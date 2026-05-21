"""Shadow-only local GGUF planning for portable micro-model experiments."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.backend.model_profiles import resolve_model_path
from app.backend.portable_paths import RuntimePathProfile
from app.backend.runtime_config import RuntimeConfig
from app.reasoning.ollama_client import _parse_json_object

ALLOWED_INTENTS = {
    "direct_source_lookup",
    "ruling_lookup",
    "source_only",
    "compare_views",
    "explain_term",
    "summarize_source",
    "fatwa_lookup",
    "transcript_lookup",
}
ALLOWED_MODES = {
    "research",
    "source_only",
    "compare_views",
    "quick_answer",
    "deep_study",
}
BENCHMARK_PROMPTS = (
    {
        "label": "direct_source_lookup",
        "question": "Show me the source for Mala Budda Minhu.",
        "rule_router_intent": "direct_source_lookup",
    },
    {
        "label": "source_only",
        "question": "Give me only primary source text on ablution.",
        "rule_router_intent": "source_only",
    },
    {
        "label": "ruling_lookup",
        "question": "What is the Hanafi view on wiping over socks?",
        "rule_router_intent": "ruling_lookup",
    },
    {
        "label": "compare_views",
        "question": "Compare Hanafi and Shafi'i views on wiping over socks.",
        "rule_router_intent": "compare_views",
    },
    {
        "label": "explain_term",
        "question": "What does fard mean in this context?",
        "rule_router_intent": "explain_term",
    },
)


@dataclass(slots=True)
class MicroShadowResult:
    timestamp: str
    session_id: str | None
    request_id: str | None
    original_question: str
    effective_question: str
    rule_router_intent: str
    micro_predicted_intent: str | None
    micro_suggested_mode: str | None
    micro_rewritten_query: str | None
    confidence: float | None
    latency_ms: int
    end_to_end_latency_ms: int
    warmup_latency_ms: int
    cache_state: str
    warmup_applied: bool
    model_path: str | None
    model_name: str | None
    output_valid: bool
    success: bool
    error: str | None

    def as_log_row(self) -> dict[str, Any]:
        return asdict(self)


class MicroLLMShadowPlanner:
    """Attempt local GGUF planning without changing production answer flow."""

    def __init__(self, *, path_profile: RuntimePathProfile, log_path: Path) -> None:
        self.path_profile = path_profile
        self.log_path = log_path
        self._lock = threading.RLock()
        self._cached_model: Any = None
        self._cached_model_signature: tuple[Any, ...] | None = None
        self._warmed_signature: tuple[Any, ...] | None = None

    def run(
        self,
        *,
        config: RuntimeConfig,
        original_question: str,
        effective_question: str,
        rule_router_intent: str,
        session_id: str | None,
        request_id: str | None,
        write_log: bool = True,
        warmup_first: bool = False,
    ) -> dict[str, Any]:
        total_started = perf_counter()
        model_path = resolve_model_path(
            root=self.path_profile.root,
            configured_path=config.micro_model_path,
        )
        model_signature = (
            self._model_signature(config=config, model_path=model_path)
            if model_path
            else None
        )
        cache_state = (
            "warm"
            if model_signature and self._cached_model_signature == model_signature
            else "cold"
        )
        warmup_applied = False
        warmup_latency_ms = 0
        if warmup_first and cache_state == "cold":
            warmup_result = self.warmup(config=config)
            warmup_latency_ms = int(warmup_result.get("latency_ms") or 0)
            warmup_applied = bool(warmup_result.get("success"))
            if warmup_applied:
                cache_state = "warm"

        base = MicroShadowResult(
            timestamp=_utc_now(),
            session_id=session_id,
            request_id=request_id,
            original_question=original_question,
            effective_question=effective_question,
            rule_router_intent=rule_router_intent,
            micro_predicted_intent=None,
            micro_suggested_mode=None,
            micro_rewritten_query=None,
            confidence=None,
            latency_ms=0,
            end_to_end_latency_ms=0,
            warmup_latency_ms=warmup_latency_ms,
            cache_state=cache_state,
            warmup_applied=warmup_applied,
            model_path=str(model_path) if model_path else None,
            model_name=config.micro_model_name,
            output_valid=False,
            success=False,
            error=None,
        )
        result = base
        inference_started = perf_counter()
        try:
            result = self._evaluate(
                config=config,
                base=base,
                model_path=model_path,
            )
        except _MicroShadowInferenceFailure:
            result = base
            result.error = "micro_shadow_inference_failed"
            result.success = False
            result.output_valid = False
        except Exception as exc:  # pragma: no cover - defensive
            result = base
            result.error = f"{type(exc).__name__}: {exc}"
            result.success = False
            result.output_valid = False
        result.latency_ms = int((perf_counter() - inference_started) * 1000)
        result.end_to_end_latency_ms = int((perf_counter() - total_started) * 1000)
        if write_log:
            self._append_log(result)
        return result.as_log_row()

    def smoke_test(self, *, config: RuntimeConfig) -> dict[str, Any]:
        return self.run(
            config=config,
            original_question="Show me sources on ablution.",
            effective_question="Show me sources on ablution.",
            rule_router_intent="direct_source_lookup",
            session_id="portable-readiness",
            request_id="portable-shadow-smoke",
            write_log=False,
            warmup_first=config.micro_shadow_warmup_before_smoke,
        )

    def warmup(self, *, config: RuntimeConfig) -> dict[str, Any]:
        started = perf_counter()
        model_path = resolve_model_path(
            root=self.path_profile.root,
            configured_path=config.micro_model_path,
        )
        if not config.micro_model_enabled:
            return _warmup_result(
                success=False,
                latency_ms=0,
                error="micro_model_disabled",
                model_path=model_path,
            )
        if config.micro_model_provider != "local_gguf":
            return _warmup_result(
                success=False,
                latency_ms=0,
                error="unsupported_micro_model_provider",
                model_path=model_path,
            )
        if model_path is None:
            return _warmup_result(
                success=False,
                latency_ms=0,
                error="micro_model_path_not_configured",
                model_path=model_path,
            )
        if not model_path.exists() or not model_path.is_file():
            return _warmup_result(
                success=False,
                latency_ms=0,
                error="micro_model_file_missing",
                model_path=model_path,
            )
        if not _backend_available():
            return _warmup_result(
                success=False,
                latency_ms=0,
                error="local_gguf_shadow_backend_unavailable",
                model_path=model_path,
            )
        signature = self._model_signature(config=config, model_path=model_path)
        if self._cached_model_signature == signature and self._warmed_signature == signature:
            return _warmup_result(
                success=True,
                latency_ms=0,
                error=None,
                model_path=model_path,
                cache_state="warm",
            )
        try:
            llm = self._load_model(model_path=model_path, config=config)
            llm.create_completion(
                prompt="{}",
                max_tokens=1,
                temperature=0.0,
                top_p=0.1,
                repeat_penalty=1.0,
            )
            self._warmed_signature = signature
            return _warmup_result(
                success=True,
                latency_ms=int((perf_counter() - started) * 1000),
                error=None,
                model_path=model_path,
                cache_state="warm",
            )
        except Exception:
            return _warmup_result(
                success=False,
                latency_ms=int((perf_counter() - started) * 1000),
                error="micro_shadow_inference_failed",
                model_path=model_path,
            )

    def reset_cache(self) -> None:
        with self._lock:
            self._cached_model = None
            self._cached_model_signature = None
            self._warmed_signature = None

    def benchmark(
        self,
        *,
        config: RuntimeConfig,
        prompts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        benchmark_prompts = list(prompts or BENCHMARK_PROMPTS)
        self.reset_cache()
        results: list[dict[str, Any]] = []
        for index, prompt in enumerate(benchmark_prompts):
            result = self.run(
                config=config,
                original_question=prompt["question"],
                effective_question=prompt["question"],
                rule_router_intent=prompt["rule_router_intent"],
                session_id="micro-benchmark",
                request_id=f"micro-benchmark-{index + 1}",
                write_log=False,
                warmup_first=False,
            )
            result["label"] = prompt["label"]
            results.append(result)

        latencies = [
            int(result.get("latency_ms") or 0)
            for result in results
            if result.get("success")
        ]
        warm_latencies = [
            int(result.get("latency_ms") or 0)
            for result in results
            if result.get("success") and result.get("cache_state") == "warm"
        ]
        cold_latencies = [
            int(result.get("latency_ms") or 0)
            for result in results
            if result.get("success") and result.get("cache_state") == "cold"
        ]
        success_count = sum(1 for result in results if result.get("success"))
        output_valid_count = sum(1 for result in results if result.get("output_valid"))
        failure_count = len(results) - success_count
        return {
            "generated_at": _utc_now(),
            "model_name": config.micro_model_name,
            "model_path": str(
                resolve_model_path(
                    root=self.path_profile.root,
                    configured_path=config.micro_model_path,
                )
                or ""
            ),
            "tuning": tuning_snapshot(config),
            "prompt_count": len(results),
            "success_count": success_count,
            "failure_count": failure_count,
            "output_valid_count": output_valid_count,
            "cold_latency_ms": cold_latencies[0] if cold_latencies else None,
            "warm_latency_ms": _average(warm_latencies),
            "average_latency_ms": _average(latencies),
            "p95_latency_ms": _p95(latencies),
            "success": success_count == len(results),
            "results": results,
        }

    def _evaluate(
        self,
        *,
        config: RuntimeConfig,
        base: MicroShadowResult,
        model_path: Path | None,
    ) -> MicroShadowResult:
        if not config.micro_model_enabled:
            base.error = "micro_model_disabled"
            return base
        if config.micro_model_role != "shadow":
            base.error = "micro_model_not_in_shadow_role"
            return base
        if config.micro_model_provider != "local_gguf":
            base.error = "unsupported_micro_model_provider"
            return base
        if model_path is None:
            base.error = "micro_model_path_not_configured"
            return base
        if not model_path.exists() or not model_path.is_file():
            base.error = "micro_model_file_missing"
            return base
        if model_path.suffix.lower() != ".gguf":
            base.error = "micro_model_path_not_gguf"
            return base
        if not _backend_available():
            base.error = "local_gguf_shadow_backend_unavailable"
            return base

        try:
            payload = self._run_local_plan(
                model_path=model_path,
                question=base.effective_question,
                config=config,
            )
        except Exception as exc:
            raise _MicroShadowInferenceFailure(str(exc)) from exc

        predicted_intent = _normalize_intent(payload.get("predicted_intent"))
        suggested_mode = _normalize_mode(payload.get("suggested_mode"))
        confidence = _normalize_confidence(payload.get("confidence"))
        rewritten_query = _normalize_query(
            payload.get("rewritten_query"),
            fallback=base.effective_question,
        )
        if predicted_intent is None or suggested_mode is None:
            raise _MicroShadowInferenceFailure("structured_shadow_labels_unusable")

        base.micro_predicted_intent = predicted_intent
        base.micro_suggested_mode = suggested_mode
        base.micro_rewritten_query = rewritten_query
        base.confidence = confidence
        base.success = True
        base.output_valid = True
        base.error = None
        return base

    def _run_local_plan(
        self,
        *,
        model_path: Path,
        question: str,
        config: RuntimeConfig,
    ) -> dict[str, Any]:
        llm = self._load_model(model_path=model_path, config=config)
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return one JSON object only. "
                        "Keys: predicted_intent, suggested_mode, rewritten_query, confidence. "
                        "Intent must be one of: "
                        + ", ".join(sorted(ALLOWED_INTENTS))
                        + ". Mode must be one of: "
                        + ", ".join(sorted(ALLOWED_MODES))
                        + ". Keep rewritten_query short."
                    ),
                },
                {"role": "user", "content": f"Q: {question}"},
            ],
            temperature=config.micro_shadow_temperature,
            top_p=config.micro_shadow_top_p,
            repeat_penalty=config.micro_shadow_repeat_penalty,
            max_tokens=config.micro_shadow_max_tokens,
            response_format={"type": "json_object"},
        )
        content = str(response["choices"][0]["message"]["content"])
        return _parse_json_object(content)

    def _load_model(self, *, model_path: Path, config: RuntimeConfig) -> Any:
        signature = self._model_signature(config=config, model_path=model_path)
        with self._lock:
            if self._cached_model is None or self._cached_model_signature != signature:
                llama_class = _llama_class()
                kwargs = {
                    "model_path": str(model_path),
                    "n_ctx": config.micro_shadow_n_ctx,
                    "n_threads": config.micro_shadow_n_threads,
                    "n_batch": config.micro_shadow_n_batch,
                    "use_mlock": config.micro_shadow_use_mlock,
                    "use_mmap": config.micro_shadow_use_mmap,
                    "verbose": False,
                }
                try:
                    self._cached_model = llama_class(**kwargs)
                except TypeError:
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("use_mlock", None)
                    fallback_kwargs.pop("use_mmap", None)
                    self._cached_model = llama_class(**fallback_kwargs)
                self._cached_model_signature = signature
                self._warmed_signature = None
            return self._cached_model

    def _model_signature(
        self,
        *,
        config: RuntimeConfig,
        model_path: Path,
    ) -> tuple[Any, ...]:
        return (
            str(model_path),
            config.micro_shadow_n_ctx,
            config.micro_shadow_n_threads,
            config.micro_shadow_n_batch,
            config.micro_shadow_use_mlock,
            config.micro_shadow_use_mmap,
        )

    def _append_log(self, result: MicroShadowResult) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.as_log_row(), ensure_ascii=False) + "\n")


def shadow_backend_status() -> dict[str, Any]:
    available = _backend_available()
    return {
        "available": available,
        "backend_name": "llama_cpp" if available else None,
        "backend_version": _backend_version() if available else None,
    }


def tuning_snapshot(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "n_ctx": config.micro_shadow_n_ctx,
        "n_threads": config.micro_shadow_n_threads,
        "n_batch": config.micro_shadow_n_batch,
        "max_tokens": config.micro_shadow_max_tokens,
        "temperature": config.micro_shadow_temperature,
        "top_p": config.micro_shadow_top_p,
        "repeat_penalty": config.micro_shadow_repeat_penalty,
        "use_mlock": config.micro_shadow_use_mlock,
        "use_mmap": config.micro_shadow_use_mmap,
        "warmup_on_startup": config.micro_shadow_warmup_on_startup,
        "warmup_before_smoke": config.micro_shadow_warmup_before_smoke,
    }


def write_benchmark_artifacts(report: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    readiness_dir = repo_root / "docs" / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    json_path = readiness_dir / "micro-llm-benchmark.json"
    markdown_path = readiness_dir / "micro-llm-benchmark.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_benchmark_markdown(report),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


def render_benchmark_markdown(report: dict[str, Any]) -> str:
    if "fast_router" in report and "smart_shadow" in report:
        fast_router = report.get("fast_router") or {}
        smart_shadow = report.get("smart_shadow") or {}
        comparison = report.get("comparison") or {}
        lines = [
            "# Micro LLM Benchmark",
            "",
            f"Generated at: `{report.get('generated_at')}`",
            f"- Prompt count: `{report.get('prompt_count')}`",
            f"- Overall success: `{report.get('success')}`",
            "",
            "## Fast Router",
            f"- Provider: `{fast_router.get('provider') or 'unknown'}`",
            f"- Model name: `{fast_router.get('model_name') or 'unknown'}`",
            f"- Success count: `{fast_router.get('success_count')}`",
            f"- Failure count: `{fast_router.get('failure_count')}`",
            f"- Output-valid count: `{fast_router.get('output_valid_count')}`",
            f"- Intent accuracy count: `{fast_router.get('intent_accuracy_count')}`",
            f"- Mode accuracy count: `{fast_router.get('mode_accuracy_count')}`",
            f"- Average latency ms: `{fast_router.get('average_latency_ms')}`",
            f"- P95 latency ms: `{fast_router.get('p95_latency_ms')}`",
            f"- Under 3s: `{fast_router.get('under_3s')}`",
            "",
            "## Smart Shadow",
            f"- Model name: `{smart_shadow.get('model_name') or 'unknown'}`",
            f"- Model path: `{smart_shadow.get('model_path') or 'unknown'}`",
            f"- Success count: `{smart_shadow.get('success_count')}`",
            f"- Failure count: `{smart_shadow.get('failure_count')}`",
            f"- Output-valid count: `{smart_shadow.get('output_valid_count')}`",
            f"- Intent accuracy count: `{smart_shadow.get('intent_accuracy_count')}`",
            f"- Mode accuracy count: `{smart_shadow.get('mode_accuracy_count')}`",
            f"- Cold latency ms: `{smart_shadow.get('cold_latency_ms')}`",
            f"- Warm latency ms: `{smart_shadow.get('warm_latency_ms')}`",
            f"- Average latency ms: `{smart_shadow.get('average_latency_ms')}`",
            f"- P95 latency ms: `{smart_shadow.get('p95_latency_ms')}`",
            "",
            "## Agreement",
            f"- Shared prompt count: `{comparison.get('shared_prompt_count')}`",
            f"- Intent agreement count: `{comparison.get('intent_agreement_count')}`",
            f"- Mode agreement count: `{comparison.get('mode_agreement_count')}`",
            f"- Average rewritten-query overlap: `{comparison.get('average_rewritten_query_overlap')}`",
        ]
        lines.extend(["", "## Smart Shadow Tuning"])
        for key, value in (smart_shadow.get("tuning") or {}).items():
            lines.append(f"- `{key}`: `{value}`")

        lines.extend(["", "## Per-Prompt Comparison"])
        for item in comparison.get("per_prompt", []):
            lines.append(
                "- "
                f"`{item.get('label')}` "
                f"intent_agreement=`{item.get('intent_agreement')}` "
                f"mode_agreement=`{item.get('mode_agreement')}` "
                f"query_overlap=`{item.get('rewritten_query_overlap')}` "
                f"fast_latency_ms=`{item.get('fast_latency_ms')}` "
                f"smart_latency_ms=`{item.get('smart_latency_ms')}`"
            )

        lines.extend(["", "## Fast Router Results"])
        for result in fast_router.get("results", []):
            lines.append(
                "- "
                f"`{result.get('label')}` "
                f"success=`{result.get('success')}` "
                f"output_valid=`{result.get('output_valid')}` "
                f"latency_ms=`{result.get('latency_ms')}` "
                f"intent=`{result.get('micro_predicted_intent') or 'none'}` "
                f"mode=`{result.get('micro_suggested_mode') or 'none'}` "
                f"error=`{result.get('error') or 'none'}`"
            )

        lines.extend(["", "## Smart Shadow Results"])
        for result in smart_shadow.get("results", []):
            lines.append(
                "- "
                f"`{result.get('label')}` "
                f"success=`{result.get('success')}` "
                f"output_valid=`{result.get('output_valid')}` "
                f"cache_state=`{result.get('cache_state')}` "
                f"latency_ms=`{result.get('latency_ms')}` "
                f"intent=`{result.get('micro_predicted_intent') or 'none'}` "
                f"mode=`{result.get('micro_suggested_mode') or 'none'}` "
                f"error=`{result.get('error') or 'none'}`"
            )
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# Micro LLM Benchmark",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"- Model name: `{report.get('model_name') or 'unknown'}`",
        f"- Model path: `{report.get('model_path') or 'unknown'}`",
        f"- Prompt count: `{report.get('prompt_count')}`",
        f"- Success count: `{report.get('success_count')}`",
        f"- Failure count: `{report.get('failure_count')}`",
        f"- Output-valid count: `{report.get('output_valid_count')}`",
        f"- Cold latency ms: `{report.get('cold_latency_ms')}`",
        f"- Warm latency ms: `{report.get('warm_latency_ms')}`",
        f"- Average latency ms: `{report.get('average_latency_ms')}`",
        f"- P95 latency ms: `{report.get('p95_latency_ms')}`",
        "",
        "## Tuning",
    ]
    for key, value in (report.get("tuning") or {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Results"])
    for result in report.get("results", []):
        lines.append(
            "- "
            f"`{result.get('label')}` "
            f"success=`{result.get('success')}` "
            f"output_valid=`{result.get('output_valid')}` "
            f"cache_state=`{result.get('cache_state')}` "
            f"latency_ms=`{result.get('latency_ms')}` "
            f"intent=`{result.get('micro_predicted_intent') or 'none'}` "
            f"mode=`{result.get('micro_suggested_mode') or 'none'}` "
            f"error=`{result.get('error') or 'none'}`"
        )
    lines.append("")
    return "\n".join(lines)


class _MicroShadowInferenceFailure(RuntimeError):
    """Internal signal for truthful shadow inference failure mapping."""


def _normalize_intent(value: Any) -> str | None:
    candidate = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    candidate = {
        "information_retrieval": "direct_source_lookup",
        "source_lookup": "direct_source_lookup",
        "retrieval": "direct_source_lookup",
        "direct_lookup": "direct_source_lookup",
        "comparison": "compare_views",
        "compare": "compare_views",
        "summary": "summarize_source",
        "summarize": "summarize_source",
        "explanation": "explain_term",
        "define_term": "explain_term",
    }.get(candidate, candidate)
    if candidate in ALLOWED_INTENTS:
        return candidate
    return None


def _normalize_mode(value: Any) -> str | None:
    candidate = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    candidate = {
        "retrieval": "research",
        "source": "source_only",
        "sources_only": "source_only",
        "comparison": "compare_views",
        "compare": "compare_views",
        "quick": "quick_answer",
        "deep": "deep_study",
    }.get(candidate, candidate)
    if candidate in ALLOWED_MODES:
        return candidate
    return None


def _normalize_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if candidate < 0:
        return 0.0
    if candidate > 1:
        return 1.0
    return round(candidate, 4)


def _normalize_query(value: Any, *, fallback: str) -> str:
    candidate = str(value or "").strip()
    return candidate or fallback


def _average(values: list[int]) -> int | None:
    if not values:
        return None
    return int(sum(values) / len(values))


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return int(ordered[index])


def _warmup_result(
    *,
    success: bool,
    latency_ms: int,
    error: str | None,
    model_path: Path | None,
    cache_state: str = "cold",
) -> dict[str, Any]:
    return {
        "success": success,
        "latency_ms": latency_ms,
        "error": error,
        "model_path": str(model_path) if model_path else None,
        "cache_state": cache_state,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _backend_available() -> bool:
    return importlib.util.find_spec("llama_cpp") is not None


def _backend_version() -> str | None:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def _llama_class() -> Any:
    from llama_cpp import Llama  # type: ignore

    return Llama
