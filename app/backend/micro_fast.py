"""Deterministic fast micro-router tier for advisory-only shadow planning."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.backend.runtime_config import RuntimeConfig
from app.reasoning.intent_router import route_query_intent

DEFAULT_FAST_MODEL_NAME = "hj-fast-router-rules"
FAST_BENCHMARK_PROMPTS = (
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
class MicroFastResult:
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
    cache_state: str
    provider: str
    model_name: str | None
    model_path: str | None
    output_valid: bool
    success: bool
    error: str | None

    def as_log_row(self) -> dict[str, Any]:
        return asdict(self)


class MicroFastRouter:
    """Run deterministic router/rewrite logic without changing production behavior."""

    def __init__(self, *, log_path: Path) -> None:
        self.log_path = log_path

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
    ) -> dict[str, Any]:
        started = perf_counter()
        provider = str(config.micro_fast_provider or "rules")
        model_name = str(config.micro_fast_name or DEFAULT_FAST_MODEL_NAME)
        base = MicroFastResult(
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
            cache_state="rules",
            provider=provider,
            model_name=model_name,
            model_path=None,
            output_valid=False,
            success=False,
            error=None,
        )
        if not config.micro_fast_enabled:
            base.error = "micro_fast_disabled"
            return self._finalize(base, started, write_log)
        if provider != "rules":
            base.error = "micro_fast_provider_unavailable"
            return self._finalize(base, started, write_log)

        predicted_intent, confidence = _predict_intent(
            question=effective_question,
            rule_router_intent=rule_router_intent,
        )
        suggested_mode = _mode_from_intent(predicted_intent)
        rewritten_query = _rewrite_query(effective_question)

        base.micro_predicted_intent = predicted_intent
        base.micro_suggested_mode = suggested_mode
        base.micro_rewritten_query = rewritten_query
        base.confidence = confidence
        base.output_valid = bool(predicted_intent and suggested_mode and rewritten_query)
        base.success = base.output_valid
        if not base.success:
            base.error = "micro_fast_output_invalid"
        return self._finalize(base, started, write_log)

    def smoke_test(self, *, config: RuntimeConfig) -> dict[str, Any]:
        return self.run(
            config=config,
            original_question="Show me sources on ablution.",
            effective_question="Show me sources on ablution.",
            rule_router_intent="direct_source_lookup",
            session_id="portable-fast-router",
            request_id="portable-fast-router-smoke",
            write_log=False,
        )

    def benchmark(
        self,
        *,
        config: RuntimeConfig,
        prompts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        benchmark_prompts = list(prompts or FAST_BENCHMARK_PROMPTS)
        results: list[dict[str, Any]] = []
        for index, prompt in enumerate(benchmark_prompts):
            result = self.run(
                config=config,
                original_question=prompt["question"],
                effective_question=prompt["question"],
                rule_router_intent=prompt["rule_router_intent"],
                session_id="micro-fast-benchmark",
                request_id=f"micro-fast-benchmark-{index + 1}",
                write_log=False,
            )
            result["label"] = prompt["label"]
            result["expected_mode"] = _mode_from_intent(prompt["rule_router_intent"])
            result["intent_match"] = result.get("micro_predicted_intent") == prompt["rule_router_intent"]
            result["mode_match"] = result.get("micro_suggested_mode") == result["expected_mode"]
            results.append(result)

        latencies = [
            int(result.get("latency_ms") or 0)
            for result in results
            if result.get("success")
        ]
        success_count = sum(1 for result in results if result.get("success"))
        output_valid_count = sum(1 for result in results if result.get("output_valid"))
        intent_accuracy_count = sum(1 for result in results if result.get("intent_match"))
        mode_accuracy_count = sum(1 for result in results if result.get("mode_match"))
        return {
            "generated_at": _utc_now(),
            "tier": "micro_fast",
            "provider": str(config.micro_fast_provider or "rules"),
            "model_name": str(config.micro_fast_name or DEFAULT_FAST_MODEL_NAME),
            "prompt_count": len(results),
            "success_count": success_count,
            "failure_count": len(results) - success_count,
            "output_valid_count": output_valid_count,
            "intent_accuracy_count": intent_accuracy_count,
            "mode_accuracy_count": mode_accuracy_count,
            "average_latency_ms": _average(latencies),
            "p95_latency_ms": _p95(latencies),
            "under_3s": bool(latencies) and max(latencies) < 3000,
            "success": success_count == len(results),
            "results": results,
        }

    def _finalize(
        self,
        result: MicroFastResult,
        started: float,
        write_log: bool,
    ) -> dict[str, Any]:
        elapsed_ms = (perf_counter() - started) * 1000
        result.latency_ms = max(1, int(round(elapsed_ms))) if elapsed_ms > 0 else 0
        payload = result.as_log_row()
        if write_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload


def compare_tier_outputs(
    fast_result: dict[str, Any] | None,
    smart_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not fast_result and not smart_result:
        return {
            "available": False,
            "intent_agreement": None,
            "mode_agreement": None,
            "rewritten_query_overlap": None,
            "latency_delta_ms": None,
            "same_request_id": None,
            "fast_was_faster": None,
        }
    fast_query = str((fast_result or {}).get("micro_rewritten_query") or "")
    smart_query = str((smart_result or {}).get("micro_rewritten_query") or "")
    fast_tokens = set(re.findall(r"[a-z0-9']+", fast_query.casefold()))
    smart_tokens = set(re.findall(r"[a-z0-9']+", smart_query.casefold()))
    union = fast_tokens | smart_tokens
    overlap = round(len(fast_tokens & smart_tokens) / len(union), 3) if union else None
    fast_latency = int((fast_result or {}).get("latency_ms") or 0)
    smart_latency = int((smart_result or {}).get("latency_ms") or 0)
    same_request_id = None
    if fast_result and smart_result:
        same_request_id = (fast_result.get("request_id") or None) == (
            smart_result.get("request_id") or None
        )
    return {
        "available": bool(fast_result or smart_result),
        "intent_agreement": (
            fast_result.get("micro_predicted_intent") == smart_result.get("micro_predicted_intent")
            if fast_result and smart_result
            else None
        ),
        "mode_agreement": (
            fast_result.get("micro_suggested_mode") == smart_result.get("micro_suggested_mode")
            if fast_result and smart_result
            else None
        ),
        "rewritten_query_overlap": overlap,
        "latency_delta_ms": (
            smart_latency - fast_latency if fast_result and smart_result else None
        ),
        "same_request_id": same_request_id,
        "fast_was_faster": (
            fast_latency <= smart_latency if fast_result and smart_result else None
        ),
    }


def _predict_intent(*, question: str, rule_router_intent: str) -> tuple[str, float]:
    routed = route_query_intent(
        question=question,
        answer_mode="research",
        selected_madhhab=_infer_selected_madhhab(question),
    )
    predicted = routed.intent_id or rule_router_intent
    normalized_question = question.casefold()
    if predicted == rule_router_intent:
        confidence = 0.96
    elif any(
        marker in normalized_question
        for marker in ("compare", "only", "source", "fatwa", "transcript", "summary", "summarize")
    ):
        confidence = 0.84
    else:
        confidence = 0.72
    return predicted, confidence


def _mode_from_intent(intent_id: str) -> str:
    if intent_id == "source_only":
        return "source_only"
    if intent_id == "compare_views":
        return "compare_views"
    return "research"


def _rewrite_query(question: str) -> str:
    lowered = re.sub(r"\s+", " ", question.strip())
    lowered = re.sub(
        r"(?i)\b(now|please|just|show me|give me|find|can you|could you)\b",
        " ",
        lowered,
    )
    lowered = re.sub(r"(?i)\bno synthesis\b", " ", lowered)
    lowered = re.sub(r"(?i)\bonly\b", " ", lowered)
    lowered = re.sub(r"(?i)\bcompare\b", " compare ", lowered)
    lowered = re.sub(r"(?i)\bviews?\b", " ", lowered)
    lowered = re.sub(r"(?i)\bwhat is\b", " ", lowered)
    lowered = re.sub(r"(?i)\bwhat does\b", " ", lowered)
    lowered = re.sub(r"(?i)\bmean in this context\b", " definition context ", lowered)
    lowered = re.sub(r"[^\w\s']+", " ", lowered)
    tokens: list[str] = []
    stopwords = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "give",
        "me",
        "on",
        "or",
        "please",
        "show",
        "sources",
        "text",
        "texts",
        "the",
        "this",
        "to",
    }
    for token in re.findall(r"[a-z0-9']+", lowered.casefold()):
        if token in stopwords:
            continue
        if token in tokens:
            continue
        tokens.append(token)
    if not tokens:
        return question.strip()
    return " ".join(tokens[:12])


def _infer_selected_madhhab(question: str) -> str:
    normalized = question.casefold()
    if "hanafi" in normalized:
        return "hanafi"
    if "shafi" in normalized:
        return "shafii"
    if "maliki" in normalized:
        return "maliki"
    if "hanbali" in normalized:
        return "hanbali"
    return "not_specified"


def _average(values: list[int]) -> int | None:
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
