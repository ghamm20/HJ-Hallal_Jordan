"""Minimal ask pipeline for Halal Jordan."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from app.citations.renderer import render_answer
from app.reasoning.answer_grounding import AnswerGrounder
from app.reasoning.authority_policy import resolve_legal_role
from app.reasoning.config_loader import ContractArtifacts, load_contract_artifacts
from app.reasoning.intent_router import route_query_intent
from app.reasoning.ollama_client import OllamaClient, OllamaClientProtocol
from app.reasoning.prompt_builder import (
    build_fast_ollama_messages,
    build_ollama_messages,
)
from app.reasoning.schema_validator import normalize_and_validate_answer
from app.reasoning.study_path import study_path_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AskRequest:
    question: str
    selected_madhhab: str
    answer_mode: str
    retrieved_sources: list[dict[str, Any]]
    research_depth: str = "balanced_research"
    greeting_style: str | None = None
    tone_level: str | None = None
    ollama_model: str | None = None
    generation_path: str = "full"
    generation_max_tokens: int | None = None
    user_context: dict[str, Any] | None = None


@dataclass(slots=True)
class AskResponse:
    answer: dict[str, Any]
    prompt_messages: list[dict[str, str]]
    rendered_text: str
    model_name: str
    answer_diagnostics: dict[str, Any] = field(default_factory=dict)
    stage_latency_ms: dict[str, int] = field(default_factory=dict)


class AskPipeline:
    """Minimal retrieval-grounded ask pipeline."""

    def __init__(
        self,
        repo_root: Path | None = None,
        client: OllamaClientProtocol | None = None,
        artifacts: ContractArtifacts | None = None,
        *,
        quote_window_chars: int = 180,
    ) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self.artifacts = artifacts or load_contract_artifacts(self.repo_root)
        self.client = client or OllamaClient()
        self.grounder = AnswerGrounder(
            self.repo_root,
            self.artifacts,
            quote_window_chars=quote_window_chars,
        )

    def ask(self, request: AskRequest) -> AskResponse:
        self._validate_request(request)
        query_intent = route_query_intent(
            question=request.question,
            answer_mode=request.answer_mode,
            selected_madhhab=request.selected_madhhab,
        )
        effective_answer_mode = _effective_answer_mode(
            requested_answer_mode=request.answer_mode,
            query_intent_id=query_intent.intent_id,
        )
        ranked_sources = self._rank_sources(
            request.retrieved_sources,
            request.selected_madhhab,
            query_intent=query_intent,
            answer_mode=effective_answer_mode,
        )
        enriched_sources = self.grounder.enrich_sources(
            question=request.question,
            selected_madhhab=request.selected_madhhab,
            answer_mode=effective_answer_mode,
            retrieved_sources=ranked_sources,
            query_intent=query_intent,
        )
        prompt_sources = _apply_pipeline_source_constraints(
            enriched_sources,
            query_intent=query_intent,
            answer_mode=effective_answer_mode,
        )
        stage_latency_ms: dict[str, int] = {}
        model_name = self.client.resolve_model(request.ollama_model)
        if _is_source_only_discipline(
            query_intent=query_intent,
            answer_mode=effective_answer_mode,
        ) and not prompt_sources:
            render_started = perf_counter()
            answer = normalize_and_validate_answer(
                candidate={
                    "answer_mode": effective_answer_mode,
                    "selected_madhhab": request.selected_madhhab,
                    "direct_answer": "No eligible primary sources found under current constraints",
                    "citations": [],
                    "ui_state": {
                        "greeting_style": request.greeting_style
                        or self.artifacts.default_greeting_style,
                        "tone_level": request.tone_level
                        or self.artifacts.default_tone_level,
                        "selected_madhhab_visible": True,
                        "answer_mode_visible": True,
                    },
                },
                schema=self.artifacts.answer_response_schema,
                request_context={
                    "question": request.question,
                    "answer_mode": effective_answer_mode,
                    "selected_madhhab": request.selected_madhhab,
                    "greeting_style": request.greeting_style
                    or self.artifacts.default_greeting_style,
                    "tone_level": request.tone_level
                    or self.artifacts.default_tone_level,
                    "repo_root": str(self.repo_root),
                },
                retrieved_sources=[],
            )
            evidence_model = self.grounder.build_evidence_model(
                answer=answer,
                selected_madhhab=request.selected_madhhab,
                answer_mode=effective_answer_mode,
                enriched_sources=[],
                all_enriched_sources=enriched_sources,
                query_intent=query_intent,
            )
            rendered_text = render_answer(answer, evidence_model)
            stage_latency_ms["prompt_build"] = 0
            stage_latency_ms["model_inference"] = 0
            stage_latency_ms["answer_render"] = int(
                (perf_counter() - render_started) * 1000
            )
            return AskResponse(
                answer=answer,
                prompt_messages=[],
                rendered_text=rendered_text,
                model_name=model_name,
                answer_diagnostics={
                    "generation_path": request.generation_path,
                    "requested_answer_mode": request.answer_mode,
                    "effective_answer_mode": effective_answer_mode,
                    "scholar_id": query_intent.scholar_id,
                    "unknown_scholar": query_intent.unknown_scholar,
                    "evidence_backfill_applied": evidence_model.evidence_backfill_applied,
                    "evidence_backfill_buckets": list(evidence_model.evidence_backfill_buckets),
                    "source_layer_composition": dict(evidence_model.source_layer_composition),
                    "metadata_completeness": dict(evidence_model.metadata_completeness),
                    "ocr_usage": dict(evidence_model.ocr_usage),
                    "teaching_layer_used": evidence_model.teaching_layer_used,
                    "teaching_sources_count": evidence_model.teaching_sources_count,
                    "teaching_layer_reason": evidence_model.teaching_layer_reason,
                    "teaching_layer_excluded_reason": evidence_model.teaching_layer_excluded_reason,
                    **study_path_diagnostics(answer),
                },
                stage_latency_ms=stage_latency_ms,
            )
        prompt_started = perf_counter()
        _ask_log("[ASK] prompt build start")
        prompt_messages = build_ollama_messages(
            question=request.question,
            selected_madhhab=request.selected_madhhab,
            answer_mode=effective_answer_mode,
            greeting_style=request.greeting_style
            or self.artifacts.default_greeting_style,
            tone_level=request.tone_level or self.artifacts.default_tone_level,
            retrieved_sources=prompt_sources,
            prompt_template=self.artifacts.prompt_template,
            response_schema=self.artifacts.answer_response_schema,
            retrieval_policy=self.artifacts.retrieval_policy,
            query_intent=query_intent,
            user_context=request.user_context,
        )
        if request.generation_path == "fast":
            prompt_messages = build_fast_ollama_messages(
                question=request.question,
                selected_madhhab=request.selected_madhhab,
                answer_mode=effective_answer_mode,
                greeting_style=request.greeting_style
                or self.artifacts.default_greeting_style,
                tone_level=request.tone_level or self.artifacts.default_tone_level,
                retrieved_sources=prompt_sources,
                response_schema=self.artifacts.answer_response_schema,
                query_intent=query_intent,
                user_context=request.user_context,
            )
        stage_latency_ms["prompt_build"] = int((perf_counter() - prompt_started) * 1000)
        _ask_log(
            f"[ASK] prompt build done: {stage_latency_ms['prompt_build'] / 1000:.3f}"
        )
        model_started = perf_counter()
        _ask_log("[ASK] model start")
        raw_output = self.client.chat_json(
            model_name=model_name,
            messages=prompt_messages,
            schema=self.artifacts.answer_response_schema,
            options=_generation_options(
                generation_path=request.generation_path,
                generation_max_tokens=request.generation_max_tokens,
            ),
        )
        stage_latency_ms["model_inference"] = int(
            (perf_counter() - model_started) * 1000
        )
        _ask_log(
            f"[ASK] model done: {stage_latency_ms['model_inference'] / 1000:.3f}"
        )
        render_started = perf_counter()
        answer = normalize_and_validate_answer(
            candidate=raw_output,
            schema=self.artifacts.answer_response_schema,
            request_context={
                "question": request.question,
                "answer_mode": effective_answer_mode,
                "selected_madhhab": request.selected_madhhab,
                "greeting_style": request.greeting_style
                or self.artifacts.default_greeting_style,
                "tone_level": request.tone_level or self.artifacts.default_tone_level,
                "repo_root": str(self.repo_root),
            },
            retrieved_sources=prompt_sources,
        )
        evidence_model = self.grounder.build_evidence_model(
            answer=answer,
            selected_madhhab=request.selected_madhhab,
            answer_mode=effective_answer_mode,
            enriched_sources=prompt_sources,
            all_enriched_sources=enriched_sources,
            query_intent=query_intent,
        )
        rendered_text = render_answer(answer, evidence_model)
        stage_latency_ms["answer_render"] = int(
            (perf_counter() - render_started) * 1000
        )
        return AskResponse(
            answer=answer,
            prompt_messages=prompt_messages,
            rendered_text=rendered_text,
            model_name=model_name,
            answer_diagnostics={
                "generation_path": request.generation_path,
                "requested_answer_mode": request.answer_mode,
                "effective_answer_mode": effective_answer_mode,
                "scholar_id": query_intent.scholar_id,
                "unknown_scholar": query_intent.unknown_scholar,
                "evidence_backfill_applied": evidence_model.evidence_backfill_applied,
                "evidence_backfill_buckets": list(evidence_model.evidence_backfill_buckets),
                "source_layer_composition": dict(evidence_model.source_layer_composition),
                "metadata_completeness": dict(evidence_model.metadata_completeness),
                "ocr_usage": dict(evidence_model.ocr_usage),
                "teaching_layer_used": evidence_model.teaching_layer_used,
                "teaching_sources_count": evidence_model.teaching_sources_count,
                "teaching_layer_reason": evidence_model.teaching_layer_reason,
                "teaching_layer_excluded_reason": evidence_model.teaching_layer_excluded_reason,
                **study_path_diagnostics(answer),
            },
            stage_latency_ms=stage_latency_ms,
        )

    def _validate_request(self, request: AskRequest) -> None:
        if not request.question.strip():
            raise ValueError("question must not be empty")
        if request.answer_mode not in self.artifacts.answer_mode_ids:
            raise ValueError(f"unsupported answer_mode: {request.answer_mode}")
        if request.selected_madhhab not in self.artifacts.selected_madhhab_ids:
            raise ValueError(
                f"unsupported selected_madhhab: {request.selected_madhhab}"
            )
        if not request.retrieved_sources:
            raise ValueError("retrieved_sources must contain at least one snippet")
        greeting_style = request.greeting_style or self.artifacts.default_greeting_style
        tone_level = request.tone_level or self.artifacts.default_tone_level
        if greeting_style not in self.artifacts.greeting_style_ids:
            raise ValueError(f"unsupported greeting_style: {greeting_style}")
        if tone_level not in self.artifacts.tone_level_ids:
            raise ValueError(f"unsupported tone_level: {tone_level}")
        if request.generation_path not in {"full", "fast"}:
            raise ValueError(f"unsupported generation_path: {request.generation_path}")

    def _rank_sources(
        self,
        retrieved_sources: list[dict[str, Any]],
        selected_madhhab: str,
        *,
        query_intent: Any | None = None,
        answer_mode: str = "research",
        enforce_source_constraints: bool = False,
    ) -> list[dict[str, Any]]:
        source_priority = {
            name: index
            for index, name in enumerate(
                self.artifacts.retrieval_policy["ranking_order_for_legal_questions"]
            )
        }
        constrained_sources = (
            _apply_pipeline_source_constraints(
                retrieved_sources,
                query_intent=query_intent,
                answer_mode=answer_mode,
            )
            if enforce_source_constraints
            else list(retrieved_sources)
        )

        def score(source: dict[str, Any]) -> tuple[int, int, str]:
            madhhab = str(source.get("madhhab", "")).lower()
            source_type = str(source.get("source_type", "inference"))
            selected_match = 0
            if selected_madhhab == "compare_all":
                selected_match = 1
            elif selected_madhhab == "not_specified":
                selected_match = 1
            elif madhhab == selected_madhhab:
                selected_match = 0
            elif not madhhab:
                selected_match = 1
            else:
                selected_match = 2
            priority = source_priority.get(source_type, len(source_priority))
            title = str(source.get("title", ""))
            return (_ranking_floor(source), selected_match, priority, title.lower())

        return sorted(constrained_sources, key=score)


def _generation_options(
    *,
    generation_path: str,
    generation_max_tokens: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": 0.1 if generation_path == "fast" else 0.3,
        "num_ctx": 2048,
    }
    max_tokens = 128
    if generation_max_tokens is not None:
        max_tokens = min(int(generation_max_tokens), 128)
    options["num_predict"] = max_tokens
    return options


def _ask_log(message: str) -> None:
    print(message, flush=True)


def _effective_answer_mode(
    *,
    requested_answer_mode: str,
    query_intent_id: str,
) -> str:
    if query_intent_id == "source_only":
        return "source_only"
    if query_intent_id == "compare_views":
        return "compare_views"
    if query_intent_id == "scholar_perspective":
        return "scholar_perspective"
    if requested_answer_mode == "scholar_perspective":
        return "research"
    return requested_answer_mode


def _is_teaching_layer_source(source: dict[str, Any]) -> bool:
    source_role_boundary = str(source.get("source_role_boundary", "") or "").strip().lower()
    source_family = str(source.get("source_family", "") or "").strip().lower()
    return source_role_boundary == "teaching_layer" or source_family == "classes"


def _is_source_only_discipline(
    *,
    query_intent: Any | None,
    answer_mode: str,
) -> bool:
    return (
        (query_intent is not None and query_intent.intent_id == "source_only")
        or answer_mode == "source_only"
    )


def _is_direct_source_lookup_discipline(query_intent: Any | None) -> bool:
    return query_intent is not None and query_intent.intent_id == "direct_source_lookup"


def _is_primary_source(source: dict[str, Any]) -> bool:
    return resolve_legal_role(source) == "primary_text"


def _source_allowed_under_constraints(
    source: dict[str, Any],
    *,
    query_intent: Any | None,
    answer_mode: str,
) -> bool:
    if _is_source_only_discipline(
        query_intent=query_intent,
        answer_mode=answer_mode,
    ):
        return _is_primary_source(source)
    if _is_direct_source_lookup_discipline(query_intent) and _is_teaching_layer_source(source):
        return False
    return True


def _apply_pipeline_source_constraints(
    sources: list[dict[str, Any]],
    *,
    query_intent: Any | None,
    answer_mode: str,
) -> list[dict[str, Any]]:
    return [
        source
        for source in sources
        if _source_allowed_under_constraints(
            source,
            query_intent=query_intent,
            answer_mode=answer_mode,
        )
    ]


def _ranking_floor(source: dict[str, Any]) -> int:
    if _is_teaching_layer_source(source):
        return 1
    return 0


def load_fixture_sources(
    fixture_path: Path,
    fixture_name: str,
) -> list[dict[str, Any]]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    try:
        return payload[fixture_name]
    except KeyError as exc:
        raise KeyError(
            f"unknown fixture set '{fixture_name}' in {fixture_path}"
        ) from exc
