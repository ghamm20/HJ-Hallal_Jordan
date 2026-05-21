"""Conservative fast-path classification for low-risk chat preparation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.backend.runtime_config import RuntimeConfig
from app.reasoning.intent_router import QueryIntent

SOURCE_LAYER_MARKERS = (
    "qur",
    "hadith",
    "fiqh",
    "commentary",
    "fatwa",
    "transcript",
    "manual",
)

QUICK_SOURCE_CHECK = "quick_source_check"
BALANCED_RESEARCH = "balanced_research"
DEEP_RESEARCH = "deep_research"
RESEARCH_DEPTH_IDS = {
    QUICK_SOURCE_CHECK,
    BALANCED_RESEARCH,
    DEEP_RESEARCH,
}
RESEARCH_DEPTH_ALIASES = {
    "quick": QUICK_SOURCE_CHECK,
    "quick_source_check": QUICK_SOURCE_CHECK,
    "quick source check": QUICK_SOURCE_CHECK,
    "balanced": BALANCED_RESEARCH,
    "balanced_research": BALANCED_RESEARCH,
    "balanced research": BALANCED_RESEARCH,
    "deep": DEEP_RESEARCH,
    "deep_research": DEEP_RESEARCH,
    "deep research": DEEP_RESEARCH,
}


@dataclass(slots=True)
class FastPathDecision:
    query_complexity: str
    eligible_for_fast_path: bool
    path_used: str
    fast_path_reason: str | None
    fast_path_rejected_reason: str | None
    retrieval_top_k: int
    retrieval_candidate_limit: int
    generation_max_tokens: int | None
    signals: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchDepthDecision:
    requested_research_depth: str
    effective_research_depth: str
    depth_auto_upgraded: bool
    depth_upgrade_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_research_depth(value: str | None) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    if not normalized:
        return BALANCED_RESEARCH
    return RESEARCH_DEPTH_ALIASES.get(normalized, BALANCED_RESEARCH)


def plan_research_depth(
    *,
    requested_research_depth: str | None,
    query_intent: QueryIntent,
    continuity_used: bool,
) -> ResearchDepthDecision:
    requested = normalize_research_depth(requested_research_depth)
    if requested != QUICK_SOURCE_CHECK:
        return ResearchDepthDecision(
            requested_research_depth=requested,
            effective_research_depth=requested,
            depth_auto_upgraded=False,
            depth_upgrade_reason=None,
        )

    if query_intent.intent_id == "compare_views":
        return ResearchDepthDecision(
            requested_research_depth=requested,
            effective_research_depth=DEEP_RESEARCH,
            depth_auto_upgraded=True,
            depth_upgrade_reason="compare_request_requires_deep_research",
        )
    if query_intent.intent_id in {
        "ruling_lookup",
        "fatwa_lookup",
        "transcript_lookup",
    }:
        return ResearchDepthDecision(
            requested_research_depth=requested,
            effective_research_depth=DEEP_RESEARCH,
            depth_auto_upgraded=True,
            depth_upgrade_reason="authority_lookup_requires_deep_research",
        )
    if continuity_used and query_intent.intent_id not in {
        "direct_source_lookup",
        "source_only",
    }:
        return ResearchDepthDecision(
            requested_research_depth=requested,
            effective_research_depth=BALANCED_RESEARCH,
            depth_auto_upgraded=True,
            depth_upgrade_reason="followup_requires_balanced_research",
        )
    if query_intent.intent_id in {"summarize_source", "explain_term"}:
        return ResearchDepthDecision(
            requested_research_depth=requested,
            effective_research_depth=BALANCED_RESEARCH,
            depth_auto_upgraded=True,
            depth_upgrade_reason="explanatory_request_requires_balanced_research",
        )
    return ResearchDepthDecision(
        requested_research_depth=requested,
        effective_research_depth=requested,
        depth_auto_upgraded=False,
        depth_upgrade_reason=None,
    )


def finalize_research_depth(
    *,
    decision: ResearchDepthDecision,
    fast_path_decision: FastPathDecision,
    retrieval_alignment: dict[str, Any],
) -> ResearchDepthDecision:
    if (
        decision.requested_research_depth != QUICK_SOURCE_CHECK
        or decision.effective_research_depth != QUICK_SOURCE_CHECK
    ):
        return decision
    if fast_path_decision.path_used == "fast":
        return decision
    if retrieval_alignment.get("should_degrade"):
        return ResearchDepthDecision(
            requested_research_depth=decision.requested_research_depth,
            effective_research_depth=BALANCED_RESEARCH,
            depth_auto_upgraded=True,
            depth_upgrade_reason="weak_evidence_requires_balanced_research",
        )
    return ResearchDepthDecision(
        requested_research_depth=decision.requested_research_depth,
        effective_research_depth=_quick_upgrade_target_from_fast_path(
            fast_path_decision.fast_path_rejected_reason
        ),
        depth_auto_upgraded=True,
        depth_upgrade_reason=_quick_upgrade_reason_from_fast_path(
            fast_path_decision.fast_path_rejected_reason
        ),
    )


def classify_fast_path(
    *,
    config: RuntimeConfig,
    question: str,
    answer_mode: str,
    query_intent: QueryIntent,
    selected_madhhab: str,
    continuity_used: bool,
    micro_fast_confidence: float | None,
    micro_fast_intent: str | None,
    research_depth: str = BALANCED_RESEARCH,
) -> FastPathDecision:
    normalized_research_depth = normalize_research_depth(research_depth)
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    token_count = len(re.findall(r"\w+", normalized))
    layer_hits = sum(1 for marker in SOURCE_LAYER_MARKERS if marker in normalized)
    compare_requested = (
        query_intent.intent_id == "compare_views" or answer_mode == "compare_views"
    )
    study_path_requested = answer_mode == "study_path"
    layered_intent = query_intent.intent_id in {
        "ruling_lookup",
        "fatwa_lookup",
        "transcript_lookup",
    }
    source_only_requested = (
        query_intent.intent_id == "source_only" or answer_mode == "source_only"
    )
    source_only_primary_layers = (
        source_only_requested
        and layer_hits <= 2
        and not any(
            marker in normalized
            for marker in ("commentary", "fatwa", "transcript", "modern")
        )
    )
    multi_source_synthesis = (
        (layer_hits >= 2 and not source_only_primary_layers)
        or "across different collections" in normalized
        or "across" in normalized and "collection" in normalized
        or "show how this topic appears across" in normalized
        or "qur'an, hadith, and" in normalized
        or "quran, hadith, and" in normalized
    )
    vague_followup = continuity_used and not source_only_requested
    definitional = query_intent.intent_id == "explain_term"
    summarize_requested = query_intent.intent_id == "summarize_source"
    direct_lookup = query_intent.intent_id == "direct_source_lookup"
    explicit_source_lookup = any(
        phrase in normalized
        for phrase in (
            "show me the source",
            "show me sources",
            "find the source",
            "find sources",
            "give me sources",
            "strongest sources",
            "source for",
        )
    )
    micro_fast_low_confidence = (
        micro_fast_confidence is not None and micro_fast_confidence < 0.75
    )
    risky_disagreement = bool(
        micro_fast_intent
        and micro_fast_intent != query_intent.intent_id
        and query_intent.intent_id in {"compare_views", "ruling_lookup", "fatwa_lookup"}
    )

    signals = {
        "compare_requested": compare_requested,
        "study_path_requested": study_path_requested,
        "layered_intent": layered_intent,
        "multi_source_synthesis": multi_source_synthesis,
        "vague_followup": vague_followup,
        "source_only_requested": source_only_requested,
        "source_only_primary_layers": source_only_primary_layers,
        "definitional": definitional,
        "summarize_requested": summarize_requested,
        "direct_lookup": direct_lookup,
        "explicit_source_lookup": explicit_source_lookup,
        "token_count": token_count,
        "source_layer_markers": layer_hits,
        "micro_fast_low_confidence": micro_fast_low_confidence,
        "risky_disagreement": risky_disagreement,
        "selected_madhhab": selected_madhhab,
        "research_depth": normalized_research_depth,
    }

    if study_path_requested:
        complexity = "complex"
        rejected_reason = "study_path_requires_full_pipeline"
    elif compare_requested:
        complexity = "complex"
        rejected_reason = "compare_views_requires_full_pipeline"
    elif layered_intent:
        complexity = "complex"
        rejected_reason = "layered_authority_question_requires_full_pipeline"
    elif vague_followup:
        complexity = "complex"
        rejected_reason = "followup_requires_full_pipeline"
    elif multi_source_synthesis:
        complexity = "complex"
        rejected_reason = "multi_source_synthesis_requires_full_pipeline"
    elif risky_disagreement:
        complexity = "complex"
        rejected_reason = "risky_micro_fast_disagreement"
    elif definitional:
        complexity = "complex"
        rejected_reason = "definition_requests_require_full_pipeline"
    elif summarize_requested:
        complexity = "complex"
        rejected_reason = "summary_requests_require_full_pipeline"
    elif micro_fast_low_confidence:
        complexity = "moderate"
        rejected_reason = "micro_fast_confidence_too_low"
    elif source_only_requested:
        complexity = "simple"
        rejected_reason = None
    elif direct_lookup and explicit_source_lookup and token_count <= 12:
        complexity = "simple"
        rejected_reason = None
    elif direct_lookup and explicit_source_lookup:
        complexity = "moderate"
        rejected_reason = None
    else:
        complexity = "complex"
        rejected_reason = "intent_requires_full_pipeline"

    if normalized_research_depth == DEEP_RESEARCH:
        return FastPathDecision(
            query_complexity=complexity,
            eligible_for_fast_path=False,
            path_used="full",
            fast_path_reason=None,
            fast_path_rejected_reason="deep_research_requested",
            retrieval_top_k=config.deep_research_top_k,
            retrieval_candidate_limit=config.deep_research_candidate_limit,
            generation_max_tokens=config.deep_research_max_tokens,
            signals=signals,
        )

    eligible = bool(config.fast_path_enabled) and rejected_reason is None
    retrieval_top_k = config.rerank_limit
    retrieval_candidate_limit = config.max_retrieval_candidates
    generation_max_tokens: int | None = None
    fast_reason: str | None = None
    if eligible and complexity == "simple":
        if normalized_research_depth == QUICK_SOURCE_CHECK:
            retrieval_top_k = config.quick_research_simple_top_k
            retrieval_candidate_limit = config.quick_research_simple_candidate_limit
            generation_max_tokens = config.quick_research_simple_max_tokens
        else:
            retrieval_top_k = config.fast_path_simple_top_k
            retrieval_candidate_limit = config.fast_path_simple_candidate_limit
            generation_max_tokens = config.fast_path_simple_max_tokens
        fast_reason = f"simple_{query_intent.intent_id}"
    elif eligible and complexity == "moderate":
        if normalized_research_depth == QUICK_SOURCE_CHECK:
            retrieval_top_k = config.quick_research_moderate_top_k
            retrieval_candidate_limit = config.quick_research_moderate_candidate_limit
            generation_max_tokens = config.quick_research_moderate_max_tokens
        else:
            retrieval_top_k = config.fast_path_moderate_top_k
            retrieval_candidate_limit = config.fast_path_moderate_candidate_limit
            generation_max_tokens = config.fast_path_moderate_max_tokens
        fast_reason = f"moderate_{query_intent.intent_id}"

    return FastPathDecision(
        query_complexity=complexity,
        eligible_for_fast_path=eligible,
        path_used="fast" if eligible else "full",
        fast_path_reason=fast_reason,
        fast_path_rejected_reason=(
            rejected_reason
            if not eligible
            else None
        ),
        retrieval_top_k=retrieval_top_k,
        retrieval_candidate_limit=retrieval_candidate_limit,
        generation_max_tokens=generation_max_tokens,
        signals=signals,
    )


def finalize_fast_path(
    *,
    decision: FastPathDecision,
    config: RuntimeConfig,
    snippets: list[dict[str, Any]],
    retrieval_alignment: dict[str, Any],
) -> FastPathDecision:
    if not decision.eligible_for_fast_path:
        return decision

    if retrieval_alignment.get("should_degrade"):
        return _with_full_path(
            decision,
            rejected_reason="retrieval_low_alignment_requires_full_pipeline",
        )

    source_count = len(snippets)
    if source_count == 0:
        return _with_full_path(
            decision,
            rejected_reason="no_sources_for_fast_path",
        )

    layer_values = {
        str(
            snippet.get("source_role_boundary")
            or snippet.get("source_classification")
            or snippet.get("source_type")
            or "unknown"
        )
        for snippet in snippets
    }
    if len(layer_values) > config.fast_path_max_layer_count:
        return _with_full_path(
            decision,
            rejected_reason="source_layers_require_full_pipeline",
        )

    if decision.signals.get("source_only_requested") and (
        source_count > config.fast_path_source_only_max_sources
    ):
        return _with_full_path(
            decision,
            rejected_reason="source_only_result_set_too_large",
        )

    refined_reason = decision.fast_path_reason or "fast_path_eligible"
    if decision.signals.get("source_only_requested"):
        refined_reason = f"{refined_reason}_small_source_only_set"
    elif source_count <= 2:
        refined_reason = f"{refined_reason}_high_confidence_small_set"

    return FastPathDecision(
        query_complexity=decision.query_complexity,
        eligible_for_fast_path=True,
        path_used="fast",
        fast_path_reason=refined_reason,
        fast_path_rejected_reason=None,
        retrieval_top_k=decision.retrieval_top_k,
        retrieval_candidate_limit=decision.retrieval_candidate_limit,
        generation_max_tokens=decision.generation_max_tokens,
        signals={
            **decision.signals,
            "source_count": source_count,
            "layer_count": len(layer_values),
        },
    )


def _with_full_path(
    decision: FastPathDecision,
    *,
    rejected_reason: str,
) -> FastPathDecision:
    return FastPathDecision(
        query_complexity=decision.query_complexity,
        eligible_for_fast_path=False,
        path_used="full",
        fast_path_reason=None,
        fast_path_rejected_reason=rejected_reason,
        retrieval_top_k=decision.retrieval_top_k,
        retrieval_candidate_limit=decision.retrieval_candidate_limit,
        generation_max_tokens=None,
        signals=dict(decision.signals),
    )


def _quick_upgrade_target_from_fast_path(rejected_reason: str | None) -> str:
    deep_reasons = {
        "study_path_requires_full_pipeline",
        "compare_views_requires_full_pipeline",
        "layered_authority_question_requires_full_pipeline",
        "multi_source_synthesis_requires_full_pipeline",
        "source_layers_require_full_pipeline",
    }
    if rejected_reason in deep_reasons:
        return DEEP_RESEARCH
    return BALANCED_RESEARCH


def _quick_upgrade_reason_from_fast_path(rejected_reason: str | None) -> str:
    reason_map = {
        "study_path_requires_full_pipeline": "study_path_requires_deep_research",
        "compare_views_requires_full_pipeline": "compare_request_requires_deep_research",
        "layered_authority_question_requires_full_pipeline": "authority_lookup_requires_deep_research",
        "followup_requires_full_pipeline": "followup_requires_balanced_research",
        "multi_source_synthesis_requires_full_pipeline": "layered_question_requires_deep_research",
        "micro_fast_confidence_too_low": "confidence_requires_balanced_research",
        "source_only_result_set_too_large": "result_set_requires_balanced_research",
        "source_layers_require_full_pipeline": "source_layers_require_deep_research",
        "no_sources_for_fast_path": "weak_evidence_requires_balanced_research",
        "intent_requires_full_pipeline": "complexity_requires_balanced_research",
        "risky_micro_fast_disagreement": "risky_disagreement_requires_balanced_research",
    }
    return reason_map.get(rejected_reason or "", "complexity_requires_balanced_research")
