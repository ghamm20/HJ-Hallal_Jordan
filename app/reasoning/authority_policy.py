"""Internal authority policy for retrieval weighting and answer assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.reasoning.intent_router import QueryIntent

SOURCE_CLASSIFICATIONS = (
    "quran",
    "hadith",
    "fiqh_manual",
    "commentary",
    "tasawwuf_text",
    "fatwa",
    "scholar_transcript",
    "transcript",
    "unknown",
)
SCHOOL_MADHHAB_IDS = {"hanafi", "shafii", "maliki", "hanbali"}

LEGAL_ROLE_LABELS = {
    "primary_text": "Primary Text",
    "madhhab_authority": "Madhhab Authority",
    "explanatory_commentary": "Explanatory Commentary",
    "spiritual_guidance": "Spiritual Guidance",
    "modern_application": "Modern Application",
    "informal_explanation": "Informal Explanation",
    "teaching_layer": "Teaching / Explanation",
    "unknown": "Unknown",
}

LEGAL_ROLE_BY_CLASSIFICATION = {
    "quran": "primary_text",
    "hadith": "primary_text",
    "fiqh_manual": "madhhab_authority",
    "commentary": "explanatory_commentary",
    "tasawwuf_text": "spiritual_guidance",
    "fatwa": "modern_application",
    "scholar_transcript": "informal_explanation",
    "transcript": "informal_explanation",
    "unknown": "unknown",
}


@dataclass(slots=True)
class AuthorityPolicy:
    policy_id: str
    classification_order: tuple[str, ...]
    role_order: tuple[str, ...]
    prefer_selected_madhhab: bool
    preserve_distinct_positions: bool
    collapse_same_section_sequences: bool
    target_fatwa_material: bool
    target_transcript_material: bool


POLICY_CONFIGS: dict[str, AuthorityPolicy] = {
    "research": AuthorityPolicy(
        policy_id="research",
        classification_order=(
            "quran",
            "hadith",
            "fiqh_manual",
            "tasawwuf_text",
            "commentary",
            "fatwa",
            "scholar_transcript",
            "transcript",
            "unknown",
        ),
        role_order=(
            "primary_text",
            "madhhab_authority",
            "explanatory_commentary",
            "spiritual_guidance",
            "modern_application",
            "informal_explanation",
            "unknown",
        ),
        prefer_selected_madhhab=True,
        preserve_distinct_positions=False,
        collapse_same_section_sequences=True,
        target_fatwa_material=False,
        target_transcript_material=False,
    ),
    "source_only": AuthorityPolicy(
        policy_id="source_only",
        classification_order=(
            "quran",
            "hadith",
            "fiqh_manual",
            "tasawwuf_text",
            "commentary",
            "fatwa",
            "scholar_transcript",
            "transcript",
            "unknown",
        ),
        role_order=(
            "primary_text",
            "madhhab_authority",
            "explanatory_commentary",
            "spiritual_guidance",
            "modern_application",
            "informal_explanation",
            "unknown",
        ),
        prefer_selected_madhhab=False,
        preserve_distinct_positions=False,
        collapse_same_section_sequences=True,
        target_fatwa_material=False,
        target_transcript_material=False,
    ),
    "compare_views": AuthorityPolicy(
        policy_id="compare_views",
        classification_order=(
            "fiqh_manual",
            "commentary",
            "quran",
            "hadith",
            "tasawwuf_text",
            "fatwa",
            "scholar_transcript",
            "transcript",
            "unknown",
        ),
        role_order=(
            "madhhab_authority",
            "primary_text",
            "explanatory_commentary",
            "spiritual_guidance",
            "modern_application",
            "informal_explanation",
            "unknown",
        ),
        prefer_selected_madhhab=True,
        preserve_distinct_positions=True,
        collapse_same_section_sequences=True,
        target_fatwa_material=False,
        target_transcript_material=False,
    ),
    "hanafi_first": AuthorityPolicy(
        policy_id="hanafi_first",
        classification_order=(
            "fiqh_manual",
            "quran",
            "hadith",
            "tasawwuf_text",
            "commentary",
            "fatwa",
            "scholar_transcript",
            "transcript",
            "unknown",
        ),
        role_order=(
            "madhhab_authority",
            "primary_text",
            "explanatory_commentary",
            "spiritual_guidance",
            "modern_application",
            "informal_explanation",
            "unknown",
        ),
        prefer_selected_madhhab=True,
        preserve_distinct_positions=False,
        collapse_same_section_sequences=True,
        target_fatwa_material=False,
        target_transcript_material=False,
    ),
    "ruling_lookup": AuthorityPolicy(
        policy_id="ruling_lookup",
        classification_order=(
            "fiqh_manual",
            "commentary",
            "fatwa",
            "quran",
            "hadith",
            "tasawwuf_text",
            "scholar_transcript",
            "transcript",
            "unknown",
        ),
        role_order=(
            "madhhab_authority",
            "explanatory_commentary",
            "spiritual_guidance",
            "modern_application",
            "primary_text",
            "informal_explanation",
            "unknown",
        ),
        prefer_selected_madhhab=True,
        preserve_distinct_positions=False,
        collapse_same_section_sequences=True,
        target_fatwa_material=False,
        target_transcript_material=False,
    ),
    "study_path": AuthorityPolicy(
        policy_id="study_path",
        classification_order=(
            "quran",
            "hadith",
            "fiqh_manual",
            "tasawwuf_text",
            "scholar_transcript",
            "transcript",
            "commentary",
            "fatwa",
            "unknown",
        ),
        role_order=(
            "primary_text",
            "madhhab_authority",
            "spiritual_guidance",
            "informal_explanation",
            "explanatory_commentary",
            "modern_application",
            "unknown",
        ),
        prefer_selected_madhhab=True,
        preserve_distinct_positions=False,
        collapse_same_section_sequences=True,
        target_fatwa_material=False,
        target_transcript_material=False,
    ),
}


def resolve_authority_policy(
    *,
    query_intent: QueryIntent | None,
    answer_mode: str,
    selected_madhhab: str,
) -> AuthorityPolicy:
    if query_intent is not None and query_intent.intent_id == "ruling_lookup":
        base = POLICY_CONFIGS["ruling_lookup"]
    elif (
        query_intent is not None
        and query_intent.intent_id == "compare_views"
    ) or answer_mode == "compare_views":
        base = POLICY_CONFIGS["compare_views"]
    elif (
        query_intent is not None
        and query_intent.intent_id == "source_only"
    ) or answer_mode == "source_only":
        base = POLICY_CONFIGS["source_only"]
    elif answer_mode == "study_path":
        base = POLICY_CONFIGS["study_path"]
    elif selected_madhhab == "hanafi":
        base = POLICY_CONFIGS["hanafi_first"]
    else:
        base = POLICY_CONFIGS["research"]

    return AuthorityPolicy(
        policy_id=base.policy_id,
        classification_order=base.classification_order,
        role_order=base.role_order,
        prefer_selected_madhhab=base.prefer_selected_madhhab,
        preserve_distinct_positions=base.preserve_distinct_positions,
        collapse_same_section_sequences=base.collapse_same_section_sequences,
        target_fatwa_material=(
            query_intent is not None and query_intent.intent_id == "fatwa_lookup"
        ),
        target_transcript_material=(
            query_intent is not None and query_intent.intent_id == "transcript_lookup"
        ),
    )


def resolve_source_classification(source: dict[str, Any]) -> str:
    classification = str(
        source.get("source_classification") or source.get("source_type") or "unknown"
    ).strip().lower()
    if classification in SOURCE_CLASSIFICATIONS:
        return classification
    return "unknown"


def resolve_legal_role(source: dict[str, Any]) -> str:
    source_role_boundary = str(source.get("source_role_boundary", "") or "").strip().lower()
    source_family = str(source.get("source_family", "") or "").strip().lower()
    if source_role_boundary == "teaching_layer" or source_family == "classes":
        return "informal_explanation"
    classification = resolve_source_classification(source)
    if classification == "fiqh_manual":
        madhhab = str(source.get("madhhab", "") or "").strip().lower()
        if madhhab in SCHOOL_MADHHAB_IDS:
            return "madhhab_authority"
        return "explanatory_commentary"
    return LEGAL_ROLE_BY_CLASSIFICATION.get(classification, "unknown")


def authority_role_label(role: str) -> str:
    return LEGAL_ROLE_LABELS.get(role, "Unknown")


def authority_priority_rank(
    source: dict[str, Any],
    *,
    policy: AuthorityPolicy,
) -> tuple[int, int]:
    classification = resolve_source_classification(source)
    role = resolve_legal_role(source)
    try:
        classification_rank = policy.classification_order.index(classification)
    except ValueError:
        classification_rank = len(policy.classification_order)
    try:
        role_rank = policy.role_order.index(role)
    except ValueError:
        role_rank = len(policy.role_order)
    return classification_rank, role_rank


def authority_bonus(
    source: dict[str, Any],
    *,
    policy: AuthorityPolicy,
    selected_madhhab: str,
) -> float:
    classification = resolve_source_classification(source)
    role = resolve_legal_role(source)
    candidate_madhhab = str(source.get("madhhab", "") or "").strip().lower()
    classification_rank, role_rank = authority_priority_rank(source, policy=policy)
    bonus = 0.0
    bonus += max(0, len(policy.classification_order) - classification_rank) * 0.09
    bonus += max(0, len(policy.role_order) - role_rank) * 0.11

    if (
        policy.prefer_selected_madhhab
        and selected_madhhab not in {"", "not_specified", "compare_all"}
        and candidate_madhhab == selected_madhhab
    ):
        if role == "madhhab_authority":
            bonus += 0.55
        elif role == "explanatory_commentary":
            bonus += 0.2
        elif role == "modern_application":
            bonus += 0.08

    if policy.policy_id == "compare_views":
        if role == "madhhab_authority":
            bonus += 0.28
        elif role == "explanatory_commentary":
            bonus += 0.1
        elif role in {"modern_application", "informal_explanation"}:
            bonus -= 0.08

    if policy.policy_id == "ruling_lookup":
        if role == "madhhab_authority":
            bonus += 0.45
        elif role == "explanatory_commentary":
            bonus += 0.18
        elif role == "modern_application":
            bonus -= 0.02
        elif role == "primary_text":
            bonus -= 0.05
        elif role == "informal_explanation":
            bonus -= 0.42

    if policy.policy_id == "source_only":
        if role == "primary_text":
            bonus += 0.24
        elif role == "modern_application":
            bonus -= 0.22
        elif role == "informal_explanation":
            bonus -= 0.4

    if policy.policy_id == "hanafi_first" and role == "madhhab_authority":
        if candidate_madhhab == "hanafi":
            bonus += 0.18
        elif candidate_madhhab:
            bonus -= 0.08

    if classification == "fatwa" and not policy.target_fatwa_material:
        bonus -= 0.18
    if classification == "transcript" and not policy.target_transcript_material:
        bonus -= 0.5
    if policy.target_fatwa_material and classification == "fatwa":
        bonus += 0.45
    if policy.target_transcript_material and classification == "transcript":
        bonus += 0.65
    return bonus
