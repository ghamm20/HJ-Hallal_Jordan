"""Metadata-aware reranking for retrieval candidates."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from app.reasoning.authority_policy import (
    authority_bonus,
    authority_priority_rank,
    resolve_authority_policy,
    resolve_legal_role,
    resolve_source_classification,
)
from app.reasoning.intent_router import QueryIntent
from app.reasoning.trust_engine import load_profile as load_trust_profile
from app.reasoning.trust_engine import score as compute_trust_score
from app.retrieval.families import build_duplicate_family_key, normalize_title_key

REPETITIVE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

PRAYER_FOCUS_TERMS = {
    "pray",
    "prayer",
    "prayers",
    "salat",
    "salah",
    "rakah",
    "rakats",
    "rakat",
    "takbir",
    "ruku",
    "sujud",
    "sajdah",
    "tashahhud",
    "fajr",
    "dhuhr",
    "asr",
    "maghrib",
    "isha",
}
PRAYER_METHOD_TERMS = {
    "opening takbir",
    "raise your hands",
    "takbir",
    "recitation",
    "qiyam",
    "stand",
    "standing",
    "facing the qibla",
    "bow",
    "bowing",
    "ruku",
    "prostrate",
    "prostration",
    "sujud",
    "sit for tashahhud",
    "tashahhud",
    "qibla",
    "pray as you have seen me pray",
}
PRAYER_METHOD_DISTRACTOR_TERMS = {
    "imam",
    "followers",
    "congregation",
    "sitting while leading",
    "if he prays sitting",
    "eid",
    "tashriq",
    "sunrise",
    "sunset",
    "makkah",
    "forgetfulness",
}

PURIFICATION_FOCUS_TERMS = {
    "wudu",
    "wudhu",
    "ablution",
    "purification",
    "taharah",
    "ghusl",
    "tayammum",
    "tayamum",
    "book of purification",
    "purity",
}
PURIFICATION_METHOD_TERMS = {
    "wash the face",
    "wash your faces",
    "wash the arms",
    "wash your hands",
    "hands up to the elbows",
    "wipe the head",
    "wipe over your heads",
    "wash the feet",
    "feet up to the ankles",
    "rinse the mouth",
    "rinse the nose",
    "perform wudu",
    "make wudu",
    "wudu",
    "ablution",
}
PURIFICATION_METHOD_DISTRACTOR_TERMS = {
    "tayammum",
    "tayamum",
    "dry ablution",
    "ghusl",
    "junub",
    "janabah",
    "major impurity",
    "sexual discharge",
    "wet dream",
    "menstruating",
}
PRAYER_PREREQUISITE_TERMS = {
    "when you want to pray",
    "before prayer",
    "for prayer",
    "required",
    "necessary",
    "condition",
    "validity",
    "valid",
    "state of purity",
    "purity",
}

DIVERSITY_PASSES = (
    {"source_limit": 1, "family_limit": 1, "section_limit": 1, "allow_repetitive": False},
    {"source_limit": 1, "family_limit": 1, "section_limit": 2, "allow_repetitive": False},
    {"source_limit": 2, "family_limit": 1, "section_limit": 2, "allow_repetitive": False},
    {"source_limit": 1, "family_limit": 2, "section_limit": 2, "allow_repetitive": False},
    {"source_limit": 2, "family_limit": 2, "section_limit": 2, "allow_repetitive": False},
    {"source_limit": 2, "family_limit": 2, "section_limit": 3, "allow_repetitive": True},
    {"source_limit": 3, "family_limit": 3, "section_limit": 4, "allow_repetitive": True},
)
SUPPORTED_MADHHABS = ("hanafi", "shafii", "maliki", "hanbali")
SCHOOL_AUTHORITY_CLASSES = {"fiqh_manual", "commentary", "fatwa"}


def rerank_candidates(
    candidates: list[dict[str, Any]],
    *,
    selected_madhhab: str,
    retrieval_policy: dict[str, Any],
    source_type_registry: dict[str, Any],
    top_k: int = 5,
    query_intent: QueryIntent | None = None,
    answer_mode: str = "research",
    trust_profile_id: str = "default",
) -> list[dict[str, Any]]:
    authority_policy = resolve_authority_policy(
        query_intent=query_intent,
        answer_mode=answer_mode,
        selected_madhhab=selected_madhhab,
    )

    # Weight and Trust Engine pre-pass. Computes a transparent breakdown
    # per candidate and attaches it for downstream rendering. The default
    # profile contributes zero, so behavior only changes when a non-default
    # trust_profile_id is selected.
    trust_profile = load_trust_profile(trust_profile_id)
    for candidate in candidates:
        breakdown = compute_trust_score(candidate, profile=trust_profile)
        candidate["_trust_breakdown"] = breakdown.to_dict()
        candidate["_trust_bonus"] = breakdown.total
    policy_priority = {
        name: index
        for index, name in enumerate(
            retrieval_policy["ranking_order_for_legal_questions"]
        )
    }
    authority_priority = {
        item["id"]: item["display_priority"]
        for item in source_type_registry["source_types"]
    }

    def sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        source_type = str(candidate.get("source_type", ""))
        return (
            _teaching_rank_floor(candidate),
            _madhhab_intent_rank(
                candidate,
                query_intent=query_intent,
                selected_madhhab=selected_madhhab,
            ),
            -_effective_relevance_score(
                candidate,
                query_intent=query_intent,
                selected_madhhab=selected_madhhab,
                authority_policy=authority_policy,
            ),
            _selected_madhhab_rank(
                candidate,
                selected_madhhab,
                query_intent=query_intent,
            ),
            *authority_priority_rank(candidate, policy=authority_policy),
            policy_priority.get(source_type, len(policy_priority)),
            authority_priority.get(source_type, 999),
            *_family_canonical_rank(candidate),
            str(candidate.get("reference", "")).lower(),
        )

    ranked = sorted(candidates, key=sort_key)
    if query_intent is not None and query_intent.intent_id in {"source_only", "direct_source_lookup"}:
        ranked = [
            candidate for candidate in ranked if not _is_teaching_layer_candidate(candidate)
        ]
    return _select_diverse_top_candidates(
        ranked,
        top_k=top_k,
        query_intent=query_intent,
        authority_policy_id=authority_policy.policy_id,
        collapse_same_section_sequences=authority_policy.collapse_same_section_sequences,
        preserve_distinct_positions=authority_policy.preserve_distinct_positions,
    )


def get_duplicate_family_key(candidate: dict[str, Any]) -> str:
    return build_duplicate_family_key(
        source_type=str(candidate.get("source_type", "") or ""),
        title=str(candidate.get("title", "") or ""),
        source_path=str(candidate.get("source_path", "") or ""),
        collection=str(candidate.get("collection", "") or ""),
    )


def get_section_cluster_key(candidate: dict[str, Any]) -> str:
    section_label = normalize_title_key(str(candidate.get("section_label", "") or ""))
    if not section_label:
        return ""
    scope = str(candidate.get("source_path", "") or "").strip().lower()
    if not scope:
        scope = get_duplicate_family_key(candidate)
    if not scope:
        return ""
    return f"{scope}|{section_label}"


def candidates_are_repetitive(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_source = str(left.get("source_path", "")).strip().lower()
    right_source = str(right.get("source_path", "")).strip().lower()
    left_family = get_duplicate_family_key(left)
    right_family = get_duplicate_family_key(right)
    same_source = bool(left_source and left_source == right_source)
    same_family = bool(left_family and left_family == right_family)
    if not same_source and not same_family:
        return False

    left_fingerprint = _candidate_text_fingerprint(left)
    right_fingerprint = _candidate_text_fingerprint(right)
    if left_fingerprint and left_fingerprint == right_fingerprint:
        return True

    left_tokens = _significant_text_tokens(str(left.get("text", "") or ""))
    right_tokens = _significant_text_tokens(str(right.get("text", "") or ""))
    if not left_tokens or not right_tokens:
        return False

    token_overlap = _token_jaccard(left_tokens, right_tokens)
    prefix_overlap = _prefix_overlap(
        " ".join(left_tokens[:24]),
        " ".join(right_tokens[:24]),
    )
    left_section = normalize_title_key(str(left.get("section_label", "") or ""))
    right_section = normalize_title_key(str(right.get("section_label", "") or ""))
    if token_overlap >= 0.9:
        return True
    if prefix_overlap >= 0.92 and min(len(left_tokens), len(right_tokens)) >= 8:
        return True
    if left_section and left_section == right_section and token_overlap >= 0.72:
        return True
    if (
        str(left.get("section_kind", "")) == "contents_like"
        and str(right.get("section_kind", "")) == "contents_like"
    ):
        return True
    return False


def _effective_relevance_score(
    candidate: dict[str, Any],
    *,
    query_intent: QueryIntent | None,
    selected_madhhab: str,
    authority_policy: Any,
) -> float:
    score = float(candidate.get("retrieval_score", 0))
    score += _loader_hint_bonus(candidate)
    score += _family_preference_bonus(candidate)
    score += float(candidate.get("_trust_bonus", 0.0))
    score += authority_bonus(
        candidate,
        policy=authority_policy,
        selected_madhhab=selected_madhhab,
    )
    score += _intent_authority_bonus(
        candidate,
        query_intent=query_intent,
        selected_madhhab=selected_madhhab,
        authority_policy_id=authority_policy.policy_id,
    )
    score += _madhhab_intent_bonus(
        candidate,
        query_intent=query_intent,
        selected_madhhab=selected_madhhab,
        authority_policy_id=authority_policy.policy_id,
    )
    score += _worship_topic_bonus(candidate, query_intent=query_intent)
    if str(candidate.get("section_label", "")).strip():
        score += 0.15
    section_kind = str(candidate.get("section_kind", "")).strip().lower()
    if section_kind == "contents_like":
        score -= 1.6
    elif section_kind in {
        "book_heading",
        "chapter_heading",
        "major_heading",
        "numbered_section",
        "supplement_heading",
        "title_like",
    }:
        score += 0.08

    extraction_status = str(candidate.get("extraction_status", "")).strip().lower()
    if extraction_status == "partial":
        score -= 0.12
    warning_count = int(candidate.get("document_quality_warnings", 0) or 0)
    score -= min(warning_count, 6) * 0.03
    if _copy_title_rank(candidate):
        score -= 0.35

    return score


def _selected_madhhab_rank(
    candidate: dict[str, Any],
    selected_madhhab: str,
    *,
    query_intent: QueryIntent | None,
) -> int:
    requested_madhhabs = _requested_madhhabs(
        query_intent=query_intent,
        selected_madhhab=selected_madhhab,
    )
    candidate_madhhab = str(candidate.get("madhhab", "")).strip().lower()
    if not requested_madhhabs:
        return 1
    if candidate_madhhab in requested_madhhabs:
        return 0
    if not candidate_madhhab:
        return 1
    return 2


def _select_diverse_top_candidates(
    ranked_candidates: list[dict[str, Any]],
    *,
    top_k: int,
    query_intent: QueryIntent | None,
    authority_policy_id: str,
    collapse_same_section_sequences: bool,
    preserve_distinct_positions: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_indexes: set[int] = set()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    madhhab_counts: Counter[str] = Counter()

    if authority_policy_id == "study_path":
        prioritized = _select_study_path_layered_candidates(
            ranked_candidates,
            top_k=top_k,
        )
        if len(prioritized) >= top_k:
            return prioritized
        if prioritized:
            selected.extend(prioritized)
            for index, candidate in enumerate(ranked_candidates):
                if any(candidate is prior for prior in prioritized):
                    selected_indexes.add(index)
            for candidate in prioritized:
                source_key = str(candidate.get("source_path", "")).strip().lower()
                family_key = get_duplicate_family_key(candidate)
                section_key = get_section_cluster_key(candidate)
                madhhab_key = _diversity_madhhab_key(
                    candidate,
                    query_intent,
                    preserve_distinct_positions=preserve_distinct_positions,
                )
                if source_key:
                    source_counts[source_key] += 1
                if family_key:
                    family_counts[family_key] += 1
                if section_key:
                    section_counts[section_key] += 1
                if madhhab_key:
                    madhhab_counts[madhhab_key] += 1

    if preserve_distinct_positions:
        requested_madhhabs = _requested_madhhabs(
            query_intent=query_intent,
            selected_madhhab="compare_all" if authority_policy_id == "compare_views" else "",
        )
        for madhhab in requested_madhhabs:
            for index, candidate in enumerate(ranked_candidates):
                if index in selected_indexes:
                    continue
                if not _is_requested_madhhab_candidate(candidate, madhhab):
                    continue
                if any(candidates_are_repetitive(candidate, prior) for prior in selected):
                    continue
                selected.append(candidate)
                selected_indexes.add(index)
                source_key = str(candidate.get("source_path", "")).strip().lower()
                family_key = get_duplicate_family_key(candidate)
                section_key = get_section_cluster_key(candidate)
                madhhab_key = _diversity_madhhab_key(
                    candidate,
                    query_intent,
                    preserve_distinct_positions=preserve_distinct_positions,
                )
                if source_key:
                    source_counts[source_key] += 1
                if family_key:
                    family_counts[family_key] += 1
                if section_key:
                    section_counts[section_key] += 1
                if madhhab_key:
                    madhhab_counts[madhhab_key] += 1
                break
            if len(selected) >= top_k:
                return selected

    for config in DIVERSITY_PASSES:
        for index, candidate in enumerate(ranked_candidates):
            if index in selected_indexes:
                continue
            source_key = str(candidate.get("source_path", "")).strip().lower()
            family_key = get_duplicate_family_key(candidate)
            section_key = get_section_cluster_key(candidate)
            madhhab_key = _diversity_madhhab_key(
                candidate,
                query_intent,
                preserve_distinct_positions=preserve_distinct_positions,
            )
            if source_key and source_counts[source_key] >= config["source_limit"]:
                continue
            if family_key and family_counts[family_key] >= config["family_limit"]:
                continue
            if section_key and section_counts[section_key] >= config["section_limit"]:
                continue
            if madhhab_key and madhhab_counts[madhhab_key] >= _madhhab_limit(
                config,
                query_intent,
                preserve_distinct_positions=preserve_distinct_positions,
            ):
                continue
            if not config["allow_repetitive"] and any(
                candidates_are_repetitive(candidate, prior)
                for prior in selected
            ):
                continue
            if (
                collapse_same_section_sequences
                and authority_policy_id in {"compare_views", "ruling_lookup", "hanafi_first"}
                and not config["allow_repetitive"]
                and any(
                    candidates_are_same_section_sequence(candidate, prior)
                    for prior in selected
                )
            ):
                continue
            selected.append(candidate)
            selected_indexes.add(index)
            if source_key:
                source_counts[source_key] += 1
            if family_key:
                family_counts[family_key] += 1
            if section_key:
                section_counts[section_key] += 1
            if madhhab_key:
                madhhab_counts[madhhab_key] += 1
            if len(selected) >= top_k:
                return selected

    for index, candidate in enumerate(ranked_candidates):
        if index in selected_indexes:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _select_study_path_layered_candidates(
    ranked_candidates: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_indexes: set[int] = set()
    selected_buckets: set[str] = set()
    target_buckets = (
        "quran",
        "hadith",
        "fiqh",
        "tasawwuf",
        "scholar_commentary",
    )

    for bucket in target_buckets:
        for index, candidate in enumerate(ranked_candidates):
            if index in selected_indexes:
                continue
            if _study_path_layer_bucket(candidate) != bucket:
                continue
            if any(candidates_are_repetitive(candidate, prior) for prior in selected):
                continue
            selected.append(candidate)
            selected_indexes.add(index)
            selected_buckets.add(bucket)
            break
        if len(selected) >= top_k:
            return selected

    for index, candidate in enumerate(ranked_candidates):
        if index in selected_indexes:
            continue
        bucket = _study_path_layer_bucket(candidate)
        if bucket and bucket in selected_buckets and len(selected_buckets) < len(target_buckets):
            continue
        if any(candidates_are_repetitive(candidate, prior) for prior in selected):
            continue
        selected.append(candidate)
        selected_indexes.add(index)
        if bucket:
            selected_buckets.add(bucket)
        if len(selected) >= top_k:
            break
    return selected


def _study_path_layer_bucket(candidate: dict[str, Any]) -> str:
    if _is_teaching_layer_candidate(candidate):
        return "scholar_commentary"
    source_classification = resolve_source_classification(candidate)
    if source_classification in {"quran", "hadith", "tasawwuf_text"}:
        return {
            "quran": "quran",
            "hadith": "hadith",
            "tasawwuf_text": "tasawwuf",
        }[source_classification]
    if source_classification == "fiqh_manual":
        return "fiqh"
    if source_classification in {"scholar_transcript", "transcript"}:
        return "scholar_commentary"
    return ""


def _family_canonical_rank(candidate: dict[str, Any]) -> tuple[int, int, int, int, int]:
    normalized_blob = _normalized_candidate_blob(candidate)
    extraction_status = str(candidate.get("extraction_status", "")).strip().lower()
    extraction_rank = {
        "": 0,
        "success": 0,
        "partial": 1,
        "failed": 2,
    }.get(extraction_status, 1)
    return (
        extraction_rank,
        _loader_hint_rank(candidate),
        _copy_title_rank(candidate),
        int(" mobile " in f" {normalized_blob} "),
        len(str(candidate.get("source_path", ""))),
    )


def _intent_authority_bonus(
    candidate: dict[str, Any],
    *,
    query_intent: QueryIntent | None,
    selected_madhhab: str,
    authority_policy_id: str,
) -> float:
    if query_intent is None:
        return 0.0

    source_classification = resolve_source_classification(candidate)
    bonus = 0.0
    teaching_layer_candidate = _is_teaching_layer_candidate(candidate)

    candidate_madhhab = str(candidate.get("madhhab", "")).strip().lower()
    if (
        query_intent.prefer_selected_madhhab
        and selected_madhhab not in {"", "not_specified", "compare_all"}
        and candidate_madhhab == selected_madhhab
        and source_classification in {"fiqh_manual", "commentary", "fatwa"}
        and authority_policy_id not in {"source_only"}
    ):
        bonus += 0.16
    if query_intent.intent_id == "compare_views":
        if candidate_madhhab and candidate_madhhab != selected_madhhab:
            bonus += 0.18
        if source_classification == "fiqh_manual":
            bonus += 0.18
        if source_classification == "tasawwuf_text":
            bonus -= 1.1
        if source_classification == "scholar_transcript":
            bonus -= 1.2
    if query_intent.intent_id in {"source_only", "direct_source_lookup"} and source_classification in {"fatwa", "transcript", "scholar_transcript"}:
        bonus -= 0.35
    if source_classification == "tasawwuf_text":
        if query_intent.target_spiritual_guidance:
            bonus += 0.28
        elif query_intent.intent_id in {"ruling_lookup", "fatwa_lookup"}:
            bonus -= 1.1
    if query_intent.intent_id == "transcript_lookup" and source_classification == "transcript":
        bonus += 0.9
    if source_classification == "scholar_transcript":
        if query_intent.intent_id in {"ruling_lookup", "fatwa_lookup"}:
            bonus -= 1.2
        elif query_intent.intent_id == "transcript_lookup":
            bonus += 0.95
        elif query_intent.target_scholar_commentary:
            bonus += 1.25
        elif query_intent.target_spiritual_guidance or query_intent.intent_id in {"explain_term", "summarize_source"}:
            bonus += 0.14
        else:
            bonus -= 0.24
    if teaching_layer_candidate:
        if query_intent.intent_id in {"source_only", "direct_source_lookup"}:
            bonus -= 1.4
        elif query_intent.intent_id == "ruling_lookup":
            bonus -= 0.7
        elif query_intent.intent_id == "compare_views":
            bonus -= 0.42
        elif query_intent.target_teaching_layer or authority_policy_id == "study_path":
            bonus += 0.7
        else:
            bonus += 0.12
    if query_intent.intent_id == "explain_term" and source_classification == "commentary":
        bonus += 0.16
    return bonus


def _madhhab_intent_bonus(
    candidate: dict[str, Any],
    *,
    query_intent: QueryIntent | None,
    selected_madhhab: str,
    authority_policy_id: str,
) -> float:
    if authority_policy_id == "source_only":
        return 0.0
    requested_madhhabs = _requested_madhhabs(
        query_intent=query_intent,
        selected_madhhab=selected_madhhab,
    )
    if not requested_madhhabs:
        return 0.0
    source_classification = resolve_source_classification(candidate)
    if source_classification not in SCHOOL_AUTHORITY_CLASSES:
        return 0.0
    candidate_madhhab = str(candidate.get("madhhab", "")).strip().lower()
    if not candidate_madhhab:
        return 0.0
    if candidate_madhhab in requested_madhhabs:
        if query_intent is not None and query_intent.intent_id == "compare_views":
            return 0.42 if source_classification == "fiqh_manual" else 0.28
        return 0.78 if source_classification == "fiqh_manual" else 0.52
    if len(requested_madhhabs) == 1:
        return -0.08
    return 0.0


def _worship_topic_bonus(
    candidate: dict[str, Any],
    *,
    query_intent: QueryIntent | None,
) -> float:
    if query_intent is None:
        return 0.0

    topic = str(query_intent.worship_topic or "").strip().lower()
    if topic not in {
        "prayer",
        "prayer_method",
        "purification",
        "purification_method",
        "prayer_with_purification_prerequisite",
    }:
        return 0.0

    focused_haystack = " ".join(
        str(candidate.get(field, "") or "")
        for field in ("title", "section_label", "reference", "quote")
    ).casefold()
    body_haystack = str(candidate.get("text", "") or "").casefold()
    prayer_focus_hits = _focus_term_matches(focused_haystack, PRAYER_FOCUS_TERMS)
    prayer_method_focus_hits = _focus_term_matches(
        focused_haystack, PRAYER_METHOD_TERMS
    )
    purification_focus_hits = _focus_term_matches(
        focused_haystack, PURIFICATION_FOCUS_TERMS
    )
    purification_method_focus_hits = _focus_term_matches(
        focused_haystack, PURIFICATION_METHOD_TERMS
    )
    prayer_method_distractor_hits = _focus_term_matches(
        focused_haystack,
        PRAYER_METHOD_DISTRACTOR_TERMS,
    )
    purification_method_distractor_hits = _focus_term_matches(
        focused_haystack,
        PURIFICATION_METHOD_DISTRACTOR_TERMS,
    )
    prerequisite_focus_hits = _focus_term_matches(
        focused_haystack,
        PRAYER_PREREQUISITE_TERMS,
    )
    prayer_body_hits = _focus_term_matches(body_haystack, PRAYER_FOCUS_TERMS)
    prayer_method_body_hits = _focus_term_matches(body_haystack, PRAYER_METHOD_TERMS)
    purification_body_hits = _focus_term_matches(
        body_haystack, PURIFICATION_FOCUS_TERMS
    )
    purification_method_body_hits = _focus_term_matches(
        body_haystack, PURIFICATION_METHOD_TERMS
    )
    prayer_method_body_distractor_hits = _focus_term_matches(
        body_haystack,
        PRAYER_METHOD_DISTRACTOR_TERMS,
    )
    purification_method_body_distractor_hits = _focus_term_matches(
        body_haystack,
        PURIFICATION_METHOD_DISTRACTOR_TERMS,
    )
    prerequisite_body_hits = _focus_term_matches(
        body_haystack,
        PRAYER_PREREQUISITE_TERMS,
    )
    source_classification = resolve_source_classification(candidate)

    if topic == "prayer_method":
        distractor_hits = prayer_method_distractor_hits + prayer_method_body_distractor_hits
        if prayer_method_focus_hits:
            bonus = 0.88 + min(prayer_method_focus_hits, 3) * 0.12
            if distractor_hits and prayer_method_focus_hits < 2:
                bonus -= 0.72 + min(distractor_hits, 3) * 0.08
            return bonus
        if prayer_method_body_hits >= 2:
            bonus = 0.28
            if distractor_hits:
                bonus -= 0.36
            return bonus
        if prayer_focus_hits:
            if distractor_hits:
                return -0.42
            return 0.08
        if purification_focus_hits or purification_method_focus_hits:
            return -0.72
        if source_classification in {"fiqh_manual", "hadith", "commentary"}:
            return -0.28
        return 0.0

    if topic == "prayer":
        if prayer_focus_hits and not purification_focus_hits:
            return 0.58
        if prayer_focus_hits and purification_focus_hits:
            return 0.22
        if prayer_body_hits and not purification_focus_hits:
            return 0.06
        if purification_focus_hits and not prayer_focus_hits:
            return -0.62
        if (
            purification_body_hits
            and not prayer_focus_hits
            and not prayer_body_hits
        ):
            return -0.24
        if source_classification in {"fiqh_manual", "hadith", "commentary"}:
            return -0.22
        return 0.0

    if topic == "purification_method":
        distractor_hits = (
            purification_method_distractor_hits + purification_method_body_distractor_hits
        )
        if purification_method_focus_hits:
            bonus = 0.88 + min(purification_method_focus_hits, 3) * 0.12
            if distractor_hits and purification_method_focus_hits < 2:
                bonus -= 0.76 + min(distractor_hits, 3) * 0.08
            return bonus
        if purification_method_body_hits >= 2:
            bonus = 0.24
            if distractor_hits:
                bonus -= 0.42
            return bonus
        if purification_focus_hits:
            if distractor_hits:
                return -0.38
            return 0.08
        if prayer_focus_hits or prayer_method_focus_hits:
            return -0.48
        if source_classification in {"fiqh_manual", "hadith", "commentary"}:
            return -0.24
        return 0.0

    if topic == "purification":
        if purification_focus_hits and not prayer_focus_hits:
            return 0.58
        if purification_focus_hits and prayer_focus_hits:
            return 0.22
        if purification_body_hits and not prayer_focus_hits:
            return 0.04
        if prayer_focus_hits and not purification_focus_hits:
            return -0.36
        if (
            prayer_body_hits
            and not purification_focus_hits
            and not purification_body_hits
        ):
            return -0.16
        if source_classification in {"fiqh_manual", "hadith", "commentary"}:
            return -0.22
        return 0.0

    if prayer_focus_hits and purification_focus_hits:
        bonus = 0.34
        if prerequisite_focus_hits or prerequisite_body_hits:
            bonus += 0.26
        return bonus
    if prayer_focus_hits and purification_body_hits:
        bonus = 0.24
        if prerequisite_focus_hits or prerequisite_body_hits:
            bonus += 0.18
        return bonus
    if purification_focus_hits and prayer_body_hits:
        bonus = 0.28
        if prerequisite_focus_hits or prerequisite_body_hits:
            bonus += 0.18
        return bonus
    if prayer_focus_hits:
        return 0.16
    if purification_focus_hits:
        return 0.24
    return 0.0


def _focus_term_matches(text: str, terms: set[str]) -> int:
    matches = 0
    for term in terms:
        if " " in term:
            if term in text:
                matches += 1
            continue
        if re.search(rf"\b{re.escape(term)}\b", text):
            matches += 1
    return matches


def _diversity_madhhab_key(
    candidate: dict[str, Any],
    query_intent: QueryIntent | None,
    *,
    preserve_distinct_positions: bool,
) -> str:
    if not preserve_distinct_positions:
        return ""
    candidate_madhhab = str(candidate.get("madhhab", "")).strip().lower()
    source_classification = resolve_source_classification(candidate)
    if not candidate_madhhab or source_classification not in SCHOOL_AUTHORITY_CLASSES:
        return ""
    return candidate_madhhab


def _madhhab_limit(
    config: dict[str, Any],
    query_intent: QueryIntent | None,
    *,
    preserve_distinct_positions: bool,
) -> int:
    if not preserve_distinct_positions:
        return 999
    if not config["allow_repetitive"]:
        return 1
    return 2


def candidates_are_same_section_sequence(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_source = str(left.get("source_path", "") or "").strip().lower()
    right_source = str(right.get("source_path", "") or "").strip().lower()
    left_family = get_duplicate_family_key(left)
    right_family = get_duplicate_family_key(right)
    same_scope = bool(
        (left_source and left_source == right_source)
        or (left_family and left_family == right_family)
    )
    if not same_scope:
        return False

    left_section = normalize_title_key(str(left.get("section_label", "") or ""))
    right_section = normalize_title_key(str(right.get("section_label", "") or ""))
    if not left_section or left_section != right_section:
        return False

    left_page = _page_number(left)
    right_page = _page_number(right)
    if left_page is not None and right_page is not None and abs(left_page - right_page) <= 2:
        return True

    left_page_ref = str(left.get("page_reference", "") or "")
    right_page_ref = str(right.get("page_reference", "") or "")
    if left_page_ref and left_page_ref == right_page_ref:
        return True

    left_chunk = int(left.get("page_chunk_index", 0) or 0)
    right_chunk = int(right.get("page_chunk_index", 0) or 0)
    if left_chunk and right_chunk and abs(left_chunk - right_chunk) == 1:
        return True
    return False


def _requested_madhhabs(
    *,
    query_intent: QueryIntent | None,
    selected_madhhab: str,
) -> tuple[str, ...]:
    if query_intent is not None and query_intent.requested_madhhabs:
        return tuple(query_intent.requested_madhhabs)
    normalized = str(selected_madhhab or "").strip().lower()
    if normalized == "compare_all":
        return SUPPORTED_MADHHABS
    if normalized in SUPPORTED_MADHHABS:
        return (normalized,)
    return ()


def _madhhab_intent_rank(
    candidate: dict[str, Any],
    *,
    query_intent: QueryIntent | None,
    selected_madhhab: str,
) -> int:
    requested_madhhabs = _requested_madhhabs(
        query_intent=query_intent,
        selected_madhhab=selected_madhhab,
    )
    if not requested_madhhabs:
        return 1
    source_classification = resolve_source_classification(candidate)
    if source_classification not in SCHOOL_AUTHORITY_CLASSES:
        return 1
    candidate_madhhab = str(candidate.get("madhhab", "")).strip().lower()
    if candidate_madhhab in requested_madhhabs:
        return 0
    if candidate_madhhab:
        return 2
    return 1


def _teaching_rank_floor(candidate: dict[str, Any]) -> int:
    if _is_teaching_layer_candidate(candidate):
        return 1
    return 0


def _is_requested_madhhab_candidate(candidate: dict[str, Any], madhhab: str) -> bool:
    source_classification = resolve_source_classification(candidate)
    if source_classification not in SCHOOL_AUTHORITY_CLASSES:
        return False
    if _is_teaching_layer_candidate(candidate):
        return False
    return str(candidate.get("madhhab", "")).strip().lower() == madhhab


def _is_teaching_layer_candidate(candidate: dict[str, Any]) -> bool:
    source_role_boundary = str(candidate.get("source_role_boundary", "") or "").strip().lower()
    source_family = str(candidate.get("source_family", "") or "").strip().lower()
    return source_role_boundary == "teaching_layer" or source_family == "classes"


def _loader_hint_bonus(candidate: dict[str, Any]) -> float:
    loader_hint = str(candidate.get("loader_hint", "")).strip().lower()
    if loader_hint == "quran_translation_csv":
        return 1.0
    if loader_hint == "quran_ayah_text":
        return 0.85
    if loader_hint in {"plain_text", "csv_rows"}:
        return 0.25
    return 0.0


def _loader_hint_rank(candidate: dict[str, Any]) -> int:
    loader_hint = str(candidate.get("loader_hint", "")).strip().lower()
    if loader_hint == "quran_translation_csv":
        return 0
    if loader_hint == "quran_ayah_text":
        return 1
    if loader_hint in {"plain_text", "csv_rows"}:
        return 2
    if loader_hint == "normalized_pdf_json":
        return 3
    return 4


def _family_preference_bonus(candidate: dict[str, Any]) -> float:
    family_key = get_duplicate_family_key(candidate)
    loader_hint = str(candidate.get("loader_hint", "")).strip().lower()
    normalized_blob = _normalized_candidate_blob(candidate)
    bonus = 0.0

    if family_key.endswith("|english_rwwad"):
        if loader_hint == "quran_translation_csv":
            bonus += 0.65
        if " pure " in f" {normalized_blob} " and not _copy_title_rank(candidate):
            bonus += 0.12
    if family_key.endswith("|mukhtasar_al_quduri") and "mukhtasar" in normalized_blob:
        bonus += 0.18
    if family_key.endswith("|muwatta_imam_malik") and " al muwatta " in f" {normalized_blob} ":
        bonus += 0.08
    return bonus


def _normalized_candidate_blob(candidate: dict[str, Any]) -> str:
    combined = " ".join(
        str(candidate.get(field, "") or "")
        for field in ("title", "source_path", "collection", "section_label")
    )
    normalized = unicodedata.normalize("NFKC", combined).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _candidate_text_fingerprint(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("text", "") or "")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:360]


def _significant_text_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    filtered: list[str] = []
    for token in tokens:
        if token in REPETITIVE_STOPWORDS:
            continue
        if token.isascii() and len(token) < 3:
            continue
        filtered.append(token)
    return filtered


def _token_jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _prefix_overlap(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    common = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        common += 1
    return common / max(len(left), len(right))


def _page_number(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("page_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _copy_title_rank(candidate: dict[str, Any]) -> int:
    title = str(candidate.get("title", ""))
    return int(bool(re.search(r"\(\d+\)$", title.strip())))
