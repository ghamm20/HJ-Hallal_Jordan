"""Simple keyword-first retrieval over local chunks."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.reasoning.intent_router import QueryIntent

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}
PRAYER_FOCUS_TERMS = {
    "pray",
    "prayer",
    "prayers",
    "salat",
    "salah",
    "rakah",
    "rakahs",
    "rakat",
    "rakats",
    "takbir",
    "sujud",
    "sajdah",
    "ruku",
    "tashahhud",
    "witr",
    "fajr",
    "dhuhr",
    "asr",
    "maghrib",
    "isha",
    "travellers' prayers",
    "times of the prayers",
    "characteristics of prayer",
    "chapters on salat",
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
    "dry ablution",
    "impurity",
    "najasa",
    "book of purification",
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
    "purity",
    "state of purity",
}

SEARCH_CACHE_KEYS = (
    "_search_haystack_normalized",
    "_search_tokens",
    "_search_token_counts",
    "_search_token_set",
    "_search_title_tokens",
    "_search_title_token_set",
    "_search_reference_tokens",
    "_search_reference_token_set",
    "_search_section_tokens",
    "_search_section_token_set",
    "_search_source_type_tokens",
    "_search_source_type_token_set",
)

SCHOOL_AUTHORITY_CLASSES = {"fiqh_manual", "commentary", "fatwa"}
SEARCH_INDEX_CACHE_LIMIT = 64


@dataclass(slots=True)
class PreparedSearchIndex:
    token_to_chunk_ids: dict[str, tuple[int, ...]]
    authority_chunk_ids_by_madhhab: dict[str, tuple[int, ...]]
    source_classification_chunk_ids: dict[str, tuple[int, ...]]
    primary_chunk_ids: tuple[int, ...]
    chunk_count: int
    query_cache: dict[tuple[str, str, int], tuple[int, ...]] = field(default_factory=dict)


_PREPARED_INDEXES: dict[int, PreparedSearchIndex] = {}


def prepare_search_index(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Precompute normalized search fields once so repeated queries stay cheap."""
    prepared_index = _build_prepared_search_index(chunks)
    _register_prepared_search_index(chunks, prepared_index)
    return chunks


def register_prepared_search_index(
    chunks: list[dict[str, Any]],
    *,
    token_to_chunk_ids: dict[str, tuple[int, ...]],
    authority_chunk_ids_by_madhhab: dict[str, tuple[int, ...]],
    source_classification_chunk_ids: dict[str, tuple[int, ...]],
    primary_chunk_ids: tuple[int, ...],
) -> PreparedSearchIndex:
    prepared_index = PreparedSearchIndex(
        token_to_chunk_ids=dict(token_to_chunk_ids),
        authority_chunk_ids_by_madhhab=dict(authority_chunk_ids_by_madhhab),
        source_classification_chunk_ids=dict(source_classification_chunk_ids),
        primary_chunk_ids=tuple(primary_chunk_ids),
        chunk_count=len(chunks),
    )
    _register_prepared_search_index(chunks, prepared_index)
    return prepared_index


def export_prepared_search_metadata(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    prepared_index = _get_prepared_search_index(chunks)
    return {
        "token_postings": [
            {"token": token, "chunk_ids": list(indices)}
            for token, indices in sorted(prepared_index.token_to_chunk_ids.items())
        ],
        "authority_chunk_ids_by_madhhab": {
            madhhab: list(indices)
            for madhhab, indices in sorted(
                prepared_index.authority_chunk_ids_by_madhhab.items()
            )
        },
        "source_classification_chunk_ids": {
            source_classification: list(indices)
            for source_classification, indices in sorted(
                prepared_index.source_classification_chunk_ids.items()
            )
        },
        "primary_chunk_ids": list(prepared_index.primary_chunk_ids),
        "chunk_count": prepared_index.chunk_count,
        "token_count": len(prepared_index.token_to_chunk_ids),
    }


def _build_prepared_search_index(chunks: list[dict[str, Any]]) -> PreparedSearchIndex:
    token_to_chunk_ids: dict[str, list[int]] = {}
    authority_chunk_ids_by_madhhab: dict[str, list[int]] = {}
    source_classification_chunk_ids: dict[str, list[int]] = {}
    primary_chunk_ids: list[int] = []

    for index, chunk in enumerate(chunks):
        cached_fields = _cached_search_fields(chunk)
        for token in cached_fields["token_set"]:
            token_to_chunk_ids.setdefault(token, []).append(index)
        source_classification = str(
            chunk.get("source_classification") or chunk.get("source_type") or ""
        ).strip().lower()
        if source_classification:
            source_classification_chunk_ids.setdefault(source_classification, []).append(index)
        if source_classification in {"quran", "hadith"}:
            primary_chunk_ids.append(index)
        if source_classification in SCHOOL_AUTHORITY_CLASSES:
            madhhab = str(chunk.get("madhhab", "") or "").strip().lower()
            if madhhab:
                authority_chunk_ids_by_madhhab.setdefault(madhhab, []).append(index)

    return PreparedSearchIndex(
        token_to_chunk_ids={
            token: tuple(indices) for token, indices in token_to_chunk_ids.items()
        },
        authority_chunk_ids_by_madhhab={
            madhhab: tuple(indices)
            for madhhab, indices in authority_chunk_ids_by_madhhab.items()
        },
        source_classification_chunk_ids={
            source_classification: tuple(indices)
            for source_classification, indices in source_classification_chunk_ids.items()
        },
        primary_chunk_ids=tuple(primary_chunk_ids),
        chunk_count=len(chunks),
    )


def _register_prepared_search_index(
    chunks: list[dict[str, Any]],
    prepared_index: PreparedSearchIndex,
) -> None:
    _PREPARED_INDEXES[id(chunks)] = prepared_index


def search_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int = 25,
    query_intent: QueryIntent | None = None,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    question_normalized = _normalize_text(question)
    question_tokens = _tokenize(question_normalized)
    prepared_index = _get_prepared_search_index(chunks)
    candidate_indices = _candidate_indices_for_query(
        prepared_index,
        question_normalized=question_normalized,
        question_tokens=question_tokens,
        query_intent=query_intent,
        top_k=top_k,
    )
    candidates: list[dict[str, Any]] = []

    for position, chunk_index in enumerate(candidate_indices, start=1):
        if deadline is not None and position % 128 == 0 and perf_counter() >= deadline:
            raise TimeoutError("retrieval_search_timeout")
        chunk = chunks[chunk_index]
        retrieval_score = _score_chunk(
            question_normalized=question_normalized,
            question_tokens=question_tokens,
            chunk=chunk,
            query_intent=query_intent,
        )
        if retrieval_score <= 0:
            continue
        candidate = dict(chunk)
        candidate["retrieval_score"] = retrieval_score
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -float(item.get("retrieval_score", 0)),
            str(item.get("reference", "")).lower(),
            str(item.get("title", "")).lower(),
        )
    )
    return candidates[:top_k]


def semantic_search(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    vector_store: Any,
    embedder: Any,
    top_k: int = 25,
    query_intent: QueryIntent | None = None,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("retrieval_search_timeout")
    chunk_by_id = {
        str(chunk.get("chunk_id", "") or "").strip(): chunk
        for chunk in chunks
        if str(chunk.get("chunk_id", "") or "").strip()
    }
    query_vector = embedder.encode_query(question)
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("retrieval_search_timeout")
    matches = vector_store.search(query_vector=query_vector, top_k=top_k)
    candidates: list[dict[str, Any]] = []
    for match in matches:
        if deadline is not None and perf_counter() >= deadline:
            raise TimeoutError("retrieval_search_timeout")
        chunk_id = str(match.get("chunk_id", "") or "").strip()
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        candidate = dict(chunk)
        candidate["retrieval_score"] = float(match.get("score", 0.0))
        candidate["semantic_score"] = float(match.get("score", 0.0))
        candidate["keyword_score"] = 0.0
        candidate["retrieval_method"] = "semantic"
        if query_intent is not None:
            candidate["semantic_intent_id"] = query_intent.intent_id
        candidates.append(candidate)
    return candidates


def hybrid_search(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    vector_store: Any,
    embedder: Any,
    top_k: int = 25,
    query_intent: QueryIntent | None = None,
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    keyword_candidates = search_chunks(
        question,
        chunks,
        top_k=top_k,
        query_intent=query_intent,
        deadline=deadline,
    )
    semantic_candidates = semantic_search(
        question,
        chunks,
        vector_store=vector_store,
        embedder=embedder,
        top_k=top_k,
        query_intent=query_intent,
        deadline=deadline,
    )
    merged: dict[str, dict[str, Any]] = {}
    _apply_rank_scores(
        merged,
        keyword_candidates,
        method_key="keyword_score",
        weight=keyword_weight,
    )
    _apply_rank_scores(
        merged,
        semantic_candidates,
        method_key="semantic_score",
        weight=semantic_weight,
    )
    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -float(item.get("retrieval_score", 0.0)),
            -float(item.get("semantic_score", 0.0)),
            -float(item.get("keyword_score", 0.0)),
            str(item.get("reference", "")).lower(),
            str(item.get("title", "")).lower(),
        ),
    )
    for candidate in ranked:
        keyword_score = float(candidate.get("keyword_score", 0.0))
        semantic_score = float(candidate.get("semantic_score", 0.0))
        if keyword_score > 0 and semantic_score > 0:
            candidate["retrieval_method"] = "hybrid"
        elif semantic_score > 0:
            candidate["retrieval_method"] = "semantic"
        else:
            candidate["retrieval_method"] = "keyword"
    return ranked[:top_k]


def _score_chunk(
    *,
    question_normalized: str,
    question_tokens: list[str],
    chunk: dict[str, Any],
    query_intent: QueryIntent | None,
) -> float:
    cached_fields = _cached_search_fields(chunk)
    haystack_normalized = str(cached_fields["haystack_normalized"])
    haystack_tokens = list(cached_fields["haystack_tokens"])
    if not haystack_tokens:
        return 0.0
    source_classification = str(
        chunk.get("source_classification") or chunk.get("source_type") or ""
    ).strip().lower()
    if (
        query_intent is not None
        and source_classification == "tasawwuf_text"
        and query_intent.intent_id in {"ruling_lookup", "compare_views", "fatwa_lookup"}
    ):
        return 0.0
    if (
        query_intent is not None
        and source_classification == "scholar_transcript"
        and query_intent.intent_id in {"ruling_lookup", "compare_views", "fatwa_lookup"}
    ):
        return 0.0

    haystack_counts = cached_fields["haystack_counts"]
    unique_question_tokens = set(question_tokens)
    token_score = 0.0
    for token in unique_question_tokens:
        count = haystack_counts.get(token, 0)
        if count <= 0:
            continue
        token_score += 1.0 + min(count - 1, 2) * 0.2

    if question_normalized and question_normalized in haystack_normalized:
        token_score += 4.0

    title_tokens = cached_fields["title_token_set"]
    reference_tokens = cached_fields["reference_token_set"]
    section_tokens = cached_fields["section_token_set"]
    source_type_tokens = cached_fields["source_type_token_set"]
    token_score += 0.75 * len(unique_question_tokens & title_tokens)
    token_score += 0.5 * len(unique_question_tokens & reference_tokens)
    token_score += 0.85 * len(unique_question_tokens & section_tokens)
    token_score += 0.25 * len(unique_question_tokens & source_type_tokens)
    section_normalized = _normalize_text(str(chunk.get("section_label", "")))
    if question_normalized and section_normalized and question_normalized in section_normalized:
        token_score += 2.5
    token_score += _intent_bonus(chunk, query_intent)
    token_score += _worship_topic_bonus(
        cached_fields=cached_fields,
        source_classification=source_classification,
        query_intent=query_intent,
    )
    return token_score


def _cached_search_fields(chunk: dict[str, Any]) -> dict[str, Any]:
    if all(key in chunk for key in SEARCH_CACHE_KEYS):
        return {
            "haystack_normalized": chunk["_search_haystack_normalized"],
            "haystack_tokens": chunk["_search_tokens"],
            "haystack_counts": chunk["_search_token_counts"],
            "token_set": chunk["_search_token_set"],
            "title_tokens": chunk["_search_title_tokens"],
            "title_token_set": chunk["_search_title_token_set"],
            "reference_tokens": chunk["_search_reference_tokens"],
            "reference_token_set": chunk["_search_reference_token_set"],
            "section_tokens": chunk["_search_section_tokens"],
            "section_token_set": chunk["_search_section_token_set"],
            "source_type_tokens": chunk["_search_source_type_tokens"],
            "source_type_token_set": chunk["_search_source_type_token_set"],
        }

    normalized_search_text = str(chunk.get("normalized_search_text", "") or "").strip()
    if normalized_search_text:
        haystack_normalized = normalized_search_text
    else:
        haystack_parts = [
            str(chunk.get("title", "")),
            str(chunk.get("collection", "")),
            str(chunk.get("author", "")),
            str(chunk.get("reference", "")),
            str(chunk.get("section_label", "")),
            str(chunk.get("book", "")),
            str(chunk.get("chapter", "")),
            str(chunk.get("section", "")),
            str(chunk.get("quote", "")),
            str(chunk.get("text", "")),
            str(chunk.get("source_type", "")),
            str(chunk.get("source_classification", "")),
            str(chunk.get("source_family", "")),
            str(chunk.get("canonical_family", "")),
            str(chunk.get("document_kind", "")),
            str(chunk.get("role", "")),
            str(chunk.get("domain", "")),
            str(chunk.get("authority_level", "")),
            str(chunk.get("source_role_flag", "")),
            str(chunk.get("source_role_boundary", "")),
            str(chunk.get("commentary_target", "")),
            str(chunk.get("fatwa_authority", "")),
            str(chunk.get("madhhab", "")),
            _source_descriptor_text(chunk),
        ]
        haystack_normalized = _normalize_text(" ".join(haystack_parts))
    haystack_tokens = tuple(_tokenize(haystack_normalized))
    haystack_counts = Counter(haystack_tokens)
    token_set = frozenset(haystack_counts)
    title_tokens = tuple(_tokenize(str(chunk.get("title", ""))))
    title_token_set = frozenset(title_tokens)
    reference_tokens = tuple(_tokenize(str(chunk.get("reference", ""))))
    reference_token_set = frozenset(reference_tokens)
    section_tokens = tuple(_tokenize(str(chunk.get("section_label", ""))))
    section_token_set = frozenset(section_tokens)
    source_type_tokens = tuple(_tokenize(str(chunk.get("source_type", ""))))
    source_type_token_set = frozenset(source_type_tokens)
    chunk["_search_haystack_normalized"] = haystack_normalized
    chunk["_search_tokens"] = haystack_tokens
    chunk["_search_token_counts"] = haystack_counts
    chunk["_search_token_set"] = token_set
    chunk["_search_title_tokens"] = title_tokens
    chunk["_search_title_token_set"] = title_token_set
    chunk["_search_reference_tokens"] = reference_tokens
    chunk["_search_reference_token_set"] = reference_token_set
    chunk["_search_section_tokens"] = section_tokens
    chunk["_search_section_token_set"] = section_token_set
    chunk["_search_source_type_tokens"] = source_type_tokens
    chunk["_search_source_type_token_set"] = source_type_token_set
    return {
        "haystack_normalized": haystack_normalized,
        "haystack_tokens": haystack_tokens,
        "haystack_counts": haystack_counts,
        "token_set": token_set,
        "title_tokens": title_tokens,
        "title_token_set": title_token_set,
        "reference_tokens": reference_tokens,
        "reference_token_set": reference_token_set,
        "section_tokens": section_tokens,
        "section_token_set": section_token_set,
        "source_type_tokens": source_type_tokens,
        "source_type_token_set": source_type_token_set,
    }


def _intent_bonus(chunk: dict[str, Any], query_intent: QueryIntent | None) -> float:
    if query_intent is None:
        return 0.0

    source_classification = str(
        chunk.get("source_classification") or chunk.get("source_type") or ""
    ).strip().lower()
    document_kind = str(chunk.get("document_kind", "") or "").strip().lower()
    section_kind = str(chunk.get("section_kind", "") or "").strip().lower()

    bonus = 0.0
    authority_index = query_intent.authority_order.index(source_classification) if source_classification in query_intent.authority_order else len(query_intent.authority_order)
    bonus += max(0, len(query_intent.authority_order) - authority_index) * 0.08

    if source_classification == "tasawwuf_text":
        if query_intent.intent_id in {"ruling_lookup", "compare_views", "fatwa_lookup"}:
            return -10.0
        if query_intent.target_spiritual_guidance:
            bonus += 0.6
        else:
            bonus -= 0.5
    if source_classification == "scholar_transcript":
        if query_intent.intent_id in {"ruling_lookup", "compare_views", "fatwa_lookup"}:
            return -10.0
        if query_intent.target_scholar_commentary:
            bonus += 1.45
        elif query_intent.target_spiritual_guidance or query_intent.intent_id in {"explain_term", "summarize_source"}:
            bonus += 0.18
        else:
            bonus -= 0.35

    if query_intent.prefer_primary_texts and source_classification in {"quran", "hadith"}:
        bonus += 0.3
    if query_intent.prefer_selected_madhhab and str(chunk.get("madhhab", "")).strip():
        bonus += 0.15
    if query_intent.prefer_definitional_material and section_kind in {
        "book_heading",
        "chapter_heading",
        "major_heading",
        "numbered_section",
    }:
        bonus += 0.2
    if query_intent.suppress_synthesis and source_classification in {"fatwa", "transcript", "scholar_transcript"}:
        bonus -= 0.4
    if query_intent.intent_id == "fatwa_lookup" and source_classification == "fatwa":
        bonus += 0.45
    if query_intent.intent_id == "transcript_lookup" and source_classification in {"transcript", "scholar_transcript"}:
        bonus += 0.9
    if query_intent.intent_id == "explain_term" and document_kind in {"tafsir", "sharh"}:
        bonus += 0.15
    return bonus


def _worship_topic_bonus(
    *,
    cached_fields: dict[str, Any],
    source_classification: str,
    query_intent: QueryIntent | None,
) -> float:
    if query_intent is None:
        return 0.0

    topic = getattr(query_intent, "worship_topic", "general")
    if topic == "general":
        return 0.0

    haystack_normalized = str(cached_fields["haystack_normalized"])
    prayer_hits = _focus_term_matches(haystack_normalized, PRAYER_FOCUS_TERMS)
    prayer_method_hits = _focus_term_matches(haystack_normalized, PRAYER_METHOD_TERMS)
    prayer_method_distractor_hits = _focus_term_matches(
        haystack_normalized,
        PRAYER_METHOD_DISTRACTOR_TERMS,
    )
    purification_hits = _focus_term_matches(
        haystack_normalized,
        PURIFICATION_FOCUS_TERMS,
    )
    purification_method_hits = _focus_term_matches(
        haystack_normalized,
        PURIFICATION_METHOD_TERMS,
    )
    purification_method_distractor_hits = _focus_term_matches(
        haystack_normalized,
        PURIFICATION_METHOD_DISTRACTOR_TERMS,
    )
    prerequisite_hits = _focus_term_matches(haystack_normalized, PRAYER_PREREQUISITE_TERMS)
    bonus = 0.0

    if topic == "prayer_method":
        if prayer_method_hits:
            bonus += 1.95 + min(prayer_method_hits, 4) * 0.28
            if source_classification == "fiqh_manual":
                bonus += 0.22
        elif prayer_hits:
            bonus += 0.5 + min(prayer_hits, 3) * 0.1
        if prayer_method_distractor_hits and prayer_method_hits < 2:
            bonus -= 1.9 + min(prayer_method_distractor_hits, 3) * 0.2
        if purification_hits and not prayer_method_hits:
            bonus -= 1.8 + min(purification_hits, 3) * 0.18
        if source_classification == "fiqh_manual" and not prayer_method_hits:
            bonus -= 0.65
        return bonus

    if topic == "prayer":
        if prayer_hits:
            bonus += 1.45 + min(prayer_hits, 4) * 0.25
            if source_classification == "fiqh_manual":
                bonus += 0.25
        if purification_hits and not prayer_hits:
            bonus -= 1.7 + min(purification_hits, 3) * 0.15
        if source_classification == "fiqh_manual" and not prayer_hits:
            bonus -= 0.45
        return bonus

    if topic == "purification_method":
        if purification_method_hits:
            bonus += 1.95 + min(purification_method_hits, 4) * 0.28
            if source_classification == "fiqh_manual":
                bonus += 0.22
        elif purification_hits:
            bonus += 0.5 + min(purification_hits, 3) * 0.1
        if purification_method_distractor_hits and purification_method_hits < 2:
            bonus -= 1.95 + min(purification_method_distractor_hits, 3) * 0.22
        if prayer_hits and not purification_method_hits:
            bonus -= 1.25 + min(prayer_hits, 3) * 0.12
        if source_classification == "fiqh_manual" and not purification_method_hits:
            bonus -= 0.48
        return bonus

    if topic == "purification":
        if purification_hits:
            bonus += 1.45 + min(purification_hits, 4) * 0.25
            if source_classification == "fiqh_manual":
                bonus += 0.25
        if prayer_hits and not purification_hits:
            bonus -= 1.1 + min(prayer_hits, 3) * 0.1
        if source_classification == "fiqh_manual" and not purification_hits:
            bonus -= 0.3
        return bonus

    if topic == "prayer_with_purification_prerequisite":
        if prayer_hits and purification_hits:
            bonus += 1.3
            if prerequisite_hits:
                bonus += 0.5 + min(prerequisite_hits, 3) * 0.1
        elif prayer_hits or purification_hits:
            bonus += 0.45
            if prerequisite_hits:
                bonus += 0.18
        return bonus

    return 0.0


def _apply_rank_scores(
    merged: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    method_key: str,
    weight: float,
) -> None:
    if not candidates or weight <= 0:
        return
    total = len(candidates)
    for index, candidate in enumerate(candidates):
        chunk_id = str(candidate.get("chunk_id", "") or "").strip()
        if not chunk_id:
            continue
        payload = merged.setdefault(chunk_id, dict(candidate))
        rank_score = weight * (total - index) / total
        payload[method_key] = float(payload.get(method_key, 0.0)) + rank_score
        payload["retrieval_score"] = float(payload.get("retrieval_score", 0.0)) + rank_score
        payload.setdefault("keyword_score", 0.0)
        payload.setdefault("semantic_score", 0.0)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", normalized).strip()


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


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    filtered: list[str] = []
    for token in tokens:
        if token in ENGLISH_STOPWORDS:
            continue
        if token.isascii() and len(token) < 2:
            continue
        filtered.append(token)
    return filtered


def _source_descriptor_text(chunk: dict[str, Any]) -> str:
    source_classification = str(
        chunk.get("source_classification") or chunk.get("source_type") or ""
    ).strip().lower()
    if source_classification == "scholar_transcript":
        return "scholar commentary teacher explanation transcript"
    if source_classification == "tasawwuf_text":
        return "spiritual guidance tasawwuf classical sufism"
    if source_classification == "fiqh_manual":
        return "fiqh manual madhhab authority"
    return ""


def _get_prepared_search_index(chunks: list[dict[str, Any]]) -> PreparedSearchIndex:
    prepared = _PREPARED_INDEXES.get(id(chunks))
    if prepared is None or prepared.chunk_count != len(chunks):
        prepare_search_index(chunks)
        prepared = _PREPARED_INDEXES[id(chunks)]
    return prepared


def _candidate_indices_for_query(
    prepared_index: PreparedSearchIndex,
    *,
    question_normalized: str,
    question_tokens: list[str],
    query_intent: QueryIntent | None,
    top_k: int,
) -> tuple[int, ...]:
    cache_key = (
        question_normalized,
        _query_cache_signature(query_intent),
        top_k,
    )
    cached = prepared_index.query_cache.get(cache_key)
    if cached is not None:
        return cached

    candidate_scores: Counter[int] = Counter()
    query_terms = _query_index_terms(question_tokens, query_intent)
    postings = [
        (term, prepared_index.token_to_chunk_ids.get(term, ()))
        for term in query_terms
        if prepared_index.token_to_chunk_ids.get(term)
    ]
    postings.sort(key=lambda item: (len(item[1]), item[0]))
    for rank, (_term, indices) in enumerate(postings[:6], start=1):
        bonus = max(0.35, 1.35 - (rank - 1) * 0.15)
        for chunk_index in indices:
            candidate_scores[chunk_index] += bonus

    metadata_indices = _metadata_candidate_indices(prepared_index, query_intent)
    for chunk_index in metadata_indices:
        candidate_scores[chunk_index] += 0.2

    if not candidate_scores:
        result = tuple(range(prepared_index.chunk_count))
    else:
        max_candidates = _max_scored_candidate_count(top_k=top_k, match_count=len(candidate_scores))
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        result = tuple(chunk_index for chunk_index, _score in ranked[:max_candidates])

    prepared_index.query_cache[cache_key] = result
    if len(prepared_index.query_cache) > SEARCH_INDEX_CACHE_LIMIT:
        oldest_key = next(iter(prepared_index.query_cache))
        prepared_index.query_cache.pop(oldest_key, None)
    return result


def _query_cache_signature(query_intent: QueryIntent | None) -> str:
    if query_intent is None:
        return "none"
    requested_madhhabs = ",".join(sorted(query_intent.requested_madhhabs))
    return "|".join(
        (
            query_intent.intent_id,
            query_intent.worship_topic,
            requested_madhhabs,
            "spiritual" if query_intent.target_spiritual_guidance else "",
            "scholar" if query_intent.target_scholar_commentary else "",
            "teaching" if query_intent.target_teaching_layer else "",
            "primary" if query_intent.prefer_primary_texts else "",
            "define" if query_intent.prefer_definitional_material else "",
            "nosynth" if query_intent.suppress_synthesis else "",
        )
    )


def _query_index_terms(
    question_tokens: list[str],
    query_intent: QueryIntent | None,
) -> tuple[str, ...]:
    terms = set(question_tokens)
    if query_intent is not None:
        terms.update(query_intent.requested_madhhabs)
        if query_intent.target_spiritual_guidance:
            terms.update({"tasawwuf", "sincerity", "repentance", "heart"})
        if query_intent.target_scholar_commentary or query_intent.intent_id == "transcript_lookup":
            terms.update({"transcript", "commentary", "scholar"})
        if query_intent.intent_id in {"source_only", "direct_source_lookup"}:
            terms.update({"quran", "hadith"})
    return tuple(sorted(terms))


def _metadata_candidate_indices(
    prepared_index: PreparedSearchIndex,
    query_intent: QueryIntent | None,
) -> set[int]:
    if query_intent is None:
        return set()

    indices: set[int] = set()
    for madhhab in query_intent.requested_madhhabs:
        indices.update(prepared_index.authority_chunk_ids_by_madhhab.get(madhhab, ()))
    if query_intent.target_spiritual_guidance:
        indices.update(prepared_index.source_classification_chunk_ids.get("tasawwuf_text", ()))
    if query_intent.target_scholar_commentary or query_intent.intent_id == "transcript_lookup":
        indices.update(prepared_index.source_classification_chunk_ids.get("scholar_transcript", ()))
        indices.update(prepared_index.source_classification_chunk_ids.get("transcript", ()))
    if query_intent.intent_id in {"source_only", "direct_source_lookup"}:
        indices.update(prepared_index.primary_chunk_ids)
    return indices


def _max_scored_candidate_count(*, top_k: int, match_count: int) -> int:
    if match_count <= 0:
        return 0
    target = max(1024, top_k * 12)
    return min(match_count, target)
