"""Grounded study-path assembly helpers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.citations.display_cleanup import clean_display_excerpt, clean_display_label

LAYER_ORDER = (
    ("quran", "Qur'an"),
    ("hadith", "Hadith"),
    ("fiqh", "Fiqh"),
    ("tasawwuf", "Tasawwuf"),
    ("scholar_commentary", "Scholar Commentary"),
)

LAYER_BY_SOURCE_CLASSIFICATION = {
    "quran": "quran",
    "hadith": "hadith",
    "fiqh_manual": "fiqh",
    "tasawwuf_text": "tasawwuf",
    "scholar_transcript": "scholar_commentary",
    "transcript": "scholar_commentary",
}

TOPIC_KEYWORDS = (
    ("pray", "Prayer"),
    ("prayer", "Prayer"),
    ("salat", "Prayer"),
    ("salah", "Prayer"),
    ("purification", "Purification"),
    ("wudu", "Purification"),
    ("ablution", "Purification"),
    ("fiqh", "Fiqh Study"),
    ("sincerity", "Sincerity"),
    ("ikhlas", "Sincerity"),
    ("repentance", "Repentance"),
    ("tawba", "Repentance"),
    ("intention", "Intention"),
    ("niyyah", "Intention"),
)

KEY_TERM_RULES = (
    ("pray", "prayer"),
    ("prayer", "prayer"),
    ("salat", "salat"),
    ("salah", "salah"),
    ("purification", "purification"),
    ("wudu", "wudu"),
    ("ablution", "ablution"),
    ("fiqh", "fiqh"),
    ("sincerity", "ikhlas"),
    ("ikhlas", "ikhlas"),
    ("repentance", "tawba"),
    ("tawba", "tawba"),
    ("intention", "niyyah"),
    ("niyyah", "niyyah"),
    ("heart", "heart"),
    ("tasawwuf", "tasawwuf"),
)


def build_study_path_payload(
    *,
    repo_root: Path,
    question: str,
    selected_madhhab: str,
    retrieved_sources: list[dict[str, Any]],
    overview: str,
) -> dict[str, Any]:
    source_layers = _build_source_layers(retrieved_sources)
    madhhab_readiness = _build_madhhab_readiness(
        repo_root=repo_root,
        retrieved_sources=retrieved_sources,
    )
    corpus_gaps = _build_corpus_gaps(
        selected_madhhab=selected_madhhab,
        source_layers=source_layers,
        madhhab_readiness=madhhab_readiness,
    )
    return {
        "topic": _derive_topic(question),
        "overview": overview.strip(),
        "source_layers": source_layers,
        "reading_list": _build_reading_list(source_layers),
        "lesson_path": _build_lesson_path(
            question=question,
            selected_madhhab=selected_madhhab,
            source_layers=source_layers,
            madhhab_readiness=madhhab_readiness,
        ),
        "key_terms": _build_key_terms(
            question=question,
            selected_madhhab=selected_madhhab,
            source_layers=source_layers,
        ),
        "what_to_avoid": _build_what_to_avoid(
            selected_madhhab=selected_madhhab,
            source_layers=source_layers,
            madhhab_readiness=madhhab_readiness,
        ),
        "madhhab_readiness": madhhab_readiness,
        "corpus_gaps": corpus_gaps,
    }


def study_path_diagnostics(answer: dict[str, Any]) -> dict[str, Any]:
    source_layers = answer.get("source_layers")
    if not isinstance(source_layers, dict):
        return {}
    source_layers_found: list[str] = []
    missing_layers: list[str] = []
    for layer_key, _label in LAYER_ORDER:
        items = source_layers.get(layer_key)
        if isinstance(items, list) and items:
            source_layers_found.append(layer_key)
        else:
            missing_layers.append(layer_key)
    return {
        "source_layers_found": source_layers_found,
        "missing_layers": missing_layers,
        "madhhab_readiness": answer.get("madhhab_readiness", {}),
        "corpus_gaps": answer.get("corpus_gaps", []),
    }


def layer_items(answer: dict[str, Any], layer_key: str) -> list[dict[str, Any]]:
    source_layers = answer.get("source_layers", {})
    items = source_layers.get(layer_key, [])
    return items if isinstance(items, list) else []


def layer_labels() -> list[tuple[str, str]]:
    return list(LAYER_ORDER)


def _build_source_layers(retrieved_sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key, _label in LAYER_ORDER}
    seen_by_layer: dict[str, set[tuple[str, str, str]]] = {
        key: set() for key, _label in LAYER_ORDER
    }
    for source in retrieved_sources:
        layer_key = _source_layer_key(source)
        if not layer_key:
            continue
        entry = _study_source_entry(source)
        lookup_key = (
            entry["title"].strip().lower(),
            entry["reference"].strip().lower(),
            entry["source_type"].strip().lower(),
        )
        if lookup_key in seen_by_layer[layer_key]:
            continue
        grouped[layer_key].append(entry)
        seen_by_layer[layer_key].add(lookup_key)
    return grouped


def _source_layer_key(source: dict[str, Any]) -> str | None:
    source_classification = str(
        source.get("source_classification") or source.get("source_type") or ""
    ).strip().lower()
    return LAYER_BY_SOURCE_CLASSIFICATION.get(source_classification)


def _study_source_entry(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": clean_display_label(
            str(source.get("human_title") or source.get("title") or "Retrieved Source").strip()
        ),
        "source_type": str(
            source.get("source_classification") or source.get("source_type") or "inference"
        ).strip(),
        "reference": clean_display_label(str(source.get("reference") or "Provided snippet").strip()),
        "collection": clean_display_label(str(source.get("collection") or "").strip()),
        "author": clean_display_label(str(source.get("author") or "").strip()),
        "madhhab": str(source.get("madhhab") or "").strip(),
        "quote": clean_display_excerpt(
            str(source.get("quote_window") or source.get("quote") or "").strip()
        ),
        "source_path": str(source.get("source_path") or "").strip(),
    }


def _build_reading_list(source_layers: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    beginner = _reading_items_for_layers(source_layers, ("quran", "hadith", "tasawwuf"))
    intermediate = _reading_items_for_layers(source_layers, ("fiqh", "tasawwuf"))
    advanced = _reading_items_for_layers(source_layers, ("fiqh", "scholar_commentary", "hadith"))
    return {
        "beginner": beginner,
        "intermediate": intermediate,
        "advanced": advanced,
    }


def _reading_items_for_layers(
    source_layers: dict[str, list[dict[str, Any]]],
    layer_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for layer_key in layer_keys:
        for source in source_layers.get(layer_key, [])[:2]:
            lookup_key = (
                str(source.get("title") or "").strip().lower(),
                layer_key,
            )
            if lookup_key in seen:
                continue
            items.append(
                {
                    "title": str(source.get("title") or "").strip(),
                    "layer": _layer_label(layer_key),
                    "collection": str(source.get("collection") or "").strip(),
                    "author": str(source.get("author") or "").strip(),
                }
            )
            seen.add(lookup_key)
    return items


def _build_lesson_path(
    *,
    question: str,
    selected_madhhab: str,
    source_layers: dict[str, list[dict[str, Any]]],
    madhhab_readiness: dict[str, bool],
) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    topic = _derive_topic(question)
    primary_sources = source_layers.get("quran", []) + source_layers.get("hadith", [])
    if primary_sources:
        lessons.append(
            _lesson_entry(
                title=f"Start with the primary frame for {topic}",
                objective="Read the primary texts first before leaning on explanation or later application.",
                sources=primary_sources,
                explanation="This layer gives the most direct textual frame currently grounded in the retrieved corpus.",
                practice_prompt="Read the cited passages and note the first command, description, or principle they establish.",
            )
        )
    fiqh_sources = source_layers.get("fiqh", [])
    if fiqh_sources:
        lessons.append(
            _lesson_entry(
                title="Study the fiqh structure",
                objective="See how the legal learning path is organized inside the admitted fiqh manuals.",
                sources=fiqh_sources,
                explanation=(
                    f"The fiqh layer is where madhhab-specific legal learning is anchored. "
                    f"The current strongest supported lane is {selected_madhhab}."
                    if selected_madhhab == "hanafi"
                    else "The fiqh layer is where madhhab-specific legal learning is anchored."
                ),
                practice_prompt="List the conditions, obligations, or sequence points that recur across the cited fiqh material.",
            )
        )
    tasawwuf_sources = source_layers.get("tasawwuf", [])
    if tasawwuf_sources:
        lessons.append(
            _lesson_entry(
                title="Study the inner discipline",
                objective="Use the classical spiritual layer to connect practice with sincerity, intention, and reform of the heart.",
                sources=tasawwuf_sources,
                explanation="Tasawwuf sources are presented here as spiritual guidance, not as legal-ruling authority.",
                practice_prompt="Write one short reflection on what inward quality these texts are trying to cultivate.",
            )
        )
    scholar_sources = source_layers.get("scholar_commentary", [])
    if scholar_sources:
        lessons.append(
            _lesson_entry(
                title="Use scholar commentary for teaching context",
                objective="Let grounded teaching material clarify sequence, emphasis, and study order without replacing higher-order source layers.",
                sources=scholar_sources,
                explanation="Scholar commentary is supportive explanation only and should be read under the primary and fiqh layers above it.",
                practice_prompt="Note one teaching point that clarifies how to study the topic, then tie it back to one cited source anchor.",
            )
        )
    if not lessons:
        lessons.append(
            {
                "lesson_title": f"Corpus gap for {topic}",
                "objective": "No grounded study path could be assembled from the current admitted layers.",
                "source_anchors": [],
                "short_explanation": "The current corpus does not provide enough admitted material in the requested layers to build a grounded learning path.",
                "practice_prompt": "",
            }
        )

    if (
        selected_madhhab == "hanafi"
        or selected_madhhab in {"", "not_specified", "compare_all"}
        or madhhab_readiness.get(_readiness_key(selected_madhhab), False)
    ):
        return lessons

    lessons.append(
        {
            "lesson_title": "Madhhab coverage limit",
            "objective": "Avoid treating this study path as a full school-specific map.",
            "source_anchors": [],
            "short_explanation": "This madhhab is not yet sufficiently represented in the current corpus.",
            "practice_prompt": "",
        }
    )
    return lessons


def _lesson_entry(
    *,
    title: str,
    objective: str,
    sources: list[dict[str, Any]],
    explanation: str,
    practice_prompt: str,
) -> dict[str, Any]:
    anchors = []
    for source in sources[:3]:
        title_text = str(source.get("title") or "").strip()
        reference = str(source.get("reference") or "").strip()
        if title_text and reference:
            anchors.append(f"{title_text} - {reference}")
        elif title_text:
            anchors.append(title_text)
    return {
        "lesson_title": title,
        "objective": objective,
        "source_anchors": anchors,
        "short_explanation": explanation,
        "practice_prompt": practice_prompt,
    }


def _build_key_terms(
    *,
    question: str,
    selected_madhhab: str,
    source_layers: dict[str, list[dict[str, Any]]],
) -> list[str]:
    normalized = _normalize_text(question)
    terms: list[str] = []
    for needle, term in KEY_TERM_RULES:
        if needle in normalized and term not in terms:
            terms.append(term)
    if selected_madhhab not in {"", "not_specified", "compare_all"}:
        terms.append(selected_madhhab)
    if source_layers.get("tasawwuf") and "spiritual discipline" not in terms:
        terms.append("spiritual discipline")
    return terms[:8]


def _build_what_to_avoid(
    *,
    selected_madhhab: str,
    source_layers: dict[str, list[dict[str, Any]]],
    madhhab_readiness: dict[str, bool],
) -> list[str]:
    notes = [
        "Do not treat later commentary as a substitute for Qur'an, hadith, or fiqh manuals.",
    ]
    if source_layers.get("tasawwuf"):
        notes.append("Do not treat spiritual guidance texts as legal-ruling authority.")
    if source_layers.get("scholar_commentary"):
        notes.append("Do not treat scholar commentary as an override on primary texts or fiqh manuals.")
    if selected_madhhab not in {"", "not_specified", "compare_all"} and not madhhab_readiness.get(
        _readiness_key(selected_madhhab), False
    ):
        notes.append("Do not assume the selected madhhab is fully represented in the current corpus.")
    if selected_madhhab not in {"", "not_specified", "compare_all"} and not source_layers.get("fiqh"):
        notes.append("Do not infer a madhhab-specific legal path where no grounded fiqh source was retrieved.")
    return notes


def _build_corpus_gaps(
    *,
    selected_madhhab: str,
    source_layers: dict[str, list[dict[str, Any]]],
    madhhab_readiness: dict[str, bool],
) -> list[str]:
    gaps: list[str] = []
    if selected_madhhab not in {"", "not_specified", "compare_all"} and not madhhab_readiness.get(
        _readiness_key(selected_madhhab), False
    ):
        gaps.append("This madhhab is not yet sufficiently represented in the current corpus.")
    if not source_layers.get("quran"):
        gaps.append("No grounded Qur'an source was found for this layer.")
    if not source_layers.get("hadith"):
        gaps.append("No grounded Hadith source was found for this layer.")
    if selected_madhhab in {"hanafi", "shafii", "maliki", "hanbali"} and not source_layers.get("fiqh"):
        gaps.append("No grounded fiqh manual source was found for the requested study path.")
    if not source_layers.get("tasawwuf"):
        gaps.append("No grounded tasawwuf source was found for this layer.")
    if not source_layers.get("scholar_commentary"):
        gaps.append("No grounded scholar commentary was found for this layer.")
    return gaps


def _build_madhhab_readiness(
    *,
    repo_root: Path,
    retrieved_sources: list[dict[str, Any]],
) -> dict[str, bool]:
    non_hanafi_ready_min_documents = 3
    baseline_counts = _baseline_madhhab_counts(repo_root)
    local_counts = {
        "hanafi": 0,
        "shafii": 0,
        "maliki": 0,
        "hanbali": 0,
    }
    for source in retrieved_sources:
        madhhab = str(source.get("madhhab") or "").strip().lower()
        if madhhab in local_counts:
            local_counts[madhhab] += 1
    return {
        "hanafi_ready": baseline_counts.get("hanafi", 0) > 0 or local_counts["hanafi"] > 0,
        "shafi_ready": (
            max(
                baseline_counts.get("shafii", 0),
                baseline_counts.get("shafi", 0),
                local_counts["shafii"],
            )
            >= non_hanafi_ready_min_documents
        ),
        "maliki_ready": max(
            baseline_counts.get("maliki", 0),
            local_counts["maliki"],
        )
        >= non_hanafi_ready_min_documents,
        "hanbali_ready": max(
            baseline_counts.get("hanbali", 0),
            local_counts["hanbali"],
        )
        >= non_hanafi_ready_min_documents,
    }


def _derive_topic(question: str) -> str:
    normalized = _normalize_text(question)
    for needle, label in TOPIC_KEYWORDS:
        if needle in normalized:
            return label
    cleaned = re.sub(r"\s+", " ", question or "").strip(" ?.")
    return cleaned or "Islamic Study Path"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _layer_label(layer_key: str) -> str:
    for key, label in LAYER_ORDER:
        if key == layer_key:
            return label
    return layer_key.replace("_", " ").title()


def _readiness_key(selected_madhhab: str) -> str:
    if selected_madhhab == "shafii":
        return "shafi_ready"
    return f"{selected_madhhab}_ready"


@lru_cache(maxsize=4)
def _baseline_madhhab_counts(repo_root: Path) -> dict[str, int]:
    baseline_path = repo_root / "docs" / "readiness" / "corpus-baseline.json"
    if not baseline_path.exists():
        return {}
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    compare_readiness = (
        payload.get("corpus", {}).get("compare_views_readiness", {})
        if isinstance(payload, dict)
        else {}
    )
    counts = compare_readiness.get("labeled_madhhab_counts", {})
    return counts if isinstance(counts, dict) else {}
