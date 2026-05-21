"""Query intent routing for retrieval and grounded answer assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.reasoning.scholar_resolver import resolve_scholar_reference

LEGAL_TERMS = {
    "allowed",
    "break",
    "disliked",
    "forbidden",
    "halal",
    "haram",
    "makruh",
    "obligatory",
    "permissible",
    "prohibited",
    "ruling",
    "valid",
    "wajib",
}
COMPARE_TERMS = {"compare", "difference", "versus", "vs", "views"}
SOURCE_ONLY_TERMS = {
    "source only",
    "source-only",
    "source-only material",
    "sources only",
    "only sources",
    "just the sources",
    "show sources",
    "give sources",
    "primary source text",
    "primary source texts",
    "primary texts",
    "only primary texts",
    "only primary source",
    "only primary source text",
    "no synthesis",
    "strip out commentary",
}
FATWA_TERMS = {"fatwa", "ifta", "mufti", "dar al ifta"}
TRANSCRIPT_TERMS = {"transcript", "lecture", "talk", "khutbah", "shaykh said"}
SCHOLAR_COMMENTARY_TERMS = {
    "scholar commentary",
    "teacher commentary",
    "scholar transcript",
    "teacher transcript",
    "using scholar commentary",
    "with scholar commentary",
}
TEACHING_LAYER_TERMS = {
    "teach me",
    "walk me through",
    "walk through",
    "why is this",
    "why does",
    "why do",
    "why did",
    "explain the reasoning",
    "teaching layer",
    "class material",
    "study note",
}
SUMMARY_TERMS = {"summarize", "summary", "overview", "outline"}
DEFINITION_TERMS = {"define", "definition", "explain term", "meaning of"}
DIRECT_SOURCE_TERMS = {
    "ayah",
    "ayat",
    "chapter",
    "hadith",
    "quran",
    "qur'an",
    "source",
    "surah",
    "verse",
}
EVIDENCE_LAYER_TERMS = {
    "primary text",
    "primary texts",
    "quran",
    "qur'an",
    "hadith",
    "fiqh manual",
    "fiqh manuals",
    "commentary",
    "fatwa",
    "modern fatwa",
    "transcript",
}
SPIRITUAL_TERMS = {
    "ikhlas",
    "sincerity",
    "niyyah",
    "intention",
    "repentance",
    "tawba",
    "tazkiyah",
    "heart",
    "nafs",
    "dhikr",
    "remembrance",
    "humility",
    "patience",
    "discipline",
    "spiritual",
}
PRAYER_TERMS = {
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
    "ruku",
    "tashahhud",
    "witr",
    "fajr",
    "dhuhr",
    "asr",
    "maghrib",
    "isha",
}
PURIFICATION_TERMS = {
    "wudu",
    "wudhu",
    "ablution",
    "purification",
    "taharah",
    "ghusl",
    "tayammum",
    "tayamum",
    "impurity",
    "najasa",
    "cleanliness",
}
PRAYER_METHOD_PATTERNS = {
    "how should i pray",
    "how do i pray",
    "how to pray",
    "teach me how to pray",
    "how should i perform salah",
    "how should i perform salat",
    "how is prayer performed",
}
PURIFICATION_METHOD_PATTERNS = {
    "how do i make wudu",
    "how should i make wudu",
    "how to make wudu",
    "how do i perform wudu",
    "how should i perform wudu",
    "how do i make ablution",
    "how should i make ablution",
    "how to make ablution",
}
TASAWWUF_REFERENCE_TERMS = {
    "ghazali",
    "ihya",
    "book of assistance",
    "haddad",
    "qushayri",
    "qushayri risala",
    "al-hikam",
    "hikam",
    "ibn ata",
    "ataillah",
    "futuh al-ghayb",
    "futuh al ghayb",
    "futuh al ghaib",
    "jilani",
    "abd al-qadir",
    "abd al qadir",
    "tasawwuf",
    "sufism",
    "sufi",
}

INTENT_AUTHORITY_ORDER = {
    "direct_source_lookup": ("quran", "hadith", "tasawwuf_text", "commentary", "fiqh_manual", "fatwa", "scholar_transcript", "transcript"),
    "ruling_lookup": ("quran", "hadith", "fiqh_manual", "commentary", "fatwa", "tasawwuf_text", "scholar_transcript", "transcript"),
    "source_only": ("quran", "hadith", "tasawwuf_text", "fiqh_manual", "commentary", "fatwa", "scholar_transcript", "transcript"),
    "compare_views": ("fiqh_manual", "quran", "hadith", "commentary", "fatwa", "tasawwuf_text", "scholar_transcript", "transcript"),
    "scholar_perspective": ("fiqh_manual", "commentary", "scholar_transcript", "quran", "hadith", "tasawwuf_text", "fatwa", "transcript"),
    "explain_term": ("quran", "hadith", "tasawwuf_text", "commentary", "fiqh_manual", "fatwa", "scholar_transcript", "transcript"),
    "summarize_source": ("quran", "hadith", "tasawwuf_text", "fiqh_manual", "commentary", "fatwa", "scholar_transcript", "transcript"),
    "fatwa_lookup": ("fatwa", "fiqh_manual", "commentary", "quran", "hadith", "tasawwuf_text", "scholar_transcript", "transcript"),
    "transcript_lookup": ("scholar_transcript", "transcript", "commentary", "fiqh_manual", "hadith", "quran", "tasawwuf_text", "fatwa"),
}
SUPPORTED_MADHHABS = ("hanafi", "shafii", "maliki", "hanbali")
MADHHAB_PATTERN_MAP = {
    "hanafi": re.compile(r"\bhanafi\b"),
    "shafii": re.compile(r"\bshaf(?:i['’]i|ii|i)\b"),
    "maliki": re.compile(r"\bmaliki\b"),
    "hanbali": re.compile(r"\bhanbali\b"),
}
ALL_FOUR_MADHHAB_PATTERNS = (
    "all four madhhabs",
    "all four schools",
    "four madhhabs",
    "four schools",
)


@dataclass(slots=True)
class QueryIntent:
    intent_id: str
    preserve_disagreement: bool
    prefer_direct_excerpts: bool
    prefer_selected_madhhab: bool
    prefer_primary_texts: bool
    prefer_definitional_material: bool
    suppress_synthesis: bool
    target_spiritual_guidance: bool
    target_scholar_commentary: bool
    target_teaching_layer: bool
    worship_topic: str
    authority_order: tuple[str, ...]
    scholar_id: str
    scholar_name: str
    scholar_madhhab: str
    scholar_period: str
    scholar_methodology_notes: str
    scholar_known_works: tuple[str, ...]
    scholar_source_families: tuple[str, ...]
    scholar_retrieval_tags: tuple[str, ...]
    unknown_scholar: str
    detected_madhhab_intent: str
    requested_madhhabs: tuple[str, ...]


def route_query_intent(
    *,
    question: str,
    answer_mode: str,
    selected_madhhab: str,
) -> QueryIntent:
    normalized = _normalize(question)
    scholar_resolution = resolve_scholar_reference(question)
    intent_id = _infer_intent_id(
        normalized_question=normalized,
        answer_mode=answer_mode,
        selected_madhhab=selected_madhhab,
        scholar_recognized=scholar_resolution.recognized,
    )
    requested_madhhabs = _detect_requested_madhhabs(
        normalized_question=normalized,
        selected_madhhab=selected_madhhab,
        intent_id=intent_id,
    )
    target_spiritual_guidance = _targets_spiritual_guidance(
        normalized_question=normalized,
        intent_id=intent_id,
    )
    target_scholar_commentary = _targets_scholar_commentary(
        normalized_question=normalized,
        intent_id=intent_id,
    )
    target_teaching_layer = _targets_teaching_layer(
        normalized_question=normalized,
        intent_id=intent_id,
    )
    worship_topic = _detect_worship_topic(normalized_question=normalized)
    return QueryIntent(
        intent_id=intent_id,
        preserve_disagreement=intent_id == "compare_views",
        prefer_direct_excerpts=intent_id in {"direct_source_lookup", "source_only", "summarize_source"},
        prefer_selected_madhhab=bool(requested_madhhabs),
        prefer_primary_texts=intent_id in {
            "direct_source_lookup",
            "source_only",
            "explain_term",
            "summarize_source",
        },
        prefer_definitional_material=intent_id == "explain_term",
        suppress_synthesis=intent_id == "source_only",
        target_spiritual_guidance=target_spiritual_guidance,
        target_scholar_commentary=target_scholar_commentary,
        target_teaching_layer=target_teaching_layer,
        worship_topic=worship_topic,
        authority_order=INTENT_AUTHORITY_ORDER[intent_id],
        scholar_id=scholar_resolution.profile.scholar_id if scholar_resolution.profile else "",
        scholar_name=scholar_resolution.profile.name if scholar_resolution.profile else "",
        scholar_madhhab=scholar_resolution.profile.madhhab if scholar_resolution.profile else "",
        scholar_period=scholar_resolution.profile.period if scholar_resolution.profile else "",
        scholar_methodology_notes=(
            scholar_resolution.profile.methodology_notes if scholar_resolution.profile else ""
        ),
        scholar_known_works=(
            scholar_resolution.profile.known_works if scholar_resolution.profile else ()
        ),
        scholar_source_families=(
            scholar_resolution.profile.source_families if scholar_resolution.profile else ()
        ),
        scholar_retrieval_tags=(
            scholar_resolution.profile.retrieval_tags if scholar_resolution.profile else ()
        ),
        unknown_scholar=scholar_resolution.unknown_scholar,
        detected_madhhab_intent=",".join(requested_madhhabs),
        requested_madhhabs=requested_madhhabs,
    )


def _infer_intent_id(
    *,
    normalized_question: str,
    answer_mode: str,
    selected_madhhab: str,
    scholar_recognized: bool,
) -> str:
    if (
        answer_mode == "source_only"
        or _contains_phrase(normalized_question, SOURCE_ONLY_TERMS)
        or _looks_like_source_only_request(normalized_question)
    ):
        return "source_only"
    if answer_mode == "compare_views" or _contains_word(normalized_question, COMPARE_TERMS):
        return "compare_views"
    if answer_mode == "scholar_perspective" and scholar_recognized:
        return "scholar_perspective"
    if scholar_recognized:
        return "scholar_perspective"
    if _looks_like_multi_layer_source_request(normalized_question):
        return "direct_source_lookup"
    if _looks_like_definition_query(normalized_question):
        return "explain_term"
    if _contains_phrase(normalized_question, FATWA_TERMS):
        return "fatwa_lookup"
    if _contains_phrase(normalized_question, TRANSCRIPT_TERMS):
        return "transcript_lookup"
    if _contains_phrase(normalized_question, SUMMARY_TERMS):
        return "summarize_source"
    if answer_mode == "study_path":
        return "direct_source_lookup"
    if _looks_like_prayer_method_request(normalized_question):
        return "direct_source_lookup"
    if _looks_like_purification_method_request(normalized_question):
        return "direct_source_lookup"
    if _contains_word(normalized_question, LEGAL_TERMS):
        return "ruling_lookup"
    if selected_madhhab not in {"", "not_specified", "compare_all"} and _mentions_juristic_focus(
        normalized_question
    ):
        return "ruling_lookup"
    if _looks_like_direct_source_lookup(normalized_question):
        return "direct_source_lookup"
    if selected_madhhab not in {"", "not_specified", "compare_all"}:
        return "ruling_lookup"
    return "direct_source_lookup"


def _looks_like_direct_source_lookup(normalized_question: str) -> bool:
    if _contains_word(normalized_question, DIRECT_SOURCE_TERMS):
        return True
    if re.search(r"\b\d{1,3}:\d{1,3}\b", normalized_question):
        return True
    return False


def _looks_like_source_only_request(normalized_question: str) -> bool:
    if "source-only" in normalized_question:
        return True
    if re.search(
        r"\b(show|give)\s+only\b.+\b(quran|qur'an|hadith|sources?|primary)\b",
        normalized_question,
    ):
        return True
    if re.search(r"\bonly\b.+\b(quran|qur'an|hadith|sources?|primary texts?)\b", normalized_question):
        return True
    return False


def _mentions_juristic_focus(normalized_question: str) -> bool:
    return bool(
        re.search(
            r"\b(view|position|requirements|conditions|wiping|ablution|purification|wudu|wudhu)\b",
            normalized_question,
        )
    )


def _looks_like_definition_query(normalized_question: str) -> bool:
    if _contains_phrase(normalized_question, DEFINITION_TERMS):
        return True
    return bool(re.search(r"\bwhat does\b.+\bmean\b", normalized_question))


def _looks_like_multi_layer_source_request(normalized_question: str) -> bool:
    matched_layers = {
        term
        for term in EVIDENCE_LAYER_TERMS
        if term in normalized_question
    }
    if len(matched_layers) < 2:
        return False
    return "across" in normalized_question or "show how" in normalized_question


def _targets_spiritual_guidance(
    *,
    normalized_question: str,
    intent_id: str,
) -> bool:
    if _contains_phrase(normalized_question, TASAWWUF_REFERENCE_TERMS):
        return True
    if intent_id in {"ruling_lookup", "compare_views", "fatwa_lookup"}:
        return False
    if "purification of the heart" in normalized_question:
        return True
    if _contains_word(normalized_question, SPIRITUAL_TERMS):
        if _contains_word(normalized_question, LEGAL_TERMS):
            return False
        if _mentions_juristic_focus(normalized_question):
            return False
        return True
    return False


def _targets_scholar_commentary(
    *,
    normalized_question: str,
    intent_id: str,
) -> bool:
    if intent_id == "scholar_perspective":
        return True
    if intent_id in {"ruling_lookup", "compare_views", "fatwa_lookup"}:
        return False
    if _contains_phrase(normalized_question, SCHOLAR_COMMENTARY_TERMS):
        return True
    return "scholar" in normalized_question and "commentary" in normalized_question


def _targets_teaching_layer(
    *,
    normalized_question: str,
    intent_id: str,
) -> bool:
    if intent_id in {"explain_term", "summarize_source"}:
        return True
    if "why" in normalized_question:
        return True
    return _contains_phrase(normalized_question, TEACHING_LAYER_TERMS)


def _detect_worship_topic(*, normalized_question: str) -> str:
    prayer_focused = _contains_phrase(normalized_question, PRAYER_TERMS)
    purification_focused = _contains_phrase(normalized_question, PURIFICATION_TERMS)
    if prayer_focused and purification_focused:
        return "prayer_with_purification_prerequisite"
    if prayer_focused and _looks_like_prayer_method_request(normalized_question):
        return "prayer_method"
    if purification_focused and _looks_like_purification_method_request(normalized_question):
        return "purification_method"
    if purification_focused:
        return "purification"
    if prayer_focused:
        return "prayer"
    return "general"


def _looks_like_prayer_method_request(normalized_question: str) -> bool:
    return any(pattern in normalized_question for pattern in PRAYER_METHOD_PATTERNS)


def _looks_like_purification_method_request(normalized_question: str) -> bool:
    return any(pattern in normalized_question for pattern in PURIFICATION_METHOD_PATTERNS)


def _detect_requested_madhhabs(
    *,
    normalized_question: str,
    selected_madhhab: str,
    intent_id: str,
) -> tuple[str, ...]:
    selected = str(selected_madhhab or "").strip().lower()
    if (
        selected == "compare_all"
        or any(pattern in normalized_question for pattern in ALL_FOUR_MADHHAB_PATTERNS)
    ):
        return SUPPORTED_MADHHABS

    detected: list[str] = []
    for madhhab in SUPPORTED_MADHHABS:
        if MADHHAB_PATTERN_MAP[madhhab].search(normalized_question):
            detected.append(madhhab)

    if detected:
        return tuple(dict.fromkeys(detected))

    if intent_id == "compare_views":
        return SUPPORTED_MADHHABS

    if selected in SUPPORTED_MADHHABS:
        return (selected,)
    return ()


def _contains_word(text: str, values: set[str]) -> bool:
    tokens = set(re.findall(r"\w+", text))
    return any(value in tokens for value in values)


def _contains_phrase(text: str, values: set[str]) -> bool:
    return any(value in text for value in values)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
