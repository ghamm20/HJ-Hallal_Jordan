"""Normalize and validate answer payloads against the canonical schema."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.reasoning.study_path import build_study_path_payload


def normalize_and_validate_answer(
    *,
    candidate: dict[str, Any],
    schema: dict[str, Any],
    request_context: dict[str, str],
    retrieved_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    properties = schema["properties"]
    normalized: dict[str, Any] = {}

    normalized["schema_version"] = _normalize_schema_version(
        candidate.get("schema_version")
    )
    requested_answer_mode = str(request_context.get("answer_mode") or "").strip()
    if requested_answer_mode == "study_path":
        normalized["answer_mode"] = "study_path"
    else:
        normalized["answer_mode"] = _coerce_enum(
            candidate.get("answer_mode"),
            properties["answer_mode"]["enum"],
            requested_answer_mode,
        )
    normalized["selected_madhhab"] = _coerce_enum(
        request_context["selected_madhhab"],
        properties["selected_madhhab"]["enum"],
        request_context["selected_madhhab"],
    )
    normalized["greeting"] = _normalize_greeting(
        candidate.get("greeting"),
        request_context["greeting_style"],
    )
    normalized["direct_answer"] = _normalize_string(
        candidate.get("direct_answer"),
        _default_direct_answer(normalized["answer_mode"]),
    )
    normalized["madhhab_position"] = _normalize_string(
        candidate.get("madhhab_position"),
        _default_madhhab_position(normalized["selected_madhhab"]),
    )
    normalized["evidence_summary"] = _normalize_string(
        candidate.get("evidence_summary"),
        _default_evidence_summary(retrieved_sources),
    )
    normalized["disagreement_note"] = _normalize_optional_string(
        candidate.get("disagreement_note"),
        _default_disagreement(normalized["answer_mode"], retrieved_sources),
    )
    normalized["uncertainty_note"] = _normalize_optional_string(
        candidate.get("uncertainty_note"),
        _default_uncertainty(retrieved_sources),
    )
    normalized["evidence_strength"] = _coerce_optional_enum(
        candidate.get("evidence_strength"),
        properties["evidence_strength"]["enum"],
        _default_evidence_strength(retrieved_sources),
    )
    normalized["citations"] = _normalize_citations(
        candidate.get("citations"),
        properties["citations"]["items"],
        retrieved_sources,
    )
    normalized["source_breakdown"] = _normalize_source_breakdown(
        candidate.get("source_breakdown"),
        retrieved_sources,
    )
    normalized["ui_state"] = _normalize_ui_state(
        candidate.get("ui_state"),
        properties["ui_state"],
        request_context,
    )
    if normalized["answer_mode"] == "study_path":
        study_path_payload = build_study_path_payload(
            repo_root=Path(str(request_context["repo_root"])),
            question=str(request_context.get("question") or ""),
            selected_madhhab=normalized["selected_madhhab"],
            retrieved_sources=retrieved_sources,
            overview=_normalize_string(
                candidate.get("overview"),
                normalized["direct_answer"],
            ),
        )
        normalized["topic"] = _normalize_string(
            candidate.get("topic"),
            str(study_path_payload["topic"]),
        )
        normalized["overview"] = _normalize_string(
            candidate.get("overview"),
            str(study_path_payload["overview"]),
        )
        normalized["source_layers"] = study_path_payload["source_layers"]
        normalized["reading_list"] = study_path_payload["reading_list"]
        normalized["lesson_path"] = study_path_payload["lesson_path"]
        normalized["key_terms"] = study_path_payload["key_terms"]
        normalized["what_to_avoid"] = study_path_payload["what_to_avoid"]
        normalized["madhhab_readiness"] = study_path_payload["madhhab_readiness"]
        normalized["corpus_gaps"] = study_path_payload["corpus_gaps"]
        readiness_key = _study_path_readiness_key(normalized["selected_madhhab"])
        if readiness_key and not normalized["madhhab_readiness"].get(readiness_key, False):
            normalized["madhhab_position"] = (
                "This madhhab is not yet sufficiently represented in the current corpus."
            )
            normalized["uncertainty_note"] = _merge_notes(
                normalized.get("uncertainty_note"),
                "This madhhab is not yet sufficiently represented in the current corpus.",
            )

    filtered = {
        key: value
        for key, value in normalized.items()
        if value not in (None, "", [], {})
        or key in schema["required"]
    }
    _validate_object(filtered, schema)
    return filtered


def _normalize_greeting(value: Any, greeting_style: str) -> str:
    if greeting_style == "none":
        return ""
    if value:
        return str(value).strip()
    if greeting_style == "full_islamic":
        return "Assalamu 'alaykum wa rahmatullahi wa barakatuh."
    return "Assalamu 'alaykum."


def _normalize_schema_version(value: Any) -> str:
    candidate = str(value).strip() if value is not None else ""
    parts = candidate.split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return candidate
    return "1.0.0"


def _default_direct_answer(answer_mode: str) -> str:
    if answer_mode == "source_only":
        return "The retrieved sources are presented below with minimal framing."
    return "The provided sources suggest a grounded answer, but the citations below should remain primary."


def _default_madhhab_position(selected_madhhab: str) -> str:
    if selected_madhhab == "not_specified":
        return "No madhhab was selected."
    if selected_madhhab == "compare_all":
        return "The answer compares the available views across the provided sources."
    return f"The answer prioritizes the {selected_madhhab} view where the provided sources support it."


def _default_evidence_summary(retrieved_sources: list[dict[str, Any]]) -> str:
    return (
        f"This answer is grounded in {len(retrieved_sources)} retrieved source "
        "snippet(s) supplied to the pipeline."
    )


def _default_disagreement(
    answer_mode: str,
    retrieved_sources: list[dict[str, Any]],
) -> str | None:
    madhhabs = {
        str(source.get("madhhab", "")).strip().lower()
        for source in retrieved_sources
        if source.get("madhhab")
    }
    if answer_mode == "compare_views" and len(madhhabs) > 1:
        return "The retrieved sources include more than one madhhab perspective."
    if len(madhhabs) > 1:
        return (
            "The retrieved sources span more than one madhhab perspective and should "
            "not be flattened into a single claim without qualification."
        )
    return None


def _default_uncertainty(retrieved_sources: list[dict[str, Any]]) -> str | None:
    source_classes = {
        str(
            source.get("source_classification")
            or source.get("source_type")
            or "unknown"
        ).strip().lower()
        for source in retrieved_sources
    }
    notes: list[str] = []
    if len(retrieved_sources) <= 2:
        notes.append(
            "This answer is limited to the provided snippets and should not be read as an independent fatwa."
        )
    if not {"quran", "hadith"} & source_classes and retrieved_sources:
        notes.append(
            "No primary text was retrieved in this set."
        )
    if {"fatwa", "transcript", "scholar_transcript"} & source_classes:
        notes.append(
            "Modern application materials are present and should be read alongside the higher-order source layers."
        )
    if "scholar_transcript" in source_classes:
        notes.append(
            "Scholar transcript materials in this set are commentary and teaching context, not legal-ruling authority."
        )
    if "tasawwuf_text" in source_classes:
        notes.append(
            "Tasawwuf materials in this set reflect classical spiritual guidance, not legal rulings."
        )
    if notes:
        return " ".join(notes)
    return None


def _default_evidence_strength(retrieved_sources: list[dict[str, Any]]) -> str:
    if len(retrieved_sources) >= 4:
        return "moderate"
    if len(retrieved_sources) >= 2:
        return "limited"
    return "insufficient"


def _normalize_string(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _normalize_optional_string(value: Any, default: str | None = None) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or default


def _coerce_enum(value: Any, allowed: list[str], default: str) -> str:
    candidate = str(value).strip() if value is not None else ""
    return candidate if candidate in allowed else default


def _coerce_optional_enum(
    value: Any,
    allowed: list[str],
    default: str | None,
) -> str | None:
    if value is None:
        return default
    candidate = str(value).strip()
    return candidate if candidate in allowed else default


def _normalize_citations(
    value: Any,
    citation_schema: dict[str, Any],
    retrieved_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allowed_keys = set(citation_schema["properties"].keys())
    allowed_source_types = set(citation_schema["properties"]["source_type"]["enum"])
    source_index = {
        _citation_lookup_key(source): source for source in retrieved_sources
    }
    citations: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            lookup_key = _citation_lookup_key(item)
            source = source_index.get(lookup_key)
            if source is None:
                continue
            citation = {
                key: _citation_value(
                    source,
                    key,
                    allowed_source_types=allowed_source_types,
                )
                for key in allowed_keys
                if _citation_value(
                    source,
                    key,
                    allowed_source_types=allowed_source_types,
                )
                not in (None, "")
            }
            if (
                _citation_has_required_fields(citation)
                and citation["source_type"] in allowed_source_types
            ):
                citations.append(citation)
    if citations:
        return citations

    fallback: list[dict[str, Any]] = []
    for source in retrieved_sources:
        citation = {
            "title": str(
                _citation_value(
                    source,
                    "title",
                    allowed_source_types=allowed_source_types,
                )
                or "Retrieved Source"
            ).strip(),
            "source_type": str(
                _citation_value(
                    source,
                    "source_type",
                    allowed_source_types=allowed_source_types,
                )
                or "inference"
            ).strip(),
            "reference": str(source.get("reference", "Provided snippet")).strip(),
        }
        for optional_key in ("collection", "author", "madhhab", "quote", "source_path"):
            optional_value = _citation_value(
                source,
                optional_key,
                allowed_source_types=allowed_source_types,
            )
            if optional_value not in (None, ""):
                citation[optional_key] = str(optional_value)
        if (
            _citation_has_required_fields(citation)
            and citation["source_type"] in allowed_source_types
        ):
            fallback.append(citation)
    return fallback


def _citation_has_required_fields(citation: dict[str, Any]) -> bool:
    return all(citation.get(key) for key in ("title", "source_type", "reference"))


def _citation_lookup_key(source: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(source.get("title", "")).strip().lower(),
        str(source.get("reference", "")).strip().lower(),
        str(source.get("source_type", "")).strip().lower(),
    )


def _normalize_source_breakdown(
    value: Any,
    retrieved_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        breakdown: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            source_type = item.get("source_type")
            count = item.get("count")
            if isinstance(source_type, str) and isinstance(count, int):
                breakdown.append({"source_type": source_type, "count": count})
        if breakdown:
            return breakdown

    counts = Counter(
        str(
            source.get("source_classification")
            or source.get("source_type")
            or "unknown"
        )
        for source in retrieved_sources
    )
    return [
        {"source_type": source_type, "count": count}
        for source_type, count in sorted(counts.items())
    ]


def _citation_value(
    source: dict[str, Any],
    key: str,
    *,
    allowed_source_types: set[str],
) -> Any:
    if key == "title":
        return source.get("human_title") or source.get("title")
    if key == "quote":
        return source.get("quote_window") or source.get("quote")
    if key == "source_type":
        candidate = str(source.get("source_type", "") or "").strip()
        if candidate in allowed_source_types:
            return candidate
        classification = str(source.get("source_classification", "") or "").strip()
        if classification in allowed_source_types:
            return classification
        return "inference"
    return source.get(key)


def _normalize_ui_state(
    value: Any,
    ui_state_schema: dict[str, Any],
    request_context: dict[str, str],
) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    normalized = {
        "greeting_style": _coerce_enum(
            value.get("greeting_style"),
            ui_state_schema["properties"]["greeting_style"]["enum"],
            request_context["greeting_style"],
        ),
        "tone_level": _coerce_enum(
            value.get("tone_level"),
            ui_state_schema["properties"]["tone_level"]["enum"],
            request_context["tone_level"],
        ),
        "selected_madhhab_visible": bool(
            value.get("selected_madhhab_visible", True)
        ),
        "answer_mode_visible": bool(value.get("answer_mode_visible", True)),
        "confidence_visible": bool(value.get("confidence_visible", True)),
        "disagreement_visible": bool(value.get("disagreement_visible", True)),
    }
    return normalized


def _merge_notes(existing: str | None, note: str) -> str:
    existing_text = str(existing or "").strip()
    if not existing_text:
        return note
    if note in existing_text:
        return existing_text
    return f"{existing_text} {note}"


def _study_path_readiness_key(selected_madhhab: str) -> str | None:
    if selected_madhhab in {"", "not_specified", "compare_all"}:
        return None
    if selected_madhhab == "shafii":
        return "shafi_ready"
    return f"{selected_madhhab}_ready"


def _validate_object(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if schema.get("additionalProperties") is False:
            unexpected = set(value) - set(properties)
            if unexpected:
                raise ValueError(
                    f"{path} has unexpected properties: {sorted(unexpected)}"
                )
        for name in required:
            if name not in value:
                raise ValueError(f"{path}.{name} is required")
        for name, item in value.items():
            if name in properties:
                _validate_object(item, properties[name], f"{path}.{name}")
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise TypeError(f"{path} must be an array")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_object(item, item_schema, f"{path}[{index}]")
        return

    if schema_type == "string":
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            raise ValueError(f"{path} must be one of {enum}")
        return

    if schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{path} must be an integer")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{path} must be >= {minimum}")
        return

    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a boolean")
        return
