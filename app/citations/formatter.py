"""Helpers for rendering cleaner, human-readable citation output."""

from __future__ import annotations

import json
from typing import Any, Mapping

from app.citations.display_cleanup import clean_display_excerpt, clean_display_label
from app.retrieval.metadata_normalizer import format_madhhab_label


def format_evidence_entry(
    source: Any,
    *,
    include_quote: bool = True,
) -> list[str]:
    title = str(
        getattr(source, "human_title", "")
        or getattr(source, "title", "")
        or "Retrieved Source"
    )
    title = clean_display_label(title)
    source_type_label = _source_frame_label(source)
    madhhab = format_madhhab_label(getattr(source, "madhhab", "") or "")
    reference = str(getattr(source, "reference", "") or "")
    hierarchy_label = clean_display_label(str(getattr(source, "hierarchy_label", "") or ""))
    section_label = clean_display_label(str(getattr(source, "section_label", "") or ""))
    fatwa_authority = str(getattr(source, "fatwa_authority", "") or "")
    quote = clean_display_excerpt(str(getattr(source, "quote", "") or ""))
    extraction_note = _extraction_note(source)

    header = f"- {title} [{source_type_label}]"
    if reference:
        header += f" - {reference}"
    if madhhab:
        header += f" ({madhhab})"

    lines = [header]
    if hierarchy_label:
        lines.append(f"  Source: {hierarchy_label}")
    elif section_label:
        lines.append(f"  Section: {section_label}")
    if fatwa_authority:
        lines.append(f"  Authority: {fatwa_authority}")
    if extraction_note:
        lines.append(f"  Extraction: {extraction_note}")
    if include_quote and quote:
        lines.append(f'  Quote: "{quote}"')
    return lines


def format_source_line(source: Any) -> str:
    title = str(
        getattr(source, "human_title", "")
        or getattr(source, "title", "")
        or "Retrieved Source"
    )
    title = clean_display_label(title)
    source_type_label = _source_frame_label(source)
    madhhab = format_madhhab_label(getattr(source, "madhhab", "") or "")
    reference = str(getattr(source, "reference", "") or "")
    hierarchy_label = clean_display_label(str(getattr(source, "hierarchy_label", "") or ""))
    section_label = clean_display_label(str(getattr(source, "section_label", "") or ""))
    extraction_note = _extraction_note(source)

    header = f"- {title} [{source_type_label}]"
    if madhhab:
        header += f" ({madhhab})"
    parts = [header]
    if hierarchy_label:
        parts.append(hierarchy_label)
    elif section_label:
        parts.append(section_label)
    if reference:
        parts.append(reference)
    if extraction_note:
        parts.append(extraction_note)
    return " - ".join(parts)


def _source_frame_label(source: Any) -> str:
    source_role_boundary = str(getattr(source, "source_role_boundary", "") or "").strip().lower()
    source_family = str(getattr(source, "source_family", "") or "").strip().lower()
    source_classification = str(getattr(source, "source_classification", "") or "").strip().lower()
    document_kind = str(getattr(source, "document_kind", "") or "").strip().lower()
    title = str(getattr(source, "human_title", "") or getattr(source, "title", "") or "").strip().lower()
    collection = str(getattr(source, "collection", "") or "").strip().lower()
    fallback = str(getattr(source, "source_type_label", "") or "Unknown")
    if source_role_boundary == "teaching_layer" or source_family == "classes":
        return "Teaching / Explanation"
    if source_classification == "quran":
        if document_kind == "translation" or "translation" in title or "translation" in collection:
            return "Translated Primary Text"
        return "Direct Primary Text"
    if source_classification == "hadith":
        return "Hadith Collection Text"
    if source_classification == "fiqh_manual":
        return "Fiqh Manual Text"
    if source_classification == "tasawwuf_text":
        return "Spiritual Guidance (Tasawwuf)"
    if source_classification == "commentary":
        return "Commentary"
    if source_classification == "scholar_transcript":
        return "Scholar Commentary"
    if source_classification == "fatwa":
        return "Modern Fatwa/Application"
    if source_classification == "transcript":
        return "Transcript / Teacher Material"
    return fallback


def format_trust_line(breakdown: Any) -> str:
    """Render a compact, human-readable Trust Weighting line.

    Returns an empty string when:
      - the breakdown is missing or unparseable
      - the profile is 'default' (neutral — never noisy when nothing happened)
      - no components carry a non-zero contribution

    Otherwise returns a single line like:
      ``Trust [hadith_focused]: +0.55 sahih, +0.36 isnad (0.8), -0.18 source_distance = +0.73``

    This is the operationalization of the charter's transparency rule:
    when the Weight and Trust Engine moves a source, the user can see
    exactly why.
    """

    parsed = _parse_breakdown(breakdown)
    if not parsed:
        return ""

    profile_id = str(parsed.get("profile_id", "") or "")
    if profile_id == "default" or not profile_id:
        return ""

    components = parsed.get("components") or []
    nonzero_components = [
        component
        for component in components
        if isinstance(component, Mapping)
        and float(component.get("contribution", 0.0) or 0.0) != 0.0
    ]
    if not nonzero_components:
        return ""

    # Show up to four most-impactful contributions (by absolute value),
    # preserving sign so the user sees both boosts and penalties.
    ranked = sorted(
        nonzero_components,
        key=lambda c: abs(float(c.get("contribution", 0.0) or 0.0)),
        reverse=True,
    )[:4]

    fragments = [_format_trust_fragment(component) for component in ranked]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return ""

    total = float(parsed.get("total", 0.0) or 0.0)
    return f"Trust [{profile_id}]: " + ", ".join(fragments) + f" = {_signed(total)}"


def _format_trust_fragment(component: Mapping[str, Any]) -> str:
    contribution = float(component.get("contribution", 0.0) or 0.0)
    if contribution == 0.0:
        return ""
    signal = str(component.get("signal", "") or "")
    observed = component.get("observed")
    label = _trust_signal_label(signal, observed)
    if not label:
        return ""
    return f"{_signed(contribution)} {label}"


def _trust_signal_label(signal: str, observed: Any) -> str:
    if signal == "authenticity_grade":
        return str(observed or "authenticity")
    if signal == "ijma_strength":
        return f"ijma:{observed}"
    if signal == "era":
        return f"era:{observed}"
    if signal == "madhhab":
        return f"madhhab:{observed}"
    if signal == "methodology_tag":
        return f"method:{observed}"
    if signal == "isnad_strength":
        try:
            return f"isnad ({float(observed):.2f})"
        except (TypeError, ValueError):
            return "isnad"
    if signal == "corroboration_count":
        return f"corroboration (x{observed})"
    if signal == "source_distance":
        return f"distance ({observed})"
    if signal == "scholar_authority":
        try:
            return f"scholar_authority ({float(observed):.2f})"
        except (TypeError, ValueError):
            return "scholar_authority"
    if signal == "unknown_signals":
        if isinstance(observed, list):
            return f"unknown_signals (x{len(observed)})"
        return "unknown_signals"
    return signal or "signal"


def _signed(value: float) -> str:
    if value >= 0:
        return f"+{value:.2f}"
    return f"{value:.2f}"


def _parse_breakdown(breakdown: Any) -> dict[str, Any] | None:
    if breakdown is None:
        return None
    if isinstance(breakdown, Mapping):
        return dict(breakdown)
    if isinstance(breakdown, str):
        stripped = breakdown.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            return None
        if isinstance(payload, dict):
            return payload
        return None
    return None


def _extraction_note(source: Any) -> str:
    ocr_derived = bool(getattr(source, "ocr_derived", False))
    ocr_backend = str(getattr(source, "ocr_backend", "") or "")
    extraction_quality = str(getattr(source, "extraction_quality", "") or "")
    extraction_status = str(getattr(source, "extraction_status", "") or "")
    notes: list[str] = []
    if ocr_derived:
        backend_label = f" via {ocr_backend}" if ocr_backend else ""
        notes.append(f"OCR-derived{backend_label}")
    if extraction_status == "partial":
        notes.append("partial extraction")
    elif extraction_quality in {"machine_extract_with_gaps", "ocr_recovered_with_gaps"}:
        notes.append("extraction with gaps")
    return ", ".join(notes)
