"""Plain-text renderer for grounded Halal Jordan answers."""

from __future__ import annotations

import json
from typing import Any

from app.citations.formatter import format_evidence_entry, format_source_line
from app.reasoning.answer_grounding import AnswerEvidenceModel, GroundedSource
from app.reasoning.confidence_taxonomy import classify_confidence, render_confidence_lines
from app.reasoning.disagreement_map import parse_disagreement_map, render_disagreement_lines
from app.reasoning.evidence_ladder import classify_sources, render_ladder_lines
from app.reasoning.study_path import layer_labels


def _append_structural_layers(
    lines: list[str],
    *,
    answer: dict[str, Any],
    evidence_model: AnswerEvidenceModel | None,
) -> None:
    """Append the three structural layers — Evidence Ladder, Where
    Scholars Diverged, Scholarly Confidence — to a rendered answer.

    Each layer is silent if it has nothing to show. The order is
    deliberate: the ladder shows what the answer rests on, the
    disagreement map shows where scholars differ on it, and the
    confidence label sits last as a single epistemic summary.
    """

    if evidence_model is None:
        return

    sources = list(evidence_model.sources or [])
    if sources:
        ladder = classify_sources(sources)
        if not ladder.is_empty():
            lines.append("")
            lines.extend(render_ladder_lines(ladder, source_formatter=format_source_line))

    disagreement = parse_disagreement_map(answer.get("disagreement_map"))
    disagreement_lines = render_disagreement_lines(disagreement)
    if disagreement_lines:
        lines.append("")
        lines.extend(disagreement_lines)

    assessment = classify_confidence(evidence_model=evidence_model, answer=answer)
    confidence_lines = render_confidence_lines(assessment)
    if confidence_lines:
        lines.append("")
        lines.extend(confidence_lines)


def _active_methodology_disclaimer(
    evidence_model: AnswerEvidenceModel | None,
) -> tuple[str, str]:
    """If any cited source carries a scholar methodology breakdown,
    return ``(scholar_name, disclaimer)`` for surfacing at the top of
    the answer.

    The charter requires that when a scholar methodology profile is
    active, the system always discloses that it is methodology modeling
    — not the actual scholar speaking, not revelation, not divine
    certainty. This is the rendering side of that rule.
    """

    if evidence_model is None:
        return ("", "")
    for source in evidence_model.sources or []:
        breakdown_json = getattr(source, "trust_breakdown_json", "") or ""
        if not breakdown_json:
            continue
        try:
            breakdown = json.loads(breakdown_json)
        except (TypeError, ValueError):
            continue
        if not isinstance(breakdown, dict):
            continue
        if not breakdown.get("is_scholar_methodology"):
            continue
        disclaimer = str(breakdown.get("methodology_disclaimer") or "").strip()
        if not disclaimer:
            continue
        scholar_name = str(breakdown.get("scholar_name") or "").strip()
        return (scholar_name, disclaimer)
    return ("", "")


def _prepend_methodology_disclaimer(
    lines: list[str],
    evidence_model: AnswerEvidenceModel | None,
) -> None:
    scholar_name, disclaimer = _active_methodology_disclaimer(evidence_model)
    if not disclaimer:
        return
    banner = ["Methodology Disclosure"]
    if scholar_name:
        banner.append(f"Active profile models the methodology associated with: {scholar_name}.")
    banner.append(disclaimer)
    banner.append("")
    # Insert after greeting (if present) so the disclosure leads the answer.
    insert_at = 0
    if lines and lines[0] and not lines[0].startswith("Mode:"):
        # Skip past greeting + blank line
        for idx, line in enumerate(lines):
            if line.startswith("Mode:"):
                insert_at = idx
                break
    lines[insert_at:insert_at] = banner


def render_answer(
    answer: dict[str, Any],
    evidence_model: AnswerEvidenceModel | None = None,
) -> str:
    if answer.get("answer_mode") == "study_path":
        return _render_study_path_answer(answer, evidence_model)
    if answer.get("answer_mode") == "scholar_perspective":
        return _render_scholar_perspective_answer(answer, evidence_model)
    if evidence_model is None:
        return _render_legacy_answer(answer)

    minimal_mode = answer["answer_mode"] == "source_only" or evidence_model.suppress_synthesis
    lines: list[str] = []
    greeting = answer.get("greeting")
    if greeting:
        lines.append(greeting)
        lines.append("")

    lines.append(f"Mode: {answer['answer_mode']}")
    lines.append(f"Selected Madhhab: {answer['selected_madhhab']}")
    should_render_direct_answer = bool(answer.get("direct_answer", "").strip()) and (
        not minimal_mode or not evidence_model.sources
    )
    if should_render_direct_answer:
        lines.append("")
        lines.append("Direct Answer")
        lines.append(answer.get("direct_answer", ""))

    madhhab_position = answer.get("madhhab_position")
    if madhhab_position and not minimal_mode:
        lines.append("")
        lines.append("Madhhab Position")
        lines.append(madhhab_position)

    evidence_summary = answer.get("evidence_summary")
    if evidence_summary and not minimal_mode:
        lines.append("")
        lines.append("Evidence Summary")
        lines.append(evidence_summary)

    if evidence_model.comparison_positions and not minimal_mode:
        lines.append("")
        lines.append("Comparison Positions")
        for position in evidence_model.comparison_positions:
            lines.append(f"- {position.label}")
            if position.source_types_used:
                lines.append(
                    "  Source Types: " + ", ".join(position.source_types_used)
                )
            lines.append("  Supporting Sources:")
            for source in position.supporting_sources:
                lines.append(f"  {format_source_line(source)}")
            for note in position.uncertainty_notes:
                lines.append(f"  Ambiguity: {note}")

    _render_group(
        lines,
        "Primary Texts",
        evidence_model.primary_evidence,
        include_quotes=True,
    )
    _render_group(
        lines,
        "Spiritual Guidance (Tasawwuf)",
        evidence_model.spiritual_guidance,
        include_quotes=True,
        notes=["This reflects classical spiritual teachings, not legal rulings."],
    )
    _render_group(
        lines,
        _selected_madhhab_view_title(answer),
        evidence_model.hanafi_authority,
        include_quotes=not minimal_mode,
    )
    if not minimal_mode:
        _render_group(
            lines,
            "Other Views",
            evidence_model.other_views,
            include_quotes=True,
            notes=evidence_model.disagreement_notes,
        )
        _render_group(
            lines,
            "Supporting Commentary",
            evidence_model.supporting_commentary,
            include_quotes=True,
        )
        _render_group(
            lines,
            "Teaching / Explanation",
            evidence_model.teaching_explanation,
            include_quotes=True,
            notes=[
                "Teaching materials support explanation and study context only; they do not function as primary text or madhhab authority."
            ],
        )
        _render_group(
            lines,
            "Modern Fatwa/Application",
            evidence_model.modern_application,
            include_quotes=True,
        )

    if evidence_model.uncertainty_notes:
        lines.append("")
        lines.append("Limits / Uncertainty")
        for note in evidence_model.uncertainty_notes:
            lines.append(f"- {note}")

    citations = evidence_model.sources
    lines.append("")
    lines.append("Sources")
    if citations:
        for source in citations:
            lines.append(format_source_line(source))
    else:
        lines.append("- None")

    evidence_strength = answer.get("evidence_strength")
    if evidence_strength:
        lines.append("")
        lines.append(f"Evidence Strength: {evidence_strength}")

    _append_structural_layers(lines, answer=answer, evidence_model=evidence_model)
    _prepend_methodology_disclaimer(lines, evidence_model)
    return "\n".join(lines)


def _render_scholar_perspective_answer(
    answer: dict[str, Any],
    evidence_model: AnswerEvidenceModel | None = None,
) -> str:
    if evidence_model is None:
        return _render_legacy_answer(answer)

    lines: list[str] = []
    greeting = answer.get("greeting")
    if greeting:
        lines.append(greeting)
        lines.append("")

    lines.append("Mode: scholar_perspective")
    lines.append(f"Selected Madhhab: {answer.get('selected_madhhab', 'not_specified')}")

    scholar = evidence_model.scholar_attribution
    if scholar:
        lines.append("")
        lines.append("Scholar Attribution")
        lines.append(f"Scholar: {scholar.get('name', 'Unknown Scholar')}")
        if scholar.get("madhhab"):
            lines.append(f"Madhhab: {scholar['madhhab']}")
        if scholar.get("period"):
            lines.append(f"Period: {scholar['period']}")
        if scholar.get("disclaimer"):
            lines.append(str(scholar["disclaimer"]))
        if scholar.get("methodology_notes"):
            lines.append(f"Methodology Note: {scholar['methodology_notes']}")

    lines.append("")
    lines.append("From Recorded Works and Positions")
    lines.append(str(answer.get("direct_answer") or "").strip())

    evidence_summary = str(answer.get("evidence_summary") or "").strip()
    if evidence_summary:
        lines.append("")
        lines.append("Evidence Summary")
        lines.append(evidence_summary)

    if evidence_model.scholar_direct_sources:
        _render_group(
            lines,
            "Directly Attributed Material",
            evidence_model.scholar_direct_sources,
            include_quotes=True,
        )
    if evidence_model.scholar_context_sources:
        _render_group(
            lines,
            "Contextually Related Material",
            evidence_model.scholar_context_sources,
            include_quotes=True,
            notes=[
                "These sources are contextually related to the scholar or school but are not presented as direct attribution."
            ],
        )
    if evidence_model.scholar_thin_attribution_note:
        lines.append("")
        lines.append("Attribution Limits")
        lines.append(f"- {evidence_model.scholar_thin_attribution_note}")

    madhhab_position = str(answer.get("madhhab_position") or "").strip()
    if madhhab_position:
        lines.append("")
        lines.append("Madhhab Context")
        lines.append(madhhab_position)

    if evidence_model.uncertainty_notes:
        lines.append("")
        lines.append("Uncertainty / Limits")
        for note in evidence_model.uncertainty_notes:
            lines.append(f"- {note}")

    lines.append("")
    lines.append("Sources")
    if evidence_model.sources:
        for source in evidence_model.sources:
            lines.append(format_source_line(source))
    else:
        lines.append("- None")

    evidence_strength = answer.get("evidence_strength")
    if evidence_strength:
        lines.append("")
        lines.append(f"Evidence Strength: {evidence_strength}")

    _append_structural_layers(lines, answer=answer, evidence_model=evidence_model)
    _prepend_methodology_disclaimer(lines, evidence_model)
    return "\n".join(lines)


def _render_study_path_answer(
    answer: dict[str, Any],
    evidence_model: AnswerEvidenceModel | None = None,
) -> str:
    lines: list[str] = []
    greeting = answer.get("greeting")
    if greeting:
        lines.append(greeting)
        lines.append("")

    lines.append("Mode: study_path")
    lines.append(f"Selected Madhhab: {answer.get('selected_madhhab', 'not_specified')}")

    topic = str(answer.get("topic") or "").strip()
    if topic:
        lines.append("")
        lines.append("Topic")
        lines.append(topic)

    overview = str(answer.get("overview") or answer.get("direct_answer") or "").strip()
    if overview:
        lines.append("")
        lines.append("Overview")
        lines.append(overview)

    madhhab_readiness = answer.get("madhhab_readiness", {})
    if isinstance(madhhab_readiness, dict) and madhhab_readiness:
        lines.append("")
        lines.append("Madhhab Readiness")
        for key in ("hanafi_ready", "shafi_ready", "maliki_ready", "hanbali_ready"):
            if key in madhhab_readiness:
                lines.append(f"- {key}: {madhhab_readiness[key]}")

    source_layers = answer.get("source_layers", {})
    if isinstance(source_layers, dict):
        for layer_key, label in layer_labels():
            lines.append("")
            lines.append(label)
            items = source_layers.get(layer_key, [])
            if not isinstance(items, list) or not items:
                lines.append("No grounded source found for this layer.")
                continue
            for item in items:
                lines.append(_study_source_line(item))
                quote = str(item.get("quote") or "").strip()
                if quote:
                    lines.append(f"  Quote: {quote}")
            if layer_key == "tasawwuf":
                lines.append("- Note: This reflects classical spiritual teachings, not legal rulings.")
            if layer_key == "scholar_commentary":
                lines.append("- Note: Scholar commentary supports explanation and teaching context only.")

    reading_list = answer.get("reading_list", {})
    if isinstance(reading_list, dict):
        lines.append("")
        lines.append("Reading List")
        _render_reading_list_group(lines, "Beginner", reading_list.get("beginner"))
        _render_reading_list_group(lines, "Intermediate", reading_list.get("intermediate"))
        _render_reading_list_group(lines, "Advanced / Deeper Study", reading_list.get("advanced"))

    lesson_path = answer.get("lesson_path", [])
    if isinstance(lesson_path, list):
        lines.append("")
        lines.append("Lesson Path")
        if lesson_path:
            for lesson in lesson_path:
                lines.append(f"- {lesson.get('lesson_title', 'Lesson')}")
                lines.append(f"  Objective: {lesson.get('objective', '')}")
                anchors = lesson.get("source_anchors", [])
                if anchors:
                    lines.append("  Source Anchors:")
                    for anchor in anchors:
                        lines.append(f"  - {anchor}")
                lines.append(f"  Explanation: {lesson.get('short_explanation', '')}")
                practice_prompt = str(lesson.get("practice_prompt") or "").strip()
                if practice_prompt:
                    lines.append(f"  Practice / Reflection: {practice_prompt}")
        else:
            lines.append("- No grounded lesson path could be assembled.")

    key_terms = answer.get("key_terms", [])
    if isinstance(key_terms, list):
        lines.append("")
        lines.append("Key Terms")
        if key_terms:
            for term in key_terms:
                lines.append(f"- {term}")
        else:
            lines.append("- None")

    what_to_avoid = answer.get("what_to_avoid", [])
    if isinstance(what_to_avoid, list):
        lines.append("")
        lines.append("What to Avoid")
        if what_to_avoid:
            for item in what_to_avoid:
                lines.append(f"- {item}")
        else:
            lines.append("- None")

    corpus_gaps = answer.get("corpus_gaps", [])
    uncertainty_note = str(answer.get("uncertainty_note") or "").strip()
    if uncertainty_note or corpus_gaps:
        lines.append("")
        lines.append("Uncertainty / Corpus Gaps")
        if uncertainty_note:
            lines.append(f"- {uncertainty_note}")
        if isinstance(corpus_gaps, list):
            for gap in corpus_gaps:
                lines.append(f"- {gap}")

    citations = (
        evidence_model.sources
        if evidence_model is not None
        else answer.get("citations", [])
    )
    lines.append("")
    lines.append("Sources")
    if citations:
        for source in citations:
            lines.append(format_source_line(source))
    else:
        lines.append("- None")

    evidence_strength = answer.get("evidence_strength")
    if evidence_strength:
        lines.append("")
        lines.append(f"Evidence Strength: {evidence_strength}")

    _append_structural_layers(lines, answer=answer, evidence_model=evidence_model)
    _prepend_methodology_disclaimer(lines, evidence_model)
    return "\n".join(lines)


def _render_group(
    lines: list[str],
    title: str,
    sources: list[Any],
    *,
    include_quotes: bool,
    notes: list[str] | None = None,
) -> None:
    if not sources and not notes:
        return
    lines.append("")
    lines.append(title)
    for source in sources:
        lines.extend(format_evidence_entry(source, include_quote=include_quotes))
    for note in notes or []:
        lines.append(f"- Note: {note}")


def _render_legacy_answer(answer: dict[str, Any]) -> str:
    lines: list[str] = []

    greeting = answer.get("greeting")
    if greeting:
        lines.append(greeting)
        lines.append("")

    lines.append(f"Mode: {answer['answer_mode']}")
    lines.append(f"Selected Madhhab: {answer['selected_madhhab']}")
    lines.append("")
    lines.append("Direct Answer")
    lines.append(answer.get("direct_answer", ""))

    madhhab_position = answer.get("madhhab_position")
    if madhhab_position:
        lines.append("")
        lines.append("Madhhab Position")
        lines.append(madhhab_position)

    evidence_summary = answer.get("evidence_summary")
    if evidence_summary:
        lines.append("")
        lines.append("Evidence Summary")
        lines.append(evidence_summary)

    citations = answer.get("citations", [])
    lines.append("")
    lines.append("Citations")
    if citations:
        for citation in citations:
            label = (
                f"- {citation['title']} "
                f"[{citation['source_type']}] {citation['reference']}"
            )
            lines.append(label)
            quote = citation.get("quote")
            if quote:
                lines.append(f"  Quote: {quote}")
    else:
        lines.append("- None")

    disagreement = answer.get("disagreement_note")
    if disagreement:
        lines.append("")
        lines.append("Disagreement")
        lines.append(disagreement)

    uncertainty = answer.get("uncertainty_note")
    if uncertainty:
        lines.append("")
        lines.append("Uncertainty")
        lines.append(uncertainty)

    evidence_strength = answer.get("evidence_strength")
    if evidence_strength:
        lines.append("")
        lines.append(f"Evidence Strength: {evidence_strength}")

    return "\n".join(lines)


def _selected_madhhab_view_title(answer: dict[str, Any]) -> str:
    selected_madhhab = str(answer.get("selected_madhhab", "") or "").strip()
    if selected_madhhab in {"", "not_specified", "compare_all"}:
        return "Selected Madhhab View"
    return f"{selected_madhhab.title()} View"


def _study_source_line(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "Retrieved Source").strip()
    source_type = str(item.get("source_type") or "unknown").strip()
    reference = str(item.get("reference") or "Provided snippet").strip()
    line = f"- {title} [{source_type}] {reference}"
    collection = str(item.get("collection") or "").strip()
    author = str(item.get("author") or "").strip()
    extras = [value for value in (collection, author) if value]
    if extras:
        line += " | " + " | ".join(extras)
    return line


def _render_reading_list_group(lines: list[str], title: str, items: Any) -> None:
    lines.append(f"- {title}")
    if not isinstance(items, list) or not items:
        lines.append("  No grounded source found for this layer.")
        return
    for item in items:
        label = str(item.get("title") or "Untitled source").strip()
        layer = str(item.get("layer") or "").strip()
        collection = str(item.get("collection") or "").strip()
        author = str(item.get("author") or "").strip()
        segments = [segment for segment in (layer, collection, author) if segment]
        if segments:
            lines.append(f"  - {label} ({'; '.join(segments)})")
        else:
            lines.append(f"  - {label}")
