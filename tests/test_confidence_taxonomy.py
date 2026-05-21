"""Tests for the Confidence Taxonomy classifier.

Pins:
  - Confidence is NEVER numeric (no float, no percentage)
  - Disagreement always downgrades to 'valid_disagreement' or below
  - Empty / unknown evidence -> None (renderer stays silent), never
    a fabricated confident label
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.reasoning.answer_grounding import AnswerEvidenceModel, GroundedSource  # noqa: E402
from app.reasoning.confidence_taxonomy import (  # noqa: E402
    LEVEL_LABELS,
    classify_confidence,
    render_confidence_lines,
)


def _grounded_source(**overrides) -> GroundedSource:
    base = dict(
        title="X",
        human_title="X",
        source_classification="hadith",
        source_type_label="Hadith",
        evidence_bucket="primary_evidence",
        role="primary_text",
        domain="hadith",
        authority_level="",
        reference="ref",
        section_label="",
        quote="",
        madhhab="",
        source_path="x.pdf",
        collection="",
        author="",
        source_family="",
        canonical_family="",
        language="english",
        hierarchy_label="",
        book="",
        chapter="",
        section="",
        document_kind="",
        source_role_boundary="",
        source_lineage="",
        commentary_target="",
        fatwa_authority="",
        legal_role="primary_text",
        legal_role_label="Primary Text",
        ocr_derived=False,
        ocr_backend="",
        ocr_status="",
        ocr_confidence="",
        extraction_status="success",
        extraction_quality="clean",
        scholar_attribution_match="",
        trust_breakdown_json="",
    )
    base.update(overrides)
    return GroundedSource(**base)


def _evidence_model(sources, **kwargs) -> AnswerEvidenceModel:
    return AnswerEvidenceModel(
        primary_evidence=[],
        spiritual_guidance=[],
        hanafi_authority=[],
        other_views=[],
        supporting_commentary=[],
        teaching_explanation=[],
        modern_application=[],
        sources=sources,
        disagreement_notes=kwargs.get("disagreement_notes", []),
        uncertainty_notes=[],
        intent_id="",
        suppress_synthesis=False,
        authority_policy_id="",
        comparison_positions=kwargs.get("comparison_positions", []),
        evidence_backfill_applied=False,
        evidence_backfill_buckets=[],
        source_layer_composition={},
        metadata_completeness={},
        ocr_usage={},
    )


# ---------------------------------------------------------------------------
# Charter rule: silent when no evidence
# ---------------------------------------------------------------------------


def test_returns_none_when_no_sources():
    assert classify_confidence(evidence_model=None, answer={}) is None
    model = _evidence_model([])
    assert classify_confidence(evidence_model=model, answer={}) is None


# ---------------------------------------------------------------------------
# Output shape — never numeric
# ---------------------------------------------------------------------------


def test_assessment_label_is_from_fixed_taxonomy():
    src = _grounded_source(source_classification="hadith", authority_level="sahih")
    model = _evidence_model([src])
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment is not None
    assert assessment.level_id in LEVEL_LABELS
    assert assessment.label == LEVEL_LABELS[assessment.level_id]
    # No numeric leakage
    assert not any(ch.isdigit() for ch in assessment.label)


def test_to_dict_has_no_numeric_confidence():
    src = _grounded_source(source_classification="quran", authority_level="")
    model = _evidence_model([src])
    assessment = classify_confidence(evidence_model=model, answer={})
    payload = assessment.to_dict()
    assert "confidence" not in payload  # explicitly not numeric
    assert "score" not in payload
    assert "level_id" in payload and "label" in payload and "reasoning" in payload


# ---------------------------------------------------------------------------
# Specific signal -> level mappings
# ---------------------------------------------------------------------------


def test_explicit_text_when_sahih_primary_and_no_disagreement():
    src = _grounded_source(
        source_classification="hadith",
        authority_level="sahih",
        title="Sahih Bukhari",
    )
    model = _evidence_model([src])
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment.level_id == "explicit_text"


def test_disagreement_downgrades_to_valid_disagreement():
    """Charter rule: scholarly disagreement must never be silenced.
    Even when sahih primary text is cited, presence of comparison
    positions downgrades to 'valid_disagreement'.
    """

    src = _grounded_source(source_classification="hadith", authority_level="sahih")
    model = _evidence_model(
        [src],
        comparison_positions=[object()],  # presence is what matters
    )
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment.level_id == "valid_disagreement"


def test_disagreement_note_string_triggers_disagreement_level():
    src = _grounded_source(source_classification="hadith", authority_level="sahih")
    model = _evidence_model([src])
    assessment = classify_confidence(
        evidence_model=model,
        answer={"disagreement_note": "Scholars differ on the timing of fajr."},
    )
    assert assessment.level_id == "valid_disagreement"


def test_weakly_evidenced_when_majority_daif():
    sources = [
        _grounded_source(source_classification="hadith", authority_level="daif"),
        _grounded_source(source_classification="hadith", authority_level="daif"),
        _grounded_source(source_classification="hadith", authority_level=""),
    ]
    model = _evidence_model(sources)
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment.level_id == "weakly_evidenced"


def test_contemporary_extrapolation_when_only_fatwas():
    sources = [
        _grounded_source(source_classification="fatwa"),
        _grounded_source(source_classification="fatwa"),
    ]
    model = _evidence_model(sources)
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment.level_id == "contemporary_extrapolation"


def test_strong_madhhab_position_when_fiqh_manual_matches_selected():
    src = _grounded_source(source_classification="fiqh_manual", madhhab="hanafi")
    model = _evidence_model([src])
    assessment = classify_confidence(
        evidence_model=model,
        answer={"selected_madhhab": "hanafi"},
    )
    assert assessment.level_id == "strong_madhhab_position"


def test_majority_position_when_multiple_fiqh_manuals_no_selected_madhhab():
    sources = [
        _grounded_source(source_classification="fiqh_manual", madhhab="hanafi"),
        _grounded_source(source_classification="fiqh_manual", madhhab="shafii"),
    ]
    model = _evidence_model(sources)
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment.level_id == "majority_position"


def test_speculative_default_when_no_strong_signals():
    sources = [
        _grounded_source(source_classification="commentary"),
    ]
    model = _evidence_model(sources)
    assessment = classify_confidence(evidence_model=model, answer={})
    assert assessment.level_id == "speculative"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_silent_for_none():
    assert render_confidence_lines(None) == []


def test_render_includes_label_and_why():
    src = _grounded_source(source_classification="quran")
    assessment = classify_confidence(
        evidence_model=_evidence_model([src]),
        answer={},
    )
    lines = render_confidence_lines(assessment)
    rendered = "\n".join(lines)
    assert "Scholarly Confidence" in rendered
    assert "Level:" in rendered
    assert "Why:" in rendered
