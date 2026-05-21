"""End-to-end test: the three structural layers appear in rendered answers.

Verifies the renderer attaches Evidence Ladder, Where Scholars Diverged,
and Scholarly Confidence sections when applicable, and stays silent
otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.citations.renderer import render_answer  # noqa: E402
from app.reasoning.answer_grounding import AnswerEvidenceModel, GroundedSource  # noqa: E402


def _grounded_source(**overrides) -> GroundedSource:
    base = dict(
        title="Sahih al-Bukhari",
        human_title="Sahih al-Bukhari",
        source_classification="hadith",
        source_type_label="Hadith",
        evidence_bucket="primary_evidence",
        role="primary_text",
        domain="hadith",
        authority_level="sahih",
        reference="Book 1, Hadith 1",
        section_label="",
        quote="Actions are by intentions",
        madhhab="",
        source_path="x.pdf",
        collection="Sahih al-Bukhari",
        author="al-Bukhari",
        source_family="hadith",
        canonical_family="bukhari",
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


def _model(sources, **kwargs) -> AnswerEvidenceModel:
    return AnswerEvidenceModel(
        primary_evidence=kwargs.get("primary_evidence", []),
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


def _answer(**overrides):
    base = {
        "answer_mode": "research",
        "selected_madhhab": "hanafi",
        "direct_answer": "...",
    }
    base.update(overrides)
    return base


def test_evidence_ladder_appears_when_sources_present():
    src = _grounded_source(source_classification="hadith", authority_level="sahih")
    rendered = render_answer(_answer(), _model([src]))
    assert "Evidence Ladder" in rendered
    assert "Sahih Hadith" in rendered


def test_confidence_appears_when_evidence_present():
    src = _grounded_source(source_classification="hadith", authority_level="sahih")
    rendered = render_answer(_answer(), _model([src]))
    assert "Scholarly Confidence" in rendered


def test_disagreement_map_appears_when_provided():
    src = _grounded_source()
    answer = _answer(
        disagreement_map={
            "point": "Test divergence",
            "principle": "Test principle",
            "positions": [
                {
                    "label": "Hanafi",
                    "evidence_priorities": ["A"],
                    "ruling_summary": "Silent",
                }
            ],
        }
    )
    rendered = render_answer(answer, _model([src]))
    assert "Where Scholars Diverged" in rendered
    assert "Test divergence" in rendered
    assert "Test principle" in rendered


def test_disagreement_map_silent_when_absent():
    src = _grounded_source()
    rendered = render_answer(_answer(), _model([src]))
    assert "Where Scholars Diverged" not in rendered


def test_three_layers_render_in_expected_order():
    src = _grounded_source(source_classification="quran")
    answer = _answer(
        disagreement_map={
            "point": "x",
            "positions": [{"label": "A", "ruling_summary": "y"}],
        }
    )
    rendered = render_answer(answer, _model([src]))
    # Order: Evidence Ladder, then Where Scholars Diverged, then Scholarly Confidence
    ladder_idx = rendered.index("Evidence Ladder")
    diverged_idx = rendered.index("Where Scholars Diverged")
    confidence_idx = rendered.index("Scholarly Confidence")
    assert ladder_idx < diverged_idx < confidence_idx


def test_render_unaffected_when_no_evidence_model():
    """Legacy path (no evidence_model) must still render. The three
    layers are evidence-model-dependent and stay quiet.
    """

    rendered = render_answer(
        _answer(
            direct_answer="Legacy answer",
            citations=[],
        ),
        evidence_model=None,
    )
    assert "Evidence Ladder" not in rendered
    assert "Scholarly Confidence" not in rendered
    assert "Where Scholars Diverged" not in rendered
    assert "Legacy answer" in rendered
