"""End-to-end: the methodology disclaimer surfaces in rendered answers.

When any cited source carries a scholar methodology breakdown, the
rendered answer must prepend a 'Methodology Disclosure' banner. This
is the charter's disclosure rule enforced at the rendering layer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.citations.renderer import (  # noqa: E402
    _active_methodology_disclaimer,
    _prepend_methodology_disclaimer,
)
from app.reasoning.answer_grounding import AnswerEvidenceModel, GroundedSource  # noqa: E402
from app.reasoning.trust_engine import load_profile, score  # noqa: E402


def _grounded_source(*, trust_breakdown_json: str = "") -> GroundedSource:
    return GroundedSource(
        title="Sahih al-Bukhari",
        human_title="Sahih al-Bukhari",
        source_classification="hadith",
        source_type_label="Hadith Collection Text",
        evidence_bucket="primary_evidence",
        role="primary_text",
        domain="hadith",
        authority_level="sahih",
        reference="Book 1, Hadith 1",
        section_label="Revelation",
        quote="Actions are by intentions...",
        madhhab="",
        source_path="x.pdf",
        collection="Sahih al-Bukhari",
        author="Imam al-Bukhari",
        source_family="hadith_sahih_collections",
        canonical_family="bukhari",
        language="english",
        hierarchy_label="Book 1 / Chapter 1 / Hadith 1",
        book="Book of Revelation",
        chapter="",
        section="",
        document_kind="hadith_collection",
        source_role_boundary="primary_text",
        source_lineage="bukhari",
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
        trust_breakdown_json=trust_breakdown_json,
    )


def _evidence_model_with_source(source: GroundedSource) -> AnswerEvidenceModel:
    return AnswerEvidenceModel(
        primary_evidence=[source],
        spiritual_guidance=[],
        hanafi_authority=[],
        other_views=[],
        supporting_commentary=[],
        teaching_explanation=[],
        modern_application=[],
        sources=[source],
        disagreement_notes=[],
        uncertainty_notes=[],
        intent_id="direct_source_lookup",
        suppress_synthesis=False,
        authority_policy_id="research",
        comparison_positions=[],
        evidence_backfill_applied=False,
        evidence_backfill_buckets=[],
        source_layer_composition={},
        metadata_completeness={},
        ocr_usage={},
    )


def test_disclaimer_detected_when_scholar_profile_breakdown_present():
    bd = score(
        {"hadith_grade": "sahih", "source_type": "hadith"},
        profile=load_profile("shaykh_jamal_methodology"),
    ).to_dict()
    source = _grounded_source(trust_breakdown_json=json.dumps(bd))
    model = _evidence_model_with_source(source)
    scholar_name, disclaimer = _active_methodology_disclaimer(model)
    assert scholar_name == "Shaykh Jamal"
    assert "methodology modeling" in disclaimer.lower()


def test_disclaimer_not_detected_for_generic_profile():
    bd = score(
        {"hadith_grade": "sahih", "source_type": "hadith"},
        profile=load_profile("hadith_focused"),
    ).to_dict()
    source = _grounded_source(trust_breakdown_json=json.dumps(bd))
    model = _evidence_model_with_source(source)
    scholar_name, disclaimer = _active_methodology_disclaimer(model)
    assert scholar_name == ""
    assert disclaimer == ""


def test_disclaimer_not_detected_when_no_evidence_model():
    scholar_name, disclaimer = _active_methodology_disclaimer(None)
    assert scholar_name == ""
    assert disclaimer == ""


def test_prepend_inserts_banner_before_mode_line():
    bd = score(
        {"hadith_grade": "sahih", "source_type": "hadith"},
        profile=load_profile("dr_umar_methodology"),
    ).to_dict()
    source = _grounded_source(trust_breakdown_json=json.dumps(bd))
    model = _evidence_model_with_source(source)

    lines = ["Mode: research", "Selected Madhhab: hanafi", "", "Direct Answer", "..."]
    _prepend_methodology_disclaimer(lines, model)
    rendered = "\n".join(lines)
    assert "Methodology Disclosure" in rendered
    assert "Dr. Umar" in rendered
    # Banner appears before the Mode line, not at the end
    assert rendered.index("Methodology Disclosure") < rendered.index("Mode: research")


def test_prepend_with_greeting_inserts_after_greeting():
    bd = score(
        {"hadith_grade": "sahih", "source_type": "hadith"},
        profile=load_profile("shaykh_jamal_methodology"),
    ).to_dict()
    source = _grounded_source(trust_breakdown_json=json.dumps(bd))
    model = _evidence_model_with_source(source)

    lines = ["As-salamu alaykum.", "", "Mode: research", "Direct Answer", "..."]
    _prepend_methodology_disclaimer(lines, model)
    rendered = "\n".join(lines)
    # Greeting is still first
    assert rendered.startswith("As-salamu alaykum.")
    # Then disclosure, then the original Mode line
    assert rendered.index("Methodology Disclosure") < rendered.index("Mode: research")


def test_malformed_breakdown_json_silently_skipped():
    source = _grounded_source(trust_breakdown_json="{not valid")
    model = _evidence_model_with_source(source)
    scholar_name, disclaimer = _active_methodology_disclaimer(model)
    assert scholar_name == ""
    assert disclaimer == ""
