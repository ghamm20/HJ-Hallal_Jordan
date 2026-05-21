"""End-to-end test: a GroundedSource carrying a trust breakdown produces
a visible 'Trust [...]' line in the rendered citation block.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.citations.formatter import format_evidence_entry  # noqa: E402
from app.reasoning.answer_grounding import GroundedSource  # noqa: E402
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
        source_path="data/raw/hadith/bukhari/vol1.pdf",
        collection="Sahih al-Bukhari",
        author="Imam al-Bukhari",
        source_family="hadith_sahih_collections",
        canonical_family="bukhari",
        language="english",
        hierarchy_label="Book 1 / Chapter 1 / Hadith 1",
        book="Book of Revelation",
        chapter="How the Divine Inspiration started",
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


def test_default_profile_breakdown_does_not_add_trust_line():
    """The default profile is neutral — the rendered block must not
    include a noisy 'Trust [default]: = +0.00' line.
    """

    candidate = {"hadith_grade": "sahih", "source_type": "hadith"}
    breakdown_json = _breakdown_json(candidate, "default")
    source = _grounded_source(trust_breakdown_json=breakdown_json)

    lines = format_evidence_entry(source, include_quote=True)
    rendered = "\n".join(lines)
    assert "Trust" not in rendered


def test_active_profile_renders_trust_line_in_evidence_block():
    candidate = {
        "hadith_grade": "sahih",
        "isnad_strength": 0.85,
        "corroboration_count": 3,
        "era": "classical",
        "source_type": "hadith",
    }
    breakdown_json = _breakdown_json(candidate, "hadith_focused")
    source = _grounded_source(trust_breakdown_json=breakdown_json)

    lines = format_evidence_entry(source, include_quote=True)
    rendered = "\n".join(lines)
    assert "Trust [hadith_focused]:" in rendered
    assert "sahih" in rendered
    # Trust line must appear before the Quote, not at the end
    trust_index = next(i for i, line in enumerate(lines) if "Trust [" in line)
    quote_index = next(i for i, line in enumerate(lines) if line.lstrip().startswith('Quote:'))
    assert trust_index < quote_index


def test_empty_breakdown_json_does_not_add_trust_line():
    source = _grounded_source(trust_breakdown_json="")
    lines = format_evidence_entry(source, include_quote=True)
    rendered = "\n".join(lines)
    assert "Trust" not in rendered


def test_malformed_breakdown_json_is_silent_not_fatal():
    """Defensive: a malformed _trust_breakdown_json must not crash the
    renderer — it must silently produce no Trust line.
    """

    source = _grounded_source(trust_breakdown_json="{not valid json")
    lines = format_evidence_entry(source, include_quote=True)
    rendered = "\n".join(lines)
    assert "Trust" not in rendered
    # And the rest of the citation block still renders correctly
    assert any("Sahih al-Bukhari" in line for line in lines)


def _breakdown_json(candidate: dict, profile_id: str) -> str:
    import json
    breakdown = score(candidate, profile=load_profile(profile_id)).to_dict()
    return json.dumps(breakdown)
