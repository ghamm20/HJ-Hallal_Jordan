"""Tests for the Evidence Ladder classifier.

Pins the charter rule: code never invents hadith grades or ijma claims.
A source with unknown grade does NOT get the sahih lift; it falls into
the commentary tier (unless a documented collection prior justifies a
higher placement).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.reasoning.evidence_ladder import (  # noqa: E402
    EVIDENCE_LADDER,
    TIER_BY_ID,
    classify_sources,
    render_ladder_lines,
)


def _source(**kwargs):
    base = {
        "title": "",
        "source_classification": "",
        "source_type": "",
        "hadith_grade": "",
        "madhhab": "",
        "collection": "",
        "reference": "",
        "section_label": "",
    }
    base.update(kwargs)
    return base


def test_quran_source_lands_in_quran_tier():
    result = classify_sources([_source(source_classification="quran", title="Al-Baqarah")])
    assert "quran" in result.tiers
    assert len(result.tiers["quran"]) == 1


def test_sahih_hadith_lands_in_sahih_tier():
    result = classify_sources([
        _source(source_classification="hadith", hadith_grade="sahih", title="Bukhari 1"),
    ])
    assert "sahih_hadith" in result.tiers
    assert len(result.tiers["sahih_hadith"]) == 1


def test_mutawatir_hadith_lands_in_mutawatir_tier():
    result = classify_sources([
        _source(source_classification="hadith", hadith_grade="mutawatir", title="X"),
    ])
    assert "mutawatir_hadith" in result.tiers


def test_daif_hadith_lands_in_weak_evidence_tier():
    result = classify_sources([
        _source(source_classification="hadith", hadith_grade="daif", title="X"),
    ])
    assert "weak_evidence" in result.tiers


def test_hadith_with_unknown_grade_does_not_get_sahih_lift():
    """Charter rule: code never invents authenticity. A hadith without
    explicit grading and without a documented collection prior must NOT
    land in the sahih tier — it falls to commentary.
    """

    result = classify_sources([
        _source(
            source_classification="hadith",
            title="Random hadith",
            collection="some other collection",
        )
    ])
    assert "sahih_hadith" not in result.tiers
    assert "commentary" in result.tiers
    # And the reason explains why
    entry = result.tiers["commentary"][0]
    assert "without" in entry.reason.lower() or "unclass" in entry.reason.lower()


def test_fiqh_manual_with_ijma_claim_lands_in_ijma_tier():
    result = classify_sources([
        _source(
            source_classification="fiqh_manual",
            ijma_strength="ijma",
            title="Al-Mughni",
        ),
    ])
    assert "ijma" in result.tiers


def test_fiqh_manual_without_ijma_lands_in_madhhab_reasoning():
    result = classify_sources([
        _source(source_classification="fiqh_manual", title="Hanafi manual"),
    ])
    assert "madhhab_reasoning" in result.tiers


def test_fatwa_lands_in_modern_fatwa_tier():
    result = classify_sources([_source(source_classification="fatwa", title="Modern ruling")])
    assert "modern_fatwa" in result.tiers


def test_tasawwuf_text_lands_in_commentary_tier():
    """Tasawwuf is spiritual guidance, not primary text or legal authority.
    Charter rule: tasawwuf must never be treated as ruling-tier evidence.
    """

    result = classify_sources([
        _source(source_classification="tasawwuf_text", title="Ihya")
    ])
    assert "commentary" in result.tiers
    # Critically: not in primary text or ijma tiers
    assert "quran" not in result.tiers
    assert "sahih_hadith" not in result.tiers
    assert "ijma" not in result.tiers


def test_populated_tiers_returned_in_ladder_order():
    sources = [
        _source(source_classification="fatwa"),
        _source(source_classification="quran"),
        _source(source_classification="hadith", hadith_grade="sahih"),
    ]
    result = classify_sources(sources)
    populated = result.populated_tiers()
    populated_ids = [tier.tier_id for tier in populated]
    # Must be in canonical order: quran first, then sahih, then modern_fatwa
    assert populated_ids == ["quran", "sahih_hadith", "modern_fatwa"]


def test_renderer_silent_for_empty_ladder():
    result = classify_sources([])
    lines = render_ladder_lines(result, source_formatter=lambda s: "X")
    assert lines == []


def test_renderer_emits_section_with_populated_tiers():
    sources = [
        _source(source_classification="quran", title="Q"),
        _source(source_classification="hadith", hadith_grade="sahih", title="H"),
    ]
    result = classify_sources(sources)
    lines = render_ladder_lines(result, source_formatter=lambda s: s["title"])
    rendered = "\n".join(lines)
    assert "Evidence Ladder" in rendered
    assert "Qur'an" in rendered
    assert "Sahih Hadith" in rendered
    # Each tier shows the count badge
    assert "[1]" in rendered


def test_ladder_canonical_order_intact():
    expected = [
        "quran",
        "mutawatir_hadith",
        "sahih_hadith",
        "hasan_hadith",
        "athar",
        "ijma",
        "qiyas",
        "madhhab_reasoning",
        "modern_fatwa",
        "commentary",
        "weak_evidence",
    ]
    assert [tier.tier_id for tier in EVIDENCE_LADDER] == expected
