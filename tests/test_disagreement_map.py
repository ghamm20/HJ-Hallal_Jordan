"""Tests for the Disagreement Map structured layer.

Pins the charter rule: code never fabricates disagreement. When the
model provides no structured disagreement data, the renderer stays
silent — never emits an empty 'Where Scholars Diverged' section.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.reasoning.disagreement_map import (  # noqa: E402
    DisagreementMap,
    DisagreementPosition,
    parse_disagreement_map,
    render_disagreement_lines,
)


# ---------------------------------------------------------------------------
# Defensive parsing — never invent
# ---------------------------------------------------------------------------


def test_parse_none_returns_none():
    assert parse_disagreement_map(None) is None


def test_parse_empty_dict_returns_none():
    assert parse_disagreement_map({}) is None


def test_parse_empty_list_returns_none():
    assert parse_disagreement_map([]) is None


def test_parse_garbage_types_returns_none():
    assert parse_disagreement_map("just a string") is None
    assert parse_disagreement_map(42) is None
    assert parse_disagreement_map(True) is None


def test_parse_dict_with_only_blanks_returns_none():
    raw = {"point": "", "principle": "", "positions": [], "notes": []}
    assert parse_disagreement_map(raw) is None


def test_position_without_label_dropped():
    raw = {
        "point": "Wiping over leather socks",
        "positions": [
            {"label": "Hanafi", "ruling_summary": "Permitted within limits"},
            {"ruling_summary": "No label — should be dropped"},
        ],
    }
    parsed = parse_disagreement_map(raw)
    assert parsed is not None
    assert len(parsed.positions) == 1
    assert parsed.positions[0].label == "Hanafi"


# ---------------------------------------------------------------------------
# Structured parse — preserves the three required fields the user asked for
# ---------------------------------------------------------------------------


def test_parse_full_structure_preserves_three_axes():
    """The user's explicit ask: show WHERE divergence occurred, WHICH
    PRINCIPLE caused it, and WHAT EVIDENCE each side prioritized.
    """

    raw = {
        "point": "Saying 'amin' aloud in jahri prayers",
        "principle": "Status of the ma'mum following the imam vs. independent recitation",
        "positions": [
            {
                "label": "Hanafi",
                "holders": ["Abu Hanifa", "Hanafi school"],
                "evidence_priorities": [
                    "Hadith of silent following",
                    "Following the imam principle",
                ],
                "ruling_summary": "Said silently",
                "citations": ["Bukhari 780"],
            },
            {
                "label": "Jumhur (Shafi'i, Maliki, Hanbali)",
                "holders": ["al-Shafi'i", "Malik", "Ahmad"],
                "evidence_priorities": ["Hadith of aloud amin", "Athar of Sahaba"],
                "ruling_summary": "Said aloud in jahri prayers",
                "citations": ["Bukhari 780", "Tirmidhi 248"],
            },
        ],
    }
    parsed = parse_disagreement_map(raw)
    assert parsed is not None
    assert parsed.point == "Saying 'amin' aloud in jahri prayers"
    assert parsed.principle.startswith("Status of the ma'mum")
    assert len(parsed.positions) == 2
    hanafi = parsed.positions[0]
    assert hanafi.holders == ("Abu Hanifa", "Hanafi school")
    assert "Following the imam principle" in hanafi.evidence_priorities
    assert hanafi.ruling_summary == "Said silently"
    assert "Bukhari 780" in hanafi.citations


def test_parse_list_of_positions_accepted_for_backward_compat():
    raw = [
        {"label": "Hanafi", "ruling_summary": "A"},
        {"label": "Shafi'i", "ruling_summary": "B"},
    ]
    parsed = parse_disagreement_map(raw)
    assert parsed is not None
    assert len(parsed.positions) == 2


def test_evidence_priorities_accepts_comma_string():
    raw = {
        "point": "Test",
        "positions": [
            {"label": "X", "evidence_priorities": "A, B, C"},
        ],
    }
    parsed = parse_disagreement_map(raw)
    assert parsed.positions[0].evidence_priorities == ("A", "B", "C")


# ---------------------------------------------------------------------------
# Rendering — silent when empty, structured when populated
# ---------------------------------------------------------------------------


def test_render_silent_for_none():
    assert render_disagreement_lines(None) == []


def test_render_silent_for_empty_map():
    empty = DisagreementMap(point="", principle="")
    assert render_disagreement_lines(empty) == []


def test_render_includes_three_axes():
    dm = DisagreementMap(
        point="Test point",
        principle="Test principle",
        positions=(
            DisagreementPosition(
                label="Hanafi",
                holders=("Abu Hanifa",),
                evidence_priorities=("Following imam",),
                ruling_summary="Silent",
                citations=("Bukhari 780",),
            ),
        ),
    )
    lines = render_disagreement_lines(dm)
    rendered = "\n".join(lines)
    assert "Where Scholars Diverged" in rendered
    assert "Point: Test point" in rendered
    assert "Principle of divergence: Test principle" in rendered
    assert "Hanafi" in rendered
    assert "Held by: Abu Hanifa" in rendered
    assert "Evidence prioritized: Following imam" in rendered
    assert "Ruling: Silent" in rendered
    assert "Cites: Bukhari 780" in rendered


def test_render_handles_position_without_optional_fields():
    dm = DisagreementMap(
        point="X",
        principle="",
        positions=(DisagreementPosition(label="Solo"),),
    )
    lines = render_disagreement_lines(dm)
    rendered = "\n".join(lines)
    assert "Solo" in rendered
    assert "Held by" not in rendered  # no holders -> no line
    assert "Cites" not in rendered  # no citations -> no line
