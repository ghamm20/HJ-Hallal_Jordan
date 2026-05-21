"""Tests for the citation formatter's Trust Weighting line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.citations.formatter import format_trust_line  # noqa: E402
from app.reasoning.trust_engine import load_profile, score  # noqa: E402


def _breakdown_for(candidate: dict, profile_id: str) -> dict:
    profile = load_profile(profile_id)
    return score(candidate, profile=profile).to_dict()


# ---------------------------------------------------------------------------
# Silence rules — never noisy when nothing happened
# ---------------------------------------------------------------------------


def test_returns_empty_for_missing_breakdown():
    assert format_trust_line(None) == ""
    assert format_trust_line("") == ""
    assert format_trust_line({}) == ""


def test_returns_empty_for_default_profile_breakdown():
    """The default profile is neutral — its line would say nothing useful,
    so we stay silent rather than producing 'Trust [default]: = +0.00'.
    """

    rich_candidate = {
        "hadith_grade": "sahih",
        "isnad_strength": 0.9,
        "source_type": "hadith",
    }
    breakdown = _breakdown_for(rich_candidate, "default")
    assert format_trust_line(breakdown) == ""


def test_returns_empty_when_all_contributions_are_zero():
    """A non-default profile applied to a signal-free candidate still
    produces no rendered line — every component is zero, so there is
    nothing to show.
    """

    # Empty candidate -> only unknown_signals component might fire under
    # profiles with unknown_penalty. exploratory has unknown_penalty=0.
    breakdown = _breakdown_for({}, "exploratory")
    assert format_trust_line(breakdown) == ""


# ---------------------------------------------------------------------------
# Rendering quality — shape of the produced line
# ---------------------------------------------------------------------------


def test_line_includes_profile_id_and_signed_total():
    candidate = {
        "hadith_grade": "sahih",
        "isnad_strength": 0.85,
        "source_type": "hadith",
        "era": "classical",
    }
    breakdown = _breakdown_for(candidate, "hadith_focused")
    line = format_trust_line(breakdown)
    assert line.startswith("Trust [hadith_focused]:")
    assert " = +" in line or " = -" in line  # signed total
    assert "sahih" in line
    assert "isnad" in line


def test_line_accepts_json_string_form():
    """The pipeline serializes the breakdown to JSON for snippet transport;
    the renderer must accept the string form too.
    """

    candidate = {"hadith_grade": "sahih", "source_type": "hadith"}
    breakdown = _breakdown_for(candidate, "hadith_focused")
    line_from_dict = format_trust_line(breakdown)
    line_from_json = format_trust_line(json.dumps(breakdown))
    assert line_from_dict == line_from_json
    assert line_from_dict  # non-empty


def test_line_caps_components_at_four_most_impactful():
    """A candidate with many contributing signals must produce a compact
    line — only the top-four most-impactful contributions show, by
    absolute value.
    """

    candidate = {
        "hadith_grade": "sahih",
        "isnad_strength": 0.95,
        "corroboration_count": 5,
        "ijma_strength": "ijma",
        "era": "classical",
        "source_type": "hadith",
        "scholar_authority": 0.9,
        "madhhab": "shafii",
        "methodology_tags": ["rigorous_isnad"],
    }
    breakdown = _breakdown_for(candidate, "strict_classical")
    line = format_trust_line(breakdown)
    # Count fragments between profile prefix and the `=` suffix
    assert line.count(",") <= 3, f"expected at most 4 fragments, got: {line!r}"


def test_signed_fragments_show_penalties_with_minus():
    """When a profile penalizes a signal, that contribution renders with a
    leading minus so users can see the downweighting.
    """

    candidate = {"hadith_grade": "daif", "source_type": "hadith"}
    breakdown = _breakdown_for(candidate, "hadith_focused")
    line = format_trust_line(breakdown)
    assert "-" in line  # negative contribution rendered
    assert "daif" in line


def test_unknown_penalty_is_labeled_with_signal_count():
    """When the unknown_signals component fires, it shows how many signals
    were missing — useful feedback that richer metadata would help.
    """

    candidate = {"hadith_grade": "sahih", "source_type": "hadith"}
    breakdown = _breakdown_for(candidate, "strict_classical")
    line = format_trust_line(breakdown)
    # strict_classical has unknown_penalty > 0 and the candidate has many unknowns
    assert "unknown_signals" in line
