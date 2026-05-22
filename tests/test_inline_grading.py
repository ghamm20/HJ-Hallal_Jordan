"""Tests for the inline hadith-grading parser.

Pins the charter rule that code never invents authenticity:
  - Conflicting grades within one chunk -> empty (silent)
  - Casual mentions ("this is a sahih book") -> empty
  - Only fires on parenthetical / bracketed / attributed forms
  - Existing explicit grades are never overridden
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.inline_grading import (  # noqa: E402
    apply_inline_grading,
    parse_inline_grade,
)


# ---------------------------------------------------------------------------
# Parser — happy paths
# ---------------------------------------------------------------------------


def test_parenthetical_sahih():
    assert parse_inline_grade("Narrated Abu Hurayrah... (Sahih)") == "sahih"


def test_parenthetical_hasan():
    assert parse_inline_grade("Long hadith body. (Hasan)") == "hasan"


def test_parenthetical_daif():
    assert parse_inline_grade("Long hadith body. (Da'if)") == "daif"
    assert parse_inline_grade("Long hadith body. (Weak)") == "daif"


def test_bracketed_sahih():
    assert parse_inline_grade("Long hadith text [Sahih]") == "sahih"


def test_attributed_albani_sahih():
    assert parse_inline_grade("Long hadith. Al-Albani: Sahih") == "sahih"
    assert parse_inline_grade("Long hadith. Albani: Hasan") == "hasan"


def test_attributed_graded_by():
    assert parse_inline_grade("Long hadith body. Graded sahih by al-Albani.") == "sahih"
    assert parse_inline_grade("Body. Graded weak by Shaykh X.") == "daif"


def test_tirmidhi_own_grade_line():
    assert parse_inline_grade("Narration...\nHasan Sahih\n") == "sahih"
    assert parse_inline_grade("Narration...\nHasan Gharib.\n") == "hasan"
    assert parse_inline_grade("Narration...\nSahih.\n") == "sahih"


# ---------------------------------------------------------------------------
# Charter rule: never fabricate, never inflate
# ---------------------------------------------------------------------------


def test_casual_mention_does_not_trigger():
    """A casual mention like 'this book is sahih' must NOT register as a
    grade — it isn't a structured annotation. The parser only fires on
    parenthetical, bracketed, or attributed forms.
    """

    assert parse_inline_grade("This is a sahih book and a great read.") == ""
    assert parse_inline_grade("The chain of narrators is hasan in nature.") == ""
    assert parse_inline_grade("Many hadith here are weak according to most scholars.") == ""


def test_conflicting_authentic_and_weak_returns_empty():
    """A chunk that contains both (Sahih) and (Da'if) markers is
    ambiguous — we must NOT arbitrate. Return empty so the chunk falls
    back to whatever the collection prior says (or nothing).
    """

    text = "First narration. (Sahih) Second narration. (Da'if)"
    assert parse_inline_grade(text) == ""


def test_multiple_authentic_grades_picks_conservative():
    """Sahih + Hasan both present (both acceptable, not in conflict):
    pick the lower (more conservative) one.
    """

    text = "First. (Sahih) Second. (Hasan)"
    assert parse_inline_grade(text) == "hasan"


def test_multiple_weak_grades_picks_more_severe():
    text = "First. (Da'if) Second. (Fabricated)"
    assert parse_inline_grade(text) == "mawdu"


def test_empty_or_too_short_returns_empty():
    assert parse_inline_grade("") == ""
    assert parse_inline_grade("(Sahih)") == ""  # too short to be meaningful


# ---------------------------------------------------------------------------
# apply_inline_grading — chunk-level behavior
# ---------------------------------------------------------------------------


def test_apply_does_not_touch_non_hadith_sources():
    """Don't scan fiqh manuals or tasawwuf text for hadith grades —
    'sahih' may appear in those discussions referentially, not as
    authentic markers.
    """

    chunk = {
        "source_classification": "fiqh_manual",
        "text": "The hadith on this matter is graded sahih by al-Albani.",
    }
    out = apply_inline_grading(chunk)
    assert "hadith_grade" not in out


def test_apply_does_not_override_existing_grade():
    chunk = {
        "source_classification": "hadith",
        "text": "Narration body. (Sahih)",
        "hadith_grade": "daif",  # explicit, must win
    }
    out = apply_inline_grading(chunk)
    assert out["hadith_grade"] == "daif"


def test_apply_sets_grade_when_unset_and_text_has_pattern():
    chunk = {
        "source_classification": "hadith",
        "text": "Narration body, fairly long. (Sahih) - end.",
    }
    out = apply_inline_grading(chunk)
    assert out["hadith_grade"] == "sahih"
    assert out["hadith_grade_source"] == "inline_text"


def test_apply_silent_when_no_pattern_present():
    chunk = {
        "source_classification": "hadith",
        "text": "Just a hadith body with no grading annotation present.",
    }
    out = apply_inline_grading(chunk)
    assert "hadith_grade" not in out


def test_apply_silent_on_conflict_even_when_unset():
    chunk = {
        "source_classification": "hadith",
        "text": "Narration A. (Sahih) Narration B. (Weak)",
    }
    out = apply_inline_grading(chunk)
    assert "hadith_grade" not in out  # silent on conflict
