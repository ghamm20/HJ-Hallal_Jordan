"""Tests for trust diagnostics in RetrievalDebugResult.

These exercise the private helper directly so the tests do not need a
fully bootstrapped corpus/index — the helper consumes the same shape
the reranker produces (candidates with ``_trust_breakdown`` attached).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.reasoning.trust_engine import load_profile, score  # noqa: E402
from app.retrieval.pipeline import _trust_diagnostics  # noqa: E402


def _candidate_with_breakdown(metadata: dict, profile_id: str) -> dict:
    profile = load_profile(profile_id)
    breakdown = score(metadata, profile=profile).to_dict()
    candidate = dict(metadata)
    candidate["_trust_breakdown"] = breakdown
    return candidate


def test_diagnostics_empty_when_no_candidates():
    out = _trust_diagnostics(reranked=[], trust_profile_id="hadith_focused")
    assert out["candidates_scored"] == 0
    assert out["candidates_with_nonzero_bonus"] == 0
    assert out["top_candidate_breakdown"] is None


def test_default_profile_diagnostics_marked_inactive():
    candidates = [
        _candidate_with_breakdown(
            {"hadith_grade": "sahih", "source_type": "hadith"},
            "default",
        )
    ]
    out = _trust_diagnostics(reranked=candidates, trust_profile_id="default")
    assert out["active"] is False
    assert out["profile_id"] == "default"
    # default profile contributes zero, so zero candidates have nonzero bonus
    assert out["candidates_with_nonzero_bonus"] == 0


def test_active_profile_reports_nonzero_bonuses():
    candidates = [
        _candidate_with_breakdown(
            {"hadith_grade": "sahih", "source_type": "hadith"},
            "hadith_focused",
        ),
        _candidate_with_breakdown(
            {"hadith_grade": "daif", "source_type": "hadith"},
            "hadith_focused",
        ),
        _candidate_with_breakdown({}, "hadith_focused"),
    ]
    out = _trust_diagnostics(reranked=candidates, trust_profile_id="hadith_focused")
    assert out["active"] is True
    assert out["candidates_scored"] == 3
    assert out["candidates_with_nonzero_bonus"] >= 2  # sahih and daif both move


def test_signal_coverage_reports_fraction_per_signal():
    candidates = [
        _candidate_with_breakdown(
            {"hadith_grade": "sahih", "madhhab": "hanafi", "source_type": "hadith"},
            "hanafi_heavy",
        ),
        _candidate_with_breakdown(
            {"madhhab": "shafii", "source_type": "fiqh_manual"},
            "hanafi_heavy",
        ),
        _candidate_with_breakdown({}, "hanafi_heavy"),
    ]
    out = _trust_diagnostics(reranked=candidates, trust_profile_id="hanafi_heavy")
    coverage = out["signal_coverage"]
    # madhhab present on 2 of 3 candidates
    assert coverage.get("madhhab") == round(2 / 3, 3)
    # authenticity_grade present on 1 of 3
    assert coverage.get("authenticity_grade") == round(1 / 3, 3)


def test_top_candidate_breakdown_is_full_dict():
    candidates = [
        _candidate_with_breakdown(
            {"hadith_grade": "sahih", "source_type": "hadith"},
            "hadith_focused",
        )
    ]
    out = _trust_diagnostics(reranked=candidates, trust_profile_id="hadith_focused")
    top = out["top_candidate_breakdown"]
    assert isinstance(top, dict)
    assert top["profile_id"] == "hadith_focused"
    assert "components" in top
    assert "signals" in top
