"""Tests for the Weight and Trust Engine.

These tests pin the charter's non-negotiable rules:

1. Unknown signals never inflate scores. A candidate with no metadata
   scores zero (or negative under unknown_penalty) — never positive — no
   matter how aggressive the profile.
2. Every contribution is visible in the breakdown. The sum of component
   contributions equals the total.
3. Profile switches actually change outcomes for the same candidate.
4. Default profile contributes exactly zero — wiring is safe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.reasoning.trust_engine import (  # noqa: E402
    TrustProfile,
    TrustSignals,
    extract_signals,
    list_profiles,
    load_profile,
    score,
)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def test_empty_candidate_yields_all_unknown_signals():
    signals = extract_signals({})
    assert signals.authenticity_grade == "unknown"
    assert signals.isnad_strength is None
    assert signals.corroboration_count == 0
    assert signals.ijma_strength == "unknown"
    assert signals.era == "unknown"
    assert signals.source_distance is None
    assert signals.scholar_authority is None
    assert signals.madhhab == ""
    assert signals.methodology_tags == ()


def test_authenticity_alias_normalization():
    assert extract_signals({"hadith_grade": "Saheeh"}).authenticity_grade == "sahih"
    assert extract_signals({"authenticity": "WEAK"}).authenticity_grade == "daif"
    assert extract_signals({"grading": "fabricated"}).authenticity_grade == "mawdu"
    assert extract_signals({"grading": "unrecognized"}).authenticity_grade == "unknown"


def test_isnad_strength_clamped_to_unit_interval():
    assert extract_signals({"isnad_strength": 1.5}).isnad_strength == 1.0
    assert extract_signals({"isnad_strength": -0.2}).isnad_strength == 0.0
    assert extract_signals({"isnad_strength": 0.7}).isnad_strength == 0.7
    assert extract_signals({"isnad_strength": "not a number"}).isnad_strength is None


def test_source_distance_inferred_from_classification():
    assert extract_signals({"source_type": "quran"}).source_distance == 0
    assert extract_signals({"source_classification": "fiqh_manual"}).source_distance == 1
    assert extract_signals({"source_classification": "fatwa"}).source_distance == 2
    assert extract_signals({}).source_distance is None
    # Explicit overrides take precedence
    assert extract_signals({"source_type": "fiqh_manual", "source_distance": 3}).source_distance == 3


def test_methodology_tags_accept_string_or_list():
    from_string = extract_signals({"methodology_tags": "rigorous_isnad, literalist"})
    from_list = extract_signals({"methodology_tags": ["Rigorous_Isnad", "literalist"]})
    assert from_string.methodology_tags == ("rigorous_isnad", "literalist")
    assert from_list.methodology_tags == ("rigorous_isnad", "literalist")


# ---------------------------------------------------------------------------
# Charter rule: unknown signals never inflate
# ---------------------------------------------------------------------------


def test_empty_candidate_never_scores_positive_under_any_profile():
    """The charter rule operationalized.

    For every shipped profile, an empty/unknown candidate must score <= 0.
    No combination of weights may turn ignorance into a positive bonus.
    """

    empty_candidate: dict = {}
    for profile_id in list_profiles():
        profile = load_profile(profile_id)
        breakdown = score(empty_candidate, profile=profile)
        assert breakdown.total <= 0.0, (
            f"profile {profile_id!r} scored an empty candidate at "
            f"{breakdown.total} — unknowns must never inflate"
        )


def test_partial_metadata_only_contributes_known_signals():
    profile = load_profile("hadith_focused")
    candidate = {"hadith_grade": "sahih"}  # everything else unknown
    breakdown = score(candidate, profile=profile)

    contributing_signals = {c.signal for c in breakdown.components if c.contribution > 0}
    assert "authenticity_grade" in contributing_signals
    # No fabricated signals
    assert "isnad_strength" not in contributing_signals
    assert "ijma_strength" not in contributing_signals
    assert "era" not in contributing_signals


# ---------------------------------------------------------------------------
# Breakdown transparency
# ---------------------------------------------------------------------------


def test_components_sum_to_total():
    profile = load_profile("strict_classical")
    candidate = {
        "hadith_grade": "sahih",
        "isnad_strength": 0.85,
        "corroboration_count": 3,
        "ijma_strength": "majority",
        "era": "classical",
        "source_type": "hadith",
        "scholar_authority": 0.7,
        "madhhab": "shafii",
        "methodology_tags": ["rigorous_isnad"],
    }
    breakdown = score(candidate, profile=profile)
    component_sum = sum(c.contribution for c in breakdown.components)
    assert breakdown.total == pytest.approx(component_sum, abs=1e-9)


def test_breakdown_records_unknowns_list():
    profile = load_profile("hadith_focused")
    candidate = {"hadith_grade": "sahih"}
    breakdown = score(candidate, profile=profile)
    assert "isnad_strength" in breakdown.unknowns
    assert "ijma_strength" in breakdown.unknowns
    # The signal we DID provide should not appear in unknowns
    assert "authenticity_grade" not in breakdown.unknowns


def test_breakdown_to_dict_is_json_serializable():
    profile = load_profile("hanafi_heavy")
    candidate = {"madhhab": "hanafi", "source_type": "fiqh_manual", "era": "classical"}
    breakdown = score(candidate, profile=profile)
    payload = breakdown.to_dict()
    # Round-trips through JSON without errors — important for citation rendering
    serialized = json.dumps(payload)
    revived = json.loads(serialized)
    assert revived["profile_id"] == "hanafi_heavy"
    assert revived["total"] == pytest.approx(breakdown.total)


# ---------------------------------------------------------------------------
# Default profile is safe
# ---------------------------------------------------------------------------


def test_default_profile_contributes_exactly_zero():
    """Wiring guarantee: adding the trust engine with the default profile
    must not perturb the reranker.
    """

    profile = load_profile("default")
    rich_candidate = {
        "hadith_grade": "sahih",
        "isnad_strength": 0.9,
        "corroboration_count": 5,
        "ijma_strength": "ijma",
        "era": "classical",
        "source_type": "hadith",
        "scholar_authority": 0.9,
        "madhhab": "hanafi",
        "methodology_tags": ["rigorous_isnad", "hanafi_usul"],
    }
    breakdown = score(rich_candidate, profile=profile)
    assert breakdown.total == 0.0
    assert all(c.contribution == 0.0 for c in breakdown.components)


# ---------------------------------------------------------------------------
# Profile sensitivity
# ---------------------------------------------------------------------------


def test_strict_profile_penalizes_modern_isolated_more_than_exploratory():
    candidate = {
        "hadith_grade": "daif",
        "ijma_strength": "isolated",
        "era": "modern",
        "source_type": "fatwa",
    }
    strict = score(candidate, profile=load_profile("strict_classical"))
    exploratory = score(candidate, profile=load_profile("exploratory"))
    assert strict.total < exploratory.total, (
        "strict_classical must penalize a daif, isolated, modern fatwa more "
        "heavily than the exploratory profile does"
    )


def test_hanafi_heavy_prefers_hanafi_over_shafii_for_same_candidate():
    hanafi_candidate = {"madhhab": "hanafi", "source_type": "fiqh_manual", "era": "classical"}
    shafii_candidate = {"madhhab": "shafii", "source_type": "fiqh_manual", "era": "classical"}
    profile = load_profile("hanafi_heavy")
    hanafi_score = score(hanafi_candidate, profile=profile).total
    shafii_score = score(shafii_candidate, profile=profile).total
    assert hanafi_score > shafii_score


def test_hadith_focused_rewards_sahih_and_punishes_mawdu_heavily():
    profile = load_profile("hadith_focused")
    sahih = score({"hadith_grade": "sahih", "source_type": "hadith"}, profile=profile)
    mawdu = score({"hadith_grade": "mawdu", "source_type": "hadith"}, profile=profile)
    assert sahih.total > 0
    assert mawdu.total < -1.0  # Fabricated narrations get a crushing penalty


def test_strictness_threshold_flag_set_when_total_below():
    profile = load_profile("strict_classical")
    weak = score({"hadith_grade": "daif", "era": "contemporary"}, profile=profile)
    assert weak.below_strictness_threshold is True

    strong = score(
        {
            "hadith_grade": "sahih",
            "isnad_strength": 0.9,
            "era": "classical",
            "ijma_strength": "ijma",
            "source_type": "hadith",
        },
        profile=profile,
    )
    assert strong.below_strictness_threshold is False


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


def test_shipped_profiles_all_load():
    expected_ids = {
        "default",
        "hadith_focused",
        "hanafi_heavy",
        "strict_classical",
        "exploratory",
    }
    assert expected_ids.issubset(set(list_profiles()))
    for profile_id in expected_ids:
        profile = load_profile(profile_id)
        assert isinstance(profile, TrustProfile)
        assert profile.profile_id == profile_id
        assert profile.description, f"{profile_id} missing description (charter requires WHY)"


def test_unknown_profile_id_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("does_not_exist_12345")


def test_default_falls_back_to_neutral_when_file_missing(tmp_path):
    # Simulate a fresh checkout that hasn't shipped profiles yet
    profile = load_profile("default", repo_root=tmp_path)
    assert profile.profile_id == "default"
    # Neutral profile means an empty candidate scores exactly zero
    assert score({}, profile=profile).total == 0.0


# ---------------------------------------------------------------------------
# TrustSignals helpers
# ---------------------------------------------------------------------------


def test_known_and_unknown_signals_are_disjoint_and_complete():
    signals = TrustSignals(
        authenticity_grade="sahih",
        madhhab="hanafi",
    )
    known = set(signals.known_signals())
    unknown = set(signals.unknown_signals())
    assert known.isdisjoint(unknown)
    assert known | unknown == {
        "authenticity_grade",
        "isnad_strength",
        "corroboration_count",
        "ijma_strength",
        "era",
        "source_distance",
        "scholar_authority",
        "madhhab",
        "methodology_tags",
    }
