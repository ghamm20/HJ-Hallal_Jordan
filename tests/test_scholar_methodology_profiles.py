"""Tests for Scholar Methodology Profiles (charter's Scholar Embodiment Layer).

Pins the non-negotiable charter rule for these profiles: every scholar
methodology profile MUST carry a disclosure that it is methodology
modeling, not the actual scholar speaking. The loader refuses to load
without one; the breakdown carries it through; the renderer surfaces it.
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
    _profile_from_payload,
    list_profiles,
    list_profiles_with_metadata,
    load_profile,
    score,
)


SHIPPED_SCHOLAR_PROFILE_IDS = ("shaykh_jamal_methodology", "dr_umar_methodology")


# ---------------------------------------------------------------------------
# Loader: charter disclosure rule
# ---------------------------------------------------------------------------


def test_loader_rejects_scholar_methodology_without_disclaimer():
    """The charter forbids loading a scholar methodology profile without
    an explicit disclosure. The loader must enforce this structurally —
    not as a best-effort warning.
    """

    payload = {
        "profile_id": "test_missing_disclaimer",
        "description": "test",
        "mode": "balanced",
        "is_scholar_methodology": True,
        "scholar_name": "Test Scholar",
        "methodology_disclaimer": "",
    }
    with pytest.raises(ValueError, match="methodology_disclaimer"):
        _profile_from_payload(payload)


def test_loader_accepts_scholar_methodology_with_disclaimer():
    payload = {
        "profile_id": "test_with_disclaimer",
        "description": "test",
        "mode": "balanced",
        "is_scholar_methodology": True,
        "scholar_name": "Test Scholar",
        "methodology_disclaimer": "Methodology modeling, not the actual scholar speaking.",
    }
    profile = _profile_from_payload(payload)
    assert profile.is_scholar_methodology is True
    assert profile.scholar_name == "Test Scholar"
    assert "methodology modeling" in profile.methodology_disclaimer.lower()


def test_non_scholar_profile_does_not_require_disclaimer():
    payload = {
        "profile_id": "test_generic",
        "description": "generic profile",
        "mode": "balanced",
    }
    profile = _profile_from_payload(payload)
    assert profile.is_scholar_methodology is False
    assert profile.methodology_disclaimer == ""


# ---------------------------------------------------------------------------
# Shipped profiles
# ---------------------------------------------------------------------------


def test_shipped_scholar_methodology_profiles_load():
    available = set(list_profiles())
    for profile_id in SHIPPED_SCHOLAR_PROFILE_IDS:
        assert profile_id in available, f"missing shipped scholar profile: {profile_id}"
        profile = load_profile(profile_id)
        assert profile.is_scholar_methodology is True
        assert profile.scholar_name, f"{profile_id} missing scholar_name"
        assert profile.methodology_disclaimer, f"{profile_id} missing methodology_disclaimer"
        # Charter: disclaimer must contain the methodology-modeling framing
        assert "methodology modeling" in profile.methodology_disclaimer.lower()


# ---------------------------------------------------------------------------
# Metadata listing for the button UI
# ---------------------------------------------------------------------------


def test_list_profiles_with_metadata_returns_expected_fields():
    entries = list_profiles_with_metadata()
    assert len(entries) >= 7  # default + 4 generic + 2 scholar minimum
    for entry in entries:
        assert "profile_id" in entry
        assert "description" in entry
        assert "mode" in entry
        assert "is_scholar_methodology" in entry
        assert "scholar_name" in entry
        assert "methodology_disclaimer" in entry


def test_scholar_profiles_carry_disclaimer_in_metadata():
    entries = {e["profile_id"]: e for e in list_profiles_with_metadata()}
    for profile_id in SHIPPED_SCHOLAR_PROFILE_IDS:
        entry = entries[profile_id]
        assert entry["is_scholar_methodology"] is True
        assert entry["methodology_disclaimer"].strip(), (
            f"{profile_id} metadata missing non-empty disclaimer"
        )


def test_generic_profiles_marked_not_scholar_methodology():
    entries = {e["profile_id"]: e for e in list_profiles_with_metadata()}
    for profile_id in ("default", "hadith_focused", "hanafi_heavy", "strict_classical", "exploratory"):
        if profile_id in entries:
            assert entries[profile_id]["is_scholar_methodology"] is False


# ---------------------------------------------------------------------------
# Breakdown carries scholar metadata through
# ---------------------------------------------------------------------------


def test_breakdown_carries_scholar_disclosure_fields():
    profile = load_profile("shaykh_jamal_methodology")
    breakdown = score({"hadith_grade": "sahih", "source_type": "hadith"}, profile=profile)
    payload = breakdown.to_dict()
    assert payload["is_scholar_methodology"] is True
    assert payload["scholar_name"] == "Shaykh Jamal"
    assert "methodology modeling" in payload["methodology_disclaimer"].lower()


def test_generic_profile_breakdown_omits_scholar_disclosure():
    profile = load_profile("hadith_focused")
    breakdown = score({"hadith_grade": "sahih", "source_type": "hadith"}, profile=profile)
    payload = breakdown.to_dict()
    assert payload["is_scholar_methodology"] is False
    assert payload["scholar_name"] == ""
    assert payload["methodology_disclaimer"] == ""


# ---------------------------------------------------------------------------
# Empty/unknown candidate never scores positive under scholar profiles
# ---------------------------------------------------------------------------


def test_scholar_profile_obeys_unknowns_rule():
    """The charter rule (unknowns never inflate) applies to scholar
    profiles too. Re-verifying here so a future profile edit can't
    accidentally break it without being caught.
    """

    for profile_id in SHIPPED_SCHOLAR_PROFILE_IDS:
        profile = load_profile(profile_id)
        breakdown = score({}, profile=profile)
        assert breakdown.total <= 0.0, (
            f"scholar profile {profile_id} scored an empty candidate at "
            f"{breakdown.total} — must never be positive"
        )
