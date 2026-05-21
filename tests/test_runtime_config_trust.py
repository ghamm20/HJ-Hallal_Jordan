"""Tests for trust_profile_id wiring in RuntimeConfig."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.runtime_config import RuntimeConfig  # noqa: E402


def test_default_trust_profile_id_is_default():
    config = RuntimeConfig()
    assert config.trust_profile_id == "default"


def test_each_shipped_profile_is_accepted():
    for profile_id in ("default", "hadith_focused", "hanafi_heavy", "strict_classical", "exploratory"):
        config = RuntimeConfig(trust_profile_id=profile_id)
        assert config.trust_profile_id == profile_id


def test_unknown_trust_profile_id_is_rejected():
    with pytest.raises(ValueError, match="trust_profile_id"):
        RuntimeConfig(trust_profile_id="not_a_real_profile")


def test_empty_trust_profile_id_falls_back_to_default():
    config = RuntimeConfig(trust_profile_id="")
    assert config.trust_profile_id == "default"
