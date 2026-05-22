"""Regression test for the flight-test bug: profile set must propagate
to the next /api/ask call.

The bug: /api/profile/set writes through ops.config_store, but the
public _public_ask helper was constructing a new RetrievalPipeline
that loaded config from a separate path. The profile change was
invisible to the next ask.

This test fails if the fix regresses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.backend.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_profile_set_then_ask_uses_new_profile(client):
    """Set a non-default profile, then immediately ask. The answer
    payload must report the profile we just set, not 'default'.
    """

    # Reset to a clean baseline
    client.post("/api/profile/set", json={"profile_id": "default"})

    # Set a non-default profile
    r = client.post("/api/profile/set", json={"profile_id": "hadith_focused"})
    assert r.status_code == 200
    assert r.json()["profile_id"] == "hadith_focused"

    # Ask a question — the answer must come back with hadith_focused active
    r = client.post("/api/ask", json={"question": "intentions sincerity"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["profile_id"] == "hadith_focused", (
        f"profile change did not propagate — answer reports {payload['profile_id']!r}, "
        f"expected 'hadith_focused'. The flight-test bug regressed."
    )

    # Reset for other tests
    client.post("/api/profile/set", json={"profile_id": "default"})


def test_profile_change_visible_in_subsequent_ask(client):
    """Set profile A, ask. Set profile B, ask. Each ask must reflect
    the most-recent profile set.
    """

    client.post("/api/profile/set", json={"profile_id": "shaykh_jamal_methodology"})
    r = client.post("/api/ask", json={"question": "purification"})
    assert r.json()["profile_id"] == "shaykh_jamal_methodology"

    client.post("/api/profile/set", json={"profile_id": "dr_umar_methodology"})
    r = client.post("/api/ask", json={"question": "purification"})
    assert r.json()["profile_id"] == "dr_umar_methodology"

    client.post("/api/profile/set", json={"profile_id": "default"})
