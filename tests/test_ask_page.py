"""Tests for the public /ask page and /api/ask endpoint.

These exercise the simple front door directly via FastAPI's TestClient.
Tests are tolerant of corpora that take time to bootstrap — they only
verify shape, not specific retrieval results.
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


# ---------------------------------------------------------------------------
# /ask HTML page
# ---------------------------------------------------------------------------


def test_ask_page_renders(client):
    response = client.get("/ask")
    assert response.status_code == 200
    body = response.text
    assert "Halal Jordan" in body
    assert "Your question" in body
    assert "/api/ask" in body  # the JS posts here
    assert "/profiles" in body  # profile chip links to selector


def test_ask_page_shows_current_profile_chip(client):
    response = client.get("/ask")
    assert "Active profile:" in response.text


def test_ask_page_is_self_contained(client):
    """Charter rule: works in any environment, including offline. The
    page must not pull external CSS or JS frameworks.
    """

    body = client.get("/ask").text
    # No external CDN references
    assert "cdn." not in body.lower()
    assert "googleapis" not in body
    assert "unpkg" not in body
    # No framework hints
    assert "react" not in body.lower()
    assert "vue.global" not in body


# ---------------------------------------------------------------------------
# /api/ask endpoint
# ---------------------------------------------------------------------------


def test_api_ask_rejects_empty_question(client):
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 400


def test_api_ask_rejects_missing_question(client):
    response = client.post("/api/ask", json={})
    assert response.status_code == 400


def test_api_ask_returns_rendered_text_and_structure(client):
    response = client.post(
        "/api/ask",
        json={"question": "intentions sincerity", "selected_madhhab": "not_specified"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "rendered_text" in payload
    assert "profile_id" in payload
    assert "evidence_ladder" in payload
    assert "confidence" in payload or payload.get("confidence") is None
    assert "source_count" in payload


def test_api_ask_evidence_ladder_in_canonical_order(client):
    response = client.post(
        "/api/ask",
        json={"question": "Quran ayah hadith intentions"},
    )
    payload = response.json()
    ladder = payload.get("evidence_ladder", [])
    # populated_tiers preserves ladder order — ranks must be ascending
    ranks = [tier["rank"] for tier in ladder]
    assert ranks == sorted(ranks), f"ladder out of order: {ranks}"


def test_api_ask_does_not_require_auth(client):
    """No cookies, no headers — the endpoint must work for the average
    user out of the box.
    """

    # TestClient resets between calls; this call carries no auth.
    response = client.post(
        "/api/ask",
        json={"question": "purification wudu"},
    )
    assert response.status_code == 200
