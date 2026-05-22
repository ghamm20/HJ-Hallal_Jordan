"""Tests for the root-route reshuffle: / is now the simple /ask page,
the previous complex chat UI lives at /workspace.

Charter directive operationalized: "give me a button" — the default
front door is the one-question, one-button surface.
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


def test_root_now_serves_ask_page(client):
    """The default page is the simple ask surface, not the full
    workspace UI.
    """

    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "Halal Jordan" in body
    assert "Your question" in body  # ask-page input label
    assert "/api/ask" in body  # ask-page JS endpoint
    # And it doesn't show the workspace's complex auth shell
    assert "initialAuthView" not in body


def test_workspace_route_still_serves_full_ui(client):
    """The full chat / projects / memory UI is preserved at /workspace."""

    response = client.get("/workspace")
    assert response.status_code == 200
    body = response.text
    assert "Halal Jordan" in body
    # Workspace HTML contains the manager shell — much larger and richer
    assert len(body) > len(client.get("/").text)


def test_ask_route_still_works(client):
    """/ask remains a valid alias for the simple page."""

    response = client.get("/ask")
    assert response.status_code == 200
    assert "Your question" in response.text


def test_root_points_workspace_link_correctly(client):
    """The simple page should link to /workspace for users who want the
    full chat UI, NOT back to / (which would loop).
    """

    body = client.get("/").text
    assert "/workspace" in body


def test_profiles_page_still_works(client):
    response = client.get("/profiles")
    assert response.status_code == 200
    assert "Choose a Reasoning Profile" in response.text


def test_admin_route_still_works(client):
    response = client.get("/admin")
    assert response.status_code == 200
