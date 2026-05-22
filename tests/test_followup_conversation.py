"""Tests for the follow-up conversation layer on /ask.

The page maintains a list of prior questions and sends them as
conversation_context on follow-ups. The backend uses them as a
retrieval-time topic bias only — never fabricates dialogue continuity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.main import _build_retrieval_query  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.backend.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Retrieval query composer — current question must dominate
# ---------------------------------------------------------------------------


def test_no_context_returns_question_unchanged():
    assert _build_retrieval_query("about wudu", []) == "about wudu"
    assert _build_retrieval_query("about wudu", None) == "about wudu"


def test_context_adds_topic_words_after_question():
    out = _build_retrieval_query(
        "what breaks it?",
        ["wudu and ritual purity"],
    )
    # Current question dominates: it appears first
    assert out.startswith("what breaks it?")
    # Topic words appended in parens for soft biasing
    assert "wudu" in out
    assert "ritual" in out
    assert "purity" in out


def test_context_word_count_capped():
    """A long prior question must not flood the retrieval query.
    Cap at 12 topic words to keep the current question dominant.
    """

    long_context = " ".join(["alpha{}beta{}".format(i, i) for i in range(50)])
    out = _build_retrieval_query("focused question", [long_context])
    word_count_in_paren = out.count("alpha")
    assert word_count_in_paren <= 12


def test_short_words_filtered_out():
    """Words shorter than 4 chars are uninformative and dropped."""

    out = _build_retrieval_query("X", ["a an the of to is in"])
    # No topic_words means no parenthetical addition
    assert out == "X"


def test_duplicates_collapsed():
    out = _build_retrieval_query("X", ["wudu wudu wudu purification"])
    # wudu appears at most once
    assert out.count("wudu") == 1


# ---------------------------------------------------------------------------
# Endpoint — accepts and reports conversation_context
# ---------------------------------------------------------------------------


def test_endpoint_accepts_conversation_context(client):
    response = client.post(
        "/api/ask",
        json={
            "question": "follow up",
            "conversation_context": ["about wudu", "what breaks it"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_context_used"] == ["about wudu", "what breaks it"]
    # The composed retrieval query carries topic words from prior turns
    assert "wudu" in payload["retrieval_question"].lower()


def test_endpoint_ignores_garbage_context(client):
    """Non-list context is ignored rather than causing an error."""

    response = client.post(
        "/api/ask",
        json={"question": "wudu", "conversation_context": "not a list"},
    )
    assert response.status_code == 200
    assert response.json()["conversation_context_used"] == []


def test_endpoint_caps_context_at_five(client):
    response = client.post(
        "/api/ask",
        json={
            "question": "follow up",
            "conversation_context": [f"prior question {i}" for i in range(10)],
        },
    )
    assert response.status_code == 200
    used = response.json()["conversation_context_used"]
    assert len(used) <= 5
    # Most recent 5 kept
    assert used[-1] == "prior question 9"


# ---------------------------------------------------------------------------
# Page renders the multi-turn UI
# ---------------------------------------------------------------------------


def test_page_renders_clear_conversation_button(client):
    body = client.get("/").text
    assert 'id="clear-btn"' in body
    assert "Clear conversation" in body


def test_page_renders_turns_container(client):
    body = client.get("/").text
    assert 'id="turns"' in body
    # Placeholder turn exists for first-time visitors
    assert 'id="placeholder"' in body


def test_page_js_sends_conversation_context(client):
    body = client.get("/").text
    assert "conversation_context" in body
    assert "conversation.push" in body
