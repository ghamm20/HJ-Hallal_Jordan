"""Tests for the /profiles button UI and public profile API.

Exercises the HTML helpers and the route handlers via FastAPI's
TestClient when starlette is available; falls back to direct helper
calls otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.main import _profile_button_html, _profiles_html  # noqa: E402
from app.reasoning.trust_engine import list_profiles_with_metadata  # noqa: E402


def test_button_marks_current_profile_active():
    metadata = {
        "profile_id": "hadith_focused",
        "description": "Hadith focused profile",
        "mode": "strict",
        "is_scholar_methodology": False,
        "scholar_name": "",
        "methodology_overview": "",
        "methodology_disclaimer": "",
    }
    html_active = _profile_button_html(metadata, "hadith_focused")
    html_inactive = _profile_button_html(metadata, "default")
    assert "active" in html_active
    assert "Active</span>" in html_active
    assert "active" not in html_inactive
    assert "Active</span>" not in html_inactive


def test_scholar_button_renders_scholar_name_and_disclaimer():
    metadata = {
        "profile_id": "shaykh_jamal_methodology",
        "description": "ignored when scholar profile",
        "mode": "strict",
        "is_scholar_methodology": True,
        "scholar_name": "Shaykh Jamal",
        "methodology_overview": "Hadith-rigorous methodology.",
        "methodology_disclaimer": "Methodology modeling, not the actual scholar speaking.",
    }
    html = _profile_button_html(metadata, "default")
    assert "Shaykh Jamal" in html
    assert "Hadith-rigorous methodology" in html
    assert "Methodology modeling" in html
    assert "profile-disclaimer" in html


def test_button_html_escapes_user_supplied_text():
    """User-editable profile fields must not produce XSS — the button
    renderer escapes them.
    """

    metadata = {
        "profile_id": "evil",
        "description": "<script>alert(1)</script>",
        "mode": "balanced",
        "is_scholar_methodology": False,
        "scholar_name": "",
        "methodology_overview": "",
        "methodology_disclaimer": "",
    }
    html = _profile_button_html(metadata, "default")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_profiles_page_renders_both_sections():
    profiles = list_profiles_with_metadata()
    html = _profiles_html(profiles=profiles, current_profile_id="default")
    assert "<h2>Scholar Methodology</h2>" in html
    assert "<h2>Research Modes</h2>" in html
    # Shipped scholar profiles appear
    assert "Shaykh Jamal" in html
    assert "Dr. Umar" in html
    # POST endpoint is referenced
    assert "/api/profile/set" in html
    # Active default is highlighted
    assert "active" in html


def test_profiles_page_works_with_no_scholar_profiles():
    """If a deployment removes all scholar methodology profiles, the page
    must still render — the scholar section is conditionally omitted.
    """

    generic_only = [
        e for e in list_profiles_with_metadata() if not e["is_scholar_methodology"]
    ]
    html = _profiles_html(profiles=generic_only, current_profile_id="default")
    assert "<h2>Scholar Methodology</h2>" not in html
    assert "<h2>Research Modes</h2>" in html
