"""Tests for the Tone Architecture document and prompt extension.

These are structural presence checks. The tone is text, not code — but
the document and prompt fields must be present and contain the
non-negotiable phrases from the charter. This catches accidental
deletions during prompt refactors.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tone_md_exists_at_repo_root():
    tone_path = REPO_ROOT / "TONE.md"
    assert tone_path.exists(), "TONE.md must exist as a charter companion"


def test_tone_md_names_recognized_emotional_states():
    """The five recognized states from the user's directive must be
    explicitly named in TONE.md."""

    text = (REPO_ROOT / "TONE.md").read_text(encoding="utf-8").lower()
    for state in ("lonely", "ashamed", "grieving", "confused", "spiritually exhausted"):
        assert state in text, f"TONE.md missing recognized state: {state!r}"


def test_tone_md_carries_core_principle():
    text = (REPO_ROOT / "TONE.md").read_text(encoding="utf-8")
    assert "Meet the emotional state first" in text


def test_tone_md_forbids_legalistic_openings():
    text = (REPO_ROOT / "TONE.md").read_text(encoding="utf-8").lower()
    # The rule appears as a non-negotiable item
    assert "no legalistic openings" in text


def test_prompt_references_tone_architecture():
    prompt_path = REPO_ROOT / "prompts" / "retrieval_grounded_answer.md"
    text = prompt_path.read_text(encoding="utf-8").lower()
    assert "tone architecture" in text or "tone.md" in text


def test_prompt_names_the_five_states():
    prompt_path = REPO_ROOT / "prompts" / "retrieval_grounded_answer.md"
    text = prompt_path.read_text(encoding="utf-8").lower()
    for state in ("lonely", "ashamed", "grieving", "confused", "spiritually exhausted"):
        assert state in text, f"prompt missing state: {state!r}"


def test_prompt_describes_structural_layers():
    prompt_path = REPO_ROOT / "prompts" / "retrieval_grounded_answer.md"
    text = prompt_path.read_text(encoding="utf-8")
    assert "Evidence Ladder" in text
    assert "Scholarly Confidence" in text
    assert "disagreement_map" in text


def test_prompt_forbids_inventing_disagreement_structure():
    prompt_path = REPO_ROOT / "prompts" / "retrieval_grounded_answer.md"
    text = prompt_path.read_text(encoding="utf-8").lower()
    assert "never fabricate" in text or "do not invent" in text or "leave the field" in text
