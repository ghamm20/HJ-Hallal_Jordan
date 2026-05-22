"""Parse inline hadith grading annotations from chunk text.

Sunan collections (Abu Dawud, Tirmidhi, Nasai, Ibn Majah) often carry
per-hadith authenticity gradings embedded in the text of each
translated narration. Common patterns:

  - Parenthetical:  ``(Sahih)``, ``(Hasan)``, ``(Da'if)``, ``(Weak)``
  - Bracketed:      ``[Sahih]``, ``[Hasan]``
  - Attributed:     ``Graded sahih by al-Albani``, ``Albani: Sahih``
  - Tirmidhi's own: ``Hasan Sahih``, ``Hasan Gharib``

This parser is intentionally **conservative**. It honors the charter
rule "code never invents authenticity":

  - Only fires on parenthetical, bracketed, or explicitly-attributed
    forms — never on a casual mention like "this is a sahih book".
  - When multiple conflicting grades appear in the same chunk, returns
    an empty result rather than guessing which one is authoritative.
  - When the chunk text is too short or noisy, returns empty.

The chunker calls this AFTER ``apply_collection_enrichment``, and only
attaches the parsed grade when no explicit / collection-prior grade is
already set.
"""

from __future__ import annotations

import re
from typing import Any

# Strict patterns. Each captures a normalized grade token.
# Parenthetical and bracketed grades — the most common annotation form.
_PAREN_PATTERNS = (
    (re.compile(r"\((sahih|saheeh)\)", re.IGNORECASE), "sahih"),
    (re.compile(r"\(hasan(?:\s+sahih)?\)", re.IGNORECASE), "hasan"),
    (re.compile(r"\(da'?if\)", re.IGNORECASE), "daif"),
    (re.compile(r"\(weak\)", re.IGNORECASE), "daif"),
    (re.compile(r"\(mawdu['ʻ]?\)", re.IGNORECASE), "mawdu"),
    (re.compile(r"\(fabricated\)", re.IGNORECASE), "mawdu"),
    (re.compile(r"\[(sahih|saheeh)\]", re.IGNORECASE), "sahih"),
    (re.compile(r"\[hasan(?:\s+sahih)?\]", re.IGNORECASE), "hasan"),
    (re.compile(r"\[da'?if\]", re.IGNORECASE), "daif"),
    (re.compile(r"\[weak\]", re.IGNORECASE), "daif"),
)

# Attributed gradings — "Graded sahih by al-Albani", etc.
_ATTRIBUTED_PATTERNS = (
    (re.compile(r"graded\s+(sahih|saheeh|authentic)\s+by\b", re.IGNORECASE), "sahih"),
    (re.compile(r"graded\s+hasan\s+by\b", re.IGNORECASE), "hasan"),
    (re.compile(r"graded\s+(da'?if|weak)\s+by\b", re.IGNORECASE), "daif"),
    (re.compile(r"graded\s+(mawdu|fabricated)\s+by\b", re.IGNORECASE), "mawdu"),
    (re.compile(r"al[\s-]?albani\s*[:\-]\s*(sahih|saheeh)\b", re.IGNORECASE), "sahih"),
    (re.compile(r"al[\s-]?albani\s*[:\-]\s*hasan\b", re.IGNORECASE), "hasan"),
    (re.compile(r"al[\s-]?albani\s*[:\-]\s*(da'?if|weak)\b", re.IGNORECASE), "daif"),
    (re.compile(r"\balbani\s*[:\-]\s*(sahih|saheeh)\b", re.IGNORECASE), "sahih"),
    (re.compile(r"\balbani\s*[:\-]\s*hasan\b", re.IGNORECASE), "hasan"),
    (re.compile(r"\balbani\s*[:\-]\s*(da'?if|weak)\b", re.IGNORECASE), "daif"),
)

# Tirmidhi's own grade lines — usually a standalone short line.
_TIRMIDHI_OWN_PATTERNS = (
    (re.compile(r"^\s*(this hadith is\s+)?hasan\s+sahih\s*\.?\s*$", re.IGNORECASE | re.MULTILINE), "sahih"),
    (re.compile(r"^\s*(this hadith is\s+)?(sahih|saheeh)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE), "sahih"),
    (re.compile(r"^\s*(this hadith is\s+)?hasan(\s+gharib)?\s*\.?\s*$", re.IGNORECASE | re.MULTILINE), "hasan"),
    (re.compile(r"^\s*(this hadith is\s+)?(da'?if|weak)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE), "daif"),
)

_GRADE_RANK = {"sahih": 3, "hasan": 2, "daif": 1, "mawdu": 0}


def parse_inline_grade(text: str) -> str:
    """Return the most-specific unambiguous grade found in ``text``, or
    empty string when uncertain.

    Returns one of: ``"sahih"``, ``"hasan"``, ``"daif"``, ``"mawdu"``,
    or ``""``. When conflicting grades appear (e.g. a chunk that
    contains both "(sahih)" and "(da'if)"), returns ``""`` rather than
    arbitrating between them — the charter forbids fabricating
    authenticity in ambiguous cases.
    """

    if not text or len(text) < 8:
        return ""

    found: set[str] = set()
    for pattern, grade in _PAREN_PATTERNS + _ATTRIBUTED_PATTERNS + _TIRMIDHI_OWN_PATTERNS:
        if pattern.search(text):
            found.add(grade)

    if not found:
        return ""

    if len(found) == 1:
        return next(iter(found))

    # Multiple grades present. If they're not in conflict (e.g. only
    # ``sahih`` and ``hasan`` both — both are "acceptable" grades),
    # pick the lower (more conservative) one. If sahih/hasan appear
    # alongside daif/mawdu, that's a real conflict — stay silent.
    has_authentic = bool(found & {"sahih", "hasan"})
    has_weak = bool(found & {"daif", "mawdu"})
    if has_authentic and has_weak:
        return ""  # genuine conflict
    if has_weak:
        # Multiple weak grades: pick the more severe (lower rank)
        return min(found, key=lambda g: _GRADE_RANK[g])
    # Multiple authentic grades: pick the lower one (more conservative)
    return min(found, key=lambda g: _GRADE_RANK[g])


def apply_inline_grading(chunk: dict[str, Any]) -> dict[str, Any]:
    """Set ``hadith_grade`` on a chunk from inline-text gradings, when
    the field is currently empty.

    Only acts on chunks whose source is classified as hadith — there
    is no reason to look for hadith grades in fiqh or tasawwuf text.

    Returns the (possibly modified) chunk. Never overrides existing
    grades; records ``hadith_grade_source="inline_text"`` when it does
    attach a grade, so the rendering layer can be transparent about
    provenance.
    """

    classification = (
        str(chunk.get("source_classification") or chunk.get("source_type") or "")
        .strip()
        .lower()
    )
    if classification != "hadith":
        return chunk

    existing = str(chunk.get("hadith_grade") or "").strip().lower()
    if existing:
        return chunk

    text = str(chunk.get("text") or chunk.get("normalized_search_text") or "")
    grade = parse_inline_grade(text)
    if not grade:
        return chunk

    enriched = dict(chunk)
    enriched["hadith_grade"] = grade
    enriched["hadith_grade_source"] = "inline_text"
    return enriched
