"""Shared duplicate-family heuristics for retrieval and audit tooling."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

FAMILY_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "quran": (
        ("english_rwwad", r"\benglish rwwad\b"),
    ),
    "fiqh_manual": (
        ("mukhtasar_al_quduri", r"\bquduri\b"),
        ("minhaj_al_talibin", r"\bminhaj(?:\s+al)?(?:\s+talibin)?\b"),
        ("al_majmu", r"\bmajmu\b|\bal[-\s]?majmu\b"),
        ("mukhtasar_khalil", r"\bkhalil\b|\bmukhtasar(?:\s+of)?\s+khalil\b"),
        ("zad_al_mustaqni", r"\bzad(?:\s+al)?(?:\s+mustaqni)?\b"),
        ("al_mughni", r"\bmughni\b|\bal[-\s]?mughni\b"),
    ),
    "hadith": (
        ("sahih_muslim", r"\bsahih muslim\b"),
        ("sahih_al_bukhari", r"\bsahih al bukhari\b|\bsahih bukhari\b"),
        ("sunan_abu_dawud", r"\bsunan abu dawud\b|\babu dawud\b"),
        ("jami_at_tirmidhi", r"\bjami at tirmidhi\b|\btirmidhi\b"),
        ("sunan_an_nasai", r"\bsunan an nasai\b|\bnasai\b"),
        ("sunan_ibn_majah", r"\bsunan ibn majah\b|\bibn majah\b"),
        ("muwatta_imam_malik", r"\bmuwatta\b"),
    ),
}


def normalize_title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\(\d+\)$", "", normalized).strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def build_duplicate_family_key(
    *,
    source_type: str,
    title: str,
    source_path: str,
    collection: str = "",
) -> str:
    normalized_source_type = source_type.strip().lower()
    if not normalized_source_type:
        return ""

    stem_key = normalize_title_key(Path(source_path).stem)
    normalized_blob = " ".join(
        part
        for part in (
            normalize_title_key(title),
            normalize_title_key(collection),
            stem_key,
        )
        if part
    )
    for family_id, pattern in FAMILY_ALIASES.get(normalized_source_type, ()):
        if re.search(pattern, normalized_blob):
            return f"{normalized_source_type}|{family_id}"

    fallback_key = normalize_title_key(title) or stem_key
    if not fallback_key:
        return ""
    return f"{normalized_source_type}|{fallback_key}"
