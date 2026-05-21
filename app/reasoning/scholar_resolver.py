"""Resolve named scholar references against local scholar profiles."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ScholarProfile:
    scholar_id: str
    name: str
    name_transliterated: str
    madhhab: str
    period: str
    known_works: tuple[str, ...]
    methodology_notes: str
    source_families: tuple[str, ...]
    retrieval_tags: tuple[str, ...]

    @property
    def aliases(self) -> tuple[str, ...]:
        values = [
            self.name,
            self.name_transliterated,
            *self.known_works,
            *self.retrieval_tags,
        ]
        normalized = [_normalize_key(value) for value in values if value]
        return tuple(dict.fromkeys(value for value in normalized if value))


@dataclass(slots=True)
class ScholarResolution:
    recognized: bool
    scholar_id: str
    detected_name: str
    unknown_scholar: str
    profile: ScholarProfile | None


def resolve_scholar_reference(
    question: str,
    repo_root: Path | None = None,
) -> ScholarResolution:
    normalized_question = _normalize_key(question)
    if not normalized_question:
        return ScholarResolution(False, "", "", "", None)

    profiles = load_scholar_profiles(repo_root)
    for profile in profiles.values():
        if any(alias and alias in normalized_question for alias in profile.aliases):
            return ScholarResolution(
                True,
                profile.scholar_id,
                profile.name,
                "",
                profile,
            )

    unknown = _extract_unknown_scholar_name(question)
    return ScholarResolution(False, "", "", unknown, None)


def match_source_to_scholar(
    source: Mapping[str, Any],
    profile: ScholarProfile | None,
) -> str:
    if profile is None:
        return ""
    normalized_blob = " ".join(
        part
        for part in (
            _normalize_key(str(source.get("title", "") or "")),
            _normalize_key(str(source.get("collection", "") or "")),
            _normalize_key(str(source.get("author", "") or "")),
            _normalize_key(str(source.get("source_path", "") or "")),
            _normalize_key(str(source.get("canonical_family", "") or "")),
            _normalize_key(str(source.get("commentary_target", "") or "")),
        )
        if part
    )
    if any(alias and alias in normalized_blob for alias in profile.aliases):
        return "direct"

    source_family = str(source.get("source_family", "") or "").strip().lower()
    source_classification = str(
        source.get("source_classification") or source.get("source_type") or ""
    ).strip().lower()
    source_madhhab = str(source.get("madhhab", "") or "").strip().lower()
    if (
        source_family in profile.source_families
        or (
            source_madhhab == profile.madhhab
            and source_classification in {"fiqh_manual", "commentary", "scholar_transcript"}
        )
    ):
        return "contextual"
    return ""


def load_scholar_profiles(repo_root: Path | None = None) -> dict[str, ScholarProfile]:
    root = str((repo_root or REPO_ROOT).resolve())
    return _load_scholar_profiles_cached(root)


@lru_cache(maxsize=8)
def _load_scholar_profiles_cached(repo_root: str) -> dict[str, ScholarProfile]:
    profile_dir = Path(repo_root) / "metadata" / "taxonomies" / "scholar_profiles"
    profiles: dict[str, ScholarProfile] = {}
    if not profile_dir.exists():
        return profiles
    for path in sorted(profile_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = ScholarProfile(
            scholar_id=str(payload.get("scholar_id", "") or ""),
            name=str(payload.get("name", "") or ""),
            name_transliterated=str(payload.get("name_transliterated", "") or ""),
            madhhab=str(payload.get("madhhab", "") or ""),
            period=str(payload.get("period", "") or ""),
            known_works=tuple(str(item) for item in payload.get("known_works", []) if item),
            methodology_notes=str(payload.get("methodology_notes", "") or ""),
            source_families=tuple(
                str(item) for item in payload.get("source_families", []) if item
            ),
            retrieval_tags=tuple(
                str(item) for item in payload.get("retrieval_tags", []) if item
            ),
        )
        if profile.scholar_id:
            profiles[profile.scholar_id] = profile
    return profiles


def _extract_unknown_scholar_name(question: str) -> str:
    compact = re.sub(r"\s+", " ", question.strip())
    patterns = (
        r"from\s+([^?.!,]+?)'?s?\s+perspective",
        r"how would\s+([^?.!,]+?)\s+approach",
        r"([^?.!,]+?)'?s\s+view on",
        r"([^?.!,]+?)\s+on\s+",
    )
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1).strip(" -'\""))
        if candidate and candidate.casefold() not in {"how", "what"}:
            return candidate
    return ""


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"'s\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
