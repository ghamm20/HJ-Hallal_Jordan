"""Weight and Trust Engine for Halal Jordan.

This module implements the charter's "Weight and Trust Engine": a transparent,
profile-driven scorer that consumes structured signals attached to a retrieval
candidate (hadith authenticity grade, isnad strength, scholar authority,
corroboration count, ijma strength, historical era, source distance,
methodology tags) and produces an additive score with a full breakdown of
every contributing component.

Two non-negotiable rules from the charter are enforced here:

1. Unknown signals never silently inflate scores. If a signal is missing,
   its contribution is zero (and optionally a small `unknown_penalty` is
   applied per missing signal to encourage richer metadata over time).
2. Every contribution is recorded in `TrustBreakdown.components` so the
   rendering layer can show users exactly why a source was weighted as it
   was — transparency over confidence.

The engine is intentionally additive. The default profile contributes zero
to every signal, so wiring this module into the reranker is safe: behavior
only changes when a non-default profile is explicitly selected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from app.reasoning.authority_policy import resolve_source_classification

REPO_ROOT = Path(__file__).resolve().parents[2]

AUTHENTICITY_GRADES = ("sahih", "hasan", "daif", "mawdu", "unknown")
IJMA_LEVELS = ("ijma", "majority", "minority", "isolated", "unknown")
ERA_LEVELS = (
    "primary",
    "classical",
    "post_classical",
    "modern",
    "contemporary",
    "unknown",
)

_AUTHENTICITY_ALIASES = {
    "sahih": "sahih",
    "saheeh": "sahih",
    "authentic": "sahih",
    "hasan": "hasan",
    "good": "hasan",
    "daif": "daif",
    "da'if": "daif",
    "weak": "daif",
    "mawdu": "mawdu",
    "mawdoo": "mawdu",
    "fabricated": "mawdu",
    "munkar": "daif",
}
_IJMA_ALIASES = {
    "ijma": "ijma",
    "ijmaa": "ijma",
    "consensus": "ijma",
    "majority": "majority",
    "jumhur": "majority",
    "minority": "minority",
    "isolated": "isolated",
    "ahad": "isolated",
    "shadhdh": "isolated",
}
_ERA_ALIASES = {
    "primary": "primary",
    "revelation": "primary",
    "sahabah": "primary",
    "salaf": "classical",
    "classical": "classical",
    "mutaqaddimun": "classical",
    "post_classical": "post_classical",
    "mutakhirun": "post_classical",
    "modern": "modern",
    "contemporary": "contemporary",
    "current": "contemporary",
}


@dataclass(slots=True, frozen=True)
class TrustSignals:
    """Structured signals extracted from a candidate's metadata.

    All fields default to a "no information" value (None or "unknown"). The
    scorer treats those as zero contribution — see the charter rule on
    unknowns.
    """

    authenticity_grade: str = "unknown"
    isnad_strength: float | None = None
    corroboration_count: int = 0
    ijma_strength: str = "unknown"
    era: str = "unknown"
    source_distance: int | None = None
    scholar_authority: float | None = None
    source_classification: str = "unknown"
    madhhab: str = ""
    methodology_tags: tuple[str, ...] = ()

    def known_signals(self) -> tuple[str, ...]:
        known: list[str] = []
        if self.authenticity_grade != "unknown":
            known.append("authenticity_grade")
        if self.isnad_strength is not None:
            known.append("isnad_strength")
        if self.corroboration_count > 0:
            known.append("corroboration_count")
        if self.ijma_strength != "unknown":
            known.append("ijma_strength")
        if self.era != "unknown":
            known.append("era")
        if self.source_distance is not None:
            known.append("source_distance")
        if self.scholar_authority is not None:
            known.append("scholar_authority")
        if self.madhhab:
            known.append("madhhab")
        if self.methodology_tags:
            known.append("methodology_tags")
        return tuple(known)

    def unknown_signals(self) -> tuple[str, ...]:
        all_signals = (
            "authenticity_grade",
            "isnad_strength",
            "corroboration_count",
            "ijma_strength",
            "era",
            "source_distance",
            "scholar_authority",
            "madhhab",
            "methodology_tags",
        )
        known = set(self.known_signals())
        return tuple(name for name in all_signals if name not in known)


@dataclass(slots=True, frozen=True)
class TrustProfile:
    """Profile mapping signals to weights.

    A profile is data, not code. Profiles are loaded from JSON in
    ``config/trust_profiles/`` and are user-extensible. The schema is at
    ``metadata/schemas/trust_profile.schema.json``.
    """

    profile_id: str
    description: str
    mode: str  # "strict" | "balanced" | "exploratory"
    authenticity_weights: Mapping[str, float] = field(default_factory=dict)
    isnad_strength_multiplier: float = 0.0
    corroboration_per_count: float = 0.0
    corroboration_cap: int = 5
    ijma_weights: Mapping[str, float] = field(default_factory=dict)
    era_weights: Mapping[str, float] = field(default_factory=dict)
    source_distance_penalty: float = 0.0
    scholar_authority_multiplier: float = 0.0
    madhhab_preferences: Mapping[str, float] = field(default_factory=dict)
    methodology_preferences: Mapping[str, float] = field(default_factory=dict)
    unknown_penalty: float = 0.0
    strictness_threshold: float | None = None

    @classmethod
    def neutral(cls, profile_id: str = "default") -> "TrustProfile":
        """A profile that contributes zero to every signal.

        Used as the safe default so that adding the trust engine to the
        pipeline does not change existing behavior until a non-default
        profile is explicitly selected.
        """

        return cls(
            profile_id=profile_id,
            description="Neutral profile — engine present, no weighting applied.",
            mode="balanced",
        )


@dataclass(slots=True, frozen=True)
class TrustComponent:
    """One contributing line in the breakdown."""

    signal: str
    observed: Any
    weight: float
    contribution: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "observed": self.observed,
            "weight": self.weight,
            "contribution": round(self.contribution, 6),
            "note": self.note,
        }


@dataclass(slots=True, frozen=True)
class TrustBreakdown:
    """Full transparent breakdown of a trust score.

    The reranker uses ``total`` as an additive bonus; the citation/rendering
    layer can surface ``components`` so users see exactly why a source was
    weighted up or down. This is the operationalization of the charter's
    "transparency is more important than confidence" rule.
    """

    profile_id: str
    total: float
    components: tuple[TrustComponent, ...]
    signals: TrustSignals
    unknowns: tuple[str, ...]
    below_strictness_threshold: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "total": round(self.total, 6),
            "components": [c.to_dict() for c in self.components],
            "signals": {
                "authenticity_grade": self.signals.authenticity_grade,
                "isnad_strength": self.signals.isnad_strength,
                "corroboration_count": self.signals.corroboration_count,
                "ijma_strength": self.signals.ijma_strength,
                "era": self.signals.era,
                "source_distance": self.signals.source_distance,
                "scholar_authority": self.signals.scholar_authority,
                "source_classification": self.signals.source_classification,
                "madhhab": self.signals.madhhab,
                "methodology_tags": list(self.signals.methodology_tags),
            },
            "unknowns": list(self.unknowns),
            "below_strictness_threshold": self.below_strictness_threshold,
        }


def extract_signals(candidate: Mapping[str, Any]) -> TrustSignals:
    """Pull structured signals from a candidate dict.

    Reads from a tolerant set of metadata keys so that documents ingested
    under earlier schemas still produce useful signals. Anything missing
    becomes "unknown"/None — never a fabricated value.
    """

    authenticity_raw = _first_string(
        candidate,
        ("hadith_grade", "authenticity_grade", "authenticity", "grading"),
    )
    authenticity = _AUTHENTICITY_ALIASES.get(authenticity_raw.lower(), "unknown")

    isnad_strength = _clamped_float(
        candidate.get("isnad_strength"), low=0.0, high=1.0
    )

    corroboration_count = _non_negative_int(candidate.get("corroboration_count"))

    ijma_raw = _first_string(
        candidate, ("ijma_strength", "consensus_level", "ijma")
    )
    ijma = _IJMA_ALIASES.get(ijma_raw.lower(), "unknown")

    era_raw = _first_string(
        candidate, ("era", "historical_period", "period_class")
    )
    era = _ERA_ALIASES.get(era_raw.lower(), "unknown")
    if era == "unknown":
        # period field on scholar profiles is human-readable like "Hanafi imam
        # of the 4th century AH" — try a light heuristic so historical works
        # carry at least a classical/post_classical signal when available.
        era = _infer_era_from_period(str(candidate.get("period", "") or ""))

    source_distance_raw = candidate.get("source_distance")
    if source_distance_raw is None:
        source_distance = _infer_source_distance(candidate)
    else:
        source_distance = _non_negative_int(source_distance_raw, default=None)

    scholar_authority = _clamped_float(
        candidate.get("scholar_authority"), low=0.0, high=1.0
    )

    classification = resolve_source_classification(dict(candidate))
    madhhab = str(candidate.get("madhhab", "") or "").strip().lower()

    methodology_tags_raw = candidate.get("methodology_tags") or ()
    if isinstance(methodology_tags_raw, str):
        methodology_tags: tuple[str, ...] = tuple(
            tag.strip().lower()
            for tag in methodology_tags_raw.split(",")
            if tag.strip()
        )
    else:
        methodology_tags = tuple(
            str(tag).strip().lower() for tag in methodology_tags_raw if str(tag).strip()
        )

    return TrustSignals(
        authenticity_grade=authenticity,
        isnad_strength=isnad_strength,
        corroboration_count=corroboration_count,
        ijma_strength=ijma,
        era=era,
        source_distance=source_distance,
        scholar_authority=scholar_authority,
        source_classification=classification,
        madhhab=madhhab,
        methodology_tags=methodology_tags,
    )


def score(
    candidate: Mapping[str, Any],
    *,
    profile: TrustProfile,
) -> TrustBreakdown:
    """Score a candidate against a profile and return the full breakdown.

    The total is purely additive; callers can use it as a reranking bonus.
    Every contribution — including zero contributions caused by unknown
    signals — is recorded so the rendering layer can show the reasoning.
    """

    signals = extract_signals(candidate)
    components: list[TrustComponent] = []
    total = 0.0

    # --- Authenticity ---
    if signals.authenticity_grade != "unknown":
        weight = float(profile.authenticity_weights.get(signals.authenticity_grade, 0.0))
        components.append(
            TrustComponent(
                signal="authenticity_grade",
                observed=signals.authenticity_grade,
                weight=weight,
                contribution=weight,
            )
        )
        total += weight

    # --- Isnad strength ---
    if signals.isnad_strength is not None and profile.isnad_strength_multiplier:
        contribution = signals.isnad_strength * profile.isnad_strength_multiplier
        components.append(
            TrustComponent(
                signal="isnad_strength",
                observed=signals.isnad_strength,
                weight=profile.isnad_strength_multiplier,
                contribution=contribution,
                note="isnad_strength * multiplier",
            )
        )
        total += contribution

    # --- Corroboration ---
    if signals.corroboration_count and profile.corroboration_per_count:
        counted = min(signals.corroboration_count, profile.corroboration_cap)
        contribution = counted * profile.corroboration_per_count
        components.append(
            TrustComponent(
                signal="corroboration_count",
                observed=signals.corroboration_count,
                weight=profile.corroboration_per_count,
                contribution=contribution,
                note=f"min(count, cap={profile.corroboration_cap}) * per_count",
            )
        )
        total += contribution

    # --- Ijma ---
    if signals.ijma_strength != "unknown":
        weight = float(profile.ijma_weights.get(signals.ijma_strength, 0.0))
        components.append(
            TrustComponent(
                signal="ijma_strength",
                observed=signals.ijma_strength,
                weight=weight,
                contribution=weight,
            )
        )
        total += weight

    # --- Era ---
    if signals.era != "unknown":
        weight = float(profile.era_weights.get(signals.era, 0.0))
        components.append(
            TrustComponent(
                signal="era",
                observed=signals.era,
                weight=weight,
                contribution=weight,
            )
        )
        total += weight

    # --- Source distance ---
    if signals.source_distance is not None and profile.source_distance_penalty:
        contribution = -signals.source_distance * profile.source_distance_penalty
        components.append(
            TrustComponent(
                signal="source_distance",
                observed=signals.source_distance,
                weight=profile.source_distance_penalty,
                contribution=contribution,
                note="-distance * penalty (further from primary = lower)",
            )
        )
        total += contribution

    # --- Scholar authority ---
    if (
        signals.scholar_authority is not None
        and profile.scholar_authority_multiplier
    ):
        contribution = signals.scholar_authority * profile.scholar_authority_multiplier
        components.append(
            TrustComponent(
                signal="scholar_authority",
                observed=signals.scholar_authority,
                weight=profile.scholar_authority_multiplier,
                contribution=contribution,
            )
        )
        total += contribution

    # --- Madhhab preference ---
    if signals.madhhab and signals.madhhab in profile.madhhab_preferences:
        weight = float(profile.madhhab_preferences[signals.madhhab])
        components.append(
            TrustComponent(
                signal="madhhab",
                observed=signals.madhhab,
                weight=weight,
                contribution=weight,
            )
        )
        total += weight

    # --- Methodology tags ---
    for tag in signals.methodology_tags:
        if tag in profile.methodology_preferences:
            weight = float(profile.methodology_preferences[tag])
            components.append(
                TrustComponent(
                    signal="methodology_tag",
                    observed=tag,
                    weight=weight,
                    contribution=weight,
                )
            )
            total += weight

    # --- Unknown penalty (encourages richer ingestion metadata) ---
    unknowns = signals.unknown_signals()
    if profile.unknown_penalty and unknowns:
        contribution = -profile.unknown_penalty * len(unknowns)
        components.append(
            TrustComponent(
                signal="unknown_signals",
                observed=list(unknowns),
                weight=profile.unknown_penalty,
                contribution=contribution,
                note="small penalty per missing structured signal",
            )
        )
        total += contribution

    below_threshold = (
        profile.strictness_threshold is not None
        and total < profile.strictness_threshold
    )

    return TrustBreakdown(
        profile_id=profile.profile_id,
        total=total,
        components=tuple(components),
        signals=signals,
        unknowns=unknowns,
        below_strictness_threshold=below_threshold,
    )


# --- Profile loading -------------------------------------------------------


def load_profile(
    profile_id: str, *, repo_root: Path | None = None
) -> TrustProfile:
    """Load a profile by id from ``config/trust_profiles/``.

    Falls back to a neutral profile if the requested id is the literal
    string ``"default"`` and no JSON file exists; raises FileNotFoundError
    for any other unknown id (we never silently substitute weights).
    """

    profiles = _load_profiles_cached(str((repo_root or REPO_ROOT).resolve()))
    if profile_id in profiles:
        return profiles[profile_id]
    if profile_id == "default":
        return TrustProfile.neutral("default")
    raise FileNotFoundError(
        f"Trust profile not found: {profile_id!r}. "
        f"Available: {sorted(profiles)}"
    )


def list_profiles(repo_root: Path | None = None) -> tuple[str, ...]:
    profiles = _load_profiles_cached(str((repo_root or REPO_ROOT).resolve()))
    ids = list(profiles.keys())
    if "default" not in ids:
        ids.append("default")
    return tuple(sorted(ids))


@lru_cache(maxsize=8)
def _load_profiles_cached(repo_root: str) -> dict[str, TrustProfile]:
    profile_dir = Path(repo_root) / "config" / "trust_profiles"
    profiles: dict[str, TrustProfile] = {}
    if not profile_dir.exists():
        return profiles
    for path in sorted(profile_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = _profile_from_payload(payload)
        if profile.profile_id:
            profiles[profile.profile_id] = profile
    return profiles


def _profile_from_payload(payload: Mapping[str, Any]) -> TrustProfile:
    return TrustProfile(
        profile_id=str(payload.get("profile_id", "") or ""),
        description=str(payload.get("description", "") or ""),
        mode=str(payload.get("mode", "balanced") or "balanced"),
        authenticity_weights=_float_map(payload.get("authenticity_weights")),
        isnad_strength_multiplier=float(
            payload.get("isnad_strength_multiplier", 0.0) or 0.0
        ),
        corroboration_per_count=float(
            payload.get("corroboration_per_count", 0.0) or 0.0
        ),
        corroboration_cap=int(payload.get("corroboration_cap", 5) or 5),
        ijma_weights=_float_map(payload.get("ijma_weights")),
        era_weights=_float_map(payload.get("era_weights")),
        source_distance_penalty=float(
            payload.get("source_distance_penalty", 0.0) or 0.0
        ),
        scholar_authority_multiplier=float(
            payload.get("scholar_authority_multiplier", 0.0) or 0.0
        ),
        madhhab_preferences=_float_map(payload.get("madhhab_preferences")),
        methodology_preferences=_float_map(payload.get("methodology_preferences")),
        unknown_penalty=float(payload.get("unknown_penalty", 0.0) or 0.0),
        strictness_threshold=(
            float(payload["strictness_threshold"])
            if payload.get("strictness_threshold") is not None
            else None
        ),
    )


# --- Helpers ---------------------------------------------------------------


def _first_string(candidate: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = candidate.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _clamped_float(value: Any, *, low: float, high: float) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < low:
        return low
    if number > high:
        return high
    return number


def _non_negative_int(value: Any, *, default: int | None = 0) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, 0)


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key).lower()] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _infer_era_from_period(period: str) -> str:
    lowered = period.lower()
    if not lowered:
        return "unknown"
    if any(token in lowered for token in ("sahabah", "companion", "revelation")):
        return "primary"
    if any(
        token in lowered
        for token in (
            "1st century",
            "2nd century",
            "3rd century",
            "4th century",
            "salaf",
            "mutaqaddim",
        )
    ):
        return "classical"
    if any(
        token in lowered
        for token in ("5th century", "6th century", "7th century", "8th century", "9th century", "mutakhir")
    ):
        return "post_classical"
    if "20th century" in lowered or "19th century" in lowered or "modern" in lowered:
        return "modern"
    if "contemporary" in lowered or "21st century" in lowered:
        return "contemporary"
    return "unknown"


def _infer_source_distance(candidate: Mapping[str, Any]) -> int | None:
    """Best-effort inference of how far a source is from a primary text.

    Returns 0 for primary texts (Quran, hadith), 1 for fiqh manuals and
    commentaries, 2 for fatwas / transcripts, None when unclassified.
    Never assumes — when classification is unknown, returns None so the
    scorer treats distance as unknown.
    """

    classification = resolve_source_classification(dict(candidate))
    distance_map = {
        "quran": 0,
        "hadith": 0,
        "fiqh_manual": 1,
        "commentary": 1,
        "tasawwuf_text": 1,
        "fatwa": 2,
        "scholar_transcript": 2,
        "transcript": 2,
    }
    if classification == "unknown":
        return None
    return distance_map.get(classification)
