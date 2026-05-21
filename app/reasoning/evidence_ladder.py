"""Evidence Ladder — visible epistemological hierarchy.

The charter (and the user's explicit ask) require that every answer
visibly show the epistemological stack the answer rests on, in order:

    1. Qur'an
    2. Mutawatir hadith
    3. Sahih hadith
    4. Athar (Companion / Successor reports)
    5. Ijma claims
    6. Qiyas
    7. Madhhab reasoning
    8. Modern fatwa
    9. Commentary
   10. Weak evidence

Users should SEE the epistemology, not just read prose.

This module is a pure classifier. It walks a list of grounded sources and
assigns each one to a tier based on observable metadata (source
classification, hadith grade, ijma signals). It never invents tier
membership — when metadata is unknown the source lands in a tier
consistent with what we DO know about it, and the breakdown records the
classification reason so the rendering layer can be transparent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(slots=True, frozen=True)
class EvidenceTier:
    tier_id: str
    rank: int  # 1 = highest authority, 10 = weakest
    label: str
    description: str


# Canonical ordered ladder. Order matters — rendering walks this list.
EVIDENCE_LADDER: tuple[EvidenceTier, ...] = (
    EvidenceTier(
        tier_id="quran",
        rank=1,
        label="Qur'an",
        description="Direct revelation. The highest authority in Islamic epistemology.",
    ),
    EvidenceTier(
        tier_id="mutawatir_hadith",
        rank=2,
        label="Mutawatir Hadith",
        description="Hadith transmitted by chains so numerous that fabrication is impossible.",
    ),
    EvidenceTier(
        tier_id="sahih_hadith",
        rank=3,
        label="Sahih Hadith",
        description="Authenticated hadith meeting the rigorous criteria of the muhaddithun.",
    ),
    EvidenceTier(
        tier_id="hasan_hadith",
        rank=3,
        label="Hasan Hadith",
        description="Sound hadith just below sahih in chain strength.",
    ),
    EvidenceTier(
        tier_id="athar",
        rank=4,
        label="Athar (Companion / Successor reports)",
        description="Reports of statements or practices of the Sahabah and Tabi'un.",
    ),
    EvidenceTier(
        tier_id="ijma",
        rank=5,
        label="Ijma Claims",
        description="Reported scholarly consensus. Strength depends on how the claim is documented.",
    ),
    EvidenceTier(
        tier_id="qiyas",
        rank=6,
        label="Qiyas (Analogical Reasoning)",
        description="Derived rulings via analogy to established primary-text rulings.",
    ),
    EvidenceTier(
        tier_id="madhhab_reasoning",
        rank=7,
        label="Madhhab Reasoning",
        description="Codified juristic positions within a specific school's usul.",
    ),
    EvidenceTier(
        tier_id="modern_fatwa",
        rank=8,
        label="Modern Fatwa",
        description="Contemporary scholarly responses applying classical principles to current questions.",
    ),
    EvidenceTier(
        tier_id="commentary",
        rank=9,
        label="Commentary",
        description="Explanatory and pedagogical material; supports understanding, not ruling.",
    ),
    EvidenceTier(
        tier_id="weak_evidence",
        rank=10,
        label="Weak Evidence",
        description="Daif hadith, isolated narrations, or material whose attestation is weak.",
    ),
)

TIER_BY_ID = {tier.tier_id: tier for tier in EVIDENCE_LADDER}


@dataclass(slots=True, frozen=True)
class LadderEntry:
    tier_id: str
    source: Any
    reason: str  # short explanation of why this source landed in this tier


@dataclass(slots=True, frozen=True)
class LadderResult:
    """Grouped, ordered output of the classifier."""

    tiers: dict[str, list[LadderEntry]] = field(default_factory=dict)

    def populated_tiers(self) -> list[EvidenceTier]:
        """Tiers that contain at least one entry, in ladder order."""

        return [tier for tier in EVIDENCE_LADDER if self.tiers.get(tier.tier_id)]

    def entries_for(self, tier_id: str) -> list[LadderEntry]:
        return list(self.tiers.get(tier_id, []))

    def is_empty(self) -> bool:
        return not any(self.tiers.values())


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_HADITH_AUTHENTICITY_KEYS = (
    "hadith_grade",
    "authenticity_grade",
    "authenticity",
    "grading",
    "authority_level",
)
_HADITH_AUTHENTICITY_NORMALIZED = {
    "sahih": "sahih",
    "saheeh": "sahih",
    "authentic": "sahih",
    "mutawatir": "mutawatir",
    "hasan": "hasan",
    "good": "hasan",
    "daif": "daif",
    "da'if": "daif",
    "weak": "daif",
    "mawdu": "mawdu",
    "mawdoo": "mawdu",
    "fabricated": "mawdu",
}

_IJMA_KEYS = ("ijma_strength", "consensus_level", "ijma")
_IJMA_PRESENT_VALUES = {"ijma", "ijmaa", "consensus", "majority", "jumhur"}


def classify_sources(sources: Sequence[Any]) -> LadderResult:
    """Group a list of sources into evidence tiers, preserving order.

    Each source is examined for observable metadata. The classifier:
      - never invents a hadith grade when none is present
      - prefers explicit metadata over inferred placement
      - records a short reason string for every assignment so the
        renderer can show *why* a source landed where it did

    Sources can be dicts, dataclasses, or any object with attribute /
    item access for the metadata fields we look at.
    """

    tiers: dict[str, list[LadderEntry]] = {}
    for source in sources:
        tier_id, reason = _classify_one(source)
        tiers.setdefault(tier_id, []).append(
            LadderEntry(tier_id=tier_id, source=source, reason=reason)
        )
    return LadderResult(tiers=tiers)


def _classify_one(source: Any) -> tuple[str, str]:
    classification = _get(source, "source_classification") or _get(source, "source_type")
    classification = str(classification or "").strip().lower()

    if classification == "quran":
        return ("quran", "source classified as Qur'an")

    if classification == "hadith":
        grade = _read_hadith_grade(source)
        if grade == "mutawatir":
            return ("mutawatir_hadith", "hadith carries mutawatir grading")
        if grade == "sahih":
            return ("sahih_hadith", "hadith graded sahih")
        if grade == "hasan":
            return ("hasan_hadith", "hadith graded hasan")
        if grade in {"daif", "mawdu"}:
            return ("weak_evidence", f"hadith graded {grade}")
        # Unknown grade: don't fabricate authenticity. Place in the
        # generic 'sahih_hadith' tier ONLY when collection-level priors
        # justify it; otherwise route to commentary so we don't
        # overstate certainty.
        collection_prior = _hadith_collection_prior(source)
        if collection_prior:
            return (
                collection_prior,
                f"collection prior ({_collection_label(source)}) without explicit grade",
            )
        return ("commentary", "hadith without explicit grading")

    if classification == "athar":
        return ("athar", "Companion / Successor report")

    if classification == "fiqh_manual":
        if _ijma_is_claimed(source):
            return ("ijma", "fiqh manual makes ijma claim")
        return ("madhhab_reasoning", "fiqh manual codifying madhhab position")

    if classification == "commentary":
        if _ijma_is_claimed(source):
            return ("ijma", "commentary records ijma claim")
        if _qiyas_is_invoked(source):
            return ("qiyas", "commentary applies qiyas")
        return ("commentary", "explanatory commentary")

    if classification == "tasawwuf_text":
        return ("commentary", "spiritual guidance (tasawwuf)")

    if classification == "fatwa":
        return ("modern_fatwa", "modern fatwa / application")

    if classification in {"scholar_transcript", "transcript"}:
        return ("commentary", "teaching / transcript material")

    return ("commentary", "unclassified source")


def _read_hadith_grade(source: Any) -> str:
    for key in _HADITH_AUTHENTICITY_KEYS:
        value = _get(source, key)
        if not value:
            continue
        normalized = _HADITH_AUTHENTICITY_NORMALIZED.get(str(value).strip().lower())
        if normalized:
            return normalized
    return ""


def _ijma_is_claimed(source: Any) -> bool:
    for key in _IJMA_KEYS:
        value = _get(source, key)
        if value and str(value).strip().lower() in _IJMA_PRESENT_VALUES:
            return True
    blob = _free_text_blob(source).lower()
    return any(token in blob for token in ("ijma", "consensus of the scholars"))


def _qiyas_is_invoked(source: Any) -> bool:
    blob = _free_text_blob(source).lower()
    return "qiyas" in blob or "analogical reasoning" in blob


def _hadith_collection_prior(source: Any) -> str:
    """Conservative collection-level priors. Used ONLY when no explicit
    grade is attached. Returns a tier id or empty string.

    Only Bukhari and Muslim get the implicit sahih lift, because their
    sahihayn status is the strongest collection-level prior in classical
    hadith methodology. Other collections do not get an implicit grade —
    they fall through to 'commentary' until ingestion enrichment attaches
    explicit grading.
    """

    blob = _free_text_blob(source).lower()
    if "bukhari" in blob or "muslim" in blob and "sahih" in blob:
        return "sahih_hadith"
    if "bukhari" in blob and "sahih" in blob:
        return "sahih_hadith"
    return ""


def _collection_label(source: Any) -> str:
    return str(
        _get(source, "collection") or _get(source, "canonical_family") or "collection"
    )


def _free_text_blob(source: Any) -> str:
    parts = []
    for key in ("title", "collection", "canonical_family", "section_label", "reference"):
        value = _get(source, key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _get(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def render_ladder_lines(
    ladder: LadderResult,
    *,
    source_formatter,
    include_empty_tiers: bool = False,
) -> list[str]:
    """Render the ladder as plain-text lines for the citation output.

    ``source_formatter(source) -> str`` is supplied by the caller so the
    ladder module stays decoupled from the renderer's exact formatting.
    """

    lines: list[str] = []
    if ladder.is_empty():
        return lines
    lines.append("Evidence Ladder")
    lines.append("Sources grouped by epistemological tier, strongest to weakest.")
    iterable: Iterable[EvidenceTier] = (
        EVIDENCE_LADDER if include_empty_tiers else ladder.populated_tiers()
    )
    for tier in iterable:
        entries = ladder.entries_for(tier.tier_id)
        marker = "[empty]" if not entries else f"[{len(entries)}]"
        lines.append(f"  {tier.rank}. {tier.label} {marker}")
        for entry in entries:
            line = source_formatter(entry.source).lstrip()
            # The formatter conventionally emits lines starting with "- ";
            # the ladder adds its own bullet, so strip a duplicate dash.
            if line.startswith("- "):
                line = line[2:]
            lines.append(f"     - {line}")
            if entry.reason:
                lines.append(f"       Reason: {entry.reason}")
    return lines
