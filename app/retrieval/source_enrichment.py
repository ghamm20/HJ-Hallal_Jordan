"""Collection-level metadata enrichment.

The Weight and Trust Engine, Evidence Ladder, and Confidence Taxonomy
all read structured signals from a candidate's metadata. Most of those
signals (era, scholar_authority, methodology_tags, sometimes
hadith_grade) are properties of the *collection a chunk comes from*,
not the chunk itself.

This module attaches those collection-level priors to documents during
ingestion, so every chunk inherits them automatically.

Two non-negotiable charter rules are enforced here:

1. **Conservative defaults.** Only collections whose entire contents
   are sahih (Sahih al-Bukhari, Sahih Muslim) carry a
   ``default_hadith_grade``. Mixed-grade collections (Abu Dawud,
   Tirmidhi, Ibn Majah, Nasai) leave ``hadith_grade`` unset rather
   than fabricating an authenticity claim. This is the operational
   form of the charter rule "code never inflates."

2. **Never override explicit values.** If the document metadata
   already carries a field, enrichment leaves it alone. Enrichment is
   a backstop, never an authority.

Provenance: each enrichment entry carries a short ``provenance`` note
so anyone auditing the system can see *why* a collection got the
priors it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class CollectionEnrichment:
    """Collection-level metadata priors.

    Keys are matched against normalized aliases (lowercased, hyphens/
    underscores collapsed). The first matching entry wins.
    """

    aliases: tuple[str, ...]
    era: str = ""
    scholar_authority: float | None = None
    methodology_tags: tuple[str, ...] = ()
    default_hadith_grade: str = ""  # ONLY when entire collection is sahih
    isnad_strength_floor: float | None = None
    madhhab: str = ""  # only when the source is intrinsic to a school
    provenance: str = ""  # short note: why these priors were assigned


# ---------------------------------------------------------------------------
# Registry — additions belong here, reviewed against the charter rules.
# ---------------------------------------------------------------------------

COLLECTION_ENRICHMENTS: tuple[CollectionEnrichment, ...] = (
    # === Hadith — the Six Books + Muwatta ===
    CollectionEnrichment(
        aliases=("sahih al-bukhari", "sahih_al-bukhari", "bukhari"),
        era="classical",
        scholar_authority=0.98,
        methodology_tags=("rigorous_isnad", "hadith_specialist"),
        default_hadith_grade="sahih",
        isnad_strength_floor=0.85,
        provenance=(
            "Entire collection authenticated by Imam al-Bukhari (d. 256 AH) "
            "according to his stringent shurut. Universally accepted as the "
            "soundest collection after the Qur'an."
        ),
    ),
    CollectionEnrichment(
        aliases=("sahih muslim", "sahih_muslim", "muslim"),
        era="classical",
        scholar_authority=0.97,
        methodology_tags=("rigorous_isnad", "hadith_specialist"),
        default_hadith_grade="sahih",
        isnad_strength_floor=0.83,
        provenance=(
            "Entire collection authenticated by Imam Muslim (d. 261 AH) "
            "to his rigorous criteria. Second in rank among the Sahihayn."
        ),
    ),
    CollectionEnrichment(
        aliases=("sunan abu dawud", "abu_dawud", "abu dawud"),
        era="classical",
        scholar_authority=0.9,
        methodology_tags=("legal_hadith_focus", "mixed_grades"),
        # NOTE: default_hadith_grade intentionally unset — Abu Dawud
        # contains sahih, hasan, and daif material. Per-hadith grading
        # must come from explicit metadata, not collection prior.
        provenance=(
            "Sunan of Abu Dawud al-Sijistani (d. 275 AH). Mixed grading; "
            "the author noted weak hadith where he included them. Defer "
            "authenticity to explicit per-hadith metadata."
        ),
    ),
    CollectionEnrichment(
        aliases=("jami al-tirmidhi", "tirmidhi", "jami at-tirmidhi"),
        era="classical",
        scholar_authority=0.88,
        methodology_tags=("explicit_per_hadith_grading", "mixed_grades"),
        provenance=(
            "Jami of al-Tirmidhi (d. 279 AH). Author explicitly grades "
            "individual hadith — defer to per-hadith metadata."
        ),
    ),
    CollectionEnrichment(
        aliases=("sunan al-nasa'i", "nasai", "nasa'i", "sunan an-nasai"),
        era="classical",
        scholar_authority=0.9,
        methodology_tags=("rigorous_selection", "mixed_grades"),
        provenance=(
            "Al-Sunan al-Sughra of al-Nasa'i (d. 303 AH). Considered "
            "among the most rigorous of the Sunan, but still mixed; "
            "defer to per-hadith metadata."
        ),
    ),
    CollectionEnrichment(
        aliases=("sunan ibn majah", "ibn_majah", "ibn majah"),
        era="classical",
        scholar_authority=0.82,
        methodology_tags=("mixed_grades",),
        provenance=(
            "Sunan of Ibn Majah (d. 273 AH). Contains sahih, hasan, "
            "daif, and a small number of fabricated narrations — defer "
            "to per-hadith metadata."
        ),
    ),
    CollectionEnrichment(
        aliases=("al-muwatta", "muwatta", "muwatta imam malik"),
        era="primary",
        scholar_authority=0.96,
        methodology_tags=("foundational_maliki_corpus", "early_compilation"),
        madhhab="maliki",
        provenance=(
            "Al-Muwatta of Imam Malik (d. 179 AH). Earliest extant "
            "hadith and fiqh compilation; foundational text of the "
            "Maliki madhhab."
        ),
    ),
    # === Hadith — compilations ===
    CollectionEnrichment(
        aliases=("riyad as-salihin", "riyad al-salihin", "riyad_al_salihin"),
        era="post_classical",
        scholar_authority=0.92,
        methodology_tags=("compilation", "authentic_focus"),
        # NOTE: default_hadith_grade unset. Al-Nawawi compiled primarily
        # from Bukhari/Muslim and other sahih/hasan sources, but the
        # collection is a thematic compilation — per-hadith grading
        # should come from the original collection metadata.
        provenance=(
            "Compilation of authentic narrations by Imam al-Nawawi "
            "(d. 676 AH), drawn primarily from Bukhari and Muslim with "
            "additional hasan material from the Sunan."
        ),
    ),
    CollectionEnrichment(
        aliases=("min nabi ilal bukhari", "min an-nabi ilal-bukhari"),
        era="contemporary",
        scholar_authority=0.7,
        methodology_tags=("transmission_history", "modern_scholarship"),
        provenance=(
            "Contemporary work on the transmission history from the "
            "Prophet to al-Bukhari. Treated as modern scholarship."
        ),
    ),
    # === Fiqh manuals — Hanafi ===
    CollectionEnrichment(
        aliases=("mukhtasar al-quduri", "quduri", "mukhtasar al quduri"),
        era="post_classical",
        scholar_authority=0.9,
        methodology_tags=("hanafi_usul",),
        madhhab="hanafi",
        provenance=(
            "Mukhtasar of al-Quduri (d. 428 AH). Foundational Hanafi "
            "primer; entry point for the school's mutun tradition."
        ),
    ),
    CollectionEnrichment(
        aliases=("mala budda minhu", "essential hanafi handbook of fiqh"),
        era="modern",
        scholar_authority=0.78,
        methodology_tags=("hanafi_usul", "modern_compilation"),
        madhhab="hanafi",
        provenance=(
            "Essential Hanafi handbook by Qazi Thanaa Ullah Pani Patti "
            "(d. 1810 CE)."
        ),
    ),
    # === Fiqh manuals — Shafi'i ===
    CollectionEnrichment(
        aliases=("minhaj al-talibin", "minhaj al talibin"),
        era="post_classical",
        scholar_authority=0.95,
        methodology_tags=("shafii_usul",),
        madhhab="shafii",
        provenance=(
            "Minhaj al-Talibin of al-Nawawi (d. 676 AH). Standard "
            "Shafi'i text studied across the school for centuries."
        ),
    ),
    CollectionEnrichment(
        aliases=("safinat al-naja", "safinat al najah", "safinat al-najah"),
        era="modern",
        scholar_authority=0.78,
        methodology_tags=("shafii_usul", "introductory"),
        madhhab="shafii",
        provenance=(
            "Safinat al-Naja by Salim al-Hadrami (d. 1855 CE). Concise "
            "introductory Shafi'i primer."
        ),
    ),
    # === Tasawwuf ===
    CollectionEnrichment(
        aliases=("ihya ulum al-din", "ihya", "ihya-v1", "ihya ulum ad-din"),
        era="post_classical",
        scholar_authority=0.95,
        methodology_tags=("spiritual_emphasis", "classical_usul"),
        provenance=(
            "Ihya 'Ulum al-Din of Abu Hamid al-Ghazali (d. 505 AH). "
            "Classical spiritual encyclopedia."
        ),
    ),
    CollectionEnrichment(
        aliases=("minhaj al-abidin", "minhaj al abidin"),
        era="post_classical",
        scholar_authority=0.95,
        methodology_tags=("spiritual_emphasis", "classical_usul"),
        provenance=(
            "Minhaj al-Abidin of al-Ghazali (d. 505 AH). Spiritual "
            "manual; the worshipful servants' path."
        ),
    ),
    CollectionEnrichment(
        aliases=("al-hikam", "hikam", "ibn ata'illah", "ibn ataillah"),
        era="post_classical",
        scholar_authority=0.92,
        methodology_tags=("spiritual_emphasis", "shadhili_methodology"),
        provenance=(
            "Al-Hikam of Ibn Ata'illah al-Iskandari (d. 709 AH). "
            "Classic Shadhili spiritual aphorisms."
        ),
    ),
    CollectionEnrichment(
        aliases=("qushayri risala", "risala al-qushayri", "qushayri"),
        era="classical",
        scholar_authority=0.93,
        methodology_tags=("spiritual_emphasis", "early_tasawwuf"),
        provenance=(
            "Al-Risala of al-Qushayri (d. 465 AH). Early classical "
            "tasawwuf reference."
        ),
    ),
    CollectionEnrichment(
        aliases=("futuh al-ghayb", "futuh al ghayb", "abd al-qadir al-jilani", "jilani"),
        era="post_classical",
        scholar_authority=0.92,
        methodology_tags=("spiritual_emphasis", "qadiri_methodology"),
        provenance=(
            "Futuh al-Ghayb of Abd al-Qadir al-Jilani (d. 561 AH). "
            "Classical Qadiri spiritual discourses."
        ),
    ),
    CollectionEnrichment(
        aliases=("book of assistance", "al-haddad", "haddad"),
        era="modern",
        scholar_authority=0.88,
        methodology_tags=("spiritual_emphasis", "ba_alawi_methodology"),
        provenance=(
            "Book of Assistance of Imam al-Haddad (d. 1132 AH / 1720 CE). "
            "Late spiritual primer."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_enrichment(metadata: Mapping[str, Any]) -> CollectionEnrichment | None:
    """Find the enrichment entry matching this document's metadata.

    Matches against normalized aliases drawn from the document's
    collection name, canonical family, source path, and title. When
    multiple entries match, the one whose matching alias is *longest*
    wins — so specific titles like ``min nabi ilal bukhari`` beat the
    generic ``bukhari`` alias on Sahih al-Bukhari.
    """

    haystack = _build_haystack(metadata)
    if not haystack:
        return None
    best: tuple[int, CollectionEnrichment] | None = None
    for enrichment in COLLECTION_ENRICHMENTS:
        for alias in enrichment.aliases:
            normalized = _normalize(alias)
            if normalized and normalized in haystack:
                length = len(normalized)
                if best is None or length > best[0]:
                    best = (length, enrichment)
    return best[1] if best else None


def apply_collection_enrichment(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge collection-level enrichment into a metadata dict.

    Charter rule: explicit values are NEVER overridden. Enrichment only
    sets fields that are currently empty / None / missing. A
    ``collection_enrichment_applied`` flag and a ``collection_enrichment_
    provenance`` note are added so the rendering layer can be transparent
    about *why* a field was set.

    Returns a new dict (does not mutate the input).
    """

    enriched = dict(metadata)
    match = find_enrichment(metadata)
    if match is None:
        return enriched

    applied: list[str] = []

    if not _is_set(enriched.get("era")) and match.era:
        enriched["era"] = match.era
        applied.append("era")
    if not _is_set(enriched.get("scholar_authority")) and match.scholar_authority is not None:
        enriched["scholar_authority"] = match.scholar_authority
        applied.append("scholar_authority")
    if match.methodology_tags and not enriched.get("methodology_tags"):
        enriched["methodology_tags"] = list(match.methodology_tags)
        applied.append("methodology_tags")
    if match.default_hadith_grade and not _is_set(enriched.get("hadith_grade")):
        enriched["hadith_grade"] = match.default_hadith_grade
        enriched["hadith_grade_source"] = "collection_prior"
        applied.append("hadith_grade")
    if match.isnad_strength_floor is not None and not _is_set(
        enriched.get("isnad_strength")
    ):
        enriched["isnad_strength"] = match.isnad_strength_floor
        enriched["isnad_strength_source"] = "collection_prior"
        applied.append("isnad_strength")
    if match.madhhab and not _is_set(enriched.get("madhhab")):
        enriched["madhhab"] = match.madhhab
        applied.append("madhhab")

    if applied:
        enriched["collection_enrichment_applied"] = applied
        enriched["collection_enrichment_provenance"] = match.provenance

    return enriched


def list_enriched_collections() -> list[dict[str, Any]]:
    """Return the registry as serializable metadata for auditing UI."""

    return [
        {
            "aliases": list(entry.aliases),
            "era": entry.era,
            "scholar_authority": entry.scholar_authority,
            "methodology_tags": list(entry.methodology_tags),
            "default_hadith_grade": entry.default_hadith_grade,
            "isnad_strength_floor": entry.isnad_strength_floor,
            "madhhab": entry.madhhab,
            "provenance": entry.provenance,
        }
        for entry in COLLECTION_ENRICHMENTS
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_haystack(metadata: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "collection",
        "canonical_family",
        "source_family",
        "title",
        "human_title",
        "source_path",
        "author",
    ):
        value = metadata.get(key)
        if value:
            parts.append(_normalize(str(value)))
    return " ".join(parts)


def _normalize(value: str) -> str:
    lowered = value.lower()
    return (
        lowered.replace("_", " ").replace("-", " ").replace("'", "")
        .replace(".", " ").replace(",", " ").replace("/", " ").replace("\\", " ")
        .strip()
    )


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True
