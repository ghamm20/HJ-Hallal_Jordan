"""Tests for collection-level metadata enrichment.

Pins the two non-negotiable charter rules for enrichment:

1. Conservative defaults. Only Sahih al-Bukhari and Sahih Muslim get
   default_hadith_grade=sahih. Mixed-grade collections (Abu Dawud,
   Tirmidhi, Ibn Majah, Nasai, Riyad as-Salihin) MUST leave
   hadith_grade unset — code never fabricates authenticity.

2. Explicit values are never overridden. Enrichment only fills empty
   fields; it never replaces metadata the ingestion pipeline already
   attached.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.source_enrichment import (  # noqa: E402
    apply_collection_enrichment,
    find_enrichment,
    list_enriched_collections,
)


# ---------------------------------------------------------------------------
# Conservative defaults — only entirely-sahih collections get the lift
# ---------------------------------------------------------------------------


def test_bukhari_gets_default_sahih_grade():
    enriched = apply_collection_enrichment({"collection": "Sahih al-Bukhari"})
    assert enriched.get("hadith_grade") == "sahih"
    assert enriched.get("hadith_grade_source") == "collection_prior"
    assert enriched.get("era") == "classical"
    assert enriched.get("scholar_authority") == 0.98


def test_muslim_gets_default_sahih_grade():
    enriched = apply_collection_enrichment({"collection": "Sahih Muslim"})
    assert enriched.get("hadith_grade") == "sahih"
    assert enriched.get("hadith_grade_source") == "collection_prior"


def test_abu_dawud_does_not_get_default_sahih_grade():
    """Charter rule: Abu Dawud contains mixed grades. The system MUST NOT
    fabricate a sahih default for the whole collection.
    """

    enriched = apply_collection_enrichment({"collection": "Sunan Abu Dawud"})
    assert "hadith_grade" not in enriched or enriched["hadith_grade"] == ""
    # But era and authority still get set
    assert enriched.get("era") == "classical"


def test_tirmidhi_does_not_get_default_sahih_grade():
    enriched = apply_collection_enrichment({"collection": "Jami al-Tirmidhi"})
    assert "hadith_grade" not in enriched or enriched["hadith_grade"] == ""


def test_ibn_majah_does_not_get_default_sahih_grade():
    enriched = apply_collection_enrichment({"collection": "Sunan Ibn Majah"})
    assert "hadith_grade" not in enriched or enriched["hadith_grade"] == ""


def test_nasai_does_not_get_default_sahih_grade():
    enriched = apply_collection_enrichment({"collection": "Sunan al-Nasa'i"})
    assert "hadith_grade" not in enriched or enriched["hadith_grade"] == ""


def test_riyad_as_salihin_does_not_get_default_sahih_grade():
    """Even though al-Nawawi compiled primarily from Bukhari/Muslim,
    Riyad as-Salihin is a thematic compilation and per-hadith grading
    should come from the original source. Enrichment must not invent.
    """

    enriched = apply_collection_enrichment({"collection": "Riyad as-Salihin"})
    assert "hadith_grade" not in enriched or enriched["hadith_grade"] == ""
    # But it does get era and methodology tags
    assert enriched.get("era") == "post_classical"
    assert "authentic_focus" in enriched.get("methodology_tags", [])


# ---------------------------------------------------------------------------
# Explicit values never overridden
# ---------------------------------------------------------------------------


def test_explicit_hadith_grade_not_overridden():
    """If a chunk already carries an explicit hadith_grade (e.g. daif),
    enrichment must NOT overwrite it with the collection default.
    """

    enriched = apply_collection_enrichment({
        "collection": "Sahih al-Bukhari",
        "hadith_grade": "daif",  # explicit, even if surprising
    })
    assert enriched["hadith_grade"] == "daif"
    # And we did NOT mark this as a collection-prior assignment
    assert enriched.get("hadith_grade_source") != "collection_prior"


def test_explicit_era_not_overridden():
    enriched = apply_collection_enrichment({
        "collection": "Sahih al-Bukhari",
        "era": "modern",  # explicit, weird but allowed
    })
    assert enriched["era"] == "modern"


def test_explicit_madhhab_not_overridden_by_muwatta_enrichment():
    enriched = apply_collection_enrichment({
        "collection": "Al-Muwatta",
        "madhhab": "shafii",  # weird but explicit
    })
    assert enriched["madhhab"] == "shafii"


# ---------------------------------------------------------------------------
# Specific shipped enrichments — spot checks
# ---------------------------------------------------------------------------


def test_minhaj_al_abidin_gets_tasawwuf_tags():
    enriched = apply_collection_enrichment({"title": "Minhaj al-Abidin"})
    assert enriched.get("era") == "post_classical"
    assert "spiritual_emphasis" in enriched.get("methodology_tags", [])


def test_ihya_gets_classical_usul_tag():
    enriched = apply_collection_enrichment({"collection": "Ihya ulum al-din"})
    assert "classical_usul" in enriched.get("methodology_tags", [])


def test_quduri_gets_hanafi_methodology_and_madhhab():
    enriched = apply_collection_enrichment({"collection": "Mukhtasar al-Quduri"})
    assert enriched.get("madhhab") == "hanafi"
    assert "hanafi_usul" in enriched.get("methodology_tags", [])


def test_minhaj_al_talibin_gets_shafii_methodology():
    enriched = apply_collection_enrichment({"collection": "Minhaj al-Talibin"})
    assert enriched.get("madhhab") == "shafii"
    assert "shafii_usul" in enriched.get("methodology_tags", [])


def test_muwatta_gets_primary_era():
    """Imam Malik was a Tabi al-Tabi'in; his Muwatta is the earliest
    extant hadith/fiqh compilation. Era = primary.
    """

    enriched = apply_collection_enrichment({"collection": "Al-Muwatta"})
    assert enriched.get("era") == "primary"
    assert enriched.get("madhhab") == "maliki"


def test_min_nabi_ilal_bukhari_is_contemporary_scholarship():
    enriched = apply_collection_enrichment({"title": "Min an-Nabi ilal-Bukhari"})
    assert enriched.get("era") == "contemporary"
    assert "transmission_history" in enriched.get("methodology_tags", [])


# ---------------------------------------------------------------------------
# Lookup behavior
# ---------------------------------------------------------------------------


def test_find_enrichment_matches_by_title_alias():
    match = find_enrichment({"title": "Sahih al-Bukhari Volume 1"})
    assert match is not None
    assert match.default_hadith_grade == "sahih"


def test_find_enrichment_returns_none_for_unknown_collection():
    match = find_enrichment({"collection": "Some Random Modern Blog Post"})
    assert match is None


def test_find_enrichment_returns_none_for_empty_metadata():
    assert find_enrichment({}) is None


def test_unknown_collection_yields_unchanged_metadata():
    metadata = {"collection": "Unknown Manuscript", "era": "modern"}
    enriched = apply_collection_enrichment(metadata)
    assert enriched == metadata
    assert "collection_enrichment_applied" not in enriched


# ---------------------------------------------------------------------------
# Provenance — auditability
# ---------------------------------------------------------------------------


def test_enrichment_records_provenance_when_applied():
    enriched = apply_collection_enrichment({"collection": "Sahih al-Bukhari"})
    assert "collection_enrichment_applied" in enriched
    assert "hadith_grade" in enriched["collection_enrichment_applied"]
    assert enriched.get("collection_enrichment_provenance")
    assert "al-Bukhari" in enriched["collection_enrichment_provenance"]


def test_list_enriched_collections_serializable():
    entries = list_enriched_collections()
    assert len(entries) >= 15
    for entry in entries:
        assert "aliases" in entry
        assert "provenance" in entry
        assert entry["provenance"], "every enrichment must carry a provenance note"


# ---------------------------------------------------------------------------
# End-to-end: enrichment flows through infer_document_metadata
# ---------------------------------------------------------------------------


def test_enrichment_flows_through_infer_document_metadata():
    from app.retrieval.index_loader import infer_document_metadata

    metadata = infer_document_metadata(
        Path("data/raw/hadith/bukhari/sahih_al_bukhari_vol_1.pdf"),
    )
    # The enrichment should have attached era and authority to a Bukhari source
    assert metadata.get("era") == "classical"
    assert metadata.get("hadith_grade") == "sahih"
