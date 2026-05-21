"""Shared metadata normalization for documents, chunks, and citations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from app.retrieval.families import build_duplicate_family_key

CANONICAL_SOURCE_TYPES = {
    "quran",
    "hadith",
    "fiqh_manual",
    "commentary",
    "tasawwuf_text",
    "fatwa",
    "scholar_transcript",
    "transcript",
}
SOURCE_FAMILY_IDS = {"quran", "hadith", "fiqh", "tasawwuf", "fatwas", "scholars", "classes"}
SCHOOL_MADHHAB_IDS = {"hanafi", "shafii", "maliki", "hanbali"}
NON_SCHOOL_MADHHAB_IDS = {"comparative"}
MADHHAB_IDS = SCHOOL_MADHHAB_IDS | NON_SCHOOL_MADHHAB_IDS
MADHHAB_LABELS = {
    "hanafi": "Hanafi",
    "shafii": "Shafi'i",
    "maliki": "Maliki",
    "hanbali": "Hanbali",
    "comparative": "Comparative",
}
GENERIC_PATH_PARTS = {
    "data",
    "raw",
    "processed",
    "normalized",
    "clean_text",
    "pdf",
    "metadata",
    "classes",
    "translations",
    "arabic",
    "comparative",
    "general",
    "seekersguidance",
    "imports_staging",
    "unknown_or_mixed",
    "review_queue",
    "major_schools",
    "topic_studies",
    "mutun",
    "shuruh",
    "fatawa",
    "usul",
    "other",
}
KNOWN_COLLECTION_DESCRIPTORS = (
    {
        "aliases": ("bukhari", "sahih al-bukhari", "sahih_al-bukhari"),
        "collection": "Sahih al-Bukhari",
        "author": "Imam al-Bukhari",
    },
    {
        "aliases": ("muslim", "sahih muslim", "sahih_muslim"),
        "collection": "Sahih Muslim",
        "author": "Imam Muslim",
    },
    {
        "aliases": ("abu_dawud", "abu dawud", "sunan abu dawud"),
        "collection": "Sunan Abu Dawud",
        "author": "Abu Dawud al-Sijistani",
    },
    {
        "aliases": ("ibn_majah", "ibn majah", "sunan ibn majah"),
        "collection": "Sunan Ibn Majah",
        "author": "Ibn Majah",
    },
    {
        "aliases": ("nasai", "nasa'i", "nasaa", "sunan an-nasai", "sunan an nasai"),
        "collection": "Sunan al-Nasa'i",
        "author": "al-Nasa'i",
    },
    {
        "aliases": ("tirmidhi", "jami at-tirmidhi", "jami at tirmidhi"),
        "collection": "Jami al-Tirmidhi",
        "author": "al-Tirmidhi",
    },
    {
        "aliases": ("muwatta", "al-muwatta", "muwatta imam malik"),
        "collection": "Al-Muwatta",
        "author": "Imam Malik",
    },
    {
        "aliases": ("mukhtasar al-quduri", "mukhtasar al quduri", "quduri"),
        "collection": "Mukhtasar al-Quduri",
        "author": "Abu al-Husayn Ahmad ibn Muhammad al-Quduri",
    },
    {
        "aliases": ("mala budda minhu", "essential hanafi handbook of fiqh"),
        "collection": "Mala Budda Minhu",
        "author": "Qazi Thanaa Ullah",
    },
    {
        "aliases": (
            "safinat al-naja",
            "safinat-al-naja",
            "safinat al najah",
            "safinat-al-najah",
            "safinat al najat",
            "matn safinat al-naja",
            "matn safinat al najat",
        ),
        "collection": "Safinat al-Naja",
        "author": "Salim ibn Abdullah ibn Sa'd ibn Samir al-Hadrami",
    },
    {
        "aliases": (
            "minhaj al-talibin",
            "minhaj-al-talibin",
            "minhaj al talibin",
            "minhaj-al-talibin-english",
            "minhaj et talibin",
            "minhajettalibin",
        ),
        "collection": "Minhaj al-Talibin",
        "author": "Imam al-Nawawi",
    },
    {
        "aliases": (
            "al-risala-ibn-abi-zayd",
            "al risala ibn abi zayd",
            "risala ibn abi zayd",
            "risala ibn abi zaid",
            "al risala ibn abi zaid",
            "al risala ibn abi zayd al qayrawani",
        ),
        "collection": "Al-Risala",
        "author": "Ibn Abi Zayd al-Qayrawani",
    },
    {
        "aliases": (
            "al-murshid-al-muin",
            "al murshid al muin",
            "al-murshid al-muin",
            "al murshid al-mu'in",
            "murshid al muin",
            "murshid al-muin",
            "al-murshid-al-muin-english-footnotes",
        ),
        "collection": "Al-Murshid al-Mu'in",
        "author": "Ibn Ashir",
    },
    {
        "aliases": (
            "akhsar al-mukhtasarat",
            "akhsar al mukhtasarat",
            "akhsar-al-mukhtasarat-english",
            "hanbali acts of worship",
            "the supreme synopsis",
        ),
        "collection": "Akhsar al-Mukhtasarat",
        "author": "Ibn Balban al-Hanbali",
    },
    {
        "aliases": (
            "sharh umdat al-fiqh",
            "sharh-umdat-al-fiqh",
            "sharh 'umdah al fiqh",
            "sharh umdah al fiqh",
            "commentary on umdat al-fiqh",
            "commentary on umdat al fiqh",
            "fiqh of worship",
            "umdah al fiqh",
            "umdat al fiqh",
        ),
        "collection": "Sharh Umdat al-Fiqh",
        "author": "Hatem al-Hajj",
    },
    {
        "aliases": ("english_rwwad", "english rwwad"),
        "collection": "English Rwwad Translation",
        "author": "The Association for Multi-lingual Islamic Content",
    },
    {
        "aliases": ("translation_of_the_meanings_quran", "translation of the meanings quran"),
        "collection": "Translation of the Meanings of the Quran",
        "author": "Muhammad Taqi-ud-Din al-Hilali; Muhammad Muhsin Khan",
    },
    {
        "aliases": ("quran-simple", "quran simple"),
        "collection": "Quran Arabic Text",
        "author": "",
    },
    {
        "aliases": ("ihya-v1", "ihya v1", "ghazali ihya"),
        "collection": "Ihya' 'Ulum al-Din",
        "author": "Imam al-Ghazali",
    },
    {
        "aliases": ("book-of-assistance", "book of assistance", "haddad book of assistance"),
        "collection": "The Book of Assistance",
        "author": "Imam al-Haddad",
    },
    {
        "aliases": ("qushayri risala",),
        "collection": "Al-Risala al-Qushayriyya",
        "author": "Imam al-Qushayri",
    },
    {
        "aliases": ("al-hikam", "hikam ibn ata allah", "ibn ata allah hikam"),
        "collection": "Al-Hikam",
        "author": "Ibn Ata'illah al-Iskandari",
    },
    {
        "aliases": ("futuh al ghayb", "futuh al ghaib", "jilani futuh"),
        "collection": "Futuh al-Ghayb",
        "author": "Abd al-Qadir al-Jilani",
    },
)

SCHOLAR_AUTHOR_BY_FOLDER = {
    "s_j_d": "Shaykh Jamaal Diwan",
    "jamal_dewan": "Jamal Dewan",
    "ibn_arabi": "Ibn Arabi",
    "drumar": "Dr. Umar",
    "sufism": "Sufism General",
}

SCHOLAR_COLLECTION_BY_FOLDER = {
    "transcripts": "Scholar Transcript Series",
    "fiqh": "Fiqh Teaching Series",
    "futuwwah": "Futuwwah Teaching Series",
    "burdah": "Burdah Teaching Series",
    "general": "General Spiritual Teaching Series",
}


def format_madhhab_label(value: Any) -> str:
    normalized = _clean_metadata_text(value).casefold()
    if not normalized:
        return ""
    return MADHHAB_LABELS.get(normalized, humanize_metadata_label(normalized))


def infer_path_madhhab(relative_path: Path | str | list[str]) -> str:
    if isinstance(relative_path, list):
        parts = [str(part).casefold() for part in relative_path if str(part).strip()]
    else:
        parts = [
            part.casefold()
            for part in Path(relative_path).as_posix().split("/")
            if part
        ]
    if "fiqh" not in parts and "classes" not in parts:
        return ""
    if "comparative" in parts:
        return "comparative"
    for madhhab in ("hanafi", "shafii", "maliki", "hanbali"):
        if madhhab in parts:
            return madhhab
    return ""


def normalize_document_metadata(
    relative_path: Path | str,
    explicit_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    explicit_metadata = explicit_metadata or {}
    path = Path(relative_path)
    parts = [part.casefold() for part in path.as_posix().split("/") if part]

    source_family = _normalize_source_family(
        explicit_metadata.get("source_family"),
        parts,
    )
    source_type = _normalize_source_type(
        explicit_metadata.get("source_type"),
        source_family=source_family,
        parts=parts,
    )
    source_classification = _normalize_source_classification(
        explicit_metadata.get("source_classification"),
        source_type=source_type,
        source_family=source_family,
        parts=parts,
    )
    title = _normalize_title(
        explicit_metadata.get("title"),
        source_path=path,
    )
    collection_descriptor = _resolve_collection_descriptor(
        title=title,
        source_path=path,
        parts=parts,
    )
    collection = _normalize_collection(
        explicit_metadata.get("collection"),
        title=title,
        source_family=source_family,
        source_path=path,
        parts=parts,
        collection_descriptor=collection_descriptor,
    )
    author = _normalize_author(
        explicit_metadata.get("author"),
        parts=parts,
        collection_descriptor=collection_descriptor,
    )
    madhhab = _normalize_madhhab(
        explicit_metadata.get("madhhab"),
        parts=parts,
    )
    language = _normalize_language(
        explicit_metadata.get("language"),
        parts=parts,
        filename=path.name,
        content_sample=_content_sample_from_metadata(explicit_metadata),
    )
    section_label = _clean_metadata_text(explicit_metadata.get("section_label"))
    section_kind = _clean_metadata_text(explicit_metadata.get("section_kind")).casefold()
    book = _clean_metadata_text(explicit_metadata.get("book"))
    chapter = _clean_metadata_text(explicit_metadata.get("chapter"))
    section = _clean_metadata_text(explicit_metadata.get("section"))
    derived_book, derived_chapter, derived_section = _derive_section_fields(
        section_label=section_label,
        section_kind=section_kind,
    )
    if not book:
        book = derived_book
    if not chapter:
        chapter = derived_chapter
    if not section:
        section = derived_section

    document_kind = _normalize_document_kind(
        explicit_metadata.get("document_kind"),
        source_type=source_type,
        source_family=source_family,
        parts=parts,
        loader_hint=_clean_metadata_text(explicit_metadata.get("loader_hint")).casefold(),
        extension=path.suffix.casefold(),
    )
    commentary_target = _normalize_commentary_target(
        explicit_metadata.get("commentary_target"),
        source_classification=source_classification,
        source_family=source_family,
        parts=parts,
        book=book,
        chapter=chapter,
        section=section,
    )
    fatwa_authority = _normalize_fatwa_authority(
        explicit_metadata.get("fatwa_authority"),
        source_classification=source_classification,
        author=author,
        parts=parts,
    )
    canonical_family = _normalize_canonical_family(
        explicit_metadata.get("canonical_family"),
        source_type=source_type,
        title=title,
        collection=collection,
        source_path=path.as_posix(),
    )

    metadata = {
        "title": title,
        "source_type": source_type,
        "source_classification": source_classification,
        "role": _normalize_role(
            explicit_metadata.get("role"),
            source_classification=source_classification,
            source_family=source_family,
        ),
        "domain": _normalize_domain(
            explicit_metadata.get("domain"),
            source_classification=source_classification,
            source_family=source_family,
        ),
        "authority_level": _normalize_authority_level(
            explicit_metadata.get("authority_level"),
            source_classification=source_classification,
            source_family=source_family,
        ),
        "source_role_flag": _normalize_source_role_flag(
            explicit_metadata.get("source_role_flag"),
            source_classification=source_classification,
            source_family=source_family,
        ),
        "source_family": source_family,
        "canonical_family": canonical_family,
        "collection": collection,
        "author": author,
        "madhhab": madhhab,
        "language": language,
        "book": book,
        "chapter": chapter,
        "section": section,
        "document_kind": document_kind,
        "source_role_boundary": _normalize_source_role_boundary(
            explicit_metadata.get("source_role_boundary"),
            source_classification=source_classification,
            document_kind=document_kind,
            source_family=source_family,
        ),
        "source_lineage": _normalize_source_lineage(
            explicit_metadata.get("source_lineage"),
            loader_hint=_clean_metadata_text(explicit_metadata.get("loader_hint")).casefold(),
            source_path=path,
            extension=path.suffix.casefold(),
            explicit_metadata=explicit_metadata,
        ),
        "commentary_target": commentary_target,
        "fatwa_authority": fatwa_authority,
        "hierarchy_label": "",
    }
    metadata["hierarchy_label"] = format_source_hierarchy(metadata)
    return metadata


def normalize_source_metadata(source: Mapping[str, Any]) -> dict[str, Any]:
    source_path = str(source.get("source_path", "") or source.get("document_id", "") or "")
    metadata = normalize_document_metadata(source_path, explicit_metadata=source)
    for key in (
        "reference",
        "page_reference",
        "page_number",
        "page_chunk_index",
        "page_chunk_total",
        "section_label",
        "section_kind",
        "loader_hint",
        "document_extension",
        "quote",
        "quote_window",
        "text",
        "source_role_boundary",
        "source_lineage",
        "ocr_derived",
        "ocr_backend",
        "ocr_status",
        "ocr_confidence",
        "extraction_status",
        "extraction_quality",
        "role",
        "domain",
        "authority_level",
    ):
        value = source.get(key)
        if value not in (None, ""):
            metadata[key] = value
    return metadata


def format_source_hierarchy(metadata: Mapping[str, Any]) -> str:
    parts: list[str] = []
    collection = humanize_metadata_label(str(metadata.get("collection", "") or ""))
    book = humanize_metadata_label(str(metadata.get("book", "") or ""))
    chapter = humanize_metadata_label(str(metadata.get("chapter", "") or ""))
    section = humanize_metadata_label(str(metadata.get("section", "") or ""))

    for value in (collection, book, chapter, section):
        if not value:
            continue
        if parts and value.casefold() == parts[-1].casefold():
            continue
        parts.append(value)
    return " > ".join(parts)


def humanize_metadata_label(value: str) -> str:
    cleaned = _clean_metadata_text(value)
    if not cleaned:
        return ""
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\.{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -._")
    if not cleaned:
        return ""
    if cleaned == cleaned.casefold():
        cleaned = cleaned.title()
    return cleaned


def _normalize_source_family(explicit_value: Any, parts: list[str]) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in SOURCE_FAMILY_IDS:
        return explicit
    for part in parts:
        if part in SOURCE_FAMILY_IDS:
            return part
    return ""


def _normalize_source_type(
    explicit_value: Any,
    *,
    source_family: str,
    parts: list[str],
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in CANONICAL_SOURCE_TYPES:
        return explicit

    if source_family == "quran":
        if "tafsir" in parts:
            return "commentary"
        return "quran"
    if source_family == "hadith":
        if any(part in {"shuruh", "commentary"} for part in parts):
            return "commentary"
        return "hadith"
    if source_family == "fiqh":
        if any(part in {"shuruh", "commentary"} for part in parts):
            return "commentary"
        if "fatawa" in parts:
            return "fatwa"
        return "fiqh_manual"
    if source_family == "tasawwuf":
        return "tasawwuf_text"
    if source_family == "fatwas":
        return "fatwa"
    if source_family == "scholars":
        return "scholar_transcript"
    if source_family == "classes":
        return "transcript"
    return ""


def _normalize_source_classification(
    explicit_value: Any,
    *,
    source_type: str,
    source_family: str,
    parts: list[str],
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in CANONICAL_SOURCE_TYPES or explicit == "unknown":
        return explicit
    if source_type in CANONICAL_SOURCE_TYPES:
        return source_type
    if source_family == "quran" and "tafsir" in parts:
        return "commentary"
    if source_family == "tasawwuf":
        return "tasawwuf_text"
    if source_family == "classes":
        return "transcript"
    return "unknown"


def _normalize_title(explicit_value: Any, *, source_path: Path) -> str:
    explicit = humanize_metadata_label(str(explicit_value or ""))
    if explicit:
        return explicit
    return humanize_metadata_label(source_path.stem) or source_path.stem


def _normalize_collection(
    explicit_value: Any,
    *,
    title: str,
    source_family: str,
    source_path: Path,
    parts: list[str],
    collection_descriptor: dict[str, str] | None,
) -> str:
    explicit = _clean_metadata_text(explicit_value)
    if explicit:
        return explicit
    if source_family == "scholars":
        scholar_collection = _scholar_collection_from_parts(parts)
        if scholar_collection:
            return scholar_collection
    if source_family == "classes":
        descriptive_part = _find_descriptive_path_part(parts)
        if descriptive_part:
            return humanize_metadata_label(descriptive_part)
    if collection_descriptor and collection_descriptor.get("collection"):
        return str(collection_descriptor["collection"])
    if source_family == "quran" and "_v" in source_path.stem:
        return source_path.stem.split("_v", 1)[0]

    descriptive_part = _find_descriptive_path_part(parts)
    if descriptive_part and descriptive_part.casefold() not in {
        source_family,
        *MADHHAB_IDS,
    }:
        return descriptive_part
    return source_path.stem or title


def _normalize_author(
    explicit_value: Any,
    *,
    parts: list[str],
    collection_descriptor: dict[str, str] | None,
) -> str:
    explicit = humanize_metadata_label(str(explicit_value or ""))
    if explicit:
        return explicit
    scholar_author = _scholar_author_from_parts(parts)
    if scholar_author:
        return scholar_author
    if collection_descriptor and collection_descriptor.get("author"):
        return str(collection_descriptor["author"])
    if "by_scholar" in parts:
        scholar_index = parts.index("by_scholar")
        if scholar_index + 1 < len(parts):
            return humanize_metadata_label(parts[scholar_index + 1])
    return ""


def _normalize_madhhab(explicit_value: Any, *, parts: list[str]) -> str:
    path_madhhab = infer_path_madhhab(parts)
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in MADHHAB_IDS:
        return explicit
    if path_madhhab:
        return path_madhhab
    return ""


def _normalize_language(
    explicit_value: Any,
    *,
    parts: list[str],
    filename: str,
    content_sample: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in {"ar", "en"}:
        return explicit
    lowered_name = filename.casefold()
    if "arabic" in parts or lowered_name.startswith("ar_"):
        return "ar"
    if "english" in lowered_name or lowered_name.startswith("en_"):
        return "en"
    if _looks_english_filename(lowered_name):
        return "en"
    sample_language = _infer_language_from_sample(content_sample)
    if sample_language:
        return sample_language
    return ""


def _normalize_document_kind(
    explicit_value: Any,
    *,
    source_type: str,
    source_family: str,
    parts: list[str],
    loader_hint: str,
    extension: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit:
        return explicit
    if source_family == "quran" and "translations" in parts:
        return "translation"
    if source_family == "quran" and "tafsir" in parts:
        return "tafsir"
    if source_family == "quran":
        return "primary_text"
    if source_type == "hadith":
        return "collection"
    if source_family == "tasawwuf":
        return "spiritual_text"
    if source_family == "fiqh" and "mutun" in parts:
        return "matn"
    if source_family == "fiqh" and "shuruh" in parts:
        return "sharh"
    if source_family == "fiqh" and "usul" in parts:
        return "usul"
    if source_family == "fiqh" and source_type == "fiqh_manual":
        return "manual"
    if source_family == "classes":
        if "study_note" in parts:
            return "study_note"
        return "class"
    if source_type == "fatwa":
        if "lectures" in parts:
            return "lecture"
        if "books" in parts:
            return "collection"
        return "fatwa"
    if source_type == "scholar_transcript":
        return "transcript"
    if source_type == "transcript":
        if source_family == "classes":
            return "class"
        return "transcript"
    if loader_hint == "normalized_pdf_json":
        return "normalized_pdf"
    if extension == ".csv":
        return "tabular_text"
    if extension == ".txt":
        return "plain_text"
    return ""


def _normalize_commentary_target(
    explicit_value: Any,
    *,
    source_classification: str,
    source_family: str,
    parts: list[str],
    book: str,
    chapter: str,
    section: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in {"quran", "hadith", "fiqh_manual"}:
        return explicit
    if source_classification != "commentary":
        return ""
    if source_family == "quran" or "tafsir" in parts:
        return "quran"
    if source_family == "hadith":
        return "hadith"
    if source_family == "fiqh" or any(value for value in (book, chapter, section)):
        return "fiqh_manual"
    return ""


def _normalize_fatwa_authority(
    explicit_value: Any,
    *,
    source_classification: str,
    author: str,
    parts: list[str],
) -> str:
    explicit = humanize_metadata_label(str(explicit_value or ""))
    if explicit:
        return explicit
    if source_classification != "fatwa":
        return ""
    if author:
        return author
    if "by_scholar" in parts:
        scholar_index = parts.index("by_scholar")
        if scholar_index + 1 < len(parts):
            return humanize_metadata_label(parts[scholar_index + 1])
    return ""


def _normalize_canonical_family(
    explicit_value: Any,
    *,
    source_type: str,
    title: str,
    collection: str,
    source_path: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit:
        return explicit
    if source_type not in CANONICAL_SOURCE_TYPES:
        return ""
    duplicate_key = build_duplicate_family_key(
        source_type=source_type,
        title=title,
        source_path=source_path,
        collection=collection,
    )
    if "|" in duplicate_key:
        return duplicate_key.split("|", 1)[1]
    return duplicate_key


def _normalize_source_role_flag(
    explicit_value: Any,
    *,
    source_classification: str,
    source_family: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in {
        "primary_text",
        "commentary",
        "authority",
        "modern_application",
        "informal_explanation",
        "teaching_layer",
        "unknown",
    }:
        return explicit
    if source_classification in {"quran", "hadith"}:
        return "primary_text"
    if source_classification == "commentary":
        return "commentary"
    if source_classification == "fiqh_manual":
        return "authority"
    if source_classification == "tasawwuf_text":
        return "spiritual_guidance"
    if source_classification == "scholar_transcript":
        return "informal_explanation"
    if source_classification == "fatwa":
        return "modern_application"
    if source_family == "classes":
        return "informal_explanation"
    if source_classification == "transcript":
        return "informal_explanation"
    return "unknown"


def _normalize_source_role_boundary(
    explicit_value: Any,
    *,
    source_classification: str,
    document_kind: str,
    source_family: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit in {
        "primary_text",
        "commentary",
        "fatwa",
        "manual",
        "spiritual_guidance",
        "translation",
        "transcript",
        "modern_application",
        "teaching_layer",
        "unknown",
    }:
        return explicit
    if document_kind == "translation":
        return "translation"
    if source_classification in {"quran", "hadith"}:
        return "primary_text"
    if source_classification == "fiqh_manual":
        return "manual"
    if source_classification == "tasawwuf_text":
        return "spiritual_guidance"
    if source_classification == "commentary":
        return "commentary"
    if source_classification == "fatwa":
        return "fatwa"
    if source_family == "classes":
        return "teaching_layer"
    if source_classification == "scholar_transcript":
        return "transcript"
    if source_classification == "transcript":
        return "transcript"
    return "unknown"


def _normalize_source_lineage(
    explicit_value: Any,
    *,
    loader_hint: str,
    source_path: Path,
    extension: str,
    explicit_metadata: Mapping[str, Any],
) -> str:
    explicit = _clean_metadata_text(explicit_value)
    if explicit:
        return explicit
    document_type = _clean_metadata_text(explicit_metadata.get("document_type")).casefold()
    parser = _clean_metadata_text(explicit_metadata.get("parser")).casefold()
    extraction_status = _clean_metadata_text(explicit_metadata.get("extraction_status")).casefold()
    ocr_backend = _clean_metadata_text(explicit_metadata.get("ocr_backend")).casefold()
    ocr_derived = _truthy(explicit_metadata.get("ocr_derived"))

    if document_type == "normalized_pdf" or loader_hint == "normalized_pdf_json":
        parser_label = parser or "unknown_parser"
        status_label = extraction_status or "unknown_status"
        if ocr_derived or ocr_backend:
            backend_label = ocr_backend or "unknown_ocr_backend"
            return f"normalized_pdf_ocr:{backend_label}:{status_label}"
        return f"normalized_pdf:{parser_label}:{status_label}"
    if document_type == "normalized_transcript" or loader_hint == "normalized_transcript_json":
        parser_label = parser or "text_file"
        status_label = extraction_status or "success"
        return f"normalized_transcript:{parser_label}:{status_label}"
    if document_type == "class_material" or loader_hint == "normalized_class_json":
        parser_label = parser or "class_manifest"
        status_label = extraction_status or "success"
        return f"normalized_class:{parser_label}:{status_label}"
    if loader_hint == "quran_translation_csv":
        return "raw_csv_translation"
    if loader_hint == "quran_ayah_text":
        return "raw_text_primary"
    if loader_hint == "csv_rows":
        return "raw_csv_rows"
    if extension == ".txt":
        return "raw_text"
    if extension == ".csv":
        return "raw_csv"
    if extension == ".md":
        return "raw_markdown"
    return ""


def _normalize_role(
    explicit_value: Any,
    *,
    source_classification: str,
    source_family: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit:
        return explicit
    if source_classification == "tasawwuf_text":
        return "spiritual_guidance"
    if source_classification == "scholar_transcript":
        return "commentary"
    if source_family == "classes":
        return "commentary"
    return ""


def _normalize_domain(
    explicit_value: Any,
    *,
    source_classification: str,
    source_family: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit:
        return explicit
    if source_classification == "tasawwuf_text":
        return "spiritual"
    if source_classification == "scholar_transcript":
        return "teaching"
    if source_family == "classes":
        return "teaching"
    return ""


def _normalize_authority_level(
    explicit_value: Any,
    *,
    source_classification: str,
    source_family: str,
) -> str:
    explicit = _clean_metadata_text(explicit_value).casefold()
    if explicit:
        return explicit
    if source_classification == "tasawwuf_text":
        return "classical"
    if source_classification == "scholar_transcript":
        return "modern"
    if source_family == "classes":
        return "modern"
    return ""


def _scholar_author_from_parts(parts: list[str]) -> str:
    for part in parts:
        if part in SCHOLAR_AUTHOR_BY_FOLDER:
            return SCHOLAR_AUTHOR_BY_FOLDER[part]
    if "general" in parts and "sufism" in parts:
        return SCHOLAR_AUTHOR_BY_FOLDER["sufism"]
    return ""


def _scholar_collection_from_parts(parts: list[str]) -> str:
    for part in reversed(parts):
        if part in SCHOLAR_COLLECTION_BY_FOLDER:
            return SCHOLAR_COLLECTION_BY_FOLDER[part]
    return ""


def _derive_section_fields(
    *,
    section_label: str,
    section_kind: str,
) -> tuple[str, str, str]:
    if not section_label:
        return "", "", ""
    lowered = section_label.casefold()
    if section_kind == "book_heading" or lowered.startswith("book "):
        return section_label, "", ""
    if section_kind == "chapter_heading" or lowered.startswith("chapter "):
        return "", section_label, ""
    return "", "", section_label


def _find_descriptive_path_part(parts: list[str]) -> str:
    for part in reversed(parts[:-1]):
        if part in GENERIC_PATH_PARTS or part in SOURCE_FAMILY_IDS or part in MADHHAB_IDS:
            continue
        return part
    return ""


def _clean_metadata_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def _resolve_collection_descriptor(
    *,
    title: str,
    source_path: Path,
    parts: list[str],
) -> dict[str, str] | None:
    lowered_path = source_path.as_posix().casefold()
    lowered_title = title.casefold()
    normalized_haystacks = [
        lowered_path,
        lowered_title,
        lowered_path.replace("_", " "),
        lowered_title.replace("_", " "),
        " ".join(parts),
    ]
    for descriptor in KNOWN_COLLECTION_DESCRIPTORS:
        for alias in descriptor["aliases"]:
            alias_normalized = alias.casefold()
            if any(alias_normalized in haystack for haystack in normalized_haystacks):
                return {
                    "collection": descriptor["collection"],
                    "author": descriptor["author"],
                }
    return None


def _content_sample_from_metadata(explicit_metadata: Mapping[str, Any]) -> str:
    explicit_sample = str(explicit_metadata.get("content_sample", "") or "").strip()
    if explicit_sample:
        return explicit_sample
    pages = explicit_metadata.get("pages")
    if isinstance(pages, list):
        page_texts: list[str] = []
        for page in pages[:3]:
            if not isinstance(page, Mapping):
                continue
            text = str(page.get("text", "") or "").strip()
            if text:
                page_texts.append(text)
            if len(" ".join(page_texts)) >= 400:
                break
        return " ".join(page_texts)[:600]
    return ""


def _looks_english_filename(lowered_name: str) -> bool:
    english_markers = (
        "translation",
        "english",
        "vol.",
        "vol ",
        "volume",
        "handbook",
        "book of",
        "sunan ",
        "sahih ",
        "imam ",
    )
    return any(marker in lowered_name for marker in english_markers)


def _infer_language_from_sample(content_sample: str) -> str:
    sample = _clean_metadata_text(content_sample)
    if not sample:
        return ""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", sample))
    latin_chars = len(re.findall(r"[A-Za-z]", sample))
    total_letters = arabic_chars + latin_chars
    if total_letters == 0:
        return ""
    if arabic_chars / total_letters >= 0.3:
        return "ar"
    if latin_chars / total_letters >= 0.6:
        return "en"
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}
