"""Load indexable local corpus documents from the existing data folders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.retrieval.admission import (
    apply_admission_policy,
    evaluate_document_admission,
    serialize_admission_decision,
)
from app.retrieval.metadata_normalizer import normalize_document_metadata
from app.retrieval.source_enrichment import apply_collection_enrichment as _apply_collection_enrichment

RAW_TEXT_EXTENSIONS = {".txt", ".csv", ".md"}
NORMALIZED_EXTENSIONS = {".txt", ".csv", ".md", ".json"}
MADHHAB_IDS = {"hanafi", "shafii", "maliki", "hanbali"}


@dataclass(slots=True)
class LoadedDocument:
    document_id: str
    title: str
    source_path: str
    absolute_path: str
    source_family: str
    source_type: str | None
    collection: str
    author: str
    madhhab: str
    language: str
    loader_hint: str
    extension: str
    content: str
    source_classification: str = ""
    source_role_flag: str = ""
    canonical_family: str = ""
    book: str = ""
    chapter: str = ""
    section: str = ""
    document_kind: str = ""
    commentary_target: str = ""
    fatwa_authority: str = ""
    source_role_boundary: str = ""
    source_lineage: str = ""
    role: str = ""
    domain: str = ""
    authority_level: str = ""
    provider: str = ""
    source_url: str = ""
    topic: str = ""
    copyright_status: str = ""
    allowed_use: str = ""
    extraction_status: str = ""
    extraction_quality: str = ""
    ocr_derived: bool = False
    ocr_backend: str = ""
    ocr_status: str = ""
    ocr_confidence: str = ""
    ocr_confidence_band: str = ""
    text_source_mix: str = ""


@dataclass(slots=True)
class CorpusBootstrap:
    documents: list[LoadedDocument]
    skipped_files: list[dict[str, str]]
    admission_records: list[dict[str, Any]] = field(default_factory=list)


def bootstrap_local_corpus(repo_root: Path) -> CorpusBootstrap:
    return bootstrap_local_corpus_with_paths(repo_root)


def bootstrap_local_corpus_with_paths(
    repo_root: Path,
    *,
    raw_root: Path | None = None,
    normalized_root: Path | None = None,
) -> CorpusBootstrap:
    documents: list[LoadedDocument] = []
    skipped_files: list[dict[str, str]] = []
    prefiltered_admission_records: list[dict[str, Any]] = []

    for path in _iter_candidate_files(
        repo_root,
        raw_root=raw_root,
        normalized_root=normalized_root,
    ):
        relative_path = path.relative_to(repo_root)
        content = _read_text_file(path)
        loader_hint = _infer_loader_hint(None, relative_path)

        if loader_hint == "normalized_pdf_json":
            loaded_document, skip_reason, prefiltered_admission = _load_normalized_pdf_document(
                path,
                content,
            )
            if loaded_document is not None:
                documents.append(loaded_document)
            else:
                skipped_files.append(
                    {
                        "source_path": relative_path.as_posix(),
                        "reason": (
                            str(prefiltered_admission.get("reason", "") or skip_reason)
                            if prefiltered_admission is not None
                            else skip_reason
                        ),
                        "status": (
                            str(prefiltered_admission.get("status", "") or "reject")
                            if prefiltered_admission is not None
                            else "reject"
                        ),
                    }
                )
                if prefiltered_admission is not None:
                    prefiltered_admission_records.append(prefiltered_admission)
            continue
        if loader_hint == "normalized_transcript_json":
            loaded_document, skip_reason = _load_normalized_transcript_document(
                path,
                content,
            )
            if loaded_document is not None:
                documents.append(loaded_document)
            else:
                skipped_files.append(
                    {
                        "source_path": relative_path.as_posix(),
                        "reason": skip_reason,
                        "status": "reject",
                    }
                )
            continue
        if loader_hint == "normalized_class_json":
            loaded_document, skip_reason = _load_normalized_class_document(
                path,
                content,
            )
            if loaded_document is not None:
                documents.append(loaded_document)
            else:
                skipped_files.append(
                    {
                        "source_path": relative_path.as_posix(),
                        "reason": skip_reason,
                        "status": "reject",
                    }
                )
            continue

        metadata = infer_document_metadata(
            relative_path,
            explicit_metadata={
                "loader_hint": loader_hint,
                "content_sample": content[:600],
            },
        )
        if not metadata["source_type"]:
            prefiltered_admission_records.append(
                serialize_admission_decision(
                    evaluate_document_admission(
                        SimpleNamespace(
                            source_path=relative_path.as_posix(),
                            content=content,
                            source_type="",
                            source_classification=metadata["source_classification"],
                            source_family=metadata["source_family"],
                            source_role_boundary=metadata["source_role_boundary"],
                            document_kind=metadata["document_kind"],
                            language=metadata["language"],
                            collection=metadata["collection"],
                            source_lineage=metadata["source_lineage"],
                            madhhab=metadata["madhhab"],
                            author=metadata["author"],
                        )
                    )
                )
            )
            skipped_files.append(
                {
                    "source_path": relative_path.as_posix(),
                    "reason": "unknown_source_type",
                }
            )
            continue
        if not content.strip():
            prefiltered_admission_records.append(
                serialize_admission_decision(
                    evaluate_document_admission(
                        SimpleNamespace(
                            source_path=relative_path.as_posix(),
                            content="",
                            source_type=metadata["source_type"],
                            source_classification=metadata["source_classification"],
                            source_family=metadata["source_family"],
                            source_role_boundary=metadata["source_role_boundary"],
                            document_kind=metadata["document_kind"],
                            language=metadata["language"],
                            collection=metadata["collection"],
                            source_lineage=metadata["source_lineage"],
                            madhhab=metadata["madhhab"],
                            author=metadata["author"],
                        )
                    )
                )
            )
            skipped_files.append(
                {
                    "source_path": relative_path.as_posix(),
                    "reason": "empty_content",
                }
            )
            continue
        documents.append(
            LoadedDocument(
                document_id=relative_path.as_posix(),
                title=metadata["title"],
                source_path=relative_path.as_posix(),
                absolute_path=str(path),
                source_family=metadata["source_family"],
                source_type=metadata["source_type"],
                collection=metadata["collection"],
                author=metadata["author"],
                madhhab=metadata["madhhab"],
                language=metadata["language"],
                loader_hint=metadata["loader_hint"],
                extension=path.suffix.lower(),
                content=content,
                source_classification=metadata["source_classification"],
                source_role_flag=metadata["source_role_flag"],
                canonical_family=metadata["canonical_family"],
                book=metadata["book"],
                chapter=metadata["chapter"],
                section=metadata["section"],
                document_kind=metadata["document_kind"],
                commentary_target=metadata["commentary_target"],
                fatwa_authority=metadata["fatwa_authority"],
                source_role_boundary=metadata["source_role_boundary"],
                source_lineage=metadata["source_lineage"],
                role=str(metadata.get("role", "") or ""),
                domain=str(metadata.get("domain", "") or ""),
                authority_level=str(metadata.get("authority_level", "") or ""),
            )
        )

    admitted_documents, admission_decisions = apply_admission_policy(documents)
    for decision in admission_decisions:
        if decision.status in {"admit", "admit_with_warnings"}:
            continue
        skipped_files.append(
            {
                "source_path": decision.source_path,
                "reason": decision.reason,
                "status": decision.status,
            }
        )

    return CorpusBootstrap(
        documents=admitted_documents,
        skipped_files=skipped_files,
        admission_records=prefiltered_admission_records
        + [serialize_admission_decision(decision) for decision in admission_decisions],
    )
def _iter_candidate_files(
    repo_root: Path,
    *,
    raw_root: Path | None = None,
    normalized_root: Path | None = None,
) -> list[Path]:
    files: list[Path] = []
    effective_raw_root = raw_root or (repo_root / "data" / "raw")
    effective_normalized_root = normalized_root or (repo_root / "data" / "processed" / "normalized")
    effective_classes_root = repo_root / "data" / "processed" / "classes"
    if effective_raw_root.exists():
        for path in effective_raw_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in RAW_TEXT_EXTENSIONS:
                continue
            if _should_skip_raw_text_candidate(
                path,
                repo_root=repo_root,
                normalized_root=effective_normalized_root,
                classes_root=effective_classes_root,
            ):
                continue
            files.append(path)
    if effective_normalized_root.exists():
        files.extend(
            path
            for path in effective_normalized_root.rglob("*")
            if path.is_file() and path.suffix.lower() in NORMALIZED_EXTENSIONS
        )
    if effective_classes_root.exists():
        files.extend(
            path
            for path in effective_classes_root.rglob("*")
            if path.is_file() and path.suffix.lower() in NORMALIZED_EXTENSIONS
        )
    return sorted(files)


def infer_document_metadata(
    relative_path: Path,
    explicit_metadata: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    normalized = normalize_document_metadata(relative_path, explicit_metadata=explicit_metadata)
    loader_hint = _infer_loader_hint(normalized["source_family"], relative_path)
    merged = {**normalized, "loader_hint": loader_hint}
    # Collection-level enrichment attaches era / scholar_authority /
    # methodology_tags / (conservatively) default_hadith_grade so the
    # Trust Engine and Evidence Ladder have signals to work with.
    # Explicit values are never overridden — enrichment is a backstop.
    return _apply_collection_enrichment(merged)


def _infer_loader_hint(source_family: str | None, relative_path: Path) -> str:
    extension = relative_path.suffix.lower()
    lowered_name = relative_path.name.lower()
    normalized_parts = relative_path.as_posix().split("/")
    if (
        extension == ".json"
        and len(normalized_parts) >= 4
        and normalized_parts[:4] == ["data", "processed", "normalized", "pdf"]
    ):
        return "normalized_pdf_json"
    if (
        extension == ".json"
        and len(normalized_parts) >= 4
        and normalized_parts[:4] == ["data", "processed", "normalized", "scholars"]
    ):
        return "normalized_transcript_json"
    if (
        extension == ".json"
        and len(normalized_parts) >= 3
        and normalized_parts[:3] == ["data", "processed", "classes"]
    ):
        return "normalized_class_json"
    if source_family == "quran" and extension == ".txt" and "quran-simple" in lowered_name:
        return "quran_ayah_text"
    if source_family == "quran" and extension == ".csv":
        return "quran_translation_csv"
    if extension == ".csv":
        return "csv_rows"
    return "plain_text"


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"could not decode {path}",
    )


def _load_normalized_pdf_document(
    path: Path,
    content: str,
) -> tuple[LoadedDocument | None, str, dict[str, Any] | None]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None, "invalid_json", None

    if payload.get("document_type") != "normalized_pdf":
        return None, "unknown_json_format", None
    extraction_status = str(payload.get("extraction_status", "")).strip()
    source_path = str(payload.get("source_path", "")).strip()
    if extraction_status not in {"success", "partial"}:
        return (
            None,
            f"pdf_extraction_{extraction_status or 'failed'}",
            _prefiltered_admission_from_payload(
                path,
                content,
                payload,
            ),
        )
    pages = payload.get("pages")
    if not isinstance(pages, list) or not any(page.get("text") for page in pages if isinstance(page, dict)):
        return (
            None,
            "pdf_without_usable_pages",
            _prefiltered_admission_from_payload(
                path,
                content,
                payload,
                override_status="reject",
                override_reason="pdf_without_usable_pages",
            ),
        )

    inferred_metadata = infer_document_metadata(Path(source_path)) if source_path else {}
    normalized_metadata = normalize_document_metadata(
        source_path or path,
        explicit_metadata={
            **payload,
            "loader_hint": "normalized_pdf_json",
        },
    )
    source_type = str(payload.get("source_type", "")).strip() or normalized_metadata.get(
        "source_type", ""
    )
    if not source_type:
        return (
            None,
            "unknown_source_type",
            _prefiltered_admission_from_payload(path, content, payload),
        )

    return (
        LoadedDocument(
            document_id=str(payload.get("normalized_json_path", path.as_posix())).strip()
            or path.as_posix(),
            title=normalized_metadata["title"],
            source_path=source_path or path.as_posix(),
            absolute_path=str(path),
            source_family=normalized_metadata["source_family"]
            or str(inferred_metadata.get("source_family", "")),
            source_type=source_type,
            collection=normalized_metadata["collection"]
            or str(inferred_metadata.get("collection", "")),
            author=normalized_metadata["author"],
            madhhab=normalized_metadata["madhhab"]
            or str(inferred_metadata.get("madhhab", "")),
            language=normalized_metadata["language"]
            or str(inferred_metadata.get("language", "")),
            loader_hint="normalized_pdf_json",
            extension=path.suffix.lower(),
            content=content,
            source_classification=normalized_metadata["source_classification"],
            source_role_flag=normalized_metadata["source_role_flag"],
            canonical_family=normalized_metadata["canonical_family"],
            book=normalized_metadata["book"],
            chapter=normalized_metadata["chapter"],
            section=normalized_metadata["section"],
            document_kind=normalized_metadata["document_kind"],
            commentary_target=normalized_metadata["commentary_target"],
            fatwa_authority=normalized_metadata["fatwa_authority"],
            source_role_boundary=normalized_metadata["source_role_boundary"],
            source_lineage=normalized_metadata["source_lineage"],
            role=str(payload.get("role", "") or normalized_metadata.get("role", "")),
            domain=str(payload.get("domain", "") or normalized_metadata.get("domain", "")),
            authority_level=str(
                payload.get("authority_level", "")
                or normalized_metadata.get("authority_level", "")
            ),
            extraction_status=extraction_status,
            extraction_quality=str(payload.get("extraction_quality", "") or ""),
            ocr_derived=bool(payload.get("ocr_derived", False)),
            ocr_backend=str(payload.get("ocr_backend", "") or ""),
            ocr_status=str(payload.get("ocr_status", "") or ""),
            ocr_confidence=str(payload.get("ocr_confidence", "") or ""),
            ocr_confidence_band=str(payload.get("ocr_confidence_band", "") or ""),
            text_source_mix=str(payload.get("text_source_mix", "") or ""),
        ),
        "",
        None,
    )


def _prefiltered_admission_from_payload(
    path: Path,
    content: str,
    payload: dict[str, Any],
    *,
    override_status: str | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    source_path = str(payload.get("source_path", "") or path.as_posix())
    normalized_metadata = normalize_document_metadata(
        source_path or path,
        explicit_metadata={
            **payload,
            "loader_hint": "normalized_pdf_json",
        },
    )
    decision = evaluate_document_admission(
        SimpleNamespace(
            source_path=source_path,
            content=content,
            loader_hint="normalized_pdf_json",
            source_type=str(payload.get("source_type", "") or normalized_metadata.get("source_type", "")),
            source_classification=str(
                payload.get("source_classification", "")
                or normalized_metadata.get("source_classification", "")
            ),
            source_family=str(
                payload.get("source_family", "") or normalized_metadata.get("source_family", "")
            ),
            source_role_boundary=str(
                payload.get("source_role_boundary", "")
                or normalized_metadata.get("source_role_boundary", "")
            ),
            document_kind=str(
                payload.get("document_kind", "")
                or normalized_metadata.get("document_kind", "")
            ),
            extraction_status=str(payload.get("extraction_status", "") or ""),
            extraction_quality=str(payload.get("extraction_quality", "") or ""),
            ocr_status=str(payload.get("ocr_status", "") or ""),
            ocr_derived=bool(payload.get("ocr_derived", False)),
            ocr_confidence=str(payload.get("ocr_confidence", "") or ""),
            ocr_confidence_band=str(payload.get("ocr_confidence_band", "") or ""),
            text_source_mix=str(payload.get("text_source_mix", "") or ""),
            language=str(payload.get("language", "") or normalized_metadata.get("language", "")),
            collection=str(payload.get("collection", "") or normalized_metadata.get("collection", "")),
            source_lineage=str(
                payload.get("source_lineage", "")
                or normalized_metadata.get("source_lineage", "")
            ),
            madhhab=str(payload.get("madhhab", "") or normalized_metadata.get("madhhab", "")),
            author=str(payload.get("author", "") or normalized_metadata.get("author", "")),
        )
    )
    if override_status:
        decision.status = override_status
    if override_reason:
        decision.reason = override_reason
    return serialize_admission_decision(decision)


def _load_normalized_transcript_document(
    path: Path,
    content: str,
) -> tuple[LoadedDocument | None, str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if payload.get("document_type") != "normalized_transcript":
        return None, "unknown_json_format"

    content_text = str(payload.get("content_text", "") or payload.get("text", "") or "").strip()
    if not content_text:
        return None, "transcript_without_usable_text"

    source_path = str(payload.get("source_path", "")).strip() or path.as_posix()
    normalized_metadata = normalize_document_metadata(
        source_path or path,
        explicit_metadata={
            **payload,
            "loader_hint": "normalized_transcript_json",
            "content_sample": content_text[:600],
        },
    )
    source_type = str(payload.get("source_type", "")).strip() or normalized_metadata.get(
        "source_type", ""
    )
    if not source_type:
        return None, "unknown_source_type"

    return (
        LoadedDocument(
            document_id=str(payload.get("normalized_json_path", path.as_posix())).strip()
            or path.as_posix(),
            title=normalized_metadata["title"],
            source_path=source_path,
            absolute_path=str(path),
            source_family=normalized_metadata["source_family"],
            source_type=source_type,
            collection=normalized_metadata["collection"],
            author=normalized_metadata["author"],
            madhhab=normalized_metadata["madhhab"],
            language=normalized_metadata["language"],
            loader_hint="normalized_transcript_json",
            extension=path.suffix.lower(),
            content=content_text,
            source_classification=normalized_metadata["source_classification"],
            source_role_flag=normalized_metadata["source_role_flag"],
            canonical_family=normalized_metadata["canonical_family"],
            book=normalized_metadata["book"],
            chapter=normalized_metadata["chapter"],
            section=normalized_metadata["section"],
            document_kind=normalized_metadata["document_kind"],
            commentary_target=normalized_metadata["commentary_target"],
            fatwa_authority=normalized_metadata["fatwa_authority"],
            source_role_boundary=normalized_metadata["source_role_boundary"],
            source_lineage=normalized_metadata["source_lineage"],
            role=str(payload.get("role", "") or normalized_metadata.get("role", "")),
            domain=str(payload.get("domain", "") or normalized_metadata.get("domain", "")),
            authority_level=str(
                payload.get("authority_level", "")
                or normalized_metadata.get("authority_level", "")
            ),
            provider=str(payload.get("provider", "") or ""),
            source_url=str(payload.get("url", "") or payload.get("source_url", "") or ""),
            topic=str(payload.get("topic", "") or ""),
            copyright_status=str(payload.get("copyright_status", "") or ""),
            allowed_use=str(payload.get("allowed_use", "") or ""),
            extraction_status=str(payload.get("extraction_status", "") or "success"),
            extraction_quality=str(
                payload.get("extraction_quality", "") or "normalized_transcript_text"
            ),
            ocr_derived=bool(payload.get("ocr_derived", False)),
            ocr_backend=str(payload.get("ocr_backend", "") or ""),
            ocr_status=str(payload.get("ocr_status", "") or "not_applicable"),
            ocr_confidence=str(payload.get("ocr_confidence", "") or ""),
            ocr_confidence_band=str(payload.get("ocr_confidence_band", "") or ""),
            text_source_mix=str(payload.get("text_source_mix", "") or ""),
        ),
        "",
    )


def _load_normalized_class_document(
    path: Path,
    content: str,
) -> tuple[LoadedDocument | None, str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if payload.get("document_type") != "class_material":
        return None, "unknown_json_format"

    content_text = str(payload.get("content_text", "") or payload.get("text", "") or "").strip()
    if not content_text:
        return None, "class_material_without_usable_text"

    source_path = str(payload.get("source_path", "")).strip() or path.as_posix()
    normalized_metadata = normalize_document_metadata(
        source_path or path,
        explicit_metadata={
            **payload,
            "source_family": "classes",
            "source_type": "transcript",
            "source_classification": "transcript",
            "loader_hint": "normalized_class_json",
            "content_sample": content_text[:600],
        },
    )
    source_type = str(payload.get("source_type", "")).strip() or normalized_metadata.get(
        "source_type", ""
    )
    if not source_type:
        return None, "unknown_source_type"

    return (
        LoadedDocument(
            document_id=str(payload.get("normalized_json_path", path.as_posix())).strip()
            or path.as_posix(),
            title=normalized_metadata["title"],
            source_path=source_path,
            absolute_path=str(path),
            source_family=normalized_metadata["source_family"],
            source_type=source_type,
            collection=normalized_metadata["collection"],
            author=normalized_metadata["author"],
            madhhab=normalized_metadata["madhhab"],
            language=normalized_metadata["language"],
            loader_hint="normalized_class_json",
            extension=path.suffix.lower(),
            content=content_text,
            source_classification=normalized_metadata["source_classification"],
            source_role_flag=normalized_metadata["source_role_flag"],
            canonical_family=normalized_metadata["canonical_family"],
            book=normalized_metadata["book"],
            chapter=normalized_metadata["chapter"],
            section=normalized_metadata["section"],
            document_kind=normalized_metadata["document_kind"],
            commentary_target=normalized_metadata["commentary_target"],
            fatwa_authority=normalized_metadata["fatwa_authority"],
            source_role_boundary=normalized_metadata["source_role_boundary"],
            source_lineage=normalized_metadata["source_lineage"],
            role=str(payload.get("role", "") or normalized_metadata.get("role", "")),
            domain=str(payload.get("domain", "") or normalized_metadata.get("domain", "")),
            authority_level=str(
                payload.get("authority_level", "")
                or normalized_metadata.get("authority_level", "")
            ),
            provider=str(payload.get("provider", "") or ""),
            source_url=str(payload.get("url", "") or payload.get("source_url", "") or ""),
            topic=str(payload.get("topic", "") or ""),
            copyright_status=str(payload.get("copyright_status", "") or ""),
            allowed_use=str(payload.get("allowed_use", "") or ""),
            extraction_status=str(payload.get("extraction_status", "") or "success"),
            extraction_quality=str(
                payload.get("extraction_quality", "") or "normalized_class_text"
            ),
            ocr_derived=bool(payload.get("ocr_derived", False)),
            ocr_backend=str(payload.get("ocr_backend", "") or ""),
            ocr_status=str(payload.get("ocr_status", "") or "not_applicable"),
            ocr_confidence=str(payload.get("ocr_confidence", "") or ""),
            ocr_confidence_band=str(payload.get("ocr_confidence_band", "") or ""),
            text_source_mix=str(payload.get("text_source_mix", "") or ""),
        ),
        "",
    )


def _should_skip_raw_text_candidate(
    path: Path,
    *,
    repo_root: Path,
    normalized_root: Path,
    classes_root: Path,
) -> bool:
    try:
        relative_path = path.relative_to(repo_root)
    except ValueError:
        return False

    normalized_parts = relative_path.as_posix().split("/")
    if len(normalized_parts) < 4 or normalized_parts[:3] != ["data", "raw", "scholars"]:
        if len(normalized_parts) >= 4 and normalized_parts[:3] == ["data", "raw", "classes"]:
            output_path = classes_root / Path(*normalized_parts[3:])
            return output_path.with_suffix(".json").exists()
        return False

    output_path = normalized_root / "scholars" / Path(*normalized_parts[3:])
    output_path = output_path.with_suffix(".json")
    return output_path.exists()
