"""Persist and load retrieval index assets for portable/offline startup."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.retrieval.chunker import ChunkBuildResult, chunk_documents_with_audit
from app.retrieval.index_loader import (
    CorpusBootstrap,
    LoadedDocument,
    NORMALIZED_EXTENSIONS,
    RAW_TEXT_EXTENSIONS,
    bootstrap_local_corpus_with_paths,
)
from app.retrieval.search import prepare_search_index
from app.retrieval.search import export_prepared_search_metadata
from app.retrieval.search import register_prepared_search_index

MANIFEST_FILENAME = "manifest.json"
CHUNKS_FILENAME = "chunks.jsonl"
DOCUMENTS_FILENAME = "documents.json"
SKIPPED_FILENAME = "skipped_files.json"
ADMISSION_FILENAME = "admission_records.json"
PREPARED_SEARCH_MANIFEST_FILENAME = "search_metadata_manifest.json"
TOKEN_POSTINGS_FILENAME = "token_postings.jsonl"
MADHHAB_AUTHORITY_BUCKETS_FILENAME = "madhhab_authority_buckets.json"
SOURCE_CLASS_BUCKETS_FILENAME = "source_class_buckets.json"
PRIMARY_CHUNK_IDS_FILENAME = "primary_chunk_ids.json"
INDEX_FORMAT = "json_manifest+json_documents+json_admission+json_skipped+jsonl_chunks"
PREPARED_SEARCH_FORMAT = (
    "json_manifest+jsonl_token_postings+json_madhhab_buckets+json_source_class_buckets+json_primary_chunk_ids"
)
PREPARED_SEARCH_METADATA_VERSION = 1


def build_persisted_retrieval_index(
    repo_root: Path,
    *,
    raw_root: Path,
    normalized_root: Path,
    index_root: Path,
    manifest_index_root: Path | None = None,
) -> dict[str, Any]:
    corpus = bootstrap_local_corpus_with_paths(
        repo_root,
        raw_root=raw_root,
        normalized_root=normalized_root,
    )
    chunk_result = chunk_documents_with_audit(corpus.documents)
    chunks = prepare_search_index(chunk_result.chunks)
    prepared_search_metadata = export_prepared_search_metadata(chunks)
    fingerprint = compute_corpus_fingerprint(
        repo_root,
        raw_root=raw_root,
        normalized_root=normalized_root,
    )
    index_root.mkdir(parents=True, exist_ok=True)

    documents_path = index_root / DOCUMENTS_FILENAME
    skipped_path = index_root / SKIPPED_FILENAME
    admission_path = index_root / ADMISSION_FILENAME
    chunks_path = index_root / CHUNKS_FILENAME
    manifest_path = index_root / MANIFEST_FILENAME
    prepared_search_manifest_path = index_root / PREPARED_SEARCH_MANIFEST_FILENAME
    token_postings_path = index_root / TOKEN_POSTINGS_FILENAME
    madhhab_buckets_path = index_root / MADHHAB_AUTHORITY_BUCKETS_FILENAME
    source_class_buckets_path = index_root / SOURCE_CLASS_BUCKETS_FILENAME
    primary_chunk_ids_path = index_root / PRIMARY_CHUNK_IDS_FILENAME

    admission_lookup = _admission_lookup(corpus.admission_records)
    documents_payload = [
        _with_admission_fields(asdict(document), admission_lookup.get(document.source_path, {}))
        for document in corpus.documents
    ]
    chunk_payloads = [
        _prepare_chunk_payload(chunk, admission_lookup.get(str(chunk.get("source_path", "") or ""), {}))
        for chunk in chunks
    ]
    documents_path.write_text(
        json.dumps(documents_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    skipped_path.write_text(
        json.dumps(corpus.skipped_files, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    admission_path.write_text(
        json.dumps(corpus.admission_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunk_payloads:
            handle.write(json.dumps(chunk, ensure_ascii=False))
            handle.write("\n")

    admission_summary = _summarize_admission_records(corpus.admission_records)
    source_breakdown = _summarize_documents(corpus.documents)
    manifest = {
        "version": 1,
        "index_format": INDEX_FORMAT,
        "generated_at": _utc_now(),
        "repo_root": ".",
        "raw_root": _relative_manifest_path(raw_root, repo_root),
        "normalized_root": _relative_manifest_path(normalized_root, repo_root),
        "index_root": _relative_manifest_path(manifest_index_root or index_root, repo_root),
        "corpus_fingerprint": fingerprint["fingerprint"],
        "source_file_count": fingerprint["file_count"],
        "raw_file_count": fingerprint["raw_file_count"],
        "normalized_file_count": fingerprint["normalized_file_count"],
        "document_count": len(corpus.documents),
        "skipped_file_count": len(corpus.skipped_files),
        "admission_record_count": len(corpus.admission_records),
        "chunk_count": len(chunk_payloads),
        "chunk_audit": _serialize_chunk_audit(chunk_result),
        "admission_summary": admission_summary,
        "source_breakdown": source_breakdown,
        "files": {
            "manifest": MANIFEST_FILENAME,
            "documents": DOCUMENTS_FILENAME,
            "skipped_files": SKIPPED_FILENAME,
            "admission_records": ADMISSION_FILENAME,
            "chunks": CHUNKS_FILENAME,
        },
    }
    manifest["core_manifest_fingerprint"] = _stable_payload_fingerprint(manifest)

    prepared_search_manifest = {
        "version": PREPARED_SEARCH_METADATA_VERSION,
        "search_metadata_format": PREPARED_SEARCH_FORMAT,
        "generated_at": _utc_now(),
        "core_manifest_fingerprint": manifest["core_manifest_fingerprint"],
        "corpus_fingerprint": fingerprint["fingerprint"],
        "chunk_count": len(chunk_payloads),
        "token_count": int(prepared_search_metadata["token_count"]),
        "authority_madhhab_count": len(
            prepared_search_metadata["authority_chunk_ids_by_madhhab"]
        ),
        "source_class_count": len(
            prepared_search_metadata["source_classification_chunk_ids"]
        ),
        "files": {
            "manifest": PREPARED_SEARCH_MANIFEST_FILENAME,
            "token_postings": TOKEN_POSTINGS_FILENAME,
            "madhhab_authority_buckets": MADHHAB_AUTHORITY_BUCKETS_FILENAME,
            "source_class_buckets": SOURCE_CLASS_BUCKETS_FILENAME,
            "primary_chunk_ids": PRIMARY_CHUNK_IDS_FILENAME,
        },
    }
    with token_postings_path.open("w", encoding="utf-8") as handle:
        for record in prepared_search_metadata["token_postings"]:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    madhhab_buckets_path.write_text(
        json.dumps(
            prepared_search_metadata["authority_chunk_ids_by_madhhab"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    source_class_buckets_path.write_text(
        json.dumps(
            prepared_search_metadata["source_classification_chunk_ids"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    primary_chunk_ids_path.write_text(
        json.dumps(prepared_search_metadata["primary_chunk_ids"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prepared_search_manifest_path.write_text(
        json.dumps(prepared_search_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest["prepared_search"] = {
        "version": PREPARED_SEARCH_METADATA_VERSION,
        "search_metadata_format": PREPARED_SEARCH_FORMAT,
        "manifest_path": PREPARED_SEARCH_MANIFEST_FILENAME,
        "token_count": int(prepared_search_metadata["token_count"]),
        "chunk_count": len(chunk_payloads),
        "generated_at": prepared_search_manifest["generated_at"],
        "loadable": True,
        "fallback_reason": "",
        "files": dict(prepared_search_manifest["files"]),
    }
    manifest["files"].update(
        {
            "prepared_search_manifest": PREPARED_SEARCH_MANIFEST_FILENAME,
            "token_postings": TOKEN_POSTINGS_FILENAME,
            "madhhab_authority_buckets": MADHHAB_AUTHORITY_BUCKETS_FILENAME,
            "source_class_buckets": SOURCE_CLASS_BUCKETS_FILENAME,
            "primary_chunk_ids": PRIMARY_CHUNK_IDS_FILENAME,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["index_size_bytes"] = _folder_size_bytes(index_root)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_persisted_retrieval_index(
    repo_root: Path,
    *,
    raw_root: Path,
    normalized_root: Path,
    index_root: Path,
) -> tuple[CorpusBootstrap, list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None:
    probe = probe_persisted_retrieval_index(
        repo_root,
        raw_root=raw_root,
        normalized_root=normalized_root,
        index_root=index_root,
    )
    if not probe["retrieval_index_loadable"]:
        return None

    document_payloads = probe["document_payloads"]
    skipped_files = probe["skipped_files"]
    admission_records = probe["admission_records"]
    chunks = probe["chunks"]
    manifest = probe["manifest"]
    allowed_fields = set(getattr(LoadedDocument, "__dataclass_fields__", {}).keys())
    documents = [
        LoadedDocument(**{key: value for key, value in payload.items() if key in allowed_fields})
        for payload in document_payloads
    ]
    corpus = CorpusBootstrap(
        documents=documents,
        skipped_files=list(skipped_files or []),
        admission_records=list(admission_records or []),
    )
    prepared_state = _load_or_rebuild_prepared_search_metadata(
        index_root=index_root,
        chunks=chunks,
        manifest=manifest,
    )
    return corpus, chunks, manifest, prepared_state


def inspect_persisted_retrieval_index(
    index_root: Path,
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    normalized_root: Path | None = None,
) -> dict[str, Any]:
    return _inspect_persisted_retrieval_index(
        index_root,
        repo_root=repo_root,
        raw_root=raw_root,
        normalized_root=normalized_root,
    )


def probe_persisted_retrieval_index(
    repo_root: Path,
    *,
    raw_root: Path,
    normalized_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    return _inspect_persisted_retrieval_index(
        index_root,
        repo_root=repo_root,
        raw_root=raw_root,
        normalized_root=normalized_root,
        include_payloads=True,
    )


def _inspect_persisted_retrieval_index(
    index_root: Path,
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    normalized_root: Path | None = None,
    include_payloads: bool = False,
) -> dict[str, Any]:
    manifest_path = index_root / MANIFEST_FILENAME
    documents_path = index_root / DOCUMENTS_FILENAME
    skipped_path = index_root / SKIPPED_FILENAME
    admission_path = index_root / ADMISSION_FILENAME
    chunks_path = index_root / CHUNKS_FILENAME
    required = (manifest_path, documents_path, skipped_path, admission_path, chunks_path)
    size_bytes = _folder_size_bytes(index_root) if index_root.exists() else 0
    manifest: dict[str, Any] = {}
    manifest_valid = False
    document_payloads: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []
    admission_records: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    fallback_reason = ""

    if not index_root.exists():
        fallback_reason = "index_directory_missing"
    elif not all(path.exists() and path.is_file() for path in required):
        fallback_reason = "index_files_missing"
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_valid = True
        except (OSError, json.JSONDecodeError):
            fallback_reason = "manifest_invalid_json"
        if manifest_valid:
            if repo_root is not None and raw_root is not None and normalized_root is not None:
                current_fingerprint = compute_corpus_fingerprint(
                    repo_root,
                    raw_root=raw_root,
                    normalized_root=normalized_root,
                )
                if manifest.get("corpus_fingerprint") != current_fingerprint["fingerprint"]:
                    fallback_reason = "corpus_fingerprint_mismatch"
            if not fallback_reason:
                try:
                    document_payloads = json.loads(documents_path.read_text(encoding="utf-8"))
                    skipped_files = json.loads(skipped_path.read_text(encoding="utf-8"))
                    admission_records = json.loads(admission_path.read_text(encoding="utf-8"))
                    chunks = [
                        json.loads(line)
                        for line in chunks_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                except (OSError, json.JSONDecodeError, TypeError):
                    fallback_reason = "index_payload_invalid"

    if not admission_records:
        admission_records = _safe_load_json_list(admission_path)
    if not document_payloads:
        document_payloads = _safe_load_json_list(documents_path)
    admission_summary = dict(manifest.get("admission_summary") or {})
    if not admission_summary and admission_records:
        admission_summary = _summarize_admission_records(admission_records)
    source_breakdown = dict(manifest.get("source_breakdown") or {})
    if not source_breakdown and document_payloads:
        source_breakdown = _summarize_document_payloads(document_payloads)
    prepared_search = _inspect_prepared_search_assets(
        index_root=index_root,
        manifest=manifest,
        retrieval_index_loadable=bool(manifest_valid and not fallback_reason),
    )

    result = {
        "index_present": bool(index_root.exists() and any(index_root.iterdir()))
        if index_root.exists()
        else False,
        "manifest_valid": manifest_valid,
        "retrieval_index_loadable": manifest_valid and not fallback_reason,
        "retrieval_bootstrap_source": (
            "persisted_index" if manifest_valid and not fallback_reason else "fallback_normalized_corpus"
        ),
        "retrieval_fallback_reason": fallback_reason,
        "index_root": str(index_root),
        "manifest_path": str(manifest_path),
        "chunks_path": str(chunks_path),
        "chunk_count": int(manifest.get("chunk_count") or len(chunks) or 0),
        "document_count": int(manifest.get("document_count") or len(document_payloads) or 0),
        "generated_at": manifest.get("generated_at"),
        "corpus_fingerprint": manifest.get("corpus_fingerprint"),
        "index_size_bytes": size_bytes,
        "index_format": str(manifest.get("index_format") or INDEX_FORMAT),
        "core_manifest_fingerprint": str(manifest.get("core_manifest_fingerprint") or ""),
        "admission_summary": admission_summary,
        "source_breakdown": source_breakdown,
        **prepared_search,
    }
    if include_payloads:
        result["manifest"] = manifest
        result["document_payloads"] = document_payloads
        result["skipped_files"] = skipped_files
        result["admission_records"] = admission_records
        result["chunks"] = chunks
    return {
        **result,
    }


def _inspect_prepared_search_assets(
    *,
    index_root: Path,
    manifest: dict[str, Any],
    retrieval_index_loadable: bool,
) -> dict[str, Any]:
    manifest_path = index_root / PREPARED_SEARCH_MANIFEST_FILENAME
    token_postings_path = index_root / TOKEN_POSTINGS_FILENAME
    madhhab_buckets_path = index_root / MADHHAB_AUTHORITY_BUCKETS_FILENAME
    source_class_buckets_path = index_root / SOURCE_CLASS_BUCKETS_FILENAME
    primary_chunk_ids_path = index_root / PRIMARY_CHUNK_IDS_FILENAME
    required = (
        manifest_path,
        token_postings_path,
        madhhab_buckets_path,
        source_class_buckets_path,
        primary_chunk_ids_path,
    )
    prepared_manifest: dict[str, Any] = {}
    prepared_manifest_valid = False
    fallback_reason = ""

    if not retrieval_index_loadable:
        fallback_reason = "retrieval_index_unavailable"
    elif not manifest_path.exists():
        fallback_reason = "prepared_search_manifest_missing"
    elif not all(path.exists() and path.is_file() for path in required):
        fallback_reason = "prepared_search_files_missing"
    else:
        try:
            prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prepared_manifest_valid = True
        except (OSError, json.JSONDecodeError):
            fallback_reason = "prepared_search_manifest_invalid"
        if prepared_manifest_valid:
            if (
                str(prepared_manifest.get("core_manifest_fingerprint") or "")
                != str(manifest.get("core_manifest_fingerprint") or "")
            ):
                fallback_reason = "prepared_search_manifest_fingerprint_mismatch"
            elif int(prepared_manifest.get("chunk_count") or 0) != int(
                manifest.get("chunk_count") or 0
            ):
                fallback_reason = "prepared_search_chunk_count_mismatch"
            elif str(prepared_manifest.get("corpus_fingerprint") or "") != str(
                manifest.get("corpus_fingerprint") or ""
            ):
                fallback_reason = "prepared_search_corpus_fingerprint_mismatch"

    return {
        "prepared_search_present": bool(manifest_path.exists()),
        "prepared_search_manifest_valid": prepared_manifest_valid,
        "prepared_search_loadable": prepared_manifest_valid and not fallback_reason,
        "prepared_search_fallback_reason": fallback_reason,
        "prepared_search_manifest_path": str(manifest_path),
        "prepared_search_token_postings_path": str(token_postings_path),
        "prepared_search_madhhab_buckets_path": str(madhhab_buckets_path),
        "prepared_search_source_class_buckets_path": str(source_class_buckets_path),
        "prepared_search_primary_chunk_ids_path": str(primary_chunk_ids_path),
        "prepared_search_generated_at": prepared_manifest.get("generated_at"),
        "prepared_search_format": str(
            prepared_manifest.get("search_metadata_format") or PREPARED_SEARCH_FORMAT
        ),
        "prepared_search_token_count": int(prepared_manifest.get("token_count") or 0),
        "prepared_search_chunk_count": int(
            prepared_manifest.get("chunk_count") or manifest.get("chunk_count") or 0
        ),
    }


def _load_or_rebuild_prepared_search_metadata(
    *,
    index_root: Path,
    chunks: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    inspection = _inspect_prepared_search_assets(
        index_root=index_root,
        manifest=manifest,
        retrieval_index_loadable=True,
    )
    if inspection["prepared_search_loadable"]:
        try:
            token_postings = _load_token_postings(index_root / TOKEN_POSTINGS_FILENAME)
            madhhab_buckets = _load_bucket_payload(
                index_root / MADHHAB_AUTHORITY_BUCKETS_FILENAME
            )
            source_class_buckets = _load_bucket_payload(
                index_root / SOURCE_CLASS_BUCKETS_FILENAME
            )
            primary_chunk_ids = _load_int_list(index_root / PRIMARY_CHUNK_IDS_FILENAME)
            register_prepared_search_index(
                chunks,
                token_to_chunk_ids=token_postings,
                authority_chunk_ids_by_madhhab=madhhab_buckets,
                source_classification_chunk_ids=source_class_buckets,
                primary_chunk_ids=primary_chunk_ids,
            )
            return {
                "prepared_search_loaded": True,
                "prepared_search_fallback_reason": "",
                "prepared_search_load_ms": int((perf_counter() - started) * 1000),
                "prepared_search_build_ms": 0,
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            inspection["prepared_search_fallback_reason"] = "prepared_search_payload_invalid"

    load_ms = int((perf_counter() - started) * 1000)
    build_started = perf_counter()
    prepare_search_index(chunks)
    return {
        "prepared_search_loaded": False,
        "prepared_search_fallback_reason": str(
            inspection.get("prepared_search_fallback_reason") or "prepared_search_missing"
        ),
        "prepared_search_load_ms": load_ms,
        "prepared_search_build_ms": int((perf_counter() - build_started) * 1000),
    }


def compute_corpus_fingerprint(
    repo_root: Path,
    *,
    raw_root: Path,
    normalized_root: Path,
) -> dict[str, Any]:
    # Inno/ZIP-style Windows installers commonly preserve timestamps with DOS-era
    # 2-second granularity. Normalize to that bucket so packaged persisted indexes
    # remain portable across install/extract locations without disabling drift checks.
    def _portable_mtime_token(path: Path) -> int:
        return (int(path.stat().st_mtime) // 2) * 2

    descriptors: list[str] = []
    raw_count = 0
    normalized_count = 0
    for path in _iter_candidate_files(raw_root, RAW_TEXT_EXTENSIONS):
        relative = path.relative_to(repo_root).as_posix()
        stat = path.stat()
        descriptors.append(f"raw|{relative}|{stat.st_size}|{_portable_mtime_token(path)}")
        raw_count += 1
    for path in _iter_candidate_files(normalized_root, NORMALIZED_EXTENSIONS):
        relative = path.relative_to(repo_root).as_posix()
        stat = path.stat()
        descriptors.append(f"normalized|{relative}|{stat.st_size}|{_portable_mtime_token(path)}")
        normalized_count += 1
    descriptors.sort()
    digest = hashlib.sha256("\n".join(descriptors).encode("utf-8")).hexdigest()
    return {
        "fingerprint": digest,
        "file_count": len(descriptors),
        "raw_file_count": raw_count,
        "normalized_file_count": normalized_count,
    }


def _iter_candidate_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _folder_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _serialize_chunk_audit(chunk_result: ChunkBuildResult) -> dict[str, Any]:
    audit = chunk_result.audit
    return {
        "raw_chunk_count": audit.raw_chunk_count,
        "final_chunk_count": audit.final_chunk_count,
        "suppressed_chunk_count": audit.suppressed_chunk_count,
        "suppressed_by_reason": dict(audit.suppressed_by_reason),
        "split_page_count": audit.split_page_count,
        "split_output_chunk_count": audit.split_output_chunk_count,
        "split_additional_chunk_count": audit.split_additional_chunk_count,
        "suppressed_examples": [
            {
                "source_path": item.source_path,
                "reference": item.reference,
                "reason": item.reason,
                "length": item.length,
                "quote": item.quote,
            }
            for item in audit.suppressed_examples
        ],
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _relative_manifest_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _admission_lookup(admission_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in admission_records:
        source_path = str(record.get("source_path", "") or "").strip()
        status = str(record.get("status", "") or "").strip()
        if not source_path or status not in {"admit", "admit_with_warnings"}:
            continue
        lookup[source_path] = dict(record)
    return lookup


def _with_admission_fields(
    payload: dict[str, Any],
    admission_record: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(payload)
    if admission_record:
        merged["admission_status"] = str(admission_record.get("status", "") or "")
        merged["admission_reason"] = str(admission_record.get("reason", "") or "")
    return merged


def _prepare_chunk_payload(
    chunk: dict[str, Any],
    admission_record: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in chunk.items()
        if not str(key).startswith("_search_")
    }
    normalized_search_text = str(chunk.get("_search_haystack_normalized", "") or "")
    if normalized_search_text and "normalized_search_text" not in payload:
        payload["normalized_search_text"] = normalized_search_text
    if admission_record:
        payload["admission_status"] = str(admission_record.get("status", "") or "")
        payload["admission_reason"] = str(admission_record.get("reason", "") or "")
    return payload


def _summarize_admission_records(admission_records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for record in admission_records:
        status = str(record.get("status", "") or "").strip() or "unknown"
        reason = str(record.get("reason", "") or "").strip() or "unknown"
        status_counts[status] += 1
        reason_counts[reason] += 1
    return {
        "total_candidates": len(admission_records),
        "admitted_count": int(status_counts.get("admit", 0)),
        "admitted_with_warnings_count": int(status_counts.get("admit_with_warnings", 0)),
        "deferred_count": int(status_counts.get("defer", 0)),
        "rejected_count": int(status_counts.get("reject", 0)),
        "warning_count": int(status_counts.get("admit_with_warnings", 0)),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
    }


def _summarize_documents(documents: list[Any]) -> dict[str, dict[str, int]]:
    payloads = [asdict(document) for document in documents]
    return _summarize_document_payloads(payloads)


def _summarize_document_payloads(documents: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "by_source_family": _count_field(documents, "source_family"),
        "by_source_classification": _count_field(documents, "source_classification"),
        "by_source_role_boundary": _count_field(documents, "source_role_boundary"),
        "by_madhhab": _count_field(documents, "madhhab"),
        "by_language": _count_field(documents, "language"),
    }


def _count_field(documents: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for payload in documents:
        value = str(payload.get(field_name, "") or "").strip() or "unknown"
        counter[value] += 1
    return dict(counter)


def _safe_load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _load_token_postings(path: Path) -> dict[str, tuple[int, ...]]:
    postings: dict[str, tuple[int, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("token_posting_record_invalid")
        token = str(payload.get("token") or "").strip()
        if not token:
            raise ValueError("token_posting_missing_token")
        postings[token] = _coerce_tuple_of_ints(payload.get("chunk_ids"))
    return postings


def _load_bucket_payload(path: Path) -> dict[str, tuple[int, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("bucket_payload_invalid")
    return {
        str(key).strip(): _coerce_tuple_of_ints(value)
        for key, value in payload.items()
        if str(key).strip()
    }


def _load_int_list(path: Path) -> tuple[int, ...]:
    return _coerce_tuple_of_ints(json.loads(path.read_text(encoding="utf-8")))


def _coerce_tuple_of_ints(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("expected_list_of_ints")
    return tuple(int(item) for item in value)


def _stable_payload_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
