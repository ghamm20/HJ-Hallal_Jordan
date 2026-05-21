"""Normalize cleaned scholar transcript text files into explicit JSON records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.chunker import chunk_documents_with_audit
from app.retrieval.index_loader import bootstrap_local_corpus_with_paths
from app.retrieval.metadata_normalizer import (
    SCHOLAR_AUTHOR_BY_FOLDER,
    SCHOLAR_COLLECTION_BY_FOLDER,
    normalize_document_metadata,
)


@dataclass(slots=True)
class ScholarTranscriptNormalizationResult:
    source_path: str
    normalized_json_path: str
    title: str
    author: str
    collection: str
    language: str
    series_key: str
    speaker_key: str
    characters: int
    line_count: int
    manifest_title: str | None = None
    manifest_video_id: str | None = None


def process_scholar_transcript_corpus(
    root: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    repo_root = root or REPO_ROOT
    raw_root = repo_root / "data" / "raw" / "scholars"
    normalized_root = repo_root / "data" / "processed" / "normalized" / "scholars"
    manifest_index = _load_manifest_index(raw_root, repo_root=repo_root)

    results: list[ScholarTranscriptNormalizationResult] = []
    skipped: list[dict[str, str]] = []
    for source_path in _iter_active_transcript_files(raw_root):
        try:
            result = normalize_transcript_file(
                repo_root,
                source_path,
                normalized_root=normalized_root,
                manifest_index=manifest_index,
                force=force,
            )
        except Exception as exc:  # pragma: no cover - defensive IO path
            skipped.append(
                {
                    "source_path": source_path.relative_to(repo_root).as_posix(),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if result is None:
            skipped.append(
                {
                    "source_path": source_path.relative_to(repo_root).as_posix(),
                    "reason": "empty_content",
                }
            )
            continue
        results.append(result)

    ingestion_summary = build_scholar_transcript_ingestion_summary(
        repo_root,
        normalized_results=results,
        skipped=skipped,
    )
    return {
        "normalized_results": [asdict(item) for item in results],
        "skipped": skipped,
        "summary": ingestion_summary,
    }


def normalize_transcript_file(
    repo_root: Path,
    source_path: Path,
    *,
    normalized_root: Path,
    manifest_index: dict[str, dict[str, Any]],
    force: bool = False,
) -> ScholarTranscriptNormalizationResult | None:
    relative_source_path = source_path.relative_to(repo_root)
    content = _read_text_file(source_path)
    normalized_content = _normalize_transcript_text(content)
    if not normalized_content:
        return None

    manifest_entry = manifest_index.get(relative_source_path.as_posix(), {})
    title = _derive_transcript_title(source_path, manifest_entry)
    explicit_metadata = {
        "loader_hint": "normalized_transcript_json",
        "source_family": "scholars",
        "source_type": "scholar_transcript",
        "source_classification": "scholar_transcript",
        "role": "commentary",
        "domain": "teaching",
        "authority_level": "modern",
        "document_kind": "transcript",
        "source_role_boundary": "transcript",
        "source_role_flag": "informal_explanation",
        "source_lineage": "normalized_transcript:text_file:success",
        "title": title,
        "content_sample": normalized_content[:600],
    }
    metadata = normalize_document_metadata(relative_source_path, explicit_metadata=explicit_metadata)
    normalized_json_path = _build_output_path(repo_root, relative_source_path)
    normalized_json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "document_type": "normalized_transcript",
        "parser": "text_file",
        "loader_hint": "normalized_transcript_json",
        "source_path": relative_source_path.as_posix(),
        "normalized_json_path": normalized_json_path.relative_to(repo_root).as_posix(),
        "title": metadata["title"],
        "source_family": metadata["source_family"],
        "source_type": metadata["source_type"],
        "source_classification": metadata["source_classification"],
        "source_role_flag": metadata["source_role_flag"],
        "source_role_boundary": metadata["source_role_boundary"],
        "source_lineage": metadata["source_lineage"],
        "canonical_family": metadata["canonical_family"],
        "collection": metadata["collection"],
        "author": metadata["author"],
        "madhhab": metadata["madhhab"],
        "language": metadata["language"],
        "book": metadata["book"],
        "chapter": metadata["chapter"],
        "section": metadata["section"],
        "document_kind": metadata["document_kind"],
        "commentary_target": metadata["commentary_target"],
        "fatwa_authority": metadata["fatwa_authority"],
        "role": metadata["role"],
        "domain": metadata["domain"],
        "authority_level": metadata["authority_level"],
        "content_text": normalized_content,
        "character_count": len(normalized_content),
        "line_count": len([line for line in normalized_content.splitlines() if line.strip()]),
        "series_key": _series_key_from_path(relative_source_path),
        "series_label": _series_label_from_path(relative_source_path),
        "speaker_key": _speaker_key_from_path(relative_source_path),
        "manifest_title": manifest_entry.get("title"),
        "manifest_video_id": manifest_entry.get("video_id"),
        "manifest_index": manifest_entry.get("index"),
        "warnings": [],
        "extraction_status": "success",
        "extraction_quality": "normalized_transcript_text",
        "ocr_derived": False,
        "ocr_backend": "",
        "ocr_status": "not_applicable",
        "ocr_confidence": "",
    }
    if force or not normalized_json_path.exists():
        normalized_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return ScholarTranscriptNormalizationResult(
        source_path=relative_source_path.as_posix(),
        normalized_json_path=normalized_json_path.relative_to(repo_root).as_posix(),
        title=metadata["title"],
        author=metadata["author"],
        collection=metadata["collection"],
        language=metadata["language"],
        series_key=payload["series_key"],
        speaker_key=payload["speaker_key"],
        characters=payload["character_count"],
        line_count=payload["line_count"],
        manifest_title=payload.get("manifest_title"),
        manifest_video_id=payload.get("manifest_video_id"),
    )


def build_scholar_transcript_ingestion_summary(
    repo_root: Path,
    *,
    normalized_results: list[ScholarTranscriptNormalizationResult],
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    corpus = bootstrap_local_corpus_with_paths(repo_root)
    scholar_documents = [
        document
        for document in corpus.documents
        if str(getattr(document, "source_classification", "") or "") == "scholar_transcript"
    ]
    scholar_chunks = chunk_documents_with_audit(scholar_documents).chunks
    return {
        "generated_at": _utc_now(),
        "normalized_transcript_count": len(normalized_results),
        "skipped_count": len(skipped),
        "indexed_scholar_document_count": len(scholar_documents),
        "indexed_scholar_chunk_count": len(scholar_chunks),
        "by_author": dict(Counter(document.author or "unknown" for document in scholar_documents)),
        "by_collection": dict(
            Counter(document.collection or "unknown" for document in scholar_documents)
        ),
        "by_source_path_root": dict(
            Counter(_speaker_key_from_source_path(document.source_path) for document in scholar_documents)
        ),
        "normalized_output_root": str(repo_root / "data" / "processed" / "normalized" / "scholars"),
        "active_raw_root": str(repo_root / "data" / "raw" / "scholars"),
    }


def write_scholar_transcript_ingestion_report(
    summary: dict[str, Any],
    normalized_results: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    repo_root: Path | None = None,
) -> str:
    root = repo_root or REPO_ROOT
    readiness_dir = root / "docs" / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    report_path = readiness_dir / "scholar-transcript-ingestion-report.md"
    report_path.write_text(
        _render_ingestion_report(summary, normalized_results, skipped),
        encoding="utf-8",
    )
    return str(report_path)


def _render_ingestion_report(
    summary: dict[str, Any],
    normalized_results: list[dict[str, Any]],
    skipped: list[dict[str, str]],
) -> str:
    lines = [
        "# Scholar Transcript Ingestion Report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Summary",
        f"- Normalized transcript files written: `{summary['normalized_transcript_count']}`",
        f"- Indexed scholar transcript documents: `{summary['indexed_scholar_document_count']}`",
        f"- Indexed scholar transcript chunks: `{summary['indexed_scholar_chunk_count']}`",
        f"- Skipped files: `{summary['skipped_count']}`",
        f"- Active raw root: `{summary['active_raw_root']}`",
        f"- Normalized output root: `{summary['normalized_output_root']}`",
        "",
        "## By Author",
    ]
    lines.extend(
        f"- `{key}`: `{value}`" for key, value in summary["by_author"].items()
    )
    lines.append("")
    lines.append("## By Collection")
    lines.extend(
        f"- `{key}`: `{value}`" for key, value in summary["by_collection"].items()
    )
    lines.append("")
    lines.append("## By Canonical Folder")
    lines.extend(
        f"- `{key}`: `{value}`"
        for key, value in summary["by_source_path_root"].items()
    )
    lines.append("")
    lines.append("## Example Normalized Outputs")
    if normalized_results:
        for item in normalized_results[:12]:
            lines.append(
                f"- `{item['source_path']}` -> `{item['normalized_json_path']}` "
                f"(author=`{item['author']}`, collection=`{item['collection']}`)"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Skipped")
    if skipped:
        lines.extend(
            f"- `{item['source_path']}`: `{item['reason']}`" for item in skipped
        )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Scholar transcripts are indexed as `scholar_transcript` and render as `Scholar Commentary`.")
    lines.append("- This layer is commentary/teaching context only and does not replace Qur'an, hadith, fiqh manuals, or tasawwuf texts.")
    lines.append("")
    return "\n".join(lines)


def _iter_active_transcript_files(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.rglob("*.txt")
        if path.is_file()
    )


def _load_manifest_index(raw_root: Path, *, repo_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest_path in raw_root.rglob("manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw_file = str(item.get("file", "") or "").strip()
            if not raw_file:
                continue
            source_path = manifest_path.parent / raw_file
            try:
                relative = source_path.relative_to(repo_root)
            except ValueError:
                continue
            index[relative.as_posix()] = item
    return index


def _build_output_path(repo_root: Path, relative_source_path: Path) -> Path:
    relative_parts = relative_source_path.as_posix().split("/")
    scholar_relative = Path(*relative_parts[3:])
    return (repo_root / "data" / "processed" / "normalized" / "scholars" / scholar_relative).with_suffix(".json")


def _derive_transcript_title(source_path: Path, manifest_entry: dict[str, Any]) -> str:
    manifest_title = str(manifest_entry.get("title", "") or "").strip()
    if manifest_title:
        return manifest_title
    stem = source_path.stem
    stem = re.sub(r"^\d+[_\-\s]+", "", stem)
    stem = re.sub(r"[_\s-]+([A-Za-z0-9_-]{8,14})$", "", stem)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -._")
    return stem or source_path.stem


def _normalize_transcript_text(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\ufeff", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"could not decode {path}")


def _series_key_from_path(relative_source_path: Path) -> str:
    parts = relative_source_path.parts
    if len(parts) >= 6:
        return parts[4]
    return "transcripts"


def _series_label_from_path(relative_source_path: Path) -> str:
    return SCHOLAR_COLLECTION_BY_FOLDER.get(_series_key_from_path(relative_source_path), "Scholar Transcript Series")


def _speaker_key_from_path(relative_source_path: Path) -> str:
    parts = relative_source_path.parts
    if len(parts) >= 5:
        return parts[3]
    return "scholars"


def _speaker_key_from_source_path(source_path: str) -> str:
    parts = Path(source_path).parts
    if len(parts) >= 5:
        return parts[3]
    return "scholars"


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize cleaned scholar transcripts into explicit JSON records."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing normalized transcript JSON files")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write docs/readiness/scholar-transcript-ingestion-report.md",
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_arg_parser().parse_args()
    result = process_scholar_transcript_corpus(REPO_ROOT, force=args.force)
    if args.write_report:
        report_path = write_scholar_transcript_ingestion_report(
            result["summary"],
            result["normalized_results"],
            result["skipped"],
            REPO_ROOT,
        )
        print(
            json.dumps(
                {
                    "report_path": report_path,
                    "summary": result["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        _render_ingestion_report(
            result["summary"],
            result["normalized_results"],
            result["skipped"],
        )
    )


if __name__ == "__main__":
    main()
