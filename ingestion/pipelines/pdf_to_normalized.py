"""Normalize machine-readable PDFs into processed JSON and text outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.backend.runtime_config import RuntimeConfigStore
from app.retrieval.admission import evaluate_document_admission
from app.retrieval.chunker import chunk_documents
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.index_loader import infer_document_metadata
from app.retrieval.index_loader import bootstrap_local_corpus_with_paths
from app.retrieval.search import prepare_search_index
from app.retrieval.vector_store import build_vector_index
from ingestion.parsers.pdf_parser import ExtractedPdfPage, PdfParseResult, parse_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PdfNormalizationRecord:
    source_path: str
    normalized_json_path: str
    clean_text_path: str
    extraction_status: str
    warnings: list[str] = field(default_factory=list)
    page_count: int | None = None
    extraction_quality: str = ""
    ocr_status: str = "not_attempted"
    ocr_derived: bool = False
    ocr_confidence: float | None = None
    ocr_confidence_band: str = "not_applicable"
    text_source_mix: str = "none"
    admission_status: str = ""
    admission_reason: str = ""


def process_pdf_corpus(
    repo_root: Path | None = None,
    *,
    force: bool = False,
    allow_ocr: bool = False,
    ocr_backend: Any | None = None,
    tesseract_path: str | None = None,
    rebuild_vector_index: bool = False,
) -> list[PdfNormalizationRecord]:
    root = repo_root or REPO_ROOT
    resolved_tesseract_path = (
        tesseract_path if allow_ocr else None
    ) or _configured_tesseract_path(root)
    records: list[PdfNormalizationRecord] = []
    for pdf_path in sorted((root / "data" / "raw").rglob("*.pdf")):
        records.append(
            normalize_pdf_file(
                pdf_path,
                repo_root=root,
                force=force,
                allow_ocr=allow_ocr,
                ocr_backend=ocr_backend,
                tesseract_path=resolved_tesseract_path,
            )
        )
    if rebuild_vector_index:
        config_store = RuntimeConfigStore(root / "config" / "runtime_config.json")
        config = config_store.load_effective() if config_store.path.exists() else config_store.defaults_model()
        corpus = bootstrap_local_corpus_with_paths(
            root,
            raw_root=root / "data" / "raw",
            normalized_root=root / "data" / "processed" / "normalized",
        )
        chunks = prepare_search_index(chunk_documents(corpus.documents))
        embedder = LocalEmbedder(
            model_name=config.embedding_model,
            device=config.embedding_device,
            enabled=config.embedding_enabled,
        )
        build_vector_index(
            chunks=chunks,
            embedder=embedder,
            index_path=root / config.vector_index_path,
        )
    return records


def normalize_pdf_file(
    pdf_path: Path,
    *,
    repo_root: Path | None = None,
    force: bool = False,
    allow_ocr: bool = False,
    ocr_backend: Any | None = None,
    tesseract_path: str | None = None,
) -> PdfNormalizationRecord:
    root = repo_root or REPO_ROOT
    relative_source_path = pdf_path.relative_to(root)
    resolved_tesseract_path = (
        tesseract_path if allow_ocr else None
    ) or _configured_tesseract_path(root)
    parse_result = parse_pdf(
        pdf_path,
        source_path=relative_source_path.as_posix(),
        allow_ocr=allow_ocr,
        ocr_backend=ocr_backend,
        tesseract_path=resolved_tesseract_path,
    )
    metadata = infer_document_metadata(
        relative_source_path,
        explicit_metadata={
            "document_type": "normalized_pdf",
            "loader_hint": "normalized_pdf_json",
            "parser": parse_result.parser,
            "extraction_status": parse_result.extraction_status,
            "extraction_quality": parse_result.extraction_quality,
            "ocr_derived": parse_result.ocr_derived,
            "ocr_backend": parse_result.ocr_backend,
            "ocr_status": parse_result.ocr_status,
            "ocr_confidence": parse_result.ocr_confidence,
            "pages": [_page_to_payload(page) for page in parse_result.pages if page.text],
        },
    )
    normalized_json_path, clean_text_path = _build_output_paths(root, relative_source_path)
    normalized_json_path.parent.mkdir(parents=True, exist_ok=True)
    clean_text_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_normalized_payload(
        parse_result=parse_result,
        metadata=metadata,
        normalized_json_path=normalized_json_path.relative_to(root).as_posix(),
        clean_text_path=clean_text_path.relative_to(root).as_posix(),
    )

    if force or not normalized_json_path.exists():
        normalized_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if parse_result.extraction_status in {"success", "partial"} and payload["pages"]:
        clean_text = _build_clean_text(parse_result.pages)
        if force or not clean_text_path.exists():
            clean_text_path.write_text(clean_text, encoding="utf-8")
    elif clean_text_path.exists():
        clean_text_path.unlink()

    return PdfNormalizationRecord(
        source_path=relative_source_path.as_posix(),
        normalized_json_path=normalized_json_path.relative_to(root).as_posix(),
        clean_text_path=clean_text_path.relative_to(root).as_posix(),
        extraction_status=parse_result.extraction_status,
        extraction_quality=parse_result.extraction_quality,
        ocr_status=parse_result.ocr_status,
        ocr_derived=parse_result.ocr_derived,
        ocr_confidence=parse_result.ocr_confidence,
        ocr_confidence_band=parse_result.ocr_confidence_band,
        text_source_mix=parse_result.text_source_mix,
        admission_status=payload["admission_status"],
        admission_reason=payload["admission_reason"],
        warnings=list(parse_result.warnings),
        page_count=parse_result.page_count,
    )


def _build_output_paths(root: Path, relative_source_path: Path) -> tuple[Path, Path]:
    relative_after_raw = Path(*relative_source_path.parts[2:])
    normalized_json_path = (
        root
        / "data"
        / "processed"
        / "normalized"
        / "pdf"
        / relative_after_raw
    ).with_suffix(".json")
    clean_text_path = (
        root
        / "data"
        / "processed"
        / "clean_text"
        / "pdf"
        / relative_after_raw
    ).with_suffix(".txt")
    return normalized_json_path, clean_text_path


def _configured_tesseract_path(root: Path) -> str | None:
    config_path = root / "config" / "runtime_config.json"
    store = RuntimeConfigStore(config_path)
    config = store.load_effective() if config_path.exists() else store.defaults_model()
    return str(config.tesseract_path or "").strip() or None


def _build_normalized_payload(
    *,
    parse_result: PdfParseResult,
    metadata: dict[str, str | None],
    normalized_json_path: str,
    clean_text_path: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0.0",
        "document_type": "normalized_pdf",
        "source_path": parse_result.source_path,
        "normalized_json_path": normalized_json_path,
        "clean_text_path": clean_text_path,
        "title": metadata["title"] or parse_result.title,
        "source_family": metadata["source_family"] or "",
        "source_type": metadata["source_type"] or "",
        "collection": metadata["collection"] or "",
        "author": metadata["author"] or "",
        "madhhab": metadata["madhhab"] or "",
        "language": metadata["language"] or "",
        "source_classification": metadata["source_classification"] or "",
        "source_role_flag": metadata["source_role_flag"] or "",
        "source_role_boundary": metadata["source_role_boundary"] or "",
        "source_lineage": metadata["source_lineage"] or "",
        "role": metadata["role"] or "",
        "domain": metadata["domain"] or "",
        "authority_level": metadata["authority_level"] or "",
        "canonical_family": metadata["canonical_family"] or "",
        "document_kind": metadata["document_kind"] or "",
        "commentary_target": metadata["commentary_target"] or "",
        "fatwa_authority": metadata["fatwa_authority"] or "",
        "parser": parse_result.parser,
        "page_count": parse_result.page_count,
        "extraction_status": parse_result.extraction_status,
        "extraction_quality": parse_result.extraction_quality,
        "ocr_derived": parse_result.ocr_derived,
        "ocr_backend": parse_result.ocr_backend,
        "ocr_status": parse_result.ocr_status,
        "ocr_confidence": parse_result.ocr_confidence,
        "ocr_confidence_band": parse_result.ocr_confidence_band,
        "text_source_mix": parse_result.text_source_mix,
        "ocr_quality_flags": list(parse_result.ocr_quality_flags),
        "warnings": list(parse_result.warnings),
        "quality_notes": dict(parse_result.quality_notes),
        "pages": [_page_to_payload(page) for page in parse_result.pages if page.text],
    }
    admission = evaluate_document_admission(
        SimpleNamespace(
            source_path=payload["source_path"],
            content=json.dumps(payload, ensure_ascii=False),
            loader_hint="normalized_pdf_json",
            source_type=payload["source_type"],
            source_classification=payload["source_classification"],
            source_family=payload["source_family"],
            source_role_boundary=payload["source_role_boundary"],
            document_kind=payload["document_kind"],
            extraction_status=payload["extraction_status"],
            extraction_quality=payload["extraction_quality"],
            ocr_status=payload["ocr_status"],
            ocr_derived=payload["ocr_derived"],
            language=payload["language"],
            collection=payload["collection"],
            source_lineage=payload["source_lineage"],
            madhhab=payload["madhhab"],
            author=payload["author"],
        )
    )
    payload["admission_status"] = admission.status
    payload["admission_reason"] = admission.reason
    payload["admission_notes"] = list(admission.notes)
    return payload


def _page_to_payload(page: ExtractedPdfPage) -> dict[str, Any]:
    return {
        "page_number": page.page_number,
        "text": page.text,
        "char_count": page.char_count,
        "warnings": list(page.warnings),
        "text_source": page.text_source,
        "ocr_confidence": page.ocr_confidence,
        "ocr_confidence_band": page.ocr_confidence_band,
    }


def _build_clean_text(pages: list[ExtractedPdfPage]) -> str:
    blocks: list[str] = []
    for page in pages:
        if not page.text:
            continue
        source_label = (
            "OCR"
            if page.text_source == "ocr"
            else page.text_source.upper()
        )
        blocks.append(f"=== Page {page.page_number} [{source_label}] ===\n{page.text}")
    return "\n\n".join(blocks).strip() + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize local PDFs for retrieval.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing normalized outputs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary instead of plain text",
    )
    parser.add_argument(
        "--allow-ocr",
        action="store_true",
        help="Attempt OCR recovery for weak PDFs when a supported OCR backend is available",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    records = process_pdf_corpus(REPO_ROOT, force=args.force, allow_ocr=args.allow_ocr)
    if args.json:
        print(
            json.dumps(
                [asdict(record) for record in records],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    successes = [
        record for record in records if record.extraction_status == "success"
    ]
    partials = [
        record for record in records if record.extraction_status == "partial"
    ]
    failures = [
        record for record in records if record.extraction_status == "failed"
    ]
    print(f"success: {len(successes)}")
    print(f"partial: {len(partials)}")
    print(f"failed: {len(failures)}")


if __name__ == "__main__":
    main()
