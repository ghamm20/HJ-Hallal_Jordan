"""Extract text from machine-readable PDFs with explicit status reporting."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExtractedPdfPage:
    page_number: int
    text: str
    char_count: int
    warnings: list[str]
    text_source: str = "pypdf"
    ocr_confidence: float | None = None
    ocr_confidence_band: str = "not_applicable"


@dataclass(slots=True)
class PdfParseResult:
    source_path: str
    title: str
    parser: str
    page_count: int | None
    extraction_status: str
    extraction_quality: str
    warnings: list[str]
    quality_notes: dict[str, Any]
    pages: list[ExtractedPdfPage]
    ocr_derived: bool = False
    ocr_backend: str = ""
    ocr_status: str = "not_attempted"
    ocr_confidence: float | None = None
    ocr_confidence_band: str = "not_applicable"
    text_source_mix: str = "none"
    ocr_quality_flags: list[str] = field(default_factory=list)


OCR_CONFIDENCE_HIGH_THRESHOLD = 0.85
OCR_CONFIDENCE_MEDIUM_THRESHOLD = 0.65


@dataclass(slots=True)
class OcrPageResult:
    text: str
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


def parse_pdf(
    pdf_path: Path,
    *,
    source_path: str | None = None,
    allow_ocr: bool = False,
    ocr_backend: Any | None = None,
    tesseract_path: str | None = None,
) -> PdfParseResult:
    PdfReader = _load_pypdf_reader()
    resolved_source_path = source_path or str(pdf_path)
    warnings: list[str] = []
    active_ocr_backend = ocr_backend or (
        _discover_ocr_backend(tesseract_path=tesseract_path) if allow_ocr else None
    )

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        return PdfParseResult(
            source_path=resolved_source_path,
            title=pdf_path.stem,
            parser="pypdf",
            page_count=None,
            extraction_status="failed",
            extraction_quality="reader_error",
            warnings=[f"reader_error:{type(exc).__name__}"],
            quality_notes={
                "pages_with_text": 0,
                "pages_without_text": 0,
                "pages_with_errors": 0,
                "total_extracted_characters": 0,
                "average_characters_per_extracted_page": 0.0,
                "extractable_page_ratio": 0.0,
                "ocr_pages_recovered": 0,
            },
            pages=[],
            ocr_status=(
                "unavailable"
                if allow_ocr and active_ocr_backend is None
                else "not_attempted"
            ),
        )

    extracted_pages: list[ExtractedPdfPage] = []
    page_count = len(reader.pages)
    pages_with_errors = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_warnings: list[str] = []
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raw_text = ""
            page_warnings.append(f"extract_error:{type(exc).__name__}")
            pages_with_errors += 1

        normalized_text = _normalize_pdf_text(raw_text)
        if not normalized_text:
            page_warnings.append("no_text_extracted")

        extracted_pages.append(
            ExtractedPdfPage(
                page_number=page_number,
                text=normalized_text,
                char_count=len(normalized_text),
                warnings=page_warnings,
            )
        )

    (
        pages_with_text,
        pages_without_text,
        total_extracted_characters,
        average_characters,
        extractable_page_ratio,
    ) = _summarize_pages(extracted_pages, page_count=page_count)
    extraction_status, extraction_quality = _assess_extraction_quality(
        pages_with_text=pages_with_text,
        pages_without_text=pages_without_text,
        pages_with_errors=pages_with_errors,
        page_count=page_count,
        total_extracted_characters=total_extracted_characters,
        average_characters=average_characters,
        warnings=warnings,
    )

    ocr_status = "not_attempted"
    ocr_backend_name = ""
    ocr_recovered_pages = 0
    ocr_confidences: list[float] = []
    ocr_quality_flags: list[str] = []
    if allow_ocr:
        if active_ocr_backend is None:
            if extraction_status in {"failed", "partial"}:
                warnings.append("ocr_unavailable")
                ocr_status = "unavailable"
        else:
            ocr_backend_name = str(getattr(active_ocr_backend, "backend_name", "") or "").strip()
            (
                ocr_recovered_pages,
                ocr_confidences,
            ) = _attempt_ocr_recovery(
                pdf_path=pdf_path,
                pages=extracted_pages,
                ocr_backend=active_ocr_backend,
            )
            if ocr_recovered_pages > 0:
                warnings.append(f"ocr_recovered_pages:{ocr_recovered_pages}")
                (
                    pages_with_text,
                    pages_without_text,
                    total_extracted_characters,
                    average_characters,
                    extractable_page_ratio,
                ) = _summarize_pages(extracted_pages, page_count=page_count)
                extraction_status, extraction_quality = _assess_extraction_quality(
                    pages_with_text=pages_with_text,
                    pages_without_text=pages_without_text,
                    pages_with_errors=pages_with_errors,
                    page_count=page_count,
                    total_extracted_characters=total_extracted_characters,
                    average_characters=average_characters,
                    warnings=warnings,
                    ocr_recovered_pages=ocr_recovered_pages,
                )
                ocr_status = (
                    "recovered_success"
                    if extraction_status == "success"
                    else "recovered_partial"
                )
            else:
                warnings.append("ocr_attempted_no_recovery")
                ocr_status = "attempted_no_recovery"

    ocr_confidence = (
        round(sum(ocr_confidences) / len(ocr_confidences), 3)
        if ocr_confidences
        else None
    )
    machine_text_pages = sum(
        1 for page in extracted_pages if page.char_count > 0 and page.text_source == "pypdf"
    )
    ocr_text_pages = sum(
        1 for page in extracted_pages if page.char_count > 0 and page.text_source == "ocr"
    )
    low_confidence_ocr_pages = sum(
        1 for page in extracted_pages if page.text_source == "ocr" and page.ocr_confidence_band == "low"
    )
    unknown_confidence_ocr_pages = sum(
        1
        for page in extracted_pages
        if page.text_source == "ocr" and page.ocr_confidence_band == "unknown"
    )
    text_source_mix = _determine_text_source_mix(
        machine_text_pages=machine_text_pages,
        ocr_text_pages=ocr_text_pages,
    )
    ocr_confidence_band = _classify_ocr_confidence(
        ocr_confidence,
        has_ocr_pages=ocr_text_pages > 0,
    )
    if text_source_mix == "mixed_machine_ocr":
        _append_warning_once(warnings, "mixed_text_sources")
        ocr_quality_flags.append("mixed_text_sources")
    if low_confidence_ocr_pages > 0 or ocr_confidence_band == "low":
        _append_warning_once(warnings, "ocr_low_confidence")
        ocr_quality_flags.append("ocr_low_confidence")
    if unknown_confidence_ocr_pages > 0 or (
        ocr_text_pages > 0 and ocr_confidence_band == "unknown"
    ):
        _append_warning_once(warnings, "ocr_confidence_unavailable")
        ocr_quality_flags.append("ocr_confidence_unavailable")

    extraction_status, extraction_quality = _assess_extraction_quality(
        pages_with_text=pages_with_text,
        pages_without_text=pages_without_text,
        pages_with_errors=pages_with_errors,
        page_count=page_count,
        total_extracted_characters=total_extracted_characters,
        average_characters=average_characters,
        warnings=warnings,
        ocr_recovered_pages=ocr_recovered_pages,
        ocr_confidence_band=ocr_confidence_band,
        low_confidence_ocr_pages=low_confidence_ocr_pages,
        unknown_confidence_ocr_pages=unknown_confidence_ocr_pages,
    )

    return PdfParseResult(
        source_path=resolved_source_path,
        title=pdf_path.stem,
        parser=(
            f"pypdf+{ocr_backend_name}"
            if ocr_recovered_pages > 0 and ocr_backend_name
            else "pypdf"
        ),
        page_count=page_count,
        extraction_status=extraction_status,
        extraction_quality=extraction_quality,
        warnings=warnings,
        quality_notes={
            "pages_with_text": pages_with_text,
            "pages_without_text": pages_without_text,
            "pages_with_errors": pages_with_errors,
            "total_extracted_characters": total_extracted_characters,
            "average_characters_per_extracted_page": round(average_characters, 2),
            "extractable_page_ratio": round(extractable_page_ratio, 4),
            "ocr_pages_recovered": ocr_recovered_pages,
            "ocr_backend_available": bool(active_ocr_backend),
            "machine_text_page_count": machine_text_pages,
            "ocr_text_page_count": ocr_text_pages,
            "text_source_mix": text_source_mix,
            "ocr_confidence_available": ocr_confidence is not None,
            "ocr_confidence_band": ocr_confidence_band,
            "low_confidence_ocr_pages": low_confidence_ocr_pages,
            "unknown_confidence_ocr_pages": unknown_confidence_ocr_pages,
            "ocr_quality_flags": list(dict.fromkeys(ocr_quality_flags)),
        },
        pages=extracted_pages,
        ocr_derived=ocr_recovered_pages > 0,
        ocr_backend=ocr_backend_name,
        ocr_status=ocr_status,
        ocr_confidence=ocr_confidence,
        ocr_confidence_band=ocr_confidence_band,
        text_source_mix=text_source_mix,
        ocr_quality_flags=list(dict.fromkeys(ocr_quality_flags)),
    )


def _normalize_pdf_text(text: str) -> str:
    normalized = text.replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def _summarize_pages(
    pages: list[ExtractedPdfPage],
    *,
    page_count: int,
) -> tuple[int, int, int, float, float]:
    pages_with_text = sum(1 for page in pages if page.char_count > 0)
    pages_without_text = max(0, page_count - pages_with_text)
    total_extracted_characters = sum(page.char_count for page in pages)
    extractable_page_ratio = pages_with_text / page_count if page_count else 0.0
    average_characters = (
        total_extracted_characters / pages_with_text if pages_with_text else 0.0
    )
    return (
        pages_with_text,
        pages_without_text,
        total_extracted_characters,
        average_characters,
        extractable_page_ratio,
    )


def _assess_extraction_quality(
    *,
    pages_with_text: int,
    pages_without_text: int,
    pages_with_errors: int,
    page_count: int,
    total_extracted_characters: int,
    average_characters: float,
    warnings: list[str],
    ocr_recovered_pages: int = 0,
    ocr_confidence_band: str = "not_applicable",
    low_confidence_ocr_pages: int = 0,
    unknown_confidence_ocr_pages: int = 0,
) -> tuple[str, str]:
    extractable_page_ratio = pages_with_text / page_count if page_count else 0.0
    if total_extracted_characters == 0:
        if "no_extractable_text" not in warnings:
            warnings.append("no_extractable_text")
        return "failed", "no_text"
    warnings[:] = [warning for warning in warnings if warning != "no_extractable_text"]

    extraction_status = "success"
    if pages_without_text > 0:
        _append_warning_once(warnings, f"pages_without_text:{pages_without_text}")
    if pages_with_errors > 0:
        _append_warning_once(warnings, f"pages_with_errors:{pages_with_errors}")
    if extractable_page_ratio < 0.75:
        _append_warning_once(warnings, "low_extractable_page_ratio")
        extraction_status = "partial"
    low_density_is_material = average_characters < 80
    if (
        ocr_recovered_pages > 0
        and pages_without_text == 0
        and total_extracted_characters >= 30
    ):
        low_density_is_material = False
    if low_density_is_material:
        _append_warning_once(warnings, "low_text_density")
        extraction_status = "partial"

    if ocr_recovered_pages > 0:
        if ocr_confidence_band == "low" or low_confidence_ocr_pages > 0:
            if extraction_status == "success":
                return extraction_status, "ocr_recovered_low_confidence"
            return extraction_status, "ocr_recovered_with_gaps_low_confidence"
        if ocr_confidence_band == "unknown" or unknown_confidence_ocr_pages > 0:
            if extraction_status == "success":
                return extraction_status, "ocr_recovered_unknown_confidence"
            return extraction_status, "ocr_recovered_with_gaps_unknown_confidence"
        if extraction_status == "success":
            return extraction_status, "ocr_recovered"
        return extraction_status, "ocr_recovered_with_gaps"
    if extraction_status == "partial":
        return extraction_status, "machine_extract_with_gaps"
    return extraction_status, "machine_extract_clean"


def _attempt_ocr_recovery(
    *,
    pdf_path: Path,
    pages: list[ExtractedPdfPage],
    ocr_backend: Any,
) -> tuple[int, list[float]]:
    recovered_pages = 0
    confidences: list[float] = []
    for page in pages:
        if page.char_count > 0:
            continue
        try:
            result = ocr_backend.extract_page_text(pdf_path, page.page_number)
        except Exception as exc:
            page.warnings.append(f"ocr_error:{type(exc).__name__}")
            continue
        if not result:
            continue
        text = _normalize_pdf_text(str(getattr(result, "text", "") or ""))
        if not text:
            page.warnings.extend(list(getattr(result, "warnings", []) or []))
            continue
        page.text = text
        page.char_count = len(text)
        page.text_source = "ocr"
        page.warnings.append("ocr_text_recovered")
        page.warnings.extend(list(getattr(result, "warnings", []) or []))
        confidence_value = getattr(result, "confidence", None)
        if confidence_value is not None:
            try:
                page.ocr_confidence = float(confidence_value)
                confidences.append(page.ocr_confidence)
            except (TypeError, ValueError):
                page.ocr_confidence = None
        page.ocr_confidence_band = _classify_ocr_confidence(
            page.ocr_confidence,
            has_ocr_pages=True,
        )
        recovered_pages += 1
    return recovered_pages, confidences


def _classify_ocr_confidence(
    confidence: float | None,
    *,
    has_ocr_pages: bool,
) -> str:
    if not has_ocr_pages:
        return "not_applicable"
    if confidence is None:
        return "unknown"
    if confidence >= OCR_CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if confidence >= OCR_CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _determine_text_source_mix(
    *,
    machine_text_pages: int,
    ocr_text_pages: int,
) -> str:
    if machine_text_pages > 0 and ocr_text_pages > 0:
        return "mixed_machine_ocr"
    if ocr_text_pages > 0:
        return "ocr_only"
    if machine_text_pages > 0:
        return "machine_only"
    return "none"


def _append_warning_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _load_pypdf_reader() -> type:
    try:
        from pypdf import PdfReader

        return PdfReader
    except ModuleNotFoundError:
        site_packages = _candidate_site_packages_paths()
        for candidate in site_packages:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        from pypdf import PdfReader

        return PdfReader


def _candidate_site_packages_paths() -> list[Path]:
    home = Path.home()
    return [
        home
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "Lib"
        / "site-packages"
    ]


def _discover_ocr_backend(*, tesseract_path: str | None = None) -> Any | None:
    _prepare_optional_site_packages()
    resolved_tesseract_path = _resolve_tesseract_path(tesseract_path)
    if not resolved_tesseract_path:
        return None
    if not _module_available("pytesseract"):
        return None
    if not _module_available("pypdfium2"):
        return None
    if not _module_available("PIL"):
        return None
    return _PytesseractPdfiumOcrBackend(resolved_tesseract_path)


def _prepare_optional_site_packages() -> None:
    for candidate in _candidate_site_packages_paths():
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class _PytesseractPdfiumOcrBackend:
    backend_name = "pytesseract_pdfium"

    def __init__(self, tesseract_cmd: str) -> None:
        self.tesseract_cmd = tesseract_cmd

    def extract_page_text(self, pdf_path: Path, page_number: int) -> OcrPageResult | None:
        import pypdfium2 as pdfium
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        document = pdfium.PdfDocument(str(pdf_path))
        if page_number < 1 or page_number > len(document):
            return None
        page = document[page_number - 1]
        bitmap = page.render(scale=2)
        image = bitmap.to_pil()
        text = pytesseract.image_to_string(image)
        confidence = _extract_tesseract_confidence(pytesseract, image)
        return OcrPageResult(text=text, confidence=confidence)


def _extract_tesseract_confidence(pytesseract_module: Any, image: Any) -> float | None:
    try:
        data = pytesseract_module.image_to_data(
            image,
            output_type=pytesseract_module.Output.DICT,
        )
    except Exception:
        return None
    raw_confidences = data.get("conf", []) if isinstance(data, dict) else []
    confidences: list[float] = []
    for raw_value in raw_confidences:
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        normalized = numeric / 100 if numeric > 1 else numeric
        if 0.0 <= normalized <= 1.0:
            confidences.append(normalized)
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 3)


def _resolve_tesseract_path(configured_path: str | None = None) -> str | None:
    candidates: list[Path] = []
    explicit = str(configured_path or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = str(os.getenv("TESSERACT_PATH", "") or "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    discovered = shutil.which("tesseract")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        [
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
