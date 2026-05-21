"""Create normalized, metadata-aware chunks for retrieval."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.retrieval.index_loader import LoadedDocument
from app.retrieval.metadata_normalizer import normalize_source_metadata


@dataclass(slots=True)
class SuppressedChunkRecord:
    source_path: str
    reference: str
    reason: str
    length: int
    quote: str


@dataclass(slots=True)
class ChunkBuildAudit:
    raw_chunk_count: int
    final_chunk_count: int
    suppressed_chunk_count: int
    suppressed_by_reason: dict[str, int]
    split_page_count: int
    split_output_chunk_count: int
    split_additional_chunk_count: int
    suppressed_examples: list[SuppressedChunkRecord]


@dataclass(slots=True)
class ChunkBuildResult:
    chunks: list[dict[str, Any]]
    audit: ChunkBuildAudit


def chunk_documents(
    documents: list[LoadedDocument],
    *,
    chunk_size_chars: int = 900,
    pdf_page_chunk_size_chars: int = 1400,
) -> list[dict[str, Any]]:
    return chunk_documents_with_audit(
        documents,
        chunk_size_chars=chunk_size_chars,
        pdf_page_chunk_size_chars=pdf_page_chunk_size_chars,
    ).chunks


def chunk_documents_with_audit(
    documents: list[LoadedDocument],
    *,
    chunk_size_chars: int = 900,
    pdf_page_chunk_size_chars: int = 1400,
    suppressed_examples_limit: int = 25,
) -> ChunkBuildResult:
    raw_chunks: list[dict[str, Any]] = []
    split_page_count = 0
    split_output_chunk_count = 0
    for document in documents:
        if document.loader_hint == "quran_ayah_text":
            raw_chunks.extend(_chunk_quran_ayah_text(document))
        elif document.loader_hint == "quran_translation_csv":
            raw_chunks.extend(_chunk_quran_translation_csv(document))
        elif document.loader_hint == "normalized_pdf_json":
            pdf_chunks, page_splits, split_outputs = _chunk_normalized_pdf_json(
                document,
                pdf_page_chunk_size_chars=pdf_page_chunk_size_chars,
            )
            raw_chunks.extend(pdf_chunks)
            split_page_count += page_splits
            split_output_chunk_count += split_outputs
        elif document.loader_hint == "csv_rows":
            raw_chunks.extend(_chunk_csv_rows(document))
        else:
            raw_chunks.extend(
                _chunk_plain_text(document, chunk_size_chars=chunk_size_chars)
            )

    filtered_chunks, suppressed_by_reason, suppressed_examples = _filter_junk_chunks(
        raw_chunks,
        suppressed_examples_limit=suppressed_examples_limit,
    )
    audit = ChunkBuildAudit(
        raw_chunk_count=len(raw_chunks),
        final_chunk_count=len(filtered_chunks),
        suppressed_chunk_count=sum(suppressed_by_reason.values()),
        suppressed_by_reason=dict(suppressed_by_reason),
        split_page_count=split_page_count,
        split_output_chunk_count=split_output_chunk_count,
        split_additional_chunk_count=max(0, split_output_chunk_count - split_page_count),
        suppressed_examples=suppressed_examples,
    )
    return ChunkBuildResult(chunks=filtered_chunks, audit=audit)


def _chunk_quran_ayah_text(document: LoadedDocument) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for raw_line in document.content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        sura, aya, ayah_text = parts
        ayah_text = _normalize_whitespace(ayah_text)
        if not ayah_text:
            continue
        reference = f"Qur'an {sura}:{aya}"
        chunks.append(
            _build_chunk(
                document=document,
                chunk_suffix=f"{sura}:{aya}",
                reference=reference,
                text=ayah_text,
                quote=ayah_text,
            )
        )
    return chunks


def _chunk_quran_translation_csv(document: LoadedDocument) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row_index, row in enumerate(_iter_csv_dict_rows(document.content), start=1):
        sura = row.get("sura", "").strip()
        aya = row.get("aya", "").strip()
        translation = _normalize_whitespace(row.get("translation", ""))
        footnotes = _normalize_whitespace(row.get("footnotes", ""))
        if not sura or not aya or not translation:
            continue
        reference = f"Qur'an {sura}:{aya}"
        text = translation if not footnotes else f"{translation}\nFootnotes: {footnotes}"
        chunks.append(
            _build_chunk(
                document=document,
                chunk_suffix=f"row-{row_index}",
                reference=reference,
                text=text,
                quote=translation,
            )
        )
    return chunks


def _chunk_csv_rows(document: LoadedDocument) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row_index, row in enumerate(_iter_csv_dict_rows(document.content), start=1):
        text = _normalize_whitespace(" ".join(value for value in row.values() if value))
        if not text:
            continue
        chunks.append(
            _build_chunk(
                document=document,
                chunk_suffix=f"row-{row_index}",
                reference=f"{document.title} row {row_index}",
                text=text,
                quote=text[:280],
            )
        )
    return chunks


def _chunk_normalized_pdf_json(
    document: LoadedDocument,
    *,
    pdf_page_chunk_size_chars: int,
) -> tuple[list[dict[str, Any]], int, int]:
    payload = json.loads(document.content)
    pages = payload.get("pages", [])
    extraction_status = str(payload.get("extraction_status", "")).strip()
    quality_warning_count = len(payload.get("warnings", []))
    chunks: list[dict[str, Any]] = []
    split_page_count = 0
    split_output_chunk_count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        raw_text = str(page.get("text", "") or "")
        if not raw_text.strip():
            continue
        try:
            page_number_int = int(page_number)
        except (TypeError, ValueError):
            page_number_int = None
        page_reference = (
            f"{document.title} p. {page_number_int}"
            if page_number_int is not None
            else f"{document.title} page"
        )
        segments = _split_pdf_page_segments(
            raw_text,
            chunk_size_chars=pdf_page_chunk_size_chars,
        )
        if len(segments) > 1:
            split_page_count += 1
            split_output_chunk_count += len(segments)
        for segment_index, segment in enumerate(segments, start=1):
            text = _normalize_whitespace(str(segment.get("text", "")))
            if not text:
                continue
            reference = (
                page_reference
                if len(segments) == 1
                else f"{page_reference} [part {segment_index}/{len(segments)}]"
            )
            extra_fields: dict[str, Any] = {
                "page_reference": page_reference,
                "page_chunk_index": segment_index,
                "page_chunk_total": len(segments),
                "extraction_status": extraction_status,
                "document_quality_warnings": quality_warning_count,
            }
            section_label = str(segment.get("section_label", "") or "").strip()
            section_kind = str(segment.get("section_kind", "") or "").strip()
            if section_label:
                extra_fields["section_label"] = section_label
            if section_kind:
                extra_fields["section_kind"] = section_kind
            chunk = _build_chunk(
                document=document,
                chunk_suffix=(
                    f"page-{page_number_int or 'unknown'}-part-{segment_index}"
                    if len(segments) > 1
                    else f"page-{page_number_int or 'unknown'}"
                ),
                reference=reference,
                text=text,
                quote=text[:280],
                extra_fields=extra_fields,
            )
            if page_number_int is not None:
                chunk["page_number"] = page_number_int
            chunks.append(chunk)
    return chunks, split_page_count, split_output_chunk_count


def _chunk_plain_text(
    document: LoadedDocument,
    *,
    chunk_size_chars: int,
) -> list[dict[str, Any]]:
    normalized = document.content.replace("\r\n", "\n")
    segments = [
        _normalize_whitespace(segment)
        for segment in re.split(r"\n\s*\n", normalized)
        if segment.strip()
    ]
    if not segments:
        segments = [
            _normalize_whitespace(segment)
            for segment in normalized.splitlines()
            if segment.strip()
        ]

    windows: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else f"{current}\n\n{segment}"
        if current and len(candidate) > chunk_size_chars:
            windows.append(current)
            current = segment
        else:
            current = candidate
    if current:
        windows.append(current)

    chunks: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        for sub_index, text in enumerate(
            _split_long_text(window, chunk_size_chars),
            start=1,
        ):
            chunk_number = f"{index}.{sub_index}" if len(windows) > 1 else str(sub_index)
            chunks.append(
                _build_chunk(
                    document=document,
                    chunk_suffix=f"chunk-{chunk_number}",
                    reference=f"{document.title} chunk {chunk_number}",
                    text=text,
                    quote=text[:280],
                )
            )
    return chunks


def _split_long_text(text: str, chunk_size_chars: int) -> list[str]:
    if len(text) <= chunk_size_chars:
        return [text]
    words = text.split()
    windows: list[str] = []
    current_words: list[str] = []
    current_length = 0
    for word in words:
        projected = current_length + len(word) + (1 if current_words else 0)
        if current_words and projected > chunk_size_chars:
            windows.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length = projected
    if current_words:
        windows.append(" ".join(current_words))
    return _merge_short_trailing_window(windows, chunk_size_chars=chunk_size_chars)


def _split_pdf_page_segments(
    text: str,
    *,
    chunk_size_chars: int,
) -> list[dict[str, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    segments = [
        segment.strip()
        for segment in re.split(r"\n\s*\n", normalized)
        if segment.strip()
    ]
    if not segments:
        segments = [segment.strip() for segment in normalized.splitlines() if segment.strip()]
    if not segments:
        normalized_text = _normalize_whitespace(text)
        return [{"text": normalized_text}] if normalized_text else []

    windows: list[dict[str, str]] = []
    current = ""
    current_label = ""
    current_kind = ""
    for segment in segments:
        cleaned_segment = _normalize_whitespace(segment)
        metadata = _extract_pdf_section_metadata(segment)
        section_label = metadata["section_label"]
        section_kind = metadata["section_kind"]
        if current and section_label:
            windows.extend(
                _emit_pdf_windows(
                    current,
                    section_label=current_label,
                    section_kind=current_kind,
                    chunk_size_chars=chunk_size_chars,
                )
            )
            current = cleaned_segment
            current_label = section_label
            current_kind = section_kind
            continue
        candidate = cleaned_segment if not current else f"{current}\n\n{cleaned_segment}"
        if current and len(candidate) > chunk_size_chars:
            windows.extend(
                _emit_pdf_windows(
                    current,
                    section_label=current_label,
                    section_kind=current_kind,
                    chunk_size_chars=chunk_size_chars,
                )
            )
            current = cleaned_segment
            current_label = section_label
            current_kind = section_kind
        else:
            if not current:
                current_label = section_label
                current_kind = section_kind
            elif not current_label and section_label:
                current_label = section_label
                current_kind = section_kind
            current = candidate
    if current:
        windows.extend(
            _emit_pdf_windows(
                current,
                section_label=current_label,
                section_kind=current_kind,
                chunk_size_chars=chunk_size_chars,
            )
        )
    return [window for window in windows if window.get("text", "").strip()]


def _emit_pdf_windows(
    text: str,
    *,
    section_label: str,
    section_kind: str,
    chunk_size_chars: int,
) -> list[dict[str, str]]:
    windows: list[dict[str, str]] = []
    for window in _split_long_text(text, chunk_size_chars):
        payload = {"text": window}
        if section_label:
            payload["section_label"] = section_label
        if section_kind:
            payload["section_kind"] = section_kind
        windows.append(payload)
    return windows


def _extract_pdf_section_metadata(text: str) -> dict[str, str]:
    lines = [_normalize_whitespace(line) for line in text.splitlines() if line.strip()]
    if not lines:
        return {"section_label": "", "section_kind": ""}

    if _looks_like_contents_segment(lines):
        return {
            "section_label": "",
            "section_kind": "contents_like",
        }

    for raw_line in lines[:3]:
        cleaned_line = _clean_section_label(raw_line)
        section_kind = _heading_kind_for_line(cleaned_line)
        if cleaned_line and section_kind:
            return {
                "section_label": cleaned_line,
                "section_kind": section_kind,
            }

    title_like_label = _infer_title_like_label(lines[:8])
    if title_like_label:
        return {
            "section_label": title_like_label,
            "section_kind": "title_like",
        }
    return {"section_label": "", "section_kind": ""}


def _looks_like_contents_segment(lines: list[str]) -> bool:
    lowered_lines = [line.casefold() for line in lines]
    if lowered_lines and "content" in lowered_lines[0]:
        return True
    heading_lines = sum(
        1
        for line in lowered_lines
        if re.match(r"^(book|chapter|supplement)\b", line)
    )
    dotted_lines = sum(1 for line in lines if re.search(r"[.\u2022]{4,}", line))
    return heading_lines >= 3 or dotted_lines >= 3


def _clean_section_label(line: str) -> str:
    cleaned = _normalize_whitespace(line)
    cleaned = re.sub(r"^[•·*\-–—<>()\[\]{}|/\\\d\s]+", "", cleaned)
    cleaned = re.sub(r"[.\u2022]{4,}\s*\d+\s*$", "", cleaned)
    cleaned = re.sub(
        r"\s+\d{1,4}(?:\s+[A-Za-z0-9#&+'/-]{1,8}){0,3}\s*$",
        "",
        cleaned,
    )
    if re.search(r"[A-Za-z]", cleaned):
        cleaned = re.sub(r"\s+\d+\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :.-")
    if len(cleaned) < 4 or len(cleaned) > 140:
        return ""
    if not re.search(r"[A-Za-z]", cleaned):
        return ""
    return cleaned


def _heading_kind_for_line(line: str) -> str:
    if not line:
        return ""
    lowered = line.casefold()
    alpha_tokens = re.findall(r"[A-Za-z]+", line)
    if not alpha_tokens:
        return ""
    if re.match(r"^chapter\b", lowered):
        return "chapter_heading"
    if re.match(r"^book\b", lowered) or "book of" in lowered:
        return "book_heading"
    if re.match(r"^supplement\b", lowered):
        return "supplement_heading"
    if re.match(r"^(\d+|[ivxlcdm]+)[\.\):]\s+", lowered):
        return "numbered_section"
    uppercase_letters = sum(character.isupper() for character in line)
    letter_count = sum(character.isalpha() for character in line)
    if letter_count and uppercase_letters / letter_count >= 0.65 and len(alpha_tokens) <= 8:
        return "major_heading"
    if len(alpha_tokens) <= 8 and len(line.split()) <= 10 and not re.search(r"[.!?]", line):
        title_case_tokens = sum(token[:1].isupper() for token in alpha_tokens)
        if title_case_tokens / max(len(alpha_tokens), 1) >= 0.75:
            return "major_heading"
    return ""


def _infer_title_like_label(lines: list[str]) -> str:
    candidates: list[str] = []
    for line in lines:
        cleaned = _clean_section_label(line)
        if not cleaned or re.search(r"www\.|@|copyright", cleaned.casefold()):
            continue
        uppercase_letters = sum(character.isupper() for character in cleaned)
        letter_count = sum(character.isalpha() for character in cleaned)
        if not letter_count:
            continue
        if uppercase_letters / letter_count >= 0.7 and len(cleaned.split()) <= 8:
            candidates.append(cleaned)
    if candidates:
        return max(candidates, key=len)
    return ""


def _merge_short_trailing_window(
    windows: list[str],
    *,
    chunk_size_chars: int,
) -> list[str]:
    if len(windows) < 2:
        return windows
    tail = windows[-1].strip()
    if not tail:
        return [window for window in windows if window.strip()]

    min_tail_chars = min(80, max(30, chunk_size_chars // 8))
    if len(tail) >= min_tail_chars:
        return windows

    max_overflow_chars = max(40, chunk_size_chars // 8)
    merged = f"{windows[-2]} {tail}".strip()
    if len(merged) > chunk_size_chars + max_overflow_chars:
        return windows
    return [*windows[:-2], merged]


def _iter_csv_dict_rows(content: str) -> list[dict[str, str]]:
    lines = content.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.lower().startswith("id,sura,aya,translation"):
            header_index = index
            break
    if header_index is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    return [dict(row) for row in reader if row]


def _build_chunk(
    *,
    document: LoadedDocument,
    chunk_suffix: str,
    reference: str,
    text: str,
    quote: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk = {
        "chunk_id": f"{document.document_id}#{chunk_suffix}",
        "document_id": document.document_id,
        "title": document.title,
        "source_type": document.source_type or "",
        "source_classification": document.source_classification or "",
        "source_role_flag": document.source_role_flag or "",
        "collection": document.collection,
        "author": document.author,
        "madhhab": document.madhhab,
        "reference": reference,
        "quote": quote,
        "source_path": document.source_path,
        "text": text,
        "source_family": document.source_family,
        "canonical_family": document.canonical_family,
        "language": document.language,
        "loader_hint": document.loader_hint,
        "document_extension": document.extension,
        "book": document.book,
        "chapter": document.chapter,
        "section": document.section,
        "document_kind": document.document_kind,
        "commentary_target": document.commentary_target,
        "fatwa_authority": document.fatwa_authority,
        "source_role_boundary": document.source_role_boundary,
        "source_lineage": document.source_lineage,
        "role": document.role,
        "domain": document.domain,
        "authority_level": document.authority_level,
        "provider": getattr(document, "provider", ""),
        "source_url": getattr(document, "source_url", ""),
        "topic": getattr(document, "topic", ""),
        "copyright_status": getattr(document, "copyright_status", ""),
        "allowed_use": getattr(document, "allowed_use", ""),
        "extraction_status": document.extraction_status,
        "extraction_quality": document.extraction_quality,
        "ocr_derived": document.ocr_derived,
        "ocr_backend": document.ocr_backend,
        "ocr_status": document.ocr_status,
        "ocr_confidence": document.ocr_confidence,
        "ocr_confidence_band": getattr(document, "ocr_confidence_band", ""),
        "text_source_mix": getattr(document, "text_source_mix", ""),
    }
    if extra_fields:
        chunk.update(extra_fields)
    chunk.update(normalize_source_metadata(chunk))
    return chunk


def _filter_junk_chunks(
    chunks: list[dict[str, Any]],
    *,
    suppressed_examples_limit: int,
) -> tuple[list[dict[str, Any]], Counter[str], list[SuppressedChunkRecord]]:
    filtered: list[dict[str, Any]] = []
    suppressed_by_reason: Counter[str] = Counter()
    suppressed_examples: list[SuppressedChunkRecord] = []

    for chunk in chunks:
        reason = _suppression_reason(chunk)
        if reason is None:
            filtered.append(chunk)
            continue
        suppressed_by_reason[reason] += 1
        if len(suppressed_examples) < suppressed_examples_limit:
            text = str(chunk.get("text", ""))
            suppressed_examples.append(
                SuppressedChunkRecord(
                    source_path=str(chunk.get("source_path", "")),
                    reference=str(chunk.get("reference", "")),
                    reason=reason,
                    length=len(text),
                    quote=text[:160],
                )
            )
    return filtered, suppressed_by_reason, suppressed_examples


def _suppression_reason(chunk: dict[str, Any]) -> str | None:
    text = str(chunk.get("text", "") or "").strip()
    source_type = str(chunk.get("source_type", "") or "").strip().lower()
    if not text:
        return "empty_fragment"
    if not any(character.isalnum() for character in text):
        return "punctuation_only_fragment"
    if _looks_like_page_number_only(text):
        return "page_number_only_fragment"
    if source_type == "quran":
        return None
    if len(text) < 4:
        return "near_empty_fragment"

    meaningful_tokens = _meaningful_word_tokens(text)
    if len(text) < 12 and not meaningful_tokens:
        return "near_empty_fragment"
    if len(text) < 20 and not meaningful_tokens and _looks_like_pdf_debris(text):
        return "pdf_extraction_debris"
    return None


def _looks_like_page_number_only(text: str) -> bool:
    normalized = _normalize_whitespace(text).casefold()
    if re.fullmatch(r"[\divxlcdm./\-\u2013\u2014 ]{1,24}", normalized):
        return True
    return bool(
        re.fullmatch(r"(page|p)\.?\s*\d+(\s*(of|/)\s*\d+)?", normalized)
    )


def _meaningful_word_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]{3,}", text, flags=re.UNICODE)


def _looks_like_pdf_debris(text: str) -> bool:
    alnum_count = sum(character.isalnum() for character in text)
    alpha_tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    longest_alpha = max((len(token) for token in alpha_tokens), default=0)
    noisy_characters = sum(
        1 for character in text if not character.isalnum() and not character.isspace()
    )
    if len(text) <= 12 and longest_alpha <= 2:
        return True
    if len(text) <= 18 and alnum_count / max(len(text), 1) < 0.55:
        return True
    if len(text) <= 18 and noisy_characters >= 2 and longest_alpha <= 2:
        return True
    return False


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
