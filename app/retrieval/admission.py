"""Corpus admission rules for retrieval-safe document loading."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.retrieval.families import build_duplicate_family_key
from app.retrieval.metadata_normalizer import SCHOOL_MADHHAB_IDS, infer_path_madhhab

ADMITTED_STATUSES = {"admit", "admit_with_warnings"}
REQUIRED_METADATA_FIELDS = (
    "language",
    "collection",
    "source_family",
    "source_lineage",
    "source_role_boundary",
)
OCR_REVIEWABLE_REASONS = {
    "ocr_low_confidence_requires_review",
    "partial_extraction_ocr_low_confidence",
    "ocr_confidence_unknown_requires_review",
    "partial_extraction_ocr_confidence_unknown",
}
FIQH_REVIEW_PATH_PARTS = {"imports_staging", "unknown_or_mixed", "review_queue"}
CLASS_REVIEW_PATH_PARTS = {"imports_staging", "review_queue"}
CLASS_ALLOWED_STORAGE_MODES = {"metadata_only", "excerpt_only", "notes_only", "permission_documented"}


@dataclass(slots=True)
class AdmissionDecision:
    source_path: str
    status: str
    reason: str
    notes: list[str] = field(default_factory=list)
    missing_required_fields: list[str] = field(default_factory=list)
    extraction_status: str = ""
    extraction_quality: str = ""
    ocr_status: str = ""
    ocr_derived: bool = False
    duplicate_status: str = ""
    canonical_source_path: str = ""
    source_type: str = ""
    source_classification: str = ""
    source_family: str = ""
    source_role_boundary: str = ""
    document_kind: str = ""
    madhhab: str = ""
    author: str = ""


def apply_admission_policy(
    documents: list[Any],
) -> tuple[list[Any], list[AdmissionDecision]]:
    duplicate_lookup = build_duplicate_admission_lookup(documents)
    admitted: list[Any] = []
    decisions: list[AdmissionDecision] = []
    for document in documents:
        duplicate_status = duplicate_lookup.get(str(getattr(document, "source_path", "") or ""))
        decision = evaluate_document_admission(
            document,
            duplicate_status=duplicate_status,
        )
        decisions.append(decision)
        if decision.status in ADMITTED_STATUSES:
            admitted.append(document)
    return admitted, decisions


def evaluate_document_admission(
    document: Any,
    *,
    duplicate_status: dict[str, str] | None = None,
) -> AdmissionDecision:
    source_path = str(getattr(document, "source_path", "") or "")
    source_type = str(getattr(document, "source_type", "") or "").strip()
    source_classification = str(
        getattr(document, "source_classification", "") or ""
    ).strip()
    source_family = str(getattr(document, "source_family", "") or "").strip()
    source_role_boundary = str(
        getattr(document, "source_role_boundary", "") or ""
    ).strip()
    document_kind = str(getattr(document, "document_kind", "") or "").strip()
    madhhab = str(getattr(document, "madhhab", "") or "").strip()
    author = str(getattr(document, "author", "") or "").strip()
    extraction_status = str(getattr(document, "extraction_status", "") or "").strip()
    extraction_quality = str(
        getattr(document, "extraction_quality", "") or ""
    ).strip()
    ocr_status = str(getattr(document, "ocr_status", "") or "").strip() or "not_attempted"
    ocr_derived = bool(getattr(document, "ocr_derived", False))
    notes: list[str] = []

    if not str(getattr(document, "content", "") or "").strip():
        return AdmissionDecision(
            source_path=source_path,
            status="reject",
            reason="empty_content",
            source_type=source_type,
            source_classification=source_classification,
            source_family=source_family,
            source_role_boundary=source_role_boundary,
            document_kind=document_kind,
            madhhab=madhhab,
            author=author,
        )

    missing_required_fields = [
        field_name
        for field_name in REQUIRED_METADATA_FIELDS
        if not str(getattr(document, field_name, "") or "").strip()
    ]
    if not source_type:
        return AdmissionDecision(
            source_path=source_path,
            status="reject",
            reason="unknown_source_type",
            missing_required_fields=missing_required_fields,
            source_classification=source_classification,
            source_family=source_family,
            source_role_boundary=source_role_boundary,
            document_kind=document_kind,
            madhhab=madhhab,
            author=author,
        )
    if source_classification in {"", "unknown"}:
        return AdmissionDecision(
            source_path=source_path,
            status="defer",
            reason="unknown_source_classification",
            missing_required_fields=missing_required_fields,
            source_type=source_type,
            source_classification=source_classification,
            source_family=source_family,
            source_role_boundary=source_role_boundary,
            document_kind=document_kind,
            madhhab=madhhab,
            author=author,
        )
    payload = _load_payload_if_available(document)
    warnings = _payload_warnings(payload)
    extractable_page_ratio = _payload_float(payload, "extractable_page_ratio")
    ocr_confidence_band = _payload_str(payload, "ocr_confidence_band")
    text_source_mix = _payload_str(payload, "text_source_mix")
    ocr_quality_flags = _payload_string_list(payload, "ocr_quality_flags")
    fiqh_path_scope = _classify_fiqh_path_scope(source_path)
    class_path_scope = _classify_class_path_scope(source_path)
    if duplicate_status:
        relation = duplicate_status.get("relation", "")
        if relation == "alternate":
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="duplicate_alternate_copy",
                notes=["duplicate_family_alternate"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                duplicate_status="alternate",
                canonical_source_path=duplicate_status.get("canonical_source_path", ""),
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        notes.append("canonical_duplicate_family_copy")

    if source_family == "fiqh":
        if fiqh_path_scope == "review_pending":
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="fiqh_madhhab_review_queue",
                notes=["review_queue_path"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if fiqh_path_scope == "comparative":
            if madhhab not in {"", "comparative"}:
                return AdmissionDecision(
                    source_path=source_path,
                    status="defer",
                    reason="comparative_source_metadata_mismatch",
                    notes=["comparative_path_requires_comparative_label"],
                    extraction_status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                    canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                )
            notes.append("comparative_context_only")
        elif fiqh_path_scope in SCHOOL_MADHHAB_IDS and madhhab and madhhab != fiqh_path_scope:
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="madhhab_path_metadata_mismatch",
                notes=[f"path_requires:{fiqh_path_scope}", f"declared_madhhab:{madhhab}"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )

    if source_family == "classes":
        provider = str(getattr(document, "provider", "") or _payload_str(payload, "provider")).strip()
        source_url = str(
            getattr(document, "source_url", "")
            or _payload_str(payload, "source_url")
            or _payload_str(payload, "url")
        ).strip()
        topic = str(getattr(document, "topic", "") or _payload_str(payload, "topic")).strip()
        copyright_status = str(
            getattr(document, "copyright_status", "")
            or _payload_str(payload, "copyright_status")
        ).strip()
        allowed_use = str(
            getattr(document, "allowed_use", "")
            or _payload_str(payload, "allowed_use")
        ).strip()
        storage_mode = str(_payload_str(payload, "storage_mode") or "").strip()

        if source_classification not in {"transcript", "scholar_transcript"} or source_role_boundary not in {
            "teaching_layer",
            "transcript",
        }:
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="class_material_wrong_authority_bucket",
                notes=["class_material_must_remain_teaching_only"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if class_path_scope == "review_pending":
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="class_material_review_queue",
                notes=["review_queue_path"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if class_path_scope in SCHOOL_MADHHAB_IDS and not madhhab:
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="class_material_madhhab_unclear",
                notes=[f"path_requires:{class_path_scope}"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if class_path_scope in SCHOOL_MADHHAB_IDS and madhhab and madhhab != class_path_scope:
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="class_material_madhhab_path_metadata_mismatch",
                notes=[f"path_requires:{class_path_scope}", f"declared_madhhab:{madhhab}"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if class_path_scope == "general" and madhhab in SCHOOL_MADHHAB_IDS:
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="class_material_general_path_mismatch",
                notes=[f"declared_madhhab:{madhhab}"],
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        class_missing_fields = []
        if not provider and not author:
            class_missing_fields.append("teacher_or_provider")
        if not source_url:
            class_missing_fields.append("source_url")
        if not topic and not str(getattr(document, "collection", "") or "").strip():
            class_missing_fields.append("topic")
        if copyright_status not in {"external_reference_only", "permission_documented"}:
            class_missing_fields.append("copyright_status")
        if class_missing_fields:
            return AdmissionDecision(
                source_path=source_path,
                status="defer",
                reason="class_material_missing_required_metadata",
                notes=class_missing_fields,
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if copyright_status == "external_reference_only":
            content_length = len(str(getattr(document, "content", "") or ""))
            effective_storage_mode = storage_mode or allowed_use
            if content_length > 1600 and effective_storage_mode not in CLASS_ALLOWED_STORAGE_MODES:
                return AdmissionDecision(
                    source_path=source_path,
                    status="defer",
                    reason="class_material_full_text_permission_required",
                    notes=["external_reference_only_requires_metadata_or_excerpt_posture"],
                    extraction_status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                )
        notes.append("teaching_layer_only")
        if provider:
            notes.append(f"provider:{provider}")
        if topic:
            notes.append(f"topic:{topic}")
        if copyright_status:
            notes.append(f"copyright:{copyright_status}")

    if str(getattr(document, "loader_hint", "") or "").strip() == "normalized_pdf_json":
        if extraction_status == "failed":
            reason = "failed_extraction"
            if ocr_status == "unavailable":
                reason = "failed_extraction_ocr_unavailable"
            return AdmissionDecision(
                source_path=source_path,
                status="reject",
                reason=reason,
                notes=list(warnings),
                extraction_status=extraction_status,
                extraction_quality=extraction_quality,
                ocr_status=ocr_status,
                ocr_derived=ocr_derived,
                duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                source_type=source_type,
                source_classification=source_classification,
                source_family=source_family,
                source_role_boundary=source_role_boundary,
                document_kind=document_kind,
                madhhab=madhhab,
                author=author,
            )
        if extraction_status == "partial":
            if "low_extractable_page_ratio" in warnings or (
                extractable_page_ratio is not None and extractable_page_ratio < 0.5
            ):
                reason = "partial_extraction_requires_review"
                if ocr_status == "unavailable":
                    reason = "partial_extraction_ocr_unavailable"
                return AdmissionDecision(
                    source_path=source_path,
                    status="defer",
                    reason=reason,
                    notes=list(warnings),
                    extraction_status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                    canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                )
            notes.append("partial_extraction")
        if ocr_derived:
            if ocr_confidence_band == "low" or extraction_quality.endswith("low_confidence"):
                reason = (
                    "partial_extraction_ocr_low_confidence"
                    if extraction_status == "partial"
                    else "ocr_low_confidence_requires_review"
                )
                reviewed = _build_reviewed_ocr_decision(
                    source_path=source_path,
                    status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                    duplicate_status=duplicate_status,
                    warnings=warnings,
                    notes=notes,
                    ocr_quality_flags=ocr_quality_flags,
                    fallback_reason=reason,
                )
                if reviewed is not None:
                    return reviewed
                return AdmissionDecision(
                    source_path=source_path,
                    status="defer",
                    reason=reason,
                    notes=_dedupe_preserve_order(notes + list(warnings) + ocr_quality_flags),
                    extraction_status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                    canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                )
            if ocr_confidence_band == "unknown" or extraction_quality.endswith("unknown_confidence"):
                reason = (
                    "partial_extraction_ocr_confidence_unknown"
                    if extraction_status == "partial"
                    else "ocr_confidence_unknown_requires_review"
                )
                reviewed = _build_reviewed_ocr_decision(
                    source_path=source_path,
                    status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                    duplicate_status=duplicate_status,
                    warnings=warnings,
                    notes=notes,
                    ocr_quality_flags=ocr_quality_flags,
                    fallback_reason=reason,
                )
                if reviewed is not None:
                    return reviewed
                return AdmissionDecision(
                    source_path=source_path,
                    status="defer",
                    reason=reason,
                    notes=_dedupe_preserve_order(notes + list(warnings) + ocr_quality_flags),
                    extraction_status=extraction_status,
                    extraction_quality=extraction_quality,
                    ocr_status=ocr_status,
                    ocr_derived=ocr_derived,
                    duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
                    canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
                    source_type=source_type,
                    source_classification=source_classification,
                    source_family=source_family,
                    source_role_boundary=source_role_boundary,
                    document_kind=document_kind,
                    madhhab=madhhab,
                    author=author,
                )
        if ocr_derived:
            notes.append("ocr_derived")
            if ocr_confidence_band:
                notes.append(f"ocr_confidence_band:{ocr_confidence_band}")
            if text_source_mix and text_source_mix not in {"machine_only", "none"}:
                notes.append(f"text_source_mix:{text_source_mix}")
            notes.extend(ocr_quality_flags)
        if warnings:
            notes.extend(warnings)

    if missing_required_fields:
        return AdmissionDecision(
            source_path=source_path,
            status="defer",
            reason="missing_required_metadata",
            notes=_dedupe_preserve_order(notes),
            missing_required_fields=missing_required_fields,
            extraction_status=extraction_status,
            extraction_quality=extraction_quality,
            ocr_status=ocr_status,
            ocr_derived=ocr_derived,
            duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
            canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
            source_type=source_type,
            source_classification=source_classification,
            source_family=source_family,
            source_role_boundary=source_role_boundary,
            document_kind=document_kind,
            madhhab=madhhab,
            author=author,
        )

    notes = _dedupe_preserve_order(notes)
    status = "admit_with_warnings" if notes else "admit"
    reason = "meets_admission_threshold"
    if status == "admit_with_warnings":
        reason = "admitted_with_quality_flags"
    return AdmissionDecision(
        source_path=source_path,
        status=status,
        reason=reason,
        notes=notes,
        extraction_status=extraction_status,
        extraction_quality=extraction_quality,
        ocr_status=ocr_status,
        ocr_derived=ocr_derived,
        duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
        canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
        source_type=source_type,
        source_classification=source_classification,
        source_family=source_family,
        source_role_boundary=source_role_boundary,
        document_kind=document_kind,
        madhhab=madhhab,
        author=author,
    )


def build_duplicate_admission_lookup(documents: list[Any]) -> dict[str, dict[str, str]]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for document in documents:
        source_path = str(getattr(document, "source_path", "") or "")
        family_key = build_duplicate_family_key(
            source_type=str(getattr(document, "source_type", "") or ""),
            title=str(getattr(document, "title", "") or ""),
            source_path=source_path,
            collection=str(getattr(document, "collection", "") or ""),
        )
        fingerprint = _document_text_fingerprint(document)
        if not family_key or not fingerprint:
            continue
        grouped.setdefault((family_key, fingerprint), []).append(document)

    lookup: dict[str, dict[str, str]] = {}
    for (_, _), family_documents in grouped.items():
        if len(family_documents) < 2:
            continue
        canonical = min(family_documents, key=_duplicate_canonical_score)
        for document in family_documents:
            source_path = str(getattr(document, "source_path", "") or "")
            lookup[source_path] = {
                "relation": "canonical" if document is canonical else "alternate",
                "canonical_source_path": str(getattr(canonical, "source_path", "") or ""),
            }
    return lookup


def summarize_admission_decisions(
    decisions: list[AdmissionDecision],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        status_counts[decision.status] = status_counts.get(decision.status, 0) + 1
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
    return {
        "total_candidates": len(decisions),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
    }


def serialize_admission_decision(decision: AdmissionDecision) -> dict[str, Any]:
    return asdict(decision)


def _classify_fiqh_path_scope(source_path: str) -> str:
    parts = [part.casefold() for part in Path(source_path).as_posix().split("/") if part]
    if "fiqh" not in parts:
        return ""
    if any(part in FIQH_REVIEW_PATH_PARTS for part in parts):
        return "review_pending"
    return infer_path_madhhab(parts)


def _classify_class_path_scope(source_path: str) -> str:
    parts = [part.casefold() for part in Path(source_path).as_posix().split("/") if part]
    if "classes" not in parts:
        return ""
    if any(part in CLASS_REVIEW_PATH_PARTS for part in parts):
        return "review_pending"
    if "comparative" in parts:
        return "comparative"
    if "general" in parts:
        return "general"
    return infer_path_madhhab(parts)


def _load_payload_if_available(document: Any) -> dict[str, Any]:
    if str(getattr(document, "loader_hint", "") or "").strip() not in {
        "normalized_pdf_json",
        "normalized_class_json",
    }:
        return {}
    try:
        return json.loads(str(getattr(document, "content", "") or ""))
    except json.JSONDecodeError:
        return {}


def _payload_warnings(payload: dict[str, Any]) -> list[str]:
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list):
        return []
    return [str(item) for item in warnings if str(item).strip()]


def _payload_float(payload: dict[str, Any], field_name: str) -> float | None:
    quality_notes = payload.get("quality_notes", {})
    if isinstance(quality_notes, dict):
        value = quality_notes.get(field_name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _payload_str(payload: dict[str, Any], field_name: str) -> str:
    direct_value = payload.get(field_name)
    if str(direct_value or "").strip():
        return str(direct_value).strip()
    quality_notes = payload.get("quality_notes", {})
    if isinstance(quality_notes, dict):
        note_value = quality_notes.get(field_name)
        if str(note_value or "").strip():
            return str(note_value).strip()
    return ""


def _payload_string_list(payload: dict[str, Any], field_name: str) -> list[str]:
    values = payload.get(field_name)
    if not isinstance(values, list):
        quality_notes = payload.get("quality_notes", {})
        values = quality_notes.get(field_name) if isinstance(quality_notes, dict) else []
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item or "").strip()]


def _build_reviewed_ocr_decision(
    *,
    source_path: str,
    status: str,
    extraction_quality: str,
    ocr_status: str,
    ocr_derived: bool,
    source_type: str,
    source_classification: str,
    source_family: str,
    source_role_boundary: str,
    document_kind: str,
    madhhab: str,
    author: str,
    duplicate_status: dict[str, str] | None,
    warnings: list[str],
    notes: list[str],
    ocr_quality_flags: list[str],
    fallback_reason: str,
) -> AdmissionDecision | None:
    review = _load_ocr_admission_review_lookup().get(source_path)
    if not review:
        return None
    reviewed_status = str(review.get("status", "") or "").strip()
    if reviewed_status not in {"admit", "admit_with_warnings", "defer", "reject"}:
        return None
    reviewed_reason = str(review.get("reason", "") or "").strip() or fallback_reason
    review_notes = [
        str(item)
        for item in review.get("notes", []) or []
        if str(item or "").strip()
    ]
    merged_notes = _dedupe_preserve_order(
        list(notes)
        + list(warnings)
        + list(ocr_quality_flags)
        + ["manual_ocr_review_applied"]
        + review_notes
    )
    return AdmissionDecision(
        source_path=source_path,
        status=reviewed_status,
        reason=reviewed_reason,
        notes=merged_notes,
        extraction_status=status,
        extraction_quality=extraction_quality,
        ocr_status=ocr_status,
        ocr_derived=ocr_derived,
        duplicate_status=duplicate_status.get("relation", "") if duplicate_status else "",
        canonical_source_path=duplicate_status.get("canonical_source_path", "") if duplicate_status else "",
        source_type=source_type,
        source_classification=source_classification,
        source_family=source_family,
        source_role_boundary=source_role_boundary,
        document_kind=document_kind,
        madhhab=madhhab,
        author=author,
    )


@lru_cache(maxsize=1)
def _load_ocr_admission_review_lookup() -> dict[str, dict[str, Any]]:
    configured = str(os.getenv("HALAL_JORDAN_OCR_ADMISSION_REVIEW_PATH", "") or "").strip()
    if configured:
        path = Path(configured).expanduser()
    else:
        path = Path(__file__).resolve().parents[2] / "config" / "ocr_admission_reviews.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source_path = str(record.get("source_path", "") or "").strip()
        if not source_path:
            continue
        lookup[source_path] = record
    return lookup


def _document_text_fingerprint(document: Any) -> str:
    loader_hint = str(getattr(document, "loader_hint", "") or "").strip()
    if loader_hint == "normalized_pdf_json":
        payload = _load_payload_if_available(document)
        page_texts: list[str] = []
        for page in payload.get("pages", []):
            if not isinstance(page, dict):
                continue
            text = str(page.get("text", "") or "").strip()
            if text:
                page_texts.append(text)
            if len(" ".join(page_texts)) >= 8000:
                break
        text = " ".join(page_texts)
    else:
        text = str(getattr(document, "content", "") or "")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:8000]


def _duplicate_canonical_score(document: Any) -> tuple[int, int, int, int]:
    extraction_status = str(getattr(document, "extraction_status", "") or "").strip().lower()
    extraction_rank = {"success": 0, "partial": 1, "failed": 2, "": 3}.get(
        extraction_status,
        2,
    )
    warnings = _payload_warnings(_load_payload_if_available(document))
    source_path = str(getattr(document, "source_path", "") or "")
    copy_rank = int(bool(re.search(r"\(\d+\)$", source_path.rsplit(".", 1)[0])))
    return (
        extraction_rank,
        len(warnings),
        copy_rank,
        len(source_path),
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in deduped:
            continue
        deduped.append(candidate)
    return deduped
