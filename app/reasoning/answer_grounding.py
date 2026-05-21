"""Internal source grounding, evidence typing, and answer assembly helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.reasoning.authority_policy import (
    authority_role_label,
    resolve_authority_policy,
    resolve_legal_role,
)
from app.reasoning.config_loader import ContractArtifacts
from app.reasoning.intent_router import QueryIntent
from app.reasoning.scholar_resolver import load_scholar_profiles, match_source_to_scholar
from app.retrieval.chunker import chunk_documents
from app.retrieval.index_loader import bootstrap_local_corpus
from app.retrieval.metadata_normalizer import (
    format_madhhab_label,
    format_source_hierarchy,
    normalize_source_metadata,
)

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

QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "show",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}


@dataclass(slots=True)
class GroundedSource:
    title: str
    human_title: str
    source_classification: str
    source_type_label: str
    evidence_bucket: str
    role: str
    domain: str
    authority_level: str
    reference: str
    section_label: str
    quote: str
    madhhab: str
    source_path: str
    collection: str
    author: str
    source_family: str
    canonical_family: str
    language: str
    hierarchy_label: str
    book: str
    chapter: str
    section: str
    document_kind: str
    source_role_boundary: str
    source_lineage: str
    commentary_target: str
    fatwa_authority: str
    legal_role: str
    legal_role_label: str
    ocr_derived: bool
    ocr_backend: str
    ocr_status: str
    ocr_confidence: str
    extraction_status: str
    extraction_quality: str
    scholar_attribution_match: str


@dataclass(slots=True)
class ComparisonPosition:
    label: str
    supporting_sources: list[GroundedSource]
    source_types_used: list[str]
    uncertainty_notes: list[str]


@dataclass(slots=True)
class AnswerEvidenceModel:
    primary_evidence: list[GroundedSource]
    spiritual_guidance: list[GroundedSource]
    hanafi_authority: list[GroundedSource]
    other_views: list[GroundedSource]
    supporting_commentary: list[GroundedSource]
    teaching_explanation: list[GroundedSource]
    modern_application: list[GroundedSource]
    sources: list[GroundedSource]
    disagreement_notes: list[str]
    uncertainty_notes: list[str]
    intent_id: str
    suppress_synthesis: bool
    authority_policy_id: str
    comparison_positions: list[ComparisonPosition]
    evidence_backfill_applied: bool
    evidence_backfill_buckets: list[str]
    source_layer_composition: dict[str, int]
    metadata_completeness: dict[str, Any]
    ocr_usage: dict[str, Any]
    teaching_layer_used: bool = False
    teaching_sources_count: int = 0
    teaching_layer_reason: str = ""
    teaching_layer_excluded_reason: str = ""
    scholar_attribution: dict[str, str] = field(default_factory=dict)
    scholar_direct_sources: list[GroundedSource] = field(default_factory=list)
    scholar_context_sources: list[GroundedSource] = field(default_factory=list)
    scholar_thin_attribution_note: str = ""
    unknown_scholar: str = ""


class AnswerGrounder:
    """Enrich retrieved snippets and assemble a source-aware evidence view."""

    _GLOBAL_CHUNK_INDEX: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    def __init__(
        self,
        repo_root: Path,
        artifacts: ContractArtifacts,
        *,
        quote_window_chars: int = 180,
    ) -> None:
        self.repo_root = repo_root
        self.artifacts = artifacts
        self.quote_window_chars = quote_window_chars
        self._chunk_index: dict[tuple[str, str], dict[str, Any]] | None = None
        self._source_type_labels = {
            str(item["id"]): str(item["label"])
            for item in artifacts.source_type_registry["source_types"]
        }

    @classmethod
    def prime_chunk_index(
        cls,
        repo_root: Path,
        chunks: list[dict[str, Any]],
    ) -> None:
        cls._GLOBAL_CHUNK_INDEX[str(repo_root.resolve())] = {
            cls._lookup_key_static(chunk): chunk
            for chunk in chunks
        }

    def enrich_sources(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
        retrieved_sources: list[dict[str, Any]],
        query_intent: QueryIntent | None = None,
    ) -> list[dict[str, Any]]:
        chunk_index = self._get_chunk_index(retrieved_sources)
        authority_policy = resolve_authority_policy(
            query_intent=query_intent,
            answer_mode=answer_mode,
            selected_madhhab=selected_madhhab,
        )
        enriched_sources: list[dict[str, Any]] = []
        for source in retrieved_sources:
            chunk = chunk_index.get(self._lookup_key(source)) if chunk_index else None
            chunk_payload = chunk or {}
            merged_source = dict(chunk or {})
            merged_source.update(source)
            normalized_source = normalize_source_metadata(merged_source)
            enriched = {
                **source,
                **normalized_source,
            }
            source_classification = str(
                normalized_source.get("source_classification", "unknown") or "unknown"
            )
            human_title = _humanize_title(
                title=str(normalized_source.get("title", "") or source.get("title", "") or ""),
                collection=str(
                    normalized_source.get("collection", "") or source.get("collection", "") or ""
                ),
                source_path=str(
                    normalized_source.get("source_path", "") or source.get("source_path", "") or ""
                ),
            )
            section_label = str(
                normalized_source.get("section_label", "")
                or chunk_payload.get("section_label", "")
            ).strip()
            quote_text = str(
                normalized_source.get("text", "") or chunk_payload.get("text", "")
            )
            quote_window = _extract_quote_window(
                text=quote_text or str(source.get("quote", "") or ""),
                question=question,
                fallback=str(source.get("quote", "") or ""),
                max_chars=self.quote_window_chars,
            )
            enriched["title"] = human_title or str(source.get("title", "") or "")
            enriched["human_title"] = enriched["title"]
            if quote_window:
                enriched["quote"] = quote_window
                enriched["quote_window"] = quote_window
            enriched["source_classification"] = source_classification
            enriched["source_type_label"] = self._source_type_label(source_classification)
            enriched["legal_role"] = resolve_legal_role(enriched)
            enriched["legal_role_label"] = authority_role_label(enriched["legal_role"])
            enriched["authority_policy_id"] = authority_policy.policy_id
            enriched["evidence_bucket"] = _assign_evidence_bucket(
                source_classification=source_classification,
                source=enriched,
                selected_madhhab=selected_madhhab,
                query_intent=query_intent,
            )
            enriched["answer_mode_hook"] = _answer_mode_hook(
                answer_mode=answer_mode,
                selected_madhhab=selected_madhhab,
                query_intent=query_intent,
            )
            if section_label:
                enriched["section_label"] = section_label
            enriched["hierarchy_label"] = format_source_hierarchy(enriched)
            if chunk:
                for key in (
                    "page_reference",
                    "page_number",
                    "page_chunk_index",
                    "page_chunk_total",
                    "section_kind",
                    "loader_hint",
                ):
                    value = chunk.get(key)
                    if value not in (None, ""):
                        enriched[key] = value
            enriched_sources.append(enriched)
        return enriched_sources

    def build_evidence_model(
        self,
        *,
        answer: dict[str, Any],
        selected_madhhab: str,
        answer_mode: str,
        enriched_sources: list[dict[str, Any]],
        all_enriched_sources: list[dict[str, Any]] | None = None,
        query_intent: QueryIntent | None = None,
    ) -> AnswerEvidenceModel:
        authority_policy = resolve_authority_policy(
            query_intent=query_intent,
            answer_mode=answer_mode,
            selected_madhhab=selected_madhhab,
        )
        sources, backfill_buckets = self._sources_for_answer(
            answer,
            enriched_sources,
            answer_mode=answer_mode,
            query_intent=query_intent,
        )
        primary_evidence: list[GroundedSource] = []
        spiritual_guidance: list[GroundedSource] = []
        hanafi_authority: list[GroundedSource] = []
        other_views: list[GroundedSource] = []
        supporting_commentary: list[GroundedSource] = []
        teaching_explanation: list[GroundedSource] = []
        modern_application: list[GroundedSource] = []
        assembled_sources: list[GroundedSource] = []

        for source in sources:
            entry = GroundedSource(
                title=str(source.get("title", "") or ""),
                human_title=str(source.get("title", "") or ""),
                source_classification=str(
                    source.get("source_classification", "unknown") or "unknown"
                ),
                source_type_label=str(
                    source.get("source_type_label", "Unknown") or "Unknown"
                ),
                evidence_bucket=str(source.get("evidence_bucket", "") or ""),
                role=str(source.get("role", "") or ""),
                domain=str(source.get("domain", "") or ""),
                authority_level=str(source.get("authority_level", "") or ""),
                reference=str(source.get("reference", "") or ""),
                section_label=str(source.get("section_label", "") or ""),
                quote=str(source.get("quote_window", source.get("quote", "")) or ""),
                madhhab=str(source.get("madhhab", "") or ""),
                source_path=str(source.get("source_path", "") or ""),
                collection=str(source.get("collection", "") or ""),
                author=str(source.get("author", "") or ""),
                source_family=str(source.get("source_family", "") or ""),
                canonical_family=str(source.get("canonical_family", "") or ""),
                language=str(source.get("language", "") or ""),
                hierarchy_label=str(source.get("hierarchy_label", "") or ""),
                book=str(source.get("book", "") or ""),
                chapter=str(source.get("chapter", "") or ""),
                section=str(source.get("section", "") or ""),
                document_kind=str(source.get("document_kind", "") or ""),
                source_role_boundary=str(source.get("source_role_boundary", "") or ""),
                source_lineage=str(source.get("source_lineage", "") or ""),
                commentary_target=str(source.get("commentary_target", "") or ""),
                fatwa_authority=str(source.get("fatwa_authority", "") or ""),
                legal_role=str(source.get("legal_role", "") or resolve_legal_role(source)),
                legal_role_label=str(
                    source.get("legal_role_label", "") or authority_role_label(
                        str(source.get("legal_role", "") or resolve_legal_role(source))
                    )
                ),
                ocr_derived=bool(source.get("ocr_derived", False)),
                ocr_backend=str(source.get("ocr_backend", "") or ""),
                ocr_status=str(source.get("ocr_status", "") or ""),
                ocr_confidence=str(source.get("ocr_confidence", "") or ""),
                extraction_status=str(source.get("extraction_status", "") or ""),
                extraction_quality=str(source.get("extraction_quality", "") or ""),
                scholar_attribution_match=str(source.get("scholar_attribution_match", "") or ""),
            )
            assembled_sources.append(entry)
            if entry.evidence_bucket == "primary_evidence":
                primary_evidence.append(entry)
            elif entry.evidence_bucket == "spiritual_guidance":
                spiritual_guidance.append(entry)
            elif entry.evidence_bucket == "hanafi_authority":
                hanafi_authority.append(entry)
            elif entry.evidence_bucket == "other_views":
                other_views.append(entry)
            elif entry.evidence_bucket == "modern_application":
                modern_application.append(entry)
            elif entry.evidence_bucket == "teaching_explanation":
                teaching_explanation.append(entry)
            else:
                supporting_commentary.append(entry)

        disagreement_notes = _collect_disagreement_notes(
            answer=answer,
            sources=assembled_sources,
            answer_mode=answer_mode,
            query_intent=query_intent,
        )
        uncertainty_notes = _collect_uncertainty_notes(
            answer=answer,
            sources=assembled_sources,
            selected_madhhab=selected_madhhab,
            primary_count=len(primary_evidence),
            hanafi_count=len(hanafi_authority),
            query_intent=query_intent,
        )
        comparison_positions = _build_comparison_positions(
            sources=assembled_sources,
            selected_madhhab=selected_madhhab,
            authority_policy_id=authority_policy.policy_id,
            preserve_distinct_positions=authority_policy.preserve_distinct_positions,
        )
        scholar_attribution = _build_scholar_attribution(
            repo_root=self.repo_root,
            sources=assembled_sources,
            query_intent=query_intent,
        )
        if scholar_attribution["thin_note"]:
            uncertainty_notes = _dedupe_preserve_order(
                [*uncertainty_notes, scholar_attribution["thin_note"]]
            )
        teaching_layer_diagnostics = _teaching_layer_diagnostics(
            all_sources=all_enriched_sources if all_enriched_sources is not None else enriched_sources,
            selected_sources=assembled_sources,
            query_intent=query_intent,
            answer_mode=answer_mode,
        )
        return AnswerEvidenceModel(
            primary_evidence=primary_evidence,
            spiritual_guidance=spiritual_guidance,
            hanafi_authority=hanafi_authority,
            other_views=other_views,
            supporting_commentary=supporting_commentary,
            teaching_explanation=teaching_explanation,
            modern_application=modern_application,
            sources=assembled_sources,
            disagreement_notes=disagreement_notes,
            uncertainty_notes=uncertainty_notes,
            intent_id=query_intent.intent_id if query_intent is not None else answer_mode,
            suppress_synthesis=(
                query_intent.suppress_synthesis if query_intent is not None else answer_mode == "source_only"
            ),
            authority_policy_id=authority_policy.policy_id,
            comparison_positions=comparison_positions,
            evidence_backfill_applied=bool(backfill_buckets),
            evidence_backfill_buckets=backfill_buckets,
            source_layer_composition=_source_layer_composition(
                primary_evidence=primary_evidence,
                spiritual_guidance=spiritual_guidance,
                hanafi_authority=hanafi_authority,
                other_views=other_views,
                supporting_commentary=supporting_commentary,
                teaching_explanation=teaching_explanation,
                modern_application=modern_application,
            ),
            metadata_completeness=_metadata_completeness_summary(assembled_sources),
            ocr_usage=_ocr_usage_summary(assembled_sources),
            teaching_layer_used=teaching_layer_diagnostics["used"],
            teaching_sources_count=teaching_layer_diagnostics["count"],
            teaching_layer_reason=teaching_layer_diagnostics["reason"],
            teaching_layer_excluded_reason=teaching_layer_diagnostics["excluded_reason"],
            scholar_attribution=scholar_attribution["header"],
            scholar_direct_sources=scholar_attribution["direct_sources"],
            scholar_context_sources=scholar_attribution["context_sources"],
            scholar_thin_attribution_note=scholar_attribution["thin_note"],
            unknown_scholar=query_intent.unknown_scholar if query_intent is not None else "",
        )

    def _classify_source_type(
        self,
        source: dict[str, Any],
        chunk: dict[str, Any] | None,
    ) -> str:
        merged = dict(chunk or {})
        merged.update(source)
        classification = str(
            normalize_source_metadata(merged).get("source_classification", "unknown")
            or "unknown"
        ).strip().lower()
        if classification in CANONICAL_SOURCE_TYPES:
            return classification
        return "unknown"

    def _source_type_label(self, source_classification: str) -> str:
        if source_classification == "unknown":
            return "Unknown"
        return self._source_type_labels.get(source_classification, source_classification)

    def _get_chunk_index(
        self,
        retrieved_sources: list[dict[str, Any]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if self._chunk_index is not None:
            return self._chunk_index

        repo_key = str(self.repo_root.resolve())
        cached_index = self._GLOBAL_CHUNK_INDEX.get(repo_key)
        if cached_index is not None:
            self._chunk_index = cached_index
            return self._chunk_index

        if not any(
            str(source.get("source_path", "")).replace("\\", "/").startswith("data/")
            for source in retrieved_sources
        ):
            self._chunk_index = {}
            return self._chunk_index

        corpus = bootstrap_local_corpus(self.repo_root)
        chunks = chunk_documents(corpus.documents)
        self.prime_chunk_index(self.repo_root, chunks)
        self._chunk_index = self._GLOBAL_CHUNK_INDEX.get(repo_key, {})
        return self._chunk_index

    def _sources_for_answer(
        self,
        answer: dict[str, Any],
        enriched_sources: list[dict[str, Any]],
        *,
        answer_mode: str,
        query_intent: QueryIntent | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        candidate_sources = _constrain_sources_for_answer(
            enriched_sources,
            query_intent=query_intent,
            answer_mode=answer_mode,
        )
        index_by_reference: dict[tuple[str, str], dict[str, Any]] = {}
        index_by_title: dict[tuple[str, str, str], dict[str, Any]] = {}
        for source in candidate_sources:
            index_by_reference[self._lookup_key(source)] = source
            index_by_title[self._citation_lookup_key(source)] = source

        selected_sources: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        backfill_buckets: list[str] = []
        for citation in answer.get("citations", []):
            source = index_by_reference.get(self._lookup_key(citation)) or index_by_title.get(
                self._citation_lookup_key(citation)
            )
            if source is None:
                source = normalize_source_metadata(dict(citation))
                source["source_classification"] = self._classify_source_type(source, None)
                source["source_type_label"] = self._source_type_label(
                    str(source["source_classification"])
                )
                source["legal_role"] = resolve_legal_role(source)
                source["legal_role_label"] = authority_role_label(source["legal_role"])
                source["evidence_bucket"] = _assign_evidence_bucket(
                    source_classification=str(source["source_classification"]),
                    source=source,
                    selected_madhhab=str(answer.get("selected_madhhab", "not_specified")),
                    query_intent=query_intent,
                )
            if not _source_allowed_for_answer(
                source,
                query_intent=query_intent,
                answer_mode=answer_mode,
            ):
                continue
            lookup_key = self._lookup_key(source)
            if lookup_key in seen:
                continue
            selected_sources.append(source)
            seen.add(lookup_key)

        if selected_sources:
            selected_sources, backfill_buckets = self._backfill_required_layers(
                selected_sources=selected_sources,
                candidate_sources=candidate_sources,
                answer=answer,
                query_intent=query_intent,
            )
            selected_sources = _constrain_sources_for_answer(
                selected_sources,
                query_intent=query_intent,
                answer_mode=answer_mode,
            )

        if not selected_sources:
            for source in candidate_sources:
                lookup_key = self._lookup_key(source)
                if lookup_key in seen:
                    continue
                selected_sources.append(source)
                seen.add(lookup_key)

        return selected_sources, backfill_buckets

    def _backfill_required_layers(
        self,
        *,
        selected_sources: list[dict[str, Any]],
        candidate_sources: list[dict[str, Any]],
        answer: dict[str, Any],
        query_intent: QueryIntent | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not candidate_sources:
            return selected_sources, []

        selected = list(selected_sources)
        seen = {self._lookup_key(source) for source in selected}
        backfill_buckets: list[str] = []

        def has_bucket(bucket: str) -> bool:
            return any(str(source.get("evidence_bucket", "") or "") == bucket for source in selected)

        def add_first_bucket(bucket: str) -> None:
            for source in candidate_sources:
                if str(source.get("evidence_bucket", "") or "") != bucket:
                    continue
                lookup_key = self._lookup_key(source)
                if lookup_key in seen:
                    continue
                selected.append(source)
                seen.add(lookup_key)
                backfill_buckets.append(bucket)
                return

        if not has_bucket("primary_evidence"):
            add_first_bucket("primary_evidence")

        selected_madhhab = str(answer.get("selected_madhhab", "") or "").strip().lower()
        if selected_madhhab == "hanafi" and not has_bucket("hanafi_authority"):
            add_first_bucket("hanafi_authority")

        if query_intent is not None and query_intent.preserve_disagreement and not has_bucket("other_views"):
            add_first_bucket("other_views")

        return selected, backfill_buckets

    @staticmethod
    def _lookup_key(source: dict[str, Any]) -> tuple[str, str]:
        return (
            str(source.get("source_path", "") or "").strip().lower(),
            str(source.get("reference", "") or "").strip().lower(),
        )

    @staticmethod
    def _lookup_key_static(source: dict[str, Any]) -> tuple[str, str]:
        return (
            str(source.get("source_path", "") or "").strip().lower(),
            str(source.get("reference", "") or "").strip().lower(),
        )

    @staticmethod
    def _citation_lookup_key(source: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(source.get("title", "") or "").strip().lower(),
            str(source.get("reference", "") or "").strip().lower(),
            str(source.get("source_type", "") or "").strip().lower(),
        )


def _assign_evidence_bucket(
    *,
    source_classification: str,
    source: dict[str, Any],
    selected_madhhab: str,
    query_intent: QueryIntent | None,
) -> str:
    if _is_teaching_layer_source_dict(source):
        return "teaching_explanation"
    madhhab = str(source.get("madhhab", "") or "").strip().lower()
    legal_role = str(source.get("legal_role", "") or resolve_legal_role(source))
    if legal_role == "primary_text":
        return "primary_evidence"
    if legal_role == "spiritual_guidance":
        return "spiritual_guidance"
    if legal_role == "madhhab_authority":
        if selected_madhhab == "hanafi" and madhhab == "hanafi":
            return "hanafi_authority"
        if (
            query_intent is not None
            and query_intent.preserve_disagreement
            and madhhab
            and madhhab != selected_madhhab
        ):
            return "other_views"
        if selected_madhhab not in {"not_specified", "compare_all", ""} and madhhab:
            return "hanafi_authority" if madhhab == selected_madhhab else "other_views"
        if madhhab and madhhab != "hanafi":
            return "other_views"
        return "supporting_commentary"
    if legal_role == "explanatory_commentary" or source_classification == "unknown":
        if (
            query_intent is not None
            and query_intent.intent_id == "compare_views"
            and madhhab
            and madhhab != selected_madhhab
        ):
            return "other_views"
        return "supporting_commentary"
    if legal_role in {"modern_application", "informal_explanation"}:
        return "modern_application"
    return "supporting_commentary"


def _answer_mode_hook(
    *,
    answer_mode: str,
    selected_madhhab: str,
    query_intent: QueryIntent | None,
) -> str:
    if query_intent is not None and query_intent.intent_id == "source_only":
        return "source_only"
    if query_intent is not None and query_intent.intent_id == "compare_views":
        return "compare_views"
    if answer_mode == "source_only":
        return "source_only"
    if answer_mode == "compare_views":
        return "compare_views"
    if selected_madhhab == "hanafi":
        return "hanafi_first"
    return "research"


def _collect_disagreement_notes(
    *,
    answer: dict[str, Any],
    sources: list[GroundedSource],
    answer_mode: str,
    query_intent: QueryIntent | None,
) -> list[str]:
    notes: list[str] = []
    explicit_note = str(answer.get("disagreement_note", "") or "").strip()
    if explicit_note:
        notes.append(explicit_note)

    madhhabs = sorted(
        {
            format_madhhab_label(source.madhhab) or source.madhhab
            for source in sources
            if source.madhhab
        }
    )
    if len(madhhabs) > 1:
        notes.append(
            "The retrieved fiqh materials reflect more than one madhhab perspective: "
            + ", ".join(madhhabs)
            + "."
        )
    if (
        (answer_mode == "compare_views" or (query_intent and query_intent.preserve_disagreement))
        and len(sources) > 1
    ):
        notes.append(
            "The retrieved materials should be read as parallel views rather than a single flattened position."
        )
    return _dedupe_preserve_order(notes)


def _collect_uncertainty_notes(
    *,
    answer: dict[str, Any],
    sources: list[GroundedSource],
    selected_madhhab: str,
    primary_count: int,
    hanafi_count: int,
    query_intent: QueryIntent | None,
) -> list[str]:
    notes: list[str] = []
    explicit_note = str(answer.get("uncertainty_note", "") or "").strip()
    if explicit_note:
        notes.append(explicit_note)

    if primary_count == 0 and sources:
        notes.append(
            "No primary text was retrieved in this set, so the answer rests on later legal or explanatory materials."
        )
    if selected_madhhab == "hanafi" and hanafi_count == 0 and sources:
        notes.append(
            "No explicitly Hanafi authority was retrieved in this set."
        )
    if any(source.evidence_bucket == "modern_application" for source in sources):
        notes.append(
            "Modern application materials should be read in light of the primary texts and school authorities, not as substitutes for them."
        )
    if any(source.evidence_bucket == "teaching_explanation" for source in sources):
        notes.append(
            "Teaching-layer materials in this answer support explanation and study context only; they do not function as primary text or madhhab authority."
        )
    if any(source.evidence_bucket == "spiritual_guidance" for source in sources):
        notes.append(
            "Tasawwuf materials in this answer reflect classical spiritual guidance, not legal rulings."
        )
    if any(source.source_classification == "unknown" for source in sources):
        notes.append(
            "Some retrieved items have incomplete source metadata, so their role in the evidence stack should be read cautiously."
        )
    if any(source.ocr_derived for source in sources):
        notes.append(
            "Some retrieved evidence is OCR-derived, so wording and citation boundaries should be checked with extra caution."
        )
    if any(source.extraction_status == "partial" for source in sources):
        notes.append(
            "Some retrieved sources come from partial extraction, so omitted text or section gaps may still exist."
        )
    if query_intent is not None and query_intent.intent_id == "compare_views" and len(
        {
            source.madhhab
            for source in sources
            if source.madhhab
        }
    ) < 2:
        notes.append(
            "Compare-views routing was requested, but the retrieved set does not yet show more than one clearly labeled madhhab view."
        )
    evidence_strength = str(answer.get("evidence_strength", "") or "").strip().lower()
    if evidence_strength in {"limited", "insufficient", "conflicting"}:
        notes.append(
            f"Evidence strength is currently assessed as {evidence_strength}."
        )
    return _dedupe_preserve_order(notes)


def _build_comparison_positions(
    *,
    sources: list[GroundedSource],
    selected_madhhab: str,
    authority_policy_id: str,
    preserve_distinct_positions: bool,
) -> list[ComparisonPosition]:
    if not preserve_distinct_positions and authority_policy_id != "compare_views":
        return []

    grouped: dict[str, list[GroundedSource]] = {}
    order: list[str] = []
    for source in sources:
        if source.evidence_bucket == "teaching_explanation":
            continue
        label = _comparison_label_for_source(source, selected_madhhab)
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        grouped[label].append(source)

    positions: list[ComparisonPosition] = []
    for label in sorted(order, key=lambda value: _comparison_position_sort_key(value, selected_madhhab)):
        grouped_sources = grouped[label]
        source_types_used = _dedupe_preserve_order(
            [source.source_type_label for source in grouped_sources if source.source_type_label]
        )
        uncertainty_notes: list[str] = []
        roles = {source.legal_role for source in grouped_sources}
        if label == "Shared Primary Texts":
            uncertainty_notes.append(
                "These shared texts inform the comparison but do not by themselves settle school-level application."
            )
        if "madhhab_authority" not in roles and roles & {
            "explanatory_commentary",
            "modern_application",
            "informal_explanation",
        }:
            uncertainty_notes.append(
                "This position is represented without a direct fiqh-manual authority in the retrieved set."
            )
        if len(grouped_sources) == 1:
            uncertainty_notes.append(
                "This position is supported by a single retrieved source in this set."
            )
        positions.append(
            ComparisonPosition(
                label=label,
                supporting_sources=grouped_sources,
                source_types_used=source_types_used,
                uncertainty_notes=_dedupe_preserve_order(uncertainty_notes),
            )
        )
    return positions


def _comparison_label_for_source(
    source: GroundedSource,
    selected_madhhab: str,
) -> str:
    if source.legal_role == "primary_text":
        return "Shared Primary Texts"
    if source.madhhab == "comparative":
        return "Comparative Context"
    selected_label = format_madhhab_label(selected_madhhab)
    source_label = format_madhhab_label(source.madhhab)
    if source.madhhab == selected_madhhab and selected_madhhab not in {"", "not_specified", "compare_all"}:
        return f"{selected_label} Position"
    if source.madhhab:
        return f"{source_label} Position"
    return "Unspecified Supporting Material"


def _comparison_position_sort_key(label: str, selected_madhhab: str) -> tuple[int, str]:
    selected_label = format_madhhab_label(selected_madhhab)
    if selected_madhhab not in {"", "not_specified", "compare_all"} and label == f"{selected_label} Position":
        return (0, label)
    if label.endswith(" Position"):
        return (1, label)
    if label == "Shared Primary Texts":
        return (2, label)
    if label == "Comparative Context":
        return (3, label)
    return (4, label)


def _build_scholar_attribution(
    *,
    repo_root: Path,
    sources: list[GroundedSource],
    query_intent: QueryIntent | None,
) -> dict[str, Any]:
    empty = {
        "header": {},
        "direct_sources": [],
        "context_sources": [],
        "thin_note": "",
    }
    if query_intent is None or query_intent.intent_id != "scholar_perspective" or not query_intent.scholar_id:
        return empty
    profile = load_scholar_profiles(repo_root).get(query_intent.scholar_id)
    if profile is None:
        return empty

    direct_sources: list[GroundedSource] = []
    context_sources: list[GroundedSource] = []
    for source in sources:
        match_type = source.scholar_attribution_match or match_source_to_scholar(
            {
                "title": source.title,
                "collection": source.collection,
                "author": source.author,
                "source_path": source.source_path,
                "canonical_family": source.canonical_family,
                "commentary_target": source.commentary_target,
                "source_family": source.source_family,
                "source_classification": source.source_classification,
                "source_type": source.source_classification,
                "madhhab": source.madhhab,
            },
            profile,
        )
        if match_type == "direct":
            direct_sources.append(source)
        elif match_type == "contextual":
            context_sources.append(source)

    thin_note = ""
    if not direct_sources:
        thin_note = (
            f"No directly attributed material for {profile.name} was found in the retrieved set. "
            "Any supporting sources below are contextual only."
        )
    elif len(direct_sources) == 1:
        thin_note = (
            f"Retrieved attribution from {profile.name} is thin and should not be generalized beyond the cited material."
        )

    return {
        "header": {
            "scholar_id": profile.scholar_id,
            "name": profile.name,
            "madhhab": profile.madhhab,
            "period": profile.period,
            "disclaimer": f"Drawn from {profile.name}'s recorded works and positions.",
            "methodology_notes": profile.methodology_notes,
        },
        "direct_sources": direct_sources,
        "context_sources": context_sources,
        "thin_note": thin_note,
    }


def _humanize_title(*, title: str, collection: str, source_path: str) -> str:
    candidate = title.strip()
    if not candidate:
        candidate = Path(source_path).stem if source_path else ""
    if _looks_machine_title(candidate):
        cleaned_collection = collection.strip()
        if cleaned_collection and not _looks_machine_title(cleaned_collection):
            candidate = cleaned_collection
        else:
            candidate = Path(source_path).stem if source_path else candidate
    candidate = candidate.replace("_", " ")
    candidate = re.sub(r"\.{2,}", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -._")
    return candidate or title or collection or "Retrieved Source"


def _looks_machine_title(value: str) -> bool:
    if not value:
        return True
    return (
        "_" in value
        or "." in value
        or bool(re.search(r"\bv\d", value, flags=re.IGNORECASE))
        or (len(value.split()) <= 2 and value.islower())
    )


def _extract_quote_window(
    *,
    text: str,
    question: str,
    fallback: str,
    max_chars: int = 180,
) -> str:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return _truncate_text(_normalize_text(fallback), max_chars=max_chars)

    phrase = _normalize_text(question)
    lowered_text = normalized_text.casefold()
    if phrase and len(phrase) >= 8:
        position = lowered_text.find(phrase.casefold())
        if position >= 0:
            return _window_around_position(normalized_text, position, max_chars=max_chars)

    tokens = _question_tokens(question)
    positions = [
        (token, lowered_text.find(token.casefold()))
        for token in sorted(tokens, key=len, reverse=True)
    ]
    for token, position in positions:
        if position >= 0:
            return _window_around_position(normalized_text, position, max_chars=max_chars)
    return _truncate_text(normalized_text, max_chars=max_chars)


def _window_around_position(text: str, position: int, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    start = max(0, position - max_chars // 3)
    end = min(len(text), start + max_chars)
    if start > 0:
        start = text.rfind(" ", 0, start) + 1 or start
    if end < len(text):
        trailing_boundary = text.find(" ", end)
        if trailing_boundary > 0:
            end = trailing_boundary
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "..."


def _question_tokens(question: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", question or "").casefold()
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    filtered: list[str] = []
    for token in tokens:
        if token in QUESTION_STOPWORDS:
            continue
        if token.isascii() and len(token) < 3:
            continue
        filtered.append(token)
    return filtered


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _source_layer_composition(
    *,
    primary_evidence: list[GroundedSource],
    spiritual_guidance: list[GroundedSource],
    hanafi_authority: list[GroundedSource],
    other_views: list[GroundedSource],
    supporting_commentary: list[GroundedSource],
    teaching_explanation: list[GroundedSource],
    modern_application: list[GroundedSource],
) -> dict[str, int]:
    return {
        "primary_evidence": len(primary_evidence),
        "spiritual_guidance": len(spiritual_guidance),
        "hanafi_authority": len(hanafi_authority),
        "other_views": len(other_views),
        "supporting_commentary": len(supporting_commentary),
        "teaching_explanation": len(teaching_explanation),
        "modern_application": len(modern_application),
    }


def _is_teaching_layer_source_dict(source: dict[str, Any]) -> bool:
    source_role_boundary = str(source.get("source_role_boundary", "") or "").strip().lower()
    source_family = str(source.get("source_family", "") or "").strip().lower()
    return source_role_boundary == "teaching_layer" or source_family == "classes"


def _is_source_only_discipline(
    *,
    query_intent: QueryIntent | None,
    answer_mode: str,
) -> bool:
    return (
        (query_intent is not None and query_intent.intent_id == "source_only")
        or answer_mode == "source_only"
    )


def _is_direct_source_lookup_discipline(query_intent: QueryIntent | None) -> bool:
    return query_intent is not None and query_intent.intent_id == "direct_source_lookup"


def _is_primary_source_dict(source: dict[str, Any]) -> bool:
    legal_role = str(source.get("legal_role", "") or resolve_legal_role(source))
    return legal_role == "primary_text"


def _source_allowed_for_answer(
    source: dict[str, Any],
    *,
    query_intent: QueryIntent | None,
    answer_mode: str,
) -> bool:
    if _is_source_only_discipline(
        query_intent=query_intent,
        answer_mode=answer_mode,
    ):
        return _is_primary_source_dict(source)
    if _is_direct_source_lookup_discipline(query_intent) and _is_teaching_layer_source_dict(source):
        return False
    return True


def _constrain_sources_for_answer(
    sources: list[dict[str, Any]],
    *,
    query_intent: QueryIntent | None,
    answer_mode: str,
) -> list[dict[str, Any]]:
    return [
        source
        for source in sources
        if _source_allowed_for_answer(
            source,
            query_intent=query_intent,
            answer_mode=answer_mode,
        )
    ]


def _teaching_layer_diagnostics(
    *,
    all_sources: list[dict[str, Any]],
    selected_sources: list[GroundedSource],
    query_intent: QueryIntent | None,
    answer_mode: str,
) -> dict[str, Any]:
    available_count = sum(1 for source in all_sources if _is_teaching_layer_source_dict(source))
    used_count = sum(1 for source in selected_sources if source.evidence_bucket == "teaching_explanation")
    excluded_reason = ""
    reason = ""
    if used_count:
        reason = "explanation_support"
        if answer_mode == "study_path":
            reason = "study_mode_support"
    elif available_count and query_intent is not None and query_intent.intent_id in {"source_only", "direct_source_lookup"}:
        excluded_reason = "excluded_for_source_discipline"
    elif available_count and query_intent is not None and query_intent.intent_id == "ruling_lookup":
        excluded_reason = "downranked_below_manual_authority"
    return {
        "used": used_count > 0,
        "count": used_count,
        "reason": reason,
        "excluded_reason": excluded_reason,
    }


def _metadata_completeness_summary(sources: list[GroundedSource]) -> dict[str, Any]:
    total = max(len(sources), 1)
    tracked_fields = (
        "author",
        "language",
        "collection",
        "source_lineage",
        "source_role_boundary",
    )
    summary: dict[str, Any] = {
        "source_count": len(sources),
        "coverage": {},
    }
    for field_name in tracked_fields:
        known = sum(
            1
            for source in sources
            if str(getattr(source, field_name, "") or "").strip().lower() not in {"", "unknown"}
        )
        summary["coverage"][field_name] = {
            "known": known,
            "unknown": len(sources) - known,
            "coverage_percent": round((known / total) * 100, 1),
        }
    return summary


def _ocr_usage_summary(sources: list[GroundedSource]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    backends: dict[str, int] = {}
    extraction_qualities: dict[str, int] = {}
    for source in sources:
        status = source.ocr_status or "not_attempted"
        statuses[status] = statuses.get(status, 0) + 1
        if source.ocr_backend:
            backends[source.ocr_backend] = backends.get(source.ocr_backend, 0) + 1
        quality = source.extraction_quality or "unknown"
        extraction_qualities[quality] = extraction_qualities.get(quality, 0) + 1
    return {
        "source_count": len(sources),
        "ocr_derived_sources": sum(1 for source in sources if source.ocr_derived),
        "ocr_status_counts": statuses,
        "ocr_backend_counts": backends,
        "extraction_quality_counts": extraction_qualities,
    }
