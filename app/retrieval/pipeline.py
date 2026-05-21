"""End-to-end retrieval pipeline returning ask-compatible snippets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from time import perf_counter
from typing import Any

from app.backend.runtime_config import RuntimeConfig, RuntimeConfigStore
from app.reasoning.config_loader import ContractArtifacts, load_contract_artifacts
from app.reasoning.intent_router import QueryIntent, route_query_intent
from app.reasoning.scholar_resolver import load_scholar_profiles, match_source_to_scholar
from app.retrieval.chunker import chunk_documents
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.index_loader import CorpusBootstrap, bootstrap_local_corpus_with_paths
from app.retrieval.persisted_index import (
    inspect_persisted_retrieval_index,
    load_persisted_retrieval_index,
)
from app.retrieval.reranker import get_duplicate_family_key
from app.retrieval.reranker import rerank_candidates
from app.retrieval.search import hybrid_search, prepare_search_index, search_chunks
from app.retrieval.vector_store import LocalVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]

SNIPPET_FIELDS = (
    "title",
    "source_type",
    "source_classification",
    "source_role_flag",
    "source_role_boundary",
    "collection",
    "author",
    "madhhab",
    "language",
    "reference",
    "quote",
    "source_path",
    "source_family",
    "canonical_family",
    "document_kind",
    "book",
    "chapter",
    "section",
    "section_label",
    "hierarchy_label",
    "source_lineage",
    "role",
    "domain",
    "authority_level",
    "extraction_status",
    "extraction_quality",
    "ocr_derived",
    "ocr_backend",
    "ocr_status",
    "ocr_confidence",
    "ocr_confidence_band",
    "text_source_mix",
    # Collection-enrichment fields (filled by source_enrichment.py during
    # infer_document_metadata). These let the Evidence Ladder, Trust
    # Engine, and Confidence Taxonomy read structured signals directly
    # from the snippet without re-inferring.
    "era",
    "hadith_grade",
    "hadith_grade_source",
    "scholar_authority",
    "isnad_strength",
    "isnad_strength_source",
    "methodology_tags",
    "collection_enrichment_applied",
    "collection_enrichment_provenance",
)


@dataclass(slots=True)
class RetrievalState:
    corpus: CorpusBootstrap
    chunks: list[dict[str, Any]]


@dataclass(slots=True)
class RetrievalDebugResult:
    question: str
    selected_madhhab: str
    answer_mode: str
    query_intent: QueryIntent
    candidate_limit: int
    candidate_count: int
    snippets: list[dict[str, str]]
    top_source_classifications: dict[str, int]
    top_source_families: dict[str, int]
    top_collections: dict[str, int]
    top_sections: dict[str, int]
    corpus_stats: dict[str, int]
    retrieval_method: str = "keyword"
    bootstrap_source: str = "uninitialized"
    bootstrap_fallback_reason: str = ""
    prepared_search_loaded: bool = False
    prepared_search_fallback_reason: str = ""
    prepared_search_load_ms: int = 0
    prepared_search_build_ms: int = 0
    warmup_total_ms: int = 0
    detected_madhhab_intent: str = ""
    madhhab_boost_applied: bool = False
    madhhab_fallback_used: bool = False
    vector_status: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)
    trust_profile_id: str = "default"
    trust_diagnostics: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    """Load, index, search, and rerank local source snippets."""

    def __init__(
        self,
        repo_root: Path | None = None,
        artifacts: ContractArtifacts | None = None,
        raw_root: Path | None = None,
        normalized_root: Path | None = None,
        index_root: Path | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self.artifacts = artifacts or load_contract_artifacts(self.repo_root)
        self.raw_root = raw_root
        self.normalized_root = normalized_root
        self.index_root = index_root or (self.repo_root / "data" / "index")
        self.runtime_config = runtime_config or _load_runtime_config(self.repo_root)
        self._state: RetrievalState | None = None
        self._last_bootstrap_source = "uninitialized"
        self._last_bootstrap_reason = ""
        self._prepared_search_loaded = False
        self._prepared_search_fallback_reason = ""
        self._prepared_search_load_ms = 0
        self._prepared_search_build_ms = 0
        self._last_bootstrap_total_ms = 0
        self._embedder: LocalEmbedder | None = None
        self._vector_store: LocalVectorStore | None = None

    def bootstrap(
        self,
        *,
        force: bool = False,
        allow_corpus_fallback: bool = True,
    ) -> RetrievalState:
        bootstrap_started = perf_counter()
        if self._state is not None and not force:
            return self._state
        if force:
            self._vector_store = None
        if self.index_root is not None:
            raw_root = self.raw_root or (self.repo_root / "data" / "raw")
            normalized_root = self.normalized_root or (
                self.repo_root / "data" / "processed" / "normalized"
            )
            persisted = load_persisted_retrieval_index(
                self.repo_root,
                raw_root=raw_root,
                normalized_root=normalized_root,
                index_root=self.index_root,
            )
            if persisted is not None:
                corpus, chunks, _manifest, prepared_state = persisted
                self._state = RetrievalState(corpus=corpus, chunks=chunks)
                self._last_bootstrap_source = "persisted_index"
                self._last_bootstrap_reason = ""
                self._prepared_search_loaded = bool(
                    prepared_state.get("prepared_search_loaded", False)
                )
                self._prepared_search_fallback_reason = str(
                    prepared_state.get("prepared_search_fallback_reason", "") or ""
                )
                self._prepared_search_load_ms = int(
                    prepared_state.get("prepared_search_load_ms", 0) or 0
                )
                self._prepared_search_build_ms = int(
                    prepared_state.get("prepared_search_build_ms", 0) or 0
                )
                self._last_bootstrap_total_ms = int(
                    (perf_counter() - bootstrap_started) * 1000
                )
                return self._state
            index_status = inspect_persisted_retrieval_index(
                self.index_root,
                repo_root=self.repo_root,
                raw_root=raw_root,
                normalized_root=normalized_root,
            )
            self._last_bootstrap_reason = str(
                index_status.get("retrieval_fallback_reason", "") or ""
            )
            if not allow_corpus_fallback:
                fallback_reason = self._last_bootstrap_reason or "index_invalid"
                raise RuntimeError(f"persisted_index_required:{fallback_reason}")
        elif not allow_corpus_fallback:
            self._last_bootstrap_reason = "index_unconfigured"
            raise RuntimeError("persisted_index_required:index_unconfigured")
        corpus = bootstrap_local_corpus_with_paths(
            self.repo_root,
            raw_root=self.raw_root,
            normalized_root=self.normalized_root,
        )
        chunks = chunk_documents(corpus.documents)
        build_started = perf_counter()
        prepare_search_index(chunks)
        self._state = RetrievalState(corpus=corpus, chunks=chunks)
        self._last_bootstrap_source = "rebuilt_from_corpus"
        self._prepared_search_loaded = False
        self._prepared_search_fallback_reason = "retrieval_bootstrap_rebuilt_from_corpus"
        self._prepared_search_load_ms = 0
        self._prepared_search_build_ms = int((perf_counter() - build_started) * 1000)
        self._last_bootstrap_total_ms = int((perf_counter() - bootstrap_started) * 1000)
        return self._state

    def retrieve(
        self,
        question: str,
        *,
        selected_madhhab: str,
        top_k: int = 5,
        answer_mode: str = "research",
        candidate_limit: int | None = None,
        deadline: float | None = None,
        allow_corpus_fallback: bool = True,
    ) -> list[dict[str, str]]:
        return self.retrieve_with_debug(
            question,
            selected_madhhab=selected_madhhab,
            top_k=top_k,
            answer_mode=answer_mode,
            candidate_limit=candidate_limit,
            deadline=deadline,
            allow_corpus_fallback=allow_corpus_fallback,
        ).snippets

    def retrieve_with_debug(
        self,
        question: str,
        *,
        selected_madhhab: str,
        top_k: int = 5,
        answer_mode: str = "research",
        candidate_limit: int | None = None,
        deadline: float | None = None,
        allow_corpus_fallback: bool = True,
    ) -> RetrievalDebugResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        state = self.bootstrap(allow_corpus_fallback=allow_corpus_fallback)
        query_intent = route_query_intent(
            question=question,
            answer_mode=answer_mode,
            selected_madhhab=selected_madhhab,
        )
        effective_candidate_limit = candidate_limit or max(top_k * 50, 500)
        retrieval_started = perf_counter()
        _ask_log("[ASK] retrieval start")
        candidates, retrieval_method, vector_status = self._retrieve_candidates(
            question=question,
            chunks=state.chunks,
            top_k=effective_candidate_limit,
            query_intent=query_intent,
            deadline=deadline,
        )
        retrieval_ms = int((perf_counter() - retrieval_started) * 1000)
        _ask_log(f"[ASK] retrieval done: {retrieval_ms / 1000:.3f}")
        candidates = self._apply_scholar_perspective_boosts(
            candidates,
            query_intent=query_intent,
        )
        rerank_started = perf_counter()
        _ask_log("[ASK] rerank start")
        trust_profile_id = str(
            getattr(self.runtime_config, "trust_profile_id", "default") or "default"
        )
        reranked = rerank_candidates(
            candidates,
            selected_madhhab=selected_madhhab,
            retrieval_policy=self.artifacts.retrieval_policy,
            source_type_registry=self.artifacts.source_type_registry,
            top_k=top_k,
            query_intent=query_intent,
            answer_mode=answer_mode,
            trust_profile_id=trust_profile_id,
        )
        rerank_ms = int((perf_counter() - rerank_started) * 1000)
        _ask_log(f"[ASK] rerank done: {rerank_ms / 1000:.3f}")
        snippets = [self._to_snippet(candidate) for candidate in reranked]
        trust_diagnostics = _trust_diagnostics(
            reranked=reranked,
            trust_profile_id=trust_profile_id,
        )
        madhhab_diagnostics = _madhhab_retrieval_diagnostics(
            candidates=candidates,
            reranked=reranked,
            query_intent=query_intent,
        )
        return RetrievalDebugResult(
            question=question,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            query_intent=query_intent,
            retrieval_method=retrieval_method,
            candidate_limit=effective_candidate_limit,
            candidate_count=len(candidates),
            snippets=snippets,
            top_source_classifications=_counter_for_candidates(
                reranked,
                field_name="source_classification",
                fallback_field="source_type",
            ),
            top_source_families=_counter_for_candidates(reranked, field_name="source_family"),
            top_collections=_counter_for_candidates(reranked, field_name="collection"),
            top_sections=_counter_for_candidates(reranked, field_name="section_label"),
            corpus_stats=self.stats(),
            vector_status=vector_status,
            bootstrap_source=self._last_bootstrap_source,
            bootstrap_fallback_reason=self._last_bootstrap_reason,
            prepared_search_loaded=self._prepared_search_loaded,
            prepared_search_fallback_reason=self._prepared_search_fallback_reason,
            prepared_search_load_ms=self._prepared_search_load_ms,
            prepared_search_build_ms=self._prepared_search_build_ms,
            warmup_total_ms=self._last_bootstrap_total_ms,
            detected_madhhab_intent=madhhab_diagnostics["detected_madhhab_intent"],
            madhhab_boost_applied=madhhab_diagnostics["madhhab_boost_applied"],
            madhhab_fallback_used=madhhab_diagnostics["madhhab_fallback_used"],
            timings_ms={
                "retrieval": retrieval_ms,
                "rerank": rerank_ms,
                "warmup_total": self._last_bootstrap_total_ms,
                "prepared_search_load": self._prepared_search_load_ms,
                "prepared_search_build": self._prepared_search_build_ms,
            },
            trust_profile_id=trust_profile_id,
            trust_diagnostics=trust_diagnostics,
        )

    def stats(self) -> dict[str, int]:
        state = self.bootstrap()
        return {
            "documents": len(state.corpus.documents),
            "chunks": len(state.chunks),
            "skipped_files": len(state.corpus.skipped_files),
        }

    def skipped_files(self) -> list[dict[str, str]]:
        return list(self.bootstrap().corpus.skipped_files)

    def _to_snippet(self, candidate: dict[str, Any]) -> dict[str, str]:
        snippet = {field: str(candidate.get(field, "") or "") for field in SNIPPET_FIELDS}
        if snippet["source_type"] not in self.artifacts.source_type_ids:
            raise ValueError(
                f"retrieval produced unsupported source_type: {snippet['source_type']}"
            )
        # Carry the trust breakdown through as a JSON-encoded string so the
        # snippet contract (dict[str, str]) is preserved. Downstream code that
        # cares about trust transparency parses _trust_breakdown_json; code
        # that doesn't is unaffected.
        trust_breakdown = candidate.get("_trust_breakdown")
        if trust_breakdown:
            import json as _json
            snippet["_trust_breakdown_json"] = _json.dumps(
                trust_breakdown, ensure_ascii=False
            )
            snippet["_trust_profile_id"] = str(trust_breakdown.get("profile_id", ""))
            snippet["_trust_total"] = str(trust_breakdown.get("total", "0"))
        return snippet

    def _retrieve_candidates(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int,
        query_intent: QueryIntent,
        deadline: float | None = None,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        vector_status = self._vector_status()
        if vector_status["hybrid_ready"]:
            try:
                candidates = hybrid_search(
                    question,
                    chunks,
                    vector_store=self._get_vector_store(),
                    embedder=self._get_embedder(),
                    top_k=top_k,
                    query_intent=query_intent,
                    keyword_weight=float(self.runtime_config.hybrid_weight_keyword),
                    semantic_weight=float(self.runtime_config.hybrid_weight_semantic),
                    deadline=deadline,
                )
                candidates = self._ensure_requested_madhhab_candidates(
                    question=question,
                    chunks=chunks,
                    candidates=candidates,
                    query_intent=query_intent,
                    deadline=deadline,
                )
                vector_status["active"] = True
                vector_status["fallback_reason"] = ""
                return candidates, "hybrid", vector_status
            except RuntimeError as exc:
                vector_status["active"] = False
                vector_status["fallback_reason"] = str(exc)
        candidates = search_chunks(
            question,
            chunks,
            top_k=top_k,
            query_intent=query_intent,
            deadline=deadline,
        )
        candidates = self._ensure_requested_madhhab_candidates(
            question=question,
            chunks=chunks,
            candidates=candidates,
            query_intent=query_intent,
            deadline=deadline,
        )
        if not vector_status["fallback_reason"]:
            vector_status["fallback_reason"] = vector_status["reason"]
        return candidates, "keyword", vector_status

    def _ensure_requested_madhhab_candidates(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        query_intent: QueryIntent,
        deadline: float | None,
    ) -> list[dict[str, Any]]:
        requested_madhhabs = tuple(query_intent.requested_madhhabs)
        if not requested_madhhabs:
            return candidates

        present_madhhabs = {
            str(candidate.get("madhhab", "")).strip().lower()
            for candidate in candidates
            if str(candidate.get("source_classification", "") or candidate.get("source_type", "")).strip().lower()
            in {"fiqh_manual", "commentary", "fatwa"}
        }
        missing_madhhabs = [
            madhhab for madhhab in requested_madhhabs if madhhab not in present_madhhabs
        ]
        if not missing_madhhabs:
            return candidates

        enriched = list(candidates)
        existing_keys = {
            (
                str(candidate.get("source_path", "")).strip().lower(),
                str(candidate.get("reference", "")).strip().lower(),
            )
            for candidate in candidates
        }
        for madhhab in missing_madhhabs:
            if deadline is not None and perf_counter() >= deadline:
                break
            filtered_chunks = [
                chunk
                for chunk in chunks
                if str(chunk.get("madhhab", "")).strip().lower() == madhhab
                and str(chunk.get("source_classification", "") or chunk.get("source_type", "")).strip().lower()
                in {"fiqh_manual", "commentary", "fatwa"}
            ]
            if not filtered_chunks:
                continue
            supplemental = search_chunks(
                question,
                filtered_chunks,
                top_k=1,
                query_intent=query_intent,
                deadline=deadline,
            )
            for candidate in supplemental:
                key = (
                    str(candidate.get("source_path", "")).strip().lower(),
                    str(candidate.get("reference", "")).strip().lower(),
                )
                if key in existing_keys:
                    continue
                enriched.append(candidate)
                existing_keys.add(key)
        return enriched

    def _vector_status(self) -> dict[str, Any]:
        if not self.runtime_config.embedding_enabled:
            return {
                "embedding_enabled": False,
                "hybrid_ready": False,
                "active": False,
                "reason": "embedding_disabled",
                "fallback_reason": "embedding_disabled",
                "index_path": str(self._vector_index_path()),
            }
        store = self._get_vector_store()
        store_loaded = store.load()
        store_status = store.status()
        embedder_status = self._get_embedder().status()
        reason = store_status["reason"]
        if not store_loaded:
            reason = store_status["reason"]
        elif not embedder_status["dependency_available"]:
            reason = embedder_status["reason"]
        elif not embedder_status["model_loaded"]:
            reason = "embedding_model_not_loaded"
        return {
            "embedding_enabled": True,
            "hybrid_ready": bool(
                store_loaded
                and embedder_status["dependency_available"]
                and embedder_status["model_loaded"]
            ),
            "active": False,
            "reason": reason,
            "fallback_reason": "",
            "index_path": str(self._vector_index_path()),
            "vector_index_loaded": store_loaded,
            "vector_store": store_status,
            "embedder": embedder_status,
        }

    def _get_embedder(self) -> LocalEmbedder:
        if self._embedder is None:
            self._embedder = LocalEmbedder(
                model_name=self.runtime_config.embedding_model,
                device=self.runtime_config.embedding_device,
                enabled=self.runtime_config.embedding_enabled,
            )
        return self._embedder

    def _get_vector_store(self) -> LocalVectorStore:
        if self._vector_store is None:
            self._vector_store = LocalVectorStore(self._vector_index_path())
        return self._vector_store

    def _vector_index_path(self) -> Path:
        configured = Path(str(self.runtime_config.vector_index_path or "data/index/vector_index.npz"))
        if configured.is_absolute():
            return configured
        return self.repo_root / configured

    def _apply_scholar_perspective_boosts(
        self,
        candidates: list[dict[str, Any]],
        *,
        query_intent: QueryIntent,
    ) -> list[dict[str, Any]]:
        if query_intent.intent_id != "scholar_perspective" or not query_intent.scholar_id:
            return candidates
        profile = load_scholar_profiles(self.repo_root).get(query_intent.scholar_id)
        if profile is None:
            return candidates
        boosted: list[dict[str, Any]] = []
        for candidate in candidates:
            match_type = match_source_to_scholar(candidate, profile)
            if not match_type:
                boosted.append(candidate)
                continue
            updated = dict(candidate)
            updated["scholar_attribution_match"] = match_type
            boost = 1.25 if match_type == "direct" else 0.35
            updated["scholar_boost_applied"] = boost
            updated["retrieval_score"] = float(updated.get("retrieval_score", 0.0) or 0.0) + boost
            boosted.append(updated)
        return boosted


def serialize_retrieval_debug(result: RetrievalDebugResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["query_intent"] = asdict(result.query_intent)
    return payload


def _load_runtime_config(repo_root: Path) -> RuntimeConfig:
    config_path = repo_root / "config" / "runtime_config.json"
    if not config_path.exists():
        return RuntimeConfig()
    store = RuntimeConfigStore(config_path)
    return store.load_effective()


def _counter_for_candidates(
    candidates: list[dict[str, Any]],
    *,
    field_name: str,
    fallback_field: str | None = None,
    limit: int = 6,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for candidate in candidates:
        value = str(candidate.get(field_name, "") or "").strip()
        if not value and fallback_field:
            value = str(candidate.get(fallback_field, "") or "").strip()
        if not value and field_name == "source_family":
            value = get_duplicate_family_key(candidate)
        if not value:
            value = "unknown"
        counter[value] += 1
    return dict(counter.most_common(limit))


def _ask_log(message: str) -> None:
    print(message, flush=True)


def _trust_diagnostics(
    *,
    reranked: list[dict[str, Any]],
    trust_profile_id: str,
) -> dict[str, Any]:
    """Compact diagnostics so admins can verify trust weighting actually fired.

    Reports the active profile, how many of the returned candidates had a
    non-zero trust contribution, what fraction of structured signals were
    present across the set (signal coverage), and the breakdown of the
    top-ranked candidate so weighting can be sanity-checked against the
    rendered answer.
    """

    total_candidates = len(reranked)
    if total_candidates == 0:
        return {
            "profile_id": trust_profile_id,
            "active": trust_profile_id != "default",
            "candidates_scored": 0,
            "candidates_with_nonzero_bonus": 0,
            "signal_coverage": {},
            "top_candidate_breakdown": None,
        }

    nonzero = 0
    signal_observed: dict[str, int] = {}
    breakdowns: list[dict[str, Any]] = []
    for candidate in reranked:
        breakdown = candidate.get("_trust_breakdown") or {}
        if not isinstance(breakdown, dict):
            continue
        breakdowns.append(breakdown)
        if float(breakdown.get("total", 0.0) or 0.0) != 0.0:
            nonzero += 1
        signals = breakdown.get("signals") or {}
        for name, value in signals.items():
            if value in (None, "", "unknown", 0, []):
                continue
            signal_observed[name] = signal_observed.get(name, 0) + 1

    signal_coverage = {
        name: round(count / total_candidates, 3)
        for name, count in sorted(signal_observed.items())
    }

    return {
        "profile_id": trust_profile_id,
        "active": trust_profile_id != "default",
        "candidates_scored": total_candidates,
        "candidates_with_nonzero_bonus": nonzero,
        "signal_coverage": signal_coverage,
        "top_candidate_breakdown": breakdowns[0] if breakdowns else None,
    }


def _madhhab_retrieval_diagnostics(
    *,
    candidates: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    query_intent: QueryIntent,
) -> dict[str, Any]:
    requested_madhhabs = tuple(query_intent.requested_madhhabs)
    if not requested_madhhabs:
        return {
            "detected_madhhab_intent": query_intent.detected_madhhab_intent,
            "madhhab_boost_applied": False,
            "madhhab_fallback_used": False,
        }
    candidate_madhhabs = {
        str(candidate.get("madhhab", "")).strip().lower()
        for candidate in candidates
        if str(candidate.get("source_classification", "") or candidate.get("source_type", "")).strip().lower()
        in {"fiqh_manual", "commentary", "fatwa"}
        and str(candidate.get("madhhab", "")).strip().lower()
    }
    boost_applied = any(madhhab in candidate_madhhabs for madhhab in requested_madhhabs)
    fallback_used = any(madhhab not in candidate_madhhabs for madhhab in requested_madhhabs)
    return {
        "detected_madhhab_intent": query_intent.detected_madhhab_intent,
        "madhhab_boost_applied": boost_applied,
        "madhhab_fallback_used": fallback_used,
    }
