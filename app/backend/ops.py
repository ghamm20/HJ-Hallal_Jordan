"""Operational services for runtime config, permissions, logging, and truth snapshots."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import uuid
from concurrent import futures
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator

import requests

from app.api.ask import AskPipeline, AskRequest
from app.backend.chat_continuity import SessionContinuityStore
from app.backend.fast_path import (
    BALANCED_RESEARCH,
    FastPathDecision,
    classify_fast_path,
    finalize_fast_path,
    finalize_research_depth,
    plan_research_depth,
)
from app.backend.live_events import LiveEventBroker, RequestStateStore
from app.backend.location_tools import (
    NullRoutingProvider,
    evaluate_location_tool,
)
from app.backend.micro_fast import (
    FAST_BENCHMARK_PROMPTS,
    MicroFastRouter,
    compare_tier_outputs,
)
from app.backend.micro_shadow import (
    MicroLLMShadowPlanner,
    shadow_backend_status,
    tuning_snapshot,
    write_benchmark_artifacts,
)
from app.backend.model_profiles import build_model_profile_status
from app.backend.portable_paths import (
    RuntimePathProfile,
    discover_config_path,
    resolve_runtime_path_profile,
)
from app.backend.permissions import RoleContext, permission_matrix
from app.backend.runtime_config import RuntimeConfig, RuntimeConfigStore
from app.backend.tasks import (
    ACTIVE_TASK_STATUSES,
    LocalTaskRunner,
    TaskCreateRequest,
    TaskDispatchRequest,
    TaskRuntimeIdentity,
    TaskStore,
    build_runtime_identity,
)
from app.backend.update_center import UpdateCenter
from app.backend.user_workspace import UserWorkspaceStore
from app.citations.renderer import render_answer
from app.reasoning.answer_grounding import AnswerGrounder
from app.reasoning.config_loader import load_contract_artifacts
from app.reasoning.intent_router import QueryIntent, route_query_intent
from app.reasoning.local_gguf_client import LocalGGUFMainModelRuntime
from app.reasoning.ollama_client import OllamaClientProtocol
from app.reasoning.schema_validator import normalize_and_validate_answer
from app.reasoning.study_path import study_path_diagnostics
from app.retrieval.pipeline import RetrievalPipeline, serialize_retrieval_debug
from app.retrieval.persisted_index import (
    build_persisted_retrieval_index,
    compute_corpus_fingerprint,
    inspect_persisted_retrieval_index,
)
from ingestion.pipelines.pdf_to_normalized import process_pdf_corpus

LOGGER = logging.getLogger(__name__)
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEBUG_FAST = os.getenv("HJ_DEBUG_FAST", "").strip().lower() in TRUTHY_ENV_VALUES
MODEL_TIMEOUT_SECONDS = 5.0
FAST_MODE_BUDGET_SECONDS = 3.0
OLLAMA_REQUEST_TIMEOUT_SECONDS = 5.0
RESEARCH_DEPTH_RETRIEVAL_TIMEOUT_FIELDS = {
    "quick_source_check": "retrieval_timeout_quick_ms",
    "balanced_research": "retrieval_timeout_balanced_ms",
    "deep_research": "retrieval_timeout_deep_ms",
}


def _ask_log(message: str) -> None:
    print(message, flush=True)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _write_startup_profile_event(event: str, **details: Any) -> None:
    path_text = str(os.getenv("HALAL_JORDAN_STARTUP_PROFILE_PATH", "") or "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": _utc_now(), "event": event, **details}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        return


def _path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _remove_tree_quietly(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)


def _safe_mode_state() -> dict[str, Any]:
    reason = str(os.getenv("HALAL_JORDAN_SAFE_MODE_REASON", "") or "").strip() or None
    return {
        "active": _env_flag("HALAL_JORDAN_SAFE_MODE"),
        "reason": reason,
        "usb_test_mode": _env_flag("HALAL_JORDAN_USB_TEST_MODE")
        or _env_flag("USB_TEST_MODE"),
    }


def _selected_runtime_listener() -> dict[str, Any]:
    host = str(
        os.getenv("HALAL_JORDAN_SELECTED_HOST")
        or os.getenv("HALAL_JORDAN_BIND_HOST")
        or "127.0.0.1"
    ).strip()
    port = None
    port_text = str(
        os.getenv("HALAL_JORDAN_SELECTED_PORT")
        or os.getenv("HALAL_JORDAN_PORT")
        or ""
    ).strip()
    if port_text:
        try:
            parsed = int(port_text)
        except ValueError:
            parsed = 0
        if parsed > 0:
            port = parsed
    url = f"http://{host}:{port}/" if port else None
    return {"host": host, "port": port, "url": url}


def _startup_retrieval_warmup_requested(config: RuntimeConfig) -> bool:
    if not bool(config.retrieval_eager_warmup_enabled):
        return False
    if os.getenv("PYTEST_CURRENT_TEST") and not _env_flag(
        "HALAL_JORDAN_FORCE_EAGER_WARMUP"
    ):
        return False
    return True


def _retrieval_timeout_seconds_for_depth(
    config: RuntimeConfig,
    research_depth: str,
) -> float:
    field_name = RESEARCH_DEPTH_RETRIEVAL_TIMEOUT_FIELDS.get(
        str(research_depth or "").strip() or "balanced_research",
        "retrieval_timeout_balanced_ms",
    )
    timeout_ms = int(getattr(config, field_name, config.retrieval_timeout_balanced_ms) or 0)
    return max(timeout_ms, 1000) / 1000.0


def _retrieval_warmup_timeout_seconds(config: RuntimeConfig) -> float:
    return max(int(config.retrieval_warmup_timeout_ms or 0), 1000) / 1000.0


def _retrieval_failure_label(reason: str) -> str:
    normalized = str(reason or "").strip()
    if not normalized:
        return "load error"
    if normalized.startswith("persisted_index_required:"):
        normalized = normalized.split(":", 1)[1]
    labels = {
        "index_directory_missing": "index missing",
        "index_files_missing": "index missing",
        "index_invalid": "index invalid",
        "manifest_invalid_json": "index invalid",
        "index_payload_invalid": "index invalid",
        "corpus_fingerprint_mismatch": "corpus fingerprint mismatch",
        "index_unconfigured": "index missing",
        "retrieval_search_timeout": "timeout",
        "retrieval_call_exceeded_budget": "timeout",
    }
    return labels.get(normalized, normalized.replace("_", " "))


def _retrieval_timeout_message(
    *,
    warmup_status: str,
    failure_reason: str,
) -> str:
    if warmup_status == "warming":
        return "Retrieval index is still warming. Please retry in a moment."
    label = _retrieval_failure_label(failure_reason)
    if label == "timeout":
        return "Retrieval timed out before grounded source search completed. Please retry."
    return f"Retrieval unavailable: {label}."


@dataclass(slots=True)
class ActionResult:
    action: str
    success: bool
    actor_id: str
    actor_role: str
    request_id: str
    duration_ms: int
    status: str
    changed: dict[str, Any]
    details: dict[str, Any]
    task_id: str | None = None
    task_type: str | None = None
    task_status: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    retry_of: str | None = None
    blocking_reason: str | None = None
    degraded_reason: str | None = None
    terminal_reason: str | None = None
    was_reconciled: bool = False
    execution_mode: str = "inline"
    runtime_instance_id: str | None = None
    error_summary: str | None = None


@dataclass(slots=True)
class OperationOutcome:
    changed: dict[str, Any]
    details: dict[str, Any]
    status: str
    task_status: str = "succeeded"
    result_summary: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    error_summary: str | None = None
    degraded_reason: str | None = None
    terminal_reason: str | None = None
    live_state_preserved: bool | None = None
    live_state_swapped: bool | None = None


class ManagedOperationError(RuntimeError):
    """Operational failure with structured task/result context."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details or {}
        self.result_summary = result_summary or {}
        self.progress = progress or {}


@dataclass(slots=True)
class ChatExecutionResult:
    status: str
    request_id: str
    answer: dict[str, Any] | None
    rendered_text: str
    model_requested: str | None
    model_used: str | None
    fallback_used: bool
    fallback_from: str | None
    fallback_to: str | None
    retrieval_summary: dict[str, Any]
    degraded: bool
    degraded_subsystem: str | None
    degraded_reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeState:
    started_at: str = field(default_factory=lambda: _utc_now())
    runtime_instance_id: str | None = None
    process_id: int | None = None
    host_id: str | None = None
    last_reload_at: str | None = None
    last_reindex_at: str | None = None
    retrieval_loaded: bool = False
    last_retrieval_stats: dict[str, int] = field(default_factory=dict)
    last_model_requested: str | None = None
    last_model_used: str | None = None
    last_model_backend: str | None = None
    last_model_fallback_used: bool = False
    last_model_fallback_from: str | None = None
    last_model_fallback_to: str | None = None
    last_model_error: str | None = None
    last_main_model_loaded: bool = False
    last_degraded_subsystem: str | None = None
    last_degraded_reason: str | None = None
    last_chat_status: str | None = None
    last_reconciliation_at: str | None = None
    last_reconciled_task_count: int = 0
    last_model_profile_status: dict[str, Any] = field(default_factory=dict)
    retrieval_warmup_status: str = "not_started"
    retrieval_warmup_started_at: str | None = None
    retrieval_warmup_finished_at: str | None = None
    retrieval_warmup_error: str | None = None
    retrieval_warmup_trigger: str | None = None
    retrieval_warmup_bootstrap_source: str | None = None
    retrieval_warmup_fallback_reason: str | None = None
    last_micro_fast: dict[str, Any] = field(default_factory=dict)
    micro_fast_assist_applied_count: int = 0
    micro_fast_assist_rejected_count: int = 0
    last_micro_fast_assist_decision: dict[str, Any] = field(default_factory=dict)
    last_micro_shadow: dict[str, Any] = field(default_factory=dict)
    last_micro_shadow_warmup: dict[str, Any] = field(default_factory=dict)
    last_micro_shadow_benchmark: dict[str, Any] = field(default_factory=dict)
    last_micro_fast_benchmark: dict[str, Any] = field(default_factory=dict)
    last_micro_tier_comparison: dict[str, Any] = field(default_factory=dict)


class SQLiteLogStore:
    """Persist chat, audit, and error logs to sqlite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.available = False
        self.last_error: str | None = None

    def ensure_ready(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        route TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        actor_role TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        latency_ms INTEGER NOT NULL,
                        question TEXT NOT NULL,
                        answer_mode TEXT NOT NULL,
                        selected_madhhab TEXT NOT NULL,
                        model_requested TEXT,
                        model_used TEXT,
                        fallback_used INTEGER NOT NULL,
                        fallback_from TEXT,
                        fallback_to TEXT,
                        retrieval_status TEXT NOT NULL,
                        retrieved_source_count INTEGER NOT NULL,
                        citation_count INTEGER NOT NULL,
                        degraded INTEGER NOT NULL,
                        degraded_reason TEXT,
                        error_class TEXT,
                        error_message TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        route TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        actor_role TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        duration_ms INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        changed_json TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        error_summary TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS error_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        route TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        actor_role TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        subsystem TEXT NOT NULL,
                        error_class TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        details_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operator_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        related_task_id TEXT,
                        request_id TEXT,
                        session_id TEXT,
                        route TEXT,
                        summary TEXT NOT NULL,
                        details_json TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
            self.available = True
            self.last_error = None
        except Exception as exc:  # pragma: no cover - exercised by degraded tests
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"

    def prune(self, retention_days: int) -> None:
        if not self.available:
            return
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        try:
            with self._connect() as connection:
                cursor = connection.cursor()
                for table_name in ("chat_logs", "audit_logs", "error_logs"):
                    cursor.execute(
                        f"DELETE FROM {table_name} WHERE timestamp < ?",
                        (cutoff,),
                    )
                connection.commit()
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"

    def write_chat_log(self, payload: dict[str, Any]) -> None:
        self._write(
            """
            INSERT INTO chat_logs (
                timestamp, route, actor_id, actor_role, request_id, session_id,
                success, latency_ms, question, answer_mode, selected_madhhab,
                model_requested, model_used, fallback_used, fallback_from,
                fallback_to, retrieval_status, retrieved_source_count,
                citation_count, degraded, degraded_reason, error_class, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["timestamp"],
                payload["route"],
                payload["actor_id"],
                payload["actor_role"],
                payload["request_id"],
                payload["session_id"],
                int(payload["success"]),
                payload["latency_ms"],
                payload["question"],
                payload["answer_mode"],
                payload["selected_madhhab"],
                payload.get("model_requested"),
                payload.get("model_used"),
                int(payload.get("fallback_used", False)),
                payload.get("fallback_from"),
                payload.get("fallback_to"),
                payload["retrieval_status"],
                payload["retrieved_source_count"],
                payload["citation_count"],
                int(payload.get("degraded", False)),
                payload.get("degraded_reason"),
                payload.get("error_class"),
                payload.get("error_message"),
            ),
        )

    def write_audit_log(self, payload: dict[str, Any]) -> None:
        self._write(
            """
            INSERT INTO audit_logs (
                timestamp, route, action, actor_id, actor_role, request_id,
                session_id, success, duration_ms, status, changed_json,
                details_json, error_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["timestamp"],
                payload["route"],
                payload["action"],
                payload["actor_id"],
                payload["actor_role"],
                payload["request_id"],
                payload["session_id"],
                int(payload["success"]),
                payload["duration_ms"],
                payload["status"],
                json.dumps(payload.get("changed", {}), ensure_ascii=False),
                json.dumps(payload.get("details", {}), ensure_ascii=False),
                payload.get("error_summary"),
            ),
        )

    def write_error_log(self, payload: dict[str, Any]) -> None:
        self._write(
            """
            INSERT INTO error_logs (
                timestamp, route, action, actor_id, actor_role, request_id,
                session_id, subsystem, error_class, error_message, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["timestamp"],
                payload["route"],
                payload["action"],
                payload["actor_id"],
                payload["actor_role"],
                payload["request_id"],
                payload["session_id"],
                payload["subsystem"],
                payload["error_class"],
                payload["error_message"],
                json.dumps(payload.get("details", {}), ensure_ascii=False),
            ),
        )

    def write_operator_event(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.available:
            return None
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO operator_events (
                        event_id, timestamp, event_type, severity, related_task_id,
                        request_id, session_id, route, summary, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["event_id"],
                        payload["timestamp"],
                        payload["event_type"],
                        payload["severity"],
                        payload.get("related_task_id"),
                        payload.get("request_id"),
                        payload.get("session_id"),
                        payload.get("route"),
                        payload["summary"],
                        json.dumps(payload.get("details", {}), ensure_ascii=False),
                    ),
                )
                row_id = cursor.lastrowid
                connection.commit()
                rows = self._fetch_rows(
                    "SELECT * FROM operator_events WHERE id = ? LIMIT 1",
                    [row_id],
                )
                return rows[0] if rows else None
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def fetch_chat_logs(self, *, limit: int, success: bool | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM chat_logs"
        params: list[Any] = []
        if success is not None:
            query += " WHERE success = ?"
            params.append(int(success))
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._fetch_rows(query, params)

    def fetch_audit_logs(self, *, limit: int, action: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM audit_logs"
        params: list[Any] = []
        if action:
            query += " WHERE action = ?"
            params.append(action)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._fetch_rows(query, params)

    def fetch_error_logs(self, *, limit: int) -> list[dict[str, Any]]:
        return self._fetch_rows(
            "SELECT * FROM error_logs ORDER BY id DESC LIMIT ?",
            [limit],
        )

    def fetch_recent_helper(self, *, kind: str, limit: int) -> list[dict[str, Any]]:
        if kind == "failures":
            return self.fetch_chat_logs(limit=limit, success=False)
        if kind == "reindexes":
            return self.fetch_audit_logs(limit=limit, action="reindex_documents")
        if kind == "reloads":
            return self.fetch_audit_logs(limit=limit, action="reload_retrieval_assets")
        if kind == "model_fallbacks":
            return self._fetch_rows(
                "SELECT * FROM chat_logs WHERE fallback_used = 1 ORDER BY id DESC LIMIT ?",
                [limit],
            )
        if kind == "permission_denials":
            return self.fetch_audit_logs(limit=limit, action="permission_denied")
        return []

    def fetch_operator_events(
        self,
        *,
        limit: int,
        severity: str | None = None,
        request_id: str | None = None,
        related_task_id: str | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM operator_events"
        params: list[Any] = []
        clauses: list[str] = []
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if request_id:
            clauses.append("request_id = ?")
            params.append(request_id)
        if related_task_id:
            clauses.append("related_task_id = ?")
            params.append(related_task_id)
        if after_id is not None:
            clauses.append("id > ?")
            params.append(after_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._fetch_rows(query, params)
        rows.reverse()
        return rows

    def latest_audit_entry(self, action: str) -> dict[str, Any] | None:
        rows = self._fetch_rows(
            "SELECT * FROM audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1",
            [action],
        )
        return rows[0] if rows else None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.path),
            "last_error": self.last_error,
        }

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _write(self, query: str, params: tuple[Any, ...]) -> None:
        if not self.available:
            return
        try:
            with self._connect() as connection:
                connection.execute(query, params)
                connection.commit()
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _fetch_rows(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            with self._connect() as connection:
                cursor = connection.execute(query, params)
                rows = [dict(row) for row in cursor.fetchall()]
            return [_decode_json_fields(row) for row in rows]
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


class RuntimeAwareOllamaClient(OllamaClientProtocol):
    """Config-driven model client for the bundled local GGUF production runtime."""

    def __init__(
        self,
        config: RuntimeConfig,
        runtime_state: RuntimeState,
        path_profile: RuntimePathProfile,
    ) -> None:
        self.config = config
        self.runtime_state = runtime_state
        self.path_profile = path_profile
        self.timeout_seconds = OLLAMA_REQUEST_TIMEOUT_SECONDS
        self._resolved_candidates: list[str] = []
        self._resolved_primary: str | None = None
        self._resolved_backends: list[dict[str, str]] = []

    def resolve_model(self, requested_model: str | None = None) -> str:
        configured = list(self.config.model_preference_order)
        requested = requested_model or (configured[0] if configured else None)
        self.runtime_state.last_model_requested = requested
        self.runtime_state.last_model_used = None
        self.runtime_state.last_model_backend = None
        self.runtime_state.last_main_model_loaded = False
        main_local_status = LocalGGUFMainModelRuntime.status(
            config=self.config,
            path_profile=self.path_profile,
        )

        if not main_local_status["ready"]:
            self.runtime_state.last_model_error = str(
                main_local_status.get("error") or "no configured production model is available"
            )
            raise RuntimeError(self.runtime_state.last_model_error)

        resolved_model = LocalGGUFMainModelRuntime.resolve_model_name(
            config=self.config,
            path_profile=self.path_profile,
        )
        if not requested_model:
            requested = resolved_model
            self.runtime_state.last_model_requested = resolved_model
        self._resolved_backends = [{"backend": "local_gguf", "model_name": resolved_model}]
        self._resolved_candidates = [resolved_model]
        self._resolved_primary = resolved_model
        self.runtime_state.last_model_used = resolved_model
        self.runtime_state.last_model_backend = "local_gguf"
        self.runtime_state.last_model_fallback_used = bool(
            requested and requested != resolved_model
        )
        self.runtime_state.last_model_fallback_from = (
            requested if requested and requested != resolved_model else None
        )
        self.runtime_state.last_model_fallback_to = (
            resolved_model if requested and requested != resolved_model else None
        )
        self.runtime_state.last_model_error = None
        return resolved_model

    def chat_json(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requested = self.runtime_state.last_model_requested or model_name
        try:
            payload = LocalGGUFMainModelRuntime.chat_json(
                config=self.config,
                path_profile=self.path_profile,
                messages=messages,
                schema=schema,
                options=options,
            )
        except Exception as exc:  # pragma: no cover - exercised by degraded tests
            self.runtime_state.last_main_model_loaded = False
            self.runtime_state.last_model_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"configured local production model failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.runtime_state.last_main_model_loaded = True
        self.runtime_state.last_model_used = model_name
        self.runtime_state.last_model_backend = "local_gguf"
        self.runtime_state.last_model_fallback_used = model_name != requested
        self.runtime_state.last_model_fallback_from = (
            requested if model_name != requested else None
        )
        self.runtime_state.last_model_fallback_to = (
            model_name if model_name != requested else None
        )
        self.runtime_state.last_model_error = None
        return payload

    def _chat_once(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("ollama_transport_removed_for_portable_runtime")

    def _list_models(self) -> list[str]:
        return []


class OpsService:
    """Wrap retrieval, ask, config, logs, and runtime truth in one service."""

    def __init__(
        self,
        repo_root: Path,
        runtime_dir: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self._runtime_dir_override = runtime_dir
        self.initial_config_path = discover_config_path(
            repo_root,
            runtime_dir_override=runtime_dir,
        )
        self.config_store = RuntimeConfigStore(self.initial_config_path)
        self.config_store.ensure_file_exists()
        self.path_profile = resolve_runtime_path_profile(
            repo_root,
            config=self.config_store.load_effective(),
            runtime_dir_override=runtime_dir,
        )
        self.path_profile.ensure_base_directories()
        self.runtime_dir = self.path_profile.runtime_dir
        self.config_store = RuntimeConfigStore(self.path_profile.config_path)
        self.config_store.ensure_file_exists()
        self.log_store = SQLiteLogStore(self.path_profile.runtime_db_path)
        self.task_store = TaskStore(self.path_profile.runtime_db_path)
        self.user_workspace = UserWorkspaceStore(self.path_profile.runtime_db_path)
        self.runtime_identity = build_runtime_identity()
        self.artifacts = load_contract_artifacts(self.repo_root)
        self.retrieval = RetrievalPipeline(
            repo_root=self.repo_root,
            artifacts=self.artifacts,
            raw_root=self.path_profile.raw_corpus_dir,
            normalized_root=self.path_profile.normalized_corpus_dir,
            index_root=self.path_profile.index_dir,
        )
        self.runtime_state = RuntimeState(
            runtime_instance_id=self.runtime_identity.runtime_instance_id,
            process_id=self.runtime_identity.process_id,
            host_id=self.runtime_identity.host_id,
        )
        self._grounder = AnswerGrounder(self.repo_root, self.artifacts)
        self._task_lock = threading.RLock()
        self._task_runner = LocalTaskRunner()
        self.event_broker = LiveEventBroker()
        self.request_states = RequestStateStore()
        self.session_continuity = SessionContinuityStore(
            db_path=self.path_profile.runtime_db_path
        )
        self.update_center = UpdateCenter(self.path_profile)
        self.micro_fast = MicroFastRouter(
            log_path=self.path_profile.runtime_dir / "micro_llm_fast.jsonl",
        )
        self.routing_provider = NullRoutingProvider()
        self.micro_shadow = MicroLLMShadowPlanner(
            path_profile=self.path_profile,
            log_path=self.path_profile.runtime_dir / "micro_llm_shadow.jsonl",
        )
        self._retrieval_warmup_lock = threading.Lock()
        self._retrieval_warmup_thread: threading.Thread | None = None
        self._retrieval_warmup_done = threading.Event()

    def _start_retrieval_warmup(
        self,
        config: RuntimeConfig,
        *,
        trigger: str,
    ) -> None:
        if self.runtime_state.retrieval_loaded:
            if self.runtime_state.retrieval_warmup_status != "ready":
                self.runtime_state.retrieval_warmup_status = "ready"
                self.runtime_state.retrieval_warmup_finished_at = (
                    self.runtime_state.retrieval_warmup_finished_at or _utc_now()
                )
            self._retrieval_warmup_done.set()
            return
        with self._retrieval_warmup_lock:
            if self.runtime_state.retrieval_loaded:
                self._retrieval_warmup_done.set()
                return
            if self._retrieval_warmup_thread and self._retrieval_warmup_thread.is_alive():
                return
            self._retrieval_warmup_done.clear()
            self.runtime_state.retrieval_warmup_status = "warming"
            self.runtime_state.retrieval_warmup_started_at = _utc_now()
            self.runtime_state.retrieval_warmup_finished_at = None
            self.runtime_state.retrieval_warmup_error = None
            self.runtime_state.retrieval_warmup_trigger = trigger
            self.runtime_state.retrieval_warmup_bootstrap_source = None
            self.runtime_state.retrieval_warmup_fallback_reason = None
            thread = threading.Thread(
                target=self._run_retrieval_warmup,
                kwargs={"config": config, "trigger": trigger},
                name=f"hj-retrieval-warmup-{trigger}",
                daemon=True,
            )
            self._retrieval_warmup_thread = thread
            thread.start()

    def _run_retrieval_warmup(
        self,
        *,
        config: RuntimeConfig,
        trigger: str,
    ) -> None:
        started = perf_counter()
        _write_startup_profile_event(
            "retrieval_warmup_started",
            trigger=trigger,
        )
        try:
            index_assets = inspect_persisted_retrieval_index(
                self.path_profile.index_dir,
                repo_root=self.repo_root,
                raw_root=self.path_profile.raw_corpus_dir,
                normalized_root=self.path_profile.normalized_corpus_dir,
            )
            current_fingerprint = compute_corpus_fingerprint(
                self.repo_root,
                raw_root=self.path_profile.raw_corpus_dir,
                normalized_root=self.path_profile.normalized_corpus_dir,
            )
            _write_startup_profile_event(
                "retrieval_warmup_probe",
                trigger=trigger,
                cwd=os.getcwd(),
                repo_root=str(self.repo_root),
                raw_root=str(self.path_profile.raw_corpus_dir),
                normalized_root=str(self.path_profile.normalized_corpus_dir),
                index_root=str(self.path_profile.index_dir),
                manifest_fingerprint=index_assets.get("corpus_fingerprint"),
                current_fingerprint=current_fingerprint.get("fingerprint"),
                manifest_chunk_count=index_assets.get("chunk_count"),
                manifest_document_count=index_assets.get("document_count"),
                retrieval_index_loadable=index_assets.get("retrieval_index_loadable"),
                retrieval_fallback_reason=index_assets.get("retrieval_fallback_reason"),
            )
            retrieval_state = self.retrieval.bootstrap(allow_corpus_fallback=False)
            self.runtime_state.retrieval_loaded = True
            self.runtime_state.last_retrieval_stats = {
                "documents": len(retrieval_state.corpus.documents),
                "chunks": len(retrieval_state.chunks),
                "skipped_files": len(retrieval_state.corpus.skipped_files),
            }
            self._grounder.quote_window_chars = config.citation_excerpt_length
            AnswerGrounder.prime_chunk_index(self.repo_root, retrieval_state.chunks)
            self.runtime_state.retrieval_warmup_status = "ready"
            self.runtime_state.retrieval_warmup_finished_at = _utc_now()
            self.runtime_state.retrieval_warmup_error = None
            self.runtime_state.retrieval_warmup_bootstrap_source = (
                self.retrieval._last_bootstrap_source
            )
            self.runtime_state.retrieval_warmup_fallback_reason = (
                self.retrieval._last_bootstrap_reason
            )
            _write_startup_profile_event(
                "retrieval_warmup_ready",
                trigger=trigger,
                latency_ms=int((perf_counter() - started) * 1000),
                bootstrap_source=self.retrieval._last_bootstrap_source,
                prepared_search_loaded=self.retrieval._prepared_search_loaded,
                prepared_search_load_ms=self.retrieval._prepared_search_load_ms,
                prepared_search_build_ms=self.retrieval._prepared_search_build_ms,
                document_count=len(retrieval_state.corpus.documents),
                chunk_count=len(retrieval_state.chunks),
            )
            self._emit_operator_event(
                event_type="retrieval_warmup_ready",
                severity="info",
                summary="Retrieval state warmed from the persisted index",
                route="startup" if trigger == "startup" else "/api/chat",
                details={
                    "trigger": trigger,
                    "documents": len(retrieval_state.corpus.documents),
                    "chunks": len(retrieval_state.chunks),
                    "latency_ms": int((perf_counter() - started) * 1000),
                    "bootstrap_source": self.retrieval._last_bootstrap_source,
                    "bootstrap_fallback_reason": self.retrieval._last_bootstrap_reason,
                },
            )
        except Exception as exc:  # pragma: no cover - startup/request warmup guard
            self.runtime_state.retrieval_loaded = False
            self.runtime_state.retrieval_warmup_status = "failed"
            self.runtime_state.retrieval_warmup_finished_at = _utc_now()
            self.runtime_state.retrieval_warmup_error = f"{type(exc).__name__}: {exc}"
            self.runtime_state.retrieval_warmup_bootstrap_source = (
                self.retrieval._last_bootstrap_source
            )
            self.runtime_state.retrieval_warmup_fallback_reason = (
                self.retrieval._last_bootstrap_reason or str(exc)
            )
            _write_startup_profile_event(
                "retrieval_warmup_failed",
                trigger=trigger,
                latency_ms=int((perf_counter() - started) * 1000),
                error=self.runtime_state.retrieval_warmup_error,
                bootstrap_source=self.retrieval._last_bootstrap_source,
                bootstrap_fallback_reason=self.runtime_state.retrieval_warmup_fallback_reason,
            )
            self._emit_operator_event(
                event_type="retrieval_warmup_failed",
                severity="warning",
                summary="Retrieval warmup failed",
                route="startup" if trigger == "startup" else "/api/chat",
                details={
                    "trigger": trigger,
                    "error": self.runtime_state.retrieval_warmup_error,
                    "latency_ms": int((perf_counter() - started) * 1000),
                    "bootstrap_source": self.retrieval._last_bootstrap_source,
                    "bootstrap_fallback_reason": self.retrieval._last_bootstrap_reason,
                },
            )
        finally:
            self._retrieval_warmup_done.set()

    def _await_retrieval_warmup(
        self,
        config: RuntimeConfig,
        *,
        trigger: str,
    ) -> dict[str, Any]:
        self._start_retrieval_warmup(config, trigger=trigger)
        wait_started = perf_counter()
        wait_timeout = _retrieval_warmup_timeout_seconds(config)
        completed = self._retrieval_warmup_done.wait(timeout=wait_timeout)
        if completed:
            self._retrieval_warmup_done.set()
        return {
            "completed": completed,
            "wait_timeout_seconds": wait_timeout,
            "waited_ms": int((perf_counter() - wait_started) * 1000),
            "status": self.runtime_state.retrieval_warmup_status,
            "error": self.runtime_state.retrieval_warmup_error,
            "bootstrap_source": self.runtime_state.retrieval_warmup_bootstrap_source,
            "bootstrap_fallback_reason": self.runtime_state.retrieval_warmup_fallback_reason,
            "prepared_search_loaded": self.retrieval._prepared_search_loaded,
            "prepared_search_fallback_reason": self.retrieval._prepared_search_fallback_reason,
            "prepared_search_load_ms": self.retrieval._prepared_search_load_ms,
            "prepared_search_build_ms": self.retrieval._prepared_search_build_ms,
            "warmup_total_ms": self.retrieval._last_bootstrap_total_ms,
        }

    def startup(self) -> None:
        startup_started = perf_counter()
        _write_startup_profile_event(
            "ops_startup_begin",
            repo_root=str(self.repo_root),
            runtime_dir=str(self.runtime_dir),
        )
        self.config_store.ensure_file_exists()
        self.log_store.ensure_ready()
        self.task_store.ensure_ready()
        self.user_workspace.ensure_ready()
        self.update_center.ensure_ready()
        config = self.config_store.load_effective()
        self.session_continuity.set_ttl_seconds(
            config.session_continuity_ttl_minutes * 60
        )
        self.session_continuity.ensure_ready()
        self.log_store.prune(config.log_retention_days)
        reconciled = self._reconcile_stale_tasks()
        LOGGER.info(
            "Halal Jordan backend starting with runtime identity %s and config %s",
            config.runtime_identity,
            {
                "ollama_url": config.ollama_url,
                "model_preference_order": config.model_preference_order,
                "max_retrieval_candidates": config.max_retrieval_candidates,
                "rerank_limit": config.rerank_limit,
                "runtime_instance_id": self.runtime_identity.runtime_instance_id,
            },
        )
        if config.micro_shadow_warmup_on_startup:
            warmup = self.micro_shadow.warmup(config=config)
            self.runtime_state.last_micro_shadow_warmup = warmup
            self._emit_operator_event(
                event_type="micro_shadow_warmup",
                severity="info" if warmup.get("success") else "warning",
                summary=(
                    "Micro shadow warmup succeeded on startup"
                    if warmup.get("success")
                    else "Micro shadow warmup failed on startup"
                ),
                route="startup",
                details={"result": warmup, "advisory_only": True},
            )
        if _startup_retrieval_warmup_requested(config):
            self._start_retrieval_warmup(config, trigger="startup")
            retrieval_prewarm = {
                "started": True,
                "status": self.runtime_state.retrieval_warmup_status,
                "trigger": "startup",
                "wait_timeout_ms": config.retrieval_warmup_timeout_ms,
                "eager_warmup_enabled": True,
            }
            self._emit_operator_event(
                event_type="retrieval_prewarm_started",
                severity="info",
                summary="Retrieval warmup started from the persisted index",
                route="startup",
                details=retrieval_prewarm,
            )
        else:
            deferred_reason = (
                "disabled_for_pytest"
                if os.getenv("PYTEST_CURRENT_TEST")
                and not _env_flag("HALAL_JORDAN_FORCE_EAGER_WARMUP")
                else "disabled_by_config"
            )
            retrieval_prewarm = {
                "started": False,
                "status": self.runtime_state.retrieval_warmup_status,
                "trigger": "deferred_until_request",
                "wait_timeout_ms": config.retrieval_warmup_timeout_ms,
                "eager_warmup_enabled": bool(config.retrieval_eager_warmup_enabled),
                "deferred_reason": deferred_reason,
            }
            self._emit_operator_event(
                event_type="retrieval_prewarm_deferred",
                severity="info",
                summary="Retrieval warmup deferred until the first grounded request",
                route="startup",
                details=retrieval_prewarm,
            )
        if (
            config.update_check_enabled
            and config.update_check_on_startup
            and str(config.update_manifest_url or "").strip()
        ):
            startup_check_id = f"startup-update-check-{uuid.uuid4().hex}"

            def run_startup_update_check() -> None:
                status = self.update_center.check_for_updates(
                    config=config,
                    trigger="startup",
                )
                last_status = str(status.get("last_update_check_status") or "")
                if last_status == "ok":
                    self._emit_operator_event(
                        event_type="update_manifest_checked",
                        severity="info",
                        summary="Update manifest checked on startup",
                        route="startup",
                        details={
                            "pending_updates_count": status.get("pending_updates_count"),
                            "downloaded_updates_count": status.get("downloaded_updates_count"),
                        },
                    )
                elif last_status == "unavailable":
                    self._emit_operator_event(
                        event_type="update_check_unavailable",
                        severity="warning",
                        summary="Update manifest check was unavailable on startup",
                        route="startup",
                        details={
                            "error": status.get("last_update_check_error"),
                            "manifest_url": status.get("update_manifest_url"),
                        },
                    )

            self._task_runner.dispatch(
                TaskDispatchRequest(
                    task_id=startup_check_id,
                    background=True,
                    execute=run_startup_update_check,
                    name=f"hj-startup-update-{startup_check_id[-8:]}",
                )
            )
        self._write_audit(
            route="startup",
            action="startup",
            actor=RoleContext(
                actor_id="system",
                role="owner",
                request_id="startup",
                session_id="startup",
            ),
            success=True,
            duration_ms=0,
            status="ok",
            changed={},
            details={
                "runtime_identity": config.runtime_identity,
                "config_path": str(self.config_store.path),
                "runtime_instance_id": self.runtime_identity.runtime_instance_id,
                "process_id": self.runtime_identity.process_id,
                "host_id": self.runtime_identity.host_id,
                "reconciled_task_count": len(reconciled),
                "retrieval_prewarm": retrieval_prewarm,
            },
        )
        _write_startup_profile_event(
            "ops_startup_ready",
            latency_ms=int((perf_counter() - startup_started) * 1000),
            retrieval_prewarm_started=bool(retrieval_prewarm.get("started")),
            retrieval_prewarm_status=retrieval_prewarm.get("status"),
            laptop_build=bool(config.laptop_build),
            main_model_enabled=bool(config.main_model_enabled),
        )

    def shutdown(self) -> None:
        if self._retrieval_warmup_thread and self._retrieval_warmup_thread.is_alive():
            self._retrieval_warmup_done.wait(timeout=5.0)
            self._retrieval_warmup_thread.join(timeout=0.5)
        self._task_runner.shutdown()
        return None

    def register_local_user(
        self,
        *,
        username: str,
        password: str,
        display_name: str,
        default_madhhab: str,
        default_answer_mode: str,
    ) -> dict[str, Any]:
        user = self.user_workspace.register_user(
            username=username,
            password=password,
            display_name=display_name,
            default_madhhab=default_madhhab,
            default_answer_mode=default_answer_mode,
        )
        self.user_workspace.remember_preference(
            user_id=int(user["id"]),
            key="default_madhhab",
            value=str(user["default_madhhab"]),
            confidence=0.95,
        )
        self.user_workspace.remember_preference(
            user_id=int(user["id"]),
            key="default_answer_mode",
            value=str(user["default_answer_mode"]),
            confidence=0.95,
        )
        return user

    def authenticate_local_user(self, *, username: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
        user = self.user_workspace.authenticate_user(username=username, password=password)
        if not user:
            return None, None
        token = self.user_workspace.create_auth_session(user_id=int(user["id"]))
        return user, token

    def current_local_user(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        return self.user_workspace.get_user_for_token(token)

    def logout_local_user(self, token: str | None) -> None:
        if token:
            self.user_workspace.revoke_auth_session(token)

    def user_sidebar_state(self, *, user_id: int) -> dict[str, Any]:
        return self.user_workspace.sidebar_state(user_id=user_id)

    def update_user_settings(
        self,
        *,
        user_id: int,
        display_name: str | None = None,
        default_madhhab: str | None = None,
        default_answer_mode: str | None = None,
    ) -> dict[str, Any]:
        return self.user_workspace.update_user_defaults(
            user_id=user_id,
            display_name=display_name,
            default_madhhab=default_madhhab,
            default_answer_mode=default_answer_mode,
        )

    def create_user_chat_session(
        self,
        *,
        user_id: int,
        title: str,
        project_id: int | None = None,
    ) -> dict[str, Any]:
        return self.user_workspace.create_chat_session(
            user_id=user_id,
            title=title,
            project_id=project_id,
        )

    def list_user_chat_sessions(
        self,
        *,
        user_id: int,
        project_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.user_workspace.list_chat_sessions(
            user_id=user_id,
            project_id=project_id,
        )

    def get_user_chat_session(self, *, user_id: int, session_id: int) -> dict[str, Any] | None:
        return self.user_workspace.get_chat_session(user_id=user_id, session_id=session_id)

    def attach_user_chat_to_project(
        self,
        *,
        user_id: int,
        session_id: int,
        project_id: int | None,
    ) -> None:
        self.user_workspace.attach_chat_to_project(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
        )

    def create_user_project(
        self,
        *,
        user_id: int,
        title: str,
        description: str,
        madhhab: str,
        study_mode: str = "standard",
        status: str = "active",
    ) -> dict[str, Any]:
        return self.user_workspace.create_project(
            user_id=user_id,
            title=title,
            description=description,
            madhhab=madhhab,
            study_mode=study_mode,
            status=status,
        )

    def list_user_projects(self, *, user_id: int) -> list[dict[str, Any]]:
        return self.user_workspace.list_projects(user_id=user_id)

    def get_user_project(
        self,
        *,
        user_id: int,
        project_id: int,
    ) -> dict[str, Any] | None:
        return self.user_workspace.get_project(user_id=user_id, project_id=project_id)

    def update_user_project_study_progress(
        self,
        *,
        user_id: int,
        project_id: int,
        active_lesson_index: int | None = None,
        completed_lesson_index: int | None = None,
        advance: bool = False,
    ) -> dict[str, Any]:
        return self.user_workspace.update_project_study_progress(
            user_id=user_id,
            project_id=project_id,
            active_lesson_index=active_lesson_index,
            completed_lesson_index=completed_lesson_index,
            advance=advance,
        )

    def add_user_memory_note(
        self,
        *,
        user_id: int,
        content: str,
        source_session_id: int | None = None,
        confidence: float = 0.6,
    ) -> dict[str, Any]:
        return self.user_workspace.add_note_memory(
            user_id=user_id,
            content=content,
            source_session_id=source_session_id,
            confidence=confidence,
        )

    def list_user_memory(self, *, user_id: int) -> list[dict[str, Any]]:
        return self.user_workspace.list_memory(user_id=user_id)

    def chat(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
        research_depth: str = BALANCED_RESEARCH,
        actor: RoleContext,
        greeting_style: str | None = None,
        tone_level: str | None = None,
        ollama_model: str | None = None,
        retrieved_sources: list[dict[str, Any]] | None = None,
        user_id: int | None = None,
        chat_session_id: int | None = None,
        project_id: int | None = None,
    ) -> ChatExecutionResult:
        started = perf_counter()
        stage_latency_ms: dict[str, int] = {}
        _ask_log(f"[ASK] start route=/api/chat request_id={actor.request_id}")
        config = self.config_store.load_effective()
        self.log_store.prune(config.log_retention_days)
        requested_answer_mode = answer_mode
        requested_selected_madhhab = selected_madhhab
        persisted_chat_session_id = chat_session_id
        resolved_project_id = project_id
        prompt_user_context: dict[str, Any] | None = None
        resolved_project: dict[str, Any] | None = None
        if user_id is not None:
            if persisted_chat_session_id is None:
                created_session = self.user_workspace.create_chat_session(
                    user_id=user_id,
                    title=self.user_workspace.auto_title_for_question(question),
                    project_id=resolved_project_id,
                )
                persisted_chat_session_id = int(created_session["id"])
                if resolved_project_id is None:
                    resolved_project_id = created_session.get("project_id")
            else:
                existing_session = self.user_workspace.get_chat_session(
                    user_id=user_id,
                    session_id=persisted_chat_session_id,
                )
                if existing_session is None:
                    raise ValueError("unknown chat session")
                if (
                    str(existing_session.get("title") or "").strip().lower() == "new chat"
                    and not existing_session.get("messages")
                ):
                    self.user_workspace.rename_chat_session(
                        user_id=user_id,
                        session_id=persisted_chat_session_id,
                        title=self.user_workspace.auto_title_for_question(question),
                    )
                if resolved_project_id is None:
                    resolved_project_id = existing_session.get("project_id")
                elif existing_session.get("project_id") != resolved_project_id:
                    self.user_workspace.attach_chat_to_project(
                        user_id=user_id,
                        session_id=persisted_chat_session_id,
                        project_id=resolved_project_id,
                    )
            if resolved_project_id is not None:
                resolved_project = self.user_workspace.get_project(
                    user_id=user_id,
                    project_id=resolved_project_id,
                )
                if resolved_project is not None:
                    if (
                        str(selected_madhhab or "").strip() in {"", "not_specified"}
                        and str(resolved_project.get("madhhab") or "").strip()
                        and str(resolved_project.get("madhhab") or "").strip() != "not_specified"
                    ):
                        selected_madhhab = str(resolved_project.get("madhhab") or "").strip()
                    if str(resolved_project.get("study_mode") or "").strip() == "study_path":
                        answer_mode = "study_path"
            prompt_user_context = self.user_workspace.build_prompt_context(
                user_id=user_id,
                project_id=resolved_project_id,
            )
            if resolved_project is not None:
                self.user_workspace.remember_project_context(
                    user_id=user_id,
                    project=resolved_project,
                    confidence=0.9,
                )
            self.user_workspace.append_chat_message(
                session_id=persisted_chat_session_id,
                role="user",
                content=question,
                answer_mode=requested_answer_mode,
                madhhab=selected_madhhab,
                evidence_json=None,
            )
            if selected_madhhab not in {"", "not_specified", "compare_all"}:
                self.user_workspace.remember_preference(
                    user_id=user_id,
                    key="selected_madhhab",
                    value=selected_madhhab,
                    source_session_id=persisted_chat_session_id,
                    confidence=0.84,
                )
            if answer_mode:
                self.user_workspace.remember_preference(
                    user_id=user_id,
                    key="answer_mode",
                    value=answer_mode,
                    source_session_id=persisted_chat_session_id,
                    confidence=0.8,
                )
        retrieval_summary: dict[str, Any] = {"status": "not_run"}
        continuity_started = perf_counter()
        continuity = self.session_continuity.resolve(
            session_id=actor.session_id,
            question=question,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
        )
        stage_latency_ms["continuity"] = int((perf_counter() - continuity_started) * 1000)
        effective_question = continuity.effective_question
        continuity_answer_mode = continuity.effective_answer_mode
        effective_selected_madhhab = continuity.effective_selected_madhhab
        baseline_query_intent = route_query_intent(
            question=effective_question,
            answer_mode=continuity_answer_mode,
            selected_madhhab=effective_selected_madhhab,
        )
        baseline_effective_answer_mode = _resolve_effective_answer_mode(
            requested_answer_mode=continuity_answer_mode,
            query_intent_id=baseline_query_intent.intent_id,
        )
        research_depth_started = perf_counter()
        research_depth_decision = plan_research_depth(
            requested_research_depth=research_depth or config.research_depth_default,
            query_intent=baseline_query_intent,
            continuity_used=continuity.used,
        )
        stage_latency_ms["research_depth"] = int(
            (perf_counter() - research_depth_started) * 1000
        )
        self.runtime_state.last_micro_fast = self.micro_fast.run(
            config=config,
            original_question=question,
            effective_question=effective_question,
            rule_router_intent=baseline_query_intent.intent_id,
            session_id=actor.session_id,
            request_id=actor.request_id,
        )
        stage_latency_ms["micro_fast"] = int(
            self.runtime_state.last_micro_fast.get("latency_ms") or 0
        )
        assist_decision = _evaluate_micro_fast_assist(
            config=config,
            fast_result=self.runtime_state.last_micro_fast,
            baseline_query_intent=baseline_query_intent,
            baseline_answer_mode=baseline_effective_answer_mode,
            effective_question=effective_question,
            selected_madhhab=effective_selected_madhhab,
        )
        query_intent = assist_decision["query_intent"]
        retrieval_question = str(
            assist_decision.get("retrieval_question") or effective_question
        )
        retrieval_question = _rewrite_retrieval_question_for_worship_topic(
            question=retrieval_question,
            query_intent=query_intent,
        )
        effective_answer_mode = str(
            assist_decision.get("effective_answer_mode") or baseline_effective_answer_mode
        )
        fast_path_started = perf_counter()
        fast_path_decision = classify_fast_path(
            config=config,
            question=effective_question,
            answer_mode=effective_answer_mode,
            query_intent=query_intent,
            selected_madhhab=effective_selected_madhhab,
            continuity_used=continuity.used,
            micro_fast_confidence=_coerce_float(
                self.runtime_state.last_micro_fast.get("confidence")
            ),
            micro_fast_intent=str(
                self.runtime_state.last_micro_fast.get("micro_predicted_intent") or ""
            )
            or None,
            research_depth=research_depth_decision.effective_research_depth,
        )
        stage_latency_ms["fast_path_classifier"] = int(
            (perf_counter() - fast_path_started) * 1000
        )
        mode_hooks = _derive_mode_hooks(
            query_intent_intent_id=query_intent.intent_id,
            selected_madhhab=effective_selected_madhhab,
            continuity_hooks=continuity.mode_hooks,
        )
        public_assist_decision = _assist_decision_public_payload(assist_decision)
        self.runtime_state.last_micro_fast_assist_decision = public_assist_decision
        if assist_decision.get("assist_applied"):
            self.runtime_state.micro_fast_assist_applied_count += 1
        else:
            self.runtime_state.micro_fast_assist_rejected_count += 1
        self._emit_operator_event(
            event_type=(
                "micro_fast_assist_applied"
                if assist_decision.get("assist_applied")
                else "micro_fast_assist_rejected"
            ),
            severity="info" if assist_decision.get("assist_applied") else "warning",
            summary=(
                "Micro fast assist applied"
                if assist_decision.get("assist_applied")
                else "Micro fast assist rejected"
            ),
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/chat",
            details={
                "assist_decision": public_assist_decision,
                "baseline_intent": baseline_query_intent.intent_id,
                "effective_intent": query_intent.intent_id,
            },
        )
        chat_diagnostics: dict[str, Any] = {
            "mode": effective_answer_mode,
            "request_id": actor.request_id,
            "session_id": actor.session_id,
            "original_question": question,
            "effective_question": effective_question,
            "retrieval_question": retrieval_question,
            "original_answer_mode": requested_answer_mode,
            "continuity_answer_mode": continuity_answer_mode,
            "baseline_effective_answer_mode": baseline_effective_answer_mode,
            "effective_answer_mode": effective_answer_mode,
            "requested_research_depth": research_depth_decision.requested_research_depth,
            "effective_research_depth": research_depth_decision.effective_research_depth,
            "depth_auto_upgraded": research_depth_decision.depth_auto_upgraded,
            "depth_upgrade_reason": research_depth_decision.depth_upgrade_reason,
            "original_selected_madhhab": requested_selected_madhhab,
            "effective_selected_madhhab": effective_selected_madhhab,
            "baseline_detected_intent": baseline_query_intent.intent_id,
            "detected_intent": query_intent.intent_id,
            "worship_topic": query_intent.worship_topic,
            "mode_hooks": mode_hooks,
            "hanafi_first_triggered": effective_selected_madhhab == "hanafi",
            "compare_views_triggered": query_intent.intent_id == "compare_views",
            "source_only_enforced": (
                query_intent.intent_id == "source_only"
                or effective_answer_mode == "source_only"
            ),
            "query_complexity": fast_path_decision.query_complexity,
            "path_used": fast_path_decision.path_used,
            "fast_path_reason": fast_path_decision.fast_path_reason,
            "fast_path_rejected_reason": fast_path_decision.fast_path_rejected_reason,
            "fast_path_classifier": fast_path_decision.as_dict(),
            "continuity": continuity.as_debug_payload(),
            "micro_fast_intent": self.runtime_state.last_micro_fast.get(
                "micro_predicted_intent"
            ),
            "micro_fast_suggested_mode": self.runtime_state.last_micro_fast.get(
                "micro_suggested_mode"
            ),
            "micro_fast_rewritten_query": self.runtime_state.last_micro_fast.get(
                "micro_rewritten_query"
            ),
            "micro_fast_confidence": self.runtime_state.last_micro_fast.get("confidence"),
            "micro_smart_status": (
                "skipped_admin_only"
                if not config.micro_smart_inline_enabled and config.micro_smart_admin_only
                else "pending"
            ),
            "assist_applied": bool(assist_decision.get("assist_applied")),
            "assist_rejected_reason": assist_decision.get("assist_rejected_reason"),
            "assist_applied_changes": assist_decision.get("applied_changes", []),
            "assist_decision": public_assist_decision,
            "authenticated_user_id": user_id,
            "chat_session_id": persisted_chat_session_id,
            "study_project_id": resolved_project_id,
            "project_study_mode": (
                str(resolved_project.get("study_mode") or "standard")
                if isinstance(resolved_project, dict)
                else "standard"
            ),
            "user_context_applied": bool(prompt_user_context),
            "routing_tool_available": self.routing_provider.available(),
            "drive_time_unavailable": False,
            "route_used": None,
            "tool_used": None,
            "stage_latency_ms": stage_latency_ms,
        }
        safe_mode = _safe_mode_state()
        chat_diagnostics.update(
            {
                "safe_mode": safe_mode["active"],
                "safe_mode_reason": safe_mode["reason"],
                "usb_test_mode": safe_mode["usb_test_mode"],
            }
        )
        self._update_request_phase(
            actor=actor,
            phase="receiving_request",
            summary="Receiving request",
            retrieval_active=False,
            model_active=False,
            degraded_active=False,
            completed=False,
            details=chat_diagnostics,
        )
        if DEBUG_FAST:
            debug_response = self._build_degraded_response(
                question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                greeting_style=greeting_style,
                tone_level=tone_level,
                sources=[],
                query_intent=query_intent,
                reason="FAST TEST OK",
            )
            self.runtime_state.last_chat_status = "debug"
            self.runtime_state.last_degraded_subsystem = "debug"
            self.runtime_state.last_degraded_reason = "debug_fast"
            chat_diagnostics.update(debug_response.get("answer_diagnostics", {}))
            chat_diagnostics.update(
                {
                    "debug_fast": True,
                    "debug_fast_reason": "HJ_DEBUG_FAST enabled",
                    "stage_latency_ms": {
                        **stage_latency_ms,
                        "total": int((perf_counter() - started) * 1000),
                    },
                }
            )
            chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                session_id=actor.session_id,
                request_id=actor.request_id,
                original_question=question,
                effective_question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                snippets=[],
                continuity=continuity,
            )
            _ask_log(
                f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
            )
            return self._finalize_chat_result(
                actor=actor,
                question=question,
                answer_mode=effective_answer_mode,
                selected_madhhab=effective_selected_madhhab,
                started=started,
                result=ChatExecutionResult(
                    status="debug",
                    request_id=actor.request_id,
                    answer=debug_response["answer"],
                    rendered_text=debug_response["rendered_text"],
                    model_requested=None,
                    model_used=None,
                    fallback_used=False,
                    fallback_from=None,
                    fallback_to=None,
                    retrieval_summary=retrieval_summary,
                    degraded=True,
                    degraded_subsystem="debug",
                    degraded_reason="debug_fast",
                    diagnostics=chat_diagnostics,
                ),
                user_id=user_id,
                chat_session_id=persisted_chat_session_id,
                project_id=resolved_project_id,
                error=None,
            )
        if safe_mode["active"]:
            safe_mode_response = self._build_degraded_response(
                question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                greeting_style=greeting_style,
                tone_level=tone_level,
                sources=[],
                query_intent=query_intent,
                reason="System running in safe mode. Limited functionality.",
            )
            self.runtime_state.last_chat_status = "degraded"
            self.runtime_state.last_degraded_subsystem = "startup"
            self.runtime_state.last_degraded_reason = safe_mode["reason"] or "safe_mode"
            chat_diagnostics.update(safe_mode_response.get("answer_diagnostics", {}))
            chat_diagnostics.update(
                {
                    "safe_mode_returned": True,
                    "stage_latency_ms": {
                        **stage_latency_ms,
                        "total": int((perf_counter() - started) * 1000),
                    },
                }
            )
            chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                session_id=actor.session_id,
                request_id=actor.request_id,
                original_question=question,
                effective_question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                snippets=[],
                continuity=continuity,
            )
            _ask_log(
                f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
            )
            return self._finalize_chat_result(
                actor=actor,
                question=question,
                answer_mode=effective_answer_mode,
                selected_madhhab=effective_selected_madhhab,
                started=started,
                result=ChatExecutionResult(
                    status="degraded",
                    request_id=actor.request_id,
                    answer=safe_mode_response["answer"],
                    rendered_text=safe_mode_response["rendered_text"],
                    model_requested=None,
                    model_used=None,
                    fallback_used=False,
                    fallback_from=None,
                    fallback_to=None,
                    retrieval_summary=retrieval_summary,
                    degraded=True,
                    degraded_subsystem="startup",
                    degraded_reason=safe_mode["reason"] or "safe_mode",
                    diagnostics=chat_diagnostics,
                ),
                user_id=user_id,
                chat_session_id=persisted_chat_session_id,
                project_id=resolved_project_id,
                error=None,
            )
        model_client = RuntimeAwareOllamaClient(config, self.runtime_state, self.path_profile)
        ask_pipeline = AskPipeline(
            repo_root=self.repo_root,
            client=model_client,
            artifacts=self.artifacts,
            quote_window_chars=config.citation_excerpt_length,
        )
        sources = list(retrieved_sources or [])
        caller_supplied_sources = bool(retrieved_sources)
        retrieval_debug_payload: dict[str, Any] = {}
        retrieval_timeout_seconds = _retrieval_timeout_seconds_for_depth(
            config,
            research_depth_decision.effective_research_depth,
        )
        try:
            retrieval_started = perf_counter()
            if not sources:
                self._update_request_phase(
                    actor=actor,
                    phase="retrieving_sources",
                    summary="Retrieving sources",
                    retrieval_active=True,
                    model_active=False,
                    degraded_active=False,
                    completed=False,
                    details={
                        **chat_diagnostics,
                        "candidate_limit": fast_path_decision.retrieval_candidate_limit,
                        "retrieval_timeout_seconds": retrieval_timeout_seconds,
                        "retrieval_warmup_status": self.runtime_state.retrieval_warmup_status,
                    },
                )
                warmup_wait = self._await_retrieval_warmup(config, trigger="request")
                chat_diagnostics["retrieval_timeout_seconds"] = retrieval_timeout_seconds
                chat_diagnostics["retrieval_warmup"] = warmup_wait
                if not self.runtime_state.retrieval_loaded:
                    retrieval_elapsed_ms = int((perf_counter() - retrieval_started) * 1000)
                    stage_latency_ms["retrieval"] = retrieval_elapsed_ms
                    stage_latency_ms["rerank"] = 0
                    stage_latency_ms["retrieval_total"] = retrieval_elapsed_ms
                    degraded_reason = (
                        "retrieval_warmup_in_progress"
                        if warmup_wait["status"] == "warming"
                        else str(
                            warmup_wait.get("bootstrap_fallback_reason")
                            or warmup_wait.get("error")
                            or "retrieval_not_ready"
                        )
                    )
                    degraded_status = (
                        "timeout_fallback"
                        if warmup_wait["status"] == "warming"
                        else "degraded"
                    )
                    self.runtime_state.last_degraded_subsystem = "retrieval"
                    self.runtime_state.last_degraded_reason = degraded_reason
                    self.runtime_state.last_chat_status = degraded_status
                    self._update_request_phase(
                        actor=actor,
                        phase="degraded_source_backed_response",
                        summary=(
                            "Retrieval index is still warming"
                            if warmup_wait["status"] == "warming"
                            else "Retrieval unavailable - returning degraded answer"
                        ),
                        retrieval_active=False,
                        model_active=False,
                        degraded_active=True,
                        completed=False,
                        outcome=degraded_status,
                        fallback_active=True,
                        details={
                            **chat_diagnostics,
                            "degraded_subsystem": "retrieval",
                            "degraded_reason": degraded_reason,
                            "retrieval_warmup": warmup_wait,
                        },
                    )
                    degraded = self._build_degraded_response(
                        question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        greeting_style=greeting_style,
                        tone_level=tone_level,
                        sources=[],
                        query_intent=query_intent,
                        reason=_retrieval_timeout_message(
                            warmup_status=str(warmup_wait["status"] or ""),
                            failure_reason=degraded_reason,
                        ),
                    )
                    chat_diagnostics.update(degraded.get("answer_diagnostics", {}))
                    if degraded_status == "timeout_fallback":
                        chat_diagnostics["retrieval_timeout_fallback"] = True
                        chat_diagnostics["timeout_fallback"] = True
                        chat_diagnostics["timeout_fallback_reason"] = degraded_reason
                    chat_diagnostics["stage_latency_ms"] = {
                        **stage_latency_ms,
                        "total": int((perf_counter() - started) * 1000),
                    }
                    chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                        session_id=actor.session_id,
                        request_id=actor.request_id,
                        original_question=question,
                        effective_question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        snippets=[],
                        continuity=continuity,
                    )
                    _ask_log(
                        f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                    )
                    return self._finalize_chat_result(
                        actor=actor,
                        question=question,
                        answer_mode=effective_answer_mode,
                        selected_madhhab=effective_selected_madhhab,
                        started=started,
                        result=ChatExecutionResult(
                            status=degraded_status,
                            request_id=actor.request_id,
                            answer=degraded["answer"],
                            rendered_text=degraded["rendered_text"],
                            model_requested=None,
                            model_used=None,
                            fallback_used=True,
                            fallback_from="retrieval",
                            fallback_to=degraded_status,
                            retrieval_summary={
                                **retrieval_summary,
                                "status": "timeout"
                                if degraded_status == "timeout_fallback"
                                else "error",
                                "bootstrap_source": warmup_wait.get("bootstrap_source") or "",
                                "bootstrap_fallback_reason": warmup_wait.get(
                                    "bootstrap_fallback_reason"
                                )
                                or "",
                            },
                            degraded=True,
                            degraded_subsystem="retrieval",
                            degraded_reason=degraded_reason,
                            diagnostics=chat_diagnostics,
                        ),
                        user_id=user_id,
                        chat_session_id=persisted_chat_session_id,
                        project_id=resolved_project_id,
                        error=None,
                    )
                retrieval_executor = futures.ThreadPoolExecutor(max_workers=1)
                retrieval_future = retrieval_executor.submit(
                    self._retrieve_sources_for_chat,
                    question=retrieval_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    query_intent=query_intent,
                    top_k=fast_path_decision.retrieval_top_k,
                    candidate_limit=fast_path_decision.retrieval_candidate_limit,
                    deadline=perf_counter() + retrieval_timeout_seconds,
                )
                try:
                    sources, retrieval_debug_payload, retrieval_stats = retrieval_future.result(
                        timeout=retrieval_timeout_seconds
                    )
                except futures.TimeoutError as exc:
                    retrieval_future.cancel()
                    retrieval_elapsed_ms = int((perf_counter() - retrieval_started) * 1000)
                    stage_latency_ms["retrieval"] = retrieval_elapsed_ms
                    stage_latency_ms["rerank"] = 0
                    stage_latency_ms["retrieval_total"] = retrieval_elapsed_ms
                    _ask_log(
                        f"[ASK] retrieval done: {stage_latency_ms['retrieval'] / 1000:.3f} (timeout)"
                    )
                    self.runtime_state.last_degraded_subsystem = "retrieval"
                    self.runtime_state.last_degraded_reason = "retrieval_call_exceeded_budget"
                    self.runtime_state.last_chat_status = "timeout_fallback"
                    self._update_request_phase(
                        actor=actor,
                        phase="degraded_source_backed_response",
                        summary="Retrieval timeout - returning fast fallback",
                        retrieval_active=False,
                        model_active=False,
                        degraded_active=True,
                        completed=False,
                        outcome="timeout_fallback",
                        fallback_active=True,
                        details={
                            **chat_diagnostics,
                            "degraded_subsystem": "retrieval",
                            "degraded_reason": "retrieval_call_exceeded_budget",
                        },
                    )
                    degraded = self._build_degraded_response(
                        question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        greeting_style=greeting_style,
                        tone_level=tone_level,
                        sources=[],
                        query_intent=query_intent,
                        reason=_retrieval_timeout_message(
                            warmup_status=self.runtime_state.retrieval_warmup_status,
                            failure_reason="retrieval_call_exceeded_budget",
                        ),
                    )
                    chat_diagnostics.update(degraded.get("answer_diagnostics", {}))
                    chat_diagnostics.update(
                        {
                            "retrieval_timeout_fallback": True,
                            "timeout_fallback": True,
                            "timeout_fallback_reason": "retrieval_call_exceeded_budget",
                            "stage_latency_ms": {
                                **stage_latency_ms,
                                "total": int((perf_counter() - started) * 1000),
                            },
                        }
                    )
                    chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                        session_id=actor.session_id,
                        request_id=actor.request_id,
                        original_question=question,
                        effective_question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        snippets=[],
                        continuity=continuity,
                    )
                    _ask_log(
                        f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                    )
                    return self._finalize_chat_result(
                        actor=actor,
                        question=question,
                        answer_mode=effective_answer_mode,
                        selected_madhhab=effective_selected_madhhab,
                        started=started,
                        result=ChatExecutionResult(
                            status="timeout_fallback",
                            request_id=actor.request_id,
                            answer=degraded["answer"],
                            rendered_text=degraded["rendered_text"],
                            model_requested=None,
                            model_used=None,
                            fallback_used=True,
                            fallback_from="retrieval",
                            fallback_to="timeout_fallback",
                            retrieval_summary={
                                **retrieval_summary,
                                "status": "timeout",
                            },
                            degraded=True,
                            degraded_subsystem="retrieval",
                            degraded_reason="retrieval_call_exceeded_budget",
                            diagnostics=chat_diagnostics,
                        ),
                        user_id=user_id,
                        chat_session_id=persisted_chat_session_id,
                        project_id=resolved_project_id,
                        error=exc,
                    )
                except Exception as exc:
                    retrieval_future.cancel()
                    retrieval_elapsed_ms = int((perf_counter() - retrieval_started) * 1000)
                    stage_latency_ms["retrieval"] = retrieval_elapsed_ms
                    stage_latency_ms["rerank"] = 0
                    stage_latency_ms["retrieval_total"] = retrieval_elapsed_ms
                    failure_reason = str(
                        self.runtime_state.retrieval_warmup_fallback_reason
                        or self.retrieval._last_bootstrap_reason
                        or exc
                    )
                    self.runtime_state.last_degraded_subsystem = "retrieval"
                    self.runtime_state.last_degraded_reason = failure_reason
                    self.runtime_state.last_chat_status = "degraded"
                    self._update_request_phase(
                        actor=actor,
                        phase="degraded_source_backed_response",
                        summary="Retrieval unavailable - returning degraded answer",
                        retrieval_active=False,
                        model_active=False,
                        degraded_active=True,
                        completed=False,
                        outcome="degraded",
                        fallback_active=True,
                        details={
                            **chat_diagnostics,
                            "degraded_subsystem": "retrieval",
                            "degraded_reason": failure_reason,
                        },
                    )
                    degraded = self._build_degraded_response(
                        question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        greeting_style=greeting_style,
                        tone_level=tone_level,
                        sources=[],
                        query_intent=query_intent,
                        reason=_retrieval_timeout_message(
                            warmup_status=self.runtime_state.retrieval_warmup_status,
                            failure_reason=failure_reason,
                        ),
                    )
                    chat_diagnostics.update(degraded.get("answer_diagnostics", {}))
                    chat_diagnostics["stage_latency_ms"] = {
                        **stage_latency_ms,
                        "total": int((perf_counter() - started) * 1000),
                    }
                    chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                        session_id=actor.session_id,
                        request_id=actor.request_id,
                        original_question=question,
                        effective_question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        snippets=[],
                        continuity=continuity,
                    )
                    _ask_log(
                        f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                    )
                    return self._finalize_chat_result(
                        actor=actor,
                        question=question,
                        answer_mode=effective_answer_mode,
                        selected_madhhab=effective_selected_madhhab,
                        started=started,
                        result=ChatExecutionResult(
                            status="degraded",
                            request_id=actor.request_id,
                            answer=degraded["answer"],
                            rendered_text=degraded["rendered_text"],
                            model_requested=None,
                            model_used=None,
                            fallback_used=True,
                            fallback_from="retrieval",
                            fallback_to="degraded",
                            retrieval_summary={
                                **retrieval_summary,
                                "status": "error",
                                "bootstrap_source": self.retrieval._last_bootstrap_source,
                                "bootstrap_fallback_reason": self.retrieval._last_bootstrap_reason,
                            },
                            degraded=True,
                            degraded_subsystem="retrieval",
                            degraded_reason=failure_reason,
                            diagnostics=chat_diagnostics,
                        ),
                        user_id=user_id,
                        chat_session_id=persisted_chat_session_id,
                        project_id=resolved_project_id,
                        error=exc,
                    )
                finally:
                    retrieval_executor.shutdown(wait=False, cancel_futures=True)
            else:
                sources, retrieval_debug_payload, retrieval_stats = self._retrieve_sources_for_chat(
                    question=retrieval_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    query_intent=query_intent,
                    top_k=fast_path_decision.retrieval_top_k,
                    candidate_limit=fast_path_decision.retrieval_candidate_limit,
                    provided_sources=sources,
                )
            retrieval_elapsed_ms = int((perf_counter() - retrieval_started) * 1000)
            retrieval_timings = retrieval_debug_payload.get("timings_ms", {})
            stage_latency_ms["retrieval"] = int(
                retrieval_timings.get("retrieval", retrieval_elapsed_ms)
            )
            stage_latency_ms["rerank"] = int(retrieval_timings.get("rerank", 0))
            stage_latency_ms["retrieval_total"] = retrieval_elapsed_ms
            self.runtime_state.retrieval_loaded = True
            self.runtime_state.last_retrieval_stats = retrieval_stats
            retrieval_summary = {
                "status": "ok" if sources else "empty",
                "source_count": len(sources),
                "candidate_count": int(retrieval_debug_payload.get("candidate_count", 0)),
                "detected_intent": query_intent.intent_id,
                "detected_madhhab_intent": retrieval_debug_payload.get(
                    "detected_madhhab_intent", ""
                ),
                "madhhab_boost_applied": bool(
                    retrieval_debug_payload.get("madhhab_boost_applied", False)
                ),
                "madhhab_fallback_used": bool(
                    retrieval_debug_payload.get("madhhab_fallback_used", False)
                ),
                "bootstrap_source": retrieval_debug_payload.get("bootstrap_source", ""),
                "bootstrap_fallback_reason": retrieval_debug_payload.get(
                    "bootstrap_fallback_reason", ""
                ),
                "prepared_search_loaded": bool(
                    retrieval_debug_payload.get("prepared_search_loaded", False)
                ),
                "prepared_search_fallback_reason": retrieval_debug_payload.get(
                    "prepared_search_fallback_reason", ""
                ),
                "prepared_search_load_ms": int(
                    retrieval_debug_payload.get("prepared_search_load_ms", 0) or 0
                ),
                "prepared_search_build_ms": int(
                    retrieval_debug_payload.get("prepared_search_build_ms", 0) or 0
                ),
                "warmup_total_ms": int(
                    retrieval_debug_payload.get("warmup_total_ms", 0) or 0
                ),
                "top_source_classifications": retrieval_debug_payload.get(
                    "top_source_classifications", {}
                ),
                "top_source_families": retrieval_debug_payload.get(
                    "top_source_families", {}
                ),
                "top_collections": retrieval_debug_payload.get("top_collections", {}),
                "top_source_role_boundaries": retrieval_debug_payload.get(
                    "top_source_role_boundaries", {}
                ),
                "top_source_lineages": retrieval_debug_payload.get(
                    "top_source_lineages", {}
                ),
                **retrieval_stats,
            }
            returned_source_truth = _source_truth_diagnostics(sources)
            retrieval_alignment = _retrieval_alignment_diagnostics(
                retrieval_question,
                sources,
                query_intent=query_intent,
            )
            if (
                caller_supplied_sources
                and _question_targets_supplied_sources(effective_question)
                and retrieval_alignment.get("should_degrade")
            ):
                retrieval_alignment = {
                    **retrieval_alignment,
                    "should_degrade": False,
                    "reason": "caller_supplied_sources_reference",
                }
            finalized_fast_path = finalize_fast_path(
                decision=fast_path_decision,
                config=config,
                snippets=sources,
                retrieval_alignment=retrieval_alignment,
            )
            if not caller_supplied_sources:
                if (
                    fast_path_decision.path_used == "fast"
                    and finalized_fast_path.path_used == "full"
                ):
                    full_retrieval_started = perf_counter()
                    sources, retrieval_debug_payload, retrieval_stats = self._retrieve_sources_for_chat(
                        question=retrieval_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        query_intent=query_intent,
                        top_k=config.rerank_limit,
                        candidate_limit=config.max_retrieval_candidates,
                        deadline=perf_counter() + retrieval_timeout_seconds,
                    )
                    stage_latency_ms["retrieval_full_fallback"] = int(
                        (perf_counter() - full_retrieval_started) * 1000
                    )
                    self.runtime_state.last_retrieval_stats = retrieval_stats
                    retrieval_summary = {
                        "status": "ok" if sources else "empty",
                        "source_count": len(sources),
                        "candidate_count": int(
                            retrieval_debug_payload.get("candidate_count", 0)
                        ),
                        "detected_intent": query_intent.intent_id,
                        "detected_madhhab_intent": retrieval_debug_payload.get(
                            "detected_madhhab_intent", ""
                        ),
                        "madhhab_boost_applied": bool(
                            retrieval_debug_payload.get("madhhab_boost_applied", False)
                        ),
                        "madhhab_fallback_used": bool(
                            retrieval_debug_payload.get("madhhab_fallback_used", False)
                        ),
                        "bootstrap_source": retrieval_debug_payload.get("bootstrap_source", ""),
                        "bootstrap_fallback_reason": retrieval_debug_payload.get(
                            "bootstrap_fallback_reason", ""
                        ),
                        "prepared_search_loaded": bool(
                            retrieval_debug_payload.get("prepared_search_loaded", False)
                        ),
                        "prepared_search_fallback_reason": retrieval_debug_payload.get(
                            "prepared_search_fallback_reason", ""
                        ),
                        "prepared_search_load_ms": int(
                            retrieval_debug_payload.get("prepared_search_load_ms", 0) or 0
                        ),
                        "prepared_search_build_ms": int(
                            retrieval_debug_payload.get("prepared_search_build_ms", 0) or 0
                        ),
                        "warmup_total_ms": int(
                            retrieval_debug_payload.get("warmup_total_ms", 0) or 0
                        ),
                        "top_source_classifications": retrieval_debug_payload.get(
                            "top_source_classifications", {}
                        ),
                        "top_source_families": retrieval_debug_payload.get(
                            "top_source_families", {}
                        ),
                        "top_collections": retrieval_debug_payload.get(
                            "top_collections", {}
                        ),
                        "top_source_role_boundaries": retrieval_debug_payload.get(
                            "top_source_role_boundaries", {}
                        ),
                        "top_source_lineages": retrieval_debug_payload.get(
                            "top_source_lineages", {}
                        ),
                        **retrieval_stats,
                    }
                    returned_source_truth = _source_truth_diagnostics(sources)
                    retrieval_alignment = _retrieval_alignment_diagnostics(
                        retrieval_question,
                        sources,
                        query_intent=query_intent,
                    )
            fast_path_decision = finalized_fast_path
            research_depth_decision = finalize_research_depth(
                decision=research_depth_decision,
                fast_path_decision=fast_path_decision,
                retrieval_alignment=retrieval_alignment,
            )
            chat_diagnostics.update(
                {
                    "requested_research_depth": research_depth_decision.requested_research_depth,
                    "effective_research_depth": research_depth_decision.effective_research_depth,
                    "depth_auto_upgraded": research_depth_decision.depth_auto_upgraded,
                    "depth_upgrade_reason": research_depth_decision.depth_upgrade_reason,
                    "retrieval_candidate_count": retrieval_summary["candidate_count"],
                    "source_count": len(sources),
                    "top_source_classifications": retrieval_summary[
                        "top_source_classifications"
                    ],
                    "top_source_families": retrieval_summary["top_source_families"],
                    "top_collections": retrieval_summary["top_collections"],
                    "top_source_role_boundaries": retrieval_summary[
                        "top_source_role_boundaries"
                    ],
                    "top_source_lineages": retrieval_summary["top_source_lineages"],
                    "returned_source_metadata_completeness": returned_source_truth[
                        "metadata_completeness"
                    ],
                    "returned_source_ocr_usage": returned_source_truth["ocr_usage"],
                    "returned_source_layer_composition": returned_source_truth[
                        "source_layer_composition"
                    ],
                    "retrieval_alignment": retrieval_alignment,
                    "query_complexity": fast_path_decision.query_complexity,
                    "path_used": fast_path_decision.path_used,
                    "fast_path_reason": fast_path_decision.fast_path_reason,
                    "fast_path_rejected_reason": fast_path_decision.fast_path_rejected_reason,
                    "fast_path_classifier": fast_path_decision.as_dict(),
                    "stage_latency_ms": stage_latency_ms,
                }
            )
            micro_shadow_payload: dict[str, Any] | None = None
            if not config.micro_smart_enabled:
                micro_shadow_payload = _skipped_micro_shadow_payload(
                    question=question,
                    effective_question=effective_question,
                    request_id=actor.request_id,
                    session_id=actor.session_id,
                    rule_router_intent=baseline_query_intent.intent_id,
                    error="micro_smart_disabled",
                )
                chat_diagnostics["micro_smart_status"] = "disabled"
                stage_latency_ms["micro_smart_shadow"] = 0
            elif not config.micro_smart_inline_enabled and config.micro_smart_admin_only:
                micro_shadow_payload = _skipped_micro_shadow_payload(
                    question=question,
                    effective_question=effective_question,
                    request_id=actor.request_id,
                    session_id=actor.session_id,
                    rule_router_intent=baseline_query_intent.intent_id,
                    error="skipped_admin_only",
                )
                chat_diagnostics["micro_smart_status"] = "skipped_admin_only"
                stage_latency_ms["micro_smart_shadow"] = 0
            elif fast_path_decision.path_used == "fast":
                micro_shadow_payload = _skipped_micro_shadow_payload(
                    question=question,
                    effective_question=effective_question,
                    request_id=actor.request_id,
                    session_id=actor.session_id,
                    rule_router_intent=baseline_query_intent.intent_id,
                    error="skipped_for_fast_path",
                )
                chat_diagnostics["micro_smart_status"] = "skipped_fast_path"
                stage_latency_ms["micro_smart_shadow"] = 0
            else:
                micro_shadow_payload = self.micro_shadow.run(
                    config=config,
                    original_question=question,
                    effective_question=effective_question,
                    rule_router_intent=baseline_query_intent.intent_id,
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                )
                self.runtime_state.last_micro_shadow = micro_shadow_payload
                self.runtime_state.last_micro_tier_comparison = compare_tier_outputs(
                    self.runtime_state.last_micro_fast,
                    micro_shadow_payload,
                )
                chat_diagnostics["micro_smart_status"] = (
                    "success" if micro_shadow_payload.get("success") else "failure"
                )
                stage_latency_ms["micro_smart_shadow"] = int(
                    micro_shadow_payload.get("latency_ms") or 0
                )
            chat_diagnostics["stage_latency_ms"] = stage_latency_ms
            self._update_request_phase(
                actor=actor,
                phase="grounding_evidence",
                summary="Grounding evidence",
                retrieval_active=False,
                model_active=False,
                degraded_active=False,
                completed=False,
                details={
                    **chat_diagnostics,
                    "retrieved_source_count": len(sources),
                    "retrieval_status": retrieval_summary["status"],
                },
            )
            if not sources:
                self._update_request_phase(
                    actor=actor,
                    phase="degraded_source_backed_response",
                    summary="Degraded source-backed response",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=True,
                    completed=False,
                    outcome="degraded",
                    details={**chat_diagnostics, "reason": "retrieval_empty"},
                )
                degraded = self._build_degraded_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode="source_only",
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=[],
                    query_intent=query_intent,
                    reason="No grounded sources were retrieved, so no answer can be synthesized safely.",
                )
                chat_diagnostics["stage_latency_ms"] = {
                    **stage_latency_ms,
                    "total": int((perf_counter() - started) * 1000),
                }
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode="source_only",
                    snippets=[],
                    continuity=continuity,
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="degraded",
                        request_id=actor.request_id,
                        answer=degraded["answer"],
                        rendered_text=degraded["rendered_text"],
                        model_requested=self.runtime_state.last_model_requested,
                        model_used=None,
                        fallback_used=False,
                        fallback_from=None,
                        fallback_to=None,
                        retrieval_summary=retrieval_summary,
                        degraded=True,
                        degraded_subsystem="retrieval",
                        degraded_reason="retrieval_empty",
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=None,
                )
            if retrieval_alignment.get("should_degrade"):
                self._update_request_phase(
                    actor=actor,
                    phase="degraded_source_backed_response",
                    summary="Degraded source-backed response",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=True,
                    completed=False,
                    outcome="degraded",
                    details={
                        **chat_diagnostics,
                        "reason": "retrieval_low_alignment",
                    },
                )
                degraded = self._build_degraded_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=[],
                    query_intent=query_intent,
                    reason=(
                        "Retrieved material does not align closely enough with the requested topic, "
                        "so no grounded answer can be synthesized safely."
                    ),
                )
                chat_diagnostics["stage_latency_ms"] = {
                    **stage_latency_ms,
                    "total": int((perf_counter() - started) * 1000),
                }
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    snippets=[],
                    continuity=continuity,
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="degraded",
                        request_id=actor.request_id,
                        answer=degraded["answer"],
                        rendered_text=degraded["rendered_text"],
                        model_requested=self.runtime_state.last_model_requested,
                        model_used=None,
                        fallback_used=False,
                        fallback_from=None,
                        fallback_to=None,
                        retrieval_summary=retrieval_summary,
                        degraded=True,
                        degraded_subsystem="retrieval",
                        degraded_reason="retrieval_low_alignment",
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=None,
                )

            location_tool_started = perf_counter()
            location_decision = evaluate_location_tool(
                question=effective_question,
                snippets=sources,
                routing_provider=self.routing_provider,
                max_comparison_sites=config.location_tool_max_comparison_sites,
            )
            stage_latency_ms["location_tool"] = int(
                (perf_counter() - location_tool_started) * 1000
            )
            chat_diagnostics.update(
                {
                    "routing_tool_available": location_decision.routing_tool_available,
                    "drive_time_unavailable": location_decision.drive_time_unavailable,
                    "route_used": location_decision.route_used,
                    "tool_used": location_decision.tool_used,
                    "location_tool": location_decision.as_dict(),
                    "stage_latency_ms": stage_latency_ms,
                }
            )
            if location_decision.query_detected and (
                location_decision.supported or location_decision.requires_clarification
            ):
                self._update_request_phase(
                    actor=actor,
                    phase="generating_response",
                    summary="Generating grounded location-tool response",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=False,
                    completed=False,
                    details={
                        **chat_diagnostics,
                        "location_tool_used": True,
                    },
                )
                answer_assembly_started = perf_counter()
                response = self._build_location_tool_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=location_decision.selected_sources or sources,
                    query_intent=query_intent,
                    direct_answer=location_decision.answer_text
                    or "No bounded grounded site comparison is available.",
                    uncertainty_note=location_decision.uncertainty_note,
                    route_used=location_decision.route_used,
                    tool_used=location_decision.tool_used,
                )
                stage_latency_ms["model"] = 0
                stage_latency_ms["answer_assembly"] = int(
                    (perf_counter() - answer_assembly_started) * 1000
                )
                self.runtime_state.last_chat_status = "ok"
                self.runtime_state.last_degraded_subsystem = None
                self.runtime_state.last_degraded_reason = None
                chat_diagnostics.update(response["answer_diagnostics"])
                chat_diagnostics["stage_latency_ms"] = {
                    **stage_latency_ms,
                    "total": int((perf_counter() - started) * 1000),
                }
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    snippets=location_decision.selected_sources or sources,
                    continuity=continuity,
                )
                _ask_log(
                    f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="ok",
                        request_id=actor.request_id,
                        answer=response["answer"],
                        rendered_text=response["rendered_text"],
                        model_requested=self.runtime_state.last_model_requested,
                        model_used=None,
                        fallback_used=False,
                        fallback_from=None,
                        fallback_to=None,
                        retrieval_summary={
                            **retrieval_summary,
                            "location_tool_used": True,
                        },
                        degraded=False,
                        degraded_subsystem=None,
                        degraded_reason=None,
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=None,
                )

            if _can_use_local_fast_response(
                fast_path_decision=fast_path_decision,
                query_intent=query_intent,
            ):
                self._update_request_phase(
                    actor=actor,
                    phase="generating_response",
                    summary="Generating fast-path response",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=False,
                    completed=False,
                    fallback_active=False,
                    details={**chat_diagnostics, "path_used": "fast"},
                )
                stage_latency_ms["model"] = 0
                answer_assembly_started = perf_counter()
                response = self._build_fast_local_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=sources,
                    query_intent=query_intent,
                )
                stage_latency_ms["answer_assembly"] = int(
                    (perf_counter() - answer_assembly_started) * 1000
                )
                self.runtime_state.last_chat_status = "ok"
                self.runtime_state.last_degraded_subsystem = None
                self.runtime_state.last_degraded_reason = None
                chat_diagnostics.update(response["answer_diagnostics"])
                chat_diagnostics["stage_latency_ms"] = {
                    **stage_latency_ms,
                    "total": int((perf_counter() - started) * 1000),
                }
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    snippets=sources,
                    continuity=continuity,
                )
                _ask_log(
                    f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="ok",
                        request_id=actor.request_id,
                        answer=response["answer"],
                        rendered_text=response["rendered_text"],
                        model_requested=self.runtime_state.last_model_requested,
                        model_used=None,
                        fallback_used=False,
                        fallback_from=None,
                        fallback_to=None,
                        retrieval_summary=retrieval_summary,
                        degraded=False,
                        degraded_subsystem=None,
                        degraded_reason=None,
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=None,
                )

            requested_model_label = ollama_model
            if (
                not requested_model_label
                and config.main_model_enabled
                and config.main_model_provider == "local_gguf"
            ):
                requested_model_label = LocalGGUFMainModelRuntime.resolve_model_name(
                    config=config,
                    path_profile=self.path_profile,
                )
            if not requested_model_label and config.model_preference_order:
                requested_model_label = config.model_preference_order[0]
            if not requested_model_label:
                requested_model_label = "unconfigured"

            if sources and not config.main_model_enabled:
                self._update_request_phase(
                    actor=actor,
                    phase="generating_response",
                    summary="Main model disabled - using retrieval-first fast mode",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=False,
                    completed=False,
                    fallback_active=False,
                    details={**chat_diagnostics, "path_used": "fast"},
                )
                stage_latency_ms["model"] = 0
                answer_assembly_started = perf_counter()
                response = self._build_fast_local_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=sources,
                    query_intent=query_intent,
                )
                stage_latency_ms["answer_assembly"] = int(
                    (perf_counter() - answer_assembly_started) * 1000
                )
                self.runtime_state.last_model_requested = None
                self.runtime_state.last_model_used = None
                self.runtime_state.last_model_backend = None
                self.runtime_state.last_model_fallback_used = False
                self.runtime_state.last_model_fallback_from = None
                self.runtime_state.last_model_fallback_to = None
                self.runtime_state.last_model_error = None
                self.runtime_state.last_main_model_loaded = False
                self.runtime_state.last_chat_status = "fast_mode"
                self.runtime_state.last_degraded_subsystem = None
                self.runtime_state.last_degraded_reason = None
                chat_diagnostics.update(response["answer_diagnostics"])
                chat_diagnostics.update(
                    {
                        "path_used": "fast",
                        "fast_mode_triggered": True,
                        "fast_mode_reason": "main_model_disabled",
                        "final_generation_unavailable": True,
                        "stage_latency_ms": {
                            **stage_latency_ms,
                            "total": int((perf_counter() - started) * 1000),
                        },
                    }
                )
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    snippets=sources,
                    continuity=continuity,
                )
                _ask_log(
                    f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="fast_mode",
                        request_id=actor.request_id,
                        answer=response["answer"],
                        rendered_text=response["rendered_text"],
                        model_requested=None,
                        model_used=None,
                        fallback_used=False,
                        fallback_from=None,
                        fallback_to=None,
                        retrieval_summary=retrieval_summary,
                        degraded=False,
                        degraded_subsystem=None,
                        degraded_reason=None,
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=None,
                )

            if (
                sources
                and (perf_counter() - started) > FAST_MODE_BUDGET_SECONDS
                and (not caller_supplied_sources or FAST_MODE_BUDGET_SECONDS <= 0)
            ):
                self._update_request_phase(
                    actor=actor,
                    phase="generating_response",
                    summary="Switching to fast retrieval summary",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=False,
                    completed=False,
                    fallback_active=True,
                    details={**chat_diagnostics, "path_used": "fast"},
                )
                stage_latency_ms["model"] = 0
                answer_assembly_started = perf_counter()
                response = self._build_fast_local_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=sources,
                    query_intent=query_intent,
                )
                stage_latency_ms["answer_assembly"] = int(
                    (perf_counter() - answer_assembly_started) * 1000
                )
                self.runtime_state.last_chat_status = "fast_mode"
                self.runtime_state.last_degraded_subsystem = None
                self.runtime_state.last_degraded_reason = None
                chat_diagnostics.update(response["answer_diagnostics"])
                chat_diagnostics.update(
                    {
                        "path_used": "fast",
                        "fast_mode_triggered": True,
                        "fast_mode_reason": "budget_exceeded_before_model",
                        "stage_latency_ms": {
                            **stage_latency_ms,
                            "total": int((perf_counter() - started) * 1000),
                        },
                    }
                )
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    snippets=sources,
                    continuity=continuity,
                )
                _ask_log(
                    f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="fast_mode",
                        request_id=actor.request_id,
                        answer=response["answer"],
                        rendered_text=response["rendered_text"],
                        model_requested=self.runtime_state.last_model_requested,
                        model_used=None,
                        fallback_used=True,
                        fallback_from=requested_model_label,
                        fallback_to="local_fast_response",
                        retrieval_summary=retrieval_summary,
                        degraded=False,
                        degraded_subsystem=None,
                        degraded_reason=None,
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=None,
                )

            self._update_request_phase(
                actor=actor,
                phase="querying_model",
                summary="Querying model",
                retrieval_active=False,
                model_active=True,
                degraded_active=False,
                completed=False,
                details={
                    **chat_diagnostics,
                    "requested_model": requested_model_label,
                },
            )
            model_started = perf_counter()
            ask_request = AskRequest(
                question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                retrieved_sources=sources,
                research_depth=research_depth_decision.effective_research_depth,
                greeting_style=greeting_style,
                tone_level=tone_level,
                ollama_model=ollama_model,
                generation_path=fast_path_decision.path_used,
                generation_max_tokens=fast_path_decision.generation_max_tokens,
                user_context=prompt_user_context,
            )
            executor = futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(ask_pipeline.ask, ask_request)
            try:
                response = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            except futures.TimeoutError as exc:
                future.cancel()
                stage_latency_ms["model"] = int((perf_counter() - model_started) * 1000)
                _ask_log(f"[ASK] model done: {stage_latency_ms['model'] / 1000:.3f} (timeout)")
                self._update_request_phase(
                    actor=actor,
                    phase="generating_response",
                    summary="Model timeout - switching to fast mode",
                    retrieval_active=False,
                    model_active=False,
                    degraded_active=False,
                    completed=False,
                    fallback_active=True,
                    details={**chat_diagnostics, "path_used": "fast"},
                )
                if sources:
                    answer_assembly_started = perf_counter()
                    fallback_response = self._build_fast_local_response(
                        question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        greeting_style=greeting_style,
                        tone_level=tone_level,
                        sources=sources,
                        query_intent=query_intent,
                    )
                    stage_latency_ms["answer_assembly"] = int(
                        (perf_counter() - answer_assembly_started) * 1000
                    )
                    self.runtime_state.last_chat_status = "fast_mode"
                    self.runtime_state.last_degraded_subsystem = None
                    self.runtime_state.last_degraded_reason = None
                    chat_diagnostics.update(fallback_response["answer_diagnostics"])
                    chat_diagnostics.update(
                        {
                            "path_used": "fast",
                            "fast_mode_triggered": True,
                            "fast_mode_reason": "model_call_exceeded_5s",
                            "model_timeout_fallback": True,
                            "timeout_fallback": True,
                            "timeout_fallback_reason": "model_call_exceeded_5s",
                            "stage_latency_ms": {
                                **stage_latency_ms,
                                "total": int((perf_counter() - started) * 1000),
                            },
                        }
                    )
                    chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                        session_id=actor.session_id,
                        request_id=actor.request_id,
                        original_question=question,
                        effective_question=effective_question,
                        selected_madhhab=effective_selected_madhhab,
                        answer_mode=effective_answer_mode,
                        snippets=sources,
                        continuity=continuity,
                    )
                    _ask_log(
                        f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                    )
                    return self._finalize_chat_result(
                        actor=actor,
                        question=question,
                        answer_mode=effective_answer_mode,
                        selected_madhhab=effective_selected_madhhab,
                        started=started,
                        result=ChatExecutionResult(
                            status="fast_mode",
                            request_id=actor.request_id,
                            answer=fallback_response["answer"],
                            rendered_text=fallback_response["rendered_text"],
                            model_requested=self.runtime_state.last_model_requested,
                            model_used=None,
                            fallback_used=True,
                            fallback_from=requested_model_label,
                            fallback_to="local_fast_response",
                            retrieval_summary=retrieval_summary,
                            degraded=False,
                            degraded_subsystem=None,
                            degraded_reason=None,
                            diagnostics=chat_diagnostics,
                        ),
                        user_id=user_id,
                        chat_session_id=persisted_chat_session_id,
                        project_id=resolved_project_id,
                        error=exc,
                    )
                degraded = self._build_degraded_response(
                    question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    greeting_style=greeting_style,
                    tone_level=tone_level,
                    sources=sources,
                    query_intent=query_intent,
                    reason="System timeout — switching to fast mode.",
                )
                self.runtime_state.last_chat_status = "timeout_fallback"
                self.runtime_state.last_degraded_subsystem = "model"
                self.runtime_state.last_degraded_reason = "model_call_exceeded_5s"
                chat_diagnostics.update(degraded.get("answer_diagnostics", {}))
                chat_diagnostics.update(
                    {
                        "model_timeout_fallback": True,
                        "timeout_fallback": True,
                        "timeout_fallback_reason": "model_call_exceeded_5s",
                        "stage_latency_ms": {
                            **stage_latency_ms,
                            "total": int((perf_counter() - started) * 1000),
                        },
                    }
                )
                chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                    session_id=actor.session_id,
                    request_id=actor.request_id,
                    original_question=question,
                    effective_question=effective_question,
                    selected_madhhab=effective_selected_madhhab,
                    answer_mode=effective_answer_mode,
                    snippets=sources,
                    continuity=continuity,
                )
                _ask_log(
                    f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
                )
                return self._finalize_chat_result(
                    actor=actor,
                    question=question,
                    answer_mode=effective_answer_mode,
                    selected_madhhab=effective_selected_madhhab,
                    started=started,
                    result=ChatExecutionResult(
                        status="timeout_fallback",
                        request_id=actor.request_id,
                        answer=degraded["answer"],
                        rendered_text=degraded["rendered_text"],
                        model_requested=self.runtime_state.last_model_requested,
                        model_used=None,
                        fallback_used=True,
                        fallback_from=requested_model_label,
                        fallback_to="timeout_fallback",
                        retrieval_summary=retrieval_summary,
                        degraded=True,
                        degraded_subsystem="model",
                        degraded_reason="model_call_exceeded_5s",
                        diagnostics=chat_diagnostics,
                    ),
                    user_id=user_id,
                    chat_session_id=persisted_chat_session_id,
                    project_id=resolved_project_id,
                    error=exc,
                )
            except Exception:
                stage_latency_ms["model"] = int((perf_counter() - model_started) * 1000)
                _ask_log(f"[ASK] model done: {stage_latency_ms['model'] / 1000:.3f}")
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            stage_latency_ms["model"] = int((perf_counter() - model_started) * 1000)
            _ask_log(f"[ASK] model done: {stage_latency_ms['model'] / 1000:.3f}")
            self._update_request_phase(
                actor=actor,
                phase="generating_response",
                summary="Generating response",
                retrieval_active=False,
                model_active=False,
                degraded_active=False,
                completed=False,
                fallback_active=self.runtime_state.last_model_fallback_used,
                details={**chat_diagnostics, "model_used": response.model_name},
            )
            self.runtime_state.last_chat_status = "ok"
            self.runtime_state.last_degraded_subsystem = None
            self.runtime_state.last_degraded_reason = None
            chat_diagnostics.update(response.answer_diagnostics)
            if response.stage_latency_ms:
                stage_latency_ms.update(response.stage_latency_ms)
            chat_diagnostics["stage_latency_ms"] = {
                **stage_latency_ms,
                "total": int((perf_counter() - started) * 1000),
            }
            chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                session_id=actor.session_id,
                request_id=actor.request_id,
                original_question=question,
                effective_question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                snippets=sources,
                continuity=continuity,
            )
            _ask_log(
                f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
            )
            return self._finalize_chat_result(
                actor=actor,
                question=question,
                answer_mode=effective_answer_mode,
                selected_madhhab=effective_selected_madhhab,
                started=started,
                result=ChatExecutionResult(
                    status="ok",
                    request_id=actor.request_id,
                    answer=response.answer,
                    rendered_text=response.rendered_text,
                    model_requested=self.runtime_state.last_model_requested,
                    model_used=response.model_name,
                    fallback_used=self.runtime_state.last_model_fallback_used,
                    fallback_from=self.runtime_state.last_model_fallback_from,
                    fallback_to=self.runtime_state.last_model_fallback_to,
                    retrieval_summary=retrieval_summary,
                    degraded=False,
                    degraded_subsystem=None,
                    degraded_reason=None,
                    diagnostics=chat_diagnostics,
                ),
                user_id=user_id,
                chat_session_id=persisted_chat_session_id,
                project_id=resolved_project_id,
                error=None,
            )
        except Exception as exc:
            self.runtime_state.last_degraded_subsystem = (
                "model" if sources else "retrieval"
            )
            self.runtime_state.last_degraded_reason = f"{type(exc).__name__}: {exc}"
            self.runtime_state.last_chat_status = "degraded"
            self._update_request_phase(
                actor=actor,
                phase="degraded_source_backed_response",
                summary="Degraded source-backed response",
                retrieval_active=False,
                model_active=False,
                degraded_active=True,
                completed=False,
                outcome="degraded",
                fallback_active=self.runtime_state.last_model_fallback_used,
                details={
                    **chat_diagnostics,
                    "degraded_subsystem": self.runtime_state.last_degraded_subsystem,
                    "degraded_reason": self.runtime_state.last_degraded_reason,
                },
            )
            degraded = self._build_degraded_response(
                question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                greeting_style=greeting_style,
                tone_level=tone_level,
                sources=sources,
                query_intent=query_intent,
                reason=(
                    "The local model is currently unavailable, so this answer is presented in degraded source-backed mode."
                    if sources
                    else "The local answer path is unavailable because retrieval or model initialization failed."
                ),
            )
            chat_diagnostics.update(degraded.get("answer_diagnostics", {}))
            chat_diagnostics["stage_latency_ms"] = {
                **stage_latency_ms,
                "total": int((perf_counter() - started) * 1000),
            }
            chat_diagnostics["session_turn"] = self.session_continuity.record_turn(
                session_id=actor.session_id,
                request_id=actor.request_id,
                original_question=question,
                effective_question=effective_question,
                selected_madhhab=effective_selected_madhhab,
                answer_mode=effective_answer_mode,
                snippets=sources,
                continuity=continuity,
            )
            _ask_log(
                f"[ASK] total: {chat_diagnostics['stage_latency_ms']['total'] / 1000:.3f}"
            )
            result = ChatExecutionResult(
                status="degraded",
                request_id=actor.request_id,
                answer=degraded["answer"],
                rendered_text=degraded["rendered_text"],
                model_requested=self.runtime_state.last_model_requested,
                model_used=self.runtime_state.last_model_used,
                fallback_used=self.runtime_state.last_model_fallback_used,
                fallback_from=self.runtime_state.last_model_fallback_from,
                fallback_to=self.runtime_state.last_model_fallback_to,
                retrieval_summary=retrieval_summary,
                degraded=True,
                degraded_subsystem=self.runtime_state.last_degraded_subsystem,
                degraded_reason=self.runtime_state.last_degraded_reason,
                diagnostics=chat_diagnostics,
            )
            return self._finalize_chat_result(
                actor=actor,
                question=question,
                answer_mode=effective_answer_mode,
                selected_madhhab=effective_selected_madhhab,
                started=started,
                result=result,
                user_id=user_id,
                chat_session_id=persisted_chat_session_id,
                project_id=resolved_project_id,
                error=exc,
            )

    def reload_retrieval_assets(
        self,
        *,
        actor: RoleContext,
        background: bool = False,
        retry_of: str | None = None,
        retry_reason: str | None = None,
    ) -> ActionResult:
        return self._run_managed_task(
            action="reload_retrieval_assets",
            route="/api/admin/retrieval/reload",
            actor=actor,
            feature_toggle="enable_reload",
            operation=self._reload_operation,
            parameters={},
            background=background,
            retry_of=retry_of,
            retry_reason=retry_reason,
        )

    def reindex_documents(
        self,
        *,
        actor: RoleContext,
        background: bool = False,
        retry_of: str | None = None,
        retry_reason: str | None = None,
    ) -> ActionResult:
        config = self.config_store.load_effective()
        return self._run_managed_task(
            action="reindex_documents",
            route="/api/admin/retrieval/reindex",
            actor=actor,
            feature_toggle="enable_reindex",
            operation=self._reindex_operation,
            parameters={
                "force_pdf_normalization": config.reindex_behavior_flags.force_pdf_normalization,
                "preserve_previous_assets_on_failure": (
                    config.reindex_behavior_flags.preserve_previous_assets_on_failure
                ),
            },
            background=background,
            retry_of=retry_of,
            retry_reason=retry_reason,
        )

    def update_runtime_config(self, *, actor: RoleContext, patch: dict[str, Any]) -> dict[str, Any]:
        config = self.config_store.load_effective()
        if not config.admin_feature_toggles.enable_config_updates:
            raise PermissionError("runtime config updates are disabled")
        started = perf_counter()
        result = self.config_store.update(patch)
        action = ActionResult(
            action="update_runtime_config",
            success=True,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            request_id=actor.request_id,
            duration_ms=int((perf_counter() - started) * 1000),
            status="ok",
            changed={"fields": result.changed_fields},
            details={
                "configured": result.configured,
                "effective": result.effective,
            },
        )
        self._write_audit(
            route="/api/admin/config",
            action=action.action,
            actor=actor,
            success=action.success,
            duration_ms=action.duration_ms,
            status=action.status,
            changed=action.changed,
            details=action.details,
        )
        return {
            "action_result": asdict(action),
            "config": {
                "configured": result.configured,
                "defaults": result.defaults,
                "effective": result.effective,
            },
        }

    def inspect_runtime_config(self) -> dict[str, Any]:
        return self.config_store.describe()

    def inspect_permissions(self, *, actor: RoleContext) -> dict[str, Any]:
        return {
            "actor": asdict(actor),
            "permission_mode": self.config_store.load_effective().permission_mode,
            "matrix": permission_matrix(),
        }

    def micro_shadow_status(self) -> dict[str, Any]:
        config = self.config_store.load_effective()
        profile_status = self._model_profile_status(config)
        fast_profile = profile_status.get("micro_fast", {})
        smart_profile = profile_status.get("micro_smart", profile_status.get("micro", {}))
        backend = shadow_backend_status()
        latest_fast = self._latest_micro_fast_result()
        latest_result = self._latest_micro_shadow_result()
        benchmark = self._load_micro_shadow_benchmark()
        latest_comparison = compare_tier_outputs(latest_fast, latest_result)
        benchmark_comparison = benchmark.get("comparison", {}) if benchmark else {}
        latest_success = bool(latest_result and latest_result.get("success"))
        latest_error = (
            str(latest_result.get("error") or "").strip()
            if latest_result
            else (
                smart_profile.get("error")
                or (
                    None
                    if backend.get("available")
                    else "local_gguf_shadow_backend_unavailable"
                )
            )
        )
        latest_status = "never_run"
        if latest_result:
            latest_status = "success" if latest_success else "failure"

        fast_success = bool(latest_fast and latest_fast.get("success"))
        fast_status = "never_run"
        if latest_fast:
            fast_status = "success" if fast_success else "failure"
        fast_error = (
            str(latest_fast.get("error") or "").strip()
            if latest_fast
            else fast_profile.get("error")
        )
        fast_router = {
            "enabled": bool(config.micro_fast_enabled),
            "provider": config.micro_fast_provider,
            "configured_role": "assist",
            "model_name": config.micro_fast_name,
            "tuning_config": {
                "deterministic_rules": config.micro_fast_provider == "rules",
                "provider": config.micro_fast_provider,
                "assist_enabled": config.micro_fast_assist_enabled,
                "assist_confidence_threshold": config.micro_fast_assist_confidence_threshold,
            },
            "model_path": None,
            "asset_present": bool(fast_profile.get("available")),
            "latest_timestamp": latest_fast.get("timestamp") if latest_fast else None,
            "latest_latency_ms": int(latest_fast.get("latency_ms") or 0) if latest_fast else 0,
            "latest_predicted_intent": latest_fast.get("micro_predicted_intent") if latest_fast else None,
            "latest_suggested_mode": latest_fast.get("micro_suggested_mode") if latest_fast else None,
            "latest_rewritten_query": latest_fast.get("micro_rewritten_query") if latest_fast else None,
            "latest_success": fast_success,
            "latest_status": fast_status,
            "last_error": fast_error,
            "benchmark_average_latency_ms": (
                benchmark.get("fast_router", {}).get("average_latency_ms")
                if benchmark
                else None
            ),
            "benchmark_p95_latency_ms": (
                benchmark.get("fast_router", {}).get("p95_latency_ms")
                if benchmark
                else None
            ),
            "benchmark_under_3s": (
                benchmark.get("fast_router", {}).get("under_3s")
                if benchmark
                else None
            ),
            "assist_applied_count": self.runtime_state.micro_fast_assist_applied_count,
            "assist_rejected_count": self.runtime_state.micro_fast_assist_rejected_count,
            "latest_assist_decision": self.runtime_state.last_micro_fast_assist_decision
            or None,
            "log_path": str(self.path_profile.runtime_dir / "micro_llm_fast.jsonl"),
        }
        smart_shadow = {
            "enabled": bool(config.micro_smart_enabled),
            "provider": config.micro_smart_provider,
            "configured_role": config.micro_smart_role,
            "model_name": config.micro_smart_name,
            "model_path": smart_profile.get("resolved_path") or smart_profile.get("path"),
            "model_size_bytes": int(smart_profile.get("size_bytes") or 0),
            "model_size_human": _human_size(int(smart_profile.get("size_bytes") or 0)),
            "asset_present": bool(smart_profile.get("available")),
            "gguf_backend_installed": bool(backend.get("available")),
            "shadow_inference_working": latest_success,
            "latest_timestamp": latest_result.get("timestamp") if latest_result else None,
            "latest_latency_ms": int(latest_result.get("latency_ms") or 0) if latest_result else 0,
            "latest_end_to_end_latency_ms": int(latest_result.get("end_to_end_latency_ms") or 0) if latest_result else 0,
            "latest_cache_state": latest_result.get("cache_state") if latest_result else None,
            "latest_output_valid": bool(latest_result and latest_result.get("output_valid")),
            "latest_warmup_applied": bool(latest_result and latest_result.get("warmup_applied")),
            "latest_warmup_latency_ms": int(latest_result.get("warmup_latency_ms") or 0) if latest_result else 0,
            "latest_predicted_intent": latest_result.get("micro_predicted_intent") if latest_result else None,
            "latest_suggested_mode": latest_result.get("micro_suggested_mode") if latest_result else None,
            "latest_rewritten_query": latest_result.get("micro_rewritten_query") if latest_result else None,
            "latest_success": latest_success,
            "latest_status": latest_status,
            "last_error": latest_error,
            "backend_name": backend.get("backend_name"),
            "backend_version": backend.get("backend_version"),
            "project_local_path": bool(smart_profile.get("project_local")),
            "within_expected_directory": bool(smart_profile.get("within_expected_directory")),
            "log_path": str(self.path_profile.runtime_dir / "micro_llm_shadow.jsonl"),
            "warmup": self.runtime_state.last_micro_shadow_warmup or None,
            "last_cold_latency_ms": benchmark.get("smart_shadow", {}).get("cold_latency_ms") if benchmark else None,
            "last_warm_latency_ms": benchmark.get("smart_shadow", {}).get("warm_latency_ms") if benchmark else None,
            "benchmark_average_latency_ms": benchmark.get("smart_shadow", {}).get("average_latency_ms") if benchmark else None,
            "benchmark_p95_latency_ms": benchmark.get("smart_shadow", {}).get("p95_latency_ms") if benchmark else None,
            "tuning_config": tuning_snapshot(config),
        }
        return {
            "micro_model_asset_present": bool(smart_profile.get("available")),
            "gguf_backend_installed": bool(backend.get("available")),
            "shadow_inference_working": latest_success,
            "configured_role": config.micro_smart_role,
            "model_name": config.micro_smart_name,
            "model_path": smart_profile.get("resolved_path") or smart_profile.get("path"),
            "model_size_bytes": int(smart_profile.get("size_bytes") or 0),
            "model_size_human": _human_size(int(smart_profile.get("size_bytes") or 0)),
            "project_local_path": bool(smart_profile.get("project_local")),
            "within_expected_directory": bool(smart_profile.get("within_expected_directory")),
            "backend_name": backend.get("backend_name"),
            "backend_version": backend.get("backend_version"),
            "shadow_log_path": str(self.path_profile.runtime_dir / "micro_llm_shadow.jsonl"),
            "latest_shadow_timestamp": latest_result.get("timestamp") if latest_result else None,
            "latest_shadow_latency_ms": int(latest_result.get("latency_ms") or 0) if latest_result else 0,
            "latest_end_to_end_latency_ms": int(latest_result.get("end_to_end_latency_ms") or 0) if latest_result else 0,
            "latest_cache_state": latest_result.get("cache_state") if latest_result else None,
            "latest_output_valid": bool(latest_result and latest_result.get("output_valid")),
            "latest_warmup_applied": bool(latest_result and latest_result.get("warmup_applied")),
            "latest_warmup_latency_ms": int(latest_result.get("warmup_latency_ms") or 0) if latest_result else 0,
            "latest_predicted_intent": latest_result.get("micro_predicted_intent") if latest_result else None,
            "latest_suggested_mode": latest_result.get("micro_suggested_mode") if latest_result else None,
            "latest_rewritten_query": latest_result.get("micro_rewritten_query") if latest_result else None,
            "latest_success": latest_success,
            "latest_status": latest_status,
            "last_error": latest_error,
            "warmup": self.runtime_state.last_micro_shadow_warmup or None,
            "last_cold_latency_ms": benchmark.get("cold_latency_ms") if benchmark else None,
            "last_warm_latency_ms": benchmark.get("warm_latency_ms") if benchmark else None,
            "benchmark_average_latency_ms": benchmark.get("average_latency_ms") if benchmark else None,
            "benchmark_p95_latency_ms": benchmark.get("p95_latency_ms") if benchmark else None,
            "benchmark_summary": benchmark,
            "benchmark_artifact_path": str(self.repo_root / "docs" / "readiness" / "micro-llm-benchmark.md"),
            "tuning_config": tuning_snapshot(config),
            "fast_router": fast_router,
            "smart_shadow": smart_shadow,
            "latest_comparison": latest_comparison,
            "benchmark_comparison": benchmark_comparison,
            "assist_applied_count": self.runtime_state.micro_fast_assist_applied_count,
            "assist_rejected_count": self.runtime_state.micro_fast_assist_rejected_count,
            "latest_assist_decision": self.runtime_state.last_micro_fast_assist_decision
            or None,
            "advisory_only": True,
        }

    def run_micro_shadow_smoke(self, *, actor: RoleContext) -> dict[str, Any]:
        config = self.config_store.load_effective()
        started = perf_counter()
        fast_result = self.micro_fast.run(
            config=config,
            original_question="Classify this question for retrieval-first planning: show me sources on ablution.",
            effective_question="Classify this question for retrieval-first planning: show me sources on ablution.",
            rule_router_intent="direct_source_lookup",
            session_id=actor.session_id,
            request_id=actor.request_id,
        )
        result = self.micro_shadow.run(
            config=config,
            original_question="Classify this question for retrieval-first planning: show me sources on ablution.",
            effective_question="Classify this question for retrieval-first planning: show me sources on ablution.",
            rule_router_intent="direct_source_lookup",
            session_id=actor.session_id,
            request_id=actor.request_id,
            warmup_first=config.micro_shadow_warmup_before_smoke,
        )
        comparison = compare_tier_outputs(fast_result, result)
        self.runtime_state.last_micro_fast = fast_result
        self.runtime_state.last_micro_shadow = result
        self.runtime_state.last_micro_tier_comparison = comparison
        duration_ms = int((perf_counter() - started) * 1000)
        success = bool(fast_result.get("success")) and bool(result.get("success"))
        error_summary = result.get("error") or fast_result.get("error") if not success else None

        self._write_audit(
            route="/api/admin/micro-llm/smoke",
            action="run_micro_shadow_smoke",
            actor=actor,
            success=success,
            duration_ms=duration_ms,
            status="ok" if success else "partial",
            changed={},
            details={
                "advisory_only": True,
                "fast_router": fast_result,
                "smart_shadow": result,
                "comparison": comparison,
            },
            error_summary=error_summary,
        )
        self._emit_operator_event(
            event_type="micro_shadow_smoke_test",
            severity="info" if success else "warning",
            summary=(
                "Micro LLM dual-tier smoke test succeeded"
                if success
                else "Micro LLM dual-tier smoke test completed with failures"
            ),
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/admin/micro-llm/smoke",
            details={
                "advisory_only": True,
                "fast_latency_ms": fast_result.get("latency_ms"),
                "smart_latency_ms": result.get("latency_ms"),
                "fast_intent": fast_result.get("micro_predicted_intent"),
                "smart_intent": result.get("micro_predicted_intent"),
                "comparison": comparison,
                "error": error_summary,
            },
        )
        if not bool(result.get("success")):
            self._write_error(
                route="/api/admin/micro-llm/smoke",
                action="run_micro_shadow_smoke",
                actor=actor,
                subsystem="micro_shadow",
                error=RuntimeError(
                    str(result.get("error") or "micro_shadow_smoke_failed")
                ),
                details={"advisory_only": True, "result": result},
            )
        if not bool(fast_result.get("success")):
            self._write_error(
                route="/api/admin/micro-llm/smoke",
                action="run_micro_shadow_smoke",
                actor=actor,
                subsystem="micro_fast",
                error=RuntimeError(
                    str(fast_result.get("error") or "micro_fast_smoke_failed")
                ),
                details={"advisory_only": True, "result": fast_result},
            )
        return {
            "action": "run_micro_shadow_smoke",
            "success": success,
            "status": "ok" if success else "partial",
            "advisory_only": True,
            "latency_ms": result.get("latency_ms"),
            "fast_router": fast_result,
            "smart_shadow": result,
            "comparison": comparison,
            "result": result,
        }

    def run_micro_shadow_benchmark(self, *, actor: RoleContext) -> dict[str, Any]:
        config = self.config_store.load_effective()
        started = perf_counter()
        report = self.build_micro_tier_benchmark_report()
        artifact_paths = write_benchmark_artifacts(report, self.repo_root)
        self.runtime_state.last_micro_fast_benchmark = report.get("fast_router", {})
        self.runtime_state.last_micro_shadow_benchmark = report.get("smart_shadow", {})
        self.runtime_state.last_micro_tier_comparison = report.get("comparison", {})
        duration_ms = int((perf_counter() - started) * 1000)
        success = bool(report.get("success"))
        self._write_audit(
            route="/api/admin/micro-llm/benchmark",
            action="run_micro_shadow_benchmark",
            actor=actor,
            success=success,
            duration_ms=duration_ms,
            status="ok" if success else "partial",
            changed={},
            details={
                "advisory_only": True,
                "report": report,
                "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
            },
            error_summary=None if success else "benchmark_contains_failures",
        )
        self._emit_operator_event(
            event_type="micro_shadow_benchmark",
            severity="info" if success else "warning",
            summary=(
                "Micro LLM dual-tier benchmark completed successfully"
                if success
                else "Micro LLM dual-tier benchmark completed with failures"
            ),
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/admin/micro-llm/benchmark",
            details={
                "advisory_only": True,
                "fast_average_latency_ms": report.get("fast_router", {}).get("average_latency_ms"),
                "smart_average_latency_ms": report.get("smart_shadow", {}).get("average_latency_ms"),
                "failure_count": report.get("failure_count"),
                "comparison": report.get("comparison"),
            },
        )
        return {
            "action": "run_micro_shadow_benchmark",
            "success": success,
            "status": "ok" if success else "partial",
            "advisory_only": True,
            "report": report,
            "fast_router": report.get("fast_router", {}),
            "smart_shadow": report.get("smart_shadow", {}),
            "comparison": report.get("comparison", {}),
            "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        }

    def build_micro_tier_benchmark_report(self) -> dict[str, Any]:
        config = self.config_store.load_effective()
        return self._build_micro_tier_benchmark_report(config)

    def portable_readiness_markdown(self) -> str:
        path = self.repo_root / "docs" / "readiness" / "portable-readiness-report.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return (
            "# Portable Readiness Report\n\n"
            "No saved portable readiness artifact is available yet.\n"
        )

    def update_center_status(self) -> dict[str, Any]:
        config = self.config_store.load_effective()
        return self.update_center.status(config=config)

    def check_for_updates(self, *, actor: RoleContext) -> dict[str, Any]:
        config = self.config_store.load_effective()
        started = perf_counter()
        status = self.update_center.check_for_updates(config=config, trigger="admin")
        duration_ms = int((perf_counter() - started) * 1000)
        self._write_audit(
            route="/api/admin/updates/check",
            action="check_for_updates",
            actor=actor,
            success=True,
            duration_ms=duration_ms,
            status="ok",
            changed={"checked": True},
            details={
                "pending_updates_count": status.get("pending_updates_count"),
                "downloaded_updates_count": status.get("downloaded_updates_count"),
                "last_update_check_status": status.get("last_update_check_status"),
            },
        )
        self._emit_operator_event(
            event_type="update_manifest_checked",
            severity="info",
            summary="Admin checked for available updates",
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/admin/updates/check",
            details={
                "pending_updates_count": status.get("pending_updates_count"),
                "downloaded_updates_count": status.get("downloaded_updates_count"),
                "last_update_check_status": status.get("last_update_check_status"),
            },
        )
        return status

    def download_update(self, *, actor: RoleContext, update_id: str) -> dict[str, Any]:
        config = self.config_store.load_effective()
        started = perf_counter()
        row = self.update_center.download_update(update_id=update_id, config=config)
        duration_ms = int((perf_counter() - started) * 1000)
        self._write_audit(
            route="/api/admin/updates/download",
            action="download_update",
            actor=actor,
            success=True,
            duration_ms=duration_ms,
            status="ok",
            changed={"update_id": update_id},
            details=row,
        )
        self._emit_operator_event(
            event_type="update_downloaded",
            severity="info",
            summary=f"Downloaded update {update_id}",
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/admin/updates/download",
            details=row,
        )
        return row

    def verify_update(self, *, actor: RoleContext, update_id: str) -> dict[str, Any]:
        started = perf_counter()
        row = self.update_center.verify_update(update_id=update_id)
        duration_ms = int((perf_counter() - started) * 1000)
        verified = row.get("verification", {}).get("status") == "verified"
        self._write_audit(
            route="/api/admin/updates/verify",
            action="verify_update",
            actor=actor,
            success=verified,
            duration_ms=duration_ms,
            status="ok" if verified else "rejected",
            changed={"update_id": update_id},
            details=row,
            error_summary=None if verified else "checksum_mismatch",
        )
        self._emit_operator_event(
            event_type="update_verified" if verified else "update_verification_failed",
            severity="info" if verified else "warning",
            summary=(
                f"Verified update {update_id}"
                if verified
                else f"Checksum rejected for update {update_id}"
            ),
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/admin/updates/verify",
            details=row,
        )
        return row

    def apply_update(self, *, actor: RoleContext, update_id: str) -> dict[str, Any]:
        started = perf_counter()
        row = self.update_center.apply_update(update_id=update_id)
        duration_ms = int((perf_counter() - started) * 1000)
        staged = row.get("apply", {}).get("status") == "review_staged"
        self._write_audit(
            route="/api/admin/updates/apply",
            action="apply_update",
            actor=actor,
            success=staged,
            duration_ms=duration_ms,
            status="ok" if staged else "blocked",
            changed={"update_id": update_id},
            details=row,
            error_summary=None if staged else "verification_required",
        )
        self._emit_operator_event(
            event_type="update_review_staged" if staged else "update_apply_blocked",
            severity="info" if staged else "warning",
            summary=(
                f"Staged update {update_id} for review"
                if staged
                else f"Update {update_id} could not be staged before verification"
            ),
            request_id=actor.request_id,
            session_id=actor.session_id,
            route="/api/admin/updates/apply",
            details=row,
        )
        return row

    def rebuild_index_if_needed(self, *, actor: RoleContext) -> dict[str, Any]:
        status = self.update_center_status()
        if not status.get("index_rebuild_needed"):
            return {
                "action": "rebuild_index_if_needed",
                "success": True,
                "status": "noop",
                "details": {
                    "index_rebuild_needed": False,
                    "message": "No staged update currently requests an index rebuild.",
                },
            }
        result = self.reindex_documents(actor=actor, background=False)
        return {
            "action": "rebuild_index_if_needed",
            "success": result.success,
            "status": result.status,
            "details": {
                "index_rebuild_needed": True,
                "reindex_result": asdict(result),
            },
        }

    def retrieval_debug(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
    ) -> dict[str, Any]:
        config = self.config_store.load_effective()
        debug = self.retrieval.retrieve_with_debug(
            question,
            selected_madhhab=selected_madhhab,
            top_k=config.rerank_limit,
            answer_mode=answer_mode,
            candidate_limit=config.max_retrieval_candidates,
        )
        return {
            "question": question,
            "selected_madhhab": selected_madhhab,
            "answer_mode": answer_mode,
            "detected_intent": debug.query_intent.intent_id,
            "query_intent": asdict(debug.query_intent),
            "mode_hooks": _derive_mode_hooks(
                query_intent_intent_id=debug.query_intent.intent_id,
                selected_madhhab=selected_madhhab,
                continuity_hooks=[],
            ),
            "snippets": debug.snippets,
            "metadata_completeness": _metadata_coverage_for_sources(debug.snippets),
            "ocr_usage": _ocr_usage_for_sources(debug.snippets),
            "source_layer_composition": _source_layer_composition_for_sources(
                debug.snippets
            ),
            "location_tool": evaluate_location_tool(
                question=question,
                snippets=debug.snippets,
                routing_provider=self.routing_provider,
                max_comparison_sites=config.location_tool_max_comparison_sites,
            ).as_dict(),
            "retrieval_status": {
                "assets_loaded": self.retrieval._state is not None,
                **debug.corpus_stats,
                "candidate_count": debug.candidate_count,
                "candidate_limit": debug.candidate_limit,
                "top_source_classifications": debug.top_source_classifications,
                "top_source_families": debug.top_source_families,
                "top_collections": debug.top_collections,
                "top_sections": debug.top_sections,
                "top_source_role_boundaries": _count_snippet_field(
                    debug.snippets,
                    field_name="source_role_boundary",
                ),
                "top_source_lineages": _count_snippet_field(
                    debug.snippets,
                    field_name="source_lineage",
                ),
            },
        }

    def _retrieval_stats_snapshot(
        self,
        *,
        index_assets: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        state = self.retrieval._state
        if state is not None:
            return {
                "documents": len(state.corpus.documents),
                "chunks": len(state.chunks),
                "skipped_files": len(state.corpus.skipped_files),
            }
        stats = dict(self.runtime_state.last_retrieval_stats or {})
        assets = index_assets or inspect_persisted_retrieval_index(
            self.path_profile.index_dir,
            repo_root=self.repo_root,
            raw_root=self.path_profile.raw_corpus_dir,
            normalized_root=self.path_profile.normalized_corpus_dir,
        )
        return {
            "documents": int(stats.get("documents") or assets.get("document_count") or 0),
            "chunks": int(stats.get("chunks") or assets.get("chunk_count") or 0),
            "skipped_files": int(
                stats.get("skipped_files")
                or assets.get("skipped_file_count")
                or 0
            ),
        }

    def _retrieve_sources_for_chat(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
        query_intent: QueryIntent,
        top_k: int,
        candidate_limit: int,
        provided_sources: list[dict[str, Any]] | None = None,
        deadline: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
        if provided_sources is not None:
            sources = list(provided_sources)
            state = self.retrieval._state
            if state is not None:
                retrieval_stats = {
                    "documents": len(state.corpus.documents),
                    "chunks": len(state.chunks),
                    "skipped_files": len(state.corpus.skipped_files),
                }
            else:
                retrieval_stats = {
                    "documents": int(
                        self.runtime_state.last_retrieval_stats.get("documents", 0)
                    ),
                    "chunks": int(
                        self.runtime_state.last_retrieval_stats.get("chunks", 0)
                    ),
                    "skipped_files": int(
                        self.runtime_state.last_retrieval_stats.get("skipped_files", 0)
                    ),
                }
            payload = {
                "question": question,
                "selected_madhhab": selected_madhhab,
                "answer_mode": answer_mode,
                "query_intent": asdict(query_intent),
                "bootstrap_source": "provided_sources",
                "bootstrap_fallback_reason": "",
                "prepared_search_loaded": False,
                "prepared_search_fallback_reason": "provided_sources",
                "prepared_search_load_ms": 0,
                "prepared_search_build_ms": 0,
                "warmup_total_ms": 0,
                "candidate_limit": 0,
                "candidate_count": 0,
                "snippets": sources,
                **_summarize_source_snippets(sources),
                "corpus_stats": retrieval_stats,
                "timings_ms": {
                    "retrieval": 0,
                    "rerank": 0,
                    "warmup_total": 0,
                    "prepared_search_load": 0,
                    "prepared_search_build": 0,
                },
            }
            return sources, payload, retrieval_stats

        retrieval_kwargs = {
            "selected_madhhab": selected_madhhab,
            "top_k": top_k,
            "answer_mode": answer_mode,
            "candidate_limit": candidate_limit,
            "deadline": deadline,
        }
        try:
            retrieval_debug = self.retrieval.retrieve_with_debug(
                question,
                allow_corpus_fallback=False,
                **retrieval_kwargs,
            )
        except TypeError as exc:
            if "allow_corpus_fallback" not in str(exc):
                raise
            retrieval_debug = self.retrieval.retrieve_with_debug(
                question,
                **retrieval_kwargs,
            )
        payload = serialize_retrieval_debug(retrieval_debug)
        return retrieval_debug.snippets, payload, self._retrieval_stats_snapshot()

    def active_tasks(self, *, limit: int) -> list[dict[str, Any]]:
        return self.task_store.list_tasks(
            limit=limit,
            statuses=sorted(ACTIVE_TASK_STATUSES),
        )

    def recent_tasks(self, *, limit: int, task_type: str | None = None) -> list[dict[str, Any]]:
        return self.task_store.list_tasks(limit=limit, task_type=task_type)

    def failed_tasks(self, *, limit: int) -> list[dict[str, Any]]:
        return self.task_store.list_tasks(limit=limit, statuses=["failed"])

    def blocked_tasks(self, *, limit: int) -> list[dict[str, Any]]:
        return self.task_store.list_tasks(limit=limit, statuses=["blocked"])

    def degraded_tasks(self, *, limit: int) -> list[dict[str, Any]]:
        return self.task_store.list_tasks(limit=limit, statuses=["degraded"])

    def task_detail(self, *, task_id: str) -> dict[str, Any] | None:
        task = self.task_store.get_task(task_id)
        if not task:
            return None
        return {
            "task": task,
            "events": self.task_store.list_events(task_id=task_id),
        }

    def operator_events(
        self,
        *,
        limit: int,
        severity: str | None = None,
        request_id: str | None = None,
        related_task_id: str | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.log_store.fetch_operator_events(
            limit=limit,
            severity=severity,
            request_id=request_id,
            related_task_id=related_task_id,
            after_id=after_id,
        )

    def request_status(self, *, request_id: str) -> dict[str, Any] | None:
        return self.request_states.get(request_id)

    def active_request_statuses(self, *, limit: int) -> list[dict[str, Any]]:
        return self.request_states.list_active(limit=limit)

    def landing_glance(self) -> dict[str, Any]:
        retrieval_status = self._snapshot_retrieval_status()
        listener = _selected_runtime_listener()
        db_status = self.log_store.status()
        baseline = self._load_corpus_baseline()
        corpus = baseline.get("corpus", {}) if isinstance(baseline, dict) else {}
        compare_views = corpus.get("compare_views_readiness", {})
        source_layers = corpus.get("source_role_boundary_breakdown", {})
        unknown_or_weak = corpus.get("unknown_or_weak_counts", {})
        admission_summary = corpus.get("admission_summary", {})

        documents = int(
            retrieval_status.get("documents")
            or corpus.get("indexed_document_count")
            or 0
        )
        chunks = int(
            retrieval_status.get("chunks")
            or corpus.get("chunk_count")
            or 0
        )
        runtime_status = (
            "ready"
            if retrieval_status.get("status") in {"ok", "not_started"}
            and db_status.get("available")
            else "degraded"
        )
        retrieval_warmup_status = str(retrieval_status.get("warmup_status") or "not_started")
        retrieval_loaded = bool(retrieval_status.get("assets_loaded", False))

        layer_parts: list[str] = []
        for key, label in (
            ("primary_text", "primary text"),
            ("translation", "translation"),
            ("manual", "fiqh manual"),
            ("commentary", "commentary"),
            ("modern_application", "modern application"),
        ):
            count = int(source_layers.get(key, 0) or 0)
            if count > 0:
                layer_parts.append(f"{count} {label}")
        layer_detail = (
            ", ".join(layer_parts) + " represented in the current admitted corpus."
            if layer_parts
            else "Source-layer composition is not currently available."
        )

        weak_pdf_count = int(unknown_or_weak.get("partial_or_failed_pdf_records", 0) or 0)
        compare_ready = bool(compare_views.get("broad_compare_views_ready", False))
        compare_gaps = list(compare_views.get("corpus_gaps", []))
        admitted_non_hanafi = int(compare_views.get("admitted_non_hanafi_documents", 0) or 0)

        trust_signals = [
            {
                "status": "ok" if retrieval_status.get("status") == "ok" else "warning",
                "title": "Corpus searchable",
                "detail": (
                    f"{documents} admitted documents and {chunks} retrieval chunks are currently loaded."
                    if retrieval_status.get("status") == "ok"
                    else (
                        "Retrieval is ready to warm on the first grounded request."
                        if retrieval_status.get("status") == "not_started"
                        else (
                        "Retrieval index is still warming from the persisted index."
                        if retrieval_status.get("status") == "warming"
                        else "Retrieval is not fully available right now, so the landing state is showing a degraded runtime."
                        )
                    )
                ),
            },
            {
                "status": "ok" if layer_parts else "warning",
                "title": "Source-layer aware",
                "detail": layer_detail,
            },
            {
                "status": "warning" if weak_pdf_count > 0 else "ok",
                "title": "OCR and weak-PDF posture",
                "detail": (
                    f"{weak_pdf_count} partial or failed PDF records remain explicitly flagged instead of being treated as clean text."
                    if weak_pdf_count > 0
                    else "No partial or failed PDF records are currently flagged in the admitted corpus baseline."
                ),
            },
            {
                "status": "ok" if compare_ready else "warning",
                "title": "Compare-views readiness",
                "detail": (
                    "Broader compare-views coverage is supported by admitted non-Hanafi material."
                    if compare_ready
                    else (
                        compare_gaps[0]
                        if compare_gaps
                        else (
                            f"Compare-views remains limited outside the Hanafi slice; admitted non-Hanafi documents: {admitted_non_hanafi}."
                        )
                    )
                ),
            },
        ]

        return {
            "runtime": {
                "status": runtime_status,
                "system_pulse": self._system_pulse(
                    db_status=db_status,
                    retrieval_status=retrieval_status,
                    active_tasks=self.active_tasks(limit=20),
                    active_requests=self.active_request_statuses(limit=20),
                ),
                "retrieval_status": retrieval_status.get("status"),
                "database_status": "ok" if db_status.get("available") else "error",
                "retrieval_loaded": retrieval_loaded,
                "retrieval_warmup_status": retrieval_warmup_status,
                "retrieval_warmup_error": retrieval_status.get("warmup_error"),
                "retrieval_warmup_trigger": retrieval_status.get("warmup_trigger"),
                "retrieval_bootstrap_source": retrieval_status.get("bootstrap_source"),
                "retrieval_fallback_reason": retrieval_status.get(
                    "bootstrap_fallback_reason"
                ),
                "prepared_search_loaded": retrieval_status.get("prepared_search_loaded"),
                "prepared_search_fallback_reason": retrieval_status.get(
                    "prepared_search_fallback_reason"
                ),
                "prepared_search_load_ms": retrieval_status.get("prepared_search_load_ms"),
                "prepared_search_build_ms": retrieval_status.get("prepared_search_build_ms"),
                "warmup_total_ms": retrieval_status.get("warmup_total_ms"),
                "selected_host": listener["host"],
                "selected_port": listener["port"],
                "selected_url": listener["url"],
                "last_reload_at": self.runtime_state.last_reload_at,
            },
            "corpus": {
                "documents": documents,
                "chunks": chunks,
                "admission_summary": admission_summary,
                "source_layers": source_layers,
                "weak_pdf_count": weak_pdf_count,
                "metadata_coverage": corpus.get("metadata_coverage", {}),
            },
            "compare_views": {
                "broad_ready": compare_ready,
                "admitted_non_hanafi_documents": admitted_non_hanafi,
                "gaps": compare_gaps[:3],
            },
            "trust_signals": trust_signals,
            "artifact": {
                "available": bool(baseline),
                "generated_at": baseline.get("generated_at") if isinstance(baseline, dict) else None,
            },
        }

    def _model_profile_status(self, config: RuntimeConfig) -> dict[str, Any]:
        status = build_model_profile_status(
            config,
            self.path_profile,
            ollama_visible_models=[],
            availability_error=None,
        )
        main_runtime = LocalGGUFMainModelRuntime.status(
            config=config,
            path_profile=self.path_profile,
        )
        status["main"].update(
            {
                "available": main_runtime["ready"],
                "backend_available": main_runtime["backend_available"],
                "backend_name": main_runtime["backend_name"],
                "backend_version": main_runtime["backend_version"],
                "loaded": self.runtime_state.last_main_model_loaded
                if self.runtime_state.last_model_backend == "local_gguf"
                else main_runtime["loaded"],
                "fallback_active": main_runtime["fallback_active"],
                "selected_for_production": main_runtime["selected_for_production"],
            }
        )
        return status

    def _runtime_health_payload(
        self,
        *,
        config: RuntimeConfig,
        listener: dict[str, Any] | None = None,
        safe_mode: dict[str, Any] | None = None,
        retrieval_status: dict[str, Any] | None = None,
        include_index_age: bool = True,
        refresh_model_profile: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        listener = listener or _selected_runtime_listener()
        safe_mode = safe_mode or _safe_mode_state()
        if refresh_model_profile:
            self.runtime_state.last_model_profile_status = self._model_profile_status(config)
        retrieval_status = retrieval_status or self._snapshot_retrieval_status(
            include_index_age=include_index_age
        )
        model_status = "ok"
        if self.runtime_state.last_model_error:
            model_status = "degraded"
        laptop_build = bool(config.laptop_build or _env_flag("HJ_LAPTOP_BUILD"))
        main_model_enabled = bool(config.main_model_enabled)
        final_generation_unavailable = not main_model_enabled
        model_disable_reason = "config.main_model_enabled=false" if final_generation_unavailable else None
        model_details = {
            "status": model_status,
            "laptop_build": laptop_build,
            "main_model_enabled": main_model_enabled,
            "main_model_provider": config.main_model_provider,
            "main_model_name": config.main_model_name,
            "main_model_path": config.main_model_path,
            "main_model_target": config.main_model_target,
            "main_model_loaded": self.runtime_state.last_main_model_loaded,
            "final_generation_unavailable": final_generation_unavailable,
            "model_disable_reason": model_disable_reason,
        }
        return retrieval_status, {
            "database": "ok" if self.log_store.status()["available"] else "error",
            "retrieval": retrieval_status["status"],
            "retrieval_warmup_status": retrieval_status.get("warmup_status"),
            "retrieval_warmup_error": retrieval_status.get("warmup_error"),
            "retrieval_bootstrap_source": retrieval_status.get("bootstrap_source"),
            "retrieval_fallback_reason": retrieval_status.get(
                "bootstrap_fallback_reason"
            ),
            "prepared_search_loaded": retrieval_status.get("prepared_search_loaded"),
            "prepared_search_fallback_reason": retrieval_status.get(
                "prepared_search_fallback_reason"
            ),
            "prepared_search_load_ms": retrieval_status.get("prepared_search_load_ms"),
            "prepared_search_build_ms": retrieval_status.get("prepared_search_build_ms"),
            "warmup_total_ms": retrieval_status.get("warmup_total_ms"),
            "model": model_status,
            "model_details": model_details,
            "laptop_build": laptop_build,
            "main_model_enabled": main_model_enabled,
            "final_generation_unavailable": final_generation_unavailable,
            "model_disable_reason": model_disable_reason,
            "safe_mode_active": safe_mode["active"],
            "safe_mode_reason": safe_mode["reason"],
            "usb_test_mode": safe_mode["usb_test_mode"],
            "selected_host": listener["host"],
            "selected_port": listener["port"],
            "selected_url": listener["url"],
            "last_degraded_subsystem": self.runtime_state.last_degraded_subsystem,
            "last_degraded_reason": self.runtime_state.last_degraded_reason,
        }

    def health_snapshot(self) -> dict[str, Any]:
        config = self.config_store.load_effective()
        safe_mode = _safe_mode_state()
        listener = _selected_runtime_listener()
        retrieval_status = self._health_retrieval_status()
        retrieval_status, runtime_health = self._runtime_health_payload(
            config=config,
            listener=listener,
            safe_mode=safe_mode,
            retrieval_status=retrieval_status,
            include_index_age=False,
            refresh_model_profile=False,
        )
        return {
            "runtime_identity": {
                "canonical_entry": config.runtime_identity,
                "compatibility_aliases": config.compatibility_aliases,
                "startup_identity": "Halal Jordan backend",
                "runtime_instance_id": self.runtime_identity.runtime_instance_id,
                "process_id": self.runtime_identity.process_id,
                "host_id": self.runtime_identity.host_id,
                "listen_host": listener["host"],
                "listen_port": listener["port"],
                "listen_url": listener["url"],
            },
            "runtime_health": runtime_health,
            "retrieval": retrieval_status,
        }

    def retrieval_status_snapshot(self) -> dict[str, Any]:
        config = self.config_store.load_effective()
        listener = _selected_runtime_listener()
        retrieval_status = self._health_retrieval_status()
        status = str(retrieval_status.get("status") or "not_started")
        stage = "ready"
        if status == "warming":
            stage = "loading_persisted_index"
        elif status == "error":
            stage = "failed"
        elif status == "not_started":
            stage = "waiting_to_start"
        error = retrieval_status.get("error") or retrieval_status.get("warmup_error")
        return {
            "ready": status == "ok",
            "stage": stage,
            "error": error,
            "status": status,
            "mode": "retrieval_first",
            "laptop_build": bool(config.laptop_build),
            "main_model_enabled": bool(config.main_model_enabled),
            "final_generation_unavailable": bool(
                config.laptop_build or not config.main_model_enabled
            ),
            "prepared_search_loaded": bool(
                retrieval_status.get("prepared_search_loaded")
            ),
            "retrieval_bootstrap_source": retrieval_status.get("bootstrap_source"),
            "retrieval_fallback_reason": retrieval_status.get(
                "bootstrap_fallback_reason"
            ),
            "warmup_status": retrieval_status.get("warmup_status"),
            "warmup_started_at": retrieval_status.get("warmup_started_at"),
            "warmup_finished_at": retrieval_status.get("warmup_finished_at"),
            "documents": int(retrieval_status.get("documents") or 0),
            "chunks": int(retrieval_status.get("chunks") or 0),
            "selected_port": listener["port"],
        }

    def runtime_snapshot(self) -> dict[str, Any]:
        config_description = self.config_store.describe()
        config = self.config_store.load_effective()
        safe_mode = _safe_mode_state()
        listener = _selected_runtime_listener()
        retrieval_status, runtime_health = self._runtime_health_payload(
            config=config,
            listener=listener,
            safe_mode=safe_mode,
            include_index_age=True,
            refresh_model_profile=True,
        )
        db_status = self.log_store.status()
        update_status = self.update_center.status(config=config)
        latest_reload = self.task_store.latest_task(task_type="reload_retrieval_assets")
        latest_reindex = self.task_store.latest_task(task_type="reindex_documents")
        latest_successful_reindex = self.task_store.latest_task(
            task_type="reindex_documents",
            statuses=["succeeded", "degraded"],
        )
        latest_failed_reindex = self.task_store.latest_task(
            task_type="reindex_documents",
            statuses=["failed"],
        )
        latest_degraded_task = self.task_store.latest_task(
            task_type="reindex_documents",
            statuses=["degraded"],
        )
        active_tasks = self.active_tasks(limit=20)
        active_requests = self.active_request_statuses(limit=20)
        recent_events = self.operator_events(limit=10)
        backend_entry = self.repo_root / "app" / "backend" / "main.py"
        admin_entry = self.repo_root / "app" / "backend" / "api.py"
        manager_ui_marker = backend_entry.stat().st_mtime if backend_entry.exists() else None

        snapshot = {
            "runtime_identity": {
                "canonical_entry": config.runtime_identity,
                "compatibility_aliases": config.compatibility_aliases,
                "startup_identity": "Halal Jordan backend",
                "runtime_instance_id": self.runtime_identity.runtime_instance_id,
                "process_id": self.runtime_identity.process_id,
                "host_id": self.runtime_identity.host_id,
                "listen_host": listener["host"],
                "listen_port": listener["port"],
                "listen_url": listener["url"],
            },
            "config": config_description,
            "model": {
                "configured_preference_order": config.model_preference_order,
                "actual_model_used": self.runtime_state.last_model_used,
                "actual_model_backend": self.runtime_state.last_model_backend,
                "requested_model": self.runtime_state.last_model_requested,
                "fallback_used": self.runtime_state.last_model_fallback_used,
                "fallback_from": self.runtime_state.last_model_fallback_from,
                "fallback_to": self.runtime_state.last_model_fallback_to,
                "last_error": self.runtime_state.last_model_error,
                "status": runtime_health["model"],
                "model_role": config.model_role,
                "micro_fast_enabled": config.micro_fast_enabled,
                "micro_fast_provider": config.micro_fast_provider,
                "micro_fast_name": config.micro_fast_name,
                "micro_fast_assist_enabled": config.micro_fast_assist_enabled,
                "micro_fast_assist_confidence_threshold": config.micro_fast_assist_confidence_threshold,
                "micro_model_enabled": config.micro_model_enabled,
                "micro_model_provider": config.micro_model_provider,
                "micro_model_name": config.micro_model_name,
                "micro_model_role": config.micro_model_role,
                "micro_model_path": config.micro_model_path,
                "micro_model_target": config.micro_model_target,
                "micro_smart_enabled": config.micro_smart_enabled,
                "micro_smart_provider": config.micro_smart_provider,
                "micro_smart_name": config.micro_smart_name,
                "micro_smart_role": config.micro_smart_role,
                "micro_smart_path": config.micro_smart_path,
                "micro_smart_target": config.micro_smart_target,
                "main_model_enabled": config.main_model_enabled,
                "main_model_provider": config.main_model_provider,
                "main_model_name": config.main_model_name,
                "main_model_path": config.main_model_path,
                "main_model_target": config.main_model_target,
                "main_model_loaded": self.runtime_state.last_main_model_loaded,
                "profile_status": self.runtime_state.last_model_profile_status,
                "fast_router_log_path": str(self.path_profile.runtime_dir / "micro_llm_fast.jsonl"),
                "shadow_log_path": str(self.path_profile.runtime_dir / "micro_llm_shadow.jsonl"),
                "last_micro_fast": self.runtime_state.last_micro_fast,
                "micro_fast_assist_applied_count": self.runtime_state.micro_fast_assist_applied_count,
                "micro_fast_assist_rejected_count": self.runtime_state.micro_fast_assist_rejected_count,
                "last_micro_fast_assist_decision": self.runtime_state.last_micro_fast_assist_decision,
                "last_micro_shadow": self.runtime_state.last_micro_shadow,
                "last_micro_shadow_warmup": self.runtime_state.last_micro_shadow_warmup,
                "last_micro_shadow_benchmark": self._load_micro_shadow_benchmark(),
                "last_micro_fast_benchmark": self.runtime_state.last_micro_fast_benchmark,
                "last_micro_tier_comparison": self.runtime_state.last_micro_tier_comparison,
                "micro_shadow_tuning": tuning_snapshot(config),
            },
            "retrieval": {
                **retrieval_status,
                "last_reload_at": self.runtime_state.last_reload_at or _row_value(latest_reload, "finished_at"),
                "last_reindex_at": self.runtime_state.last_reindex_at or _row_value(latest_reindex, "finished_at"),
                "normalized_corpus_path": str(self.path_profile.normalized_corpus_dir),
                "index_path": str(self.path_profile.index_dir),
                "mapping_path": str(self.repo_root / "metadata" / "mappings" / "retrieval_policy.json"),
            },
            "tasks": {
                "active_task_count": len(active_tasks),
                "active_tasks": active_tasks,
                "active_runner_threads": self._task_runner.active_thread_names(),
                "last_reload": latest_reload,
                "last_successful_reindex": latest_successful_reindex,
                "last_failed_reindex": latest_failed_reindex,
                "last_degraded_task": latest_degraded_task,
                "current_blocked_condition": self._current_blocked_condition(active_tasks),
                "last_reconciliation_at": self.runtime_state.last_reconciliation_at,
                "last_reconciled_task_count": self.runtime_state.last_reconciled_task_count,
            },
            "requests": {
                "active_request_count": len(active_requests),
                "active_requests": active_requests,
            },
            "events": {
                "recent": recent_events,
            },
            "updates": update_status,
            "database": db_status,
            "permissions": {
                "mode": config.permission_mode,
            },
            "runtime_health": runtime_health,
            "paths": {
                "db_path": str(self.log_store.path),
                "config_path": str(self.config_store.path),
                "raw_corpus_path": str(self.path_profile.raw_corpus_dir),
                "normalized_corpus_path": str(self.path_profile.normalized_corpus_dir),
                "index_path": str(self.path_profile.index_dir),
                "runtime_path": str(self.path_profile.runtime_dir),
                "logs_path": str(self.path_profile.logs_dir),
                "updates_root": str(self.path_profile.updates_root),
                "updates_inbox_path": str(self.path_profile.updates_inbox_dir),
                "updates_processed_path": str(self.path_profile.updates_processed_dir),
                "updates_failed_path": str(self.path_profile.updates_failed_dir),
                "updates_quarantine_path": str(self.path_profile.updates_quarantine_dir),
                "updates_manifests_path": str(self.path_profile.updates_manifests_dir),
                "models_micro_path": str(self.path_profile.micro_models_dir),
                "models_main_path": str(self.path_profile.main_models_dir),
                "mapping_path": str(self.repo_root / "metadata" / "mappings" / "retrieval_policy.json"),
            },
            "version_markers": {
                "backend_main_mtime": _iso_from_timestamp(manager_ui_marker),
                "backend_api_mtime": _iso_from_timestamp(admin_entry.stat().st_mtime) if admin_entry.exists() else None,
                "manager_ui_marker": "embedded_html",
                "admin_ui_marker": "embedded_html",
            },
            "system_pulse": self._system_pulse(
                db_status=db_status,
                retrieval_status=retrieval_status,
                active_tasks=active_tasks,
                active_requests=active_requests,
            ),
        }
        return snapshot

    def _load_corpus_baseline(self) -> dict[str, Any]:
        path = self.repo_root / "docs" / "readiness" / "corpus-baseline.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def recent_logs(self, *, kind: str, limit: int) -> list[dict[str, Any]]:
        return self.log_store.fetch_recent_helper(kind=kind, limit=limit)

    def chat_logs(self, *, limit: int, success: bool | None = None) -> list[dict[str, Any]]:
        return self.log_store.fetch_chat_logs(limit=limit, success=success)

    def audit_logs(self, *, limit: int, action: str | None = None) -> list[dict[str, Any]]:
        return self.log_store.fetch_audit_logs(limit=limit, action=action)

    def error_logs(self, *, limit: int) -> list[dict[str, Any]]:
        return self.log_store.fetch_error_logs(limit=limit)

    def _emit_operator_event(
        self,
        *,
        event_type: str,
        severity: str,
        summary: str,
        related_task_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        route: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        payload = {
            "event_id": f"evt-{uuid.uuid4().hex}",
            "timestamp": _utc_now(),
            "event_type": event_type,
            "severity": severity,
            "related_task_id": related_task_id,
            "request_id": request_id,
            "session_id": session_id,
            "route": route,
            "summary": summary,
            "details": details or {},
        }
        row = self.log_store.write_operator_event(payload)
        if row:
            self.event_broker.publish(row)
        return row

    def _update_request_phase(
        self,
        *,
        actor: RoleContext,
        phase: str,
        summary: str,
        route: str = "/api/chat",
        retrieval_active: bool = False,
        model_active: bool = False,
        degraded_active: bool = False,
        fallback_active: bool | None = None,
        completed: bool = False,
        outcome: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.request_states.update(
            actor.request_id,
            session_id=actor.session_id,
            route=route,
            phase=phase,
            retrieval_active=retrieval_active,
            model_active=model_active,
            degraded_active=degraded_active,
            fallback_active=fallback_active,
            completed=completed,
            outcome=outcome,
            summary=summary,
            details=details,
        )
        self._emit_operator_event(
            event_type="request_phase_changed",
            severity="warning" if degraded_active else "info",
            summary=summary,
            request_id=actor.request_id,
            session_id=actor.session_id,
            route=route,
            details=state,
        )
        return state

    def _system_pulse(
        self,
        *,
        db_status: dict[str, Any],
        retrieval_status: dict[str, Any],
        active_tasks: list[dict[str, Any]],
        active_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not db_status.get("available"):
            return {
                "level": "error",
                "summary": "Error: DB logging unavailable",
            }
        if retrieval_status.get("status") == "warming":
            return {
                "level": "warning",
                "summary": "Warning: retrieval index is still warming",
            }
        if retrieval_status.get("status") == "error":
            return {
                "level": "error",
                "summary": "Error: retrieval subsystem unavailable",
            }
        maintenance_task = next(
            (
                task
                for task in active_tasks
                if task.get("task_type") in {"reindex_documents", "reload_retrieval_assets"}
            ),
            None,
        )
        if maintenance_task:
            return {
                "level": "info",
                "summary": f"Maintenance running: {maintenance_task.get('task_type')} in progress",
            }
        if self.runtime_state.last_model_fallback_used:
            return {
                "level": "warning",
                "summary": "Degraded: model fallback active",
            }
        if active_requests:
            return {
                "level": "info",
                "summary": "System healthy: active requests in progress",
            }
        return {
            "level": "info",
            "summary": "System healthy",
        }

    def record_permission_denial(
        self,
        *,
        actor: RoleContext,
        route: str,
        permission: str,
    ) -> None:
        self._write_audit(
            route=route,
            action="permission_denied",
            actor=actor,
            success=False,
            duration_ms=0,
            status="denied",
            changed={},
            details={"permission": permission},
            error_summary=f"{actor.role} cannot access {permission}",
        )
        self._write_error(
            route=route,
            action="permission_denied",
            actor=actor,
            subsystem="permissions",
            error=PermissionError(f"{actor.role} cannot access {permission}"),
            details={"permission": permission},
        )
        self._emit_operator_event(
            event_type="permission_denied",
            severity="warning",
            summary=f"Permission denied for {permission}",
            request_id=actor.request_id,
            session_id=actor.session_id,
            route=route,
            details={"actor_role": actor.role, "permission": permission},
        )

    def _latest_micro_shadow_result(self) -> dict[str, Any] | None:
        current = self.runtime_state.last_micro_shadow or None
        logged = self._read_last_jsonl_row(
            self.path_profile.runtime_dir / "micro_llm_shadow.jsonl"
        )
        if current and logged:
            current_ts = str(current.get("timestamp") or "")
            logged_ts = str(logged.get("timestamp") or "")
            return current if current_ts >= logged_ts else logged
        return current or logged

    def _latest_micro_fast_result(self) -> dict[str, Any] | None:
        current = self.runtime_state.last_micro_fast or None
        logged = self._read_last_jsonl_row(
            self.path_profile.runtime_dir / "micro_llm_fast.jsonl"
        )
        if current and logged:
            current_ts = str(current.get("timestamp") or "")
            logged_ts = str(logged.get("timestamp") or "")
            return current if current_ts >= logged_ts else logged
        return current or logged

    def _read_last_jsonl_row(self, path: Path) -> dict[str, Any] | None:
        if not path.exists() or not path.is_file():
            return None
        last_line = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _load_micro_shadow_benchmark(self) -> dict[str, Any] | None:
        current = _normalize_micro_benchmark_report(self.runtime_state.last_micro_shadow_benchmark or None)
        path = self.repo_root / "docs" / "readiness" / "micro-llm-benchmark.json"
        if not path.exists() or not path.is_file():
            return current
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return current
        payload = _normalize_micro_benchmark_report(payload)
        if not isinstance(payload, dict):
            return current
        if current:
            current_ts = str(current.get("generated_at") or "")
            payload_ts = str(payload.get("generated_at") or "")
            return current if current_ts >= payload_ts else payload
        return payload

    def _build_micro_tier_benchmark_report(self, config: RuntimeConfig) -> dict[str, Any]:
        prompts = [dict(prompt) for prompt in FAST_BENCHMARK_PROMPTS]
        fast_router = self.micro_fast.benchmark(config=config, prompts=prompts)
        smart_shadow = self.micro_shadow.benchmark(config=config, prompts=prompts)
        comparison = _compare_benchmark_reports(fast_router, smart_shadow)
        return {
            "generated_at": _utc_now(),
            "prompt_count": len(prompts),
            "success": bool(fast_router.get("success")) and bool(smart_shadow.get("success")),
            "success_count": sum(
                [
                    int(fast_router.get("success_count") or 0),
                    int(smart_shadow.get("success_count") or 0),
                ]
            ),
            "failure_count": sum(
                [
                    int(fast_router.get("failure_count") or 0),
                    int(smart_shadow.get("failure_count") or 0),
                ]
            ),
            "output_valid_count": sum(
                [
                    int(fast_router.get("output_valid_count") or 0),
                    int(smart_shadow.get("output_valid_count") or 0),
                ]
            ),
            "average_latency_ms": smart_shadow.get("average_latency_ms"),
            "p95_latency_ms": smart_shadow.get("p95_latency_ms"),
            "fast_router": fast_router,
            "smart_shadow": smart_shadow,
            "comparison": comparison,
        }

    def _run_managed_task(
        self,
        *,
        action: str,
        route: str,
        actor: RoleContext,
        feature_toggle: str,
        operation: Callable[[RuntimeConfig, Callable[..., dict[str, Any]]], OperationOutcome],
        parameters: dict[str, Any],
        background: bool,
        retry_of: str | None,
        retry_reason: str | None,
    ) -> ActionResult:
        config = self.config_store.load_effective()
        feature_state = getattr(config.admin_feature_toggles, feature_toggle)
        if not feature_state:
            raise PermissionError(f"{feature_toggle} is disabled")
        if not self.task_store.available:
            raise RuntimeError("task store is unavailable")

        with self._task_lock:
            conflicts = self.task_store.find_active_conflicts(action)
            if conflicts:
                return self._build_blocked_task_result(
                    action=action,
                    route=route,
                    actor=actor,
                    parameters=parameters,
                    conflicts=conflicts,
                    background=background,
                    retry_of=retry_of,
                    retry_reason=retry_reason,
                )
            task = self.task_store.create_task(
                TaskCreateRequest(
                    task_type=action,
                    requested_by=actor.actor_id,
                    requested_role=actor.role,
                    request_route=route,
                    request_source="admin_api",
                    parameters=parameters,
                    runtime_identity=self.runtime_identity,
                    execution_mode="background" if background else "inline",
                    retry_of=retry_of,
                    retry_reason=retry_reason,
                )
            )
            self._emit_operator_event(
                event_type="task_created",
                severity="info",
                summary=f"Task created: {action}",
                related_task_id=task["task_id"],
                request_id=actor.request_id,
                session_id=actor.session_id,
                route=route,
                details=self._task_brief(task),
            )
        result_holder: dict[str, ActionResult] = {}

        def execute_reserved() -> None:
            result_holder["result"] = self._execute_reserved_task(
                task=task,
                action=action,
                route=route,
                actor=actor,
                config=config,
                operation=operation,
            )

        dispatch_request = TaskDispatchRequest(
            task_id=task["task_id"],
            background=background,
            execute=execute_reserved,
            name=f"hj-{action}-{task['task_id'][:8]}",
        )
        try:
            dispatch_result = self._task_runner.dispatch(dispatch_request)
        except Exception as exc:
            return self._build_dispatch_failure_result(
                task=task,
                action=action,
                route=route,
                actor=actor,
                error=exc,
            )

        if background:
            self._write_audit(
                route=route,
                action=f"{action}_submitted",
                actor=actor,
                success=True,
                duration_ms=0,
                status="accepted",
                changed={"task_id": task["task_id"]},
                details={
                    "task_type": action,
                    "execution_mode": dispatch_result.execution_mode,
                    "thread_name": dispatch_result.thread_name,
                },
            )
            task_row = self.task_store.get_task(task["task_id"]) or task
            return ActionResult(
                action=action,
                success=True,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                request_id=actor.request_id,
                duration_ms=0,
                status="accepted",
                changed={"task_id": task["task_id"]},
                details={
                    "background_submitted": True,
                    "thread_name": dispatch_result.thread_name,
                },
                task_id=task["task_id"],
                task_type=action,
                task_status=task_row.get("status"),
                progress=task_row.get("progress", {}),
                retry_count=int(task_row.get("retry_count", 0)),
                retry_of=task_row.get("retry_of"),
                blocking_reason=task_row.get("blocking_reason"),
                degraded_reason=task_row.get("degraded_reason"),
                terminal_reason=task_row.get("terminal_reason"),
                was_reconciled=bool(task_row.get("was_reconciled", False)),
                execution_mode=dispatch_result.execution_mode,
                runtime_instance_id=task_row.get("runtime_instance_id"),
            )

        return result_holder["result"]

    def _execute_reserved_task(
        self,
        *,
        task: dict[str, Any],
        action: str,
        route: str,
        actor: RoleContext,
        config: RuntimeConfig,
        operation: Callable[[RuntimeConfig, Callable[..., dict[str, Any]]], OperationOutcome],
    ) -> ActionResult:
        started_at = _utc_now()
        task = self.task_store.update_status(
            task_id=task["task_id"],
            status="running",
            actor_id=actor.actor_id,
            actor_role=actor.role,
            details={"request_id": actor.request_id},
            started_at=started_at,
            progress={
                "phase": "starting",
                "current_step": "task_started",
                "active_subsystem": "operations",
            },
            metadata=self._task_runtime_metadata(),
        ) or task
        self._emit_operator_event(
            event_type="task_started",
            severity="info",
            summary=f"Task started: {action}",
            related_task_id=task["task_id"],
            request_id=actor.request_id,
            session_id=actor.session_id,
            route=route,
            details=self._task_brief(task),
        )
        started = perf_counter()

        def report_progress(
            *,
            phase: str,
            current_step: str,
            items_processed: int | None = None,
            total_items: int | None = None,
            percent: int | None = None,
            active_subsystem: str | None = None,
            current_target: str | None = None,
        ) -> dict[str, Any]:
            payload = {
                "phase": phase,
                "current_step": current_step,
                "active_subsystem": active_subsystem or "operations",
            }
            if items_processed is not None:
                payload["items_processed"] = items_processed
            if total_items is not None:
                payload["total_items"] = total_items
            if percent is not None:
                payload["percent"] = percent
            if current_target is not None:
                payload["current_target"] = current_target
            updated = self.task_store.update_progress(
                task_id=task["task_id"],
                actor_id=actor.actor_id,
                actor_role=actor.role,
                progress=payload,
                metadata=self._task_runtime_metadata(),
            )
            self._emit_operator_event(
                event_type="task_progress",
                severity="info",
                summary=f"Task progress: {action} -> {phase}",
                related_task_id=task["task_id"],
                request_id=actor.request_id,
                session_id=actor.session_id,
                route=route,
                details=payload,
            )
            return updated.get("progress", payload) if updated else payload

        try:
            outcome = operation(config, report_progress)
            duration_ms = int((perf_counter() - started) * 1000)
            final_progress = dict(task.get("progress", {}))
            final_progress.update(
                outcome.progress
                or {
                    "phase": "completed",
                    "current_step": "completed",
                    "active_subsystem": "operations",
                }
            )
            finalized = self.task_store.update_status(
                task_id=task["task_id"],
                status=outcome.task_status,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                details=outcome.details,
                finished_at=_utc_now(),
                result_summary=outcome.result_summary or {
                    "live_state_disposition": outcome.details.get("live_state_disposition"),
                },
                duration_ms=duration_ms,
                progress=final_progress,
                metadata={
                    **self._task_runtime_metadata(),
                    "terminal_reason": outcome.terminal_reason or self._default_terminal_reason(outcome.task_status),
                    "degraded_reason": outcome.degraded_reason,
                    "live_state_preserved": outcome.live_state_preserved,
                    "live_state_swapped": outcome.live_state_swapped,
                },
            ) or task
            success = outcome.task_status in {"succeeded", "degraded"}
            result = self._task_action_result_from_record(
                task=finalized,
                action=action,
                actor=actor,
                duration_ms=duration_ms,
                success=success,
                status=outcome.status,
                changed=outcome.changed,
                details=outcome.details,
                error_summary=outcome.error_summary,
            )
            self._write_audit(
                route=route,
                action=action,
                actor=actor,
                success=success,
                duration_ms=duration_ms,
                status=outcome.status,
                changed={**outcome.changed, "task_id": task["task_id"]},
                details={**outcome.details, "task_status": outcome.task_status},
                error_summary=outcome.error_summary,
            )
            self._emit_operator_event(
                event_type="task_succeeded" if outcome.task_status == "succeeded" else "task_degraded",
                severity="info" if outcome.task_status == "succeeded" else "warning",
                summary=f"Task {outcome.task_status}: {action}",
                related_task_id=task["task_id"],
                request_id=actor.request_id,
                session_id=actor.session_id,
                route=route,
                details={**outcome.details, **(outcome.result_summary or {})},
            )
            if action == "reload_retrieval_assets" and outcome.task_status == "succeeded":
                self._emit_operator_event(
                    event_type="retrieval_reloaded",
                    severity="info",
                    summary="Retrieval assets reloaded",
                    related_task_id=task["task_id"],
                    request_id=actor.request_id,
                    session_id=actor.session_id,
                    route=route,
                    details=outcome.result_summary or {},
                )
            if action == "reindex_documents":
                if outcome.live_state_swapped:
                    self._emit_operator_event(
                        event_type="reindex_swap_completed",
                        severity="info" if outcome.task_status == "succeeded" else "warning",
                        summary="Reindex completed and live assets swapped",
                        related_task_id=task["task_id"],
                        request_id=actor.request_id,
                        session_id=actor.session_id,
                        route=route,
                        details=outcome.result_summary or {},
                    )
                elif outcome.live_state_preserved:
                    self._emit_operator_event(
                        event_type="reindex_swap_preserved",
                        severity="warning",
                        summary="Reindex preserved previous live assets",
                        related_task_id=task["task_id"],
                        request_id=actor.request_id,
                        session_id=actor.session_id,
                        route=route,
                        details=outcome.result_summary or {},
                    )
            return result
        except Exception as exc:
            duration_ms = int((perf_counter() - started) * 1000)
            error_details = (
                exc.details if isinstance(exc, ManagedOperationError) else {}
            )
            result_summary = (
                exc.result_summary if isinstance(exc, ManagedOperationError) else {}
            )
            progress = (
                exc.progress
                if isinstance(exc, ManagedOperationError)
                else {
                    "phase": "failed",
                    "current_step": "operation_failed",
                    "active_subsystem": "operations",
                }
            )
            finalized = self.task_store.update_status(
                task_id=task["task_id"],
                status="failed",
                actor_id=actor.actor_id,
                actor_role=actor.role,
                details=error_details,
                finished_at=_utc_now(),
                result_summary=result_summary,
                error_summary=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
                progress=progress,
                metadata={
                    **self._task_runtime_metadata(),
                    "terminal_reason": "operation_failed",
                    "interruption_reason": None,
                    "live_state_preserved": error_details.get("live_state_disposition") == "preserved",
                    "live_state_swapped": False if error_details.get("live_state_disposition") else None,
                },
            ) or task
            result = self._task_action_result_from_record(
                task=finalized,
                action=action,
                actor=actor,
                duration_ms=duration_ms,
                success=False,
                status="error",
                changed={},
                details=error_details,
                error_summary=f"{type(exc).__name__}: {exc}",
            )
            self._write_audit(
                route=route,
                action=action,
                actor=actor,
                success=False,
                duration_ms=duration_ms,
                status="error",
                changed={"task_id": task["task_id"]},
                details={**error_details, "task_status": "failed"},
                error_summary=result.error_summary,
            )
            self._write_error(
                route=route,
                action=action,
                actor=actor,
                subsystem="operations",
                error=exc,
                details={"action": action, "task_id": task["task_id"], **error_details},
            )
            self._emit_operator_event(
                event_type="task_failed",
                severity="error",
                summary=f"Task failed: {action}",
                related_task_id=task["task_id"],
                request_id=actor.request_id,
                session_id=actor.session_id,
                route=route,
                details={"error_summary": result.error_summary, **error_details},
            )
            if action == "reindex_documents" and error_details.get("live_state_disposition") == "preserved":
                self._emit_operator_event(
                    event_type="reindex_swap_preserved",
                    severity="warning",
                    summary="Reindex failed and previous live assets were preserved",
                    related_task_id=task["task_id"],
                    request_id=actor.request_id,
                    session_id=actor.session_id,
                    route=route,
                    details=error_details,
                )
            return result

    def _build_dispatch_failure_result(
        self,
        *,
        task: dict[str, Any],
        action: str,
        route: str,
        actor: RoleContext,
        error: Exception,
    ) -> ActionResult:
        finalized = self.task_store.update_status(
            task_id=task["task_id"],
            status="failed",
            actor_id=actor.actor_id,
            actor_role=actor.role,
            details={"dispatch_failed": True},
            finished_at=_utc_now(),
            result_summary={"live_state_disposition": "preserved", "swap_applied": False},
            error_summary=f"{type(error).__name__}: {error}",
            duration_ms=0,
            progress={
                "phase": "failed",
                "current_step": "dispatch_failed",
                "active_subsystem": "operations",
            },
            metadata={
                **self._task_runtime_metadata(),
                "terminal_reason": "dispatch_failed",
                "live_state_preserved": True,
                "live_state_swapped": False,
            },
        ) or task
        result = self._task_action_result_from_record(
            task=finalized,
            action=action,
            actor=actor,
            duration_ms=0,
            success=False,
            status="error",
            changed={},
            details={"dispatch_failed": True},
            error_summary=f"{type(error).__name__}: {error}",
        )
        self._write_audit(
            route=route,
            action=action,
            actor=actor,
            success=False,
            duration_ms=0,
            status="error",
            changed={"task_id": task["task_id"]},
            details={"dispatch_failed": True, "task_status": "failed"},
            error_summary=result.error_summary,
        )
        return result

    def _reconcile_stale_tasks(self) -> list[dict[str, Any]]:
        if not self.task_store.available:
            return []
        stale_tasks = self.task_store.stale_incomplete_tasks()
        reconciled: list[dict[str, Any]] = []
        actor_id = f"startup-reconciler:{self.runtime_identity.runtime_instance_id}"
        for task in stale_tasks:
            reconciled_task = self._reconcile_task_record(
                task=task,
                actor_id=actor_id,
                actor_role="owner",
            )
            if reconciled_task is None:
                continue
            reconciled.append(reconciled_task)
            self._write_audit(
                route="startup",
                action="task_reconciled",
                actor=RoleContext(
                    actor_id=actor_id,
                    role="owner",
                    request_id="startup-reconciliation",
                    session_id="startup-reconciliation",
                ),
                success=True,
                duration_ms=0,
                status=reconciled_task.get("status", "reconciled"),
                changed={"task_id": reconciled_task.get("task_id")},
                details={
                    "task_type": reconciled_task.get("task_type"),
                    "terminal_reason": reconciled_task.get("terminal_reason"),
                    "interruption_reason": reconciled_task.get("interruption_reason"),
                },
            )
            self._emit_operator_event(
                event_type="task_reconciled",
                severity="warning",
                summary=f"Task reconciled on startup: {reconciled_task.get('task_type')}",
                related_task_id=reconciled_task.get("task_id"),
                route="startup",
                details={
                    "status": reconciled_task.get("status"),
                    "terminal_reason": reconciled_task.get("terminal_reason"),
                    "interruption_reason": reconciled_task.get("interruption_reason"),
                },
            )
        self.runtime_state.last_reconciliation_at = _utc_now()
        self.runtime_state.last_reconciled_task_count = len(reconciled)
        return reconciled

    def _reconcile_task_record(
        self,
        *,
        task: dict[str, Any],
        actor_id: str,
        actor_role: str,
    ) -> dict[str, Any] | None:
        current_status = task.get("status")
        progress = dict(task.get("progress", {}))
        phase = progress.get("phase", "")
        if current_status == "queued":
            progress.update(
                {
                    "phase": "reconciled",
                    "current_step": "interrupted_before_execution",
                    "active_subsystem": "operations",
                }
            )
            return self.task_store.reconcile_task(
                task_id=task["task_id"],
                status="canceled",
                actor_id=actor_id,
                actor_role=actor_role,
                reason="startup_reconciliation_interrupted_before_start",
                details={
                    "previous_status": current_status,
                    "live_state_disposition": "preserved",
                    "no_asset_swap_confirmed": True,
                },
                progress=progress,
                metadata={
                    "live_state_preserved": True,
                    "live_state_swapped": False,
                },
            )

        if task.get("task_type") == "reload_retrieval_assets":
            progress.update(
                {
                    "phase": "reconciled",
                    "current_step": "reload_interrupted_during_process_shutdown",
                    "active_subsystem": "retrieval",
                }
            )
            return self.task_store.reconcile_task(
                task_id=task["task_id"],
                status="degraded",
                actor_id=actor_id,
                actor_role=actor_role,
                reason="startup_reconciliation_reload_interrupted",
                details={
                    "previous_status": current_status,
                    "live_state_disposition": "preserved_assumed",
                    "swap_applied": False,
                },
                progress=progress,
                metadata={
                    "degraded_reason": "reload_interrupted_previous_live_state_assumed_preserved",
                    "live_state_preserved": True,
                    "live_state_swapped": False,
                },
            )

        if task.get("task_type") == "reindex_documents":
            if phase == "swapping_live_assets":
                progress.update(
                    {
                        "phase": "reconciled",
                        "current_step": "reindex_interrupted_during_swap_boundary",
                        "active_subsystem": "retrieval",
                    }
                )
                return self.task_store.reconcile_task(
                    task_id=task["task_id"],
                    status="degraded",
                    actor_id=actor_id,
                    actor_role=actor_role,
                    reason="startup_reconciliation_reindex_interrupted_during_swap",
                    details={
                        "previous_status": current_status,
                        "live_state_disposition": "unknown_after_interruption",
                        "swap_state_uncertain": True,
                    },
                    progress=progress,
                    metadata={
                        "degraded_reason": "reindex_interrupted_during_swap_state_unknown",
                        "live_state_preserved": None,
                        "live_state_swapped": None,
                    },
                )
            progress.update(
                {
                    "phase": "reconciled",
                    "current_step": "reindex_interrupted_before_swap",
                    "active_subsystem": "retrieval",
                }
            )
            return self.task_store.reconcile_task(
                task_id=task["task_id"],
                status="failed",
                actor_id=actor_id,
                actor_role=actor_role,
                reason="startup_reconciliation_reindex_interrupted_before_swap",
                details={
                    "previous_status": current_status,
                    "live_state_disposition": "preserved_assumed",
                    "swap_applied": False,
                },
                progress=progress,
                metadata={
                    "live_state_preserved": True,
                    "live_state_swapped": False,
                },
            )

        progress.update(
            {
                "phase": "reconciled",
                "current_step": "task_interrupted_by_process_shutdown",
                "active_subsystem": "operations",
            }
        )
        return self.task_store.reconcile_task(
            task_id=task["task_id"],
            status="failed",
            actor_id=actor_id,
            actor_role=actor_role,
            reason="startup_reconciliation_interrupted_running_task",
            details={
                "previous_status": current_status,
                "live_state_disposition": "unknown",
            },
            progress=progress,
        )

    def _build_blocked_task_result(
        self,
        *,
        action: str,
        route: str,
        actor: RoleContext,
        parameters: dict[str, Any],
        conflicts: list[dict[str, Any]],
        background: bool,
        retry_of: str | None,
        retry_reason: str | None,
    ) -> ActionResult:
        task = self.task_store.create_task(
            TaskCreateRequest(
                task_type=action,
                requested_by=actor.actor_id,
                requested_role=actor.role,
                request_route=route,
                request_source="admin_api",
                parameters=parameters,
                runtime_identity=self.runtime_identity,
                execution_mode="background" if background else "inline",
                retry_of=retry_of,
                retry_reason=retry_reason,
            )
        )
        blocking_reason = self._format_blocking_reason(action, conflicts)
        finished_at = _utc_now()
        finalized = self.task_store.update_status(
            task_id=task["task_id"],
            status="blocked",
            actor_id=actor.actor_id,
            actor_role=actor.role,
            details={
                "blocked_by": [self._task_brief(conflict) for conflict in conflicts],
                "reason": blocking_reason,
            },
            finished_at=finished_at,
            result_summary={
                "live_state_disposition": "preserved",
                "blocked_by": [self._task_brief(conflict) for conflict in conflicts],
            },
            duration_ms=0,
            blocking_reason=blocking_reason,
            progress={
                "phase": "blocked",
                "current_step": "blocked_by_active_task",
                "active_subsystem": "operations",
            },
            metadata={
                **self._task_runtime_metadata(),
                "terminal_reason": "blocked_by_active_task",
                "live_state_preserved": True,
                "live_state_swapped": False,
            },
        ) or task
        result = self._task_action_result_from_record(
            task=finalized,
            action=action,
            actor=actor,
            duration_ms=0,
            success=False,
            status="blocked",
            changed={},
            details={"blocked_by": [self._task_brief(conflict) for conflict in conflicts]},
            error_summary=blocking_reason,
        )
        self._write_audit(
            route=route,
            action=action,
            actor=actor,
            success=False,
            duration_ms=0,
            status="blocked",
            changed={"task_id": task["task_id"]},
            details={"blocked_by": [self._task_brief(conflict) for conflict in conflicts]},
            error_summary=blocking_reason,
        )
        self._emit_operator_event(
            event_type="task_blocked",
            severity="warning",
            summary=f"Task blocked: {action}",
            related_task_id=task["task_id"],
            request_id=actor.request_id,
            session_id=actor.session_id,
            route=route,
            details={"blocking_reason": blocking_reason},
        )
        return result

    def _task_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime_instance_id": self.runtime_identity.runtime_instance_id,
            "started_by_runtime": self.runtime_identity.runtime_instance_id,
            "process_id": self.runtime_identity.process_id,
            "host_id": self.runtime_identity.host_id,
        }

    def _task_action_result_from_record(
        self,
        *,
        task: dict[str, Any],
        action: str,
        actor: RoleContext,
        duration_ms: int,
        success: bool,
        status: str,
        changed: dict[str, Any],
        details: dict[str, Any],
        error_summary: str | None,
    ) -> ActionResult:
        return ActionResult(
            action=action,
            success=success,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            request_id=actor.request_id,
            duration_ms=duration_ms,
            status=status,
            changed=changed,
            details=details,
            task_id=task.get("task_id"),
            task_type=task.get("task_type"),
            task_status=task.get("status"),
            progress=task.get("progress", {}),
            retry_count=int(task.get("retry_count", 0)),
            retry_of=task.get("retry_of"),
            blocking_reason=task.get("blocking_reason"),
            degraded_reason=task.get("degraded_reason"),
            terminal_reason=task.get("terminal_reason"),
            was_reconciled=bool(task.get("was_reconciled", False)),
            execution_mode=str(task.get("execution_mode") or "inline"),
            runtime_instance_id=task.get("runtime_instance_id"),
            error_summary=error_summary,
        )

    def _default_terminal_reason(self, task_status: str) -> str:
        return {
            "succeeded": "operation_completed",
            "degraded": "operation_completed_with_degraded_outcome",
            "failed": "operation_failed",
            "blocked": "blocked_by_active_task",
            "canceled": "operation_canceled",
        }.get(task_status, "operation_finished")

    def _reload_operation(
        self,
        _: RuntimeConfig,
        report_progress: Callable[..., dict[str, Any]],
    ) -> OperationOutcome:
        previous_state = self.retrieval._state
        report_progress(
            phase="reading_db",
            current_step="reading_normalized_assets",
            active_subsystem="retrieval",
        )
        report_progress(
            phase="reading_mapping",
            current_step="loading_retrieval_policy",
            active_subsystem="retrieval",
        )
        report_progress(
            phase="reading_index",
            current_step="rebuilding_in_memory_state",
            active_subsystem="retrieval",
        )
        try:
            state = self.retrieval.bootstrap(force=True)
        except Exception as exc:
            self.retrieval._state = previous_state
            raise ManagedOperationError(
                str(exc),
                details={
                    "live_state_disposition": "preserved",
                    "previous_assets_retained": previous_state is not None,
                },
                result_summary={
                    "live_state_disposition": "preserved",
                    "previous_assets_retained": previous_state is not None,
                    "swap_applied": False,
                },
                progress={
                    "phase": "reading_index",
                    "current_step": "reload_failed_before_swap",
                    "active_subsystem": "retrieval",
                },
            ) from exc
        report_progress(
            phase="warming_retrieval",
            current_step="verifying_retrieval_stats",
            active_subsystem="retrieval",
        )
        stats = {
            "documents": len(state.corpus.documents),
            "chunks": len(state.chunks),
            "skipped_files": len(state.corpus.skipped_files),
        }
        self.runtime_state.retrieval_loaded = True
        self.runtime_state.last_reload_at = _utc_now()
        self.runtime_state.last_retrieval_stats = stats
        return OperationOutcome(
            changed=stats,
            details={
                "retrieval_loaded": True,
                **stats,
                "live_state_disposition": "swapped_live_assets",
            },
            status="ok",
            task_status="succeeded",
            result_summary={
                "live_state_disposition": "swapped_live_assets",
                "swap_applied": True,
                **stats,
            },
            terminal_reason="reload_completed",
            live_state_preserved=False,
            live_state_swapped=True,
            progress={
                "phase": "completed",
                "current_step": "reload_completed",
                "active_subsystem": "retrieval",
            },
        )

    def _reindex_operation(
        self,
        config: RuntimeConfig,
        report_progress: Callable[..., dict[str, Any]],
    ) -> OperationOutcome:
        previous_state = self.retrieval._state
        live_index_root = self.path_profile.index_dir
        staging_index_root = live_index_root.parent / f"{live_index_root.name}.staging-{uuid.uuid4().hex}"
        backup_index_root = live_index_root.parent / f"{live_index_root.name}.backup-{uuid.uuid4().hex}"
        had_existing_index = _path_has_files(live_index_root)
        swapped_index = False
        old_index_preserved = False
        report_progress(
            phase="scanning",
            current_step="scanning_raw_pdfs",
            active_subsystem="ingestion",
            current_target=str(self.repo_root / "data" / "raw"),
        )
        report_progress(
            phase="extracting",
            current_step="extracting_machine_readable_pdfs",
            active_subsystem="ingestion",
        )
        report_progress(
            phase="normalizing",
            current_step="writing_normalized_outputs",
            active_subsystem="ingestion",
        )
        records = process_pdf_corpus(
            self.repo_root,
            force=config.reindex_behavior_flags.force_pdf_normalization,
            allow_ocr=config.reindex_behavior_flags.enable_ocr_recovery,
        )
        success_count = sum(1 for record in records if record.extraction_status == "success")
        partial_count = sum(1 for record in records if record.extraction_status == "partial")
        failed_count = sum(1 for record in records if record.extraction_status == "failed")
        report_progress(
            phase="writing_db",
            current_step="updating_normalized_corpus_state",
            items_processed=len(records),
            total_items=len(records),
            active_subsystem="ingestion",
        )
        report_progress(
            phase="rebuilding_index",
            current_step="building_persisted_retrieval_index",
            active_subsystem="retrieval",
        )
        try:
            _remove_tree_quietly(staging_index_root)
            _remove_tree_quietly(backup_index_root)
            persisted_index = build_persisted_retrieval_index(
                self.repo_root,
                raw_root=self.path_profile.raw_corpus_dir,
                normalized_root=self.path_profile.normalized_corpus_dir,
                index_root=staging_index_root,
                manifest_index_root=live_index_root,
            )
            if live_index_root.exists():
                live_index_root.replace(backup_index_root)
                old_index_preserved = had_existing_index
            staging_index_root.replace(live_index_root)
            swapped_index = True
            report_progress(
                phase="rebuilding_index",
                current_step="reloading_retrieval_pipeline",
                active_subsystem="retrieval",
                items_processed=int(persisted_index.get("chunk_count") or 0),
                total_items=int(persisted_index.get("chunk_count") or 0),
            )
            state = self.retrieval.bootstrap(force=True)
        except Exception as exc:
            _remove_tree_quietly(staging_index_root)
            if swapped_index:
                _remove_tree_quietly(live_index_root)
                if backup_index_root.exists():
                    backup_index_root.replace(live_index_root)
                    old_index_preserved = True
            elif not live_index_root.exists() and backup_index_root.exists():
                backup_index_root.replace(live_index_root)
                old_index_preserved = True
            if config.reindex_behavior_flags.preserve_previous_assets_on_failure:
                self.retrieval._state = previous_state
            raise ManagedOperationError(
                str(exc),
                details={
                    "live_state_disposition": (
                        "preserved"
                        if config.reindex_behavior_flags.preserve_previous_assets_on_failure
                        else "no_swap"
                    ),
                    "preserved_previous_assets_on_failure": (
                        config.reindex_behavior_flags.preserve_previous_assets_on_failure
                    ),
                    "index_rebuilt": False,
                    "old_index_preserved": old_index_preserved,
                    "index_root": str(self.path_profile.index_dir),
                },
                result_summary={
                    "live_state_disposition": (
                        "preserved"
                        if config.reindex_behavior_flags.preserve_previous_assets_on_failure
                        else "no_swap"
                    ),
                    "swap_applied": False,
                    "records_processed": len(records),
                    "pdf_success": success_count,
                    "pdf_partial": partial_count,
                    "pdf_failed": failed_count,
                    "index_rebuilt": False,
                    "old_index_preserved": old_index_preserved,
                    "index_root": str(self.path_profile.index_dir),
                },
                progress={
                    "phase": "rebuilding_index",
                    "current_step": "reindex_failed_before_swap",
                    "active_subsystem": "retrieval",
                },
            ) from exc
        finally:
            _remove_tree_quietly(staging_index_root)
            if backup_index_root.exists() and swapped_index:
                _remove_tree_quietly(backup_index_root)
        report_progress(
            phase="swapping_live_assets",
            current_step="publishing_reindexed_retrieval_state",
            active_subsystem="retrieval",
        )
        stats = {
            "documents": len(state.corpus.documents),
            "chunks": len(state.chunks),
            "skipped_files": len(state.corpus.skipped_files),
        }
        self.runtime_state.retrieval_loaded = True
        self.runtime_state.last_retrieval_stats = stats
        self.runtime_state.last_reindex_at = _utc_now()
        task_status = "succeeded" if failed_count == 0 else "degraded"
        status = "ok" if failed_count == 0 else "partial"
        return OperationOutcome(
            changed={
                "pdf_success": success_count,
                "pdf_partial": partial_count,
                "pdf_failed": failed_count,
                "index_rebuilt": True,
                "old_index_preserved": old_index_preserved,
                "index_root": str(self.path_profile.index_dir),
                **stats,
            },
            details={
                "records_processed": len(records),
                "preserved_previous_assets_on_failure": (
                    config.reindex_behavior_flags.preserve_previous_assets_on_failure
                ),
                "stats": stats,
                "index_root": str(self.path_profile.index_dir),
                "persisted_index": inspect_persisted_retrieval_index(
                    self.path_profile.index_dir,
                    repo_root=self.repo_root,
                    raw_root=self.path_profile.raw_corpus_dir,
                    normalized_root=self.path_profile.normalized_corpus_dir,
                ),
                "index_rebuilt": True,
                "old_index_preserved": old_index_preserved,
                "live_state_disposition": "swapped_live_assets",
            },
            status=status,
            task_status=task_status,
            result_summary={
                "live_state_disposition": "swapped_live_assets",
                "swap_applied": True,
                "records_processed": len(records),
                "pdf_success": success_count,
                "pdf_partial": partial_count,
                "pdf_failed": failed_count,
                "index_rebuilt": True,
                "old_index_preserved": old_index_preserved,
                "index_root": str(self.path_profile.index_dir),
                **stats,
            },
            degraded_reason=(
                "reindex_completed_with_failed_pdf_records"
                if failed_count > 0
                else None
            ),
            terminal_reason=(
                "reindex_completed_with_degraded_outcome"
                if failed_count > 0
                else "reindex_completed"
            ),
            live_state_preserved=False,
            live_state_swapped=True,
            progress={
                "phase": "completed",
                "current_step": "reindex_completed",
                "active_subsystem": "retrieval",
                "items_processed": len(records),
                "total_items": len(records),
            },
        )

    def _current_blocked_condition(self, active_tasks: list[dict[str, Any]]) -> str | None:
        if not active_tasks:
            return None
        active = active_tasks[0]
        return (
            f"{active.get('task_type')} is currently {active.get('status')}, "
            "so conflicting maintenance actions are blocked."
        )

    def _format_blocking_reason(
        self,
        action: str,
        conflicts: list[dict[str, Any]],
    ) -> str:
        if not conflicts:
            return f"{action} is blocked by active maintenance work"
        primary = conflicts[0]
        return (
            f"{action} is blocked while {primary.get('task_type')} "
            f"is {primary.get('status')} ({primary.get('task_id')})."
        )

    def _task_brief(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "status": task.get("status"),
            "requested_by": task.get("requested_by"),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "progress": task.get("progress", {}),
        }

    def _snapshot_retrieval_status(
        self,
        *,
        include_index_age: bool = True,
    ) -> dict[str, Any]:
        index_assets = inspect_persisted_retrieval_index(
            self.path_profile.index_dir,
            repo_root=self.repo_root,
            raw_root=self.path_profile.raw_corpus_dir,
            normalized_root=self.path_profile.normalized_corpus_dir,
        )
        stats = self._retrieval_stats_snapshot(index_assets=index_assets)
        warmup_status = self.runtime_state.retrieval_warmup_status
        status = "ok"
        error = None
        if warmup_status == "warming":
            status = "warming"
        elif warmup_status == "failed":
            status = "error"
            error = self.runtime_state.retrieval_warmup_error
            self.runtime_state.last_degraded_subsystem = "retrieval"
            self.runtime_state.last_degraded_reason = error
        elif not self.runtime_state.retrieval_loaded and warmup_status == "not_started":
            status = "not_started"
        elif not self.runtime_state.retrieval_loaded and not index_assets.get(
            "retrieval_index_loadable"
        ):
            status = "error"
            error = _retrieval_failure_label(
                str(index_assets.get("retrieval_fallback_reason") or "index_missing")
            )
        elif not self.runtime_state.retrieval_loaded:
            status = "warming"
        return {
            "status": status,
            "assets_loaded": self.retrieval._state is not None,
            **stats,
            "index_age": self._index_age_summary() if include_index_age else None,
            "bootstrap_source": self.retrieval._last_bootstrap_source,
            "bootstrap_fallback_reason": self.retrieval._last_bootstrap_reason,
            "prepared_search_loaded": self.retrieval._prepared_search_loaded,
            "prepared_search_fallback_reason": self.retrieval._prepared_search_fallback_reason,
            "prepared_search_load_ms": self.retrieval._prepared_search_load_ms,
            "prepared_search_build_ms": self.retrieval._prepared_search_build_ms,
            "warmup_total_ms": self.retrieval._last_bootstrap_total_ms,
            "warmup_status": warmup_status,
            "warmup_started_at": self.runtime_state.retrieval_warmup_started_at,
            "warmup_finished_at": self.runtime_state.retrieval_warmup_finished_at,
            "warmup_trigger": self.runtime_state.retrieval_warmup_trigger,
            "warmup_error": self.runtime_state.retrieval_warmup_error,
            "warmup_bootstrap_source": self.runtime_state.retrieval_warmup_bootstrap_source,
            "warmup_fallback_reason": self.runtime_state.retrieval_warmup_fallback_reason,
            "index_assets": index_assets,
            "error": error,
        }

    def _health_retrieval_status(self) -> dict[str, Any]:
        state = self.retrieval._state
        if state is not None:
            stats = {
                "documents": len(state.corpus.documents),
                "chunks": len(state.chunks),
                "skipped_files": len(state.corpus.skipped_files),
            }
        else:
            stats = {
                "documents": int(self.runtime_state.last_retrieval_stats.get("documents", 0)),
                "chunks": int(self.runtime_state.last_retrieval_stats.get("chunks", 0)),
                "skipped_files": int(
                    self.runtime_state.last_retrieval_stats.get("skipped_files", 0)
                ),
            }
        warmup_status = self.runtime_state.retrieval_warmup_status
        status = "ok"
        error = None
        if warmup_status == "warming":
            status = "warming"
        elif warmup_status == "failed":
            status = "error"
            error = self.runtime_state.retrieval_warmup_error
        elif not self.runtime_state.retrieval_loaded and warmup_status == "not_started":
            status = "not_started"
        elif not self.runtime_state.retrieval_loaded:
            status = "warming"
        return {
            "status": status,
            "assets_loaded": state is not None,
            **stats,
            "index_age": None,
            "bootstrap_source": (
                self.runtime_state.retrieval_warmup_bootstrap_source
                or self.retrieval._last_bootstrap_source
            ),
            "bootstrap_fallback_reason": (
                self.runtime_state.retrieval_warmup_fallback_reason
                or self.retrieval._last_bootstrap_reason
            ),
            "prepared_search_loaded": self.retrieval._prepared_search_loaded,
            "prepared_search_fallback_reason": self.retrieval._prepared_search_fallback_reason,
            "prepared_search_load_ms": self.retrieval._prepared_search_load_ms,
            "prepared_search_build_ms": self.retrieval._prepared_search_build_ms,
            "warmup_total_ms": self.retrieval._last_bootstrap_total_ms,
            "warmup_status": warmup_status,
            "warmup_started_at": self.runtime_state.retrieval_warmup_started_at,
            "warmup_finished_at": self.runtime_state.retrieval_warmup_finished_at,
            "warmup_trigger": self.runtime_state.retrieval_warmup_trigger,
            "warmup_error": self.runtime_state.retrieval_warmup_error,
            "warmup_bootstrap_source": self.runtime_state.retrieval_warmup_bootstrap_source,
            "warmup_fallback_reason": self.runtime_state.retrieval_warmup_fallback_reason,
            "index_assets": None,
            "error": error,
        }

    def _index_age_summary(self) -> dict[str, Any] | None:
        index_root = self.path_profile.index_dir
        if not index_root.exists():
            return None
        newest = None
        for path in index_root.rglob("*"):
            if not path.is_file():
                continue
            modified = path.stat().st_mtime
            if newest is None or modified > newest:
                newest = modified
        if newest is None:
            return None
        persisted_index = inspect_persisted_retrieval_index(index_root)
        return {
            "latest_asset_mtime": _iso_from_timestamp(newest),
            "seconds_since_latest_asset": max(0, int(datetime.now(UTC).timestamp() - newest)),
            "generated_at": persisted_index.get("generated_at"),
            "chunk_count": persisted_index.get("chunk_count", 0),
            "document_count": persisted_index.get("document_count", 0),
            "bootstrap_source": self.retrieval._last_bootstrap_source,
            "bootstrap_fallback_reason": self.retrieval._last_bootstrap_reason,
        }

    def _build_fast_local_response(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
        greeting_style: str | None,
        tone_level: str | None,
        sources: list[dict[str, Any]],
        query_intent: QueryIntent,
    ) -> dict[str, Any]:
        grounder = self._grounder
        grounder.quote_window_chars = self.config_store.load_effective().citation_excerpt_length
        enriched_sources = grounder.enrich_sources(
            question=question,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            retrieved_sources=sources,
            query_intent=query_intent,
        )
        answer = normalize_and_validate_answer(
            candidate={
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "direct_answer": _fast_local_direct_answer(
                    answer_mode=answer_mode,
                    query_intent=query_intent,
                    sources=sources,
                ),
                "evidence_strength": _fast_local_evidence_strength(sources),
                "uncertainty_note": _fast_local_uncertainty_note(
                    answer_mode=answer_mode,
                    query_intent=query_intent,
                    sources=sources,
                ),
                "ui_state": {
                    "greeting_style": greeting_style or self.artifacts.default_greeting_style,
                    "tone_level": tone_level or self.artifacts.default_tone_level,
                    "selected_madhhab_visible": True,
                    "answer_mode_visible": True,
                    "confidence_visible": True,
                    "disagreement_visible": True,
                },
            },
            schema=self.artifacts.answer_response_schema,
            request_context={
                "question": question,
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "greeting_style": greeting_style or self.artifacts.default_greeting_style,
                "tone_level": tone_level or self.artifacts.default_tone_level,
                "repo_root": str(self.repo_root),
            },
            retrieved_sources=enriched_sources,
        )
        evidence_model = grounder.build_evidence_model(
            answer=answer,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            enriched_sources=enriched_sources,
            query_intent=query_intent,
        )
        return {
            "answer": answer,
            "rendered_text": render_answer(answer, evidence_model),
            "answer_diagnostics": {
                "generation_path": "fast",
                "evidence_backfill_applied": evidence_model.evidence_backfill_applied,
                "evidence_backfill_buckets": list(evidence_model.evidence_backfill_buckets),
                "source_layer_composition": dict(evidence_model.source_layer_composition),
                "metadata_completeness": dict(evidence_model.metadata_completeness),
                "ocr_usage": dict(evidence_model.ocr_usage),
                "teaching_layer_used": evidence_model.teaching_layer_used,
                "teaching_sources_count": evidence_model.teaching_sources_count,
                "teaching_layer_reason": evidence_model.teaching_layer_reason,
                "teaching_layer_excluded_reason": evidence_model.teaching_layer_excluded_reason,
                **study_path_diagnostics(answer),
            },
        }

    def _build_location_tool_response(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
        greeting_style: str | None,
        tone_level: str | None,
        sources: list[dict[str, Any]],
        query_intent: QueryIntent,
        direct_answer: str,
        uncertainty_note: str | None,
        route_used: str | None,
        tool_used: str | None,
    ) -> dict[str, Any]:
        grounder = self._grounder
        grounder.quote_window_chars = self.config_store.load_effective().citation_excerpt_length
        enriched_sources = grounder.enrich_sources(
            question=question,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            retrieved_sources=sources,
            query_intent=query_intent,
        )
        answer = normalize_and_validate_answer(
            candidate={
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "direct_answer": direct_answer,
                "uncertainty_note": uncertainty_note,
                "evidence_strength": _fast_local_evidence_strength(sources),
                "ui_state": {
                    "greeting_style": greeting_style or self.artifacts.default_greeting_style,
                    "tone_level": tone_level or self.artifacts.default_tone_level,
                    "selected_madhhab_visible": True,
                    "answer_mode_visible": True,
                    "confidence_visible": True,
                    "disagreement_visible": True,
                },
            },
            schema=self.artifacts.answer_response_schema,
            request_context={
                "question": question,
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "greeting_style": greeting_style or self.artifacts.default_greeting_style,
                "tone_level": tone_level or self.artifacts.default_tone_level,
                "repo_root": str(self.repo_root),
            },
            retrieved_sources=enriched_sources,
        )
        evidence_model = grounder.build_evidence_model(
            answer=answer,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            enriched_sources=enriched_sources,
            query_intent=query_intent,
        )
        return {
            "answer": answer,
            "rendered_text": render_answer(answer, evidence_model),
            "answer_diagnostics": {
                "generation_path": "tool",
                "route_used": route_used,
                "tool_used": tool_used,
                "evidence_backfill_applied": evidence_model.evidence_backfill_applied,
                "evidence_backfill_buckets": list(evidence_model.evidence_backfill_buckets),
                "source_layer_composition": dict(evidence_model.source_layer_composition),
                "metadata_completeness": dict(evidence_model.metadata_completeness),
                "ocr_usage": dict(evidence_model.ocr_usage),
                "teaching_layer_used": evidence_model.teaching_layer_used,
                "teaching_sources_count": evidence_model.teaching_sources_count,
                "teaching_layer_reason": evidence_model.teaching_layer_reason,
                "teaching_layer_excluded_reason": evidence_model.teaching_layer_excluded_reason,
                **study_path_diagnostics(answer),
            },
        }

    def _build_degraded_response(
        self,
        *,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
        greeting_style: str | None,
        tone_level: str | None,
        sources: list[dict[str, Any]],
        query_intent: QueryIntent | None,
        reason: str,
    ) -> dict[str, Any]:
        grounder = self._grounder
        grounder.quote_window_chars = self.config_store.load_effective().citation_excerpt_length
        enriched_sources = grounder.enrich_sources(
            question=question,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            retrieved_sources=sources,
            query_intent=query_intent,
        )
        answer = normalize_and_validate_answer(
            candidate={
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "direct_answer": reason,
                "uncertainty_note": reason,
                "evidence_strength": "limited" if sources else "insufficient",
                "ui_state": {
                    "greeting_style": greeting_style or self.artifacts.default_greeting_style,
                    "tone_level": tone_level or self.artifacts.default_tone_level,
                    "selected_madhhab_visible": True,
                    "answer_mode_visible": True,
                    "confidence_visible": True,
                    "disagreement_visible": True,
                },
            },
            schema=self.artifacts.answer_response_schema,
            request_context={
                "question": question,
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "greeting_style": greeting_style or self.artifacts.default_greeting_style,
                "tone_level": tone_level or self.artifacts.default_tone_level,
                "repo_root": str(self.repo_root),
            },
            retrieved_sources=enriched_sources,
        )
        evidence_model = grounder.build_evidence_model(
            answer=answer,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            enriched_sources=enriched_sources,
            query_intent=query_intent,
        )
        return {
            "answer": answer,
            "rendered_text": render_answer(answer, evidence_model),
            "answer_diagnostics": {
                "evidence_backfill_applied": evidence_model.evidence_backfill_applied,
                "evidence_backfill_buckets": list(evidence_model.evidence_backfill_buckets),
                "source_layer_composition": dict(evidence_model.source_layer_composition),
                "metadata_completeness": dict(evidence_model.metadata_completeness),
                "ocr_usage": dict(evidence_model.ocr_usage),
                "teaching_layer_used": evidence_model.teaching_layer_used,
                "teaching_sources_count": evidence_model.teaching_sources_count,
                "teaching_layer_reason": evidence_model.teaching_layer_reason,
                "teaching_layer_excluded_reason": evidence_model.teaching_layer_excluded_reason,
                **study_path_diagnostics(answer),
            },
        }

    def _finalize_chat_result(
        self,
        *,
        actor: RoleContext,
        question: str,
        answer_mode: str,
        selected_madhhab: str,
        started: float,
        result: ChatExecutionResult,
        error: Exception | None,
        user_id: int | None = None,
        chat_session_id: int | None = None,
        project_id: int | None = None,
        ) -> ChatExecutionResult:
        latency_ms = int((perf_counter() - started) * 1000)
        answer = result.answer or {}
        result.diagnostics = {
            **result.diagnostics,
            "latency_ms": latency_ms,
            "source_count": int(result.retrieval_summary.get("source_count", 0)),
            "citation_count": len(answer.get("citations", [])),
        }
        self._update_request_phase(
            actor=actor,
            phase="completed",
            summary="Completed" if not result.degraded else "Completed in degraded mode",
            retrieval_active=False,
            model_active=False,
            degraded_active=result.degraded,
            fallback_active=result.fallback_used,
            completed=True,
            outcome=result.status,
            details={
                **result.diagnostics,
                "retrieved_source_count": int(result.retrieval_summary.get("source_count", 0)),
            },
        )
        self.log_store.write_chat_log(
            {
                "timestamp": _utc_now(),
                "route": "/api/chat",
                "actor_id": actor.actor_id,
                "actor_role": actor.role,
                "request_id": actor.request_id,
                "session_id": actor.session_id,
                "success": not result.degraded,
                "latency_ms": latency_ms,
                "question": question,
                "answer_mode": answer_mode,
                "selected_madhhab": selected_madhhab,
                "model_requested": result.model_requested,
                "model_used": result.model_used,
                "fallback_used": result.fallback_used,
                "fallback_from": result.fallback_from,
                "fallback_to": result.fallback_to,
                "retrieval_status": json.dumps(result.retrieval_summary, ensure_ascii=False),
                "retrieved_source_count": int(result.retrieval_summary.get("source_count", 0)),
                "citation_count": len(answer.get("citations", [])),
                "degraded": result.degraded,
                "degraded_reason": result.degraded_reason,
                "error_class": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
            }
        )
        if user_id is not None and chat_session_id is not None:
            self.user_workspace.append_chat_message(
                session_id=chat_session_id,
                role="assistant",
                content=result.rendered_text,
                answer_mode=answer_mode,
                madhhab=selected_madhhab,
                evidence_json={
                    "answer": answer,
                    "rendered_text": result.rendered_text,
                    "diagnostics": result.diagnostics,
                    "project_id": project_id,
                },
            )
            if project_id is not None and str(answer.get("answer_mode") or answer_mode).strip() == "study_path":
                try:
                    self.user_workspace.save_project_study_state(
                        user_id=user_id,
                        project_id=project_id,
                        answer=answer,
                    )
                except Exception as exc:  # pragma: no cover - defensive persistence guard
                    self._emit_operator_event(
                        event_type="study_project_state_save_failed",
                        severity="warning",
                        summary="Study project state could not be saved",
                        request_id=actor.request_id,
                        session_id=actor.session_id,
                        route="/api/chat",
                        details={
                            "project_id": project_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
        if result.fallback_used:
            self._emit_operator_event(
                event_type="model_fallback",
                severity="warning",
                summary="Model fallback used for chat request",
                request_id=actor.request_id,
                session_id=actor.session_id,
                route="/api/chat",
                details={
                    "fallback_from": result.fallback_from,
                    "fallback_to": result.fallback_to,
                },
            )
        if result.degraded:
            self._emit_operator_event(
                event_type="degraded_answer_path",
                severity="warning",
                summary="Chat request completed on a degraded path",
                request_id=actor.request_id,
                session_id=actor.session_id,
                route="/api/chat",
                details={
                    "degraded_subsystem": result.degraded_subsystem,
                    "degraded_reason": result.degraded_reason,
                },
            )
        if error is not None:
            self._write_error(
                route="/api/chat",
                action="chat",
                actor=actor,
                subsystem=result.degraded_subsystem or "chat",
                error=error,
                details={
                    "question": question,
                    "answer_mode": answer_mode,
                    "selected_madhhab": selected_madhhab,
                },
            )
        return result

    def _write_audit(
        self,
        *,
        route: str,
        action: str,
        actor: RoleContext,
        success: bool,
        duration_ms: int,
        status: str,
        changed: dict[str, Any],
        details: dict[str, Any],
        error_summary: str | None = None,
    ) -> None:
        self.log_store.write_audit_log(
            {
                "timestamp": _utc_now(),
                "route": route,
                "action": action,
                "actor_id": actor.actor_id,
                "actor_role": actor.role,
                "request_id": actor.request_id,
                "session_id": actor.session_id,
                "success": success,
                "duration_ms": duration_ms,
                "status": status,
                "changed": changed,
                "details": details,
                "error_summary": error_summary,
            }
        )

    def _write_error(
        self,
        *,
        route: str,
        action: str,
        actor: RoleContext,
        subsystem: str,
        error: Exception,
        details: dict[str, Any],
    ) -> None:
        self.log_store.write_error_log(
            {
                "timestamp": _utc_now(),
                "route": route,
                "action": action,
                "actor_id": actor.actor_id,
                "actor_role": actor.role,
                "request_id": actor.request_id,
                "session_id": actor.session_id,
                "subsystem": subsystem,
                "error_class": type(error).__name__,
                "error_message": str(error),
                "details": details,
            }
        )


def _decode_json_fields(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for key in ("changed_json", "details_json", "retrieval_status"):
        value = decoded.get(key)
        if isinstance(value, str):
            try:
                decoded[key.removesuffix("_json")] = json.loads(value)
            except json.JSONDecodeError:
                decoded[key.removesuffix("_json")] = value
    return decoded


def _derive_mode_hooks(
    *,
    query_intent_intent_id: str,
    selected_madhhab: str,
    continuity_hooks: list[str],
) -> list[str]:
    hooks = list(continuity_hooks)
    if query_intent_intent_id == "compare_views" and "compare_views_behavior" not in hooks:
        hooks.append("compare_views_behavior")
    if query_intent_intent_id == "source_only" and "source_only_behavior" not in hooks:
        hooks.append("source_only_behavior")
    if selected_madhhab == "hanafi" and "hanafi_first_behavior" not in hooks:
        hooks.append("hanafi_first_behavior")
    return hooks


def _resolve_effective_answer_mode(
    *,
    requested_answer_mode: str,
    query_intent_id: str,
) -> str:
    if query_intent_id == "source_only":
        return "source_only"
    if query_intent_id == "compare_views":
        return "compare_views"
    if query_intent_id == "scholar_perspective":
        return "scholar_perspective"
    if requested_answer_mode == "scholar_perspective":
        return "research"
    return requested_answer_mode


MICRO_FAST_SAFE_INTENTS = {
    "direct_source_lookup",
    "source_only",
    "compare_views",
    "explain_term",
}


def _evaluate_micro_fast_assist(
    *,
    config: RuntimeConfig,
    fast_result: dict[str, Any],
    baseline_query_intent: QueryIntent,
    baseline_answer_mode: str,
    effective_question: str,
    selected_madhhab: str,
) -> dict[str, Any]:
    candidate_intent = str(fast_result.get("micro_predicted_intent") or "").strip()
    candidate_mode = str(fast_result.get("micro_suggested_mode") or "").strip()
    candidate_query = str(fast_result.get("micro_rewritten_query") or "").strip()
    confidence = _coerce_float(fast_result.get("confidence"))

    query_intent = baseline_query_intent
    effective_answer_mode = baseline_answer_mode
    retrieval_question = effective_question
    assist_applied = False
    assist_rejected_reason: str | None = None
    applied_changes: list[str] = []

    if not config.micro_fast_assist_enabled:
        assist_rejected_reason = "assist_disabled"
    elif not fast_result.get("success"):
        assist_rejected_reason = str(fast_result.get("error") or "micro_fast_failed")
    elif confidence is None or confidence < config.micro_fast_assist_confidence_threshold:
        assist_rejected_reason = "low_confidence"
    elif candidate_intent not in MICRO_FAST_SAFE_INTENTS:
        assist_rejected_reason = "unsafe_intent"
    elif not _is_safe_micro_fast_override(
        baseline_intent=baseline_query_intent.intent_id,
        candidate_intent=candidate_intent,
    ):
        assist_rejected_reason = "risky_disagreement"
    elif not _is_safe_micro_fast_mode_override(
        baseline_mode=baseline_answer_mode,
        candidate_mode=candidate_mode,
    ):
        assist_rejected_reason = "risky_mode_override"
    else:
        assist_applied = True
        if candidate_query and candidate_query.casefold() != effective_question.casefold():
            retrieval_question = candidate_query
            applied_changes.append("retrieval_query_rewrite")
        if candidate_intent != baseline_query_intent.intent_id:
            query_intent = route_query_intent(
                question=effective_question,
                answer_mode=candidate_mode or baseline_answer_mode,
                selected_madhhab=selected_madhhab,
            )
            applied_changes.append("intent_override")
        if candidate_mode in {"source_only", "compare_views"} and candidate_mode != baseline_answer_mode:
            effective_answer_mode = candidate_mode
            applied_changes.append("answer_mode_override")
        if not applied_changes:
            applied_changes.append("router_confirmation")

    return {
        "assist_applied": assist_applied,
        "assist_rejected_reason": assist_rejected_reason,
        "applied_changes": applied_changes,
        "retrieval_question": retrieval_question,
        "effective_answer_mode": effective_answer_mode,
        "query_intent": query_intent,
        "baseline_intent": baseline_query_intent.intent_id,
        "final_intent": query_intent.intent_id,
        "candidate_intent": candidate_intent or None,
        "candidate_mode": candidate_mode or None,
        "candidate_query": candidate_query or None,
        "confidence": confidence,
    }


def _is_safe_micro_fast_override(*, baseline_intent: str, candidate_intent: str) -> bool:
    if candidate_intent == baseline_intent:
        return True
    if baseline_intent in {"source_only", "compare_views"}:
        return False
    safe_pairs = {
        ("direct_source_lookup", "source_only"),
        ("direct_source_lookup", "compare_views"),
        ("direct_source_lookup", "explain_term"),
    }
    return (baseline_intent, candidate_intent) in safe_pairs


def _is_safe_micro_fast_mode_override(*, baseline_mode: str, candidate_mode: str) -> bool:
    if not candidate_mode:
        return True
    if candidate_mode == baseline_mode:
        return True
    if baseline_mode in {"source_only", "compare_views"}:
        return False
    return baseline_mode == "research" and candidate_mode in {"source_only", "compare_views"}


def _assist_decision_public_payload(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "assist_applied": bool(decision.get("assist_applied")),
        "assist_rejected_reason": decision.get("assist_rejected_reason"),
        "applied_changes": list(decision.get("applied_changes", [])),
        "baseline_intent": decision.get("baseline_intent"),
        "final_intent": decision.get("final_intent"),
        "effective_answer_mode": decision.get("effective_answer_mode"),
        "retrieval_question": decision.get("retrieval_question"),
        "candidate_intent": decision.get("candidate_intent"),
        "candidate_mode": decision.get("candidate_mode"),
        "candidate_query": decision.get("candidate_query"),
        "confidence": decision.get("confidence"),
    }


def _rewrite_retrieval_question_for_worship_topic(
    *,
    question: str,
    query_intent: QueryIntent,
) -> str:
    topic = str(query_intent.worship_topic or "").strip().lower()
    if topic == "prayer_method":
        return (
            "prayer method salat salah qiyam recitation takbir ruku sujud "
            "tashahhud salam"
        )
    if topic == "purification_method":
        return (
            "wudu ablution wash face hands elbows wipe head feet ankles "
            "mouth nose"
        )
    if topic == "prayer_with_purification_prerequisite":
        return (
            "wudu required before prayer salat condition validity purity "
            "for prayer"
        )
    return question


def _can_use_local_fast_response(
    *,
    fast_path_decision: FastPathDecision,
    query_intent: QueryIntent,
) -> bool:
    if fast_path_decision.path_used != "fast":
        return False
    return query_intent.intent_id in {"source_only", "direct_source_lookup"}


def _fast_local_direct_answer(
    *,
    answer_mode: str,
    query_intent: QueryIntent,
    sources: list[dict[str, Any]],
) -> str:
    if answer_mode == "source_only" or query_intent.intent_id == "source_only":
        return ""
    if query_intent.intent_id == "direct_source_lookup":
        if len(sources) == 1:
            return "The retrieved source below is the closest grounded match to the question."
        return "The retrieved sources below are the closest grounded matches to the question."
    return "The retrieved sources below are the grounded basis for this answer."


def _fast_local_evidence_strength(sources: list[dict[str, Any]]) -> str:
    if len(sources) >= 3:
        return "moderate"
    if len(sources) >= 1:
        return "limited"
    return "insufficient"


def _fast_local_uncertainty_note(
    *,
    answer_mode: str,
    query_intent: QueryIntent,
    sources: list[dict[str, Any]],
) -> str | None:
    if answer_mode == "source_only" or query_intent.intent_id == "source_only":
        return "This fast-path response is limited to the highest-confidence retrieved source text."
    if len(sources) <= 1:
        return "This fast-path response is based on a very small retrieved source set."
    return None


def _skipped_micro_shadow_payload(
    *,
    question: str,
    effective_question: str,
    request_id: str,
    session_id: str,
    rule_router_intent: str,
    error: str,
) -> dict[str, Any]:
    return {
        "timestamp": _utc_now(),
        "session_id": session_id,
        "request_id": request_id,
        "original_question": question,
        "effective_question": effective_question,
        "rule_router_intent": rule_router_intent,
        "micro_predicted_intent": None,
        "micro_suggested_mode": None,
        "micro_rewritten_query": None,
        "confidence": None,
        "latency_ms": 0,
        "end_to_end_latency_ms": 0,
        "warmup_latency_ms": 0,
        "cache_state": "skipped",
        "warmup_applied": False,
        "model_path": None,
        "model_name": None,
        "output_valid": False,
        "success": False,
        "error": error,
    }


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare_benchmark_reports(
    fast_router: dict[str, Any],
    smart_shadow: dict[str, Any],
) -> dict[str, Any]:
    fast_results = {result.get("label"): result for result in fast_router.get("results", [])}
    smart_results = {result.get("label"): result for result in smart_shadow.get("results", [])}
    shared_labels = [label for label in fast_results if label in smart_results]
    intent_agreement = 0
    mode_agreement = 0
    query_overlap_values: list[float] = []
    per_prompt: list[dict[str, Any]] = []
    for label in shared_labels:
        fast_result = fast_results[label]
        smart_result = smart_results[label]
        comparison = compare_tier_outputs(fast_result, smart_result)
        if comparison.get("intent_agreement"):
            intent_agreement += 1
        if comparison.get("mode_agreement"):
            mode_agreement += 1
        overlap = comparison.get("rewritten_query_overlap")
        if overlap is not None:
            query_overlap_values.append(float(overlap))
        per_prompt.append(
            {
                "label": label,
                "intent_agreement": comparison.get("intent_agreement"),
                "mode_agreement": comparison.get("mode_agreement"),
                "rewritten_query_overlap": overlap,
                "fast_latency_ms": fast_result.get("latency_ms"),
                "smart_latency_ms": smart_result.get("latency_ms"),
                "fast_output_valid": fast_result.get("output_valid"),
                "smart_output_valid": smart_result.get("output_valid"),
            }
        )
    agreement_count = len(shared_labels)
    return {
        "shared_prompt_count": agreement_count,
        "intent_agreement_count": intent_agreement,
        "mode_agreement_count": mode_agreement,
        "average_rewritten_query_overlap": (
            round(sum(query_overlap_values) / len(query_overlap_values), 3)
            if query_overlap_values
            else None
        ),
        "fast_router_under_3s": fast_router.get("under_3s"),
        "fast_router_average_latency_ms": fast_router.get("average_latency_ms"),
        "smart_shadow_average_latency_ms": smart_shadow.get("average_latency_ms"),
        "per_prompt": per_prompt,
    }


def _normalize_micro_benchmark_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    if "fast_router" in report and "smart_shadow" in report:
        return report
    if "average_latency_ms" not in report and "results" not in report:
        return report
    smart_shadow = dict(report)
    return {
        "generated_at": report.get("generated_at"),
        "prompt_count": report.get("prompt_count"),
        "success": report.get("success"),
        "success_count": report.get("success_count"),
        "failure_count": report.get("failure_count"),
        "output_valid_count": report.get("output_valid_count"),
        "average_latency_ms": report.get("average_latency_ms"),
        "p95_latency_ms": report.get("p95_latency_ms"),
        "fast_router": {},
        "smart_shadow": smart_shadow,
        "comparison": {},
    }


def _summarize_source_snippets(snippets: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "top_source_classifications": _count_snippet_field(
            snippets,
            field_name="source_classification",
            fallback_field="source_type",
        ),
        "top_source_families": _count_snippet_field(snippets, field_name="source_family"),
        "top_collections": _count_snippet_field(snippets, field_name="collection"),
        "top_sections": _count_snippet_field(snippets, field_name="section_label"),
        "top_source_role_boundaries": _count_snippet_field(
            snippets,
            field_name="source_role_boundary",
        ),
        "top_source_lineages": _count_snippet_field(
            snippets,
            field_name="source_lineage",
        ),
    }


def _count_snippet_field(
    snippets: list[dict[str, Any]],
    *,
    field_name: str,
    fallback_field: str | None = None,
    limit: int = 6,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for snippet in snippets:
        value = str(snippet.get(field_name, "") or "").strip()
        if not value and fallback_field:
            value = str(snippet.get(fallback_field, "") or "").strip()
        if not value:
            value = "unknown"
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:limit])


def _dedupe_strings(values: list[str | None]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in deduped:
            continue
        deduped.append(candidate)
    return deduped


def _source_truth_diagnostics(snippets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata_completeness": _metadata_coverage_for_sources(snippets),
        "ocr_usage": _ocr_usage_for_sources(snippets),
        "source_layer_composition": _source_layer_composition_for_sources(snippets),
    }


def _metadata_coverage_for_sources(snippets: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(snippets), 1)
    tracked_fields = (
        "author",
        "language",
        "collection",
        "source_lineage",
        "source_role_boundary",
    )
    coverage: dict[str, Any] = {"source_count": len(snippets), "coverage": {}}
    for field_name in tracked_fields:
        known = 0
        for snippet in snippets:
            value = str(snippet.get(field_name, "") or "").strip().lower()
            if value not in {"", "unknown"}:
                known += 1
        coverage["coverage"][field_name] = {
            "known": known,
            "unknown": len(snippets) - known,
            "coverage_percent": round((known / total) * 100, 1),
        }
    return coverage


def _ocr_usage_for_sources(snippets: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    extraction_quality_counts: dict[str, int] = {}
    for snippet in snippets:
        status = str(snippet.get("ocr_status", "") or "not_attempted")
        status_counts[status] = status_counts.get(status, 0) + 1
        backend = str(snippet.get("ocr_backend", "") or "").strip()
        if backend:
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        quality = str(snippet.get("extraction_quality", "") or "unknown")
        extraction_quality_counts[quality] = extraction_quality_counts.get(quality, 0) + 1
    return {
        "source_count": len(snippets),
        "ocr_derived_sources": sum(1 for snippet in snippets if bool(snippet.get("ocr_derived", False))),
        "ocr_status_counts": status_counts,
        "ocr_backend_counts": backend_counts,
        "extraction_quality_counts": extraction_quality_counts,
    }


def _source_layer_composition_for_sources(snippets: list[dict[str, Any]]) -> dict[str, int]:
    composition: dict[str, int] = {}
    for snippet in snippets:
        layer = str(
            snippet.get("source_role_boundary", "")
            or snippet.get("source_classification", "")
            or "unknown"
        )
        composition[layer] = composition.get(layer, 0) + 1
    return composition


ALIGNMENT_STOPWORDS = {
    "a",
    "about",
    "across",
    "and",
    "behind",
    "book",
    "can",
    "compare",
    "criteria",
    "different",
    "find",
    "first",
    "follow",
    "from",
    "give",
    "hadith",
    "hanafi",
    "is",
    "issue",
    "material",
    "me",
    "modern",
    "now",
    "only",
    "other",
    "over",
    "path",
    "part",
    "primary",
    "purification",
    "quran",
    "qur",
    "research",
    "ruling",
    "say",
    "school",
    "schools",
    "shafi",
    "show",
    "side",
    "source",
    "sources",
    "summarize",
    "teacher",
    "text",
    "texts",
    "that",
    "the",
    "this",
    "topic",
    "transcript",
    "uncertain",
    "uncertainty",
    "view",
    "views",
    "what",
    "wiping",
    "which",
    "strongest",
}


def _retrieval_alignment_diagnostics(
    question: str,
    snippets: list[dict[str, Any]],
    *,
    query_intent: Any,
) -> dict[str, Any]:
    query_tokens = _alignment_tokens(question)
    if not query_tokens:
        return {
            "query_tokens": [],
            "matched_tokens": [],
            "unmatched_tokens": [],
            "coverage_ratio": 1.0,
            "should_degrade": False,
            "reason": "insufficient_query_tokens",
        }
    haystack_tokens: set[str] = set()
    for snippet in snippets:
        haystack_tokens.update(
            _alignment_tokens(
                " ".join(
                    [
                        str(snippet.get("title", "")),
                        str(snippet.get("collection", "")),
                        str(snippet.get("reference", "")),
                        str(snippet.get("section_label", "")),
                        str(snippet.get("book", "")),
                        str(snippet.get("chapter", "")),
                        str(snippet.get("section", "")),
                        str(snippet.get("quote", "")),
                        str(snippet.get("source_classification", "")),
                        str(snippet.get("source_family", "")),
                        str(snippet.get("canonical_family", "")),
                        str(snippet.get("document_kind", "")),
                        str(snippet.get("commentary_target", "")),
                        str(snippet.get("fatwa_authority", "")),
                    ]
                )
            )
        )
    matched_tokens = [token for token in query_tokens if token in haystack_tokens]
    unmatched_tokens = [token for token in query_tokens if token not in haystack_tokens]
    coverage_ratio = round(len(matched_tokens) / len(query_tokens), 3)
    high_risk_intent = str(getattr(query_intent, "intent_id", "") or "") in {
        "ruling_lookup",
        "fatwa_lookup",
        "transcript_lookup",
        "compare_views",
    }
    domain_unmatched = [token for token in unmatched_tokens if len(token) >= 5]
    should_degrade = (
        high_risk_intent
        and len(query_tokens) >= 2
        and coverage_ratio <= 0.5
        and len(domain_unmatched) >= 2
    )
    return {
        "query_tokens": query_tokens[:12],
        "matched_tokens": matched_tokens[:12],
        "unmatched_tokens": unmatched_tokens[:12],
        "coverage_ratio": coverage_ratio,
        "should_degrade": should_degrade,
        "reason": (
            "low_query_token_coverage_for_high_risk_request"
            if should_degrade
            else "coverage_ok"
        ),
    }


def _alignment_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
        token = raw_token.strip("_")
        if not token or token in ALIGNMENT_STOPWORDS:
            continue
        if token.isascii() and len(token) < 3:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if token in ALIGNMENT_STOPWORDS or token in tokens:
            continue
        tokens.append(token)
    return tokens


def _question_targets_supplied_sources(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    generic_source_phrases = (
        "provided source",
        "provided sources",
        "these source",
        "these sources",
        "the source provided",
        "the sources provided",
    )
    return any(phrase in normalized for phrase in generic_source_phrases)


def _human_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _iso_from_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, UTC).isoformat()


def _row_value(row: dict[str, Any] | None, key: str) -> Any:
    if not row:
        return None
    return row.get(key)
