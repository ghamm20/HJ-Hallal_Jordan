"""Durable task lifecycle persistence, reconciliation helpers, and local dispatch."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

TASK_STATUSES = {
    "queued",
    "running",
    "succeeded",
    "failed",
    "blocked",
    "canceled",
    "degraded",
}
ACTIVE_TASK_STATUSES = {"queued", "running"}
TASK_CONFLICT_RULES = {
    "reload_retrieval_assets": {"reload_retrieval_assets", "reindex_documents"},
    "reindex_documents": {"reload_retrieval_assets", "reindex_documents"},
    "diagnostics": set(),
    "document_ingest": {"document_ingest", "reindex_documents"},
    "bulk_audit": set(),
    "maintenance": {"reload_retrieval_assets", "reindex_documents", "maintenance"},
}
TASK_TABLE_MIGRATIONS = {
    "runtime_instance_id": "TEXT",
    "started_by_runtime": "TEXT",
    "process_id": "INTEGER",
    "host_id": "TEXT",
    "last_heartbeat_at": "TEXT",
    "last_progress_at": "TEXT",
    "terminal_reason": "TEXT",
    "interruption_reason": "TEXT",
    "degraded_reason": "TEXT",
    "reconciled_at": "TEXT",
    "reconciled_by": "TEXT",
    "was_reconciled": "INTEGER NOT NULL DEFAULT 0",
    "live_state_preserved": "INTEGER",
    "live_state_swapped": "INTEGER",
    "execution_mode": "TEXT NOT NULL DEFAULT 'inline'",
}


@dataclass(slots=True)
class TaskRuntimeIdentity:
    runtime_instance_id: str
    process_id: int
    host_id: str


@dataclass(slots=True)
class TaskCreateRequest:
    task_type: str
    requested_by: str
    requested_role: str
    request_route: str
    request_source: str
    parameters: dict[str, Any]
    runtime_identity: TaskRuntimeIdentity
    execution_mode: str = "inline"
    retry_of: str | None = None
    retry_reason: str | None = None


@dataclass(slots=True)
class TaskDispatchRequest:
    task_id: str
    background: bool
    execute: Callable[[], None]
    name: str


@dataclass(slots=True)
class TaskDispatchResult:
    execution_mode: str
    background: bool
    thread_name: str | None = None


class LocalTaskRunner:
    """Background-safe local task runner with inline and thread dispatch modes."""

    def __init__(self) -> None:
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def dispatch(self, request: TaskDispatchRequest) -> TaskDispatchResult:
        if not request.background:
            request.execute()
            return TaskDispatchResult(
                execution_mode="inline",
                background=False,
            )

        def worker() -> None:
            try:
                request.execute()
            finally:
                with self._lock:
                    self._threads.pop(request.task_id, None)

        thread = threading.Thread(
            target=worker,
            name=request.name,
            daemon=True,
        )
        with self._lock:
            self._threads[request.task_id] = thread
        thread.start()
        return TaskDispatchResult(
            execution_mode="background",
            background=True,
            thread_name=thread.name,
        )

    def active_thread_names(self) -> list[str]:
        with self._lock:
            return [thread.name for thread in self._threads.values() if thread.is_alive()]

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        with self._lock:
            threads = list(self._threads.values())
        deadline = datetime.now(UTC).timestamp() + max(timeout_s, 0.0)
        for thread in threads:
            if not thread.is_alive():
                continue
            remaining = deadline - datetime.now(UTC).timestamp()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)


class TaskStore:
    """Persist task rows and task lifecycle events in sqlite."""

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
                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL UNIQUE,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        requested_by TEXT NOT NULL,
                        requested_role TEXT NOT NULL,
                        request_route TEXT NOT NULL,
                        request_source TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        progress_json TEXT NOT NULL,
                        result_summary_json TEXT NOT NULL,
                        error_summary TEXT,
                        duration_ms INTEGER,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        retry_of TEXT,
                        retry_reason TEXT,
                        blocking_reason TEXT,
                        runtime_instance_id TEXT,
                        started_by_runtime TEXT,
                        process_id INTEGER,
                        host_id TEXT,
                        last_heartbeat_at TEXT,
                        last_progress_at TEXT,
                        terminal_reason TEXT,
                        interruption_reason TEXT,
                        degraded_reason TEXT,
                        reconciled_at TEXT,
                        reconciled_by TEXT,
                        was_reconciled INTEGER NOT NULL DEFAULT 0,
                        live_state_preserved INTEGER,
                        live_state_swapped INTEGER,
                        execution_mode TEXT NOT NULL DEFAULT 'inline'
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        from_status TEXT,
                        to_status TEXT,
                        actor_id TEXT,
                        actor_role TEXT,
                        details_json TEXT NOT NULL
                    )
                    """
                )
                self._ensure_columns(connection)
                connection.commit()
            self.available = True
            self.last_error = None
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"

    def create_task(self, request: TaskCreateRequest) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        created_at = _utc_now()
        retry_count = self._next_retry_count(request.retry_of)
        task = {
            "task_id": task_id,
            "task_type": request.task_type,
            "status": "queued",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "requested_by": request.requested_by,
            "requested_role": request.requested_role,
            "request_route": request.request_route,
            "request_source": request.request_source,
            "parameters_json": json.dumps(request.parameters, ensure_ascii=False),
            "progress_json": json.dumps(
                {
                    "phase": "queued",
                    "current_step": "waiting_to_start",
                    "last_update_at": created_at,
                },
                ensure_ascii=False,
            ),
            "result_summary_json": json.dumps({}, ensure_ascii=False),
            "error_summary": None,
            "duration_ms": None,
            "retry_count": retry_count,
            "retry_of": request.retry_of,
            "retry_reason": request.retry_reason,
            "blocking_reason": None,
            "runtime_instance_id": request.runtime_identity.runtime_instance_id,
            "started_by_runtime": request.runtime_identity.runtime_instance_id,
            "process_id": request.runtime_identity.process_id,
            "host_id": request.runtime_identity.host_id,
            "last_heartbeat_at": None,
            "last_progress_at": created_at,
            "terminal_reason": None,
            "interruption_reason": None,
            "degraded_reason": None,
            "reconciled_at": None,
            "reconciled_by": None,
            "was_reconciled": 0,
            "live_state_preserved": None,
            "live_state_swapped": None,
            "execution_mode": request.execution_mode,
        }
        self._write(
            """
            INSERT INTO tasks (
                task_id, task_type, status, created_at, started_at, finished_at,
                requested_by, requested_role, request_route, request_source,
                parameters_json, progress_json, result_summary_json, error_summary,
                duration_ms, retry_count, retry_of, retry_reason, blocking_reason,
                runtime_instance_id, started_by_runtime, process_id, host_id,
                last_heartbeat_at, last_progress_at, terminal_reason,
                interruption_reason, degraded_reason, reconciled_at, reconciled_by,
                was_reconciled, live_state_preserved, live_state_swapped, execution_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task["task_id"],
                task["task_type"],
                task["status"],
                task["created_at"],
                task["started_at"],
                task["finished_at"],
                task["requested_by"],
                task["requested_role"],
                task["request_route"],
                task["request_source"],
                task["parameters_json"],
                task["progress_json"],
                task["result_summary_json"],
                task["error_summary"],
                task["duration_ms"],
                task["retry_count"],
                task["retry_of"],
                task["retry_reason"],
                task["blocking_reason"],
                task["runtime_instance_id"],
                task["started_by_runtime"],
                task["process_id"],
                task["host_id"],
                task["last_heartbeat_at"],
                task["last_progress_at"],
                task["terminal_reason"],
                task["interruption_reason"],
                task["degraded_reason"],
                task["reconciled_at"],
                task["reconciled_by"],
                task["was_reconciled"],
                task["live_state_preserved"],
                task["live_state_swapped"],
                task["execution_mode"],
            ),
        )
        self.record_event(
            task_id=task_id,
            event_type="created",
            from_status=None,
            to_status="queued",
            actor_id=request.requested_by,
            actor_role=request.requested_role,
            details={
                "request_route": request.request_route,
                "request_source": request.request_source,
                "parameters": request.parameters,
                "retry_of": request.retry_of,
                "retry_reason": request.retry_reason,
                "runtime_instance_id": request.runtime_identity.runtime_instance_id,
                "process_id": request.runtime_identity.process_id,
                "host_id": request.runtime_identity.host_id,
                "execution_mode": request.execution_mode,
            },
        )
        return self.get_task(task_id) or {}

    def update_status(
        self,
        *,
        task_id: str,
        status: str,
        actor_id: str | None,
        actor_role: str | None,
        details: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        result_summary: dict[str, Any] | None = None,
        error_summary: str | None = None,
        duration_ms: int | None = None,
        blocking_reason: str | None = None,
        progress: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if status not in TASK_STATUSES:
            raise ValueError(f"unsupported task status: {status}")
        existing = self.get_task(task_id)
        if not existing:
            return None

        next_progress = dict(existing.get("progress", {}))
        if progress:
            next_progress.update(progress)
        progress_timestamp = _utc_now()
        next_progress["last_update_at"] = progress_timestamp

        updates: dict[str, Any] = {
            "status": status,
            "started_at": started_at or existing.get("started_at"),
            "finished_at": finished_at or existing.get("finished_at"),
            "progress_json": json.dumps(next_progress, ensure_ascii=False),
            "result_summary_json": json.dumps(
                result_summary or existing.get("result_summary", {}),
                ensure_ascii=False,
            ),
            "error_summary": error_summary if error_summary is not None else existing.get("error_summary"),
            "duration_ms": duration_ms if duration_ms is not None else existing.get("duration_ms"),
            "blocking_reason": blocking_reason if blocking_reason is not None else existing.get("blocking_reason"),
            "last_heartbeat_at": progress_timestamp,
            "last_progress_at": progress_timestamp,
        }
        updates.update(self._normalize_metadata(metadata, existing))
        self._update_row(task_id=task_id, updates=updates)
        self.record_event(
            task_id=task_id,
            event_type="transition",
            from_status=existing["status"],
            to_status=status,
            actor_id=actor_id,
            actor_role=actor_role,
            details=details or {},
        )
        return self.get_task(task_id)

    def update_progress(
        self,
        *,
        task_id: str,
        actor_id: str | None,
        actor_role: str | None,
        progress: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_task(task_id)
        if not existing:
            return None
        merged = dict(existing.get("progress", {}))
        merged.update(progress)
        timestamp = _utc_now()
        merged["last_update_at"] = timestamp
        updates = {
            "progress_json": json.dumps(merged, ensure_ascii=False),
            "last_heartbeat_at": timestamp,
            "last_progress_at": timestamp,
        }
        updates.update(self._normalize_metadata(metadata, existing))
        self._update_row(task_id=task_id, updates=updates)
        self.record_event(
            task_id=task_id,
            event_type="progress",
            from_status=existing["status"],
            to_status=existing["status"],
            actor_id=actor_id,
            actor_role=actor_role,
            details=merged,
        )
        return self.get_task(task_id)

    def reconcile_task(
        self,
        *,
        task_id: str,
        status: str,
        actor_id: str,
        actor_role: str,
        reason: str,
        details: dict[str, Any],
        progress: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_task(task_id)
        if not existing:
            return None
        finished_at = _utc_now()
        duration_ms = _duration_ms(existing.get("started_at") or existing.get("created_at"), finished_at)
        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault("terminal_reason", reason)
        combined_metadata.setdefault("interruption_reason", "process_interruption")
        combined_metadata["was_reconciled"] = True
        combined_metadata["reconciled_at"] = finished_at
        combined_metadata["reconciled_by"] = actor_id
        updated = self.update_status(
            task_id=task_id,
            status=status,
            actor_id=actor_id,
            actor_role=actor_role,
            details=details,
            finished_at=finished_at,
            duration_ms=duration_ms,
            progress=progress,
            metadata=combined_metadata,
        )
        self.record_event(
            task_id=task_id,
            event_type="reconciled",
            from_status=existing.get("status"),
            to_status=status,
            actor_id=actor_id,
            actor_role=actor_role,
            details={
                "reason": reason,
                **details,
            },
        )
        return updated

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        rows = self._fetch_rows(
            "SELECT * FROM tasks WHERE task_id = ? LIMIT 1",
            [task_id],
        )
        return rows[0] if rows else None

    def list_tasks(
        self,
        *,
        limit: int = 20,
        statuses: list[str] | None = None,
        task_type: str | None = None,
        retried_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM tasks"
        params: list[Any] = []
        clauses: list[str] = []
        if statuses:
            clauses.append("status IN ({})".format(", ".join("?" for _ in statuses)))
            params.extend(statuses)
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        if retried_only:
            clauses.append("retry_count > 0")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._fetch_rows(query, params)

    def list_events(self, *, task_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._fetch_rows(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id ASC LIMIT ?",
            [task_id, limit],
        )

    def find_active_conflicts(self, task_type: str) -> list[dict[str, Any]]:
        conflicting_types = sorted(conflicts_for_task(task_type))
        if not conflicting_types:
            return []
        placeholders = ", ".join("?" for _ in conflicting_types)
        params: list[Any] = [*sorted(ACTIVE_TASK_STATUSES), *conflicting_types]
        query = f"""
            SELECT * FROM tasks
            WHERE status IN (?, ?)
              AND task_type IN ({placeholders})
            ORDER BY id DESC
        """
        return self._fetch_rows(query, params)

    def latest_task(
        self,
        *,
        task_type: str,
        statuses: list[str] | None = None,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM tasks WHERE task_type = ?"
        params: list[Any] = [task_type]
        if statuses:
            query += " AND status IN ({})".format(", ".join("?" for _ in statuses))
            params.extend(statuses)
        query += " ORDER BY id DESC LIMIT 1"
        rows = self._fetch_rows(query, params)
        return rows[0] if rows else None

    def stale_incomplete_tasks(self) -> list[dict[str, Any]]:
        return self.list_tasks(limit=1000, statuses=sorted(ACTIVE_TASK_STATUSES))

    def record_event(
        self,
        *,
        task_id: str,
        event_type: str,
        from_status: str | None,
        to_status: str | None,
        actor_id: str | None,
        actor_role: str | None,
        details: dict[str, Any],
    ) -> None:
        self._write(
            """
            INSERT INTO task_events (
                task_id, timestamp, event_type, from_status, to_status,
                actor_id, actor_role, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                _utc_now(),
                event_type,
                from_status,
                to_status,
                actor_id,
                actor_role,
                json.dumps(details, ensure_ascii=False),
            ),
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_columns(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute("PRAGMA table_info(tasks)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_type in TASK_TABLE_MIGRATIONS.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                f"ALTER TABLE tasks ADD COLUMN {column_name} {column_type}"
            )

    def _next_retry_count(self, retry_of: str | None) -> int:
        if not retry_of:
            return 0
        previous = self.get_task(retry_of)
        if not previous:
            return 1
        return int(previous.get("retry_count", 0)) + 1

    def _normalize_metadata(
        self,
        metadata: dict[str, Any] | None,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        if not metadata:
            return {}
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None and key not in {"live_state_preserved", "live_state_swapped"}:
                normalized[key] = existing.get(key)
            else:
                normalized[key] = value
        return normalized

    def _update_row(self, *, task_id: str, updates: dict[str, Any]) -> None:
        columns = []
        params: list[Any] = []
        for key, value in updates.items():
            columns.append(f"{key} = ?")
            params.append(_db_value(value))
        params.append(task_id)
        self._write(
            f"UPDATE tasks SET {', '.join(columns)} WHERE task_id = ?",
            params,
        )

    def _write(self, query: str, params: tuple[Any, ...] | list[Any]) -> None:
        if not self.available:
            return
        try:
            with self._connect() as connection:
                connection.execute(query, tuple(params))
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
            return [_decode_task_row(row) for row in rows]
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


def build_runtime_identity() -> TaskRuntimeIdentity:
    return TaskRuntimeIdentity(
        runtime_instance_id=str(uuid.uuid4()),
        process_id=os.getpid(),
        host_id=socket.gethostname(),
    )


def conflicts_for_task(task_type: str) -> set[str]:
    return set(TASK_CONFLICT_RULES.get(task_type, {task_type}))


def _decode_task_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for key in ("parameters_json", "progress_json", "result_summary_json", "details_json"):
        value = decoded.get(key)
        decoded_key = key.removesuffix("_json")
        if isinstance(value, str):
            try:
                decoded[decoded_key] = json.loads(value)
            except json.JSONDecodeError:
                decoded[decoded_key] = value
        elif value is not None:
            decoded[decoded_key] = value
    for key in ("was_reconciled", "live_state_preserved", "live_state_swapped"):
        if key in decoded and decoded[key] is not None:
            decoded[key] = bool(decoded[key])
    return decoded


def _db_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def _duration_ms(started_at: str | None, finished_at: str) -> int | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        finish = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    return max(0, int((finish - start).total_seconds() * 1000))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
