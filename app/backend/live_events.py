"""Live operator events and request-scoped execution state."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class OperatorEvent:
    event_id: str
    event_type: str
    ts: str
    severity: str
    summary: str
    related_task_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    route: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    id: int | None = None


class LiveEventBroker:
    """Simple process-local publish/subscribe buffer for SSE consumers."""

    def __init__(self, *, max_events: int = 500) -> None:
        self._condition = threading.Condition()
        self._events: list[dict[str, Any]] = []
        self._max_events = max_events

    def publish(self, event: dict[str, Any]) -> None:
        with self._condition:
            self._events.append(dict(event))
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]
            self._condition.notify_all()

    def backlog(
        self,
        *,
        after_id: int = 0,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._condition:
            return [
                dict(event)
                for event in self._events
                if int(event.get("id") or 0) > after_id
                and (request_id is None or event.get("request_id") == request_id)
            ]

    def wait_for_events(
        self,
        *,
        after_id: int = 0,
        timeout_seconds: float = 15.0,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                ready = [
                    dict(event)
                    for event in self._events
                    if int(event.get("id") or 0) > after_id
                    and (request_id is None or event.get("request_id") == request_id)
                ]
                if ready:
                    return ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(timeout=remaining)


class RequestStateStore:
    """Track request-scoped execution phases for live manager status."""

    def __init__(self, *, retention_seconds: int = 3600) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._retention_seconds = retention_seconds

    def update(
        self,
        request_id: str,
        *,
        session_id: str | None = None,
        route: str | None = None,
        phase: str | None = None,
        retrieval_active: bool | None = None,
        model_active: bool | None = None,
        degraded_active: bool | None = None,
        fallback_active: bool | None = None,
        completed: bool | None = None,
        outcome: str | None = None,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            existing = dict(self._states.get(request_id, {}))
            now = _utc_now()
            state = {
                "request_id": request_id,
                "session_id": session_id or existing.get("session_id"),
                "route": route or existing.get("route"),
                "phase": phase or existing.get("phase") or "receiving_request",
                "retrieval_active": (
                    retrieval_active
                    if retrieval_active is not None
                    else existing.get("retrieval_active", False)
                ),
                "model_active": (
                    model_active if model_active is not None else existing.get("model_active", False)
                ),
                "degraded_active": (
                    degraded_active
                    if degraded_active is not None
                    else existing.get("degraded_active", False)
                ),
                "fallback_active": (
                    fallback_active
                    if fallback_active is not None
                    else existing.get("fallback_active", False)
                ),
                "completed": completed if completed is not None else existing.get("completed", False),
                "outcome": outcome or existing.get("outcome"),
                "summary": summary or existing.get("summary"),
                "details": details if details is not None else existing.get("details", {}),
                "created_at": existing.get("created_at") or now,
                "updated_at": now,
            }
            if state["completed"]:
                state["completed_at"] = now
            self._states[request_id] = state
            return dict(state)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._prune_locked()
            state = self._states.get(request_id)
            return dict(state) if state else None

    def list_active(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            active = [
                dict(state)
                for state in self._states.values()
                if not state.get("completed", False)
            ]
        active.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return active[:limit]

    def _prune_locked(self) -> None:
        now = datetime.now(UTC).timestamp()
        removable = []
        for request_id, state in self._states.items():
            updated_at = state.get("updated_at")
            if not updated_at:
                continue
            try:
                updated_ts = datetime.fromisoformat(updated_at).timestamp()
            except ValueError:
                continue
            if state.get("completed") and now - updated_ts > self._retention_seconds:
                removable.append(request_id)
        for request_id in removable:
            self._states.pop(request_id, None)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
