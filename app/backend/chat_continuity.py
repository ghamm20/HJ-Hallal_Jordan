"""Conservative session continuity helpers for follow-up research turns."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.reasoning.intent_router import route_query_intent

FOLLOWUP_TERMS = {
    "that",
    "this",
    "it",
    "those",
    "them",
    "same mode",
    "same framing",
    "same format",
    "first view",
    "second view",
    "that source",
    "the source",
    "primary texts",
    "strip out commentary",
    "what part is strongest",
    "which part is strongest",
    "what is uncertain",
    "which part is uncertain",
    "keep the same mode",
    "keep same mode",
    "same mode as before",
    "now compare",
    "now give me",
    "show only",
    "summarize the disagreement",
    "plain english",
}

SOURCE_ONLY_TERMS = {
    "only primary source",
    "only primary texts",
    "only the primary texts",
    "primary source text",
    "primary source texts",
    "no synthesis",
    "strip out commentary",
    "show only the sources",
    "show only sources",
    "sources only",
    "source only",
}

COMPARE_TERMS = {
    "compare",
    "compare it",
    "compare that",
    "other view",
    "other views",
    "shafi'i view",
    "shafii view",
    "hanafi and shafi",
    "versus",
    "vs",
}

MODE_PRESERVE_TERMS = {
    "keep the same mode",
    "keep same mode",
    "same mode as before",
    "keep the same framing",
    "same framing",
}
SUMMARY_TERMS = {
    "summarize",
    "summary",
    "plain english",
    "summarize the disagreement",
}

PRIMARY_ONLY_TERMS = {
    "only primary texts",
    "only the primary texts",
    "only primary source",
    "only primary source text",
    "only the sources",
    "show only sources",
    "show only the sources",
    "strip out commentary",
}

SOURCE_REFERENCE_TERMS = {
    "that source",
    "the source",
    "that text",
    "that citation",
}

VIEW_REFERENCE_TERMS = {
    "first view",
    "second view",
    "hanafi side",
    "shafii side",
    "shafi'i side",
}

EXPLICIT_MADHHAB_TERMS = {
    "hanafi": "hanafi",
    "shafii": "shafii",
    "shafi'i": "shafii",
    "maliki": "maliki",
    "hanbali": "hanbali",
}


@dataclass(slots=True)
class SessionTurnSummary:
    request_id: str
    created_at: str
    expires_at: str
    original_question: str
    effective_question: str
    selected_madhhab: str
    answer_mode: str
    intent_id: str
    topic_anchor: str
    source_focus_label: str | None = None
    position_labels: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    source_classifications: list[str] = field(default_factory=list)
    source_families: list[str] = field(default_factory=list)
    mode_hooks: list[str] = field(default_factory=list)
    continuity_used: bool = False
    grounded_source_count: int = 0


@dataclass(slots=True)
class ContinuityResolution:
    used: bool
    reason: str
    effective_question: str
    effective_answer_mode: str
    effective_selected_madhhab: str
    prior_request_id: str | None = None
    topic_anchor: str | None = None
    source_focus_label: str | None = None
    position_focus_label: str | None = None
    mode_hooks: list[str] = field(default_factory=list)
    continuity_source: str = "none"

    def as_debug_payload(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "reason": self.reason,
            "prior_request_id": self.prior_request_id,
            "topic_anchor": self.topic_anchor,
            "source_focus_label": self.source_focus_label,
            "position_focus_label": self.position_focus_label,
            "effective_question": self.effective_question,
            "effective_answer_mode": self.effective_answer_mode,
            "effective_selected_madhhab": self.effective_selected_madhhab,
            "mode_hooks": list(self.mode_hooks),
            "continuity_source": self.continuity_source,
        }


class SessionContinuityStore:
    """Track the last resolved turn for each chat session."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        max_sessions: int = 200,
        ttl_seconds: int = 3600,
    ) -> None:
        self._lock = threading.RLock()
        self._turns: dict[str, SessionTurnSummary] = {}
        self._max_sessions = max_sessions
        self._ttl_seconds = max(1, ttl_seconds)
        self._db_path = db_path
        self._available = False

    def ensure_ready(self) -> None:
        if self._db_path is None:
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_turns (
                        session_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        original_question TEXT NOT NULL,
                        effective_question TEXT NOT NULL,
                        selected_madhhab TEXT NOT NULL,
                        answer_mode TEXT NOT NULL,
                        intent_id TEXT NOT NULL,
                        topic_anchor TEXT NOT NULL,
                        source_focus_label TEXT,
                        position_labels_json TEXT NOT NULL,
                        source_titles_json TEXT NOT NULL,
                        source_classifications_json TEXT NOT NULL,
                        source_families_json TEXT NOT NULL,
                        mode_hooks_json TEXT NOT NULL,
                        continuity_used INTEGER NOT NULL DEFAULT 0,
                        grounded_source_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.commit()
            self._available = True
            self.prune_expired()
        except Exception:
            self._available = False

    def set_ttl_seconds(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(1, ttl_seconds)

    def prune_expired(self) -> None:
        if not self._available:
            return
        now = _utc_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM session_turns WHERE expires_at <= ?",
                    (now,),
                )
                connection.commit()
        except Exception:
            self._available = False

    def resolve(
        self,
        *,
        session_id: str,
        question: str,
        selected_madhhab: str,
        answer_mode: str,
    ) -> ContinuityResolution:
        normalized_question = _normalize(question)
        self.prune_expired()
        with self._lock:
            prior = self._turns.get(session_id)
        prior_source = "memory"
        if prior and _is_expired(prior.expires_at):
            with self._lock:
                self._turns.pop(session_id, None)
            prior = None
        if not prior:
            prior = self._load_persisted_turn(session_id)
            prior_source = "persistent_store"
        if not prior:
            return ContinuityResolution(
                used=False,
                reason="no_prior_session_turn",
                effective_question=question,
                effective_answer_mode=answer_mode,
                effective_selected_madhhab=selected_madhhab,
            )
        if not _looks_like_followup(normalized_question):
            return ContinuityResolution(
                used=False,
                reason="question_appears_standalone",
                effective_question=question,
                effective_answer_mode=answer_mode,
                effective_selected_madhhab=selected_madhhab,
            )
        if prior.grounded_source_count <= 0 and not (
            prior.continuity_used and prior.topic_anchor
        ):
            return ContinuityResolution(
                used=False,
                reason="prior_turn_not_grounded",
                effective_question=question,
                effective_answer_mode=answer_mode,
                effective_selected_madhhab=selected_madhhab,
            )

        effective_mode = answer_mode
        effective_selected_madhhab = selected_madhhab
        mode_hooks: list[str] = []
        compare_requested = _contains_any_phrase(normalized_question, COMPARE_TERMS)

        explicit_madhhab = _extract_explicit_madhhab(normalized_question)
        if explicit_madhhab:
            if (
                compare_requested
                and prior.selected_madhhab not in {"", "not_specified", "compare_all"}
            ):
                effective_selected_madhhab = prior.selected_madhhab
                mode_hooks.append(f"comparison_target:{explicit_madhhab}")
            else:
                effective_selected_madhhab = explicit_madhhab
                mode_hooks.append(f"madhhab_focus:{explicit_madhhab}")
        elif selected_madhhab == "not_specified" and prior.selected_madhhab not in {
            "",
            "not_specified",
            "compare_all",
        }:
            effective_selected_madhhab = prior.selected_madhhab
            mode_hooks.append("carry_forward_madhhab")

        if _contains_any_phrase(normalized_question, SOURCE_ONLY_TERMS):
            effective_mode = "source_only"
            mode_hooks.append("source_only_override")
        elif compare_requested:
            effective_mode = "compare_views"
            mode_hooks.append("compare_views_override")
        elif _contains_any_phrase(normalized_question, SUMMARY_TERMS):
            effective_mode = answer_mode
            mode_hooks.append("summary_override")
        elif _contains_any_phrase(normalized_question, MODE_PRESERVE_TERMS):
            effective_mode = prior.answer_mode
            mode_hooks.append("preserve_prior_mode")
        elif _is_short_anchored_followup(normalized_question):
            effective_mode = prior.answer_mode
            mode_hooks.append("carry_forward_mode")

        source_focus_label = (
            prior.source_focus_label
            if _contains_any_phrase(normalized_question, SOURCE_REFERENCE_TERMS)
            else None
        )
        position_focus_label = None
        if _contains_any_phrase(normalized_question, {"first view"}) and prior.position_labels:
            position_focus_label = prior.position_labels[0]
        elif _contains_any_phrase(normalized_question, {"second view"}) and len(prior.position_labels) > 1:
            position_focus_label = prior.position_labels[1]
        elif explicit_madhhab and prior.position_labels:
            for label in prior.position_labels:
                if explicit_madhhab in label.casefold():
                    position_focus_label = label
                    break

        effective_question = _compose_followup_question(
            topic_anchor=prior.topic_anchor,
            followup_question=question,
            source_focus_label=source_focus_label,
            position_focus_label=position_focus_label,
        )
        return ContinuityResolution(
            used=True,
            reason="anchored_followup",
            prior_request_id=prior.request_id,
            topic_anchor=prior.topic_anchor,
            source_focus_label=source_focus_label,
            position_focus_label=position_focus_label,
            effective_question=effective_question,
            effective_answer_mode=effective_mode,
            effective_selected_madhhab=effective_selected_madhhab,
            mode_hooks=mode_hooks,
            continuity_source=prior_source,
        )

    def record_turn(
        self,
        *,
        session_id: str,
        request_id: str,
        original_question: str,
        effective_question: str,
        selected_madhhab: str,
        answer_mode: str,
        snippets: list[dict[str, Any]],
        continuity: ContinuityResolution,
    ) -> dict[str, Any]:
        created_at = _utc_now()
        expires_at = _future_utc(seconds=self._ttl_seconds)
        intent = route_query_intent(
            question=effective_question,
            answer_mode=answer_mode,
            selected_madhhab=selected_madhhab,
        )
        topic_anchor = continuity.topic_anchor or original_question.strip() or effective_question.strip()
        summary = SessionTurnSummary(
            request_id=request_id,
            created_at=created_at,
            expires_at=expires_at,
            original_question=original_question,
            effective_question=effective_question,
            selected_madhhab=selected_madhhab,
            answer_mode=answer_mode,
            intent_id=intent.intent_id,
            topic_anchor=topic_anchor,
            source_focus_label=_primary_source_label(snippets),
            position_labels=_position_labels(snippets),
            source_titles=_top_unique_values(snippets, "title", limit=4),
            source_classifications=_top_unique_values(
                snippets, "source_classification", fallback_field="source_type", limit=4
            ),
            source_families=_top_unique_values(snippets, "source_family", limit=4),
            mode_hooks=list(continuity.mode_hooks),
            continuity_used=continuity.used,
            grounded_source_count=len(snippets),
        )
        with self._lock:
            if len(self._turns) >= self._max_sessions and session_id not in self._turns:
                oldest_key = next(iter(self._turns))
                self._turns.pop(oldest_key, None)
            self._turns[session_id] = summary
        self._persist_turn(session_id=session_id, summary=summary)
        return asdict(summary)

    def get(self, *, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            summary = self._turns.get(session_id)
        if summary:
            return asdict(summary)
        persisted = self._load_persisted_turn(session_id)
        return asdict(persisted) if persisted else None

    def _persist_turn(self, *, session_id: str, summary: SessionTurnSummary) -> None:
        if not self._available:
            return
        payload = asdict(summary)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO session_turns (
                        session_id, request_id, created_at, expires_at,
                        original_question, effective_question, selected_madhhab,
                        answer_mode, intent_id, topic_anchor, source_focus_label,
                        position_labels_json, source_titles_json, source_classifications_json,
                        source_families_json, mode_hooks_json, continuity_used,
                        grounded_source_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        request_id=excluded.request_id,
                        created_at=excluded.created_at,
                        expires_at=excluded.expires_at,
                        original_question=excluded.original_question,
                        effective_question=excluded.effective_question,
                        selected_madhhab=excluded.selected_madhhab,
                        answer_mode=excluded.answer_mode,
                        intent_id=excluded.intent_id,
                        topic_anchor=excluded.topic_anchor,
                        source_focus_label=excluded.source_focus_label,
                        position_labels_json=excluded.position_labels_json,
                        source_titles_json=excluded.source_titles_json,
                        source_classifications_json=excluded.source_classifications_json,
                        source_families_json=excluded.source_families_json,
                        mode_hooks_json=excluded.mode_hooks_json,
                        continuity_used=excluded.continuity_used,
                        grounded_source_count=excluded.grounded_source_count
                    """,
                    (
                        session_id,
                        summary.request_id,
                        summary.created_at,
                        summary.expires_at,
                        summary.original_question,
                        summary.effective_question,
                        summary.selected_madhhab,
                        summary.answer_mode,
                        summary.intent_id,
                        summary.topic_anchor,
                        summary.source_focus_label,
                        json.dumps(summary.position_labels, ensure_ascii=False),
                        json.dumps(summary.source_titles, ensure_ascii=False),
                        json.dumps(summary.source_classifications, ensure_ascii=False),
                        json.dumps(summary.source_families, ensure_ascii=False),
                        json.dumps(summary.mode_hooks, ensure_ascii=False),
                        int(summary.continuity_used),
                        summary.grounded_source_count,
                    ),
                )
                connection.commit()
        except Exception:
            self._available = False

    def _load_persisted_turn(self, session_id: str) -> SessionTurnSummary | None:
        if not self._available:
            return None
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM session_turns
                    WHERE session_id = ?
                      AND expires_at > ?
                    LIMIT 1
                    """,
                    (session_id, _utc_now()),
                ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            summary = SessionTurnSummary(
                request_id=str(payload["request_id"]),
                created_at=str(payload["created_at"]),
                expires_at=str(payload["expires_at"]),
                original_question=str(payload["original_question"]),
                effective_question=str(payload["effective_question"]),
                selected_madhhab=str(payload["selected_madhhab"]),
                answer_mode=str(payload["answer_mode"]),
                intent_id=str(payload["intent_id"]),
                topic_anchor=str(payload["topic_anchor"]),
                source_focus_label=_nullable_text(payload.get("source_focus_label")),
                position_labels=_decode_json_list(payload.get("position_labels_json")),
                source_titles=_decode_json_list(payload.get("source_titles_json")),
                source_classifications=_decode_json_list(
                    payload.get("source_classifications_json")
                ),
                source_families=_decode_json_list(payload.get("source_families_json")),
                mode_hooks=_decode_json_list(payload.get("mode_hooks_json")),
                continuity_used=bool(payload.get("continuity_used")),
                grounded_source_count=int(payload.get("grounded_source_count") or 0),
            )
            with self._lock:
                self._turns[session_id] = summary
            return summary
        except Exception:
            self._available = False
            return None

    @contextmanager
    def _connect(self):
        if self._db_path is None:
            raise RuntimeError("session continuity store has no database path")
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _looks_like_followup(normalized_question: str) -> bool:
    if not normalized_question:
        return False
    if _contains_any_phrase(normalized_question, FOLLOWUP_TERMS):
        return True
    if normalized_question.startswith("now "):
        return True
    return bool(re.search(r"\b(that|this|it|them|those)\b", normalized_question))


def _is_short_anchored_followup(normalized_question: str) -> bool:
    tokens = re.findall(r"\w+", normalized_question)
    if len(tokens) > 12:
        return False
    return _looks_like_followup(normalized_question)


def _extract_explicit_madhhab(normalized_question: str) -> str | None:
    for key, value in EXPLICIT_MADHHAB_TERMS.items():
        if key in normalized_question:
            return value
    return None


def _contains_any_phrase(text: str, values: set[str]) -> bool:
    return any(_contains_phrase(text, value) for value in values)


def _contains_phrase(text: str, value: str) -> bool:
    pattern = r"\b" + re.escape(value).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


def _compose_followup_question(
    *,
    topic_anchor: str,
    followup_question: str,
    source_focus_label: str | None,
    position_focus_label: str | None,
) -> str:
    focus_parts: list[str] = []
    if source_focus_label:
        focus_parts.append(f"Focus source: {source_focus_label}.")
    if position_focus_label:
        focus_parts.append(f"Focus position: {position_focus_label}.")
    focus_prefix = " ".join(focus_parts).strip()
    if focus_prefix:
        return f"{topic_anchor} {focus_prefix} Follow-up: {followup_question}".strip()
    return f"{topic_anchor} Follow-up: {followup_question}".strip()


def _primary_source_label(snippets: list[dict[str, Any]]) -> str | None:
    if not snippets:
        return None
    first = snippets[0]
    title = str(first.get("title", "") or "").strip()
    reference = str(first.get("reference", "") or "").strip()
    if title and reference:
        return f"{title} ({reference})"
    return title or reference or None


def _position_labels(snippets: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for snippet in snippets:
        madhhab = str(snippet.get("madhhab", "") or "").strip().lower()
        if not madhhab:
            continue
        label = f"{madhhab.title()} Position"
        if label not in labels:
            labels.append(label)
    return labels


def _top_unique_values(
    snippets: list[dict[str, Any]],
    field_name: str,
    *,
    fallback_field: str | None = None,
    limit: int,
) -> list[str]:
    values: list[str] = []
    for snippet in snippets:
        value = str(snippet.get(field_name, "") or "").strip()
        if not value and fallback_field:
            value = str(snippet.get(fallback_field, "") or "").strip()
        if not value or value in values:
            continue
        values.append(value)
        if len(values) >= limit:
            break
    return values


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _decode_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _nullable_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _future_utc(*, seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _is_expired(expires_at: str) -> bool:
    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    return expiry <= datetime.now(UTC)
