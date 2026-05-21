"""Local user auth, chat history, project state, and lightweight memory."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class UserWorkspaceStore:
    """Persist local users, auth sessions, chats, projects, and safe memory."""

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
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        default_madhhab TEXT NOT NULL,
                        default_answer_mode TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_login_at TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        project_id INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(id),
                        FOREIGN KEY(project_id) REFERENCES study_projects(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        answer_mode TEXT,
                        madhhab TEXT,
                        evidence_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS study_projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        madhhab TEXT NOT NULL,
                        study_mode TEXT NOT NULL DEFAULT 'standard',
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_session_id INTEGER,
                        confidence REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(id),
                        FOREIGN KEY(source_session_id) REFERENCES chat_sessions(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        token_hash TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                    """
                )
                self._ensure_column(
                    connection,
                    table_name="study_projects",
                    column_name="study_mode",
                    column_sql="TEXT NOT NULL DEFAULT 'standard'",
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS study_project_state (
                        project_id INTEGER PRIMARY KEY,
                        topic TEXT,
                        overview TEXT,
                        selected_madhhab TEXT,
                        source_layers_json TEXT,
                        reading_list_json TEXT,
                        lesson_path_json TEXT,
                        key_terms_json TEXT,
                        what_to_avoid_json TEXT,
                        corpus_gaps_json TEXT,
                        missing_layers_json TEXT,
                        progress_json TEXT,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(project_id) REFERENCES study_projects(id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    DROP INDEX IF EXISTS idx_user_memory_unique_preference
                    """
                )
                cursor.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memory_unique_non_note
                    ON user_memory(user_id, memory_type, content)
                    WHERE memory_type IN ('preference', 'project_context')
                    """
                )
                connection.commit()
            self.available = True
            self.last_error = None
        except Exception as exc:  # pragma: no cover - defensive
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"

    def register_user(
        self,
        *,
        username: str,
        password: str,
        display_name: str,
        default_madhhab: str,
        default_answer_mode: str,
    ) -> dict[str, Any]:
        normalized_username = self._normalize_username(username)
        if not normalized_username:
            raise ValueError("username must not be empty")
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        display = str(display_name or "").strip() or normalized_username
        now = _utc_now()
        password_hash = _hash_password(password)
        try:
            with self._connect() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, display_name,
                        default_madhhab, default_answer_mode,
                        created_at, last_login_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_username,
                        password_hash,
                        display,
                        default_madhhab,
                        default_answer_mode,
                        now,
                        None,
                    ),
                )
                connection.commit()
                user_id = int(cursor.lastrowid)
            return self.get_user_by_id(user_id) or {}
        except sqlite3.IntegrityError as exc:
            raise ValueError("username is already taken") from exc

    def authenticate_user(self, *, username: str, password: str) -> dict[str, Any] | None:
        normalized_username = self._normalize_username(username)
        row = self._fetch_one(
            "SELECT * FROM users WHERE username = ?",
            (normalized_username,),
        )
        if not row:
            return None
        if not _verify_password(password, str(row["password_hash"])):
            return None
        now = _utc_now()
        self._write(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (now, int(row["id"])),
        )
        return self.get_user_by_id(int(row["id"]))

    def create_auth_session(self, *, user_id: int, days_valid: int = 30) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = _hash_token(token)
        created_at = _utc_now()
        expires_at = (datetime.now(UTC) + timedelta(days=days_valid)).isoformat()
        self._write(
            """
            INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, token_hash, created_at, expires_at, created_at),
        )
        return token

    def revoke_auth_session(self, token: str) -> None:
        if not token:
            return
        self._write(
            "DELETE FROM auth_sessions WHERE token_hash = ?",
            (_hash_token(token),),
        )

    def get_user_for_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        now = datetime.now(UTC)
        row = self._fetch_one(
            """
            SELECT users.*
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
            """,
            (_hash_token(token),),
        )
        if not row:
            return None
        session_row = self._fetch_one(
            "SELECT * FROM auth_sessions WHERE token_hash = ?",
            (_hash_token(token),),
        )
        if not session_row:
            return None
        expires_at = _parse_time(str(session_row["expires_at"]))
        if expires_at is not None and expires_at <= now:
            self.revoke_auth_session(token)
            return None
        self._write(
            "UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?",
            (_utc_now(), _hash_token(token)),
        )
        return self._row_to_user(row)

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        row = self._fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if not row:
            return None
        return self._row_to_user(row)

    def update_user_defaults(
        self,
        *,
        user_id: int,
        display_name: str | None = None,
        default_madhhab: str | None = None,
        default_answer_mode: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_user_by_id(user_id)
        if not current:
            raise ValueError("unknown user")
        updated_display_name = str(display_name or current["display_name"]).strip() or current["display_name"]
        updated_madhhab = default_madhhab or current["default_madhhab"]
        updated_mode = default_answer_mode or current["default_answer_mode"]
        self._write(
            """
            UPDATE users
            SET display_name = ?, default_madhhab = ?, default_answer_mode = ?
            WHERE id = ?
            """,
            (updated_display_name, updated_madhhab, updated_mode, user_id),
        )
        self.remember_preference(
            user_id=user_id,
            key="default_madhhab",
            value=updated_madhhab,
            confidence=0.95,
        )
        self.remember_preference(
            user_id=user_id,
            key="default_answer_mode",
            value=updated_mode,
            confidence=0.95,
        )
        return self.get_user_by_id(user_id) or {}

    def create_chat_session(
        self,
        *,
        user_id: int,
        title: str,
        project_id: int | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        clean_title = str(title or "").strip() or "New Chat"
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO chat_sessions (user_id, title, project_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, clean_title, project_id, now, now),
            )
            connection.commit()
            session_id = int(cursor.lastrowid)
        return self.get_chat_session(user_id=user_id, session_id=session_id) or {}

    def list_chat_sessions(self, *, user_id: int, project_id: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT chat_sessions.*,
                   (
                     SELECT content
                     FROM chat_messages
                     WHERE chat_messages.session_id = chat_sessions.id
                     ORDER BY chat_messages.id DESC
                     LIMIT 1
                   ) AS latest_message
            FROM chat_sessions
            WHERE user_id = ?
        """
        params: list[Any] = [user_id]
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        query += " ORDER BY updated_at DESC, id DESC"
        rows = self._fetch_all(query, params)
        return [self._row_to_chat_session(row) for row in rows]

    def get_chat_session(self, *, user_id: int, session_id: int) -> dict[str, Any] | None:
        row = self._fetch_one(
            "SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        if not row:
            return None
        session = self._row_to_chat_session(row)
        session["messages"] = self.list_chat_messages(user_id=user_id, session_id=session_id)
        return session

    def rename_chat_session(self, *, user_id: int, session_id: int, title: str) -> None:
        clean_title = str(title or "").strip()
        if not clean_title:
            return
        self._write(
            """
            UPDATE chat_sessions
            SET title = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (clean_title, _utc_now(), session_id, user_id),
        )

    def attach_chat_to_project(self, *, user_id: int, session_id: int, project_id: int | None) -> None:
        self._write(
            """
            UPDATE chat_sessions
            SET project_id = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (project_id, _utc_now(), session_id, user_id),
        )

    def append_chat_message(
        self,
        *,
        session_id: int,
        role: str,
        content: str,
        answer_mode: str | None,
        madhhab: str | None,
        evidence_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        payload = json.dumps(evidence_json, ensure_ascii=False) if evidence_json is not None else None
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO chat_messages (
                    session_id, role, content, answer_mode, madhhab, evidence_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, answer_mode, madhhab, payload, now),
            )
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            connection.commit()
            message_id = int(cursor.lastrowid)
        row = self._fetch_one("SELECT * FROM chat_messages WHERE id = ?", (message_id,))
        return self._row_to_chat_message(row) if row else {}

    def list_chat_messages(self, *, user_id: int, session_id: int) -> list[dict[str, Any]]:
        owned = self._fetch_one(
            "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        if not owned:
            return []
        rows = self._fetch_all(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return [self._row_to_chat_message(row) for row in rows]

    def create_project(
        self,
        *,
        user_id: int,
        title: str,
        description: str,
        madhhab: str,
        study_mode: str = "standard",
        status: str = "active",
    ) -> dict[str, Any]:
        now = _utc_now()
        clean_title = str(title or "").strip()
        if not clean_title:
            raise ValueError("project title must not be empty")
        clean_description = str(description or "").strip()
        clean_study_mode = _normalize_study_mode(study_mode)
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO study_projects (
                    user_id, title, description, madhhab, study_mode, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    clean_title,
                    clean_description,
                    madhhab,
                    clean_study_mode,
                    status,
                    now,
                    now,
                ),
            )
            connection.commit()
            project_id = int(cursor.lastrowid)
        project = self.get_project(user_id=user_id, project_id=project_id) or {}
        self.remember_project_context(
            user_id=user_id,
            project=project,
            confidence=0.9,
        )
        return project

    def list_projects(self, *, user_id: int) -> list[dict[str, Any]]:
        rows = self._fetch_all(
            """
            SELECT study_projects.*,
                   (
                     SELECT COUNT(1)
                     FROM chat_sessions
                     WHERE chat_sessions.project_id = study_projects.id
                   ) AS chat_count
            FROM study_projects
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id,),
        )
        return [self._row_to_project(row) for row in rows]

    def get_project(self, *, user_id: int, project_id: int) -> dict[str, Any] | None:
        row = self._fetch_one(
            """
            SELECT study_projects.*,
                   (
                     SELECT COUNT(1)
                     FROM chat_sessions
                     WHERE chat_sessions.project_id = study_projects.id
                   ) AS chat_count
            FROM study_projects
            WHERE study_projects.id = ? AND study_projects.user_id = ?
            """,
            (project_id, user_id),
        )
        if not row:
            return None
        return self._row_to_project(row)

    def save_project_study_state(
        self,
        *,
        user_id: int,
        project_id: int,
        answer: dict[str, Any],
    ) -> dict[str, Any]:
        project = self.get_project(user_id=user_id, project_id=project_id)
        if not project:
            raise ValueError("unknown project")
        if str(answer.get("answer_mode") or "").strip() != "study_path":
            return project
        now = _utc_now()
        existing_state = self._load_project_study_state(project_id=project_id) or {}
        lesson_path = answer.get("lesson_path") if isinstance(answer.get("lesson_path"), list) else []
        progress = self._normalize_project_progress(
            existing_progress=existing_state.get("progress"),
            lesson_path=lesson_path,
        )
        state_payload = {
            "topic": str(answer.get("topic") or "").strip(),
            "overview": str(
                answer.get("overview") or answer.get("direct_answer") or ""
            ).strip(),
            "selected_madhhab": str(answer.get("selected_madhhab") or project.get("madhhab") or "").strip(),
            "source_layers_json": _json_dump(answer.get("source_layers", {})),
            "reading_list_json": _json_dump(answer.get("reading_list", {})),
            "lesson_path_json": _json_dump(lesson_path),
            "key_terms_json": _json_dump(answer.get("key_terms", [])),
            "what_to_avoid_json": _json_dump(answer.get("what_to_avoid", [])),
            "corpus_gaps_json": _json_dump(answer.get("corpus_gaps", [])),
            "missing_layers_json": _json_dump(_derive_missing_layers(answer)),
            "progress_json": _json_dump(progress),
            "updated_at": now,
        }
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO study_project_state (
                    project_id, topic, overview, selected_madhhab, source_layers_json,
                    reading_list_json, lesson_path_json, key_terms_json, what_to_avoid_json,
                    corpus_gaps_json, missing_layers_json, progress_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    topic = excluded.topic,
                    overview = excluded.overview,
                    selected_madhhab = excluded.selected_madhhab,
                    source_layers_json = excluded.source_layers_json,
                    reading_list_json = excluded.reading_list_json,
                    lesson_path_json = excluded.lesson_path_json,
                    key_terms_json = excluded.key_terms_json,
                    what_to_avoid_json = excluded.what_to_avoid_json,
                    corpus_gaps_json = excluded.corpus_gaps_json,
                    missing_layers_json = excluded.missing_layers_json,
                    progress_json = excluded.progress_json,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    state_payload["topic"],
                    state_payload["overview"],
                    state_payload["selected_madhhab"],
                    state_payload["source_layers_json"],
                    state_payload["reading_list_json"],
                    state_payload["lesson_path_json"],
                    state_payload["key_terms_json"],
                    state_payload["what_to_avoid_json"],
                    state_payload["corpus_gaps_json"],
                    state_payload["missing_layers_json"],
                    state_payload["progress_json"],
                    state_payload["updated_at"],
                ),
            )
            cursor.execute(
                "UPDATE study_projects SET updated_at = ? WHERE id = ? AND user_id = ?",
                (now, project_id, user_id),
            )
            connection.commit()
        refreshed = self.get_project(user_id=user_id, project_id=project_id) or {}
        self.remember_project_context(
            user_id=user_id,
            project=refreshed,
            confidence=0.92,
        )
        return refreshed

    def update_project_study_progress(
        self,
        *,
        user_id: int,
        project_id: int,
        active_lesson_index: int | None = None,
        completed_lesson_index: int | None = None,
        advance: bool = False,
    ) -> dict[str, Any]:
        project = self.get_project(user_id=user_id, project_id=project_id)
        if not project:
            raise ValueError("unknown project")
        state = self._load_project_study_state(project_id=project_id)
        if not state:
            raise ValueError("study project has no saved lesson state yet")
        lesson_path = state.get("lesson_path") if isinstance(state.get("lesson_path"), list) else []
        progress = self._normalize_project_progress(
            existing_progress=state.get("progress"),
            lesson_path=lesson_path,
        )
        lesson_count = len(lesson_path)
        completed_indices = set(progress.get("completed_indices", []))
        if completed_lesson_index is not None and 0 <= completed_lesson_index < lesson_count:
            completed_indices.add(int(completed_lesson_index))
        progress["completed_indices"] = sorted(completed_indices)
        if active_lesson_index is not None and 0 <= active_lesson_index < lesson_count:
            progress["active_index"] = int(active_lesson_index)
        if advance:
            progress["active_index"] = self._next_lesson_index(
                lesson_count=lesson_count,
                completed_indices=progress["completed_indices"],
                current_index=int(progress.get("active_index") or 0),
            )
        elif completed_lesson_index is not None and progress.get("active_index") == int(completed_lesson_index):
            progress["active_index"] = self._next_lesson_index(
                lesson_count=lesson_count,
                completed_indices=progress["completed_indices"],
                current_index=int(progress.get("active_index") or 0),
            )
        progress = self._normalize_project_progress(
            existing_progress=progress,
            lesson_path=lesson_path,
        )
        now = _utc_now()
        self._write(
            """
            UPDATE study_project_state
            SET progress_json = ?, updated_at = ?
            WHERE project_id = ?
            """,
            (_json_dump(progress), now, project_id),
        )
        self._write(
            "UPDATE study_projects SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now, project_id, user_id),
        )
        refreshed = self.get_project(user_id=user_id, project_id=project_id) or {}
        self.remember_project_context(
            user_id=user_id,
            project=refreshed,
            confidence=0.9,
        )
        return refreshed

    def remember_preference(
        self,
        *,
        user_id: int,
        key: str,
        value: str,
        source_session_id: int | None = None,
        confidence: float = 0.9,
    ) -> None:
        payload = json.dumps({"key": key, "value": value}, ensure_ascii=False)
        self._upsert_memory(
            user_id=user_id,
            memory_type="preference",
            content=payload,
            source_session_id=source_session_id,
            confidence=confidence,
        )

    def remember_project_context(
        self,
        *,
        user_id: int,
        project: dict[str, Any],
        confidence: float = 0.88,
    ) -> None:
        payload = json.dumps(
            {
                "project_id": project.get("id"),
                "title": project.get("title"),
                "description": project.get("description"),
                "madhhab": project.get("madhhab"),
                "study_mode": project.get("study_mode"),
                "status": project.get("status"),
                "study_state": _project_context_snapshot(project),
            },
            ensure_ascii=False,
        )
        self._upsert_memory(
            user_id=user_id,
            memory_type="project_context",
            content=payload,
            source_session_id=None,
            confidence=confidence,
        )

    def add_note_memory(
        self,
        *,
        user_id: int,
        content: str,
        source_session_id: int | None = None,
        confidence: float = 0.6,
    ) -> dict[str, Any]:
        now = _utc_now()
        note = str(content or "").strip()
        if not note:
            raise ValueError("note must not be empty")
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO user_memory (
                    user_id, memory_type, content, source_session_id, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, "note", note, source_session_id, confidence, now, now),
            )
            connection.commit()
            memory_id = int(cursor.lastrowid)
        row = self._fetch_one("SELECT * FROM user_memory WHERE id = ?", (memory_id,))
        return self._row_to_memory(row) if row else {}

    def list_memory(self, *, user_id: int) -> list[dict[str, Any]]:
        rows = self._fetch_all(
            "SELECT * FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
            (user_id,),
        )
        return [self._row_to_memory(row) for row in rows]

    def build_prompt_context(
        self,
        *,
        user_id: int,
        project_id: int | None = None,
    ) -> dict[str, Any]:
        user = self.get_user_by_id(user_id)
        if not user:
            return {}
        memory_rows = self.list_memory(user_id=user_id)
        preferences: list[dict[str, Any]] = []
        notes: list[str] = []
        project_context: dict[str, Any] | None = None
        for row in memory_rows:
            if row["memory_type"] == "preference":
                payload = _decode_json(row["content"])
                if isinstance(payload, dict) and payload.get("key") and payload.get("value"):
                    preferences.append(
                        {
                            "key": payload["key"],
                            "value": payload["value"],
                            "confidence": row["confidence"],
                        }
                    )
            elif row["memory_type"] == "project_context":
                if project_context is not None:
                    continue
                payload = _decode_json(row["content"])
                if not isinstance(payload, dict):
                    continue
                if project_id is not None and int(payload.get("project_id") or 0) != int(project_id):
                    continue
                project_context = payload
            elif row["memory_type"] == "note":
                notes.append(str(row["content"]))
        if project_id is not None and project_context is None:
            project_context = self.get_project(user_id=user_id, project_id=project_id)
        if isinstance(project_context, dict):
            project_context = {
                **project_context,
                "study_state": _project_context_snapshot(project_context),
            }
        return {
            "user_profile": {
                "display_name": user["display_name"],
                "default_madhhab": user["default_madhhab"],
                "default_answer_mode": user["default_answer_mode"],
            },
            "preferences": preferences[:6],
            "project_context": project_context,
            "notes": notes[:4],
        }

    def sidebar_state(self, *, user_id: int) -> dict[str, Any]:
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("unknown user")
        projects = self.list_projects(user_id=user_id)
        chats = self.list_chat_sessions(user_id=user_id)
        return {
            "user": user,
            "projects": projects,
            "recent_chats": chats[:20],
            "memory": self.list_memory(user_id=user_id)[:12],
        }

    def auto_title_for_question(self, question: str) -> str:
        clean = " ".join(str(question or "").strip().split())
        if not clean:
            return "New Chat"
        if len(clean) <= 72:
            return clean
        return clean[:69].rstrip() + "..."

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def _write(self, query: str, params: tuple[Any, ...] | list[Any]) -> None:
        with self._connect() as connection:
            connection.execute(query, params)
            connection.commit()

    def _fetch_one(self, query: str, params: tuple[Any, ...] | list[Any]) -> sqlite3.Row | None:
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return row

    def _fetch_all(self, query: str, params: tuple[Any, ...] | list[Any]) -> list[sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return rows

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        *,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in columns:
            return
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
        )

    def _upsert_memory(
        self,
        *,
        user_id: int,
        memory_type: str,
        content: str,
        source_session_id: int | None,
        confidence: float,
    ) -> None:
        now = _utc_now()
        existing = self._fetch_one(
            """
            SELECT * FROM user_memory
            WHERE user_id = ? AND memory_type = ? AND content = ?
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, memory_type, content),
        )
        if existing:
            self._write(
                """
                UPDATE user_memory
                SET source_session_id = ?, confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (source_session_id, confidence, now, int(existing["id"])),
            )
            return
        self._write(
            """
            INSERT INTO user_memory (
                user_id, memory_type, content, source_session_id, confidence, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, memory_type, content, source_session_id, confidence, now, now),
        )

    def _normalize_username(self, username: str) -> str:
        return str(username or "").strip().casefold()

    def _row_to_user(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "display_name": str(row["display_name"]),
            "default_madhhab": str(row["default_madhhab"]),
            "default_answer_mode": str(row["default_answer_mode"]),
            "created_at": str(row["created_at"]),
            "last_login_at": row["last_login_at"],
        }

    def _row_to_chat_session(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "user_id": int(row["user_id"]),
            "title": str(row["title"]),
            "project_id": int(row["project_id"]) if row["project_id"] is not None else None,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "latest_message": row["latest_message"] if "latest_message" in row.keys() else None,
        }

    def _row_to_chat_message(self, row: sqlite3.Row) -> dict[str, Any]:
        evidence = _decode_json(row["evidence_json"])
        return {
            "id": int(row["id"]),
            "session_id": int(row["session_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "answer_mode": row["answer_mode"],
            "madhhab": row["madhhab"],
            "evidence_json": evidence,
            "created_at": str(row["created_at"]),
        }

    def _row_to_project(self, row: sqlite3.Row) -> dict[str, Any]:
        project = {
            "id": int(row["id"]),
            "user_id": int(row["user_id"]),
            "title": str(row["title"]),
            "description": str(row["description"]),
            "madhhab": str(row["madhhab"]),
            "study_mode": _normalize_study_mode(row["study_mode"] if "study_mode" in row.keys() else "standard"),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "chat_count": int(row["chat_count"]) if "chat_count" in row.keys() and row["chat_count"] is not None else 0,
        }
        project["study_state"] = self._load_project_study_state(project_id=project["id"])
        return project

    def _row_to_memory(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "user_id": int(row["user_id"]),
            "memory_type": str(row["memory_type"]),
            "content": str(row["content"]),
            "source_session_id": int(row["source_session_id"]) if row["source_session_id"] is not None else None,
            "confidence": float(row["confidence"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _load_project_study_state(self, *, project_id: int) -> dict[str, Any] | None:
        row = self._fetch_one(
            "SELECT * FROM study_project_state WHERE project_id = ?",
            (project_id,),
        )
        if not row:
            return None
        lesson_path = _decode_json(row["lesson_path_json"])
        if not isinstance(lesson_path, list):
            lesson_path = []
        progress = self._normalize_project_progress(
            existing_progress=_decode_json(row["progress_json"]),
            lesson_path=lesson_path,
        )
        return {
            "topic": str(row["topic"] or "").strip(),
            "overview": str(row["overview"] or "").strip(),
            "selected_madhhab": str(row["selected_madhhab"] or "").strip(),
            "source_layers": _decode_json(row["source_layers_json"]) or {},
            "reading_list": _decode_json(row["reading_list_json"]) or {},
            "lesson_path": lesson_path,
            "key_terms": _decode_json(row["key_terms_json"]) or [],
            "what_to_avoid": _decode_json(row["what_to_avoid_json"]) or [],
            "corpus_gaps": _decode_json(row["corpus_gaps_json"]) or [],
            "missing_layers": _decode_json(row["missing_layers_json"]) or [],
            "progress": progress,
            "active_lesson": (
                lesson_path[progress["active_index"]]
                if lesson_path and progress["active_index"] is not None and 0 <= progress["active_index"] < len(lesson_path)
                else None
            ),
            "updated_at": str(row["updated_at"]),
        }

    def _normalize_project_progress(
        self,
        *,
        existing_progress: Any,
        lesson_path: list[dict[str, Any]],
    ) -> dict[str, Any]:
        lesson_count = len(lesson_path)
        payload = existing_progress if isinstance(existing_progress, dict) else {}
        completed_indices = sorted(
            {
                int(index)
                for index in payload.get("completed_indices", [])
                if isinstance(index, int) and 0 <= index < lesson_count
            }
        )
        if lesson_count == 0:
            return {
                "completed_indices": [],
                "completed_count": 0,
                "total_lessons": 0,
                "active_index": None,
                "next_lesson_index": None,
                "progress_percent": 0,
            }
        active_index_raw = payload.get("active_index")
        if not isinstance(active_index_raw, int) or not (0 <= active_index_raw < lesson_count):
            active_index_raw = self._next_lesson_index(
                lesson_count=lesson_count,
                completed_indices=completed_indices,
                current_index=0,
            )
        next_lesson_index = self._next_lesson_index(
            lesson_count=lesson_count,
            completed_indices=completed_indices,
            current_index=active_index_raw,
        )
        progress_percent = int(round((len(completed_indices) / lesson_count) * 100))
        return {
            "completed_indices": completed_indices,
            "completed_count": len(completed_indices),
            "total_lessons": lesson_count,
            "active_index": active_index_raw,
            "next_lesson_index": next_lesson_index,
            "progress_percent": progress_percent,
        }

    def _next_lesson_index(
        self,
        *,
        lesson_count: int,
        completed_indices: list[int],
        current_index: int,
    ) -> int | None:
        if lesson_count <= 0:
            return None
        completed = set(completed_indices)
        for index in range(max(current_index, 0), lesson_count):
            if index not in completed:
                return index
        for index in range(lesson_count):
            if index not in completed:
                return index
        return lesson_count - 1


def _hash_password(password: str) -> str:
    iterations = 120_000
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _decode_json(value: Any) -> Any:
    if value in {None, ""}:
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_study_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return "study_path" if normalized == "study_path" else "standard"


def _derive_missing_layers(answer: dict[str, Any]) -> list[str]:
    source_layers = answer.get("source_layers")
    if not isinstance(source_layers, dict):
        return []
    labels = {
        "quran": "Qur'an",
        "hadith": "Hadith",
        "fiqh": "Fiqh",
        "tasawwuf": "Tasawwuf",
        "scholar_commentary": "Scholar Commentary",
    }
    missing: list[str] = []
    for key, label in labels.items():
        items = source_layers.get(key)
        if not isinstance(items, list) or not items:
            missing.append(label)
    return missing


def _project_context_snapshot(project: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(project, dict):
        return None
    state = project.get("study_state")
    if not isinstance(state, dict):
        return None
    reading_list = state.get("reading_list")
    reading_titles: list[str] = []
    if isinstance(reading_list, dict):
        for group_items in reading_list.values():
            if not isinstance(group_items, list):
                continue
            for item in group_items[:2]:
                if isinstance(item, dict) and str(item.get("title") or "").strip():
                    reading_titles.append(str(item["title"]).strip())
    active_lesson = state.get("active_lesson") if isinstance(state.get("active_lesson"), dict) else {}
    progress = state.get("progress") if isinstance(state.get("progress"), dict) else {}
    return {
        "study_mode": project.get("study_mode"),
        "topic": state.get("topic"),
        "overview": state.get("overview"),
        "current_lesson_title": active_lesson.get("lesson_title"),
        "reading_titles": reading_titles[:4],
        "progress": {
            "completed_count": progress.get("completed_count", 0),
            "total_lessons": progress.get("total_lessons", 0),
            "progress_percent": progress.get("progress_percent", 0),
        },
    }


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
