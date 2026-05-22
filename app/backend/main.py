"""Canonical FastAPI entrypoint for the Halal Jordan runtime."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import requests
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from app.backend.ops import OpsService
from app.backend.permissions import (
    RoleContext,
    require_permission,
)
from app.reasoning.scholar_resolver import load_scholar_profiles
from app.reasoning.trust_engine import list_profiles_with_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ASSETS_DIR = REPO_ROOT / "app" / "frontend" / "assets"
AUTH_COOKIE_NAME = "hj_auth"


class ChatRequestModel(BaseModel):
    question: str
    selected_madhhab: str = "not_specified"
    answer_mode: str = "research"
    research_depth: str = "balanced_research"
    greeting_style: str | None = None
    tone_level: str | None = None
    ollama_model: str | None = None
    retrieved_sources: list[dict[str, Any]] | None = None
    chat_session_id: int | None = None
    project_id: int | None = None


class RegisterRequestModel(BaseModel):
    username: str
    password: str
    display_name: str | None = None
    default_madhhab: str = "hanafi"
    default_answer_mode: str = "research"


class LoginRequestModel(BaseModel):
    username: str
    password: str


class UserSettingsRequestModel(BaseModel):
    display_name: str | None = None
    default_madhhab: str | None = None
    default_answer_mode: str | None = None


class CreateChatSessionRequestModel(BaseModel):
    title: str | None = None
    project_id: int | None = None


class CreateProjectRequestModel(BaseModel):
    title: str
    description: str = ""
    madhhab: str = "hanafi"
    study_mode: str = "standard"
    status: str = "active"


class AttachChatProjectRequestModel(BaseModel):
    project_id: int | None = None


class MemoryNoteRequestModel(BaseModel):
    content: str
    source_session_id: int | None = None
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class ProjectStudyProgressRequestModel(BaseModel):
    active_lesson_index: int | None = Field(default=None, ge=0)
    completed_lesson_index: int | None = Field(default=None, ge=0)
    advance: bool = False


class UpdateSelectionRequestModel(BaseModel):
    update_id: str


def create_app(
    *,
    repo_root: Path | None = None,
    runtime_dir: Path | None = None,
    ops_service: OpsService | None = None,
) -> FastAPI:
    root = repo_root or REPO_ROOT
    ops = ops_service or OpsService(root, runtime_dir=runtime_dir)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ops = ops
        application.state.ops.startup()
        try:
            yield
        finally:
            application.state.ops.shutdown()

    app = FastAPI(title="Halal Jordan", version="0.1.0", lifespan=lifespan)
    app.state.ops = ops
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.get("/", response_class=HTMLResponse)
    async def root_ui() -> HTMLResponse:
        """The default landing page is now the simple /ask surface.

        Charter directive: "current build too much for average user — give
        me a button". The full chat/workspace UI lives at /workspace for
        users who need it; the default is now the one-question, one-button
        public front door.
        """

        return HTMLResponse(
            _ask_html(
                current_profile_id=str(
                    app.state.ops.config_store.load_effective().trust_profile_id
                    or "default"
                ),
            )
        )

    @app.get("/workspace", response_class=HTMLResponse)
    async def workspace_ui() -> HTMLResponse:
        return HTMLResponse(_manager_html(repo_root=root))

    @app.get("/login", response_class=HTMLResponse)
    async def login_ui() -> HTMLResponse:
        return HTMLResponse(_manager_html(repo_root=root, initial_auth_view="login"))

    @app.get("/register", response_class=HTMLResponse)
    async def register_ui() -> HTMLResponse:
        return HTMLResponse(_manager_html(repo_root=root, initial_auth_view="register"))

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_ui() -> HTMLResponse:
        return HTMLResponse(_admin_html())

    @app.get("/profiles", response_class=HTMLResponse)
    async def profiles_ui() -> HTMLResponse:
        """One-page button selector for the active reasoning profile.

        Public on purpose: switching the active trust/scholar methodology
        profile is a low-risk, transparent operation that the charter
        wants accessible — "a button" — for the average user.
        """

        return HTMLResponse(
            _profiles_html(
                profiles=list_profiles_with_metadata(repo_root=root),
                current_profile_id=str(
                    app.state.ops.config_store.load_effective().trust_profile_id
                    or "default"
                ),
            )
        )

    @app.get("/api/profile/list")
    async def profile_list() -> dict[str, Any]:
        return {"profiles": list_profiles_with_metadata(repo_root=root)}

    @app.get("/api/profile/current")
    async def profile_current() -> dict[str, Any]:
        current_id = str(
            app.state.ops.config_store.load_effective().trust_profile_id
            or "default"
        )
        metadata = next(
            (
                entry
                for entry in list_profiles_with_metadata(repo_root=root)
                if entry["profile_id"] == current_id
            ),
            None,
        )
        return {"profile_id": current_id, "profile": metadata}

    @app.get("/ask", response_class=HTMLResponse)
    async def ask_ui() -> HTMLResponse:
        """The simplest possible front door.

        One question box, one button, profile chip linking to /profiles.
        Retrieval-first — works without Ollama. Public, no auth. This is
        the "give me a button" surface for the average user.
        """

        return HTMLResponse(
            _ask_html(
                current_profile_id=str(
                    app.state.ops.config_store.load_effective().trust_profile_id
                    or "default"
                ),
            )
        )

    @app.post("/api/ask")
    async def ask_endpoint(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Retrieval-first public ask endpoint.

        Body fields:
          - question (required): the current question.
          - selected_madhhab (optional): hanafi/shafii/maliki/hanbali/
            compare_all/not_specified. Defaults to not_specified.
          - conversation_context (optional): list of prior questions
            from the same conversation thread. Used as retrieval-time
            topic biasing — keeps follow-ups on-topic without
            fabricating dialogue continuity.

        Always performs retrieval; the active trust profile drives the
        ranking. Returns rendered plain-text + a structured payload
        (citations + Evidence Ladder + Confidence Taxonomy).
        """

        question = str((payload or {}).get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question required")
        selected_madhhab = str(
            (payload or {}).get("selected_madhhab") or "not_specified"
        ).strip().lower() or "not_specified"
        raw_context = (payload or {}).get("conversation_context") or []
        if not isinstance(raw_context, list):
            raw_context = []
        # Cap at 5 prior questions to keep the retrieval prompt focused;
        # older context drifts the topic.
        conversation_context = [
            str(item).strip()
            for item in raw_context[-5:]
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]

        return _public_ask(
            app.state.ops,
            repo_root=root,
            question=question,
            selected_madhhab=selected_madhhab,
            conversation_context=conversation_context,
        )

    @app.post("/api/profile/set")
    async def profile_set(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Set the active trust/scholar methodology profile.

        Intentionally narrow: this endpoint can only flip
        ``trust_profile_id`` and only to a known profile. Other config
        fields remain behind the authenticated admin endpoint.
        """

        requested = str((payload or {}).get("profile_id") or "").strip()
        if not requested:
            raise HTTPException(status_code=400, detail="profile_id required")
        available = {entry["profile_id"] for entry in list_profiles_with_metadata(repo_root=root)}
        if requested not in available:
            raise HTTPException(
                status_code=404,
                detail=f"unknown profile_id: {requested}. Available: {sorted(available)}",
            )
        try:
            app.state.ops.config_store.update({"trust_profile_id": requested})
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "profile_id": requested}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        snapshot = app.state.ops.health_snapshot()
        return {
            "status": "ok",
            "mode": "retrieval_first",
            "laptop_build": bool(snapshot["runtime_health"].get("laptop_build")),
            "runtime_identity": snapshot["runtime_identity"],
            "runtime_health": snapshot["runtime_health"],
        }

    @app.get("/retrieval/status")
    async def retrieval_status() -> dict[str, Any]:
        return app.state.ops.retrieval_status_snapshot()

    @app.get("/api/landing/glance")
    async def landing_glance() -> dict[str, Any]:
        return app.state.ops.landing_glance()

    @app.post("/api/auth/register")
    async def register_local_user(
        payload: RegisterRequestModel,
    ) -> JSONResponse:
        try:
            user = app.state.ops.register_local_user(
                username=payload.username,
                password=payload.password,
                display_name=payload.display_name or payload.username,
                default_madhhab=payload.default_madhhab,
                default_answer_mode=payload.default_answer_mode,
            )
            authenticated_user, token = app.state.ops.authenticate_local_user(
                username=payload.username,
                password=payload.password,
            )
            response = JSONResponse(
                {
                    "ok": True,
                    "user": authenticated_user or user,
                }
            )
            _set_auth_cookie(response, token)
            return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/auth/login")
    async def login_local_user(
        payload: LoginRequestModel,
    ) -> JSONResponse:
        user, token = app.state.ops.authenticate_local_user(
            username=payload.username,
            password=payload.password,
        )
        if not user or not token:
            raise HTTPException(status_code=401, detail="invalid credentials")
        response = JSONResponse({"ok": True, "user": user})
        _set_auth_cookie(response, token)
        return response

    @app.post("/api/auth/logout")
    async def logout_local_user(
        request: Request,
    ) -> JSONResponse:
        token = request.cookies.get(AUTH_COOKIE_NAME)
        app.state.ops.logout_local_user(token)
        response = JSONResponse({"ok": True})
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")
        return response

    @app.get("/api/auth/me")
    async def auth_me(
        request: Request,
    ) -> dict[str, Any]:
        user = _current_authenticated_user(request)
        if not user:
            return {"authenticated": False, "user": None}
        return {
            "authenticated": True,
            "user": user,
            "workspace": app.state.ops.user_sidebar_state(user_id=int(user["id"])),
        }

    @app.post("/api/chat")
    async def chat(
        request: Request,
        payload: ChatRequestModel,
        actor: RoleContext = Depends(
            require_permission("chat.ask", default_role="manager")
        ),
    ) -> dict[str, Any]:
        current_user = _current_authenticated_user(request)
        result = app.state.ops.chat(
            question=payload.question,
            selected_madhhab=payload.selected_madhhab,
            answer_mode=payload.answer_mode,
            research_depth=payload.research_depth,
            actor=actor,
            greeting_style=payload.greeting_style,
            tone_level=payload.tone_level,
            ollama_model=payload.ollama_model,
            retrieved_sources=payload.retrieved_sources,
            user_id=int(current_user["id"]) if current_user else None,
            chat_session_id=payload.chat_session_id,
            project_id=payload.project_id,
        )
        return as_json(result)

    @app.get("/api/user/workspace")
    async def user_workspace_state(
        request: Request,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        return app.state.ops.user_sidebar_state(user_id=int(user["id"]))

    @app.post("/api/user/settings")
    async def update_user_settings(
        request: Request,
        payload: UserSettingsRequestModel,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        updated = app.state.ops.update_user_settings(
            user_id=int(user["id"]),
            display_name=payload.display_name,
            default_madhhab=payload.default_madhhab,
            default_answer_mode=payload.default_answer_mode,
        )
        return {"ok": True, "user": updated}

    @app.get("/api/user/chats")
    async def user_chats(
        request: Request,
        project_id: int | None = Query(default=None),
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        return {
            "rows": app.state.ops.list_user_chat_sessions(
                user_id=int(user["id"]),
                project_id=project_id,
            )
        }

    @app.post("/api/user/chats")
    async def create_user_chat(
        request: Request,
        payload: CreateChatSessionRequestModel,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        title = payload.title or "New Chat"
        session = app.state.ops.create_user_chat_session(
            user_id=int(user["id"]),
            title=title,
            project_id=payload.project_id,
        )
        return {"ok": True, "session": session}

    @app.get("/api/user/chats/{session_id}")
    async def get_user_chat(
        request: Request,
        session_id: int,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        session = app.state.ops.get_user_chat_session(
            user_id=int(user["id"]),
            session_id=session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="chat session not found")
        return {"ok": True, "session": session}

    @app.post("/api/user/chats/{session_id}/project")
    async def attach_user_chat_project(
        request: Request,
        session_id: int,
        payload: AttachChatProjectRequestModel,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        app.state.ops.attach_user_chat_to_project(
            user_id=int(user["id"]),
            session_id=session_id,
            project_id=payload.project_id,
        )
        session = app.state.ops.get_user_chat_session(
            user_id=int(user["id"]),
            session_id=session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="chat session not found")
        return {"ok": True, "session": session}

    @app.get("/api/user/projects")
    async def user_projects(
        request: Request,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        return {"rows": app.state.ops.list_user_projects(user_id=int(user["id"]))}

    @app.post("/api/user/projects")
    async def create_user_project(
        request: Request,
        payload: CreateProjectRequestModel,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        project = app.state.ops.create_user_project(
            user_id=int(user["id"]),
            title=payload.title,
            description=payload.description,
            madhhab=payload.madhhab,
            study_mode=payload.study_mode,
            status=payload.status,
        )
        return {"ok": True, "project": project}

    @app.get("/api/user/projects/{project_id}")
    async def get_user_project(
        request: Request,
        project_id: int,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        project = app.state.ops.get_user_project(
            user_id=int(user["id"]),
            project_id=project_id,
        )
        if not project:
            raise HTTPException(status_code=404, detail="study project not found")
        return {"ok": True, "project": project}

    @app.post("/api/user/projects/{project_id}/study/progress")
    async def update_project_study_progress(
        request: Request,
        project_id: int,
        payload: ProjectStudyProgressRequestModel,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        try:
            project = app.state.ops.update_user_project_study_progress(
                user_id=int(user["id"]),
                project_id=project_id,
                active_lesson_index=payload.active_lesson_index,
                completed_lesson_index=payload.completed_lesson_index,
                advance=payload.advance,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "project": project}

    @app.get("/api/user/memory")
    async def user_memory(
        request: Request,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        return {"rows": app.state.ops.list_user_memory(user_id=int(user["id"]))}

    @app.post("/api/user/memory/notes")
    async def create_memory_note(
        request: Request,
        payload: MemoryNoteRequestModel,
    ) -> dict[str, Any]:
        user = _require_authenticated_user(request)
        note = app.state.ops.add_user_memory_note(
            user_id=int(user["id"]),
            content=payload.content,
            source_session_id=payload.source_session_id,
            confidence=payload.confidence,
        )
        return {"ok": True, "memory": note}

    @app.get("/api/admin/overview")
    async def admin_overview(
        _: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.runtime_snapshot()

    @app.get("/api/admin/micro-llm/status")
    async def micro_llm_status(
        _: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.micro_shadow_status()

    @app.post("/api/admin/micro-llm/smoke")
    async def micro_llm_smoke(
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.run_micro_shadow_smoke(actor=actor)

    @app.post("/api/admin/micro-llm/benchmark")
    async def micro_llm_benchmark(
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.run_micro_shadow_benchmark(actor=actor)

    @app.get("/api/admin/readiness/portable", response_class=PlainTextResponse)
    async def portable_readiness(
        _: RoleContext = Depends(require_permission("admin.access")),
    ) -> PlainTextResponse:
        return PlainTextResponse(
            app.state.ops.portable_readiness_markdown(),
            media_type="text/markdown",
        )

    @app.get("/api/admin/updates/status")
    async def update_center_status(
        _: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.update_center_status()

    @app.post("/api/admin/updates/check")
    async def check_for_updates(
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.check_for_updates(actor=actor)

    @app.post("/api/admin/updates/download")
    async def download_update(
        payload: UpdateSelectionRequestModel,
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        try:
            return app.state.ops.download_update(actor=actor, update_id=payload.update_id)
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/admin/updates/verify")
    async def verify_update(
        payload: UpdateSelectionRequestModel,
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        try:
            return app.state.ops.verify_update(actor=actor, update_id=payload.update_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/admin/updates/apply")
    async def apply_update(
        payload: UpdateSelectionRequestModel,
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        try:
            return app.state.ops.apply_update(actor=actor, update_id=payload.update_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/admin/updates/rebuild-index")
    async def rebuild_index_if_needed(
        actor: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.rebuild_index_if_needed(actor=actor)

    @app.get("/api/admin/config")
    async def admin_config(
        _: RoleContext = Depends(require_permission("config.inspect")),
    ) -> dict[str, Any]:
        return app.state.ops.inspect_runtime_config()

    @app.post("/api/admin/config")
    async def update_config(
        patch: dict[str, Any] = Body(...),
        actor: RoleContext = Depends(require_permission("config.update")),
    ) -> dict[str, Any]:
        try:
            return app.state.ops.update_runtime_config(actor=actor, patch=patch)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValidationError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/admin/permissions")
    async def permissions(
        actor: RoleContext = Depends(require_permission("permissions.inspect")),
    ) -> dict[str, Any]:
        return app.state.ops.inspect_permissions(actor=actor)

    @app.get("/api/admin/retrieval/status")
    async def retrieval_status(
        _: RoleContext = Depends(require_permission("admin.access")),
    ) -> dict[str, Any]:
        return app.state.ops.runtime_snapshot()["retrieval"]

    @app.post("/api/admin/retrieval/reload")
    async def reload_retrieval(
        background: bool = Query(False),
        retry_of: str | None = Query(default=None),
        retry_reason: str | None = Query(default=None),
        actor: RoleContext = Depends(require_permission("retrieval.reload")),
    ) -> dict[str, Any]:
        result = app.state.ops.reload_retrieval_assets(
            actor=actor,
            background=background,
            retry_of=retry_of,
            retry_reason=retry_reason,
        )
        return as_json(result)

    @app.post("/api/admin/retrieval/reindex")
    async def reindex_retrieval(
        background: bool = Query(False),
        retry_of: str | None = Query(default=None),
        retry_reason: str | None = Query(default=None),
        actor: RoleContext = Depends(require_permission("retrieval.reindex")),
    ) -> dict[str, Any]:
        result = app.state.ops.reindex_documents(
            actor=actor,
            background=background,
            retry_of=retry_of,
            retry_reason=retry_reason,
        )
        return as_json(result)

    @app.get("/api/admin/retrieval/debug")
    async def retrieval_debug(
        question: str = Query(...),
        selected_madhhab: str = Query("not_specified"),
        answer_mode: str = Query("research"),
        actor: RoleContext = Depends(require_permission("retrieval.debug")),
    ) -> dict[str, Any]:
        config = app.state.ops.config_store.load_effective()
        if not config.admin_feature_toggles.enable_retrieval_debug:
            raise HTTPException(status_code=403, detail="retrieval debug is disabled")
        return {
            "actor": as_json(actor),
            **app.state.ops.retrieval_debug(
                question=question,
                selected_madhhab=selected_madhhab,
                answer_mode=answer_mode,
            ),
        }

    @app.get("/api/admin/logs/chat")
    async def chat_logs(
        limit: int = Query(20, ge=1, le=200),
        success: bool | None = Query(default=None),
        _: RoleContext = Depends(require_permission("logs.inspect")),
    ) -> dict[str, Any]:
        _ensure_log_queries_enabled(app.state.ops)
        return {"rows": app.state.ops.chat_logs(limit=limit, success=success)}

    @app.get("/api/admin/logs/audit")
    async def audit_logs(
        limit: int = Query(20, ge=1, le=200),
        action: str | None = Query(default=None),
        _: RoleContext = Depends(require_permission("logs.inspect")),
    ) -> dict[str, Any]:
        _ensure_log_queries_enabled(app.state.ops)
        return {"rows": app.state.ops.audit_logs(limit=limit, action=action)}

    @app.get("/api/admin/logs/errors")
    async def error_logs(
        limit: int = Query(20, ge=1, le=200),
        _: RoleContext = Depends(require_permission("logs.inspect")),
    ) -> dict[str, Any]:
        _ensure_log_queries_enabled(app.state.ops)
        return {"rows": app.state.ops.error_logs(limit=limit)}

    @app.get("/api/admin/logs/recent")
    async def recent_logs(
        kind: str = Query("failures"),
        limit: int = Query(20, ge=1, le=200),
        _: RoleContext = Depends(require_permission("logs.inspect")),
    ) -> dict[str, Any]:
        _ensure_log_queries_enabled(app.state.ops)
        return {"rows": app.state.ops.recent_logs(kind=kind, limit=limit)}

    @app.get("/api/admin/events")
    async def operator_events(
        limit: int = Query(50, ge=1, le=200),
        severity: str | None = Query(default=None),
        request_id: str | None = Query(default=None),
        related_task_id: str | None = Query(default=None),
        after_id: int | None = Query(default=None, ge=0),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        return {
            "rows": app.state.ops.operator_events(
                limit=limit,
                severity=severity,
                request_id=request_id,
                related_task_id=related_task_id,
                after_id=after_id,
            )
        }

    @app.get("/api/admin/events/stream")
    async def operator_events_stream(
        request: Request,
        after_id: int = Query(0, ge=0),
        severity: str | None = Query(default=None),
        once: bool = Query(False),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> StreamingResponse:
        async def event_source():
            cursor = after_id
            backlog = app.state.ops.operator_events(
                limit=50,
                severity=severity,
                after_id=cursor or None,
            )
            for row in backlog:
                cursor = max(cursor, int(row.get("id") or 0))
                yield _sse_message(row)
            if once:
                return
            while True:
                if await request.is_disconnected():
                    break
                rows = await asyncio.to_thread(
                    app.state.ops.event_broker.wait_for_events,
                    after_id=cursor,
                    timeout_seconds=10.0,
                )
                if severity:
                    rows = [row for row in rows if row.get("severity") == severity]
                if not rows:
                    yield ": ping\n\n"
                    continue
                for row in rows:
                    cursor = max(cursor, int(row.get("id") or 0))
                    yield _sse_message(row)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/api/admin/tasks/active")
    async def active_tasks(
        limit: int = Query(20, ge=1, le=200),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        return {"rows": app.state.ops.active_tasks(limit=limit)}

    @app.get("/api/admin/tasks/recent")
    async def recent_tasks(
        limit: int = Query(20, ge=1, le=200),
        task_type: str | None = Query(default=None),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        return {"rows": app.state.ops.recent_tasks(limit=limit, task_type=task_type)}

    @app.get("/api/admin/tasks/failed")
    async def failed_tasks(
        limit: int = Query(20, ge=1, le=200),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        return {"rows": app.state.ops.failed_tasks(limit=limit)}

    @app.get("/api/admin/tasks/blocked")
    async def blocked_tasks(
        limit: int = Query(20, ge=1, le=200),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        return {"rows": app.state.ops.blocked_tasks(limit=limit)}

    @app.get("/api/admin/tasks/degraded")
    async def degraded_tasks(
        limit: int = Query(20, ge=1, le=200),
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        return {"rows": app.state.ops.degraded_tasks(limit=limit)}

    @app.get("/api/admin/tasks/{task_id}")
    async def task_detail(
        task_id: str,
        _: RoleContext = Depends(require_permission("tasks.inspect")),
    ) -> dict[str, Any]:
        payload = app.state.ops.task_detail(task_id=task_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="task not found")
        return payload

    @app.get("/api/chat/status/{request_id}")
    async def chat_request_status(request_id: str) -> dict[str, Any]:
        payload = app.state.ops.request_status(request_id=request_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="request not found")
        return payload

    @app.get("/api/chat/status/stream")
    async def chat_request_status_stream(
        request: Request,
        request_id: str = Query(...),
        after_id: int = Query(0, ge=0),
        once: bool = Query(False),
    ) -> StreamingResponse:
        async def event_source():
            cursor = after_id
            backlog = app.state.ops.operator_events(
                limit=50,
                request_id=request_id,
                after_id=cursor or None,
            )
            for row in backlog:
                cursor = max(cursor, int(row.get("id") or 0))
                yield _sse_message(row)
            if once:
                return
            while True:
                if await request.is_disconnected():
                    break
                rows = await asyncio.to_thread(
                    app.state.ops.event_broker.wait_for_events,
                    after_id=cursor,
                    timeout_seconds=10.0,
                    request_id=request_id,
                )
                if not rows:
                    yield ": ping\n\n"
                    continue
                for row in rows:
                    cursor = max(cursor, int(row.get("id") or 0))
                    yield _sse_message(row)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.exception_handler(PermissionError)
    async def permission_error_handler(_: Request, exc: PermissionError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"error": str(exc)})

    return app


def as_json(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


def _set_auth_cookie(response: JSONResponse, token: str | None) -> None:
    if not token:
        return
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


def _current_authenticated_user(request: Request) -> dict[str, Any] | None:
    ops: OpsService = request.app.state.ops
    token = request.cookies.get(AUTH_COOKIE_NAME)
    return ops.current_local_user(token)


def _require_authenticated_user(request: Request) -> dict[str, Any]:
    user = _current_authenticated_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="authentication required")
    return user


def _ensure_log_queries_enabled(ops: OpsService) -> None:
    if not ops.config_store.load_effective().admin_feature_toggles.enable_log_queries:
        raise HTTPException(status_code=403, detail="log queries are disabled")


def _sse_message(payload: dict[str, Any]) -> str:
    event_type = str(payload.get("event_type") or "message")
    event_id = str(payload.get("id") or payload.get("event_id") or "")
    return (
        f"id: {event_id}\n"
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


def _manager_html(*, repo_root: Path = REPO_ROOT, initial_auth_view: str = "auto") -> str:
    scholar_profiles = _manager_scholar_profile_map(repo_root)
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Halal Jordan</title>
    <link rel="stylesheet" href="/assets/style.css" />
    <script defer src="/assets/app.js"></script>
    <style>
      :root {
        color-scheme: light;
        --bg: #f3ead8;
        --bg-elevated: rgba(255, 250, 241, 0.94);
        --bg-panel: rgba(255, 249, 238, 0.89);
        --bg-panel-strong: rgba(255, 252, 246, 0.96);
        --text: #2f271f;
        --muted: #6c5f4e;
        --muted-strong: #4f4333;
        --line: rgba(123, 90, 46, 0.17);
        --line-strong: rgba(154, 114, 64, 0.34);
        --accent: #bc8f49;
        --accent-soft: rgba(227, 191, 121, 0.32);
        --accent-deep: #56684f;
        --success: #6a7f64;
        --shadow: 0 34px 76px rgba(52, 34, 14, 0.18);
        --shadow-soft: 0 20px 42px rgba(52, 34, 14, 0.11);
        --shadow-deep: 0 42px 92px rgba(40, 24, 8, 0.24);
        --radius: 22px;
      }

      * { box-sizing: border-box; }

      html {
        scroll-behavior: smooth;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        background:
          radial-gradient(circle at 50% 14%, rgba(255, 220, 136, 0.18), transparent 18%),
          linear-gradient(180deg, #120c09 0%, #1b130e 18%, #2b1d14 48%, #17110d 100%);
        color: var(--text);
        position: relative;
        overflow-x: hidden;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
          linear-gradient(180deg, rgba(19, 12, 8, 0.74), rgba(21, 13, 9, 0.42) 22%, rgba(255, 213, 116, 0.04) 44%, rgba(16, 10, 7, 0.78) 100%),
          radial-gradient(circle at 50% 16%, rgba(255, 232, 176, 0.18), rgba(255, 244, 220, 0.08) 22%, rgba(34, 21, 13, 0.2) 56%, rgba(12, 8, 6, 0.86) 100%),
          url('/assets/images/halal-jordan-hero-bg.png');
        background-position: center top;
        background-repeat: no-repeat;
        background-size: cover;
        filter: saturate(0.86) contrast(0.94) brightness(0.74);
        transform: scale(1.015);
        opacity: 0.48;
      }

      body::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
          radial-gradient(circle at 50% 18%, rgba(255, 239, 198, 0.12), transparent 14%),
          radial-gradient(circle at 50% 34%, rgba(255, 214, 116, 0.1), transparent 30%),
          linear-gradient(90deg, rgba(22, 15, 11, 0.82), rgba(22, 15, 11, 0.28) 16%, rgba(22, 15, 11, 0.28) 84%, rgba(22, 15, 11, 0.82)),
          linear-gradient(180deg, rgba(17, 11, 8, 0.68), transparent 24%, transparent 70%, rgba(17, 11, 8, 0.6));
        opacity: 0.84;
      }

      a {
        color: inherit;
        text-decoration: none;
      }

      .app-layout {
        width: min(1320px, calc(100% - 2.75rem));
        margin: 0 auto;
        display: grid;
        grid-template-columns: minmax(248px, 292px) minmax(0, 1fr);
        gap: 1.45rem;
        align-items: start;
        position: relative;
        z-index: 1;
        transition: grid-template-columns 320ms ease, gap 320ms ease;
      }

      .page-shell {
        width: min(1080px, 100%);
        min-width: 0;
        margin: 0;
        padding: 1.25rem 0 4rem;
        position: relative;
        z-index: 1;
        justify-self: center;
      }

      .workspace-sidebar {
        position: sticky;
        top: 0;
        align-self: start;
        padding: 1rem 0 2rem;
      }

      .sidebar-rail {
        display: grid;
        gap: 0.55rem;
        padding: 0.72rem;
        border-radius: 18px;
        border: 1px solid rgba(137, 100, 54, 0.18);
        background: linear-gradient(180deg, rgba(255, 249, 239, 0.86), rgba(241, 229, 205, 0.82));
        box-shadow:
          var(--shadow-soft),
          inset 0 1px 0 rgba(255, 255, 255, 0.68);
        backdrop-filter: blur(14px) saturate(1.04);
      }

      .sidebar-toggle {
        appearance: none;
        border: 1px solid rgba(150, 112, 64, 0.24);
        border-radius: 14px;
        background: rgba(255, 252, 246, 0.92);
        color: var(--text);
        padding: 0.78rem 0.86rem;
        font: inherit;
        font-weight: 700;
        text-align: left;
        cursor: pointer;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
      }

      .sidebar-toggle:hover,
      .sidebar-toggle:focus-visible {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(78, 57, 24, 0.12);
        border-color: rgba(173, 130, 71, 0.42);
      }

      .sidebar-rail-title {
        display: block;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.2rem;
      }

      .sidebar-rail-copy {
        display: block;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.35;
      }

      .sidebar-stack {
        display: grid;
        gap: 0.95rem;
        transition: opacity 220ms ease, transform 220ms ease;
      }

      .sidebar-card {
        background:
          linear-gradient(180deg, rgba(255, 249, 239, 0.86), rgba(241, 229, 205, 0.82));
        border: 1px solid rgba(137, 100, 54, 0.18);
        border-radius: 18px;
        box-shadow:
          var(--shadow-soft),
          inset 0 1px 0 rgba(255, 255, 255, 0.68);
        backdrop-filter: blur(14px) saturate(1.04);
        padding: 1rem 1rem 1.05rem;
        position: relative;
        overflow: hidden;
      }

      .sidebar-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 4px;
        background: linear-gradient(90deg, rgba(210, 168, 93, 0.42), rgba(107, 124, 103, 0.14), transparent);
      }

      .sidebar-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.65rem;
        margin-bottom: 0.8rem;
      }

      .sidebar-title {
        margin: 0.25rem 0 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.18rem;
        font-weight: 600;
      }

      .sidebar-copy,
      .sidebar-empty,
      .sidebar-meta,
      .sidebar-helper {
        color: var(--muted);
        line-height: 1.55;
      }

      .sidebar-copy,
      .sidebar-helper {
        font-size: 0.95rem;
      }

      .sidebar-meta {
        font-size: 0.85rem;
      }

      .sidebar-empty {
        font-size: 0.92rem;
        padding: 0.85rem 0.9rem;
        border-radius: 12px;
        border: 1px dashed rgba(150, 114, 65, 0.24);
        background: rgba(255, 252, 246, 0.68);
      }

      .sidebar-profile {
        display: grid;
        gap: 0.55rem;
      }

      .sidebar-profile-name {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.36rem;
        font-weight: 600;
      }

      .sidebar-profile-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
      }

      .sidebar-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        background: rgba(255, 252, 246, 0.84);
        border: 1px solid rgba(137, 100, 54, 0.16);
        color: var(--muted-strong);
        font-size: 0.82rem;
      }

      .sidebar-actions,
      .sidebar-inline-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
      }

      .sidebar-button,
      .sidebar-secondary-button,
      .sidebar-link-button {
        appearance: none;
        border: 1px solid rgba(150, 112, 64, 0.24);
        border-radius: 12px;
        font: inherit;
        cursor: pointer;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
      }

      .sidebar-button {
        background: linear-gradient(180deg, rgba(247, 221, 162, 0.98), rgba(216, 174, 96, 0.96));
        color: #2a2012;
        padding: 0.7rem 0.9rem;
        font-weight: 700;
        box-shadow: 0 12px 22px rgba(78, 57, 24, 0.15), inset 0 1px 0 rgba(255, 248, 226, 0.9);
      }

      .sidebar-secondary-button {
        background: rgba(255, 252, 246, 0.94);
        color: var(--muted-strong);
        padding: 0.62rem 0.82rem;
        font-weight: 600;
      }

      .sidebar-link-button {
        background: transparent;
        color: var(--accent-deep);
        padding: 0.25rem 0;
        border-color: transparent;
        text-align: left;
        font-weight: 600;
      }

      .sidebar-button:hover,
      .sidebar-secondary-button:hover,
      .sidebar-link-button:hover,
      .sidebar-button:focus-visible,
      .sidebar-secondary-button:focus-visible,
      .sidebar-link-button:focus-visible {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(78, 57, 24, 0.14);
        border-color: rgba(173, 130, 71, 0.42);
      }

      .sidebar-field-grid {
        display: grid;
        gap: 0.72rem;
      }

      .sidebar-field-grid label,
      .sidebar-settings-grid label {
        margin-top: 0;
        margin-bottom: 0.28rem;
        font-size: 0.9rem;
      }

      .sidebar-settings-grid {
        display: grid;
        gap: 0.78rem;
      }

      .sidebar-card input,
      .sidebar-card select,
      .sidebar-card textarea {
        padding: 0.78rem 0.88rem;
        border-radius: 12px;
      }

      .sidebar-card textarea {
        min-height: 6.2rem;
      }

      .sidebar-list {
        display: grid;
        gap: 0.55rem;
      }

      .sidebar-item {
        width: 100%;
        text-align: left;
        border-radius: 14px;
        border: 1px solid rgba(153, 116, 69, 0.16);
        background: rgba(255, 252, 246, 0.74);
        padding: 0.82rem 0.9rem;
        display: grid;
        gap: 0.28rem;
        cursor: pointer;
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
      }

      .sidebar-item:hover,
      .sidebar-item:focus-visible {
        transform: translateY(-1px);
        border-color: rgba(173, 130, 71, 0.35);
        background: rgba(255, 254, 250, 0.88);
        box-shadow: 0 14px 24px rgba(78, 57, 24, 0.08);
      }

      .sidebar-item.active {
        border-color: rgba(173, 130, 71, 0.46);
        background: linear-gradient(180deg, rgba(255, 246, 225, 0.96), rgba(247, 235, 207, 0.92));
        box-shadow: 0 16px 26px rgba(78, 57, 24, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.7);
      }

      .sidebar-item-title {
        font-weight: 700;
        color: var(--text);
      }

      .sidebar-item-meta {
        font-size: 0.83rem;
        color: var(--muted);
        line-height: 1.45;
      }

      .project-detail-shell {
        display: grid;
        gap: 0.75rem;
      }

      .project-detail-card,
      .project-lesson-card,
      .project-reading-card {
        border-radius: 14px;
        border: 1px solid rgba(153, 116, 69, 0.16);
        background: rgba(255, 252, 246, 0.76);
        padding: 0.85rem 0.92rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .project-detail-title,
      .project-subtitle,
      .project-reading-title,
      .project-lesson-title {
        color: var(--text);
      }

      .project-detail-title {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.14rem;
        font-weight: 600;
      }

      .project-subtitle,
      .project-lesson-meta,
      .project-reading-meta {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.5;
      }

      .project-progress-bar {
        width: 100%;
        height: 0.55rem;
        border-radius: 999px;
        background: rgba(206, 187, 153, 0.42);
        overflow: hidden;
      }

      .project-progress-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, rgba(205, 156, 73, 0.96), rgba(105, 127, 96, 0.9));
      }

      .project-chip-row,
      .project-inline-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }

      .project-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(255, 249, 238, 0.88);
        border: 1px solid rgba(153, 116, 69, 0.16);
        color: var(--muted-strong);
        font-size: 0.79rem;
      }

      .project-lesson-list,
      .project-reading-list {
        display: grid;
        gap: 0.6rem;
      }

      .project-lesson-card.is-active {
        border-color: rgba(173, 130, 71, 0.34);
        background: linear-gradient(180deg, rgba(255, 248, 232, 0.94), rgba(246, 236, 214, 0.9));
      }

      .project-lesson-title,
      .project-reading-title {
        font-weight: 700;
      }

      .project-inline-actions .sidebar-link-button,
      .project-inline-actions .sidebar-secondary-button {
        padding: 0.38rem 0.62rem;
      }

      .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(137, 100, 54, 0.22), transparent);
        margin: 0.25rem 0 0.35rem;
      }

      .workspace-context {
        margin-bottom: 0.95rem;
        padding: 0.82rem 0.94rem;
        border-radius: 14px;
        border: 1px solid rgba(137, 100, 54, 0.16);
        background: rgba(255, 252, 246, 0.74);
        color: var(--muted-strong);
        line-height: 1.6;
      }

      .retrieval-status-banner {
        margin-bottom: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(137, 100, 54, 0.18);
        background: rgba(255, 249, 238, 0.92);
        color: var(--ink);
        line-height: 1.55;
      }

      .retrieval-status-banner strong {
        display: inline-block;
        margin-right: 0.35rem;
      }

      .retrieval-status-banner[data-state="loading"] {
        border-color: rgba(173, 130, 71, 0.28);
        background: rgba(255, 247, 230, 0.96);
      }

      .retrieval-status-banner[data-state="error"] {
        border-color: rgba(160, 77, 61, 0.28);
        background: rgba(255, 242, 239, 0.97);
      }

      .retrieval-status-banner[data-state="ready"] {
        border-color: rgba(53, 116, 83, 0.24);
        background: rgba(239, 250, 244, 0.94);
      }

      .history-shell {
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(137, 100, 54, 0.16);
        background: linear-gradient(180deg, rgba(255, 252, 246, 0.78), rgba(246, 236, 218, 0.74));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.74);
      }

      .history-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
      }

      .history-heading {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.15rem;
        font-weight: 600;
      }

      .history-list {
        display: grid;
        gap: 0.85rem;
      }

      .history-message {
        padding: 0.95rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(137, 100, 54, 0.16);
        background: rgba(255, 252, 246, 0.84);
        box-shadow: 0 14px 26px rgba(78, 57, 24, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.7);
      }

      .history-message.user {
        background: linear-gradient(180deg, rgba(248, 238, 216, 0.9), rgba(240, 226, 198, 0.86));
      }

      .history-message.assistant {
        background: linear-gradient(180deg, rgba(255, 252, 246, 0.9), rgba(245, 236, 219, 0.88));
      }

      .history-message-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        margin-bottom: 0.55rem;
      }

      .history-role {
        font-weight: 700;
        color: var(--muted-strong);
      }

      .history-time {
        color: var(--muted);
        font-size: 0.84rem;
      }

      .history-copy {
        margin: 0;
        color: var(--text);
        line-height: 1.68;
        white-space: pre-wrap;
      }

      .history-rendered {
        display: grid;
        gap: 0.9rem;
      }

      .history-answer-shell,
      .history-study-shell {
        display: grid;
        gap: 0.85rem;
      }

      .history-answer-grid,
      .history-answer-layers,
      .history-answer-support {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
      }

      .history-answer-shell .answer-card,
      .history-study-shell .answer-card {
        padding: 0.95rem 1rem;
      }

      .history-answer-shell .answer-card-title,
      .history-study-shell .answer-card-title,
      .history-study-shell .answer-source-layer-title {
        font-size: 1.02rem;
      }

      .hero {
        position: relative;
        overflow: hidden;
        min-height: clamp(20rem, 48vh, 28rem);
        display: grid;
        place-items: center;
        padding: 0.9rem 0 0.75rem;
      }

      .hero::before,
      .hero::after {
        content: "";
        position: absolute;
        inset: auto;
        pointer-events: none;
      }

      .hero::before {
        width: 460px;
        height: 460px;
        top: -195px;
        right: -120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(229, 180, 92, 0.16), transparent 68%);
        filter: blur(22px);
      }

      .hero::after {
        width: 400px;
        height: 400px;
        bottom: -190px;
        left: -90px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(244, 215, 162, 0.16), transparent 72%);
        filter: blur(24px);
      }

      .hero-frame {
        position: relative;
        width: min(700px, 100%);
        padding: 2.2rem clamp(1.2rem, 3vw, 2.4rem) 1.8rem;
        text-align: center;
        background:
          radial-gradient(circle at 50% 13%, rgba(255, 217, 122, 0.4), rgba(255, 243, 214, 0.12) 28%, transparent 52%),
          linear-gradient(180deg, rgba(255, 250, 240, 0.88), rgba(247, 234, 211, 0.84));
        border: 1px solid rgba(177, 127, 61, 0.34);
        border-radius: 164px 164px 28px 28px / 124px 124px 28px 28px;
        box-shadow:
          0 28px 58px rgba(36, 21, 11, 0.28),
          0 14px 28px rgba(92, 56, 20, 0.14),
          inset 0 1px 0 rgba(255, 255, 255, 0.9),
          inset 0 -18px 30px rgba(197, 149, 75, 0.1),
          0 0 0 1px rgba(255, 229, 176, 0.18);
        isolation: isolate;
        backdrop-filter: blur(14px) saturate(1.04);
        animation: settleIn 720ms ease both;
      }

      .hero-frame::before {
        content: "";
        position: absolute;
        inset: 16px;
        border-radius: 146px 146px 22px 22px / 108px 108px 22px 22px;
        border: 1px solid rgba(193, 146, 76, 0.28);
        pointer-events: none;
        box-shadow:
          inset 0 0 0 1px rgba(255, 247, 222, 0.26),
          0 0 0 1px rgba(105, 73, 38, 0.02);
      }

      .hero-frame::after {
        content: "";
        position: absolute;
        inset: auto 15% 1.25rem;
        height: 26px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(223, 188, 120, 0.28), transparent 72%);
        filter: blur(12px);
        pointer-events: none;
      }

      .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0.95rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 250, 242, 0.76);
        color: var(--muted-strong);
        font-size: 0.82rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62);
      }

      .eyebrow::before,
      .eyebrow::after {
        content: "";
        width: 1.1rem;
        height: 1px;
        background: rgba(173, 130, 71, 0.55);
      }

      h1 {
        margin: 1.1rem 0 0.75rem;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: clamp(2.7rem, 6vw, 4.35rem);
        line-height: 0.98;
        letter-spacing: -0.03em;
        font-weight: 600;
        color: #302518;
        text-shadow: 0 8px 18px rgba(255, 224, 154, 0.14);
      }

      .subtitle {
        max-width: 34rem;
        margin: 0 auto;
        font-size: clamp(1.02rem, 1.85vw, 1.2rem);
        line-height: 1.62;
        color: #534633;
      }

      .supporting-text {
        max-width: 38rem;
        margin: 0.95rem auto 0;
        color: #685946;
        font-size: 0.98rem;
        line-height: 1.68;
      }

      .trust-line {
        margin: 0.95rem auto 0;
        max-width: 34rem;
        color: #5b6a54;
        font-weight: 600;
        letter-spacing: 0.01em;
      }

      .cta-row {
        margin-top: 1.45rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.75rem;
      }

      .hero-example-shell {
        margin-top: 0.85rem;
      }

      .example-toggle {
        appearance: none;
        border: 1px solid rgba(137, 100, 54, 0.2);
        border-radius: 999px;
        background: rgba(255, 250, 242, 0.78);
        color: var(--muted-strong);
        font: inherit;
        font-weight: 600;
        padding: 0.72rem 1.05rem;
        cursor: pointer;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
      }

      .example-toggle:hover,
      .example-toggle:focus-visible {
        transform: translateY(-1px);
        border-color: rgba(173, 130, 71, 0.34);
        background: rgba(255, 252, 247, 0.9);
        box-shadow: 0 14px 26px rgba(78, 57, 24, 0.1);
      }

      .hero-note {
        margin-top: 0.85rem;
        color: rgba(55, 46, 35, 0.74);
        font-size: 0.9rem;
      }

      .hero-ask-shell {
        display: grid;
        gap: 0.78rem;
        margin-top: 1.15rem;
        text-align: left;
      }

      .hero-ask-shell label {
        margin: 0;
        font-size: 0.86rem;
      }

      .hero-ask-shell textarea,
      .hero-ask-shell select {
        border-radius: 13px;
        background: rgba(255, 253, 248, 0.96);
      }

      .hero-ask-shell textarea {
        min-height: 4.6rem;
        line-height: 1.58;
        padding: 0.88rem 0.94rem;
      }

      .hero-ask-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) repeat(3, minmax(132px, 1fr));
        gap: 0.72rem;
        align-items: end;
      }

      .hero-ask-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.72rem;
        align-items: center;
      }

      .hero-ask-button {
        appearance: none;
        width: auto;
        min-width: 160px;
        min-height: 3.15rem;
        border-radius: 13px;
        border: 1px solid rgba(184, 134, 67, 0.52);
        background: linear-gradient(180deg, #f4d99a 0%, #dfb86f 52%, #c98936 100%);
        color: #2a2012;
        font: inherit;
        font-weight: 700;
        padding: 0.8rem 1.15rem;
        transition:
          transform 180ms ease,
          border-color 180ms ease,
          background 220ms ease,
          box-shadow 220ms ease;
        box-shadow:
          0 14px 28px rgba(78, 57, 24, 0.16),
          inset 0 1px 0 rgba(255, 246, 219, 0.88),
          inset 0 -10px 20px rgba(131, 87, 30, 0.12);
      }

      .hero-ask-button:hover,
      .hero-ask-button:focus-visible {
        transform: translateY(-1px);
        border-color: rgba(173, 130, 71, 0.72);
        background: linear-gradient(180deg, #f7e1a8 0%, #e4bf79 52%, #cd913e 100%);
        box-shadow:
          0 20px 34px rgba(78, 57, 24, 0.2),
          inset 0 1px 0 rgba(255, 248, 226, 0.94),
          0 0 0 0.18rem rgba(214, 176, 96, 0.12);
      }

      .hero-ask-button:active {
        transform: translateY(1px);
      }

      .hero-quick-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        align-items: center;
      }

      .hero-quick-label {
        color: var(--muted-strong);
        font-size: 0.88rem;
        font-weight: 600;
      }

      .hero-prompt-chip {
        appearance: none;
        width: auto;
        border: 1px solid rgba(173, 130, 71, 0.18);
        border-radius: 999px;
        background: rgba(255, 252, 246, 0.92);
        color: var(--muted-strong);
        font: inherit;
        font-size: 0.88rem;
        line-height: 1.35;
        padding: 0.5rem 0.8rem;
        cursor: pointer;
        transition:
          transform 180ms ease,
          border-color 180ms ease,
          box-shadow 180ms ease,
          background 180ms ease;
      }

      .hero-prompt-chip:hover,
      .hero-prompt-chip:focus-visible {
        transform: translateY(-1px);
        border-color: rgba(173, 130, 71, 0.36);
        box-shadow: 0 12px 22px rgba(78, 57, 24, 0.1);
        background: rgba(255, 254, 250, 0.98);
      }

      .guide-panel,
      .trust-panel {
        position: relative;
        overflow: hidden;
        padding: 1.15rem 1.2rem;
        background:
          linear-gradient(180deg, rgba(255, 251, 243, 0.9), rgba(248, 240, 226, 0.9));
        border: 1px solid rgba(137, 100, 54, 0.18);
        border-radius: var(--radius);
        box-shadow:
          var(--shadow-soft),
          inset 0 1px 0 rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(8px);
        animation: settleIn 760ms ease both;
      }

      .guide-panel::before,
      .trust-panel::before,
      .source-card::before,
      .workspace-card::before,
      .status-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 5px;
        background: linear-gradient(90deg, rgba(210, 168, 93, 0.42), rgba(107, 124, 103, 0.14), transparent);
        pointer-events: none;
      }

      .guide-panel::after,
      .trust-panel::after,
      .source-card::after,
      .workspace-card::after,
      .status-card::after {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: calc(var(--radius) - 1px);
        pointer-events: none;
        background:
          linear-gradient(135deg, rgba(255, 255, 255, 0.2), transparent 28%, transparent 72%, rgba(215, 172, 95, 0.08));
        opacity: 0.78;
      }

      .guide-panel h2,
      .trust-panel h2 {
        margin: 0.55rem 0 0.3rem;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.45rem;
        font-weight: 600;
      }

      .guide-panel p,
      .trust-panel p {
        margin: 0;
        color: var(--muted);
        line-height: 1.7;
      }

      .hero-actions,
      .secondary-actions {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        transition:
          transform 180ms ease,
          box-shadow 180ms ease,
          background 220ms ease,
          border-color 220ms ease,
          color 220ms ease;
        cursor: pointer;
      }

      .hero-actions {
        min-height: 3.9rem;
        padding: 0.95rem 1.8rem;
        min-width: 17rem;
        border: 1px solid rgba(184, 134, 67, 0.62);
        background: linear-gradient(180deg, #f6dd9c 0%, #e4bc74 52%, #c98c37 100%);
        color: #2a2012;
        font-weight: 700;
        font-size: 1.02rem;
        box-shadow:
          0 18px 34px rgba(78, 57, 24, 0.22),
          inset 0 1px 0 rgba(255, 246, 219, 0.88),
          inset 0 -10px 20px rgba(131, 87, 30, 0.14);
      }

      .hero-actions:hover,
      .hero-actions:focus-visible {
        transform: translateY(-2px);
        box-shadow:
          0 24px 42px rgba(78, 57, 24, 0.26),
          inset 0 1px 0 rgba(255, 250, 232, 0.94),
          inset 0 -12px 24px rgba(131, 87, 30, 0.16),
          0 0 0 0.18rem rgba(228, 190, 111, 0.16);
        background: linear-gradient(180deg, #f8e2a6 0%, #e7c27d 52%, #ce9442 100%);
      }

      .hero-actions:active {
        transform: translateY(1px) scale(0.995);
      }

      .secondary-actions {
        min-height: 3.9rem;
        padding: 0.95rem 1.5rem;
        border: 1px solid rgba(137, 100, 54, 0.18);
        background: rgba(255, 251, 244, 0.8);
        color: var(--text);
        font-weight: 600;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
      }

      .secondary-actions:hover,
      .secondary-actions:focus-visible {
        transform: translateY(-1px);
        border-color: var(--line-strong);
        background: rgba(255, 251, 244, 0.96);
        box-shadow: 0 16px 28px rgba(78, 57, 24, 0.1);
      }

      .section-grid {
        display: grid;
        gap: 1.2rem;
      }

      .prompt-grid,
      .mode-grid,
      .signal-list {
        display: grid;
        gap: 0.9rem;
      }

      .prompt-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin-top: 1.15rem;
      }

      .prompt-card,
      .mode-card {
        width: 100%;
        text-align: left;
        border-radius: 13px;
        border: 1px solid rgba(173, 130, 71, 0.15);
        background:
          linear-gradient(180deg, rgba(255, 253, 248, 0.95), rgba(248, 241, 228, 0.92));
        color: var(--text);
        padding: 0.9rem 0.98rem;
        min-height: 8rem;
        transition:
          transform 180ms ease,
          border-color 180ms ease,
          background 220ms ease,
          box-shadow 220ms ease;
      }

      .prompt-card:hover,
      .prompt-card:focus-visible,
      .mode-card:hover,
      .mode-card:focus-visible {
        transform: translateY(-2px);
        border-color: rgba(173, 130, 71, 0.36);
        background: linear-gradient(180deg, rgba(255, 255, 252, 0.98), rgba(249, 243, 232, 0.98));
        box-shadow: 0 18px 30px rgba(78, 57, 24, 0.14);
      }

      .prompt-title,
      .mode-title {
        display: block;
        color: var(--text);
        font-weight: 700;
      }

      .prompt-copy,
      .mode-copy {
        display: block;
        margin-top: 0.3rem;
        color: var(--muted);
        line-height: 1.6;
      }

      .mode-grid {
        grid-template-columns: repeat(4, minmax(160px, 1fr));
        margin-top: 1rem;
      }

      .mode-card.active {
        border-color: rgba(173, 130, 71, 0.4);
        background: linear-gradient(180deg, rgba(249, 244, 234, 0.98), rgba(241, 233, 217, 0.98));
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.82),
          0 16px 28px rgba(78, 57, 24, 0.09);
      }

      .mode-switch-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.5rem;
      }

      .mode-switch {
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: rgba(255, 250, 242, 0.94);
        color: var(--muted-strong);
        border-radius: 999px;
        padding: 0.42rem 0.78rem;
        font-size: 0.84rem;
        transition:
          transform 180ms ease,
          border-color 180ms ease,
          box-shadow 180ms ease,
          background 180ms ease;
      }

      .mode-switch:hover,
      .mode-switch:focus-visible {
        transform: translateY(-1px);
        border-color: rgba(173, 130, 71, 0.34);
        box-shadow: 0 10px 22px rgba(78, 57, 24, 0.12);
      }

      .mode-switch.active {
        border-color: rgba(173, 130, 71, 0.38);
        background: linear-gradient(180deg, rgba(249, 244, 234, 0.98), rgba(241, 233, 217, 0.98));
        color: var(--text);
      }

      .trust-meta {
        margin-top: 0.95rem;
        color: var(--muted-strong);
        font-size: 0.93rem;
      }

      .signal-list {
        margin-top: 1rem;
      }

      .signal-card {
        border-radius: 18px;
        border: 1px solid rgba(173, 130, 71, 0.14);
        background: linear-gradient(180deg, rgba(255, 253, 248, 0.92), rgba(249, 241, 228, 0.9));
        padding: 0.9rem 1rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.68);
      }

      .signal-title {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 700;
        color: var(--text);
      }

      .signal-title::before {
        content: "";
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 50%;
        background: var(--success);
      }

      .signal-card.warning .signal-title::before {
        background: var(--accent);
      }

      .signal-card.error .signal-title::before {
        background: #cf8f86;
      }

      .signal-detail {
        margin-top: 0.35rem;
        color: var(--muted);
        line-height: 1.65;
      }

      .source-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: -1.2rem;
      }

      .panel,
      .source-card,
      .workspace-card,
      .status-card {
        background:
          linear-gradient(180deg, rgba(255, 249, 239, 0.88), rgba(244, 234, 214, 0.84));
        border: 1px solid rgba(137, 100, 54, 0.18);
        border-radius: var(--radius);
        box-shadow:
          var(--shadow-soft),
          inset 0 1px 0 rgba(255, 255, 255, 0.68);
        backdrop-filter: blur(14px) saturate(1.04);
        position: relative;
        overflow: hidden;
        animation: settleIn 780ms ease both;
      }

      .source-card {
        padding: 1.25rem 1.3rem;
      }

      .section-label {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.75rem;
      }

      .source-card h2,
      .workspace-heading,
      .status-heading {
        margin: 0.5rem 0 0.35rem;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.5rem;
        font-weight: 600;
      }

      .source-card p,
      .workspace-copy,
      .status-copy {
        margin: 0;
        color: var(--muted);
        line-height: 1.7;
      }

      .workspace-trust {
        margin: 0.55rem 0 0;
        color: var(--accent-deep);
        font-weight: 600;
        line-height: 1.55;
      }

      .workspace {
        grid-template-columns: minmax(0, 2.06fr) minmax(360px, 0.94fr);
        margin-top: 1.75rem;
      }

      .progressive-section {
        opacity: 0;
        transform: translateY(18px);
        max-height: 0;
        overflow: hidden;
        pointer-events: none;
        margin-top: 0;
        transition:
          opacity 360ms ease,
          transform 360ms ease,
          max-height 420ms ease,
          margin-top 360ms ease;
      }

      .desk-entry-shell {
        display: grid;
        gap: 0.9rem;
        opacity: 0;
        transform: translateY(14px);
        transition: opacity 320ms ease 70ms, transform 320ms ease 70ms;
      }

      .workspace-card,
      .status-card {
        padding: 1.55rem 1.58rem;
      }

      .mode-preview-shell[hidden] {
        display: none !important;
      }

      .mode-preview-toggle {
        appearance: none;
        border: 1px solid rgba(150, 112, 64, 0.2);
        border-radius: 12px;
        background: rgba(255, 252, 246, 0.94);
        color: var(--muted-strong);
        padding: 0.68rem 0.86rem;
        font: inherit;
        font-weight: 600;
        cursor: pointer;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
      }

      .mode-preview-toggle:hover,
      .mode-preview-toggle:focus-visible {
        transform: translateY(-1px);
        box-shadow: 0 12px 20px rgba(78, 57, 24, 0.08);
        border-color: rgba(173, 130, 71, 0.36);
      }

      .workspace-card.is-focused {
        border-color: rgba(173, 130, 71, 0.42);
        box-shadow:
          var(--shadow-soft),
          0 0 0 0.22rem rgba(200, 157, 81, 0.14),
          0 22px 40px rgba(74, 49, 18, 0.12);
      }

      .workspace-topline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 1rem;
      }

      .pulse {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--muted-strong);
        font-size: 0.95rem;
        padding: 0.55rem 0.85rem;
        border-radius: 999px;
        background: rgba(255, 251, 244, 0.82);
        border: 1px solid rgba(107, 124, 103, 0.2);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.66);
      }

      .pulse::before {
        content: "";
        width: 0.65rem;
        height: 0.65rem;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 0 0.3rem rgba(159, 191, 154, 0.14);
      }

      label {
        display: block;
        margin-top: 0.95rem;
        margin-bottom: 0.35rem;
        color: var(--muted-strong);
        font-weight: 600;
      }

      .think-shell {
        position: relative;
        padding: 0.38rem;
        border-radius: 15px;
        background:
          linear-gradient(180deg, rgba(255, 252, 245, 0.94), rgba(241, 230, 208, 0.86));
        border: 1px solid rgba(173, 130, 71, 0.18);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.74),
          0 18px 32px rgba(46, 28, 14, 0.18);
      }

      .control-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.12fr) minmax(0, 1fr) minmax(0, 1.12fr);
        gap: 0.95rem;
        margin-top: 0.42rem;
      }

      .field-note,
      .output-note {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.6;
      }

      .field-note {
        margin-top: 0.55rem;
      }

      input,
      textarea,
      select,
      button {
        width: 100%;
        font: inherit;
      }

      input,
      textarea,
      select {
        border-radius: 14px;
        border: 1px solid rgba(173, 130, 71, 0.18);
        background: rgba(255, 253, 248, 0.98);
        color: var(--text);
        padding: 0.95rem 1rem;
        outline: none;
        transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease, transform 180ms ease;
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.78),
          inset 0 -6px 14px rgba(155, 117, 61, 0.02);
      }

      input::placeholder,
      textarea::placeholder {
        color: rgba(45, 42, 34, 0.4);
      }

      textarea {
        min-height: 10.2rem;
        line-height: 1.72;
        resize: vertical;
      }

      input:focus,
      textarea:focus,
      select:focus,
      input:focus-visible,
      textarea:focus-visible,
      select:focus-visible {
        border-color: rgba(173, 130, 71, 0.45);
        box-shadow:
          0 0 0 0.24rem rgba(214, 176, 96, 0.15),
          0 14px 28px rgba(78, 57, 24, 0.1);
        background: rgba(255, 255, 252, 1);
        transform: translateY(-1px);
      }

      .ask-button {
        margin-top: 1rem;
        min-height: 3.4rem;
        border-radius: 14px;
        border: 1px solid rgba(184, 134, 67, 0.52);
        background: linear-gradient(180deg, #f4d99a 0%, #dfb86f 52%, #c98936 100%);
        color: #2a2012;
        font-weight: 700;
        transition:
          transform 180ms ease,
          border-color 180ms ease,
          background 220ms ease,
          box-shadow 220ms ease;
        box-shadow:
          0 16px 30px rgba(78, 57, 24, 0.18),
          inset 0 1px 0 rgba(255, 246, 219, 0.88),
          inset 0 -10px 20px rgba(131, 87, 30, 0.12);
      }

      .ask-button:hover,
      .ask-button:focus-visible {
        transform: translateY(-2px);
        border-color: rgba(173, 130, 71, 0.72);
        background: linear-gradient(180deg, #f7e1a8 0%, #e4bf79 52%, #cd913e 100%);
        box-shadow:
          0 22px 36px rgba(78, 57, 24, 0.22),
          inset 0 1px 0 rgba(255, 248, 226, 0.94),
          0 0 0 0.18rem rgba(214, 176, 96, 0.12);
      }

      .ask-button:active {
        transform: translateY(1px);
      }

      .output-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        margin-top: 1.35rem;
        margin-bottom: 0.82rem;
      }

      .output {
        min-height: 15rem;
      }

      .answer-surface {
        display: grid;
        gap: 1.05rem;
      }

      .learning-center-panel {
        display: grid;
        gap: 0.9rem;
      }

      .learning-center-subtitle {
        margin: 0;
        color: var(--muted-strong);
        font-size: 0.95rem;
        line-height: 1.6;
      }

      .learning-center-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }

      .learning-center-body {
        display: grid;
        gap: 0.82rem;
      }

      .learning-center-helper-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.58rem;
      }

      .learning-center-helper-button {
        width: auto;
        border-radius: 999px;
        border: 1px solid rgba(173, 130, 71, 0.18);
        background: rgba(255, 250, 243, 0.95);
        color: var(--accent-deep);
        padding: 0.5rem 0.82rem;
        font-size: 0.88rem;
        line-height: 1.2;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .learning-center-helper-button:hover,
      .learning-center-helper-button:focus-visible {
        border-color: rgba(173, 130, 71, 0.32);
        background: rgba(255, 247, 233, 0.98);
      }

      .learning-center-empty {
        margin: 0;
        padding: 0.9rem 0.96rem;
        border-radius: 13px;
        border: 1px dashed rgba(173, 130, 71, 0.22);
        background: rgba(255, 251, 245, 0.92);
        color: var(--muted-strong);
        line-height: 1.6;
      }

      .answer-shell {
        display: grid;
        gap: 1rem;
      }

      .answer-section-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
      }

      .answer-compare-grid,
      .answer-layer-grid,
      .answer-support-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
      }

      .answer-card {
        padding: 1.08rem 1.12rem;
        border-radius: 13px;
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: linear-gradient(180deg, rgba(255, 253, 248, 0.98), rgba(246, 238, 225, 0.95));
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.74),
          0 16px 28px rgba(78, 57, 24, 0.08);
      }

      .answer-card.wide {
        grid-column: 1 / -1;
      }

      .answer-summary-card {
        background:
          linear-gradient(180deg, rgba(255, 252, 245, 0.98), rgba(245, 235, 218, 0.95));
      }

      .answer-scholar-card {
        background:
          linear-gradient(180deg, rgba(249, 244, 234, 0.98), rgba(239, 230, 212, 0.96));
        border-color: rgba(148, 110, 53, 0.24);
      }

      .answer-summary-grid {
        display: grid;
        gap: 0.75rem;
      }

      .answer-summary-title {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.35rem;
        font-weight: 600;
        line-height: 1.2;
        color: var(--text);
      }

      .answer-summary-copy {
        margin: 0;
        color: var(--muted-strong);
        line-height: 1.7;
      }

      .answer-card-header {
        display: grid;
        gap: 0.35rem;
        margin-bottom: 0.65rem;
      }

      .answer-card-title {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text);
      }

      .answer-card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
      }

      .answer-alert-stack {
        display: grid;
        gap: 0.8rem;
        margin-bottom: 1rem;
      }

      .answer-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.38rem 0.68rem;
        border-radius: 999px;
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: rgba(255, 249, 240, 0.92);
        color: var(--muted-strong);
        font-size: 0.84rem;
      }

      .answer-copy,
      .answer-empty,
      .answer-list li,
      .answer-source-meta,
      .answer-source-quote,
      .answer-note,
      .answer-reading-meta,
      .answer-lesson-meta,
      .answer-lesson-copy {
        color: var(--muted-strong);
        line-height: 1.68;
      }

      .answer-copy {
        margin: 0;
      }

      .answer-list {
        margin: 0;
        padding-left: 1.1rem;
        display: grid;
        gap: 0.45rem;
      }

      .answer-source-list {
        display: grid;
        gap: 0.72rem;
      }

      .answer-source-item {
        padding: 0.82rem 0.88rem;
        border-radius: 14px;
        border: 1px solid rgba(173, 130, 71, 0.13);
        background: rgba(255, 250, 243, 0.92);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.74);
      }

      .answer-source-item[data-source-tone="primary"] {
        border-color: rgba(100, 132, 87, 0.24);
        background: linear-gradient(180deg, rgba(250, 253, 246, 0.97), rgba(242, 247, 238, 0.95));
      }

      .answer-source-item[data-source-tone="fiqh"] {
        border-color: rgba(173, 130, 71, 0.22);
      }

      .answer-source-item[data-source-tone="tasawwuf"] {
        border-color: rgba(72, 114, 82, 0.22);
        background: linear-gradient(180deg, rgba(248, 252, 246, 0.97), rgba(240, 246, 238, 0.95));
      }

      .answer-source-item[data-source-tone="commentary"] {
        border-color: rgba(119, 96, 60, 0.18);
      }

      .answer-source-item[data-source-tone="modern"] {
        border-color: rgba(180, 111, 65, 0.22);
        background: linear-gradient(180deg, rgba(255, 248, 244, 0.97), rgba(249, 238, 228, 0.95));
      }

      .answer-source-item[data-source-tone="teaching"] {
        border-color: rgba(97, 118, 158, 0.22);
        background: linear-gradient(180deg, rgba(246, 249, 255, 0.97), rgba(238, 243, 252, 0.95));
      }

      .answer-source-head,
      .answer-source-title-row,
      .answer-source-locator {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.5rem;
      }

      .answer-source-head {
        margin-bottom: 0.42rem;
      }

      .answer-source-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
      }

      .answer-source-badge,
      .answer-warning-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.2rem 0.56rem;
        font-size: 0.76rem;
        line-height: 1.2;
      }

      .answer-source-badge {
        background: rgba(255, 247, 233, 0.98);
        border: 1px solid rgba(173, 130, 71, 0.18);
        color: var(--accent-deep);
      }

      .answer-warning-badge {
        background: rgba(255, 240, 208, 0.98);
        border: 1px solid rgba(199, 140, 37, 0.24);
        color: #7b4e10;
      }

      .answer-source-title {
        font-weight: 700;
        color: var(--text);
      }

      .answer-source-page {
        color: var(--muted);
        font-size: 0.84rem;
        white-space: nowrap;
      }

      .answer-source-layer-title {
        margin: 0;
        font-size: 1.02rem;
        color: var(--text);
      }

      .answer-source-meta {
        margin-top: 0.24rem;
        font-size: 0.93rem;
      }

      .answer-source-locator {
        margin-top: 0.36rem;
        color: var(--muted);
        font-size: 0.85rem;
      }

      .answer-source-quote {
        margin-top: 0.42rem;
        padding-left: 0.72rem;
        border-left: 3px solid rgba(173, 130, 71, 0.22);
      }

      .answer-note,
      .answer-empty {
        margin: 0;
        font-size: 0.95rem;
      }

      .answer-inline-banner,
      .continuity-banner {
        padding: 0.82rem 0.96rem;
        border-radius: 13px;
        border: 1px solid rgba(173, 130, 71, 0.2);
        background: rgba(255, 250, 243, 0.94);
        color: var(--muted-strong);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.74);
      }

      .answer-inline-banner.warning {
        background: rgba(255, 242, 216, 0.96);
        border-color: rgba(198, 133, 27, 0.26);
        color: #6d4a10;
      }

      .continuity-banner strong,
      .answer-inline-banner strong {
        color: var(--text);
      }

      .scholar-meta-grid,
      .compare-position-grid,
      .request-phase-track {
        display: grid;
        gap: 0.8rem;
      }

      .scholar-meta-grid,
      .compare-position-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .scholar-meta-row {
        padding: 0.7rem 0.8rem;
        border-radius: 12px;
        background: rgba(255, 248, 237, 0.8);
        border: 1px solid rgba(173, 130, 71, 0.12);
      }

      .scholar-meta-label {
        display: block;
        margin-bottom: 0.22rem;
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
      }

      .compare-position-card {
        padding: 0.9rem 0.94rem;
        border-radius: 13px;
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: rgba(255, 250, 242, 0.94);
      }

      .compare-position-card .answer-card-title {
        font-size: 1.02rem;
      }

      .compare-position-list {
        display: grid;
        gap: 0.42rem;
        margin-top: 0.6rem;
      }

      .compare-position-source {
        padding: 0.62rem 0.7rem;
        border-radius: 11px;
        background: rgba(249, 243, 232, 0.92);
        border: 1px solid rgba(173, 130, 71, 0.1);
      }

      .request-phase-shell {
        margin-bottom: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: rgba(255, 250, 242, 0.9);
      }

      .request-phase-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        margin-bottom: 0.82rem;
      }

      .request-phase-copy {
        color: var(--muted-strong);
        font-size: 0.92rem;
      }

      .request-phase-track {
        grid-template-columns: repeat(5, minmax(0, 1fr));
      }

      .request-phase-step {
        min-height: 72px;
        padding: 0.72rem 0.76rem;
        border-radius: 12px;
        border: 1px solid rgba(173, 130, 71, 0.12);
        background: rgba(255, 250, 243, 0.72);
        transition:
          transform 180ms ease,
          border-color 180ms ease,
          box-shadow 180ms ease,
          background 180ms ease;
      }

      .request-phase-step.active {
        border-color: rgba(173, 130, 71, 0.3);
        background: linear-gradient(180deg, rgba(250, 244, 232, 0.98), rgba(242, 233, 216, 0.96));
        box-shadow: 0 16px 28px rgba(78, 57, 24, 0.09);
        transform: translateY(-1px);
      }

      .request-phase-step.complete {
        border-color: rgba(76, 115, 82, 0.26);
        background: rgba(244, 250, 241, 0.94);
      }

      .request-phase-step.degraded {
        border-color: rgba(199, 140, 37, 0.26);
        background: rgba(255, 242, 216, 0.96);
      }

      .request-phase-label {
        display: block;
        color: var(--text);
        font-weight: 700;
      }

      .request-phase-meta {
        display: block;
        margin-top: 0.28rem;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.5;
      }

      .answer-kicker {
        color: var(--accent-deep);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.76rem;
        font-weight: 700;
      }

      .answer-reading-group {
        display: grid;
        gap: 0.68rem;
      }

      .answer-reading-heading {
        font-size: 0.84rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-deep);
        font-weight: 700;
      }

      .answer-reading-list,
      .answer-lesson-list,
      .answer-bullet-list {
        display: grid;
        gap: 0.78rem;
      }

      .answer-reading-card,
      .answer-lesson-card,
      .answer-bullet-card {
        padding: 0.8rem 0.84rem;
        border-radius: 12px;
        border: 1px solid rgba(173, 130, 71, 0.13);
        background: rgba(255, 251, 244, 0.9);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .answer-reading-title,
      .answer-lesson-title {
        font-weight: 700;
        color: var(--text);
      }

      .answer-reading-meta,
      .answer-lesson-meta {
        margin-top: 0.25rem;
        font-size: 0.94rem;
      }

      .answer-lesson-copy {
        margin-top: 0.46rem;
      }

      .answer-footnote {
        margin-top: 0.72rem;
        font-size: 0.92rem;
        color: var(--muted);
        line-height: 1.65;
      }

      .answer-fallback-note {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 0.92rem;
      }

      pre,
      .output {
        white-space: pre-wrap;
        word-break: break-word;
      }

      pre {
        margin: 0;
        padding: 1.25rem 1.18rem;
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(255, 253, 248, 0.96), rgba(247, 239, 226, 0.94));
        border: 1px solid rgba(173, 130, 71, 0.16);
        color: var(--text);
        font-family: inherit;
        line-height: 1.68;
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.72),
          inset 0 -8px 16px rgba(149, 113, 59, 0.03);
      }

      .study-path-shell {
        display: grid;
        gap: 1rem;
      }

      .study-path-summary,
      .study-layer-shell,
      .study-support-card,
      .study-tab-panel {
        border-radius: 16px;
        border: 1px solid rgba(173, 130, 71, 0.18);
        background: linear-gradient(180deg, rgba(255, 253, 248, 0.97), rgba(246, 238, 225, 0.94));
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.72),
          0 18px 30px rgba(78, 57, 24, 0.08);
        padding: 1rem 1.05rem;
      }

      .study-summary-grid {
        display: grid;
        gap: 0.85rem;
      }

      .study-summary-title {
        margin: 0;
        font-size: 1.28rem;
        line-height: 1.25;
        color: var(--text);
      }

      .study-summary-copy {
        margin: 0;
        color: var(--muted-strong);
        line-height: 1.7;
      }

      .study-layer-shell {
        display: grid;
        gap: 0.95rem;
      }

      .study-layer-shell-header {
        display: grid;
        gap: 0.55rem;
      }

      .study-layer-shell-title {
        margin: 0;
        font-size: 1.08rem;
        color: var(--text);
      }

      .study-layer-shell-copy {
        margin: 0;
        color: var(--muted-strong);
        line-height: 1.66;
      }

      .study-summary-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
      }

      .study-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.78rem;
        border-radius: 999px;
        border: 1px solid rgba(173, 130, 71, 0.18);
        background: rgba(255, 250, 241, 0.9);
        color: var(--muted-strong);
        font-size: 0.87rem;
      }

      .study-tabs {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
        gap: 0.7rem;
      }

      .study-tab {
        appearance: none;
        border: 1px solid rgba(173, 130, 71, 0.18);
        background: rgba(255, 250, 241, 0.86);
        color: var(--muted-strong);
        border-radius: 14px;
        padding: 0.78rem 0.88rem;
        font: inherit;
        cursor: pointer;
        text-align: left;
        display: grid;
        gap: 0.24rem;
        min-height: 4.1rem;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
      }

      .study-tab:hover,
      .study-tab:focus-visible {
        transform: translateY(-1px);
        border-color: rgba(173, 130, 71, 0.36);
        box-shadow: 0 12px 20px rgba(78, 57, 24, 0.08);
      }

      .study-tab.active {
        background: linear-gradient(180deg, rgba(228, 192, 118, 0.28), rgba(255, 249, 239, 0.96));
        border-color: rgba(173, 130, 71, 0.42);
        color: var(--text);
      }

      .study-tab-label {
        font-weight: 700;
        color: var(--text);
      }

      .study-tab-meta {
        font-size: 0.86rem;
        line-height: 1.5;
        color: var(--muted);
      }

      .study-layer-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
      }

      .study-layer-title {
        margin: 0;
        font-size: 1.04rem;
        color: var(--text);
      }

      .study-layer-count {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.36rem 0.64rem;
        border-radius: 999px;
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: rgba(255, 251, 244, 0.9);
        font-size: 0.84rem;
        color: var(--muted-strong);
        white-space: nowrap;
      }

      .study-layer-note {
        margin: 0;
        color: var(--muted);
        line-height: 1.65;
      }

      .study-layer-summary {
        margin: 0 0 0.92rem;
        color: var(--muted-strong);
        line-height: 1.68;
      }

      .study-source-list,
      .study-reading-list,
      .study-lesson-list,
      .study-bullet-list {
        display: grid;
        gap: 0.8rem;
      }

      .study-source-card,
      .study-reading-card,
      .study-lesson-card,
      .study-bullet-card {
        padding: 0.8rem 0.86rem;
        border-radius: 12px;
        border: 1px solid rgba(173, 130, 71, 0.14);
        background: rgba(255, 251, 244, 0.88);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .study-source-title,
      .study-reading-title,
      .study-lesson-title {
        font-weight: 700;
        color: var(--text);
      }

      .study-source-meta,
      .study-reading-meta,
      .study-lesson-meta,
      .study-source-quote,
      .study-lesson-copy,
      .study-empty {
        margin-top: 0.35rem;
        color: var(--muted-strong);
        line-height: 1.65;
      }

      .study-source-quote {
        padding-left: 0.8rem;
        border-left: 3px solid rgba(173, 130, 71, 0.24);
      }

      .study-missing-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
      }

      .study-missing-card {
        padding: 0.84rem 0.88rem;
        border-radius: 12px;
        border: 1px solid rgba(173, 130, 71, 0.14);
        background: rgba(255, 251, 244, 0.88);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .study-missing-title {
        font-weight: 700;
        color: var(--text);
      }

      .study-missing-copy {
        margin-top: 0.34rem;
        color: var(--muted-strong);
        line-height: 1.62;
      }

      .study-layer-footnote {
        margin-top: 0.75rem;
        color: var(--muted);
        font-size: 0.9rem;
      }

      .study-support-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
      }

      .study-reading-group {
        display: grid;
        gap: 0.6rem;
        padding: 0.9rem;
        border-radius: 12px;
        border: 1px solid rgba(173, 130, 71, 0.12);
        background: rgba(255, 251, 244, 0.72);
      }

      .study-reading-heading {
        font-size: 0.92rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(131, 93, 45, 0.88);
      }

      .study-reading-copy {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
        font-size: 0.91rem;
      }

      .study-lesson-header {
        display: grid;
        gap: 0.45rem;
      }

      .study-lesson-step {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        width: fit-content;
        padding: 0.22rem 0.56rem;
        border-radius: 999px;
        border: 1px solid rgba(173, 130, 71, 0.16);
        background: rgba(255, 248, 236, 0.9);
        color: rgba(131, 93, 45, 0.92);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }

      .study-anchor-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.45rem;
      }

      .study-anchor-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.34rem 0.64rem;
        border-radius: 999px;
        border: 1px solid rgba(173, 130, 71, 0.14);
        background: rgba(255, 249, 238, 0.88);
        color: var(--muted-strong);
        font-size: 0.84rem;
        line-height: 1.45;
      }

      .status-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0 0;
      }

      .status-list li {
        padding: 0.78rem 0.88rem;
        border: 1px solid rgba(173, 130, 71, 0.14);
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(255, 253, 248, 0.92), rgba(247, 239, 226, 0.9));
        color: var(--muted-strong);
        margin-bottom: 0.7rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.68);
      }

      .status-list li:last-child {
        margin-bottom: 0;
      }

      .runtime-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 251, 244, 0.86);
        color: var(--muted-strong);
        font-size: 0.92rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .runtime-chip strong {
        color: var(--text);
      }

      .glance {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1.25rem;
      }

      .glance .source-card {
        min-height: 100%;
      }

      .link-row {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        color: #54674f;
        font-weight: 600;
        transition: transform 180ms ease, color 180ms ease;
      }

      .link-row:hover,
      .link-row:focus-visible {
        transform: translateX(2px);
        color: #43543f;
      }

      body[data-ui-stage="landing"] .secondary-actions,
      body[data-ui-stage="landing"] .trust-panel,
      body[data-ui-stage="landing"] .source-grid,
      body[data-ui-stage="landing"] .glance,
      body[data-ui-stage="landing"] .workspace {
        opacity: 0;
        transform: translateY(18px);
        max-height: 0;
        overflow: hidden;
        pointer-events: none;
        margin-top: 0;
      }

      body[data-ui-stage="landing"] .app-layout {
        width: min(1120px, calc(100% - 2rem));
        grid-template-columns: 1fr;
      }

      body[data-ui-stage="landing"] .workspace-sidebar {
        display: none;
      }

      body[data-ui-stage="landing"] .page-shell {
        width: min(840px, 100%);
        padding-top: 1rem;
      }

      body[data-ui-stage="workspace"] .workspace,
      body[data-ui-stage="engaged"] .workspace,
      body[data-ui-stage="engaged"] .source-grid,
      body[data-ui-stage="engaged"] .glance {
        opacity: 1;
        transform: translateY(0);
        max-height: 12000px;
        pointer-events: auto;
      }

      body[data-ui-stage="workspace"] .workspace,
      body[data-ui-stage="engaged"] .workspace {
        margin-top: 1.15rem;
      }

      body[data-ui-stage="workspace"] .workspace {
        max-width: min(860px, 100%);
        margin-inline: auto;
      }

      body[data-ui-stage="workspace"] .source-grid,
      body[data-ui-stage="workspace"] .glance,
      body[data-ui-stage="workspace"] .trust-panel {
        opacity: 0;
        transform: translateY(18px);
        max-height: 0;
        overflow: hidden;
        pointer-events: none;
        margin-top: 0;
      }

      body[data-ui-stage="landing"] .trust-panel,
      body[data-ui-stage="workspace"] .trust-panel {
        display: none;
      }

      body[data-ui-stage="workspace"] .hero,
      body[data-ui-stage="engaged"] .hero {
        min-height: auto;
        padding: 0 0 0.7rem;
      }

      body[data-ui-stage="workspace"] .hero-frame {
        width: min(500px, 100%);
        padding: 1.55rem 1.35rem 1.2rem;
        border-radius: 74px 74px 20px 20px / 56px 56px 20px 20px;
        box-shadow:
          0 14px 28px rgba(36, 21, 11, 0.16),
          0 8px 16px rgba(92, 56, 20, 0.08),
          inset 0 1px 0 rgba(255, 255, 255, 0.88);
      }

      body[data-ui-stage="engaged"] .hero-frame {
        width: min(620px, 100%);
        padding: 2rem 1.7rem 1.6rem;
        border-radius: 92px 92px 22px 22px / 72px 72px 22px 22px;
        box-shadow:
          0 18px 38px rgba(36, 21, 11, 0.2),
          0 10px 18px rgba(92, 56, 20, 0.1),
          inset 0 1px 0 rgba(255, 255, 255, 0.88);
      }

      body[data-ui-stage="workspace"] .hero-frame::before {
        inset: 11px;
        border-radius: 58px 58px 15px 15px / 42px 42px 15px 15px;
      }

      body[data-ui-stage="engaged"] .hero-frame::before {
        inset: 12px;
        border-radius: 74px 74px 16px 16px / 56px 56px 16px 16px;
      }

      body[data-ui-stage="workspace"] h1 {
        font-size: clamp(1.9rem, 3vw, 2.35rem);
        margin: 0.72rem 0 0.4rem;
      }

      body[data-ui-stage="engaged"] h1 {
        font-size: clamp(2.2rem, 4vw, 3rem);
        margin: 0.9rem 0 0.55rem;
      }

      body[data-ui-stage="workspace"] .supporting-text,
      body[data-ui-stage="engaged"] .supporting-text,
      body[data-ui-stage="workspace"] .hero-note,
      body[data-ui-stage="engaged"] .hero-note,
      body[data-ui-stage="workspace"] .hero-ask-shell,
      body[data-ui-stage="engaged"] .hero-ask-shell {
        display: none;
      }

      body[data-ui-stage="workspace"] .subtitle,
      body[data-ui-stage="engaged"] .subtitle {
        max-width: 31rem;
        font-size: 1rem;
        line-height: 1.55;
      }

      body[data-ui-stage="workspace"] .trust-line,
      body[data-ui-stage="engaged"] .trust-line {
        font-size: 0.94rem;
      }

      body[data-ui-stage="workspace"] .workspace-context,
      body[data-ui-stage="workspace"] #historyShell,
      body[data-ui-stage="workspace"] .output-header,
      body[data-ui-stage="workspace"] #requestPhaseShell,
      body[data-ui-stage="workspace"] .answer-surface,
      body[data-ui-stage="workspace"] #output,
      body[data-ui-stage="workspace"] .pulse,
      body[data-ui-stage="workspace"] .workspace-copy,
      body[data-ui-stage="workspace"] .workspace-trust,
      body[data-ui-stage="workspace"] #researchDepthField,
      body[data-ui-stage="workspace"] #researchDepthNote {
        display: none;
      }

      body[data-ui-stage="workspace"] .desk-entry-shell,
      body[data-ui-stage="engaged"] .desk-entry-shell {
        opacity: 1;
        transform: translateY(0);
      }

      body[data-ui-stage="workspace"] .desk-entry-shell {
        max-width: 760px;
        margin-inline: auto;
      }

      body[data-ui-stage="workspace"] .control-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      body[data-ui-stage="workspace"] .ask-button {
        max-width: 220px;
      }

      body[data-request-state="idle"] .workspace {
        grid-template-columns: minmax(0, 1fr);
        max-width: min(1040px, 100%);
        margin-inline: auto;
      }

      body[data-request-state="idle"] #statusPanel {
        display: none;
      }

      body[data-request-state="active"] #statusPanel {
        display: block;
      }

      body[data-sidebar-state="collapsed"] .app-layout {
        grid-template-columns: minmax(132px, 132px) minmax(0, 1fr);
        gap: 1rem;
      }

      body[data-ui-stage="landing"][data-sidebar-state="collapsed"] .app-layout {
        grid-template-columns: 1fr;
      }

      body[data-sidebar-state="collapsed"] .sidebar-stack {
        display: none;
      }

      body[data-sidebar-state="collapsed"] .sidebar-rail {
        justify-items: stretch;
      }

      body[data-sidebar-state="collapsed"] .sidebar-toggle {
        text-align: left;
        padding-inline: 0.78rem;
      }

      .sidebar-toggle:disabled {
        cursor: default;
        transform: none;
        box-shadow: none;
        opacity: 0.84;
      }

      .hero-frame,
      .landing-guides,
      .source-grid,
      .workspace-card,
      .status-card {
        will-change: transform, opacity;
      }

      .landing-guides { animation: settleIn 820ms ease both; }
      .source-grid { animation: settleIn 900ms ease both; }
      .workspace-card { animation: settleIn 980ms ease both; }
      .status-card { animation: settleIn 1060ms ease both; }

      @keyframes settleIn {
        from {
          opacity: 0;
          transform: translateY(18px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      :focus-visible {
        outline: 3px solid rgba(173, 130, 71, 0.28);
        outline-offset: 3px;
      }

      @media (max-width: 920px) {
        .app-layout {
          width: min(100%, calc(100% - 1rem));
          grid-template-columns: 1fr;
        }

        body[data-sidebar-state="collapsed"] .app-layout {
          grid-template-columns: 1fr;
        }

        .workspace-sidebar {
          position: static;
          padding-bottom: 0;
        }

        .landing-guides,
        .source-grid,
        .workspace,
        .glance,
        .prompt-grid,
        .mode-grid,
        .hero-ask-grid,
        .history-answer-grid,
        .history-answer-layers,
        .history-answer-support {
          grid-template-columns: 1fr;
        }

        .hero {
          min-height: auto;
          padding-top: 1rem;
        }

        .hero-frame {
          border-radius: 48px;
          padding-top: 2.35rem;
        }

        .hero-frame::before {
          inset: 14px;
          border-radius: 36px;
        }

        .control-grid {
          grid-template-columns: 1fr;
        }

        .study-support-grid {
          grid-template-columns: 1fr;
        }

        .answer-section-grid {
          grid-template-columns: 1fr;
        }

        .answer-layer-grid,
        .answer-support-grid {
          grid-template-columns: 1fr;
        }
      }

      @media (prefers-reduced-motion: reduce) {
        html {
          scroll-behavior: auto;
        }

        *,
        *::before,
        *::after {
          transition: none !important;
          animation: none !important;
        }
      }
    </style>
  </head>
  <body data-ui-stage="landing" data-sidebar-state="collapsed" data-request-state="idle">
    <div class="app-layout">
      <aside class="workspace-sidebar" id="workspaceSidebar" aria-label="Workspace sidebar" hidden>
        <div class="sidebar-rail">
          <button class="sidebar-toggle" id="sidebarToggle" type="button" aria-expanded="false" onclick="toggleSidebar()" disabled>
            <span class="sidebar-rail-title" id="sidebarRailTitle">Open Workspace</span>
            <span class="sidebar-rail-copy" id="sidebarRailCopy">Chats, projects, and settings.</span>
          </button>
        </div>
        <div class="sidebar-stack" id="sidebarStack" aria-hidden="true">
          <section class="sidebar-card" id="authPanel">
            <div id="sidebarAuthState">
              <div class="section-label">Local Access</div>
              <h2 class="sidebar-title">Loading workspace access</h2>
              <p class="sidebar-copy">Checking whether a saved local session is available.</p>
            </div>
          </section>

          <section class="sidebar-card" aria-label="Recent chats">
            <div class="sidebar-card-header">
              <div>
                <div class="section-label">Chats</div>
                <h2 class="sidebar-title">Recent chats</h2>
              </div>
              <button class="sidebar-button" type="button" onclick="createNewChat()">New Chat</button>
            </div>
            <div class="sidebar-helper" id="chatListHelper">
              Sign in to keep persistent chat history across refreshes.
            </div>
            <div class="sidebar-divider"></div>
            <div class="sidebar-list" id="chatList">
              <div class="sidebar-empty">No persistent chats yet.</div>
            </div>
          </section>

          <section class="sidebar-card" aria-label="Study projects">
            <div class="sidebar-card-header">
              <div>
                <div class="section-label">Study Projects</div>
                <h2 class="sidebar-title">Learning projects</h2>
              </div>
              <button class="sidebar-secondary-button" type="button" onclick="toggleProjectComposer()">Create Project</button>
            </div>
            <div class="sidebar-field-grid" id="projectComposer" hidden>
              <div>
                <label for="projectTitle">Project title</label>
                <input id="projectTitle" type="text" placeholder="Prayer Study" />
              </div>
              <div>
                <label for="projectDescription">Goal / description</label>
                <textarea id="projectDescription" rows="3" placeholder="Beginner path for prayer, study order, and grounded reading anchors."></textarea>
              </div>
              <div>
                <label for="projectMadhhab">Project madhhab</label>
                <select id="projectMadhhab">
                  <option value="hanafi">hanafi</option>
                  <option value="shafii">shafii</option>
                  <option value="maliki">maliki</option>
                  <option value="hanbali">hanbali</option>
                  <option value="not_specified">not_specified</option>
                </select>
              </div>
              <div>
                <label for="projectStudyMode">Project mode</label>
                <select id="projectStudyMode">
                  <option value="standard">standard</option>
                  <option value="study_path">study_path</option>
                </select>
              </div>
              <div class="sidebar-inline-actions">
                <button class="sidebar-button" type="button" onclick="createProject()">Save Project</button>
                <button class="sidebar-secondary-button" type="button" onclick="toggleProjectComposer(false)">Cancel</button>
              </div>
            </div>
            <div class="sidebar-divider"></div>
            <div class="sidebar-list" id="projectList">
              <div class="sidebar-empty">No study projects yet.</div>
            </div>
            <div class="sidebar-inline-actions" id="projectActions" hidden>
              <button class="sidebar-secondary-button" type="button" onclick="clearProjectFilter()">Show All Chats</button>
              <button class="sidebar-secondary-button" type="button" onclick="linkActiveChatToProject()">Link Active Chat</button>
            </div>
            <div class="sidebar-divider"></div>
            <div class="project-detail-shell" id="projectDetailPanel">
              <div class="sidebar-empty">Select a study project to see saved lessons, progress, and grounded reading anchors.</div>
            </div>
          </section>

          <section class="sidebar-card" aria-label="Settings">
            <div class="section-label">Settings</div>
            <h2 class="sidebar-title">Workspace defaults</h2>
            <div class="sidebar-settings-grid" id="settingsPanel">
              <div class="sidebar-empty">Sign in to save default madhhab, answer mode, and profile settings.</div>
            </div>
          </section>
        </div>
      </aside>

      <div class="page-shell">
      <section class="hero" aria-labelledby="hero-title">
        <div class="hero-frame">
          <div class="eyebrow">Grounded Islamic Research</div>
          <h1 id="hero-title">Halal Jordan</h1>
          <p class="subtitle">
            A source-cited Islamic research assistant built for clarity, discipline, and trust.
          </p>
          <p class="supporting-text">
            Search Qur'an, hadith, fiqh manuals, and clearly labeled commentary with visible source
            layers and grounded answers.
          </p>
          <p class="trust-line">Grounded in primary texts and scholarly sources.</p>
          <div class="hero-ask-shell" aria-label="Quick ask">
            <div>
              <label for="landingQuestion">Question</label>
              <textarea id="landingQuestion" rows="2" placeholder="Ask a grounded question. Example: How should I pray?">What does the Qur'an say about the straight path?</textarea>
            </div>
            <div class="hero-ask-grid">
              <div>
                <label for="landingMode">Answer mode</label>
                <select id="landingMode">
                  <option value="research">Research</option>
                  <option value="source_only">Source Only</option>
                  <option value="compare_views">Compare Views</option>
                  <option value="scholar_perspective">Scholar Perspective</option>
                  <option value="quick_answer">Quick Answer</option>
                  <option value="deep_study">Deep Study</option>
                  <option value="study_path">Study Path</option>
                </select>
              </div>
              <div>
                <label for="landingMadhhab">Madhhab</label>
                <select id="landingMadhhab">
                  <option value="not_specified">not_specified</option>
                  <option value="hanafi">hanafi</option>
                  <option value="shafii">shafii</option>
                  <option value="maliki">maliki</option>
                  <option value="hanbali">hanbali</option>
                  <option value="compare_all">compare_all</option>
                </select>
              </div>
              <div>
                <label for="landingResearchDepth">Research Depth</label>
                <select id="landingResearchDepth">
                  <option value="quick_source_check">Quick Source Check</option>
                  <option value="balanced_research" selected>Balanced Research</option>
                  <option value="deep_research">Deep Research</option>
                </select>
              </div>
              <div class="hero-ask-actions">
                <button class="hero-ask-button" id="heroAskButton" type="button" onclick="runHeroAsk()">Ask</button>
              </div>
            </div>
            <div class="hero-quick-row" aria-label="Example prompts">
              <span class="hero-quick-label">Examples:</span>
              <button class="hero-prompt-chip" type="button" onclick="launchHeroExample('Show me the source for Mala Budda Minhu.', 'research', 'hanafi', 'quick_source_check')">Mala Budda Minhu</button>
              <button class="hero-prompt-chip" type="button" onclick="launchHeroExample('Give me only primary source text on ablution. No synthesis.', 'source_only', 'not_specified', 'quick_source_check')">Primary source ablution</button>
              <button class="hero-prompt-chip" type="button" onclick="launchHeroExample('Compare Hanafi and Shafi\\'i views on wiping over socks.', 'compare_views', 'hanafi', 'deep_research')">Compare wiping over socks</button>
            </div>
          </div>
          <div class="cta-row">
            <a
              class="hero-actions"
              href="#ask-workspace"
              id="launchButton"
              onclick="enterHalalJordan(event)"
            >
              Focus Research Desk
            </a>
            <a class="secondary-actions" href="#ask-workspace" onclick="focusWorkspace(event)">Open Full Desk</a>
          </div>
        </div>
      </section>
      <section class="section-grid workspace progressive-section" id="ask-workspace" aria-label="Research desk" hidden>
        <div class="workspace-card">
          <div class="workspace-topline">
            <div>
              <div class="section-label">Research Desk</div>
              <h2 class="workspace-heading">Manager-facing grounded ask flow</h2>
              <p class="workspace-copy">
                Enter a question, choose a mode, and review the answer with live request status.
              </p>
              <p class="workspace-trust">Grounded in primary texts and scholarly sources.</p>
            </div>
            <div class="pulse" id="pulse">System ready.</div>
          </div>

          <div class="workspace-context" id="workspaceContext">
            Working anonymously. Sign in to keep chat history, study projects, and preferences across refreshes.
          </div>

          <div class="retrieval-status-banner" id="retrievalStatusBanner" data-state="loading" role="status" aria-live="polite">
            <strong>Index loading.</strong>
            Grounded answers will unlock when the persisted source index is ready.
          </div>

          <section class="history-shell" id="historyShell" aria-label="Conversation history">
            <div class="history-header">
              <div>
                <div class="section-label">Conversation History</div>
                <h3 class="history-heading" id="historyTitle">Current session</h3>
              </div>
              <div class="sidebar-meta" id="historyMeta">History will appear here once a saved session is active.</div>
            </div>
            <div class="history-list" id="conversationHistory">
              <div class="sidebar-empty">Sign in and start a chat to keep a structured conversation history.</div>
            </div>
          </section>

          <div class="desk-entry-shell" id="deskEntryShell">
            <label for="question">Question</label>
            <div class="think-shell">
              <textarea id="question" rows="5">What does the Qur'an say about the straight path?</textarea>
            </div>
            <div class="field-note">
              Start with a source, concept, ruling, or comparison. The grounded path stays visible as you go.
            </div>

            <div class="control-grid">
              <div class="control-field">
                <label for="mode">Answer mode</label>
                <select id="mode">
                  <option value="research">Research</option>
                  <option value="source_only">Source Only</option>
                  <option value="compare_views">Compare Views</option>
                  <option value="scholar_perspective">Scholar Perspective</option>
                  <option value="quick_answer">Quick Answer</option>
                  <option value="deep_study">Deep Study</option>
                  <option value="study_path">Study Path</option>
                </select>
              </div>

              <div class="control-field">
                <label for="madhhab">Selected madhhab</label>
                <select id="madhhab">
                  <option value="not_specified">not_specified</option>
                  <option value="hanafi">hanafi</option>
                  <option value="shafii">shafii</option>
                  <option value="maliki">maliki</option>
                  <option value="hanbali">hanbali</option>
                  <option value="compare_all">compare_all</option>
                </select>
              </div>

              <div class="control-field" id="researchDepthField">
                <label for="researchDepth">Research Depth</label>
                <select id="researchDepth">
                  <option value="quick_source_check">Quick Source Check</option>
                  <option value="balanced_research" selected>Balanced Research</option>
                  <option value="deep_research">Deep Research</option>
                </select>
              </div>
            </div>
            <div class="field-note" id="researchDepthNote">
              Balanced Research: Recommended. Strong grounding with practical speed.
            </div>

            <button class="ask-button" id="askButton" type="button" onclick="runAsk()">Ask</button>
          </div>
          <div class="output-header">
            <div>
              <div class="section-label">Rendered Answer</div>
              <div class="output-note">Live request truth remains visible in the status panel beside the answer.</div>
            </div>
            <div class="answer-header-controls">
              <div class="answer-mode-indicator" id="answerModeIndicator">Active mode: Research</div>
              <div class="mode-switch-row" aria-label="Quick answer mode switch">
                <button class="mode-switch active" data-mode-switch="research" type="button" onclick="switchDeskMode('research')">Research</button>
                <button class="mode-switch" data-mode-switch="source_only" type="button" onclick="switchDeskMode('source_only')">Source Only</button>
                <button class="mode-switch" data-mode-switch="compare_views" type="button" onclick="switchDeskMode('compare_views')">Compare Views</button>
                <button class="mode-switch" data-mode-switch="scholar_perspective" type="button" onclick="switchDeskMode('scholar_perspective')">Scholar Perspective</button>
              </div>
            </div>
          </div>
          <section class="request-phase-shell" id="requestPhaseShell" aria-live="polite" hidden>
            <div class="request-phase-header">
              <div class="section-label">Request Progress</div>
              <div class="request-phase-copy" id="requestPhaseCopy">Waiting to begin.</div>
            </div>
            <div class="request-phase-track" id="requestPhaseTrack"></div>
          </section>
          <div class="answer-surface">
            <div class="study-path-shell" id="studyPathPanel" hidden>
              <div class="answer-alert-stack" id="studyAlertStack"></div>
              <div class="study-path-summary" id="studyPathSummary"></div>
              <section class="study-layer-shell">
                <div class="study-layer-shell-header">
                  <div>
                    <div class="section-label">Source Layers</div>
                    <h3 class="study-layer-shell-title">Read by source family</h3>
                  </div>
                  <p class="study-layer-shell-copy">Source layers appear only when grounded evidence exists. Primary texts remain primary, fiqh stays legal study, tasawwuf stays spiritual guidance, and scholar commentary stays secondary.</p>
                </div>
                <div class="study-tabs" id="studyTabs" role="tablist" aria-label="Study source layers"></div>
                <div class="study-tab-panel" id="studyTabPanel" role="tabpanel" aria-live="polite"></div>
              </section>
              <section class="study-support-card">
                <div class="section-label">Missing Layers</div>
                <div id="studyMissingLayers"></div>
              </section>
              <div class="study-support-grid">
                <section class="study-support-card">
                  <div class="section-label">Reading List</div>
                  <div id="studyReadingList"></div>
                </section>
                <section class="study-support-card">
                  <div class="section-label">Lesson Path</div>
                  <div id="studyLessonPath"></div>
                </section>
              </div>
              <div class="study-support-grid">
                <section class="study-support-card">
                  <div class="section-label">Key Terms</div>
                  <div id="studyKeyTerms"></div>
                </section>
                <section class="study-support-card">
                  <div class="section-label">What to Avoid</div>
                  <div id="studyAvoid"></div>
                </section>
              </div>
              <section class="study-support-card">
                <div class="section-label">Uncertainty / Corpus Gaps</div>
                <div id="studyGaps"></div>
              </section>
            </div>
            <pre class="output" id="output">Ready.</pre>
            <div class="answer-shell" id="answerStructured" hidden>
              <div class="answer-alert-stack" id="answerAlertStack"></div>
              <section class="answer-card answer-scholar-card wide" id="answerScholarBlock" hidden></section>
              <section class="answer-card answer-summary-card wide" id="answerSummary"></section>
              <div class="answer-section-grid" id="answerSectionGrid"></div>
              <div class="answer-compare-grid" id="answerCompareGrid"></div>
              <div class="answer-layer-grid" id="answerLayerGrid"></div>
              <div class="answer-support-grid" id="answerSupportGrid"></div>
            </div>
            <section class="answer-card wide learning-center-panel" id="learningCenterPanel" hidden>
              <div class="answer-card-header">
                <div>
                  <div class="section-label">Learning Center</div>
                  <h3 class="answer-card-title">Teaching / Explanation</h3>
                </div>
                <div class="learning-center-meta" id="learningCenterMeta"></div>
              </div>
              <p class="learning-center-subtitle">Explanations, class-style notes, and study support. Not a ruling authority.</p>
              <div class="learning-center-body" id="learningCenterBody"></div>
              <div class="learning-center-helper-row" aria-label="Learning Center helper prompts">
                <button class="learning-center-helper-button" type="button" onclick="launchLearningHelper('explain_ruling')">Explain this ruling</button>
                <button class="learning-center-helper-button" type="button" onclick="launchLearningHelper('teach_reasoning')">Teach me the reasoning</button>
                <button class="learning-center-helper-button" type="button" onclick="launchLearningHelper('study_notes')">Show study notes</button>
                <button class="learning-center-helper-button" type="button" onclick="launchLearningHelper('uncertainty')">What is uncertain?</button>
              </div>
            </section>
          </div>
        </div>

        <aside class="status-card" id="statusPanel">
          <div class="section-label">Live Request State</div>
          <h2 class="status-heading">Request status</h2>
          <p class="status-copy">
            The manager view shows honest progress through retrieval, grounding, model use, and
            degraded operation when relevant.
          </p>
          <ul class="status-list" id="statusList">
            <li>Idle</li>
          </ul>
        </aside>
      </section>

      <section class="section-grid progressive-section" id="runtimeTrustPanel" aria-label="Corpus and trust state" hidden>
        <aside class="trust-panel" aria-live="polite">
          <div class="section-label">Corpus and Trust State</div>
          <h2>Enter with live corpus truth in view.</h2>
          <p id="landingGlanceText">
            Loading current runtime, retrieval posture, and corpus readiness.
          </p>
          <div class="runtime-chip" id="runtimeChip">
            <strong>Status:</strong> Loading
          </div>
          <div class="trust-meta" id="runtimeMeta">Waiting for the latest landing glance.</div>
          <div class="signal-list" id="trustSignals">
            <div class="signal-card">
              <div class="signal-title">Loading trust signals</div>
              <div class="signal-detail">Corpus, OCR posture, and compare-views readiness will appear here.</div>
            </div>
          </div>
        </aside>
      </section>

      <section class="section-grid source-grid progressive-section" id="sourceCommitmentSection" aria-label="Source commitment" hidden>
        <article class="source-card">
          <div class="section-label">Primary Texts</div>
          <h2>Qur'an and hadith stay distinct.</h2>
          <p>
            The answer path preserves primary text, translated primary text, hadith collection text,
            and later layers instead of blending them into one undifferentiated claim.
          </p>
        </article>
        <article class="source-card">
          <div class="section-label">Fiqh Discipline</div>
          <h2>Madhhab-aware without overclaiming.</h2>
          <p>
            Hanafi-first behavior stays visible when selected, while disagreement and uncertainty
            remain plainly labeled whenever the evidence is mixed or incomplete.
          </p>
        </article>
        <article class="source-card">
          <div class="section-label">Grounding Promise</div>
          <h2>Operator truth remains inspectable.</h2>
          <p>
            Retrieval status, degraded paths, source layers, and runtime honesty remain intact for
            operators without turning the manager experience into an admin console.
          </p>
        </article>
      </section>

      <section class="glance progressive-section" id="runtimeGlance" aria-label="Runtime glance" hidden>
        <article class="source-card" id="modePreviewCard">
          <div class="section-label">Mode Preview</div>
          <h2>Choose the framing before you ask.</h2>
          <p>
            Research balances concise synthesis and citations. Source-only keeps interpretation to a
            minimum. Compare views preserves distinct positions instead of blending them together.
          </p>
          <button class="mode-preview-toggle" id="modePreviewToggle" type="button" aria-expanded="false" onclick="toggleModePreview()">
            Show mode preview
          </button>
          <div class="mode-preview-shell" id="modePreviewPanel" hidden>
            <div class="mode-grid" id="modePreview">
              <button class="mode-card active" data-mode-card="research" type="button" onclick="previewMode('research')">
                <span class="mode-title">Research</span>
                <span class="mode-copy">Balanced answer with citations and concise synthesis.</span>
              </button>
              <button class="mode-card" data-mode-card="source_only" type="button" onclick="previewMode('source_only')">
                <span class="mode-title">Source-only</span>
                <span class="mode-copy">Minimal framing with direct excerpts and metadata first.</span>
              </button>
              <button class="mode-card" data-mode-card="compare_views" type="button" onclick="previewMode('compare_views')">
                <span class="mode-title">Compare views</span>
                <span class="mode-copy">Distinct position groups with disagreement kept visible.</span>
              </button>
              <button class="mode-card" data-mode-card="scholar_perspective" type="button" onclick="previewMode('scholar_perspective')">
                <span class="mode-title">Scholar perspective</span>
                <span class="mode-copy">Attributed synthesis drawn only from recorded works and positions.</span>
              </button>
            </div>
          </div>
        </article>
        <article class="source-card">
          <div class="section-label">Current Posture</div>
          <h2>Ready for grounded asks</h2>
          <p id="runtimeGlanceSummary">
            The landing state will show current retrieval health and corpus scope.
          </p>
          <div class="trust-meta" id="corpusScopeText">
            Loading admitted corpus counts and compare-views readiness.
          </div>
        </article>
        <article class="source-card">
          <div class="section-label">Entry Path</div>
          <h2>Begin in the research desk</h2>
          <p>
            The launch button, prompt cards, and mode preview all move into the same manager-facing
            workspace below, keeping the transition calm and deliberate.
          </p>
          <a class="link-row" href="#ask-workspace" onclick="enterHalalJordan(event)">Go to research desk</a>
        </article>
      </section>
      </div>
    </div>
    <script>
      let currentStream = null;
      let workspaceHighlightTimer = null;
      const initialAuthView = "__INITIAL_AUTH_VIEW__";
      const scholarProfileMap = __SCHOLAR_PROFILE_MAP__;
      const requestPhaseDefinitions = [
        {key: 'receiving', label: 'Receiving', copy: 'Validating the request and preparing grounded routing.'},
        {key: 'retrieving', label: 'Retrieving', copy: 'Collecting grounded source candidates from the live corpus.'},
        {key: 'grounding', label: 'Grounding', copy: 'Assembling evidence layers and checking retrieval integrity.'},
        {key: 'querying', label: 'Querying Model', copy: 'Passing the grounded pack into the answer model.'},
        {key: 'generating', label: 'Generating', copy: 'Rendering the response with source-aware answer structure.'},
      ];
      const workspaceState = {
        authenticated: false,
        user: null,
        workspace: null,
        activeChatId: null,
        activeProjectId: null,
        activeSession: null,
        authView: initialAuthView === 'register' ? 'register' : 'login',
        userAdjustedMode: false,
        userAdjustedMadhhab: false,
        enteredWorkspace: false,
        hasAsked: false,
        requestActive: false,
        sidebarExpanded: false,
        examplesExpanded: false,
        modePreviewExpanded: false,
        currentRequestPhase: 'receiving',
        currentRequestPhaseCopy: 'Waiting to begin.',
        degradedRequestPath: false,
        retrievalReady: false,
        retrievalStage: 'loading_persisted_index',
        retrievalError: '',
      };

      function formatTimestamp(value) {
        if (!value) {
          return 'unknown time';
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
          return value;
        }
        return date.toLocaleString();
      }

      function pushStatus(message) {
        const list = document.getElementById('statusList');
        const item = document.createElement('li');
        item.textContent = message;
        list.prepend(item);
        while (list.children.length > 8) {
          list.removeChild(list.lastChild);
        }
      }

      function renderRetrievalStatus(status) {
        const banner = document.getElementById('retrievalStatusBanner');
        const pulse = document.getElementById('pulse');
        if (!banner) {
          return;
        }
        const ready = Boolean(status && status.ready);
        const stage = String((status && status.stage) || 'loading_persisted_index');
        const error = nonEmptyString(status && status.error);
        const changed =
          workspaceState.retrievalReady !== ready ||
          workspaceState.retrievalStage !== stage ||
          workspaceState.retrievalError !== error;
        workspaceState.retrievalReady = ready;
        workspaceState.retrievalStage = stage;
        workspaceState.retrievalError = error;

        if (error) {
          banner.hidden = false;
          banner.dataset.state = 'error';
          banner.innerHTML = `<strong>Source index failed to load. See diagnostics.</strong>${error ? ` ${escapeHtml(error)}` : ''}`;
          if (pulse && !workspaceState.requestActive) {
            pulse.textContent = 'Source index failed to load. See diagnostics.';
          }
        } else if (!ready) {
          banner.hidden = false;
          banner.dataset.state = 'loading';
          banner.innerHTML = '<strong>Index loading.</strong> Loading retrieval index from the persisted source index.';
          if (pulse && !workspaceState.requestActive) {
            pulse.textContent = 'Index loading...';
          }
        } else {
          banner.hidden = true;
          banner.dataset.state = 'ready';
          banner.innerHTML = '<strong>Index ready.</strong> Grounded source search is available.';
          if (pulse && !workspaceState.requestActive) {
            pulse.textContent = 'Grounded source index ready.';
          }
        }

        if (changed) {
          loadLandingGlance();
        }
      }

      async function refreshRetrievalStatus() {
        try {
          const response = await fetch('/retrieval/status');
          const payload = await response.json();
          renderRetrievalStatus(payload);
        } catch (_) {
          renderRetrievalStatus({
            ready: false,
            stage: 'status_unavailable',
            error: 'Retrieval status endpoint is unavailable.'
          });
        }
      }

      function highlightWorkspace() {
        const workspaceCard = document.querySelector('.workspace-card');
        workspaceCard.classList.add('is-focused');
        if (workspaceHighlightTimer) {
          window.clearTimeout(workspaceHighlightTimer);
        }
        workspaceHighlightTimer = window.setTimeout(() => {
          workspaceCard.classList.remove('is-focused');
        }, 1800);
      }

      function modeDisplayLabel(mode) {
        const labels = {
          research: 'Research',
          source_only: 'Source Only',
          compare_views: 'Compare Views',
          scholar_perspective: 'Scholar Perspective',
          quick_answer: 'Quick Answer',
          deep_study: 'Deep Study',
          study_path: 'Study Path',
        };
        return labels[mode] || cleanDisplayLabel(mode || 'research');
      }

      function syncDeskModeUi(mode) {
        const indicator = document.getElementById('answerModeIndicator');
        if (indicator) {
          indicator.textContent = 'Active mode: ' + modeDisplayLabel(mode);
        }
        document.querySelectorAll('[data-mode-switch]').forEach((button) => {
          button.classList.toggle('active', button.getAttribute('data-mode-switch') === mode);
        });
      }

      function setModeSelection(mode) {
        const modeField = document.getElementById('mode');
        const landingModeField = document.getElementById('landingMode');
        if (modeField) {
          modeField.value = mode;
        }
        if (landingModeField) {
          landingModeField.value = mode;
        }
        document.querySelectorAll('[data-mode-card]').forEach((card) => {
          card.classList.toggle('active', card.getAttribute('data-mode-card') === mode);
        });
        syncDeskModeUi(mode);
      }

      function switchDeskMode(mode) {
        setModeSelection(mode);
        document.getElementById('pulse').textContent = modeDisplayLabel(mode) + ' mode prepared.';
        pushStatus(modeDisplayLabel(mode) + ' mode selected from the research desk.');
      }

      function normalizeRequestPhase(rawPhase) {
        const phase = String(rawPhase || '').trim().toLowerCase();
        if (!phase) {
          return 'receiving';
        }
        if (phase.includes('receiving')) {
          return 'receiving';
        }
        if (phase.includes('retriev')) {
          return 'retrieving';
        }
        if (phase.includes('ground')) {
          return 'grounding';
        }
        if (phase.includes('query')) {
          return 'querying';
        }
        if (phase.includes('generat') || phase.includes('degraded_source_backed') || phase.includes('completed')) {
          return 'generating';
        }
        return 'receiving';
      }

      function renderRequestPhaseIndicator() {
        const shell = document.getElementById('requestPhaseShell');
        const copy = document.getElementById('requestPhaseCopy');
        const track = document.getElementById('requestPhaseTrack');
        if (!shell || !copy || !track) {
          return;
        }
        if (!workspaceState.requestActive && !workspaceState.hasAsked) {
          shell.hidden = true;
          track.innerHTML = '';
          copy.textContent = 'Waiting to begin.';
          return;
        }
        shell.hidden = false;
        const activeIndex = requestPhaseDefinitions.findIndex((entry) => entry.key === workspaceState.currentRequestPhase);
        copy.textContent = workspaceState.currentRequestPhaseCopy || 'Grounded request in progress.';
        track.innerHTML = requestPhaseDefinitions.map((entry, index) => {
          const classes = ['request-phase-step'];
          if (index < activeIndex) {
            classes.push('complete');
          }
          if (index === activeIndex) {
            classes.push(workspaceState.degradedRequestPath ? 'degraded' : 'active');
          }
          return `
            <article class="${classes.join(' ')}">
              <span class="request-phase-label">${escapeHtml(entry.label)}</span>
              <span class="request-phase-meta">${escapeHtml(entry.copy)}</span>
            </article>
          `;
        }).join('');
      }

      function setRequestPhase(phase, summary, degraded) {
        workspaceState.currentRequestPhase = normalizeRequestPhase(phase);
        workspaceState.currentRequestPhaseCopy = summary || 'Grounded request in progress.';
        workspaceState.degradedRequestPath = Boolean(degraded);
        renderRequestPhaseIndicator();
      }

      function setResearchDepthSelection(depth) {
        const researchDepthField = document.getElementById('researchDepth');
        const landingResearchDepthField = document.getElementById('landingResearchDepth');
        const researchDepthNote = document.getElementById('researchDepthNote');
        const normalizedDepth = depth || 'balanced_research';
        const descriptions = {
          quick_source_check: 'Quick Source Check: Fastest. Best for direct source lookups and simple questions.',
          balanced_research: 'Balanced Research: Recommended. Strong grounding with practical speed.',
          deep_research: 'Deep Research: Slowest. Best for nuanced, comparative, or layered questions.'
        };
        if (researchDepthField) {
          researchDepthField.value = normalizedDepth;
        }
        if (landingResearchDepthField) {
          landingResearchDepthField.value = normalizedDepth;
        }
        if (researchDepthNote) {
          researchDepthNote.textContent = descriptions[normalizedDepth] || descriptions.balanced_research;
        }
      }

      function setMadhhabSelection(madhhab) {
        const normalized = madhhab || 'not_specified';
        const madhhabField = document.getElementById('madhhab');
        const landingMadhhabField = document.getElementById('landingMadhhab');
        if (madhhabField) {
          madhhabField.value = normalized;
        }
        if (landingMadhhabField) {
          landingMadhhabField.value = normalized;
        }
      }

      function syncHeroAskToDesk() {
        const landingQuestion = document.getElementById('landingQuestion');
        const deskQuestion = document.getElementById('question');
        if (landingQuestion && deskQuestion) {
          deskQuestion.value = landingQuestion.value;
        }
        setModeSelection(document.getElementById('landingMode').value);
        setMadhhabSelection(document.getElementById('landingMadhhab').value);
        setResearchDepthSelection(document.getElementById('landingResearchDepth').value);
      }

      function syncDeskAskToHero() {
        const landingQuestion = document.getElementById('landingQuestion');
        const deskQuestion = document.getElementById('question');
        if (landingQuestion && deskQuestion) {
          landingQuestion.value = deskQuestion.value;
        }
        setModeSelection(document.getElementById('mode').value);
        setMadhhabSelection(document.getElementById('madhhab').value);
        setResearchDepthSelection(document.getElementById('researchDepth').value);
      }

      function currentUiStage() {
        if (!workspaceState.enteredWorkspace) {
          return 'landing';
        }
        return workspaceState.hasAsked ? 'engaged' : 'workspace';
      }

      function syncUiState() {
        const stage = currentUiStage();
        if (stage !== 'engaged') {
          workspaceState.sidebarExpanded = false;
        }
        document.body.setAttribute('data-ui-stage', stage);
        document.body.setAttribute('data-sidebar-state', workspaceState.sidebarExpanded ? 'expanded' : 'collapsed');
        document.body.setAttribute('data-request-state', workspaceState.requestActive ? 'active' : 'idle');
        const workspaceSidebar = document.getElementById('workspaceSidebar');
        if (workspaceSidebar) {
          workspaceSidebar.hidden = stage === 'landing';
        }
        const askWorkspace = document.getElementById('ask-workspace');
        if (askWorkspace) {
          askWorkspace.hidden = stage === 'landing';
        }
        const runtimeTrustPanel = document.getElementById('runtimeTrustPanel');
        if (runtimeTrustPanel) {
          runtimeTrustPanel.hidden = stage !== 'engaged';
        }
        const sourceCommitmentSection = document.getElementById('sourceCommitmentSection');
        if (sourceCommitmentSection) {
          sourceCommitmentSection.hidden = stage !== 'engaged';
        }
        const runtimeGlance = document.getElementById('runtimeGlance');
        if (runtimeGlance) {
          runtimeGlance.hidden = stage !== 'engaged';
        }
        const sidebarToggle = document.getElementById('sidebarToggle');
        if (sidebarToggle) {
          sidebarToggle.setAttribute('aria-expanded', workspaceState.sidebarExpanded ? 'true' : 'false');
          sidebarToggle.disabled = stage !== 'engaged';
        }
        const sidebarStack = document.getElementById('sidebarStack');
        if (sidebarStack) {
          sidebarStack.setAttribute('aria-hidden', workspaceState.sidebarExpanded && stage === 'engaged' ? 'false' : 'true');
        }
        const sidebarRailTitle = document.getElementById('sidebarRailTitle');
        const sidebarRailCopy = document.getElementById('sidebarRailCopy');
        if (sidebarRailTitle) {
          sidebarRailTitle.textContent = stage === 'engaged' ? 'Open Workspace' : 'Workspace';
        }
        if (sidebarRailCopy) {
          if (stage === 'workspace') {
            sidebarRailCopy.textContent = 'Saved chats and study tools open after the first grounded answer.';
          } else if (stage === 'engaged') {
            sidebarRailCopy.textContent = 'Chats, projects, and settings.';
          } else {
            sidebarRailCopy.textContent = 'Chats, projects, and settings.';
          }
        }
        const modePreviewPanel = document.getElementById('modePreviewPanel');
        const modePreviewToggle = document.getElementById('modePreviewToggle');
        if (modePreviewPanel) {
          modePreviewPanel.hidden = !workspaceState.modePreviewExpanded;
        }
        if (modePreviewToggle) {
          modePreviewToggle.setAttribute('aria-expanded', workspaceState.modePreviewExpanded ? 'true' : 'false');
          modePreviewToggle.textContent = workspaceState.modePreviewExpanded ? 'Hide mode preview' : 'Show mode preview';
        }
        renderRequestPhaseIndicator();
      }

      function setEnteredWorkspace(value) {
        workspaceState.enteredWorkspace = value !== false;
        syncUiState();
      }

      function markAsked() {
        workspaceState.hasAsked = true;
        syncUiState();
      }

      function setRequestActive(active) {
        workspaceState.requestActive = Boolean(active);
        syncUiState();
      }

      function toggleSidebar(force) {
        if (currentUiStage() !== 'engaged') {
          return;
        }
        workspaceState.sidebarExpanded = typeof force === 'boolean'
          ? force
          : !workspaceState.sidebarExpanded;
        syncUiState();
      }

      function toggleModePreview(force) {
        workspaceState.modePreviewExpanded = typeof force === 'boolean'
          ? force
          : !workspaceState.modePreviewExpanded;
        syncUiState();
      }

      function focusWorkspace(eventOrOptions, maybeOptions) {
        if (eventOrOptions && typeof eventOrOptions.preventDefault === 'function') {
          eventOrOptions.preventDefault();
        }
        const options = eventOrOptions && typeof eventOrOptions.preventDefault === 'function'
          ? (maybeOptions || {})
          : (eventOrOptions || {});
        const focusQuestion = !options || options.focusQuestion !== false;
        setEnteredWorkspace(true);
        const workspace = document.getElementById('ask-workspace');
        workspace.scrollIntoView({behavior: 'smooth', block: 'start'});
        window.setTimeout(() => {
          highlightWorkspace();
          if (focusQuestion) {
            document.getElementById('question').focus();
          }
        }, 220);
      }

      function enterHalalJordan(event) {
        event.preventDefault();
        if (currentUiStage() === 'landing') {
          const landingQuestion = document.getElementById('landingQuestion');
          const hasText = landingQuestion && landingQuestion.value.trim();
          if (hasText) {
            runHeroAsk();
            return;
          }
          if (landingQuestion) {
            landingQuestion.focus();
          }
          document.getElementById('pulse').textContent = 'Type a question here, or open the full desk below.';
          pushStatus('Hero ask box focused.');
          return;
        }
        document.getElementById('pulse').textContent = 'Research desk ready.';
        pushStatus('Entered the research desk.');
        focusWorkspace({focusQuestion: true});
      }

      function previewMode(mode) {
        const labels = {
          research: 'Research mode prepared.',
          source_only: 'Source-only mode prepared.',
          compare_views: 'Compare-views mode prepared.',
          scholar_perspective: 'Scholar Perspective mode prepared.',
          study_path: 'Study Path mode prepared.'
        };
        setModeSelection(mode);
        document.getElementById('pulse').textContent = labels[mode] || 'Mode prepared.';
        pushStatus(labels[mode] || 'Mode prepared.');
        focusWorkspace({focusQuestion: true});
      }

      function launchPreset(question, mode, madhhab, researchDepth) {
        document.getElementById('question').value = question;
        document.getElementById('landingQuestion').value = question;
        setMadhhabSelection(madhhab || 'not_specified');
        setModeSelection(mode || 'research');
        setResearchDepthSelection(researchDepth || 'balanced_research');
        resetAnswerSurface('Featured prompt loaded. Review the prompt or ask immediately.');
        document.getElementById('pulse').textContent = 'Featured prompt ready.';
        pushStatus('Featured prompt loaded into the research desk.');
        focusWorkspace({focusQuestion: true});
      }

      function launchHeroExample(question, mode, madhhab, researchDepth) {
        document.getElementById('landingQuestion').value = question;
        document.getElementById('question').value = question;
        setModeSelection(mode || 'research');
        setMadhhabSelection(madhhab || 'not_specified');
        setResearchDepthSelection(researchDepth || 'balanced_research');
        const landingQuestion = document.getElementById('landingQuestion');
        if (landingQuestion) {
          landingQuestion.focus();
          landingQuestion.setSelectionRange(landingQuestion.value.length, landingQuestion.value.length);
        }
        document.getElementById('pulse').textContent = 'Example loaded into the ask box.';
        pushStatus('Example prompt loaded on the landing screen.');
      }

      function learningHelperBaseQuestion() {
        const deskQuestion = nonEmptyString(document.getElementById('question').value);
        const landingQuestion = nonEmptyString(document.getElementById('landingQuestion').value);
        return deskQuestion || landingQuestion || 'this topic';
      }

      function launchLearningHelper(kind) {
        const baseQuestion = learningHelperBaseQuestion();
        const prompts = {
          explain_ruling: `Explain this ruling: ${baseQuestion}`,
          teach_reasoning: `Teach me the reasoning behind: ${baseQuestion}`,
          study_notes: `Show study notes for: ${baseQuestion}`,
          uncertainty: `What is uncertain about: ${baseQuestion}`,
        };
        const nextQuestion = prompts[kind] || `Explain this: ${baseQuestion}`;
        const questionField = document.getElementById('question');
        const landingQuestionField = document.getElementById('landingQuestion');
        if (questionField) {
          questionField.value = nextQuestion;
        }
        if (landingQuestionField) {
          landingQuestionField.value = nextQuestion;
        }
        if (document.getElementById('mode').value !== 'source_only') {
          workspaceState.userAdjustedMode = true;
          setModeSelection('research');
        }
        syncDeskAskToHero();
        document.getElementById('pulse').textContent = 'Learning helper prepared in the ask box.';
        pushStatus('Learning Center helper prompt loaded.');
        focusWorkspace({focusQuestion: true});
      }

      async function runHeroAsk() {
        const landingQuestion = document.getElementById('landingQuestion');
        if (!landingQuestion || !landingQuestion.value.trim()) {
          landingQuestion && landingQuestion.focus();
          document.getElementById('pulse').textContent = 'Type a question first.';
          pushStatus('Waiting for a grounded question.');
          return;
        }
        syncHeroAskToDesk();
        await runAsk();
      }

      function renderTrustSignals(signals) {
        const root = document.getElementById('trustSignals');
        root.innerHTML = '';
        if (!signals || !signals.length) {
          root.innerHTML = '<div class="signal-card"><div class="signal-title">No trust signals available</div><div class="signal-detail">The landing summary could not load corpus truth.</div></div>';
          return;
        }
        signals.forEach((signal) => {
          const card = document.createElement('div');
          card.className = 'signal-card ' + (signal.status || 'info');
          const title = document.createElement('div');
          title.className = 'signal-title';
          title.textContent = signal.title;
          const detail = document.createElement('div');
          detail.className = 'signal-detail';
          detail.textContent = signal.detail;
          card.appendChild(title);
          card.appendChild(detail);
          root.appendChild(card);
        });
      }

      async function loadLandingGlance() {
        try {
          const response = await fetch('/api/landing/glance');
          const data = await response.json();
          const runtime = data.runtime || {};
          const corpus = data.corpus || {};
          const compareViews = data.compare_views || {};
          const overall = runtime.status || 'unknown';
            const retrieval = runtime.retrieval_status || 'unknown';
            const retrievalWarmupStatus = runtime.retrieval_warmup_status || 'not_started';
            const selectedPort = runtime.selected_port || 'unknown';
            const retrievalFailed = retrieval === 'error';
            const pulseSummary = runtime.system_pulse && runtime.system_pulse.summary
              ? runtime.system_pulse.summary
              : 'Runtime truth loaded.';

            document.getElementById('runtimeChip').innerHTML =
              '<strong>Status:</strong> ' + overall + ' / retrieval ' + retrieval + ' / warmup ' + retrievalWarmupStatus + ' / port ' + selectedPort;
          document.getElementById('landingGlanceText').textContent =
            retrievalFailed
              ? 'Source index failed to load. See diagnostics.'
              : retrievalWarmupStatus === 'warming'
              ? 'Index loading from the persisted source index. Grounded asks become fully ready when warmup completes.'
              : pulseSummary;
          document.getElementById('runtimeGlanceSummary').textContent =
            retrievalFailed
              ? 'The app is live, but the source index failed to load and grounded retrieval is unavailable until diagnostics are fixed.'
              : retrievalWarmupStatus === 'warming'
              ? 'The runtime is live, but the source index is still loading before grounded source search is fully ready.'
              : overall === 'ready'
              ? 'The current runtime is ready and the admitted corpus is searchable.'
              : 'The current runtime is degraded, so the landing experience is showing cautionary state.';
            document.getElementById('runtimeMeta').textContent =
              String(corpus.documents || 0) + ' admitted documents, ' +
              String(corpus.chunks || 0) + ' retrieval chunks. Last reload: ' +
              formatTimestamp(runtime.last_reload_at) + '. Selected port: ' + selectedPort + '.';
          document.getElementById('corpusScopeText').textContent =
            compareViews.broad_ready
              ? 'Broader compare-views coverage is available in the current admitted corpus.'
              : 'Compare-views remains limited outside the current Hanafi-heavy slice; non-Hanafi admitted documents: ' +
                String(compareViews.admitted_non_hanafi_documents || 0) + '.';
          renderTrustSignals(data.trust_signals || []);
        } catch (_) {
          document.getElementById('runtimeChip').innerHTML =
            '<strong>Status:</strong> unavailable';
          document.getElementById('landingGlanceText').textContent =
            'Runtime and corpus truth could not be loaded from the local service.';
          document.getElementById('runtimeGlanceSummary').textContent =
            'The landing page is keeping a cautious empty state.';
          document.getElementById('runtimeMeta').textContent =
            'No live landing summary is available right now.';
          document.getElementById('corpusScopeText').textContent =
            'Corpus scope is unavailable until the landing summary can be loaded.';
          renderTrustSignals([]);
        }
      }

      function getSessionId() {
        let sessionId = window.localStorage.getItem('hj-manager-session-id');
        if (!sessionId) {
          sessionId = crypto.randomUUID();
          window.localStorage.setItem('hj-manager-session-id', sessionId);
        }
        return sessionId;
      }

      function escapeHtml(value) {
        return String(value || '')
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }

      function escapeRegex(value) {
        return String(value || '').replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
      }

      function normalizeWhitespace(value) {
        return String(value || '').replace(/\s+/g, ' ').trim();
      }

      function applySafeDisplayReplacements(value) {
        let text = String(value || '');
        const replacements = {
          auih: 'Allah',
          'au&': 'Allah',
          im8m: 'Imam',
          qtblah: 'Qiblah',
          wth: 'with'
        };
        Object.entries(replacements).forEach(([source, target]) => {
          text = text.replace(new RegExp(`\\b${escapeRegex(source)}\\b`, 'gi'), target);
        });
        return text;
      }

      function trimDisplayToken(token) {
        return String(token || '').replace(/^[\[\]{}()<>\"'`.,;:!?|]+|[\[\]{}()<>\"'`.,;:!?|]+$/g, '');
      }

      function hasMixedLettersAndDigits(token) {
        return /[A-Za-z]/.test(token) && /\d/.test(token);
      }

      function looksLikeGarbledLabelToken(token) {
        if (!token || token.length <= 2) {
          return false;
        }
        if (hasMixedLettersAndDigits(token)) {
          return true;
        }
        const symbolCount = Array.from(token).filter((char) => !/[A-Za-z0-9]/.test(char) && char !== '-' && char !== "'").length;
        if (symbolCount >= 2) {
          return true;
        }
        if (token.length >= 5 && !/[aeiou]/i.test(token) && /[A-Za-z]/.test(token)) {
          return true;
        }
        return false;
      }

      function looksLikeGarbledExcerptToken(token) {
        if (!token || token.length <= 3) {
          return false;
        }
        if (hasMixedLettersAndDigits(token)) {
          return true;
        }
        const symbolCount = Array.from(token).filter((char) => !/[A-Za-z0-9]/.test(char) && char !== '-' && char !== "'").length;
        return symbolCount >= 3;
      }

      function cleanDisplayText(value, options) {
        const excerpt = Boolean(options && options.excerpt);
        const original = normalizeWhitespace(applySafeDisplayReplacements(value));
        if (!original) {
          return '';
        }
        const cleanedTokens = [];
        original.split(' ').forEach((token) => {
          const cleaned = trimDisplayToken(token);
          if (!cleaned) {
            return;
          }
          if ((excerpt ? looksLikeGarbledExcerptToken(cleaned) : looksLikeGarbledLabelToken(cleaned))) {
            return;
          }
          cleanedTokens.push(cleaned);
        });
        const collapsed = normalizeWhitespace(cleanedTokens.join(' '));
        if (!collapsed) {
          return original;
        }
        const minimumLength = excerpt ? Math.max(12, Math.floor(original.length * 0.6)) : Math.max(8, Math.floor(original.length * 0.45));
        if (collapsed.length < minimumLength) {
          return original;
        }
        return collapsed;
      }

      function cleanDisplayLabel(value) {
        return cleanDisplayText(value, {excerpt: false});
      }

      function cleanDisplayExcerpt(value) {
        return cleanDisplayText(value, {excerpt: true});
      }

      function nonEmptyString(value) {
        const normalized = normalizeWhitespace(value);
        return normalized || '';
      }

      function uniqueNonEmpty(values) {
        const seen = new Set();
        const output = [];
        values.forEach((value) => {
          const normalized = nonEmptyString(value);
          if (!normalized) {
            return;
          }
          const key = normalized.toLowerCase();
          if (seen.has(key)) {
            return;
          }
          seen.add(key);
          output.push(normalized);
        });
        return output;
      }

      function citationSourceLabel(citation) {
        const sourceType = String(citation && citation.source_type ? citation.source_type : '').toLowerCase();
        if (sourceType === 'quran') {
          return "Qur'an";
        }
        if (sourceType === 'hadith') {
          return 'Hadith';
        }
        if (sourceType === 'fiqh_manual') {
          return 'Fiqh Manual';
        }
        if (sourceType === 'tasawwuf_text') {
          return 'Tasawwuf';
        }
        if (sourceType === 'scholar_transcript') {
          return 'Scholar Work';
        }
        if (sourceType === 'commentary') {
          return 'Commentary';
        }
        if (sourceType === 'fatwa') {
          return 'Fatwa';
        }
        if (sourceType === 'transcript') {
          return 'Scholar Work';
        }
        return 'Grounded Source';
      }

      function groupedCitationsForDisplay(citations) {
        const groups = {
          primary_texts: [],
          fiqh: [],
          tasawwuf: [],
          teaching_explanation: [],
          scholar_commentary: [],
          commentary: [],
          modern_application: []
        };
        (Array.isArray(citations) ? citations : []).forEach((citation) => {
          if (isTeachingCitation(citation)) {
            groups.teaching_explanation.push(citation);
            return;
          }
          const sourceType = String(citation && citation.source_type ? citation.source_type : '').toLowerCase();
          if (sourceType === 'quran' || sourceType === 'hadith') {
            groups.primary_texts.push(citation);
            return;
          }
          if (sourceType === 'fiqh_manual') {
            groups.fiqh.push(citation);
            return;
          }
          if (sourceType === 'tasawwuf_text') {
            groups.tasawwuf.push(citation);
            return;
          }
          if (sourceType === 'scholar_transcript' || sourceType === 'transcript') {
            groups.scholar_commentary.push(citation);
            return;
          }
          if (sourceType === 'fatwa') {
            groups.modern_application.push(citation);
            return;
          }
          groups.commentary.push(citation);
        });
        return groups;
      }

      function derivedMissingLayers(citations) {
        const present = new Set();
        (Array.isArray(citations) ? citations : []).forEach((citation) => {
          const sourceType = String(citation && citation.source_type ? citation.source_type : '').toLowerCase();
          if (sourceType === 'quran') {
            present.add('quran');
          } else if (sourceType === 'hadith') {
            present.add('hadith');
          } else if (sourceType === 'fiqh_manual') {
            present.add('fiqh');
          } else if (sourceType === 'tasawwuf_text') {
            present.add('tasawwuf');
          } else if (sourceType === 'scholar_transcript' || sourceType === 'transcript') {
            present.add('scholar_commentary');
          }
        });
        const definitions = [
          {key: 'quran', label: "Qur'an"},
          {key: 'hadith', label: 'Hadith'},
          {key: 'fiqh', label: 'Fiqh'},
          {key: 'tasawwuf', label: 'Tasawwuf'},
          {key: 'scholar_commentary', label: 'Scholar Commentary'}
        ];
        return {
          presentCount: present.size,
          missing: definitions.filter((entry) => !present.has(entry.key))
        };
      }

      function citationSourceTone(citation) {
        if (isTeachingCitation(citation)) {
          return 'teaching';
        }
        const sourceType = String(citation && citation.source_type ? citation.source_type : '').toLowerCase();
        if (sourceType === 'quran' || sourceType === 'hadith') {
          return 'primary';
        }
        if (sourceType === 'fiqh_manual') {
          return 'fiqh';
        }
        if (sourceType === 'tasawwuf_text') {
          return 'tasawwuf';
        }
        if (sourceType === 'fatwa') {
          return 'modern';
        }
        return 'commentary';
      }

      function sourceWarningLabel(citation) {
        const extractionStatus = String(citation && citation.extraction_status ? citation.extraction_status : '').toLowerCase();
        const extractionQuality = String(citation && citation.extraction_quality ? citation.extraction_quality : '').toLowerCase();
        const ocrStatus = String(citation && citation.ocr_status ? citation.ocr_status : '').toLowerCase();
        if (extractionStatus === 'partial') {
          return 'Partial extraction';
        }
        if (extractionQuality === 'machine_extract_with_gaps' || extractionQuality === 'ocr_recovered_with_gaps') {
          return 'OCR / extraction gaps';
        }
        if (ocrStatus && !['clean', 'not_attempted', 'not_needed', 'not_applicable', 'none', 'ok'].includes(ocrStatus)) {
          return 'OCR-derived excerpt';
        }
        return '';
      }

      function buildSourceLocator(citation) {
        return uniqueNonEmpty([
          cleanDisplayLabel(citation.reference || ''),
          cleanDisplayLabel(citation.book || ''),
          cleanDisplayLabel(citation.chapter || ''),
          cleanDisplayLabel(citation.section || ''),
          cleanDisplayLabel(citation.section_label || ''),
        ]).map(escapeHtml).join(' • ');
      }

      function buildSourceMeta(citation) {
        return uniqueNonEmpty([
          cleanDisplayLabel(citation.collection || ''),
          cleanDisplayLabel(citation.author || ''),
          cleanDisplayLabel(citation.madhhab || ''),
        ]).map(escapeHtml).join(' • ');
      }

      function buildSourceCard(citation) {
        const title = escapeHtml(cleanDisplayLabel(citation.title || 'Retrieved Source'));
        const meta = buildSourceMeta(citation);
        const locator = buildSourceLocator(citation);
        const quote = cleanDisplayExcerpt(citation.quote || '');
        const sourceLabel = escapeHtml(citationSourceLabel(citation));
        const tone = escapeHtml(citationSourceTone(citation));
        const warning = sourceWarningLabel(citation);
        const pageReference = escapeHtml(cleanDisplayLabel(citation.page_reference || citation.page_number || ''));
        return `
          <article class="answer-source-item" data-source-tone="${tone}">
            <div class="answer-source-head">
              <div class="answer-source-badges">
                <span class="answer-source-badge">${sourceLabel}</span>
                ${warning ? `<span class="answer-warning-badge">${escapeHtml(warning)}</span>` : ''}
              </div>
              ${pageReference ? `<div class="answer-source-page">${pageReference}</div>` : ''}
            </div>
            <div class="answer-source-title-row">
              <div class="answer-source-title">${title}</div>
            </div>
            ${locator ? `<div class="answer-source-locator">${locator}</div>` : ''}
            <div class="answer-source-meta">${meta || 'Grounded source metadata available.'}</div>
            ${quote ? `<div class="answer-source-quote">${escapeHtml(quote)}</div>` : ''}
          </article>
        `;
      }

      function hasStructuredAnswer(answer, data) {
        if (!answer || typeof answer !== 'object') {
          return false;
        }
        if (Array.isArray(answer.citations) && answer.citations.length) {
          return true;
        }
        return Boolean(
          nonEmptyString(answer.direct_answer)
          || nonEmptyString(answer.madhhab_position)
          || nonEmptyString(answer.evidence_summary)
          || nonEmptyString(answer.uncertainty_note)
          || nonEmptyString(answer.evidence_strength)
          || (data && data.diagnostics && Object.keys(data.diagnostics).length)
        );
      }

      function resetAnswerSurface(message) {
        const output = document.getElementById('output');
        const structured = document.getElementById('answerStructured');
        const studyPathPanel = document.getElementById('studyPathPanel');
        const learningCenterPanel = document.getElementById('learningCenterPanel');
        output.hidden = false;
        output.textContent = message || 'Ready.';
        structured.hidden = true;
        studyPathPanel.hidden = true;
        learningCenterPanel.hidden = true;
        document.getElementById('answerSummary').innerHTML = '';
        document.getElementById('answerSectionGrid').innerHTML = '';
        document.getElementById('answerCompareGrid').innerHTML = '';
        document.getElementById('answerLayerGrid').innerHTML = '';
        document.getElementById('answerSupportGrid').innerHTML = '';
        document.getElementById('answerAlertStack').innerHTML = '';
        document.getElementById('answerScholarBlock').innerHTML = '';
        document.getElementById('answerScholarBlock').hidden = true;
        document.getElementById('studyPathSummary').innerHTML = '';
        document.getElementById('studyAlertStack').innerHTML = '';
        document.getElementById('studyTabs').innerHTML = '';
        document.getElementById('studyTabPanel').innerHTML = '';
        document.getElementById('studyMissingLayers').innerHTML = '';
        document.getElementById('studyReadingList').innerHTML = '';
        document.getElementById('studyLessonPath').innerHTML = '';
        document.getElementById('studyKeyTerms').innerHTML = '';
        document.getElementById('studyAvoid').innerHTML = '';
        document.getElementById('studyGaps').innerHTML = '';
        document.getElementById('learningCenterMeta').innerHTML = '';
        document.getElementById('learningCenterBody').innerHTML = '';
      }

      function isTeachingCitation(citation) {
        const roleBoundary = String(citation && citation.source_role_boundary ? citation.source_role_boundary : '').toLowerCase();
        const authorityLayer = String(citation && citation.authority_layer ? citation.authority_layer : '').toLowerCase();
        const sourceType = String(citation && citation.source_type ? citation.source_type : '').toLowerCase();
        const sourceClassification = String(citation && citation.source_classification ? citation.source_classification : '').toLowerCase();
        return roleBoundary === 'teaching_layer'
          || authorityLayer === 'explanatory_teaching'
          || sourceType === 'class_material'
          || sourceClassification === 'class_material';
      }

      function resolveRenderedAnswerText(data) {
        const answer = data && data.answer ? data.answer : {};
        return nonEmptyString(data && data.rendered_text)
          || nonEmptyString(answer.direct_answer)
          || nonEmptyString(answer.evidence_summary)
          || nonEmptyString(answer.madhhab_position)
          || '';
      }

      function renderRenderedAnswerText(text) {
        const output = document.getElementById('output');
        if (!output) {
          return;
        }
        output.hidden = false;
        output.textContent = text || 'No rendered answer is available yet.';
      }

      function setAskButtonsPending(active) {
        [
          ['heroAskButton', 'Ask'],
          ['askButton', 'Ask'],
        ].forEach(([id, fallbackLabel]) => {
          const button = document.getElementById(id);
          if (!button) {
            return;
          }
          const defaultLabel = button.dataset.defaultLabel || button.textContent || fallbackLabel;
          button.dataset.defaultLabel = defaultLabel;
          button.disabled = Boolean(active);
          button.textContent = active ? 'Retrieving sources...' : defaultLabel;
          button.setAttribute('aria-busy', active ? 'true' : 'false');
        });
      }

      function scrollAnswerSurfaceIntoView() {
        const targets = [
          document.getElementById('requestPhaseShell'),
          document.getElementById('output'),
          document.getElementById('answerStructured'),
          document.getElementById('studyPathPanel'),
        ];
        const target = targets.find((entry) => entry && !entry.hidden);
        if (target && typeof target.scrollIntoView === 'function') {
          target.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
      }

      function normalizeLookupValue(value) {
        return normalizeWhitespace(applySafeDisplayReplacements(value)).toLowerCase();
      }

      function resolveScholarProfileForAnswer(answer, diagnostics) {
        const requestedId = nonEmptyString(diagnostics && diagnostics.scholar_id);
        if (requestedId && scholarProfileMap[requestedId]) {
          return scholarProfileMap[requestedId];
        }
        const citations = Array.isArray(answer && answer.citations) ? answer.citations : [];
        const haystack = citations.map((citation) => [
          citation.author || '',
          citation.collection || '',
          citation.title || '',
          citation.reference || '',
        ].join(' ')).join(' ');
        const normalizedHaystack = normalizeLookupValue(haystack);
        for (const profile of Object.values(scholarProfileMap)) {
          const aliases = Array.isArray(profile.aliases) ? profile.aliases : [];
          if (aliases.some((alias) => alias && normalizedHaystack.includes(String(alias).toLowerCase()))) {
            return profile;
          }
        }
        const fallbackAuthor = nonEmptyString(citations.find((citation) => nonEmptyString(citation.author))?.author);
        if (!fallbackAuthor && answer.answer_mode !== 'scholar_perspective') {
          return null;
        }
        return {
          scholar_id: requestedId || '',
          name: fallbackAuthor || 'Requested scholar',
          name_transliterated: '',
          madhhab: nonEmptyString(answer.selected_madhhab) || 'not_specified',
          period: 'Period not loaded',
          methodology_notes: 'Drawn from recorded works and positions only.',
          known_works: [],
          retrieval_tags: [],
          aliases: [],
        };
      }

      function comparisonGroupLabel(citation) {
        const madhhab = normalizeLookupValue(citation && citation.madhhab ? citation.madhhab : '');
        if (madhhab === 'hanafi') {
          return 'Hanafi';
        }
        if (madhhab === 'shafii') {
          return "Shafi'i";
        }
        if (madhhab === 'maliki') {
          return 'Maliki';
        }
        if (madhhab === 'hanbali') {
          return 'Hanbali';
        }
        return 'Shared / Unspecified';
      }

      function buildComparePositionGroups(citations) {
        const groups = new Map();
        (Array.isArray(citations) ? citations : []).forEach((citation) => {
          const label = comparisonGroupLabel(citation);
          if (!groups.has(label)) {
            groups.set(label, []);
          }
          groups.get(label).push(citation);
        });
        return Array.from(groups.entries()).map(([label, items]) => ({label, items}));
      }

      function renderAnswerAlerts(answer, diagnostics, data, rootId) {
        const root = document.getElementById(rootId || 'answerAlertStack');
        const alerts = [];
        const degradedReason = nonEmptyString((data && data.degraded_reason) || (diagnostics && diagnostics.degraded_reason) || '');
        if (data && data.degraded) {
          alerts.push(`
            <section class="answer-inline-banner warning">
              <strong>Limited source retrieval — answer may be incomplete.</strong>
              ${degradedReason ? ` <span>${escapeHtml(degradedReason)}</span>` : ''}
            </section>
          `);
        }
        const continuity = diagnostics && diagnostics.continuity && typeof diagnostics.continuity === 'object'
          ? diagnostics.continuity
          : {};
        if (continuity.used && continuity.topic_anchor) {
          alerts.push(`
            <section class="continuity-banner">
              <strong>Continuing from:</strong> ${escapeHtml(cleanDisplayLabel(continuity.topic_anchor))}
            </section>
          `);
        }
        if (root) {
          root.innerHTML = alerts.join('');
        }
      }

      function renderScholarAttributionBlock(answer, diagnostics) {
        const root = document.getElementById('answerScholarBlock');
        if (!root) {
          return;
        }
        const profile = resolveScholarProfileForAnswer(answer, diagnostics);
        if (answer.answer_mode !== 'scholar_perspective' || !profile) {
          root.hidden = true;
          root.innerHTML = '';
          return;
        }
        const methodology = nonEmptyString(profile.methodology_notes)
          || 'Drawn from recorded works and positions only.';
        root.hidden = false;
        root.innerHTML = `
          <div class="answer-card-header">
            <div class="section-label">Scholar Attribution</div>
            <h3 class="answer-card-title">From ${escapeHtml(profile.name || 'recorded scholar material')}</h3>
          </div>
          <p class="answer-copy">Drawn from ${escapeHtml(profile.name || 'this scholar')}'s recorded works and positions. This is not a live ruling and it does not extrapolate beyond retrieved attribution.</p>
          <div class="scholar-meta-grid">
            <div class="scholar-meta-row">
              <span class="scholar-meta-label">Scholar</span>
              <strong>${escapeHtml(profile.name || 'Unknown scholar')}</strong>
            </div>
            <div class="scholar-meta-row">
              <span class="scholar-meta-label">Madhhab</span>
              <strong>${escapeHtml(profile.madhhab || answer.selected_madhhab || 'not_specified')}</strong>
            </div>
            <div class="scholar-meta-row">
              <span class="scholar-meta-label">Period</span>
              <strong>${escapeHtml(profile.period || 'Period not loaded')}</strong>
            </div>
            <div class="scholar-meta-row">
              <span class="scholar-meta-label">Method</span>
              <strong>${escapeHtml(methodology)}</strong>
            </div>
          </div>
        `;
      }

      function renderCompareViewSections(answer, diagnostics) {
        const root = document.getElementById('answerCompareGrid');
        const citations = Array.isArray(answer.citations) ? answer.citations : [];
        const compareActive = answer.answer_mode === 'compare_views' || Boolean(diagnostics && diagnostics.compare_views_triggered);
        if (!compareActive || !citations.length) {
          root.innerHTML = '';
          return;
        }
        const groups = buildComparePositionGroups(citations).filter((entry) => entry.items.length);
        if (!groups.length) {
          root.innerHTML = '';
          return;
        }
        root.innerHTML = `
          <section class="answer-card wide">
            <div class="answer-card-header">
              <div class="section-label">Compare Views</div>
              <h3 class="answer-card-title">Grounded comparison positions</h3>
            </div>
            <div class="compare-position-grid">
              ${groups.map((group) => {
                const sources = group.items.slice(0, 4);
                return `
                  <article class="compare-position-card">
                    <div class="answer-card-header">
                      <div class="section-label">${escapeHtml(group.label)}</div>
                      <h4 class="answer-card-title">${escapeHtml(group.label)} position</h4>
                    </div>
                    <div class="compare-position-list">
                      ${sources.map((citation) => `
                        <div class="compare-position-source">
                          <strong>${escapeHtml(cleanDisplayLabel(citation.title || citation.collection || 'Grounded source'))}</strong>
                          <div class="answer-source-meta">${buildSourceMeta(citation) || escapeHtml(cleanDisplayLabel(citation.reference || 'Grounded source metadata available.'))}</div>
                        </div>
                      `).join('')}
                    </div>
                    ${answer.uncertainty_note ? `<div class="answer-note" style="margin-top:0.72rem;"><strong>Uncertainty:</strong> ${escapeHtml(answer.uncertainty_note)}</div>` : ''}
                  </article>
                `;
              }).join('')}
            </div>
          </section>
        `;
      }

      function renderLearningCenter(data) {
        const panel = document.getElementById('learningCenterPanel');
        const meta = document.getElementById('learningCenterMeta');
        const body = document.getElementById('learningCenterBody');
        if (!panel || !meta || !body) {
          return;
        }
        const answer = data && data.answer ? data.answer : {};
        const diagnostics = data && data.diagnostics ? data.diagnostics : {};
        const citations = Array.isArray(answer.citations) ? answer.citations : [];
        const teachingCitations = citations.filter((citation) => isTeachingCitation(citation));
        const teachingLayerUsed = Boolean(diagnostics.teaching_layer_used);
        const teachingSourcesCount = Number(diagnostics.teaching_sources_count || teachingCitations.length || 0);
        const teachingLayerReason = nonEmptyString(diagnostics.teaching_layer_reason);
        meta.innerHTML = [
          `<span class="answer-chip"><strong>Teaching sources:</strong> ${escapeHtml(String(teachingSourcesCount))}</span>`,
          teachingLayerReason ? `<span class="answer-chip"><strong>Reason:</strong> ${escapeHtml(cleanDisplayLabel(teachingLayerReason))}</span>` : '',
        ].filter(Boolean).join('');
        if (teachingLayerUsed) {
          body.innerHTML = `
            <div class="learning-center-empty">
              Teaching material is explanatory support only. It does not function as a primary text, madhhab authority, or modern fatwa authority.
            </div>
            ${teachingCitations.length ? `<div class="answer-source-list">${teachingCitations.map((item) => buildSourceCard(item)).join('')}</div>` : '<p class="learning-center-empty">Teaching-layer support was used for this answer, but no separate teaching source cards were available.</p>'}
          `;
        } else {
          body.innerHTML = '<p class="learning-center-empty">No teaching-layer material was used for this answer.</p>';
        }
        panel.hidden = false;
      }

      function renderAnswerSurface(data) {
        const answer = data && data.answer ? data.answer : {};
        const renderedText = resolveRenderedAnswerText(data);
        syncDeskModeUi(answer.answer_mode || document.getElementById('mode').value);
        if (answer.answer_mode === 'study_path') {
          renderStudyPath(answer, data && data.diagnostics ? data.diagnostics : {}, data);
          renderLearningCenter(data);
          scrollAnswerSurfaceIntoView();
          return;
        }
        if (hasStructuredAnswer(answer, data)) {
          renderStructuredAnswer(data);
          renderLearningCenter(data);
          scrollAnswerSurfaceIntoView();
          return;
        }
        resetAnswerSurface(renderedText || JSON.stringify(data, null, 2));
        renderLearningCenter(data);
        scrollAnswerSurfaceIntoView();
      }

      function renderStructuredAnswer(data) {
        const answer = data && data.answer ? data.answer : {};
        const diagnostics = data && data.diagnostics ? data.diagnostics : {};
        const renderedText = resolveRenderedAnswerText(data);
        const structured = document.getElementById('answerStructured');
        const studyPathPanel = document.getElementById('studyPathPanel');
        renderRenderedAnswerText(renderedText);
        structured.hidden = false;
        studyPathPanel.hidden = true;
        renderAnswerAlerts(answer, diagnostics, data, 'answerAlertStack');
        renderScholarAttributionBlock(answer, diagnostics);
        renderAnswerSummary(answer, diagnostics, data);
        renderAnswerMainSections(answer);
        renderCompareViewSections(answer, diagnostics);
        renderAnswerLayerSections(answer);
        renderAnswerSupportSections(answer, diagnostics);
      }

      function renderAnswerSummary(answer, diagnostics, data) {
        const root = document.getElementById('answerSummary');
        const citations = Array.isArray(answer.citations) ? answer.citations : [];
        const chips = [
          `<span class="answer-chip"><strong>Mode:</strong> ${escapeHtml(modeDisplayLabel(answer.answer_mode || 'research'))}</span>`,
          `<span class="answer-chip"><strong>Madhhab:</strong> ${escapeHtml(answer.selected_madhhab || 'not_specified')}</span>`,
          `<span class="answer-chip"><strong>Sources:</strong> ${escapeHtml(String(diagnostics.source_count || citations.length || 0))}</span>`
        ];
        if (answer.evidence_strength) {
          chips.push(`<span class="answer-chip"><strong>Evidence:</strong> ${escapeHtml(answer.evidence_strength)}</span>`);
        }
        if (diagnostics.path_used) {
          chips.push(`<span class="answer-chip"><strong>Path:</strong> ${escapeHtml(diagnostics.path_used)}</span>`);
        }
        if (diagnostics.latency_ms) {
          chips.push(`<span class="answer-chip"><strong>Latency:</strong> ${escapeHtml(String(diagnostics.latency_ms))} ms</span>`);
        }
        if (data && data.degraded) {
          chips.push('<span class="answer-chip"><strong>Retrieval:</strong> degraded</span>');
        }
        const summaryCopy = nonEmptyString(answer.evidence_summary)
          || (answer.answer_mode === 'source_only'
            ? 'Direct excerpts and source metadata are organized below with minimal framing.'
            : 'Grounded sections and source layers are organized below for review.');
        root.innerHTML = `
          <div class="answer-summary-grid">
            <div>
              <div class="answer-kicker">Rendered Answer</div>
              <h3 class="answer-summary-title">Grounded research answer</h3>
            </div>
            <p class="answer-summary-copy">${escapeHtml(summaryCopy)}</p>
            <div class="answer-card-meta">${chips.join('')}</div>
          </div>
        `;
      }

      function renderAnswerMainSections(answer) {
        const root = document.getElementById('answerSectionGrid');
        const sections = [];
        const directAnswer = nonEmptyString(answer.direct_answer);
        if (directAnswer) {
          sections.push({
            title: 'Direct Answer',
            content: directAnswer,
            wide: true,
          });
        }
        const madhhabPosition = nonEmptyString(answer.madhhab_position);
        if (madhhabPosition && !/^no madhhab was selected\.?$/i.test(madhhabPosition)) {
          sections.push({
            title: 'Madhhab Position',
            content: madhhabPosition,
          });
        }
        const evidenceSummary = nonEmptyString(answer.evidence_summary);
        if (evidenceSummary && evidenceSummary !== directAnswer) {
          sections.push({
            title: 'Evidence Summary',
            content: evidenceSummary,
          });
        }
        const evidenceStrength = nonEmptyString(answer.evidence_strength);
        if (evidenceStrength) {
          sections.push({
            title: 'Evidence Strength',
            content: evidenceStrength,
          });
        }
        if (!sections.length) {
          root.innerHTML = '';
          return;
        }
        root.innerHTML = sections.map((section) => `
          <section class="answer-card${section.wide ? ' wide' : ''}">
            <div class="answer-card-header">
              <div class="section-label">${escapeHtml(section.title)}</div>
              <h3 class="answer-card-title">${escapeHtml(section.title)}</h3>
            </div>
            <p class="answer-copy">${escapeHtml(section.content)}</p>
          </section>
        `).join('');
      }

      function renderAnswerLayerSections(answer) {
        const root = document.getElementById('answerLayerGrid');
        const citations = Array.isArray(answer.citations) ? answer.citations : [];
        const groups = groupedCitationsForDisplay(citations);
        const definitions = [
          {
            key: 'primary_texts',
            title: 'Primary Texts',
            footnote: '',
          },
          {
            key: 'fiqh',
            title: 'Fiqh / Madhhab Authority',
            footnote: '',
          },
          {
            key: 'tasawwuf',
            title: 'Spiritual Guidance / Tasawwuf',
            footnote: 'This reflects classical spiritual teachings, not legal rulings.',
          },
          {
            key: 'scholar_commentary',
            title: 'Scholar Commentary',
            footnote: 'Scholar commentary supports explanation and teaching context only.',
          },
          {
            key: 'commentary',
            title: 'Supporting Commentary',
            footnote: '',
          },
          {
            key: 'modern_application',
            title: 'Modern Fatwa / Application',
            footnote: 'Modern application materials should be read in light of the primary texts and school authorities.',
          }
        ];
        const cards = [];
        const nonTeachingCitations = citations.filter((item) => !isTeachingCitation(item));
        definitions.forEach((definition) => {
          const items = groups[definition.key] || [];
          if (!items.length) {
            return;
          }
          cards.push(`
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">${escapeHtml(definition.title)}</div>
                <h3 class="answer-source-layer-title">${escapeHtml(definition.title)}</h3>
              </div>
              <div class="answer-source-list">
                ${items.map((item) => buildSourceCard(item)).join('')}
              </div>
              ${definition.footnote ? `<div class="answer-footnote">${escapeHtml(definition.footnote)}</div>` : ''}
            </section>
          `);
        });
        if (nonTeachingCitations.length) {
          cards.push(`
            <section class="answer-card wide">
              <div class="answer-card-header">
                <div class="section-label">Sources</div>
                <h3 class="answer-card-title">Sources</h3>
              </div>
              <div class="answer-source-list">
                ${nonTeachingCitations.map((item) => buildSourceCard(item)).join('')}
              </div>
            </section>
          `);
        }
        root.innerHTML = cards.join('');
      }

      function renderAnswerSupportSections(answer, diagnostics) {
        const root = document.getElementById('answerSupportGrid');
        const citations = Array.isArray(answer.citations) ? answer.citations : [];
        const supportCards = [];
        const derivedLayers = derivedMissingLayers(citations);
        if (derivedLayers.presentCount > 0 && derivedLayers.missing.length) {
          supportCards.push(`
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">Missing Layers</div>
                <h3 class="answer-card-title">Missing Layers</h3>
              </div>
              <div class="answer-bullet-list">
                ${derivedLayers.missing.map((layer) => `
                  <div class="answer-bullet-card"><strong>${escapeHtml(layer.label)}:</strong> No grounded source found for this layer.</div>
                `).join('')}
              </div>
            </section>
          `);
        }

        const gapItems = uniqueNonEmpty([
          answer.uncertainty_note || '',
          ...(Array.isArray(answer.corpus_gaps) ? answer.corpus_gaps : []),
        ]);
        if (gapItems.length) {
          supportCards.push(`
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">Uncertainty / Corpus Gaps</div>
                <h3 class="answer-card-title">Uncertainty / Corpus Gaps</h3>
              </div>
              <div class="answer-bullet-list">
                ${gapItems.map((item) => `<div class="answer-bullet-card">${escapeHtml(item)}</div>`).join('')}
              </div>
            </section>
          `);
        }

        if (answer.reading_list && typeof answer.reading_list === 'object') {
          const groups = [
            {key: 'beginner', label: 'Beginner'},
            {key: 'intermediate', label: 'Intermediate'},
            {key: 'advanced', label: 'Advanced / Deeper Study'}
          ];
          supportCards.push(`
            <section class="answer-card wide">
              <div class="answer-card-header">
                <div class="section-label">Reading List</div>
                <h3 class="answer-card-title">Reading List</h3>
              </div>
              <div class="answer-reading-list">
                ${groups.map((group) => {
                  const items = Array.isArray(answer.reading_list[group.key]) ? answer.reading_list[group.key] : [];
                  return `
                    <section class="answer-reading-group">
                      <div class="answer-reading-heading">${escapeHtml(group.label)}</div>
                      ${items.length ? items.map((item) => {
                        const meta = uniqueNonEmpty([item.layer, item.collection, item.author]).map(escapeHtml).join(' • ');
                        return `
                          <article class="answer-reading-card">
                            <div class="answer-reading-title">${escapeHtml(cleanDisplayLabel(item.title || 'Untitled source'))}</div>
                            <div class="answer-reading-meta">${meta || 'Grounded reading anchor.'}</div>
                          </article>
                        `;
                      }).join('') : '<div class="answer-empty">No grounded source found for this layer.</div>'}
                    </section>
                  `;
                }).join('')}
              </div>
            </section>
          `);
        }

        if (Array.isArray(answer.lesson_path) && answer.lesson_path.length) {
          supportCards.push(`
            <section class="answer-card wide">
              <div class="answer-card-header">
                <div class="section-label">Lesson Path</div>
                <h3 class="answer-card-title">Lesson Path</h3>
              </div>
              <div class="answer-lesson-list">
                ${answer.lesson_path.map((lesson) => {
                  const anchors = Array.isArray(lesson.source_anchors) ? lesson.source_anchors.filter(Boolean) : [];
                  return `
                    <article class="answer-lesson-card">
                      <div class="answer-lesson-title">${escapeHtml(cleanDisplayLabel(lesson.lesson_title || 'Lesson'))}</div>
                      <div class="answer-lesson-meta"><strong>Objective:</strong> ${escapeHtml(lesson.objective || '')}</div>
                      <div class="answer-lesson-copy">${escapeHtml(lesson.short_explanation || '')}</div>
                      ${anchors.length ? `<div class="answer-lesson-meta"><strong>Source Anchors:</strong> ${anchors.map((anchor) => escapeHtml(cleanDisplayLabel(anchor))).join(' • ')}</div>` : ''}
                      ${lesson.practice_prompt ? `<div class="answer-lesson-meta"><strong>Practice / Reflection:</strong> ${escapeHtml(lesson.practice_prompt)}</div>` : ''}
                    </article>
                  `;
                }).join('')}
              </div>
            </section>
          `);
        }

        const keyTerms = Array.isArray(answer.key_terms) ? answer.key_terms.filter(Boolean) : [];
        if (keyTerms.length) {
          supportCards.push(`
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">Key Terms</div>
                <h3 class="answer-card-title">Key Terms</h3>
              </div>
              <div class="answer-bullet-list">
                ${keyTerms.map((item) => `<div class="answer-bullet-card">${escapeHtml(cleanDisplayLabel(item))}</div>`).join('')}
              </div>
            </section>
          `);
        }

        const avoidItems = Array.isArray(answer.what_to_avoid) ? answer.what_to_avoid.filter(Boolean) : [];
        if (avoidItems.length) {
          supportCards.push(`
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">What To Avoid</div>
                <h3 class="answer-card-title">What To Avoid</h3>
              </div>
              <div class="answer-bullet-list">
                ${avoidItems.map((item) => `<div class="answer-bullet-card">${escapeHtml(item)}</div>`).join('')}
              </div>
            </section>
          `);
        }

        root.innerHTML = supportCards.join('');
      }

      function renderStudyPath(answer, diagnostics, data) {
        const structured = document.getElementById('answerStructured');
        const studyPathPanel = document.getElementById('studyPathPanel');
        const renderedText = resolveRenderedAnswerText(data);
        const gapItems = [];
        if (answer.uncertainty_note) {
          gapItems.push(answer.uncertainty_note);
        }
        if (Array.isArray(answer.corpus_gaps)) {
          answer.corpus_gaps.forEach((item) => gapItems.push(item));
        }
        renderRenderedAnswerText(renderedText);
        structured.hidden = true;
        studyPathPanel.hidden = false;
        renderAnswerAlerts(answer, diagnostics, data, 'studyAlertStack');
        renderStudySummary(answer);
        renderStudyTabs(answer);
        renderStudyReadingList(answer.reading_list || {});
        renderStudyLessonPath(answer.lesson_path || []);
        renderStudyMissingLayers(answer);
        renderStudyBulletSection('studyKeyTerms', answer.key_terms || [], 'No grounded key terms were assembled.');
        renderStudyBulletSection('studyAvoid', answer.what_to_avoid || [], 'No cautionary notes were needed for this study path.');
        renderStudyBulletSection('studyGaps', gapItems, 'No major corpus gap was surfaced for this topic.');
      }

      function renderStudySummary(answer) {
        const root = document.getElementById('studyPathSummary');
        const readiness = answer.madhhab_readiness || {};
        const selectedMadhhab = answer.selected_madhhab || 'not_specified';
        const readinessKey = selectedMadhhab === 'shafii'
          ? 'shafi_ready'
          : selectedMadhhab + '_ready';
        const selectedReady = selectedMadhhab === 'not_specified'
          || selectedMadhhab === 'compare_all'
          || readiness[readinessKey] !== false;
        root.innerHTML = `
          <div class="study-summary-grid">
            <div>
              <div class="section-label">Study Path</div>
              <h3 class="study-summary-title">${escapeHtml(answer.topic || 'Grounded Study Path')}</h3>
            </div>
            <p class="study-summary-copy">${escapeHtml(answer.overview || answer.direct_answer || 'No overview is available yet.')}</p>
            <div class="study-summary-meta">
              <span class="study-chip"><strong>Madhhab:</strong> ${escapeHtml(selectedMadhhab)}</span>
              <span class="study-chip"><strong>Readiness:</strong> ${selectedReady ? 'supported where evidence exists' : 'limited in current corpus'}</span>
              <span class="study-chip"><strong>Mode:</strong> study_path</span>
            </div>
          </div>
        `;
      }

      function studyLayerDefinitions() {
        return [
          {
            key: 'quran',
            label: "Qur'an",
            descriptor: 'Primary texts',
            summary: 'Direct revelation anchors the topic and sets the first study boundary.',
            footnote: ''
          },
          {
            key: 'hadith',
            label: 'Hadith',
            descriptor: 'Primary texts',
            summary: 'Prophetic reports clarify practice, wording, and lived application.',
            footnote: ''
          },
          {
            key: 'fiqh',
            label: 'Fiqh',
            descriptor: 'Legal study',
            summary: 'Madhhab-based fiqh organizes obligations, order, and legal structure where corpus support exists.',
            footnote: ''
          },
          {
            key: 'tasawwuf',
            label: 'Tasawwuf',
            descriptor: 'Spiritual guidance',
            summary: 'Inner discipline and sincerity are kept clearly separate from legal rulings.',
            footnote: 'This reflects classical spiritual teachings, not legal rulings.'
          },
          {
            key: 'scholar_commentary',
            label: 'Scholar Commentary',
            descriptor: 'Teaching context',
            summary: 'Modern commentary may explain, sequence, and contextualize, but it does not override primary texts or fiqh manuals.',
            footnote: 'Scholar Commentary supports explanation and teaching context only.'
          }
        ];
      }

      function studyReadingGroups(readingList) {
        const payload = readingList && typeof readingList === 'object' ? readingList : {};
        return [
          {
            key: 'beginner',
            label: 'Beginner',
            copy: 'Start with the clearest admitted anchors for first study.',
            items: Array.isArray(payload.beginner) ? payload.beginner : []
          },
          {
            key: 'intermediate',
            label: 'Intermediate',
            copy: 'Add structure, commentary, and stronger detail once the basics are stable.',
            items: Array.isArray(payload.intermediate) ? payload.intermediate : []
          },
          {
            key: 'deeper',
            label: 'Deeper Study',
            copy: 'Broaden nuance only where grounded sources are actually available.',
            items: Array.isArray(payload.deeper)
              ? payload.deeper
              : (Array.isArray(payload.advanced) ? payload.advanced : [])
          }
        ];
      }

      function renderStudyTabs(answer) {
        const layers = answer.source_layers || {};
        const root = document.getElementById('studyTabs');
        const panel = document.getElementById('studyTabPanel');
        const definitions = studyLayerDefinitions();
        root.innerHTML = '';
        const available = definitions.filter((entry) => Array.isArray(layers[entry.key]) && layers[entry.key].length);
        if (!available.length) {
          panel.innerHTML = `
            <div class="study-layer-header">
              <div>
                <div class="section-label">Source Layers</div>
                <h3 class="study-layer-title">No grounded layer available</h3>
              </div>
            </div>
            <p class="study-layer-note">No grounded source found for this layer.</p>
          `;
          return;
        }
        const firstActive = available[0];
        available.forEach((entry) => {
          const button = document.createElement('button');
          button.type = 'button';
          const isActive = entry.key === firstActive.key;
          button.className = 'study-tab' + (isActive ? ' active' : '');
          button.id = `study-tab-${entry.key}`;
          button.setAttribute('role', 'tab');
          button.setAttribute('aria-controls', 'studyTabPanel');
          button.setAttribute('aria-selected', isActive ? 'true' : 'false');
          const count = Array.isArray(layers[entry.key]) ? layers[entry.key].length : 0;
          button.innerHTML = `
            <span class="study-tab-label">${escapeHtml(entry.label)}</span>
            <span class="study-tab-meta">${escapeHtml(`${count} grounded source${count === 1 ? '' : 's'} • ${entry.descriptor}`)}</span>
          `;
          button.addEventListener('click', () => {
            root.querySelectorAll('.study-tab').forEach((tab) => {
              tab.classList.remove('active');
              tab.setAttribute('aria-selected', 'false');
            });
            button.classList.add('active');
            button.setAttribute('aria-selected', 'true');
            renderStudyLayerPanel(entry, layers[entry.key] || []);
          });
          root.appendChild(button);
        });
        panel.setAttribute('aria-labelledby', `study-tab-${firstActive.key}`);
        renderStudyLayerPanel(firstActive, layers[firstActive.key] || []);
      }

      function renderStudyMissingLayers(answer) {
        const root = document.getElementById('studyMissingLayers');
        const layers = answer.source_layers || {};
        const definitions = studyLayerDefinitions();
        const missing = definitions.filter((entry) => !Array.isArray(layers[entry.key]) || !layers[entry.key].length);
        if (!missing.length) {
          root.innerHTML = '<div class="study-empty">All major source layers were grounded for this path.</div>';
          return;
        }
        let html = '<div class="study-missing-list">';
        missing.forEach((entry) => {
          html += `
            <article class="study-missing-card">
              <div class="study-missing-title">${escapeHtml(entry.label)}</div>
              <div class="study-missing-copy">No grounded source found for this layer.</div>
            </article>
          `;
        });
        html += '</div>';
        root.innerHTML = html;
      }

      function renderStudyLayerPanel(definition, items) {
        const panel = document.getElementById('studyTabPanel');
        panel.setAttribute('aria-labelledby', `study-tab-${definition.key}`);
        if (!Array.isArray(items) || !items.length) {
          panel.innerHTML = `
            <div class="study-layer-header">
              <div>
                <div class="section-label">Source Layer</div>
                <h3 class="study-layer-title">${escapeHtml(definition.label)}</h3>
              </div>
            </div>
            <p class="study-layer-note">No grounded source found for this layer.</p>
          `;
          return;
        }
        let html = `
          <div class="study-layer-header">
            <div>
              <div class="section-label">${escapeHtml(definition.descriptor)}</div>
              <h3 class="study-layer-title">${escapeHtml(definition.label)}</h3>
            </div>
            <div class="study-layer-count">${escapeHtml(`${items.length} grounded anchor${items.length === 1 ? '' : 's'}`)}</div>
          </div>
          <p class="study-layer-summary">${escapeHtml(definition.summary)}</p>
          <div class="study-source-list">
        `;
        items.forEach((item) => {
          const meta = uniqueNonEmpty([
            cleanDisplayLabel(item.reference || ''),
            cleanDisplayLabel(item.collection || ''),
            cleanDisplayLabel(item.author || ''),
            cleanDisplayLabel(item.madhhab || '')
          ]).map(escapeHtml).join(' • ');
          const title = cleanDisplayLabel(item.title || 'Retrieved Source');
          const quote = nonEmptyString(cleanDisplayExcerpt(item.quote || item.snippet || item.excerpt || ''));
          html += `
            <article class="study-source-card">
              <div class="study-source-title">${escapeHtml(title)}</div>
              <div class="study-source-meta">${meta || 'Grounded source metadata available.'}</div>
              ${quote ? `<div class="study-source-quote">${escapeHtml(quote)}</div>` : ''}
            </article>
          `;
        });
        html += '</div>';
        if (definition.footnote) {
          html += `<div class="study-layer-footnote">${escapeHtml(definition.footnote)}</div>`;
        }
        panel.innerHTML = html;
      }

      function renderStudyReadingList(readingList) {
        const root = document.getElementById('studyReadingList');
        const groups = studyReadingGroups(readingList);
        let html = '<div class="study-reading-list">';
        groups.forEach((group) => {
          const items = Array.isArray(group.items) ? group.items : [];
          html += `
            <section class="study-reading-group">
              <div class="study-reading-heading">${escapeHtml(group.label)}</div>
              <p class="study-reading-copy">${escapeHtml(group.copy)}</p>
          `;
          if (!items.length) {
            html += '<div class="study-empty">No grounded reading anchor is available for this stage yet.</div>';
          } else {
            items.forEach((item) => {
              const meta = uniqueNonEmpty([
                item.layer,
                cleanDisplayLabel(item.collection || ''),
                cleanDisplayLabel(item.author || '')
              ]).map(escapeHtml).join(' • ');
              html += `
                <article class="study-reading-card">
                  <div class="study-reading-title">${escapeHtml(cleanDisplayLabel(item.title || 'Untitled source'))}</div>
                  <div class="study-reading-meta">${meta || 'Grounded reading anchor.'}</div>
                </article>
              `;
            });
          }
          html += '</section>';
        });
        html += '</div>';
        root.innerHTML = html;
      }

      function renderStudyLessonPath(lessonPath) {
        const root = document.getElementById('studyLessonPath');
        if (!Array.isArray(lessonPath) || !lessonPath.length) {
          root.innerHTML = '<div class="study-empty">No grounded lesson path could be assembled.</div>';
          return;
        }
        let html = '<div class="study-lesson-list">';
        lessonPath.forEach((lesson, index) => {
          const anchors = Array.isArray(lesson.source_anchors)
            ? lesson.source_anchors.filter(Boolean).map((anchor) => cleanDisplayLabel(anchor))
            : [];
          html += `
            <article class="study-lesson-card">
              <div class="study-lesson-header">
                <div class="study-lesson-step">Lesson ${index + 1}</div>
                <div class="study-lesson-title">${escapeHtml(cleanDisplayLabel(lesson.lesson_title || `Lesson ${index + 1}`))}</div>
              </div>
              <div class="study-lesson-meta"><strong>Objective:</strong> ${escapeHtml(lesson.objective || '')}</div>
              <div class="study-lesson-copy">${escapeHtml(lesson.short_explanation || '')}</div>
              ${anchors.length ? `
                <div class="study-lesson-meta"><strong>Source Anchors:</strong></div>
                <div class="study-anchor-list">${anchors.map((anchor) => `<span class="study-anchor-chip">${escapeHtml(anchor)}</span>`).join('')}</div>
              ` : ''}
              ${lesson.practice_prompt ? `<div class="study-lesson-meta"><strong>Practice / Reflection:</strong> ${escapeHtml(lesson.practice_prompt)}</div>` : ''}
            </article>
          `;
        });
        html += '</div>';
        root.innerHTML = html;
      }

      function renderStudyBulletSection(elementId, items, emptyText) {
        const root = document.getElementById(elementId);
        const normalized = Array.isArray(items)
          ? items.map((item) => elementId === 'studyKeyTerms' ? cleanDisplayLabel(item) : nonEmptyString(item)).filter(Boolean)
          : [];
        if (!normalized.length) {
          root.innerHTML = `<div class="study-empty">${escapeHtml(emptyText)}</div>`;
          return;
        }
        let html = '<div class="study-bullet-list">';
        normalized.forEach((item) => {
          html += `<div class="study-bullet-card">${escapeHtml(item)}</div>`;
        });
        html += '</div>';
        root.innerHTML = html;
      }

      function workspaceStorageKey(suffix) {
        const userKey = workspaceState.user && workspaceState.user.id ? String(workspaceState.user.id) : 'anonymous';
        return `hj-workspace-${userKey}-${suffix}`;
      }

      function activeProjects() {
        return workspaceState.workspace && Array.isArray(workspaceState.workspace.projects)
          ? workspaceState.workspace.projects
          : [];
      }

      function activeChats() {
        return workspaceState.workspace && Array.isArray(workspaceState.workspace.recent_chats)
          ? workspaceState.workspace.recent_chats
          : [];
      }

      function currentProject() {
        return activeProjects().find((project) => Number(project.id) === Number(workspaceState.activeProjectId)) || null;
      }

      function currentProjectStudyState() {
        const project = currentProject();
        return project && project.study_state && typeof project.study_state === 'object'
          ? project.study_state
          : null;
      }

      function persistWorkspaceSelection() {
        if (!workspaceState.authenticated || !workspaceState.user) {
          return;
        }
        if (workspaceState.activeChatId) {
          window.localStorage.setItem(workspaceStorageKey('active-chat'), String(workspaceState.activeChatId));
        } else {
          window.localStorage.removeItem(workspaceStorageKey('active-chat'));
        }
        if (workspaceState.activeProjectId) {
          window.localStorage.setItem(workspaceStorageKey('active-project'), String(workspaceState.activeProjectId));
        } else {
          window.localStorage.removeItem(workspaceStorageKey('active-project'));
        }
      }

      function hydrateWorkspaceSelection() {
        if (!workspaceState.authenticated || !workspaceState.user) {
          workspaceState.activeChatId = null;
          workspaceState.activeProjectId = null;
          return;
        }
        const savedChat = Number(window.localStorage.getItem(workspaceStorageKey('active-chat')) || 0);
        const savedProject = Number(window.localStorage.getItem(workspaceStorageKey('active-project')) || 0);
        const chats = activeChats();
        const projects = activeProjects();
        workspaceState.activeChatId = chats.some((chat) => Number(chat.id) === savedChat) ? savedChat : null;
        workspaceState.activeProjectId = projects.some((project) => Number(project.id) === savedProject) ? savedProject : null;
      }

      function filteredChats() {
        const chats = activeChats();
        if (!workspaceState.activeProjectId) {
          return chats;
        }
        return chats.filter((chat) => Number(chat.project_id || 0) === Number(workspaceState.activeProjectId));
      }

      function getRequestSessionId() {
        if (workspaceState.authenticated && workspaceState.activeChatId) {
          return `chat-${workspaceState.activeChatId}`;
        }
        return getSessionId();
      }

      function applyWorkspaceDefaults() {
        const user = workspaceState.user || {};
        const project = currentProject();
        const mode = project && project.study_mode === 'study_path'
          ? 'study_path'
          : (user.default_answer_mode || 'research');
        const madhhab = project && project.madhhab && project.madhhab !== 'not_specified'
          ? project.madhhab
          : (user.default_madhhab || 'not_specified');
        if (!workspaceState.userAdjustedMode) {
          setModeSelection(mode);
        }
        if (!workspaceState.userAdjustedMadhhab) {
          setMadhhabSelection(madhhab || 'not_specified');
        }
      }

      function toggleProjectComposer(force) {
        const composer = document.getElementById('projectComposer');
        composer.hidden = typeof force === 'boolean' ? !force : !composer.hidden;
      }

      function renderWorkspaceContext() {
        const root = document.getElementById('workspaceContext');
        if (!workspaceState.authenticated || !workspaceState.user) {
          root.textContent = 'Working anonymously. Sign in to keep chat history, study projects, and preferences across refreshes.';
          return;
        }
        const user = workspaceState.user;
        const project = currentProject();
        const session = workspaceState.activeSession;
        const parts = [
          `${user.display_name || user.username} is signed in locally.`,
          `Default madhhab: ${user.default_madhhab || 'not_specified'}.`,
          `Default answer mode: ${user.default_answer_mode || 'research'}.`,
        ];
        if (project) {
          parts.push(`Active project: ${project.title}.`);
          parts.push(`Project mode: ${project.study_mode === 'study_path' ? 'study_path' : 'standard'}.`);
          if (project.description) {
            parts.push(project.description);
          }
          const progress = project.study_state && project.study_state.progress ? project.study_state.progress : null;
          if (progress && Number(progress.total_lessons || 0) > 0) {
            parts.push(`Study progress: ${progress.completed_count || 0} of ${progress.total_lessons || 0} lessons complete.`);
          }
        }
        if (session) {
          parts.push(`Current chat: ${session.title}.`);
        } else {
          parts.push('Create or select a chat to keep a persistent study trail.');
        }
        root.textContent = parts.join(' ');
      }

      function renderAuthPanel() {
        const root = document.getElementById('sidebarAuthState');
        if (workspaceState.authenticated && workspaceState.user) {
          const user = workspaceState.user;
          const project = currentProject();
          root.innerHTML = `
            <div class="section-label">Local Profile</div>
            <div class="sidebar-profile">
              <div>
                <h2 class="sidebar-profile-name">${escapeHtml(user.display_name || user.username)}</h2>
                <div class="sidebar-meta">@${escapeHtml(user.username || '')}</div>
              </div>
              <div class="sidebar-profile-badges">
                <span class="sidebar-chip"><strong>Madhhab:</strong> ${escapeHtml(user.default_madhhab || 'not_specified')}</span>
                <span class="sidebar-chip"><strong>Mode:</strong> ${escapeHtml(user.default_answer_mode || 'research')}</span>
                ${project ? `<span class="sidebar-chip"><strong>Project:</strong> ${escapeHtml(project.title)}</span>` : ''}
              </div>
              <p class="sidebar-copy">Preferences and project context influence framing only. They never override grounded source evidence.</p>
              <div class="sidebar-actions">
                <button class="sidebar-secondary-button" type="button" onclick="logoutUser()">Sign Out</button>
              </div>
            </div>
          `;
          return;
        }

        const loginActive = workspaceState.authView !== 'register';
        root.innerHTML = `
          <div class="section-label">Local Access</div>
          <h2 class="sidebar-title">${loginActive ? 'Sign in locally' : 'Create a local account'}</h2>
          <p class="sidebar-copy">
            Local sign-in enables persistent chats, study projects, and safe memory. Anonymous asking still works without persistence.
          </p>
          <form class="sidebar-field-grid" onsubmit="${loginActive ? 'submitLogin(event)' : 'submitRegister(event)'}">
            <div>
              <label for="authUsername">${loginActive ? 'Username' : 'Choose a username'}</label>
              <input id="authUsername" name="username" type="text" autocomplete="username" required />
            </div>
            ${loginActive ? '' : `
              <div>
                <label for="authDisplayName">Display name</label>
                <input id="authDisplayName" name="display_name" type="text" autocomplete="nickname" />
              </div>
            `}
            <div>
              <label for="authPassword">${loginActive ? 'Password' : 'Create a password'}</label>
              <input id="authPassword" name="password" type="password" autocomplete="${loginActive ? 'current-password' : 'new-password'}" required />
            </div>
            ${loginActive ? '' : `
              <div>
                <label for="authDefaultMadhhab">Default madhhab</label>
                <select id="authDefaultMadhhab" name="default_madhhab">
                  <option value="hanafi">hanafi</option>
                  <option value="shafii">shafii</option>
                  <option value="maliki">maliki</option>
                  <option value="hanbali">hanbali</option>
                  <option value="not_specified">not_specified</option>
                </select>
              </div>
              <div>
                <label for="authDefaultMode">Default answer mode</label>
                <select id="authDefaultMode" name="default_answer_mode">
                  <option value="research">research</option>
                  <option value="source_only">source_only</option>
                  <option value="compare_views">compare_views</option>
                  <option value="scholar_perspective">scholar_perspective</option>
                  <option value="study_path">study_path</option>
                </select>
              </div>
            `}
            <div class="sidebar-inline-actions">
              <button class="sidebar-button" type="submit">${loginActive ? 'Login' : 'Register'}</button>
              <button class="sidebar-link-button" type="button" onclick="setAuthView('${loginActive ? 'register' : 'login'}')">
                ${loginActive ? 'Need an account? Register' : 'Already have an account? Login'}
              </button>
            </div>
          </form>
        `;
      }

      function renderChatList() {
        const root = document.getElementById('chatList');
        const helper = document.getElementById('chatListHelper');
        if (!workspaceState.authenticated) {
          helper.textContent = 'Sign in to keep persistent chat history across refreshes.';
          root.innerHTML = '<div class="sidebar-empty">Anonymous asks stay local to the current page load.</div>';
          return;
        }
        const chats = filteredChats();
        helper.textContent = workspaceState.activeProjectId
          ? 'Showing chats linked to the selected study project.'
          : 'Your most recent grounded research sessions.';
        if (!chats.length) {
          root.innerHTML = '<div class="sidebar-empty">No chats match the current filter yet.</div>';
          return;
        }
        root.innerHTML = chats.map((chat) => `
          <button
            class="sidebar-item${Number(chat.id) === Number(workspaceState.activeChatId) ? ' active' : ''}"
            type="button"
            onclick="selectChat(${Number(chat.id)})"
          >
            <div class="sidebar-item-title">${escapeHtml(chat.title || 'Untitled chat')}</div>
            <div class="sidebar-item-meta">${escapeHtml(cleanDisplayLabel(chat.latest_message || 'No saved messages yet.'))}</div>
            <div class="sidebar-item-meta">Updated ${escapeHtml(formatTimestamp(chat.updated_at))}</div>
          </button>
        `).join('');
      }

      function renderProjectList() {
        const root = document.getElementById('projectList');
        const actions = document.getElementById('projectActions');
        if (!workspaceState.authenticated) {
          root.innerHTML = '<div class="sidebar-empty">Sign in to create or reuse study projects.</div>';
          actions.hidden = true;
          return;
        }
        const projects = activeProjects();
        actions.hidden = projects.length === 0;
        if (!projects.length) {
          root.innerHTML = '<div class="sidebar-empty">No study projects yet.</div>';
          return;
        }
        root.innerHTML = projects.map((project) => `
          <button
            class="sidebar-item${Number(project.id) === Number(workspaceState.activeProjectId) ? ' active' : ''}"
            type="button"
            onclick="selectProject(${Number(project.id)})"
          >
            <div class="sidebar-item-title">${escapeHtml(project.title || 'Untitled project')}</div>
            <div class="sidebar-item-meta">${escapeHtml(project.description || 'No goal recorded yet.')}</div>
            <div class="sidebar-item-meta">Madhhab: ${escapeHtml(project.madhhab || 'not_specified')} • Mode: ${escapeHtml(project.study_mode || 'standard')} • Status: ${escapeHtml(project.status || 'active')}</div>
            <div class="sidebar-item-meta">${escapeHtml(projectProgressLine(project))}</div>
          </button>
        `).join('');
      }

      function projectProgressLine(project) {
        const progress = project && project.study_state && project.study_state.progress
          ? project.study_state.progress
          : null;
        if (!progress || !Number(progress.total_lessons || 0)) {
          return project && project.study_mode === 'study_path'
            ? 'No saved lesson path yet.'
            : 'Standard project.';
        }
        return `${Number(progress.completed_count || 0)} of ${Number(progress.total_lessons || 0)} lessons complete`;
      }

      function flattenProjectReadingItems(readingList) {
        if (!readingList || typeof readingList !== 'object') {
          return [];
        }
        const orderedGroups = ['beginner', 'intermediate', 'deeper', 'advanced'];
        const seen = new Set();
        const items = [];
        orderedGroups.forEach((groupKey) => {
          const groupItems = Array.isArray(readingList[groupKey]) ? readingList[groupKey] : [];
          groupItems.forEach((item) => {
            const title = nonEmptyString(item && item.title ? item.title : '');
            const dedupeKey = `${groupKey}:${title.toLowerCase()}`;
            if (seen.has(dedupeKey)) {
              return;
            }
            seen.add(dedupeKey);
            items.push({...item, group_key: groupKey});
          });
        });
        return items.slice(0, 8);
      }

      function renderProjectDetailPanel() {
        const root = document.getElementById('projectDetailPanel');
        if (!workspaceState.authenticated) {
          root.innerHTML = '<div class="sidebar-empty">Sign in to keep project-level lessons and reading lists.</div>';
          return;
        }
        const project = currentProject();
        if (!project) {
          root.innerHTML = '<div class="sidebar-empty">Select a study project to see saved lessons, progress, and grounded reading anchors.</div>';
          return;
        }
        const state = currentProjectStudyState();
        const progress = state && state.progress ? state.progress : null;
        const lessons = state && Array.isArray(state.lesson_path) ? state.lesson_path : [];
        const readingItems = flattenProjectReadingItems(state && state.reading_list ? state.reading_list : {});
        const percent = progress ? Number(progress.progress_percent || 0) : 0;
        const completedCount = progress ? Number(progress.completed_count || 0) : 0;
        const totalLessons = progress ? Number(progress.total_lessons || lessons.length || 0) : lessons.length;
        const nextIndex = progress && Number.isInteger(progress.next_lesson_index) ? progress.next_lesson_index : null;
        const activeIndex = progress && Number.isInteger(progress.active_index) ? progress.active_index : null;
        const activeLesson = state && state.active_lesson && typeof state.active_lesson === 'object'
          ? state.active_lesson
          : null;

        if (project.study_mode !== 'study_path') {
          root.innerHTML = `
            <section class="project-detail-card">
              <div class="section-label">Active Project</div>
              <h3 class="project-detail-title">${escapeHtml(project.title || 'Untitled project')}</h3>
              <p class="sidebar-copy">${escapeHtml(project.description || 'This project is using the standard chat workspace.')}</p>
              <div class="project-chip-row">
                <span class="project-chip"><strong>Mode:</strong> standard</span>
                <span class="project-chip"><strong>Madhhab:</strong> ${escapeHtml(project.madhhab || 'not_specified')}</span>
                <span class="project-chip"><strong>Chats:</strong> ${escapeHtml(String(project.chat_count || 0))}</span>
              </div>
            </section>
          `;
          return;
        }

        if (!state) {
          root.innerHTML = `
            <section class="project-detail-card">
              <div class="section-label">Study Path Project</div>
              <h3 class="project-detail-title">${escapeHtml(project.title || 'Untitled project')}</h3>
              <p class="sidebar-copy">${escapeHtml(project.description || 'Ask inside this project to generate a grounded study path.')}</p>
              <div class="project-chip-row">
                <span class="project-chip"><strong>Mode:</strong> study_path</span>
                <span class="project-chip"><strong>Madhhab:</strong> ${escapeHtml(project.madhhab || 'not_specified')}</span>
              </div>
              <div class="sidebar-empty">No saved study path yet. Ask a grounded learning question in this project to create lessons and a reading list.</div>
            </section>
          `;
          return;
        }

        root.innerHTML = `
          <section class="project-detail-card">
            <div class="section-label">Active Study Project</div>
            <h3 class="project-detail-title">${escapeHtml(project.title || state.topic || 'Study project')}</h3>
            <div class="project-subtitle">${escapeHtml(state.topic || project.description || 'Grounded study path')}</div>
            <p class="sidebar-copy">${escapeHtml(state.overview || project.description || 'Grounded study output is saved at the project level for continuity.')}</p>
            <div class="project-chip-row">
              <span class="project-chip"><strong>Mode:</strong> study_path</span>
              <span class="project-chip"><strong>Madhhab:</strong> ${escapeHtml(project.madhhab || state.selected_madhhab || 'not_specified')}</span>
              <span class="project-chip"><strong>Chats:</strong> ${escapeHtml(String(project.chat_count || 0))}</span>
            </div>
            <div class="project-progress-bar" aria-label="Study progress">
              <div class="project-progress-fill" style="width:${Math.max(0, Math.min(percent, 100))}%"></div>
            </div>
            <div class="project-subtitle">${escapeHtml(`${completedCount} of ${totalLessons} lessons complete${activeLesson && activeLesson.lesson_title ? ` • Current: ${activeLesson.lesson_title}` : ''}`)}</div>
            <div class="project-inline-actions">
              ${Number.isInteger(nextIndex) ? `<button class="sidebar-secondary-button" type="button" onclick="continueProjectLesson()">Continue Next Lesson</button>` : ''}
              <button class="sidebar-secondary-button" type="button" onclick="createNewChat()">New Chat In Project</button>
            </div>
          </section>
          <section class="project-detail-card">
            <div class="section-label">Lessons</div>
            <div class="project-lesson-list">
              ${lessons.length ? lessons.map((lesson, index) => `
                <article class="project-lesson-card${activeIndex === index ? ' is-active' : ''}">
                  <div class="project-lesson-title">${escapeHtml(lesson.lesson_title || `Lesson ${index + 1}`)}</div>
                  <div class="project-lesson-meta">${escapeHtml(lesson.objective || '')}</div>
                  ${lesson.short_explanation ? `<div class="project-lesson-meta">${escapeHtml(lesson.short_explanation)}</div>` : ''}
                  <div class="project-inline-actions">
                    <button class="sidebar-link-button" type="button" onclick="revisitProjectLesson(${index})">Revisit Lesson</button>
                    <button class="sidebar-link-button" type="button" onclick="markProjectLessonComplete(${index})">${Array.isArray(progress && progress.completed_indices) && progress.completed_indices.includes(index) ? 'Completed' : 'Mark Complete'}</button>
                  </div>
                </article>
              `).join('') : '<div class="sidebar-empty">No grounded lesson path could be assembled yet.</div>'}
            </div>
          </section>
          <section class="project-detail-card">
            <div class="section-label">Saved Reading List</div>
            <div class="project-reading-list">
              ${readingItems.length ? readingItems.map((item) => `
                <article class="project-reading-card">
                  <div class="project-reading-title">${escapeHtml(cleanDisplayLabel(item.title || 'Untitled source'))}</div>
                  <div class="project-reading-meta">${escapeHtml([item.layer, item.collection, item.author].filter(Boolean).join(' • ') || 'Grounded reading anchor.')}</div>
                </article>
              `).join('') : '<div class="sidebar-empty">No grounded reading list is saved for this project yet.</div>'}
            </div>
          </section>
        `;
      }

      function renderSettingsPanel() {
        const root = document.getElementById('settingsPanel');
        if (!workspaceState.authenticated || !workspaceState.user) {
          root.innerHTML = '<div class="sidebar-empty">Sign in to save default madhhab, answer mode, and profile settings.</div>';
          return;
        }
        const user = workspaceState.user;
        root.innerHTML = `
          <div>
            <label for="settingsDisplayName">Display name</label>
            <input id="settingsDisplayName" type="text" value="${escapeHtml(user.display_name || user.username || '')}" />
          </div>
          <div>
            <label for="settingsMadhhab">Default madhhab</label>
            <select id="settingsMadhhab">
              <option value="hanafi"${user.default_madhhab === 'hanafi' ? ' selected' : ''}>hanafi</option>
              <option value="shafii"${user.default_madhhab === 'shafii' ? ' selected' : ''}>shafii</option>
              <option value="maliki"${user.default_madhhab === 'maliki' ? ' selected' : ''}>maliki</option>
              <option value="hanbali"${user.default_madhhab === 'hanbali' ? ' selected' : ''}>hanbali</option>
              <option value="not_specified"${user.default_madhhab === 'not_specified' ? ' selected' : ''}>not_specified</option>
            </select>
          </div>
          <div>
            <label for="settingsAnswerMode">Default answer mode</label>
            <select id="settingsAnswerMode">
              <option value="research"${user.default_answer_mode === 'research' ? ' selected' : ''}>research</option>
              <option value="source_only"${user.default_answer_mode === 'source_only' ? ' selected' : ''}>source_only</option>
              <option value="compare_views"${user.default_answer_mode === 'compare_views' ? ' selected' : ''}>compare_views</option>
              <option value="scholar_perspective"${user.default_answer_mode === 'scholar_perspective' ? ' selected' : ''}>scholar_perspective</option>
              <option value="study_path"${user.default_answer_mode === 'study_path' ? ' selected' : ''}>study_path</option>
            </select>
          </div>
          <div class="sidebar-inline-actions">
            <button class="sidebar-button" type="button" onclick="saveSettings()">Save Settings</button>
          </div>
        `;
      }

      function renderSidebarChrome() {
        renderAuthPanel();
        renderChatList();
        renderProjectList();
        renderProjectDetailPanel();
        renderSettingsPanel();
        renderWorkspaceContext();
      }

      function buildHistorySectionCard(title, content, wide) {
        return `
          <section class="answer-card${wide ? ' wide' : ''}">
            <div class="answer-card-header">
              <div class="section-label">${escapeHtml(title)}</div>
              <h3 class="answer-card-title">${escapeHtml(title)}</h3>
            </div>
            <p class="answer-copy">${escapeHtml(content)}</p>
          </section>
        `;
      }

      function buildHistoryLayerCard(title, items, footnote) {
        if (!Array.isArray(items) || !items.length) {
          return '';
        }
        return `
          <section class="answer-card">
            <div class="answer-card-header">
              <div class="section-label">${escapeHtml(title)}</div>
              <h3 class="answer-source-layer-title">${escapeHtml(title)}</h3>
            </div>
            <div class="answer-source-list">
              ${items.map((item) => buildSourceCard(item)).join('')}
            </div>
            ${footnote ? `<div class="answer-footnote">${escapeHtml(footnote)}</div>` : ''}
          </section>
        `;
      }

      function buildHistoryReadingListCard(readingList) {
        if (!readingList || typeof readingList !== 'object') {
          return '';
        }
        const groups = studyReadingGroups(readingList);
        return `
          <section class="answer-card wide">
            <div class="answer-card-header">
              <div class="section-label">Reading List</div>
              <h3 class="answer-card-title">Reading List</h3>
            </div>
            <div class="answer-reading-list">
              ${groups.map((group) => {
                const items = Array.isArray(group.items) ? group.items : [];
                return `
                  <section class="answer-reading-group">
                    <div class="answer-reading-heading">${escapeHtml(group.label)}</div>
                    ${items.length ? items.map((item) => {
                      const meta = uniqueNonEmpty([item.layer, item.collection, item.author]).map(escapeHtml).join(' • ');
                      return `
                        <article class="answer-reading-card">
                          <div class="answer-reading-title">${escapeHtml(cleanDisplayLabel(item.title || 'Untitled source'))}</div>
                          <div class="answer-reading-meta">${meta || 'Grounded reading anchor.'}</div>
                        </article>
                      `;
                    }).join('') : '<div class="answer-empty">No grounded source found for this layer.</div>'}
                  </section>
                `;
              }).join('')}
            </div>
          </section>
        `;
      }

      function buildHistoryLessonPathCard(lessonPath) {
        if (!Array.isArray(lessonPath) || !lessonPath.length) {
          return '';
        }
        return `
          <section class="answer-card wide">
            <div class="answer-card-header">
              <div class="section-label">Lesson Path</div>
              <h3 class="answer-card-title">Lesson Path</h3>
            </div>
            <div class="answer-lesson-list">
              ${lessonPath.map((lesson, index) => {
                const anchors = Array.isArray(lesson.source_anchors) ? lesson.source_anchors.filter(Boolean) : [];
                return `
                  <article class="answer-lesson-card">
                    <div class="answer-kicker">Lesson ${index + 1}</div>
                    <div class="answer-lesson-title">${escapeHtml(cleanDisplayLabel(lesson.lesson_title || 'Lesson'))}</div>
                    <div class="answer-lesson-meta"><strong>Objective:</strong> ${escapeHtml(lesson.objective || '')}</div>
                    <div class="answer-lesson-copy">${escapeHtml(lesson.short_explanation || '')}</div>
                    ${anchors.length ? `<div class="answer-lesson-meta"><strong>Source Anchors:</strong> ${anchors.map((anchor) => escapeHtml(cleanDisplayLabel(anchor))).join(' • ')}</div>` : ''}
                    ${lesson.practice_prompt ? `<div class="answer-lesson-meta"><strong>Practice / Reflection:</strong> ${escapeHtml(lesson.practice_prompt)}</div>` : ''}
                  </article>
                `;
              }).join('')}
            </div>
          </section>
        `;
      }

      function buildHistoryBulletCard(title, items) {
        if (!Array.isArray(items) || !items.length) {
          return '';
        }
        return `
          <section class="answer-card">
            <div class="answer-card-header">
              <div class="section-label">${escapeHtml(title)}</div>
              <h3 class="answer-card-title">${escapeHtml(title)}</h3>
            </div>
            <div class="answer-bullet-list">
              ${items.map((item) => `<div class="answer-bullet-card">${escapeHtml(item)}</div>`).join('')}
            </div>
          </section>
        `;
      }

      function buildHistoryScholarCard(answer, diagnostics) {
        const profile = resolveScholarProfileForAnswer(answer, diagnostics);
        if (answer.answer_mode !== 'scholar_perspective' || !profile) {
          return '';
        }
        return `
          <section class="answer-card answer-scholar-card wide">
            <div class="answer-card-header">
              <div class="section-label">Scholar Attribution</div>
              <h3 class="answer-card-title">From ${escapeHtml(profile.name || 'recorded scholar material')}</h3>
            </div>
            <p class="answer-copy">Drawn from recorded works and positions only. This remains attribution-based rather than a live ruling.</p>
            <div class="scholar-meta-grid">
              <div class="scholar-meta-row">
                <span class="scholar-meta-label">Scholar</span>
                <strong>${escapeHtml(profile.name || 'Unknown scholar')}</strong>
              </div>
              <div class="scholar-meta-row">
                <span class="scholar-meta-label">Madhhab</span>
                <strong>${escapeHtml(profile.madhhab || answer.selected_madhhab || 'not_specified')}</strong>
              </div>
              <div class="scholar-meta-row">
                <span class="scholar-meta-label">Period</span>
                <strong>${escapeHtml(profile.period || 'Period not loaded')}</strong>
              </div>
              <div class="scholar-meta-row">
                <span class="scholar-meta-label">Method</span>
                <strong>${escapeHtml(profile.methodology_notes || 'Drawn from recorded works and positions only.')}</strong>
              </div>
            </div>
          </section>
        `;
      }

      function buildHistoryCompareCard(answer, diagnostics, citations) {
        const compareActive = answer.answer_mode === 'compare_views' || Boolean(diagnostics && diagnostics.compare_views_triggered);
        if (!compareActive || !Array.isArray(citations) || !citations.length) {
          return '';
        }
        const groups = buildComparePositionGroups(citations).filter((entry) => entry.items.length);
        if (!groups.length) {
          return '';
        }
        return `
          <section class="answer-card wide">
            <div class="answer-card-header">
              <div class="section-label">Compare Views</div>
              <h3 class="answer-card-title">Grounded comparison positions</h3>
            </div>
            <div class="compare-position-grid">
              ${groups.map((group) => `
                <article class="compare-position-card">
                  <div class="answer-card-header">
                    <div class="section-label">${escapeHtml(group.label)}</div>
                    <h4 class="answer-card-title">${escapeHtml(group.label)} position</h4>
                  </div>
                  <div class="compare-position-list">
                    ${group.items.slice(0, 4).map((citation) => `
                      <div class="compare-position-source">
                        <strong>${escapeHtml(cleanDisplayLabel(citation.title || citation.collection || 'Grounded source'))}</strong>
                        <div class="answer-source-meta">${buildSourceMeta(citation) || escapeHtml(cleanDisplayLabel(citation.reference || 'Grounded source metadata available.'))}</div>
                      </div>
                    `).join('')}
                  </div>
                </article>
              `).join('')}
            </div>
          </section>
        `;
      }

      function buildStructuredHistoryMarkup(answer, diagnostics) {
        const citations = Array.isArray(answer.citations) ? answer.citations : [];
        const groups = groupedCitationsForDisplay(citations);
        const sections = [];
        const alertCards = [];
        if (diagnostics && diagnostics.degraded_reason) {
          alertCards.push(`
            <section class="answer-inline-banner warning">
              <strong>Limited source retrieval — answer may be incomplete.</strong>
              <span> ${escapeHtml(diagnostics.degraded_reason)}</span>
            </section>
          `);
        }
        if (diagnostics && diagnostics.continuity && diagnostics.continuity.used && diagnostics.continuity.topic_anchor) {
          alertCards.push(`
            <section class="continuity-banner">
              <strong>Continuing from:</strong> ${escapeHtml(cleanDisplayLabel(diagnostics.continuity.topic_anchor))}
            </section>
          `);
        }
        const directAnswer = nonEmptyString(answer.direct_answer);
        if (directAnswer) {
          sections.push(buildHistorySectionCard('Direct Answer', directAnswer, true));
        }
        const madhhabPosition = nonEmptyString(answer.madhhab_position);
        if (madhhabPosition && !/^no madhhab was selected\.?$/i.test(madhhabPosition)) {
          sections.push(buildHistorySectionCard('Madhhab Position', madhhabPosition, false));
        }
        const evidenceSummary = nonEmptyString(answer.evidence_summary);
        if (evidenceSummary && evidenceSummary !== directAnswer) {
          sections.push(buildHistorySectionCard('Evidence Summary', evidenceSummary, false));
        }
        const evidenceStrength = nonEmptyString(answer.evidence_strength);
        if (evidenceStrength) {
          sections.push(buildHistorySectionCard('Evidence Strength', evidenceStrength, false));
        }
        const scholarCard = buildHistoryScholarCard(answer, diagnostics);
        if (scholarCard) {
          sections.unshift(scholarCard);
        }
        const compareCard = buildHistoryCompareCard(answer, diagnostics, citations);
        if (compareCard) {
          sections.push(compareCard);
        }

        const layerCards = [
          buildHistoryLayerCard('Primary Texts', groups.primary_texts || [], ''),
          buildHistoryLayerCard('Fiqh / Madhhab Authority', groups.fiqh || [], ''),
          buildHistoryLayerCard('Spiritual Guidance / Tasawwuf', groups.tasawwuf || [], 'This reflects classical spiritual teachings, not legal rulings.'),
          buildHistoryLayerCard('Scholar Commentary', groups.scholar_commentary || [], 'Scholar commentary supports explanation and teaching context only.'),
          buildHistoryLayerCard('Supporting Commentary', groups.commentary || [], ''),
        ].filter(Boolean);
        if ((groups.modern_application || []).length) {
          layerCards.push(
            buildHistoryLayerCard(
              'Modern Fatwa / Application',
              groups.modern_application || [],
              'Modern application materials should be read in light of the primary texts and school authorities.'
            )
          );
        }
        if (citations.length) {
          layerCards.push(`
            <section class="answer-card wide">
              <div class="answer-card-header">
                <div class="section-label">Sources</div>
                <h3 class="answer-card-title">Sources</h3>
              </div>
              <div class="answer-source-list">${citations.map((item) => buildSourceCard(item)).join('')}</div>
            </section>
          `);
        }

        const supportCards = [];
        const derivedLayers = derivedMissingLayers(citations);
        if (derivedLayers.presentCount > 0 && derivedLayers.missing.length) {
          supportCards.push(`
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">Missing Layers</div>
                <h3 class="answer-card-title">Missing Layers</h3>
              </div>
              <div class="answer-bullet-list">
                ${derivedLayers.missing.map((layer) => `<div class="answer-bullet-card"><strong>${escapeHtml(layer.label)}:</strong> No grounded source found for this layer.</div>`).join('')}
              </div>
            </section>
          `);
        }
        const gapItems = uniqueNonEmpty([
          answer.uncertainty_note || '',
          ...(Array.isArray(answer.corpus_gaps) ? answer.corpus_gaps : []),
        ]);
        if (gapItems.length) {
          supportCards.push(buildHistoryBulletCard('Uncertainty / Corpus Gaps', gapItems));
        }
        if (answer.reading_list && typeof answer.reading_list === 'object') {
          supportCards.push(buildHistoryReadingListCard(answer.reading_list));
        }
        if (Array.isArray(answer.lesson_path) && answer.lesson_path.length) {
          supportCards.push(buildHistoryLessonPathCard(answer.lesson_path));
        }
        const keyTerms = Array.isArray(answer.key_terms) ? answer.key_terms.filter(Boolean).map((item) => cleanDisplayLabel(item)) : [];
        if (keyTerms.length) {
          supportCards.push(buildHistoryBulletCard('Key Terms', keyTerms));
        }
        const avoidItems = Array.isArray(answer.what_to_avoid) ? answer.what_to_avoid.filter(Boolean) : [];
        if (avoidItems.length) {
          supportCards.push(buildHistoryBulletCard('What To Avoid', avoidItems));
        }

        const chips = [
          `<span class="answer-chip"><strong>Mode:</strong> ${escapeHtml(modeDisplayLabel(answer.answer_mode || 'research'))}</span>`,
          `<span class="answer-chip"><strong>Madhhab:</strong> ${escapeHtml(answer.selected_madhhab || 'not_specified')}</span>`,
        ];
        if (citations.length) {
          chips.push(`<span class="answer-chip"><strong>Sources:</strong> ${citations.length}</span>`);
        }
        if (diagnostics && diagnostics.path_used) {
          chips.push(`<span class="answer-chip"><strong>Path:</strong> ${escapeHtml(diagnostics.path_used)}</span>`);
        }

        return `
          <div class="history-answer-shell">
            ${alertCards.length ? `<div class="answer-alert-stack">${alertCards.join('')}</div>` : ''}
            <section class="answer-card answer-summary-card wide">
              <div class="answer-summary-grid">
                <div>
                  <div class="section-label">Structured Answer</div>
                  <h3 class="answer-summary-title">${escapeHtml(answer.direct_answer || answer.evidence_summary || 'Grounded answer')}</h3>
                </div>
                <div class="answer-card-meta">${chips.join('')}</div>
              </div>
            </section>
            ${sections.length ? `<div class="history-answer-grid">${sections.join('')}</div>` : ''}
            ${layerCards.length ? `<div class="history-answer-layers">${layerCards.join('')}</div>` : ''}
            ${supportCards.length ? `<div class="history-answer-support">${supportCards.join('')}</div>` : ''}
          </div>
        `;
      }

      function buildStudyPathHistoryMarkup(answer) {
        const layers = answer.source_layers || {};
        const layerDefinitions = studyLayerDefinitions();
        const layerCards = layerDefinitions.map((definition) => {
          const items = Array.isArray(layers[definition.key]) ? layers[definition.key] : [];
          if (!items.length) {
            return '';
          }
          return `
            <section class="answer-card">
              <div class="answer-card-header">
                <div class="section-label">${escapeHtml(definition.label)}</div>
                <h3 class="answer-source-layer-title">${escapeHtml(definition.label)}</h3>
              </div>
              <div class="answer-source-list">
                ${items.map((item) => buildSourceCard(item)).join('')}
              </div>
              ${definition.footnote ? `<div class="answer-footnote">${escapeHtml(definition.footnote)}</div>` : ''}
            </section>
          `;
        }).filter(Boolean);

        const missing = layerDefinitions
          .filter((definition) => !Array.isArray(layers[definition.key]) || !layers[definition.key].length)
          .map((definition) => `${definition.label}: No grounded source found for this layer.`);
        const supportCards = [];
        if (missing.length) {
          supportCards.push(buildHistoryBulletCard('Missing Layers', missing));
        }
        if (answer.reading_list && typeof answer.reading_list === 'object') {
          supportCards.push(buildHistoryReadingListCard(answer.reading_list));
        }
        if (Array.isArray(answer.lesson_path) && answer.lesson_path.length) {
          supportCards.push(buildHistoryLessonPathCard(answer.lesson_path));
        }
        const keyTerms = Array.isArray(answer.key_terms) ? answer.key_terms.filter(Boolean).map((item) => cleanDisplayLabel(item)) : [];
        if (keyTerms.length) {
          supportCards.push(buildHistoryBulletCard('Key Terms', keyTerms));
        }
        const avoidItems = Array.isArray(answer.what_to_avoid) ? answer.what_to_avoid.filter(Boolean) : [];
        if (avoidItems.length) {
          supportCards.push(buildHistoryBulletCard('What To Avoid', avoidItems));
        }
        const gapItems = uniqueNonEmpty([
          answer.uncertainty_note || '',
          ...(Array.isArray(answer.corpus_gaps) ? answer.corpus_gaps : []),
        ]);
        if (gapItems.length) {
          supportCards.push(buildHistoryBulletCard('Uncertainty / Corpus Gaps', gapItems));
        }

        return `
          <div class="history-study-shell">
            <section class="answer-card answer-summary-card wide">
              <div class="answer-summary-grid">
                <div>
                  <div class="section-label">Study Path</div>
                  <h3 class="answer-summary-title">${escapeHtml(answer.topic || 'Grounded Study Path')}</h3>
                </div>
                <p class="answer-copy">${escapeHtml(answer.overview || answer.direct_answer || 'No overview is available yet.')}</p>
                <div class="answer-card-meta">
                  <span class="answer-chip"><strong>Madhhab:</strong> ${escapeHtml(answer.selected_madhhab || 'not_specified')}</span>
                  <span class="answer-chip"><strong>Mode:</strong> study_path</span>
                </div>
              </div>
            </section>
            ${layerCards.length ? `<div class="history-answer-layers">${layerCards.join('')}</div>` : ''}
            ${supportCards.length ? `<div class="history-answer-support">${supportCards.join('')}</div>` : ''}
          </div>
        `;
      }

      function previewMessageAnswer(messageId) {
        const session = workspaceState.activeSession;
        const messages = session && Array.isArray(session.messages) ? session.messages : [];
        const target = messages.find((message) => Number(message.id) === Number(messageId));
        if (!target || !target.evidence_json) {
          return;
        }
        setEnteredWorkspace(true);
        markAsked();
        renderAnswerSurface({
          answer: target.evidence_json.answer || {},
          rendered_text: target.evidence_json.rendered_text || target.content || '',
          diagnostics: target.evidence_json.diagnostics || {},
        });
        document.getElementById('pulse').textContent = 'Loaded saved answer into the desk.';
      }

      function renderConversationHistory(messages) {
        const root = document.getElementById('conversationHistory');
        const title = document.getElementById('historyTitle');
        const meta = document.getElementById('historyMeta');
        const shell = document.getElementById('historyShell');
        const session = workspaceState.activeSession;
        title.textContent = session && session.title ? session.title : 'Current session';
        if (!workspaceState.authenticated) {
          if (shell) {
            shell.hidden = true;
          }
          meta.textContent = 'Anonymous asks are not persisted. Sign in to restore saved sessions.';
          root.innerHTML = '<div class="sidebar-empty">Sign in and start a chat to keep a structured conversation history.</div>';
          return;
        }
        if (shell) {
          shell.hidden = false;
        }
        meta.textContent = session
          ? `Saved ${Array.isArray(messages) ? messages.length : 0} messages.`
          : 'Select a chat or create a new one to begin a persistent study session.';
        if (!Array.isArray(messages) || !messages.length) {
          if (shell && !session) {
            shell.hidden = true;
          }
          root.innerHTML = '<div class="sidebar-empty">This chat has no saved messages yet.</div>';
          return;
        }
        if (shell) {
          shell.hidden = false;
        }
        root.innerHTML = messages.map((message) => {
          const role = String(message.role || 'assistant');
          const structuredPayload = message.evidence_json || {};
          const answer = structuredPayload.answer || null;
          let contentMarkup = `<p class="history-copy">${escapeHtml(message.content || '')}</p>`;
          if (role === 'assistant' && answer && typeof answer === 'object') {
            contentMarkup = `
              <div class="history-rendered">
                ${answer.answer_mode === 'study_path'
                  ? buildStudyPathHistoryMarkup(answer)
                  : buildStructuredHistoryMarkup(answer, structuredPayload.diagnostics || {})}
                <div class="sidebar-inline-actions">
                  <button class="sidebar-link-button" type="button" onclick="previewMessageAnswer(${Number(message.id)})">View in answer panel</button>
                </div>
              </div>
            `;
          }
          return `
            <article class="history-message ${escapeHtml(role)}">
              <div class="history-message-header">
                <div class="history-role">${escapeHtml(role === 'assistant' ? 'Halal Jordan' : role.charAt(0).toUpperCase() + role.slice(1))}</div>
                <div class="history-time">${escapeHtml(formatTimestamp(message.created_at))}</div>
              </div>
              ${contentMarkup}
            </article>
          `;
        }).join('');
      }

      function renderLatestSavedAnswer() {
        const session = workspaceState.activeSession;
        const messages = session && Array.isArray(session.messages) ? session.messages : [];
        const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant' && message.evidence_json && message.evidence_json.answer);
        if (!latestAssistant) {
          resetAnswerSurface('Ready for the next grounded question.');
          return;
        }
        markAsked();
        previewMessageAnswer(latestAssistant.id);
      }

      async function refreshWorkspaceState(options) {
        const config = options || {};
        const response = await fetch('/api/auth/me');
        const payload = await response.json();
        workspaceState.authenticated = Boolean(payload && payload.authenticated);
        workspaceState.user = payload && payload.user ? payload.user : null;
        workspaceState.workspace = payload && payload.workspace ? payload.workspace : null;
        workspaceState.activeSession = null;
        hydrateWorkspaceSelection();
        renderSidebarChrome();
        applyWorkspaceDefaults();
        if (!workspaceState.authenticated) {
          renderConversationHistory([]);
          return;
        }
        const chats = filteredChats();
        if (!workspaceState.activeChatId && chats.length) {
          workspaceState.activeChatId = Number(chats[0].id);
          persistWorkspaceSelection();
        }
        renderChatList();
        if (config.loadSession !== false && workspaceState.activeChatId) {
          await loadChatSession(workspaceState.activeChatId, {renderLatestAnswer: config.renderLatestAnswer !== false});
        } else {
          renderConversationHistory([]);
        }
      }

      function setAuthView(view) {
        workspaceState.authView = view === 'register' ? 'register' : 'login';
        renderAuthPanel();
      }

      async function submitLogin(event) {
        event.preventDefault();
        const username = document.getElementById('authUsername').value;
        const password = document.getElementById('authPassword').value;
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username, password}),
        });
        const payload = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = payload.detail || 'Login failed.';
          pushStatus(payload.detail || 'Login failed.');
          return;
        }
        document.getElementById('pulse').textContent = 'Login successful.';
        pushStatus('Logged in locally.');
        await refreshWorkspaceState({renderLatestAnswer: true});
      }

      async function submitRegister(event) {
        event.preventDefault();
        const payload = {
          username: document.getElementById('authUsername').value,
          password: document.getElementById('authPassword').value,
          display_name: document.getElementById('authDisplayName').value,
          default_madhhab: document.getElementById('authDefaultMadhhab').value,
          default_answer_mode: document.getElementById('authDefaultMode').value,
        };
        const response = await fetch('/api/auth/register', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        const body = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = body.detail || 'Registration failed.';
          pushStatus(body.detail || 'Registration failed.');
          return;
        }
        document.getElementById('pulse').textContent = 'Account created and signed in.';
        pushStatus('Local account created.');
        await refreshWorkspaceState({renderLatestAnswer: true});
      }

      async function logoutUser() {
        await fetch('/api/auth/logout', {method: 'POST'});
        workspaceState.authenticated = false;
        workspaceState.user = null;
        workspaceState.workspace = null;
        workspaceState.activeChatId = null;
        workspaceState.activeProjectId = null;
        workspaceState.activeSession = null;
        workspaceState.sidebarExpanded = false;
        renderSidebarChrome();
        renderConversationHistory([]);
        syncUiState();
        document.getElementById('pulse').textContent = 'Signed out. Anonymous mode is active.';
        pushStatus('Signed out of local workspace.');
      }

      async function saveSettings() {
        if (!workspaceState.authenticated) {
          return;
        }
        const payload = {
          display_name: document.getElementById('settingsDisplayName').value,
          default_madhhab: document.getElementById('settingsMadhhab').value,
          default_answer_mode: document.getElementById('settingsAnswerMode').value,
        };
        const response = await fetch('/api/user/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          document.getElementById('pulse').textContent = 'Could not save settings.';
          pushStatus('Saving settings failed.');
          return;
        }
        workspaceState.userAdjustedMode = false;
        workspaceState.userAdjustedMadhhab = false;
        document.getElementById('pulse').textContent = 'Workspace defaults saved.';
        pushStatus('Local settings saved.');
        await refreshWorkspaceState({renderLatestAnswer: false});
      }

      async function createNewChat() {
        setEnteredWorkspace(true);
        if (!workspaceState.authenticated) {
          workspaceState.activeSession = null;
          workspaceState.activeChatId = null;
          renderConversationHistory([]);
          resetAnswerSurface('Anonymous chat ready. Sign in to persist this conversation.');
          document.getElementById('pulse').textContent = 'Anonymous chat ready.';
          pushStatus('Started an anonymous chat.');
          focusWorkspace({focusQuestion: true});
          return;
        }
        const response = await fetch('/api/user/chats', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            title: 'New Chat',
            project_id: workspaceState.activeProjectId || null,
          }),
        });
        const payload = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = payload.detail || 'Could not create chat.';
          pushStatus(payload.detail || 'Could not create chat.');
          return;
        }
        workspaceState.activeChatId = Number(payload.session.id);
        workspaceState.activeSession = payload.session;
        persistWorkspaceSelection();
        await refreshWorkspaceState({renderLatestAnswer: false});
        resetAnswerSurface('New persistent chat ready.');
        document.getElementById('pulse').textContent = 'New persistent chat ready.';
        pushStatus('Created a new persistent chat.');
        focusWorkspace({focusQuestion: true});
      }

      async function loadChatSession(sessionId, options) {
        if (!workspaceState.authenticated) {
          return;
        }
        setEnteredWorkspace(true);
        const response = await fetch(`/api/user/chats/${Number(sessionId)}`);
        const payload = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = payload.detail || 'Could not load chat session.';
          pushStatus(payload.detail || 'Could not load chat session.');
          return;
        }
        workspaceState.activeChatId = Number(sessionId);
        workspaceState.activeSession = payload.session;
        if (payload.session && payload.session.project_id) {
          workspaceState.activeProjectId = Number(payload.session.project_id);
        }
        persistWorkspaceSelection();
        renderSidebarChrome();
        renderConversationHistory(payload.session.messages || []);
        if (!options || options.renderLatestAnswer !== false) {
          renderLatestSavedAnswer();
        }
      }

      async function selectChat(sessionId) {
        await loadChatSession(sessionId, {renderLatestAnswer: true});
        document.getElementById('pulse').textContent = 'Persistent chat loaded.';
        pushStatus('Loaded saved chat history.');
      }

      function clearProjectFilter() {
        workspaceState.activeProjectId = null;
        workspaceState.userAdjustedMode = false;
        workspaceState.userAdjustedMadhhab = false;
        persistWorkspaceSelection();
        renderSidebarChrome();
        applyWorkspaceDefaults();
        document.getElementById('pulse').textContent = 'Project filter cleared.';
        pushStatus('Showing chats from every project.');
      }

      function selectProject(projectId) {
        setEnteredWorkspace(true);
        workspaceState.activeProjectId = Number(projectId);
        workspaceState.userAdjustedMode = false;
        workspaceState.userAdjustedMadhhab = false;
        persistWorkspaceSelection();
        renderSidebarChrome();
        applyWorkspaceDefaults();
        document.getElementById('pulse').textContent = 'Study project prepared.';
        pushStatus('Selected a study project. New chats will attach to it.');
      }

      async function createProject() {
        if (!workspaceState.authenticated) {
          document.getElementById('pulse').textContent = 'Sign in first to create a study project.';
          pushStatus('Project creation requires local sign-in.');
          return;
        }
        const payload = {
          title: document.getElementById('projectTitle').value,
          description: document.getElementById('projectDescription').value,
          madhhab: document.getElementById('projectMadhhab').value,
          study_mode: document.getElementById('projectStudyMode').value,
          status: 'active',
        };
        const response = await fetch('/api/user/projects', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        const body = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = body.detail || 'Could not create project.';
          pushStatus(body.detail || 'Could not create project.');
          return;
        }
        workspaceState.activeProjectId = Number(body.project.id);
        persistWorkspaceSelection();
        toggleProjectComposer(false);
        document.getElementById('projectTitle').value = '';
        document.getElementById('projectDescription').value = '';
        document.getElementById('projectStudyMode').value = 'standard';
        document.getElementById('pulse').textContent = 'Study project created.';
        pushStatus('Created a study project.');
        await refreshWorkspaceState({renderLatestAnswer: false});
      }

      async function updateProjectStudyProgress(payload, successMessage) {
        if (!workspaceState.authenticated || !workspaceState.activeProjectId) {
          return;
        }
        const response = await fetch(`/api/user/projects/${Number(workspaceState.activeProjectId)}/study/progress`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        const body = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = body.detail || 'Could not update study progress.';
          pushStatus(body.detail || 'Could not update study progress.');
          return;
        }
        await refreshWorkspaceState({renderLatestAnswer: true});
        document.getElementById('pulse').textContent = successMessage;
        pushStatus(successMessage);
      }

      async function revisitProjectLesson(index) {
        await updateProjectStudyProgress(
          {active_lesson_index: Number(index)},
          'Study lesson loaded for continuation.'
        );
      }

      async function markProjectLessonComplete(index) {
        await updateProjectStudyProgress(
          {completed_lesson_index: Number(index)},
          'Marked lesson complete.'
        );
      }

      async function continueProjectLesson() {
        await updateProjectStudyProgress(
          {advance: true},
          'Advanced to the next grounded lesson.'
        );
      }

      async function linkActiveChatToProject() {
        if (!workspaceState.authenticated || !workspaceState.activeChatId || !workspaceState.activeProjectId) {
          return;
        }
        const response = await fetch(`/api/user/chats/${Number(workspaceState.activeChatId)}/project`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({project_id: workspaceState.activeProjectId}),
        });
        const payload = await response.json();
        if (!response.ok) {
          document.getElementById('pulse').textContent = payload.detail || 'Could not link chat to project.';
          pushStatus(payload.detail || 'Could not link chat to project.');
          return;
        }
        workspaceState.activeSession = payload.session;
        document.getElementById('pulse').textContent = 'Active chat linked to the study project.';
        pushStatus('Linked the active chat to the current project.');
        await refreshWorkspaceState({renderLatestAnswer: false});
      }

      async function runAsk() {
        const questionField = document.getElementById('question');
        const question = questionField && questionField.value ? questionField.value.trim() : '';
        if (!question) {
          resetAnswerSurface('Enter a question to begin.');
          if (questionField) {
            questionField.focus();
          }
          document.getElementById('pulse').textContent = 'Type a grounded question first.';
          pushStatus('Waiting for a grounded question.');
          scrollAnswerSurfaceIntoView();
          return;
        }
        if (currentStream) {
          currentStream.close();
          currentStream = null;
        }
        setEnteredWorkspace(true);
        markAsked();
        setRequestActive(true);
        setRequestPhase('retrieving_sources', 'Retrieving sources...', false);
        const requestId = crypto.randomUUID();
        const sessionId = getRequestSessionId();
        setAskButtonsPending(true);
        resetAnswerSurface('Retrieving sources...');
        document.getElementById('pulse').textContent = 'Request in progress. Request ID: ' + requestId;
        document.getElementById('statusList').innerHTML = '';
        pushStatus('Request ID: ' + requestId);
        scrollAnswerSurfaceIntoView();
        currentStream = new EventSource('/api/chat/status/stream?request_id=' + encodeURIComponent(requestId));
        currentStream.onmessage = (event) => {
          const payload = JSON.parse(event.data);
          const summary = payload.summary || (payload.details && payload.details.summary) || payload.event_type;
          pushStatus(summary);
        };
        currentStream.addEventListener('request_phase_changed', (event) => {
          const payload = JSON.parse(event.data);
          const phase = (payload.details && payload.details.phase) || payload.summary;
          document.getElementById('pulse').textContent = 'Request status: ' + phase;
          setRequestPhase(phase, payload.summary || phase, Boolean(payload.details && payload.details.degraded_active));
          pushStatus(payload.summary);
          if (payload.details && payload.details.detected_intent && phase === 'receiving_request') {
            pushStatus('Detected intent: ' + payload.details.detected_intent);
          }
        });
        currentStream.addEventListener('degraded_answer_path', (event) => {
          const payload = JSON.parse(event.data);
          document.getElementById('pulse').textContent = 'Degraded path active.';
          setRequestPhase('generating_response', payload.summary || 'Degraded source-backed response active.', true);
          pushStatus(payload.summary);
        });
        const payload = {
          question,
          answer_mode: document.getElementById('mode').value,
          selected_madhhab: document.getElementById('madhhab').value,
          research_depth: document.getElementById('researchDepth').value,
          chat_session_id: workspaceState.activeChatId,
          project_id: workspaceState.activeProjectId,
        };
        try {
          const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Request-Id': requestId,
              'X-Session-Id': sessionId
            },
            body: JSON.stringify(payload)
          });
          const rawText = await response.text();
          let data = null;
          try {
            data = rawText ? JSON.parse(rawText) : {};
          } catch (parseError) {
            if (response.ok) {
              throw parseError;
            }
          }
          if (!response.ok) {
            throw new Error((data && resolveRenderedAnswerText(data)) || rawText || 'Request failed.');
          }
          renderAnswerSurface(data);
          setRequestPhase('completed', data.degraded ? 'Completed in degraded mode.' : 'Completed.', Boolean(data.degraded));
          if (workspaceState.authenticated) {
            const persistedChatId = Number((data.diagnostics && data.diagnostics.chat_session_id) || workspaceState.activeChatId || 0);
            const persistedProjectId = Number((data.diagnostics && data.diagnostics.study_project_id) || workspaceState.activeProjectId || 0);
            if (persistedChatId) {
              workspaceState.activeChatId = persistedChatId;
            }
            if (persistedProjectId) {
              workspaceState.activeProjectId = persistedProjectId;
            }
            persistWorkspaceSelection();
            await refreshWorkspaceState({renderLatestAnswer: true});
          }
          document.getElementById('pulse').textContent =
            data.degraded ? 'Completed in degraded mode.' : 'Completed.';
          scrollAnswerSurfaceIntoView();
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          resetAnswerSurface('Request failed: ' + message);
          setRequestPhase('receiving_request', 'Request failed before completion.', true);
          document.getElementById('pulse').textContent = 'Request failed.';
          pushStatus('Request failed before completion.');
          pushStatus(message);
          scrollAnswerSurfaceIntoView();
        } finally {
          setRequestActive(false);
          setAskButtonsPending(false);
          if (currentStream) {
            setTimeout(() => {
              if (currentStream) {
                currentStream.close();
                currentStream = null;
              }
            }, 500);
          }
        }
      }

      refreshRetrievalStatus();
      window.setInterval(() => {
        refreshRetrievalStatus();
      }, 2500);
      loadLandingGlance();
      setModeSelection(document.getElementById('mode').value);
      setMadhhabSelection(document.getElementById('madhhab').value);
      setResearchDepthSelection(document.getElementById('researchDepth').value);
      syncDeskAskToHero();
      syncUiState();
      document.getElementById('mode').addEventListener('change', (event) => {
        workspaceState.userAdjustedMode = true;
        setModeSelection(event.target.value);
      });
      document.getElementById('landingMode').addEventListener('change', (event) => {
        workspaceState.userAdjustedMode = true;
        setModeSelection(event.target.value);
      });
      document.getElementById('madhhab').addEventListener('change', () => {
        workspaceState.userAdjustedMadhhab = true;
        setMadhhabSelection(document.getElementById('madhhab').value);
      });
      document.getElementById('landingMadhhab').addEventListener('change', () => {
        workspaceState.userAdjustedMadhhab = true;
        setMadhhabSelection(document.getElementById('landingMadhhab').value);
      });
      document.getElementById('researchDepth').addEventListener('change', (event) => {
        setResearchDepthSelection(event.target.value);
      });
      document.getElementById('landingResearchDepth').addEventListener('change', (event) => {
        setResearchDepthSelection(event.target.value);
      });
      document.getElementById('question').addEventListener('input', () => {
        const landingQuestion = document.getElementById('landingQuestion');
        if (landingQuestion) {
          landingQuestion.value = document.getElementById('question').value;
        }
      });
      document.getElementById('landingQuestion').addEventListener('input', () => {
        document.getElementById('question').value = document.getElementById('landingQuestion').value;
      });
      document.getElementById('landingQuestion').addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          runHeroAsk();
        }
      });
      refreshWorkspaceState({renderLatestAnswer: true}).catch(() => {
        renderSidebarChrome();
        renderConversationHistory([]);
      });
    </script>
  </body>
</html>
""".replace("__SCHOLAR_PROFILE_MAP__", json.dumps(scholar_profiles, ensure_ascii=False)).replace("__INITIAL_AUTH_VIEW__", initial_auth_view)


def _manager_scholar_profile_map(repo_root: Path) -> dict[str, dict[str, Any]]:
    try:
        profiles = load_scholar_profiles(repo_root)
    except Exception:
        return {}
    return {
        scholar_id: {
            "scholar_id": profile.scholar_id,
            "name": profile.name,
            "name_transliterated": profile.name_transliterated,
            "madhhab": profile.madhhab,
            "period": profile.period,
            "methodology_notes": profile.methodology_notes,
            "known_works": list(profile.known_works),
            "retrieval_tags": list(profile.retrieval_tags),
            "aliases": list(profile.aliases),
        }
        for scholar_id, profile in profiles.items()
    }


def _public_ask(
    ops: Any,
    *,
    repo_root: Path,
    question: str,
    selected_madhhab: str,
    conversation_context: list[str] | None = None,
) -> dict[str, Any]:
    """Run retrieval + render an answer using the same structural layers
    the full chat pipeline uses. No LLM call required — works in any
    environment, including fresh laptops without Ollama running.

    The Trust Engine, Evidence Ladder, Confidence Taxonomy, and (when
    applicable) Methodology Disclosure all surface in the rendered
    output exactly as they would in a synthesized answer.

    Conversation context (a list of prior questions from the same
    thread) is used as retrieval-time topic biasing only — the answer
    is still about the CURRENT question. We never fabricate dialogue
    continuity; the prior questions just keep the retrieval focused.
    """

    from app.citations.renderer import render_answer
    from app.reasoning.answer_grounding import AnswerEvidenceModel, GroundedSource
    from app.reasoning.confidence_taxonomy import classify_confidence
    from app.reasoning.evidence_ladder import classify_sources
    from app.reasoning.trust_engine import list_profiles_with_metadata
    from app.retrieval.pipeline import RetrievalPipeline

    # Retrieval pipeline reads trust_profile_id from runtime config and
    # attaches the breakdown to each snippet automatically.
    pipeline = RetrievalPipeline(repo_root=repo_root)
    retrieval_question = _build_retrieval_query(question, conversation_context)
    debug = pipeline.retrieve_with_debug(
        retrieval_question,
        selected_madhhab=selected_madhhab,
        top_k=6,
        answer_mode="research",
    )

    grounded = [_grounded_source_from_snippet(snippet) for snippet in debug.snippets]
    evidence_model = AnswerEvidenceModel(
        primary_evidence=[g for g in grounded if g.source_classification in {"quran", "hadith"}],
        spiritual_guidance=[g for g in grounded if g.source_classification == "tasawwuf_text"],
        hanafi_authority=[],
        other_views=[],
        supporting_commentary=[g for g in grounded if g.source_classification == "commentary"],
        teaching_explanation=[],
        modern_application=[g for g in grounded if g.source_classification == "fatwa"],
        sources=grounded,
        disagreement_notes=[],
        uncertainty_notes=(
            ["No sources were retrieved for this question."] if not grounded else []
        ),
        intent_id=debug.query_intent.intent_id,
        suppress_synthesis=True,
        authority_policy_id="research",
        comparison_positions=[],
        evidence_backfill_applied=False,
        evidence_backfill_buckets=[],
        source_layer_composition={},
        metadata_completeness={},
        ocr_usage={},
    )

    minimal_answer = {
        "answer_mode": "source_only",
        "selected_madhhab": selected_madhhab,
        "direct_answer": (
            "Here is what local sources say. This is research output — "
            "the engine surfaces evidence and weighting transparently, "
            "rather than issuing a ruling."
            if grounded
            else "No local sources matched this question. Try rephrasing, "
            "or check the corpus stats on the admin page."
        ),
    }
    rendered_text = render_answer(minimal_answer, evidence_model)

    # Structured payload for the page to display however it likes.
    ladder = classify_sources(grounded)
    confidence = classify_confidence(evidence_model=evidence_model, answer=minimal_answer)

    profile_meta = next(
        (
            entry
            for entry in list_profiles_with_metadata(repo_root=repo_root)
            if entry["profile_id"] == debug.trust_profile_id
        ),
        None,
    )

    return {
        "question": question,
        "retrieval_question": retrieval_question,
        "selected_madhhab": selected_madhhab,
        "rendered_text": rendered_text,
        "profile_id": debug.trust_profile_id,
        "profile": profile_meta,
        "evidence_ladder": [
            {
                "tier_id": tier.tier_id,
                "rank": tier.rank,
                "label": tier.label,
                "count": len(ladder.entries_for(tier.tier_id)),
            }
            for tier in ladder.populated_tiers()
        ],
        "confidence": confidence.to_dict() if confidence else None,
        "source_count": len(grounded),
        "trust_diagnostics": debug.trust_diagnostics,
        "conversation_context_used": list(conversation_context or []),
    }


def _build_retrieval_query(question: str, context: list[str] | None) -> str:
    """Compose a retrieval query that biases toward the conversation
    topic without burying the current question.

    The current question is what the user wants answered. Prior
    questions act as soft topic markers — appended in a compact form so
    BM25 picks up shared terms but the current question still dominates
    relevance.
    """

    if not context:
        return question
    seen: set[str] = set()
    topic_words: list[str] = []
    for prior in context:
        for word in str(prior).split():
            normalized = word.strip().lower()
            if len(normalized) < 4 or normalized in seen:
                continue
            seen.add(normalized)
            topic_words.append(word.strip())
            if len(topic_words) >= 12:
                break
        if len(topic_words) >= 12:
            break
    if not topic_words:
        return question
    return f"{question} ({' '.join(topic_words)})"


def _grounded_source_from_snippet(snippet: dict[str, str]) -> Any:
    """Build a GroundedSource from a retrieval snippet dict.

    Snippets are str-typed for transport; GroundedSource fields are
    str or bool. Trust breakdown rides as JSON in the snippet and
    flows straight through.
    """

    from app.reasoning.answer_grounding import GroundedSource

    def _b(value: str) -> bool:
        return str(value or "").strip().lower() in {"true", "1", "yes"}

    return GroundedSource(
        title=snippet.get("title", ""),
        human_title=snippet.get("title", ""),
        source_classification=snippet.get("source_classification") or snippet.get("source_type", ""),
        source_type_label="",
        evidence_bucket="primary_evidence",
        role=snippet.get("role", ""),
        domain=snippet.get("domain", ""),
        authority_level=snippet.get("authority_level", "") or snippet.get("hadith_grade", ""),
        reference=snippet.get("reference", ""),
        section_label=snippet.get("section_label", ""),
        quote=snippet.get("quote", ""),
        madhhab=snippet.get("madhhab", ""),
        source_path=snippet.get("source_path", ""),
        collection=snippet.get("collection", ""),
        author=snippet.get("author", ""),
        source_family=snippet.get("source_family", ""),
        canonical_family=snippet.get("canonical_family", ""),
        language=snippet.get("language", ""),
        hierarchy_label=snippet.get("hierarchy_label", ""),
        book=snippet.get("book", ""),
        chapter=snippet.get("chapter", ""),
        section=snippet.get("section", ""),
        document_kind=snippet.get("document_kind", ""),
        source_role_boundary=snippet.get("source_role_boundary", ""),
        source_lineage=snippet.get("source_lineage", ""),
        commentary_target="",
        fatwa_authority="",
        legal_role="",
        legal_role_label="",
        ocr_derived=_b(snippet.get("ocr_derived", "")),
        ocr_backend=snippet.get("ocr_backend", ""),
        ocr_status=snippet.get("ocr_status", ""),
        ocr_confidence=snippet.get("ocr_confidence", ""),
        extraction_status=snippet.get("extraction_status", ""),
        extraction_quality=snippet.get("extraction_quality", ""),
        scholar_attribution_match="",
        trust_breakdown_json=snippet.get("_trust_breakdown_json", ""),
    )


def _ask_html(*, current_profile_id: str) -> str:
    """Self-contained single-page UI for the public /ask endpoint.

    Multi-turn: each Q&A appears as a card; follow-ups stay on the same
    page and inherit the conversation topic. No JS framework, no
    external assets. Works offline, on phones, on laptops, on tablets.

    Conversation context (the list of prior questions) is sent on
    follow-ups to bias retrieval — the answer is still about the
    current question, not a synthesized dialogue continuation.
    """

    import html as _html

    profile_label = _html.escape(_humanize_profile_id(current_profile_id))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Halal Jordan</title>
    <style>
      * {{ box-sizing: border-box; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 0;
        padding: 1.5rem 1rem 4rem;
        background: #f3ead8;
        color: #2f271f;
        line-height: 1.55;
      }}
      .container {{ max-width: 48rem; margin: 0 auto; }}
      header {{ padding: 0.5rem 0 1rem; }}
      h1 {{ margin: 0 0 0.25rem; font-size: 1.6rem; }}
      header p {{ margin: 0; color: #6c5f4e; font-size: 0.95rem; }}
      .profile-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        margin-top: 0.6rem;
        background: rgba(255, 250, 241, 0.94);
        border: 1px solid rgba(120, 95, 60, 0.25);
        border-radius: 999px;
        text-decoration: none;
        color: #2f271f;
        font-size: 0.85rem;
      }}
      .profile-chip:hover {{ border-color: rgba(120, 95, 60, 0.55); }}
      .profile-chip strong {{ color: #6c5f4e; font-weight: 600; }}
      form {{ margin: 1.25rem 0 0; display: grid; gap: 0.75rem; }}
      label {{ font-weight: 600; font-size: 0.95rem; }}
      textarea {{
        width: 100%;
        min-height: 5rem;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 2px solid rgba(120, 95, 60, 0.25);
        font: inherit;
        background: rgba(255, 252, 246, 0.96);
        resize: vertical;
      }}
      textarea:focus {{ outline: none; border-color: #b8862c; }}
      .row {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
      select {{
        padding: 0.55rem 0.7rem;
        border-radius: 10px;
        border: 1px solid rgba(120, 95, 60, 0.25);
        background: rgba(255, 252, 246, 0.96);
        font: inherit;
      }}
      button.primary {{
        padding: 0.7rem 1.6rem;
        border-radius: 999px;
        border: none;
        background: #b8862c;
        color: white;
        font: inherit;
        font-weight: 600;
        cursor: pointer;
      }}
      button.primary:hover {{ background: #9d731f; }}
      button.primary:disabled {{ background: #b9b2a4; cursor: progress; }}
      #turns {{ margin-top: 1.25rem; display: grid; gap: 1rem; }}
      .turn {{
        padding: 1rem 1.25rem;
        background: rgba(255, 252, 246, 0.96);
        border-radius: 14px;
        border: 1px solid rgba(120, 95, 60, 0.2);
      }}
      .turn .q {{
        font-weight: 600;
        color: #6c5f4e;
        margin: 0 0 0.5rem;
        font-size: 0.95rem;
      }}
      .turn .meta {{
        font-size: 0.78rem;
        color: #9b8d77;
        margin: 0 0 0.6rem;
      }}
      .turn pre {{
        margin: 0;
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.88rem;
        line-height: 1.55;
      }}
      .turn.placeholder pre {{ font-family: inherit; font-style: italic; color: #6c5f4e; }}
      .hint {{ font-size: 0.8rem; color: #6c5f4e; margin-top: 0.25rem; }}
      .toolbar {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
      button.ghost {{
        background: transparent;
        border: 1px solid rgba(120, 95, 60, 0.3);
        color: #6c5f4e;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        font: inherit;
        cursor: pointer;
      }}
      button.ghost:hover {{ border-color: #b8862c; color: #2f271f; }}
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>Halal Jordan</h1>
        <p>Ask a question. The system finds local sources and shows you the
        evidence transparently — not a ruling.</p>
        <a class="profile-chip" href="/profiles">
          <strong>Active profile:</strong>
          <span id="profile-label">{profile_label}</span>
          <span aria-hidden="true">›</span>
        </a>
      </header>
      <form id="ask-form">
        <label for="question" id="question-label">Your question</label>
        <textarea id="question" name="question" placeholder="e.g. What do hadith say about sincerity of intention?"></textarea>
        <div class="row">
          <select id="madhhab" name="madhhab">
            <option value="not_specified">Madhhab: any</option>
            <option value="hanafi">Hanafi</option>
            <option value="shafii">Shafi'i</option>
            <option value="maliki">Maliki</option>
            <option value="hanbali">Hanbali</option>
            <option value="compare_all">Compare all four</option>
          </select>
          <button class="primary" type="submit">Ask</button>
          <button class="ghost" type="button" id="clear-btn" hidden>Clear conversation</button>
        </div>
        <p class="hint">Retrieval-only mode — works in any environment.
        Follow-ups stay on topic without fabricating dialogue continuity.
        Full chat synthesis is at <a href="/workspace">the workspace</a> if you need it.</p>
      </form>
      <div id="turns">
        <div class="turn placeholder" id="placeholder">
          <pre>Your answer will appear here.</pre>
        </div>
      </div>
    </div>
    <script>
      (function () {{
        var form = document.getElementById("ask-form");
        var turnsEl = document.getElementById("turns");
        var placeholder = document.getElementById("placeholder");
        var btn = form.querySelector("button.primary");
        var clearBtn = document.getElementById("clear-btn");
        var questionLabel = document.getElementById("question-label");
        var questionInput = document.getElementById("question");
        var conversation = [];  // list of prior question strings

        function humanizeProfile(id) {{
          return String(id || "").replace(/_/g, " ").replace(/\\b\\w/g, function (c) {{
            return c.toUpperCase();
          }});
        }}

        function refreshUI() {{
          if (conversation.length === 0) {{
            questionLabel.textContent = "Your question";
            questionInput.placeholder = "e.g. What do hadith say about sincerity of intention?";
            clearBtn.hidden = true;
          }} else {{
            questionLabel.textContent = "Follow up";
            questionInput.placeholder = "Ask something related, or pivot to a new question.";
            clearBtn.hidden = false;
          }}
        }}

        function appendTurn(question, profileLabel, renderedText, sourceCount, confidenceLabel) {{
          if (placeholder && placeholder.parentNode) {{
            placeholder.parentNode.removeChild(placeholder);
          }}
          var turn = document.createElement("div");
          turn.className = "turn";

          var qEl = document.createElement("p");
          qEl.className = "q";
          qEl.textContent = question;
          turn.appendChild(qEl);

          var metaEl = document.createElement("p");
          metaEl.className = "meta";
          var bits = [];
          if (profileLabel) bits.push("Profile: " + profileLabel);
          if (typeof sourceCount === "number") bits.push(sourceCount + " source" + (sourceCount === 1 ? "" : "s"));
          if (confidenceLabel) bits.push("Confidence: " + confidenceLabel);
          metaEl.textContent = bits.join(" · ");
          turn.appendChild(metaEl);

          var pre = document.createElement("pre");
          pre.textContent = renderedText || "(no output)";
          turn.appendChild(pre);

          turnsEl.appendChild(turn);
          turn.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
        }}

        clearBtn.addEventListener("click", function () {{
          conversation = [];
          turnsEl.innerHTML = "";
          var ph = document.createElement("div");
          ph.id = "placeholder";
          ph.className = "turn placeholder";
          var pre = document.createElement("pre");
          pre.textContent = "Your answer will appear here.";
          ph.appendChild(pre);
          turnsEl.appendChild(ph);
          placeholder = ph;
          refreshUI();
        }});

        form.addEventListener("submit", function (e) {{
          e.preventDefault();
          var question = questionInput.value.trim();
          var madhhab = document.getElementById("madhhab").value;
          if (!question) return;
          btn.disabled = true;
          var oldText = btn.textContent;
          btn.textContent = "Thinking…";

          fetch("/api/ask", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
              question: question,
              selected_madhhab: madhhab,
              conversation_context: conversation.slice(),
            }}),
          }})
            .then(function (r) {{
              if (!r.ok) return r.text().then(function (t) {{ throw new Error(t || ("HTTP " + r.status)); }});
              return r.json();
            }})
            .then(function (data) {{
              var profileLabel = humanizeProfile(data.profile_id || "");
              var confLabel = data.confidence ? data.confidence.label : "";
              appendTurn(question, profileLabel, data.rendered_text, data.source_count, confLabel);
              conversation.push(question);
              if (data.profile_id) {{
                document.getElementById("profile-label").textContent = profileLabel;
              }}
              questionInput.value = "";
              refreshUI();
            }})
            .catch(function (err) {{
              appendTurn(question, "", "Error: " + (err.message || err), null, "");
            }})
            .finally(function () {{
              btn.disabled = false;
              btn.textContent = oldText;
              questionInput.focus();
            }});
        }});

        refreshUI();
      }})();
    </script>
  </body>
</html>"""


def _profiles_html(*, profiles: list[dict[str, Any]], current_profile_id: str) -> str:
    """One-page button selector for the active reasoning profile.

    Deliberately self-contained: no JS framework, no external assets, no
    auth. Works in any browser, including offline laptop builds and
    minimal environments. This is the user-facing 'button' surface for
    the charter's Scholar Methodology Profiles and Weight & Trust
    Engine.
    """

    scholar_buttons: list[str] = []
    generic_buttons: list[str] = []
    for profile in profiles:
        button_html = _profile_button_html(profile, current_profile_id)
        if profile.get("is_scholar_methodology"):
            scholar_buttons.append(button_html)
        else:
            generic_buttons.append(button_html)

    scholar_section = ""
    if scholar_buttons:
        scholar_section = (
            "<section><h2>Scholar Methodology</h2>"
            "<p class=\"section-note\">Methodology modeling — not the actual scholar speaking, "
            "not a fatwa, not divine certainty. Each profile is a transparent weighting "
            "pattern you can audit and edit.</p>"
            "<div class=\"button-grid\">" + "".join(scholar_buttons) + "</div></section>"
        )

    generic_section = (
        "<section><h2>Research Modes</h2>"
        "<p class=\"section-note\">Generic weighting profiles. Pick one for the kind of "
        "research you're doing right now.</p>"
        "<div class=\"button-grid\">" + "".join(generic_buttons) + "</div></section>"
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Halal Jordan — Profiles</title>
    <style>
      * {{ box-sizing: border-box; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 0;
        padding: 1.5rem;
        background: #f3ead8;
        color: #2f271f;
        line-height: 1.5;
      }}
      .container {{ max-width: 64rem; margin: 0 auto; }}
      header {{ padding: 0.5rem 0 1.5rem; }}
      h1 {{ margin: 0 0 0.25rem; font-size: 1.75rem; }}
      header p {{ margin: 0; color: #6c5f4e; }}
      a.home-link {{ color: #6c5f4e; text-decoration: none; font-size: 0.9rem; }}
      a.home-link:hover {{ text-decoration: underline; }}
      h2 {{ margin: 1.75rem 0 0.5rem; font-size: 1.25rem; }}
      .section-note {{ color: #6c5f4e; font-size: 0.95rem; margin: 0 0 1rem; }}
      .button-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr));
        gap: 1rem;
      }}
      .profile-button {{
        display: block;
        width: 100%;
        text-align: left;
        background: rgba(255, 250, 241, 0.94);
        border: 2px solid rgba(120, 95, 60, 0.2);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        font: inherit;
        color: inherit;
        cursor: pointer;
        transition: transform 0.08s ease, border-color 0.08s ease, background 0.08s ease;
      }}
      .profile-button:hover {{
        border-color: rgba(120, 95, 60, 0.5);
        transform: translateY(-1px);
      }}
      .profile-button.active {{
        background: #f6d27a;
        border-color: #b8862c;
      }}
      .profile-title {{
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0 0 0.25rem;
      }}
      .profile-meta {{ font-size: 0.85rem; color: #6c5f4e; margin: 0 0 0.5rem; }}
      .profile-description {{ font-size: 0.95rem; margin: 0; }}
      .profile-disclaimer {{
        font-size: 0.8rem;
        color: #7a5a2a;
        margin: 0.6rem 0 0;
        padding: 0.5rem 0.6rem;
        background: rgba(247, 220, 160, 0.45);
        border-left: 3px solid #b8862c;
        border-radius: 4px;
      }}
      .badge {{
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: #b8862c;
        color: white;
        margin-left: 0.4rem;
        vertical-align: middle;
        text-transform: uppercase;
      }}
      .toast {{
        position: fixed;
        bottom: 1rem;
        left: 50%;
        transform: translateX(-50%);
        background: #2f271f;
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 999px;
        opacity: 0;
        transition: opacity 0.2s ease;
        z-index: 100;
      }}
      .toast.show {{ opacity: 1; }}
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>Choose a Reasoning Profile</h1>
        <p>One click switches how Halal Jordan weights evidence. Active profile is highlighted.</p>
        <p><a class="home-link" href="/">&larr; Back to Halal Jordan</a></p>
      </header>
      {scholar_section}
      {generic_section}
    </div>
    <div id="toast" class="toast"></div>
    <script>
      (function () {{
        var toastEl = document.getElementById("toast");
        function showToast(message) {{
          toastEl.textContent = message;
          toastEl.classList.add("show");
          setTimeout(function () {{ toastEl.classList.remove("show"); }}, 2200);
        }}
        document.querySelectorAll(".profile-button").forEach(function (btn) {{
          btn.addEventListener("click", function () {{
            var profileId = btn.getAttribute("data-profile-id");
            if (!profileId) return;
            btn.disabled = true;
            fetch("/api/profile/set", {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{ profile_id: profileId }}),
            }})
              .then(function (r) {{
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
              }})
              .then(function (data) {{
                showToast("Active profile: " + data.profile_id);
                document.querySelectorAll(".profile-button").forEach(function (b) {{
                  b.classList.toggle("active", b.getAttribute("data-profile-id") === profileId);
                }});
              }})
              .catch(function (err) {{ showToast("Failed: " + err.message); }})
              .finally(function () {{ btn.disabled = false; }});
          }});
        }});
      }})();
    </script>
  </body>
</html>"""


def _profile_button_html(profile: dict[str, Any], current_profile_id: str) -> str:
    """Render one profile as a big button. HTML-escaped throughout."""

    import html as _html

    profile_id = str(profile.get("profile_id", "") or "")
    is_current = profile_id == current_profile_id
    active_class = " active" if is_current else ""
    badge = '<span class="badge">Active</span>' if is_current else ""

    if profile.get("is_scholar_methodology"):
        title = _html.escape(str(profile.get("scholar_name") or profile_id))
        description = _html.escape(str(profile.get("methodology_overview") or ""))
        meta = _html.escape(f"Methodology profile · {profile.get('mode', 'balanced')}")
    else:
        title = _html.escape(_humanize_profile_id(profile_id))
        description = _html.escape(str(profile.get("description") or ""))
        meta = _html.escape(f"Research profile · {profile.get('mode', 'balanced')}")

    disclaimer_html = ""
    disclaimer_text = str(profile.get("methodology_disclaimer") or "").strip()
    if disclaimer_text:
        disclaimer_html = (
            f'<p class="profile-disclaimer">{_html.escape(disclaimer_text)}</p>'
        )

    return (
        f'<button class="profile-button{active_class}" '
        f'data-profile-id="{_html.escape(profile_id)}">'
        f'<p class="profile-title">{title}{badge}</p>'
        f'<p class="profile-meta">{meta}</p>'
        f'<p class="profile-description">{description}</p>'
        f"{disclaimer_html}"
        "</button>"
    )


def _humanize_profile_id(profile_id: str) -> str:
    return " ".join(part.capitalize() for part in profile_id.replace("_", " ").split())


def _admin_html() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Halal Jordan Admin</title>
    <style>
      body { font-family: sans-serif; margin: 2rem; max-width: 96rem; background: #faf9f6; color: #1f1f1f; }
      pre, .panel { background: #f4f4f4; padding: 1rem; white-space: pre-wrap; border: 1px solid #ddd; border-radius: 12px; }
      .grid { display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 1rem; align-items: start; }
      .stack { display: grid; gap: 1rem; }
      .list { list-style: none; padding: 0; margin: 0; }
      .list li { padding: 0.3rem 0; border-bottom: 1px solid #e5e5e5; }
      .pulse { font-weight: 600; margin-bottom: 1rem; }
      .panel-title { font-weight: 700; margin-bottom: 0.5rem; }
      .panel-note { margin: 0 0 0.85rem; color: #555; font-size: 0.95rem; line-height: 1.45; }
      .button-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 0.85rem; }
      button {
        appearance: none;
        border: 1px solid #c8b998;
        background: #fffaf0;
        color: #2a2416;
        padding: 0.65rem 0.9rem;
        border-radius: 999px;
        cursor: pointer;
        font: inherit;
      }
      button:hover { background: #f5ecd6; }
      button:disabled { opacity: 0.7; cursor: wait; }
      .micro-tier-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
      }
      .micro-tier-card {
        display: grid;
        gap: 0.45rem;
      }
      .micro-tier-label {
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #6c5b3e;
      }
      .micro-status { min-height: 16rem; }
      .update-select {
        width: 100%;
        margin-bottom: 0.85rem;
        padding: 0.65rem 0.75rem;
        border: 1px solid #d4c8aa;
        border-radius: 0.85rem;
        background: #fffdf8;
        color: #2a2416;
        font: inherit;
      }
      .update-status, .update-detail, .update-history { min-height: 10rem; }
      @media (max-width: 1100px) {
        .grid { grid-template-columns: 1fr; }
        .micro-tier-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <h1>Jenna Admin</h1>
    <p>Operational oversight console with live task and runtime supervision.</p>
    <div class="pulse" id="pulse">Connecting...</div>
    <div class="grid">
      <div class="stack">
        <div class="panel">
          <div class="panel-title">Active Tasks</div>
          <ul class="list" id="activeTasks"><li>Loading...</li></ul>
        </div>
        <div class="panel">
          <div class="panel-title">Recent Events</div>
          <ul class="list" id="recentEvents"><li>Loading...</li></ul>
        </div>
      </div>
      <div class="stack">
        <div class="panel">
          <div class="panel-title">Failures / Warnings</div>
          <ul class="list" id="failures"><li>Loading...</li></ul>
        </div>
        <div class="panel">
          <div class="panel-title">Model / Retrieval</div>
          <pre id="runtimeTruth">Loading...</pre>
        </div>
      </div>
      <div class="stack">
        <div class="panel">
          <div class="panel-title">Micro LLM Status</div>
          <p class="panel-note">Advisory only. Does not affect final answers.</p>
          <div class="button-row">
            <button id="microShadowSmokeButton" type="button" onclick="runMicroShadowSmoke()">Run Dual Smoke Test</button>
            <button id="microShadowBenchmarkButton" type="button" onclick="runMicroShadowBenchmark()">Run Dual Benchmark</button>
            <button type="button" onclick="refreshMicroShadowStatus()">Refresh Status</button>
            <button type="button" onclick="openPortableReadiness()">Open Portable Readiness</button>
          </div>
          <div class="micro-tier-grid">
            <div class="micro-tier-card">
              <div class="micro-tier-label">Fast Router Status</div>
              <pre class="micro-status" id="microFastStatus">Loading...</pre>
            </div>
            <div class="micro-tier-card">
              <div class="micro-tier-label">Smart Shadow Status</div>
              <pre class="micro-status" id="microSmartStatus">Loading...</pre>
            </div>
            <div class="micro-tier-card">
              <div class="micro-tier-label">Latest Comparison</div>
              <pre class="micro-status" id="microTierComparison">Loading...</pre>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-title">Update Center</div>
          <p class="panel-note">Optional internet check only. Downloads stay project-local in <code>updates/</code> and require explicit review before any install action.</p>
          <div class="button-row">
            <button id="updateCheckButton" type="button" onclick="checkForUpdates()">Check for Updates</button>
            <button id="updateDownloadButton" type="button" onclick="downloadSelectedUpdate()">Download Selected</button>
            <button id="updateVerifyButton" type="button" onclick="verifySelectedUpdate()">Verify Checksum</button>
            <button id="updateApplyButton" type="button" onclick="applySelectedUpdate()">Apply Selected</button>
            <button id="updateRebuildButton" type="button" onclick="rebuildIndexFromUpdates()">Rebuild Index if Needed</button>
          </div>
          <select id="updateSelection" class="update-select" onchange="renderSelectedUpdateDetail()">
            <option value="">No update selected</option>
          </select>
          <pre class="update-status" id="updateSummary">Loading...</pre>
          <pre class="update-detail" id="updateSelectedDetail">Loading...</pre>
          <pre class="update-history" id="updateHistory">Loading...</pre>
        </div>
        <div class="panel">
          <div class="panel-title">System Pulse</div>
          <pre id="systemPulse">Loading...</pre>
        </div>
        <div class="panel">
          <div class="panel-title">Overview</div>
          <pre id="overview">Loading...</pre>
        </div>
      </div>
    </div>
    <script>
      const roleHeaders = {'X-HJ-Role': 'admin'};
      let updateCenterState = null;

      function renderList(id, items, formatter) {
        const list = document.getElementById(id);
        list.innerHTML = '';
        if (!items || !items.length) {
          const item = document.createElement('li');
          item.textContent = 'None';
          list.appendChild(item);
          return;
        }
        items.slice(0, 8).forEach((entry) => {
          const item = document.createElement('li');
          item.textContent = formatter(entry);
          list.appendChild(item);
        });
      }

      function prettyJson(value) {
        return JSON.stringify(value, null, 2);
      }

      function selectedUpdateId() {
        const select = document.getElementById('updateSelection');
        return select ? String(select.value || '').trim() : '';
      }

      function setUpdateButtonsDisabled(disabled) {
        ['updateCheckButton', 'updateDownloadButton', 'updateVerifyButton', 'updateApplyButton', 'updateRebuildButton'].forEach((id) => {
          const button = document.getElementById(id);
          if (button) {
            button.disabled = disabled;
          }
        });
      }

      function renderUpdateCenter() {
        const summary = document.getElementById('updateSummary');
        const detail = document.getElementById('updateSelectedDetail');
        const history = document.getElementById('updateHistory');
        const select = document.getElementById('updateSelection');
        if (!updateCenterState) {
          summary.textContent = 'No update status loaded yet.';
          detail.textContent = 'No update selected.';
          history.textContent = 'No history loaded yet.';
          select.innerHTML = '<option value="">No update selected</option>';
          return;
        }
        summary.textContent = prettyJson({
          internet_update_check_enabled: updateCenterState.internet_update_check_enabled,
          update_manifest_url: updateCenterState.update_manifest_url || '',
          last_update_check: updateCenterState.last_update_check,
          last_update_check_status: updateCenterState.last_update_check_status,
          last_update_check_error: updateCenterState.last_update_check_error,
          pending_updates_count: updateCenterState.pending_updates_count,
          downloaded_updates_count: updateCenterState.downloaded_updates_count,
          verified_updates_count: updateCenterState.verified_updates_count,
          index_rebuild_needed: updateCenterState.index_rebuild_needed,
          offline_mode: updateCenterState.offline_mode,
          inbox_dir: updateCenterState.inbox_dir,
          processed_dir: updateCenterState.processed_dir
        });

        const currentValue = String(select.value || '');
        select.innerHTML = '<option value="">No update selected</option>';
        (updateCenterState.available_updates || []).forEach((entry) => {
          const option = document.createElement('option');
          option.value = entry.update_id;
          option.textContent = `${entry.update_id} (${entry.manifest.type} ${entry.manifest.version}) - ${entry.local_status}`;
          if (entry.update_id === currentValue) {
            option.selected = true;
          }
          select.appendChild(option);
        });
        renderSelectedUpdateDetail();
        history.textContent = prettyJson(updateCenterState.history || []);
      }

      function renderSelectedUpdateDetail() {
        const detail = document.getElementById('updateSelectedDetail');
        const updateId = selectedUpdateId();
        const entry = (updateCenterState && Array.isArray(updateCenterState.available_updates))
          ? updateCenterState.available_updates.find((item) => item.update_id === updateId)
          : null;
        if (!entry) {
          detail.textContent = 'Select an available update to review its manifest, download state, verification state, and staged apply result.';
          return;
        }
        detail.textContent = prettyJson(entry);
      }

      async function refreshUpdateCenter() {
        try {
          const response = await fetch('/api/admin/updates/status', {headers: roleHeaders});
          updateCenterState = await response.json();
        } catch (error) {
          updateCenterState = {
            internet_update_check_enabled: false,
            pending_updates_count: 0,
            downloaded_updates_count: 0,
            verified_updates_count: 0,
            index_rebuild_needed: false,
            offline_mode: true,
            last_update_check_status: 'error',
            last_update_check_error: String(error),
            available_updates: [],
            history: [],
          };
        }
        renderUpdateCenter();
      }

      async function runUpdateAction(path, buttonId, body) {
        const button = document.getElementById(buttonId);
        if (button) {
          button.disabled = true;
        }
        document.getElementById('pulse').textContent = 'Update Center is working...';
        try {
          const response = await fetch(path, {
            method: 'POST',
            headers: {
              ...roleHeaders,
              'Content-Type': 'application/json'
            },
            body: body ? JSON.stringify(body) : undefined
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.detail || 'Update action failed.');
          }
          await refreshUpdateCenter();
          await refreshEvents();
          await refreshOverview();
          document.getElementById('pulse').textContent = 'Update Center action completed.';
          return payload;
        } catch (error) {
          document.getElementById('pulse').textContent = 'Update Center action failed.';
          document.getElementById('updateSelectedDetail').textContent = prettyJson({
            success: false,
            error: String(error),
          });
          return null;
        } finally {
          if (button) {
            button.disabled = false;
          }
        }
      }

      async function checkForUpdates() {
        await runUpdateAction('/api/admin/updates/check', 'updateCheckButton');
      }

      async function downloadSelectedUpdate() {
        const updateId = selectedUpdateId();
        if (!updateId) {
          document.getElementById('pulse').textContent = 'Select an update first.';
          return;
        }
        await runUpdateAction('/api/admin/updates/download', 'updateDownloadButton', {update_id: updateId});
      }

      async function verifySelectedUpdate() {
        const updateId = selectedUpdateId();
        if (!updateId) {
          document.getElementById('pulse').textContent = 'Select an update first.';
          return;
        }
        await runUpdateAction('/api/admin/updates/verify', 'updateVerifyButton', {update_id: updateId});
      }

      async function applySelectedUpdate() {
        const updateId = selectedUpdateId();
        if (!updateId) {
          document.getElementById('pulse').textContent = 'Select an update first.';
          return;
        }
        await runUpdateAction('/api/admin/updates/apply', 'updateApplyButton', {update_id: updateId});
      }

      async function rebuildIndexFromUpdates() {
        await runUpdateAction('/api/admin/updates/rebuild-index', 'updateRebuildButton');
      }

      async function refreshOverview() {
        const response = await fetch('/api/admin/overview', {headers: roleHeaders});
        const data = await response.json();
        document.getElementById('pulse').textContent = data.system_pulse.summary;
        document.getElementById('systemPulse').textContent = prettyJson(data.system_pulse);
        document.getElementById('overview').textContent = prettyJson(data.tasks);
        document.getElementById('runtimeTruth').textContent = prettyJson({
          model: data.model,
          retrieval: data.retrieval,
          requests: data.requests
        });
        renderList('activeTasks', data.tasks.active_tasks, (entry) =>
          `${entry.task_type} - ${entry.status} - ${entry.progress && entry.progress.phase ? entry.progress.phase : 'n/a'}`
        );
      }

      async function refreshEvents() {
        const response = await fetch('/api/admin/events?limit=20', {headers: roleHeaders});
        const data = await response.json();
        renderList('recentEvents', data.rows, (entry) =>
          `${entry.severity.toUpperCase()} - ${entry.event_type} - ${entry.summary}`
        );
        renderList('failures', data.rows.filter((entry) => ['warning', 'error', 'critical'].includes(entry.severity)), (entry) =>
          `${entry.severity.toUpperCase()} - ${entry.summary}`
        );
      }

      async function refreshMicroShadowStatus() {
        const fastTarget = document.getElementById('microFastStatus');
        const smartTarget = document.getElementById('microSmartStatus');
        const compareTarget = document.getElementById('microTierComparison');
        try {
          const response = await fetch('/api/admin/micro-llm/status', {headers: roleHeaders});
          const data = await response.json();
          fastTarget.textContent = prettyJson(data.fast_router || {});
          smartTarget.textContent = prettyJson(data.smart_shadow || {});
          compareTarget.textContent = prettyJson({
            latest_comparison: data.latest_comparison || {},
            benchmark_comparison: data.benchmark_comparison || {},
            advisory_only: data.advisory_only === true
          });
        } catch (error) {
          const fallback = prettyJson({
            success: false,
            last_error: String(error),
            advisory_only: true
          });
          fastTarget.textContent = fallback;
          smartTarget.textContent = fallback;
          compareTarget.textContent = fallback;
        }
      }

      async function runMicroShadowSmoke() {
        const button = document.getElementById('microShadowSmokeButton');
        button.disabled = true;
        document.getElementById('pulse').textContent = 'Running dual micro smoke test...';
        try {
          const response = await fetch('/api/admin/micro-llm/smoke', {
            method: 'POST',
            headers: roleHeaders
          });
          const data = await response.json();
          document.getElementById('pulse').textContent = data.success
            ? 'Dual micro smoke test succeeded.'
            : 'Dual micro smoke test completed with failures.';
          await refreshMicroShadowStatus();
          await refreshEvents();
        } catch (error) {
          document.getElementById('pulse').textContent = 'Dual micro smoke test request failed.';
          const fallback = prettyJson({
            success: false,
            last_error: String(error),
            advisory_only: true
          });
          document.getElementById('microFastStatus').textContent = fallback;
          document.getElementById('microSmartStatus').textContent = fallback;
          document.getElementById('microTierComparison').textContent = fallback;
        } finally {
          button.disabled = false;
        }
      }

      async function runMicroShadowBenchmark() {
        const button = document.getElementById('microShadowBenchmarkButton');
        button.disabled = true;
        document.getElementById('pulse').textContent = 'Running dual micro benchmark...';
        try {
          const response = await fetch('/api/admin/micro-llm/benchmark', {
            method: 'POST',
            headers: roleHeaders
          });
          const data = await response.json();
          document.getElementById('pulse').textContent = data.success
            ? 'Dual micro benchmark completed.'
            : 'Dual micro benchmark completed with failures.';
          await refreshMicroShadowStatus();
          await refreshEvents();
        } catch (error) {
          document.getElementById('pulse').textContent = 'Dual micro benchmark request failed.';
          const fallback = prettyJson({
            success: false,
            last_error: String(error),
            advisory_only: true
          });
          document.getElementById('microFastStatus').textContent = fallback;
          document.getElementById('microSmartStatus').textContent = fallback;
          document.getElementById('microTierComparison').textContent = fallback;
        } finally {
          button.disabled = false;
        }
      }

      function openPortableReadiness() {
        window.open('/api/admin/readiness/portable', '_blank', 'noopener');
      }

      async function startEventStream() {
        const response = await fetch('/api/admin/events/stream', {
          headers: roleHeaders
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const frames = buffer.split('\\n\\n');
          buffer = frames.pop();
          for (const frame of frames) {
            if (!frame.includes('data: ')) continue;
            const dataLine = frame.split('\\n').find((line) => line.startsWith('data: '));
            if (!dataLine) continue;
            try {
              const payload = JSON.parse(dataLine.slice(6));
              const events = document.getElementById('recentEvents');
              const item = document.createElement('li');
              item.textContent = `${payload.severity.toUpperCase()} - ${payload.event_type} - ${payload.summary}`;
              events.prepend(item);
              while (events.children.length > 8) {
                events.removeChild(events.lastChild);
              }
              if (['warning', 'error', 'critical'].includes(payload.severity)) {
                const failures = document.getElementById('failures');
                const warningItem = document.createElement('li');
                warningItem.textContent = `${payload.severity.toUpperCase()} - ${payload.summary}`;
                failures.prepend(warningItem);
                while (failures.children.length > 8) {
                  failures.removeChild(failures.lastChild);
                }
              }
              refreshOverview();
              if (payload.event_type === 'micro_shadow_smoke_test' || payload.event_type === 'micro_shadow_benchmark') {
                refreshMicroShadowStatus();
              }
              if (String(payload.event_type || '').startsWith('update_')) {
                refreshUpdateCenter();
              }
            } catch (_) {
            }
          }
        }
      }

      refreshOverview();
      refreshEvents();
      refreshMicroShadowStatus();
      refreshUpdateCenter();
      startEventStream();
    </script>
  </body>
</html>
"""

app = create_app()

