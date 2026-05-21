"""Internal local-role permission enforcement for backend/admin routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException, Request

ROLE_ORDER = {
    "viewer": 0,
    "manager": 1,
    "admin": 2,
    "architect": 3,
    "owner": 4,
}

PERMISSION_MIN_ROLE = {
    "chat.ask": "manager",
    "admin.access": "admin",
    "config.inspect": "admin",
    "config.update": "architect",
    "permissions.inspect": "admin",
    "retrieval.debug": "admin",
    "retrieval.reload": "admin",
    "retrieval.reindex": "architect",
    "logs.inspect": "admin",
    "tasks.inspect": "admin",
}


@dataclass(slots=True)
class RoleContext:
    actor_id: str
    role: str
    request_id: str
    session_id: str


def normalize_role(role: str | None, *, default_role: str) -> str:
    candidate = str(role or "").strip().lower()
    if candidate in ROLE_ORDER:
        return candidate
    return default_role


def permissions_for_role(role: str) -> list[str]:
    normalized = normalize_role(role, default_role="viewer")
    rank = ROLE_ORDER[normalized]
    return sorted(
        permission
        for permission, min_role in PERMISSION_MIN_ROLE.items()
        if rank >= ROLE_ORDER[min_role]
    )


def has_permission(role: str, permission: str) -> bool:
    normalized = normalize_role(role, default_role="viewer")
    required = PERMISSION_MIN_ROLE.get(permission, "owner")
    return ROLE_ORDER[normalized] >= ROLE_ORDER[required]


def role_context_dependency(*, default_role: str):
    async def dependency(
        request: Request,
        x_hj_role: str | None = Header(default=None),
        x_hj_actor: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> RoleContext:
        context = build_role_context(
            request,
            default_role=default_role,
            requested_role=x_hj_role,
            actor_id=x_hj_actor,
            session_id=x_session_id,
        )
        request.state.role_context = context
        return context

    return dependency


def require_permission(permission: str, *, default_role: str = "viewer"):
    async def dependency(
        request: Request,
        x_hj_role: str | None = Header(default=None),
        x_hj_actor: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> RoleContext:
        context = build_role_context(
            request,
            default_role=default_role,
            requested_role=x_hj_role,
            actor_id=x_hj_actor,
            session_id=x_session_id,
        )
        request.state.role_context = context
        if has_permission(context.role, permission):
            return context
        ops = getattr(request.app.state, "ops", None)
        if ops is not None:
            ops.record_permission_denial(
                actor=context,
                route=request.url.path,
                permission=permission,
            )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "permission": permission,
                "role": context.role,
            },
        )

    return dependency


def build_role_context(
    request: Request,
    *,
    default_role: str,
    requested_role: str | None,
    actor_id: str | None,
    session_id: str | None,
) -> RoleContext:
    request_id = str(getattr(request.state, "request_id", "") or "unknown-request")
    normalized_role = normalize_role(requested_role, default_role=default_role)
    return RoleContext(
        actor_id=str(actor_id or f"local-{normalized_role}"),
        role=normalized_role,
        request_id=request_id,
        session_id=str(session_id or request_id),
    )


def permission_matrix() -> dict[str, Any]:
    return {
        "role_order": ROLE_ORDER,
        "permission_min_role": PERMISSION_MIN_ROLE,
        "permissions_by_role": {
            role: permissions_for_role(role)
            for role in ROLE_ORDER
        },
    }
