"""Helpers for portable model profile resolution and verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.backend.portable_paths import RuntimePathProfile

LOCAL_MODEL_EXTENSIONS = {".gguf", ".bin", ".onnx"}
MICRO_MODEL_MIN_BYTES = 1_000_000_000


def build_model_profile_status(
    config: Any,
    path_profile: RuntimePathProfile,
    *,
    ollama_visible_models: list[str] | None = None,
    availability_error: str | None = None,
) -> dict[str, Any]:
    visible_models = list(ollama_visible_models or [])
    micro_smart = build_single_model_profile_status(
        enabled=bool(
            getattr(config, "micro_smart_enabled", getattr(config, "micro_model_enabled", False))
        ),
        provider=str(
            getattr(config, "micro_smart_provider", getattr(config, "micro_model_provider", "local_gguf"))
            or "local_gguf"
        ),
        name=str(
            getattr(config, "micro_smart_name", getattr(config, "micro_model_name", "")) or ""
        ).strip(),
        configured_path=str(
            getattr(config, "micro_smart_path", getattr(config, "micro_model_path", "")) or ""
        ).strip(),
        role=str(
            getattr(config, "micro_smart_role", getattr(config, "micro_model_role", "shadow"))
            or "shadow"
        ),
        root=path_profile.root,
        expected_models_dir=path_profile.micro_models_dir,
        visible_models=visible_models,
        minimum_bytes=MICRO_MODEL_MIN_BYTES,
    )
    micro_fast = build_rule_router_profile_status(
        enabled=bool(getattr(config, "micro_fast_enabled", True)),
        provider=str(getattr(config, "micro_fast_provider", "rules") or "rules"),
        name=str(getattr(config, "micro_fast_name", "hj-fast-router-rules") or "").strip(),
    )
    return {
        "micro": dict(micro_smart),
        "micro_fast": micro_fast,
        "micro_smart": micro_smart,
        "main": build_single_model_profile_status(
            enabled=bool(getattr(config, "main_model_enabled", False)),
            provider=str(
                getattr(config, "main_model_provider", "")
                or _infer_model_provider(
                configured_path=str(getattr(config, "main_model_path", "") or "").strip(),
                configured_name=str(getattr(config, "main_model_name", "") or "").strip(),
                )
            ),
            name=str(getattr(config, "main_model_name", "") or "").strip(),
            configured_path=str(getattr(config, "main_model_path", "") or "").strip(),
            role=str(getattr(config, "model_role", "") or "final_synthesis"),
            root=path_profile.root,
            expected_models_dir=path_profile.main_models_dir,
            visible_models=visible_models,
            minimum_bytes=MICRO_MODEL_MIN_BYTES,
        ),
        "ollama_visible_models": visible_models,
        "availability_error": availability_error,
    }


def build_rule_router_profile_status(
    *,
    enabled: bool,
    provider: str,
    name: str,
) -> dict[str, Any]:
    available = enabled and provider == "rules"
    error: str | None = None
    if not enabled:
        error = "fast_router_disabled"
    elif provider != "rules":
        error = "fast_router_provider_unavailable"
    return {
        "enabled": enabled,
        "configured": enabled,
        "provider": provider,
        "name": name,
        "role": "assist",
        "path": None,
        "available": available,
        "availability_source": "deterministic_rules",
        "resolved_path": None,
        "project_local": True,
        "within_expected_directory": True,
        "exists": False,
        "is_file": False,
        "extension": None,
        "is_gguf": False,
        "size_bytes": 0,
        "size_over_1gb": False,
        "error": error,
    }


def build_single_model_profile_status(
    *,
    enabled: bool,
    provider: str,
    name: str,
    configured_path: str,
    role: str,
    root: Path,
    expected_models_dir: Path,
    visible_models: list[str],
    minimum_bytes: int,
) -> dict[str, Any]:
    configured = bool(configured_path or name)
    relative_path = str(configured_path or "").strip()
    resolved_path = resolve_model_path(root=root, configured_path=relative_path)
    exists = bool(resolved_path and resolved_path.exists())
    is_file = bool(resolved_path and resolved_path.is_file())
    extension = resolved_path.suffix.lower() if resolved_path else Path(relative_path).suffix.lower()
    size_bytes = resolved_path.stat().st_size if exists and is_file else 0
    project_local = bool(resolved_path and _is_relative_to(resolved_path, root))
    expected_dir_match = bool(
        resolved_path and _is_relative_to(resolved_path, expected_models_dir)
    )
    size_over_minimum = size_bytes > minimum_bytes if size_bytes else False
    available = False
    availability_source = "disabled"
    error: str | None = None

    if enabled and provider == "local_gguf":
        availability_source = "local_gguf_path"
        available = bool(
            configured_path
            and exists
            and is_file
            and extension == ".gguf"
            and project_local
            and expected_dir_match
            and size_over_minimum
        )
        if not configured_path:
            error = "model_path_not_configured"
        elif not exists:
            error = "configured_model_path_missing"
        elif not is_file:
            error = "configured_model_path_not_file"
        elif extension != ".gguf":
            error = "configured_model_path_not_gguf"
        elif not project_local:
            error = "configured_model_path_not_project_local"
        elif not expected_dir_match:
            error = "configured_model_path_outside_models_directory"
        elif not size_over_minimum:
            error = "configured_model_file_too_small"
    elif enabled and provider == "ollama_name":
        availability_source = "ollama_name"
        available = bool(name and name in visible_models)
        if not name:
            error = "model_name_not_configured"
        elif not available:
            error = "configured_ollama_model_missing"
    elif enabled:
        availability_source = "unknown_provider"
        error = "unsupported_model_provider"

    return {
        "enabled": enabled,
        "configured": configured,
        "provider": provider,
        "name": name,
        "role": role,
        "path": relative_path or None,
        "available": available,
        "availability_source": availability_source,
        "resolved_path": str(resolved_path) if resolved_path else None,
        "project_local": project_local,
        "within_expected_directory": expected_dir_match,
        "exists": exists,
        "is_file": is_file,
        "extension": extension or None,
        "is_gguf": extension == ".gguf",
        "size_bytes": size_bytes,
        "size_over_1gb": size_over_minimum,
        "backend_available": None,
        "backend_name": None,
        "backend_version": None,
        "loaded": False,
        "fallback_active": None,
        "selected_for_production": False,
        "error": error,
    }


def resolve_model_path(*, root: Path, configured_path: str) -> Path | None:
    candidate_text = str(configured_path or "").strip()
    if not candidate_text:
        return None
    candidate = Path(candidate_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _infer_model_provider(*, configured_path: str, configured_name: str) -> str:
    if configured_path:
        return "local_gguf"
    if configured_name:
        return "ollama_name"
    return "unconfigured"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
