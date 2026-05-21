"""Helpers for bundled Ollama runtime and project-local model store inspection."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import urlopen


def bundled_ollama_dir(project_root: Path) -> Path:
    return project_root / "runtime" / "ollama"


def bundled_ollama_executable(project_root: Path) -> Path:
    return bundled_ollama_dir(project_root) / "ollama.exe"


def project_local_ollama_models_dir(project_root: Path) -> Path:
    return project_root / "models" / "ollama"


def production_ollama_model_name(config: Any) -> str:
    order = list(getattr(config, "model_preference_order", []) or [])
    for candidate in order:
        name = str(candidate or "").strip()
        if name:
            return name
    return ""


def ollama_manifest_relative_path(model_name: str) -> Path:
    normalized = str(model_name or "").strip()
    if not normalized:
        return Path()
    namespace, _, tag = normalized.partition(":")
    namespace = namespace.replace("/", os.sep).replace("\\", os.sep)
    tag = tag or "latest"
    return Path("manifests") / "registry.ollama.ai" / "library" / namespace / tag


def ollama_manifest_path(models_dir: Path, model_name: str) -> Path:
    relative = ollama_manifest_relative_path(model_name)
    if not relative.parts:
        return Path()
    return models_dir / relative


def ollama_model_present_in_store(models_dir: Path, model_name: str) -> bool:
    manifest_path = ollama_manifest_path(models_dir, model_name)
    if not manifest_path.exists() or not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    blobs = payload.get("layers") or []
    if not isinstance(blobs, list):
        return False
    for layer in blobs:
        if not isinstance(layer, dict):
            continue
        digest = str(layer.get("digest") or "").strip()
        if not digest:
            continue
        blob_name = digest.replace(":", "-")
        blob_path = models_dir / "blobs" / blob_name
        if not blob_path.exists() or not blob_path.is_file():
            return False
    return True


def inspect_ollama_bundle(project_root: Path, config: Any) -> dict[str, Any]:
    ollama_path = bundled_ollama_executable(project_root)
    models_dir = project_local_ollama_models_dir(project_root)
    production_model = production_ollama_model_name(config)
    project_model_present = bool(
        production_model and ollama_model_present_in_store(models_dir, production_model)
    )
    host_store = Path.home() / ".ollama" / "models"
    host_global_model_present = bool(
        production_model and ollama_model_present_in_store(host_store, production_model)
    )
    host_dependency = not (ollama_path.exists() and project_model_present)
    host_cache_required = not project_model_present and host_global_model_present
    return {
        "bundled_ollama_present": ollama_path.exists(),
        "bundled_ollama_path": str(ollama_path),
        "project_local_ollama_models_dir": str(models_dir),
        "production_model_name": production_model,
        "production_model_present_in_project_store": project_model_present,
        "host_global_model_present": host_global_model_present,
        "host_ollama_dependency": host_dependency,
        "host_ollama_model_cache_used": host_cache_required,
    }


def current_ollama_server_status(
    project_root: Path,
    *,
    ollama_url: str,
    bundled_ollama_path: str | None = None,
) -> dict[str, Any]:
    state_path = project_root / "runtime" / "halal-jordan-launch-state.json"
    started_by_launcher = False
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        started_by_launcher = bool(payload.get("ollama_started_by_launcher"))

    parsed = urlparse(str(ollama_url).strip())
    port = parsed.port or 11434
    listener_info = _listener_process_info_for_port(port)
    active_listener_path = str(listener_info.get("path") or "") if listener_info else ""
    bundled_listener = bool(
        bundled_ollama_path
        and active_listener_path
        and Path(active_listener_path).resolve() == Path(bundled_ollama_path).resolve()
    )
    host_listener_detected = bool(listener_info and active_listener_path and not bundled_listener)

    tags_url = str(ollama_url).rstrip("/") + "/api/tags"
    try:
        with urlopen(tags_url, timeout=5) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ollama_health_ok": False,
            "ollama_visible_models": [],
            "ollama_error": f"{type(exc).__name__}: {exc}",
            "ollama_server_started_by_launcher": started_by_launcher,
            "host_ollama_model_cache_used": False,
            "host_ollama_listener_detected": host_listener_detected,
            "active_listener_pid": listener_info.get("pid") if listener_info else None,
            "active_listener_path": active_listener_path or None,
            "bundled_listener_active": bundled_listener,
        }

    models = [
        str(item.get("name") or "").strip()
        for item in payload.get("models", [])
        if isinstance(item, dict)
    ]
    models = [item for item in models if item]
    return {
        "ollama_health_ok": True,
        "ollama_visible_models": models,
        "ollama_error": None,
        "ollama_server_started_by_launcher": started_by_launcher,
        "host_ollama_model_cache_used": False,
        "host_ollama_listener_detected": host_listener_detected,
        "active_listener_pid": listener_info.get("pid") if listener_info else None,
        "active_listener_path": active_listener_path or None,
        "bundled_listener_active": bundled_listener,
    }


def _listener_process_info_for_port(port: int) -> dict[str, Any] | None:
    command = [
        "powershell",
        "-NoLogo",
        "-NoProfile",
        "-Command",
        (
            "$conn = Get-NetTCPConnection -LocalPort "
            + str(port)
            + " -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;"
            + " if (-not $conn) { exit 0 };"
            + " $pidValue = [int]$conn.OwningProcess;"
            + " $proc = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $pidValue) -ErrorAction SilentlyContinue;"
            + " [pscustomobject]@{ pid=$pidValue; path=($proc.ExecutablePath) } | ConvertTo-Json -Compress"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = str(completed.stdout or "").strip()
    if not output:
        return None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
