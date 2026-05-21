"""Portable local GGUF final-answer runtime backed by bundled llama.cpp."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from app.backend.model_profiles import resolve_model_path
from app.backend.portable_paths import RuntimePathProfile
from app.backend.runtime_config import RuntimeConfig
from app.reasoning.ollama_client import _parse_json_object

DEFAULT_MAIN_MODEL_N_CTX = 12288
DEFAULT_MAIN_MODEL_N_BATCH = 256
DEFAULT_MAIN_MODEL_TEMPERATURE = 0.3
DEFAULT_MAIN_MODEL_TOP_P = 0.9
DEFAULT_MAIN_MODEL_REPEAT_PENALTY = 1.05
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 11435
DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS = 60.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 180.0
WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class LocalGGUFMainModelRuntime:
    """Shared local llama.cpp server runtime for the portable final-answer model."""

    _lock = threading.RLock()
    _cached_process: subprocess.Popen[str] | None = None
    _cached_signature: tuple[Any, ...] | None = None
    _cached_backend_version: str | None = None

    @classmethod
    def status(
        cls,
        *,
        config: RuntimeConfig,
        path_profile: RuntimePathProfile,
    ) -> dict[str, Any]:
        model_path = resolve_model_path(
            root=path_profile.root,
            configured_path=config.main_model_path,
        )
        server_binary = bundled_llama_server_path(path_profile.root)
        server_status = local_gguf_server_runtime_status(
            config=config,
            path_profile=path_profile,
        )
        enabled = bool(config.main_model_enabled)
        provider = str(getattr(config, "main_model_provider", "local_gguf") or "local_gguf")
        configured = bool(config.main_model_path or config.main_model_name)
        exists = bool(model_path and model_path.exists())
        is_file = bool(model_path and model_path.is_file())
        extension = (
            model_path.suffix.lower()
            if model_path
            else Path(config.main_model_path or "").suffix.lower()
        )
        present = bool(exists and is_file and extension == ".gguf")
        backend_available = bool(server_binary.exists() and server_binary.is_file())
        ready = bool(enabled and provider == "local_gguf" and backend_available and present)

        error: str | None = None
        if not enabled:
            error = "main_model_disabled"
        elif provider != "local_gguf":
            error = "unsupported_main_model_provider"
        elif not configured:
            error = "main_model_path_not_configured"
        elif not exists:
            error = "configured_main_model_path_missing"
        elif not is_file:
            error = "configured_main_model_path_not_file"
        elif extension != ".gguf":
            error = "configured_main_model_path_not_gguf"
        elif not backend_available:
            error = "bundled_llama_server_missing"

        return {
            "enabled": enabled,
            "provider": provider,
            "configured": configured,
            "path": config.main_model_path or None,
            "resolved_path": str(model_path) if model_path else None,
            "backend_available": backend_available,
            "backend_name": "llama.cpp" if backend_available else None,
            "backend_version": cls._backend_version(path_profile.root) if backend_available else None,
            "present": present,
            "loaded": bool(server_status["server_health_ok"]),
            "ready": ready,
            "error": error,
            "selected_for_production": ready,
            "fallback_active": not ready,
            "model_name": _display_name(config=config, model_path=model_path),
            "server_url": server_status["server_url"],
            "server_binary_path": str(server_binary),
            "server_health_ok": server_status["server_health_ok"],
            "server_error": server_status["server_error"],
            "server_visible_models": list(server_status["server_visible_models"]),
            "server_state_path": server_status["server_state_path"],
        }

    @classmethod
    def chat_json(
        cls,
        *,
        config: RuntimeConfig,
        path_profile: RuntimePathProfile,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status = cls.status(config=config, path_profile=path_profile)
        if not status["ready"]:
            raise RuntimeError(str(status["error"] or "local_gguf_main_model_unavailable"))

        model_path = resolve_model_path(
            root=path_profile.root,
            configured_path=config.main_model_path,
        )
        if model_path is None:
            raise RuntimeError("main_model_path_not_configured")

        cls.ensure_server_started(config=config, path_profile=path_profile)
        request_options = dict(options or {})
        response = requests.post(
            server_chat_completions_url(path_profile.root),
            json={
                "model": cls.resolve_model_name(
                    config=config,
                    path_profile=path_profile,
                ),
                "messages": messages,
                "max_tokens": int(request_options.get("num_predict", 128) or 128),
                "temperature": float(
                    request_options.get("temperature", DEFAULT_MAIN_MODEL_TEMPERATURE)
                ),
                "top_p": float(request_options.get("top_p", DEFAULT_MAIN_MODEL_TOP_P)),
                "repeat_penalty": float(
                    request_options.get(
                        "repeat_penalty",
                        DEFAULT_MAIN_MODEL_REPEAT_PENALTY,
                    )
                ),
                "response_format": {
                    "type": "json_object",
                    "schema": schema,
                },
                "stream": False,
            },
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        content = str(payload["choices"][0]["message"]["content"])
        return _parse_json_object(content)

    @classmethod
    def ensure_server_started(
        cls,
        *,
        config: RuntimeConfig,
        path_profile: RuntimePathProfile,
    ) -> None:
        model_path = resolve_model_path(
            root=path_profile.root,
            configured_path=config.main_model_path,
        )
        if model_path is None:
            raise RuntimeError("main_model_path_not_configured")
        signature = cls._model_signature(model_path=model_path)
        expected_model_name = cls.resolve_model_name(
            config=config,
            path_profile=path_profile,
        )
        server_url = local_gguf_server_url(path_profile.root)

        current_probe = _probe_server(
            server_url=server_url,
            expected_model_name=expected_model_name,
        )
        if current_probe["server_health_ok"]:
            with cls._lock:
                cls._cached_signature = signature
            return

        with cls._lock:
            current_probe = _probe_server(
                server_url=server_url,
                expected_model_name=expected_model_name,
            )
            if current_probe["server_health_ok"]:
                cls._cached_signature = signature
                return

            if cls._cached_process is not None and cls._cached_process.poll() is None:
                _wait_for_server_ready(
                    server_url=server_url,
                    expected_model_name=expected_model_name,
                    process=cls._cached_process,
                    timeout_seconds=_server_startup_timeout_seconds(),
                    stderr_path=llama_server_error_log_path(path_profile.root),
                )
                cls._cached_signature = signature
                return

            server_binary = bundled_llama_server_path(path_profile.root)
            if not server_binary.exists():
                raise RuntimeError("bundled_llama_server_missing")
            log_dir = path_profile.logs_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            state_path = llama_server_state_path(path_profile.root)
            command = [
                str(server_binary),
                "-m",
                str(model_path),
                "--alias",
                expected_model_name,
                "--host",
                local_gguf_server_host(path_profile.root),
                "--port",
                str(local_gguf_server_port(path_profile.root)),
                "-c",
                str(DEFAULT_MAIN_MODEL_N_CTX),
                "-t",
                str(_default_threads()),
                "-b",
                str(DEFAULT_MAIN_MODEL_N_BATCH),
                "-ub",
                str(DEFAULT_MAIN_MODEL_N_BATCH),
                "--reasoning",
                "off",
                "--no-webui",
                "--jinja",
                "--offline",
            ]
            env = os.environ.copy()
            env["LLAMA_OFFLINE"] = "1"
            env.setdefault("LLAMA_LOG_VERBOSITY", "1")
            stdout_handle = llama_server_output_log_path(path_profile.root).open(
                "a",
                encoding="utf-8",
            )
            stderr_handle = llama_server_error_log_path(path_profile.root).open(
                "a",
                encoding="utf-8",
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(server_binary.parent),
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    creationflags=WINDOWS_NO_WINDOW,
                )
            finally:
                stdout_handle.close()
                stderr_handle.close()
            cls._cached_process = process
            cls._cached_signature = signature
            _write_server_state(
                state_path=state_path,
                payload={
                    "pid": process.pid,
                    "server_url": server_url,
                    "model_name": expected_model_name,
                    "model_path": str(model_path),
                    "server_binary_path": str(server_binary),
                    "started_at": datetime.now(UTC).isoformat(),
                    "started_by": "local_gguf_runtime",
                },
            )
            _wait_for_server_ready(
                server_url=server_url,
                expected_model_name=expected_model_name,
                process=process,
                timeout_seconds=_server_startup_timeout_seconds(),
                stderr_path=llama_server_error_log_path(path_profile.root),
            )

    @classmethod
    def resolve_model_name(
        cls,
        *,
        config: RuntimeConfig,
        path_profile: RuntimePathProfile,
    ) -> str:
        model_path = resolve_model_path(
            root=path_profile.root,
            configured_path=config.main_model_path,
        )
        return _display_name(config=config, model_path=model_path)

    @classmethod
    def reset_cache(cls) -> None:
        with cls._lock:
            if cls._cached_process is not None and cls._cached_process.poll() is None:
                try:
                    cls._cached_process.terminate()
                    cls._cached_process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        cls._cached_process.kill()
                    except OSError:
                        pass
            cls._cached_process = None
            cls._cached_signature = None
            cls._cached_backend_version = None

    @classmethod
    def _model_signature(cls, *, model_path: Path) -> tuple[Any, ...]:
        return (
            str(model_path.resolve()),
            DEFAULT_MAIN_MODEL_N_CTX,
            _default_threads(),
            DEFAULT_MAIN_MODEL_N_BATCH,
        )

    @classmethod
    def _backend_version(cls, root: Path) -> str | None:
        with cls._lock:
            if cls._cached_backend_version is not None:
                return cls._cached_backend_version
            server_binary = bundled_llama_server_path(root)
            if not server_binary.exists():
                return None
            try:
                completed = subprocess.run(
                    [str(server_binary), "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                    creationflags=WINDOWS_NO_WINDOW,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            output = str(completed.stdout or completed.stderr or "").strip()
            cls._cached_backend_version = output.splitlines()[0].strip() if output else None
            return cls._cached_backend_version


def local_gguf_server_runtime_status(
    *,
    config: RuntimeConfig,
    path_profile: RuntimePathProfile,
) -> dict[str, Any]:
    expected_model_name = LocalGGUFMainModelRuntime.resolve_model_name(
        config=config,
        path_profile=path_profile,
    )
    server_url = local_gguf_server_url(path_profile.root)
    probe = _probe_server(
        server_url=server_url,
        expected_model_name=expected_model_name,
    )
    state_path = llama_server_state_path(path_profile.root)
    state_payload = _read_server_state(state_path)
    return {
        "server_binary_path": str(bundled_llama_server_path(path_profile.root)),
        "server_binary_present": bundled_llama_server_path(path_profile.root).exists(),
        "server_url": server_url,
        "server_health_ok": probe["server_health_ok"],
        "server_error": probe["server_error"],
        "server_visible_models": list(probe["server_visible_models"]),
        "server_state_path": str(state_path),
        "server_state": state_payload,
    }


def bundled_llama_runtime_dir(root: Path) -> Path:
    return root / "runtime" / "llama"


def bundled_llama_server_path(root: Path) -> Path:
    return bundled_llama_runtime_dir(root) / "llama-server.exe"


def bundled_llama_cli_path(root: Path) -> Path:
    return bundled_llama_runtime_dir(root) / "llama-cli.exe"


def local_gguf_server_host(root: Path) -> str:
    return str(
        os.getenv("HALAL_JORDAN_LLAMACPP_HOST")
        or os.getenv("HALAL_JORDAN_LLAMA_HOST")
        or DEFAULT_SERVER_HOST
    ).strip()


def local_gguf_server_port(root: Path) -> int:
    configured = str(
        os.getenv("HALAL_JORDAN_LLAMACPP_PORT")
        or os.getenv("HALAL_JORDAN_LLAMA_PORT")
        or DEFAULT_SERVER_PORT
    ).strip()
    try:
        parsed = int(configured)
    except ValueError:
        parsed = DEFAULT_SERVER_PORT
    return parsed if parsed > 0 else DEFAULT_SERVER_PORT


def local_gguf_server_url(root: Path) -> str:
    configured = str(
        os.getenv("HALAL_JORDAN_LLAMACPP_URL")
        or os.getenv("HALAL_JORDAN_LLAMA_URL")
        or ""
    ).strip()
    if configured:
        return configured.rstrip("/")
    return f"http://{local_gguf_server_host(root)}:{local_gguf_server_port(root)}"


def server_chat_completions_url(root: Path) -> str:
    return local_gguf_server_url(root) + "/v1/chat/completions"


def llama_server_state_path(root: Path) -> Path:
    return root / "runtime" / "halal-jordan-llama-state.json"


def llama_server_output_log_path(root: Path) -> Path:
    return root / "logs" / "halal-jordan-llama.out.log"


def llama_server_error_log_path(root: Path) -> Path:
    return root / "logs" / "halal-jordan-llama.err.log"


def _default_threads() -> int:
    cpu_total = os.cpu_count() or 4
    return max(1, min(8, cpu_total))


def _display_name(*, config: RuntimeConfig, model_path: Path | None) -> str:
    configured_name = str(getattr(config, "main_model_name", "") or "").strip()
    if configured_name:
        return configured_name
    if model_path is not None:
        return model_path.stem
    return "local-main-gguf"


def _wait_for_server_ready(
    *,
    server_url: str,
    expected_model_name: str,
    process: subprocess.Popen[str],
    timeout_seconds: float,
    stderr_path: Path,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        probe = _probe_server(
            server_url=server_url,
            expected_model_name=expected_model_name,
        )
        if probe["server_health_ok"]:
            return
        if process.poll() is not None:
            raise RuntimeError(
                "bundled_llama_server_exited: "
                + _last_log_line(stderr_path, default="unknown_error")
            )
        time.sleep(1.0)
    raise RuntimeError(
        "bundled_llama_server_startup_timeout: "
        + _last_log_line(stderr_path, default="server_not_ready")
    )


def _probe_server(
    *,
    server_url: str,
    expected_model_name: str,
) -> dict[str, Any]:
    try:
        response = requests.get(server_url + "/v1/models", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "server_health_ok": False,
            "server_error": f"{type(exc).__name__}: {exc}",
            "server_visible_models": [],
        }
    models = [
        str(item.get("id") or "").strip()
        for item in payload.get("data", [])
        if isinstance(item, dict)
    ]
    models = [item for item in models if item]
    if expected_model_name and models and expected_model_name not in models:
        return {
            "server_health_ok": False,
            "server_error": "unexpected_loaded_model",
            "server_visible_models": models,
        }
    return {
        "server_health_ok": bool(models),
        "server_error": None if models else "no_models_visible",
        "server_visible_models": models,
    }


def _server_startup_timeout_seconds() -> float:
    try:
        configured = float(
            str(
                os.getenv("HALAL_JORDAN_LLAMACPP_STARTUP_TIMEOUT_SECONDS")
                or DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS
            ).strip()
        )
    except ValueError:
        configured = DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS
    return max(configured, 5.0)


def _read_server_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_server_state(*, state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _last_log_line(path: Path, *, default: str) -> str:
    if not path.exists():
        return default
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return default
    for line in reversed(lines):
        candidate = line.strip()
        if candidate:
            return candidate
    return default
