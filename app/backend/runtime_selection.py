"""Portable Python runtime selection helpers for Windows launchers and readiness."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def inspect_launcher_runtime(project_root: Path) -> dict[str, Any]:
    candidates = _runtime_candidates(project_root)
    selected = next((candidate for candidate in candidates if candidate["exists"]), None)

    bundled_python_paths = [
        candidate["executable"]
        for candidate in candidates
        if candidate["bundled"] and candidate["exists"]
    ]
    bundled_package_paths = [
        str(path)
        for path in _bundled_package_locations(project_root)
        if path.exists() and path.is_dir() and any(path.iterdir())
    ]

    return {
        "selected": selected,
        "host_fallback_used": bool(selected and selected["host_fallback"]),
        "bundled_python_present": bool(bundled_python_paths),
        "bundled_python_paths": bundled_python_paths,
        "bundled_packages_present": bool(bundled_package_paths),
        "bundled_package_paths": bundled_package_paths,
        "candidates": candidates,
    }


def _runtime_candidates(project_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        _candidate(
            project_root / "runtime" / "python" / "python.exe",
            label="bundled runtime python",
            source="bundled_runtime_python",
            project_local=True,
            bundled=True,
            host_fallback=False,
        ),
        _candidate(
            project_root / "runtime" / "venv" / "Scripts" / "python.exe",
            label="bundled runtime venv",
            source="bundled_runtime_venv",
            project_local=True,
            bundled=True,
            host_fallback=False,
        ),
        _candidate(
            project_root / "runtime" / "python311" / "python.exe",
            label="bundled runtime python311",
            source="bundled_runtime_python311",
            project_local=True,
            bundled=True,
            host_fallback=False,
        ),
        _candidate(
            project_root / ".venv" / "Scripts" / "python.exe",
            label="project-local .venv",
            source="project_local_dotvenv",
            project_local=True,
            bundled=False,
            host_fallback=False,
        ),
        _candidate(
            project_root / "venv" / "Scripts" / "python.exe",
            label="project-local venv",
            source="project_local_venv",
            project_local=True,
            bundled=False,
            host_fallback=False,
        ),
        _candidate(
            project_root / "python" / "python.exe",
            label="project-local python",
            source="project_local_python",
            project_local=True,
            bundled=False,
            host_fallback=False,
        ),
    ]

    host_python = shutil.which("python")
    if host_python:
        candidates.append(
            {
                "executable": host_python,
                "prefix": [],
                "label": "python on PATH",
                "source": "host_path_python",
                "project_local": False,
                "bundled": False,
                "host_fallback": True,
                "exists": True,
            }
        )

    py_launcher = shutil.which("py")
    if py_launcher:
        candidates.append(
            {
                "executable": py_launcher,
                "prefix": ["-3"],
                "label": "py launcher",
                "source": "host_py_launcher",
                "project_local": False,
                "bundled": False,
                "host_fallback": True,
                "exists": True,
            }
        )

    return candidates


def _bundled_package_locations(project_root: Path) -> list[Path]:
    return [
        project_root / "runtime" / "site-packages",
        project_root / "runtime" / "python" / "Lib" / "site-packages",
        project_root / "runtime" / "python311" / "Lib" / "site-packages",
        project_root / "runtime" / "venv" / "Lib" / "site-packages",
    ]


def _candidate(
    path: Path,
    *,
    label: str,
    source: str,
    project_local: bool,
    bundled: bool,
    host_fallback: bool,
) -> dict[str, Any]:
    return {
        "executable": str(path),
        "prefix": [],
        "label": label,
        "source": source,
        "project_local": project_local,
        "bundled": bundled,
        "host_fallback": host_fallback,
        "exists": path.exists(),
    }
