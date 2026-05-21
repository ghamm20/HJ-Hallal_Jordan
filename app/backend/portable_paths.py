"""Resolve legacy and USB-edition runtime paths from one portable profile."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from app.backend.runtime_config import RuntimeConfig


@dataclass(slots=True)
class RuntimePathProfile:
    root: Path
    portable_mode: bool
    raw_corpus_dir: Path
    normalized_corpus_dir: Path
    index_dir: Path
    updates_root: Path
    updates_inbox_dir: Path
    updates_processed_dir: Path
    updates_failed_dir: Path
    updates_quarantine_dir: Path
    updates_manifests_dir: Path
    runtime_dir: Path
    logs_dir: Path
    config_dir: Path
    micro_models_dir: Path
    main_models_dir: Path

    @property
    def config_path(self) -> Path:
        return self.config_dir / "runtime_config.json"

    @property
    def runtime_db_path(self) -> Path:
        return self.runtime_dir / "ops.db"

    @property
    def update_log_path(self) -> Path:
        return self.runtime_dir / "update_log.jsonl"

    @property
    def update_state_path(self) -> Path:
        return self.updates_manifests_dir / "update_state.json"

    def ensure_base_directories(self) -> None:
        for path in (
            self.raw_corpus_dir,
            self.normalized_corpus_dir,
            self.index_dir,
            self.updates_root,
            self.updates_inbox_dir,
            self.updates_processed_dir,
            self.updates_failed_dir,
            self.updates_quarantine_dir,
            self.updates_manifests_dir,
            self.runtime_dir,
            self.logs_dir,
            self.config_dir,
            self.micro_models_dir,
            self.main_models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def discover_config_path(
    repo_root: Path,
    *,
    runtime_dir_override: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    if runtime_dir_override is not None:
        return runtime_dir_override / "runtime_config.json"
    active_env = env or os.environ
    if _portable_mode_enabled(active_env):
        root = _portable_root(repo_root, active_env)
        return root / "config" / "runtime_config.json"
    return repo_root / "data" / "runtime" / "runtime_config.json"


def resolve_runtime_path_profile(
    repo_root: Path,
    *,
    config: RuntimeConfig,
    runtime_dir_override: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RuntimePathProfile:
    if runtime_dir_override is not None:
        runtime_root = runtime_dir_override.resolve()
        return RuntimePathProfile(
            root=repo_root.resolve(),
            portable_mode=False,
            raw_corpus_dir=(repo_root / "data" / "raw").resolve(),
            normalized_corpus_dir=(repo_root / "data" / "processed" / "normalized").resolve(),
            index_dir=(repo_root / "data" / "index").resolve(),
            updates_root=(runtime_root / "updates").resolve(),
            updates_inbox_dir=(runtime_root / "updates" / "inbox").resolve(),
            updates_processed_dir=(runtime_root / "updates" / "processed").resolve(),
            updates_failed_dir=(runtime_root / "updates" / "failed").resolve(),
            updates_quarantine_dir=(runtime_root / "updates" / "quarantine").resolve(),
            updates_manifests_dir=(runtime_root / "updates" / "manifests").resolve(),
            runtime_dir=runtime_root,
            logs_dir=runtime_root,
            config_dir=runtime_root,
            micro_models_dir=(repo_root / "models" / "micro").resolve(),
            main_models_dir=(repo_root / "models").resolve(),
        )

    active_env = env or os.environ
    portable_mode = bool(config.portable_mode or _portable_mode_enabled(active_env))
    if not portable_mode:
        root = repo_root.resolve()
        return RuntimePathProfile(
            root=root,
            portable_mode=False,
            raw_corpus_dir=(root / "data" / "raw").resolve(),
            normalized_corpus_dir=(root / "data" / "processed" / "normalized").resolve(),
            index_dir=(root / "data" / "index").resolve(),
            updates_root=(root / "updates").resolve(),
            updates_inbox_dir=(root / "updates" / "inbox").resolve(),
            updates_processed_dir=(root / "updates" / "processed").resolve(),
            updates_failed_dir=(root / "updates" / "failed").resolve(),
            updates_quarantine_dir=(root / "updates" / "quarantine").resolve(),
            updates_manifests_dir=(root / "updates" / "manifests").resolve(),
            runtime_dir=(root / "data" / "runtime").resolve(),
            logs_dir=(root / "data" / "runtime").resolve(),
            config_dir=(root / "data" / "runtime").resolve(),
            micro_models_dir=(root / "models" / "micro").resolve(),
            main_models_dir=(root / "models").resolve(),
        )

    root = _resolve_root(repo_root, config.portable_root, active_env)
    directories = config.portable_directories
    normalized_dir = _resolve_relative(root, directories.normalized_corpus)
    legacy_normalized_dir = (root / "data" / "processed" / "normalized").resolve()
    if not normalized_dir.exists() and legacy_normalized_dir.exists():
        normalized_dir = legacy_normalized_dir
    return RuntimePathProfile(
        root=root,
        portable_mode=True,
        raw_corpus_dir=_resolve_relative(root, directories.raw_corpus),
        normalized_corpus_dir=normalized_dir,
        index_dir=_resolve_relative(root, directories.index),
        updates_root=_resolve_relative(root, directories.updates),
        updates_inbox_dir=_resolve_relative(root, directories.updates) / "inbox",
        updates_processed_dir=_resolve_relative(root, directories.updates) / "processed",
        updates_failed_dir=_resolve_relative(root, directories.updates) / "failed",
        updates_quarantine_dir=_resolve_relative(root, directories.updates) / "quarantine",
        updates_manifests_dir=_resolve_relative(root, directories.updates) / "manifests",
        runtime_dir=_resolve_relative(root, directories.runtime),
        logs_dir=_resolve_relative(root, directories.logs),
        config_dir=_resolve_relative(root, directories.config),
        micro_models_dir=_resolve_relative(root, directories.models_micro),
        main_models_dir=_resolve_relative(root, directories.models_main),
    )


def _resolve_root(
    repo_root: Path,
    configured_root: str | None,
    env: Mapping[str, str],
) -> Path:
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return _portable_root(repo_root, env)


def _portable_root(repo_root: Path, env: Mapping[str, str]) -> Path:
    override = str(env.get("HALAL_JORDAN_PORTABLE_ROOT", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root.resolve()


def _portable_mode_enabled(env: Mapping[str, str]) -> bool:
    value = str(env.get("HALAL_JORDAN_PORTABLE_MODE", "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_relative(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()
