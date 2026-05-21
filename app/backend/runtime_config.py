"""Persistent runtime configuration with validation and effective/default views."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AdminFeatureToggles(BaseModel):
    enable_reload: bool = True
    enable_reindex: bool = True
    enable_config_updates: bool = True
    enable_log_queries: bool = True
    enable_retrieval_debug: bool = True
    enable_permission_inspection: bool = True


class ReindexBehaviorFlags(BaseModel):
    force_pdf_normalization: bool = True
    preserve_previous_assets_on_failure: bool = True
    enable_ocr_recovery: bool = False


class PortableDirectories(BaseModel):
    models_micro: str = "models/micro"
    models_main: str = "models"
    raw_corpus: str = "data/raw"
    normalized_corpus: str = "data/normalized"
    index: str = "data/index"
    updates: str = "updates"
    runtime: str = "runtime"
    logs: str = "logs"
    config: str = "config"


class RuntimeConfig(BaseModel):
    runtime_identity: str = "app.backend.main:app"
    compatibility_aliases: list[str] = Field(
        default_factory=lambda: ["app.backend.api:app", "main:app", "api:app"]
    )
    portable_mode: bool = Field(
        default_factory=lambda: _env_flag("HALAL_JORDAN_PORTABLE_MODE")
    )
    portable_root: str | None = Field(
        default_factory=lambda: os.getenv("HALAL_JORDAN_PORTABLE_ROOT") or None
    )
    portable_directories: PortableDirectories = Field(
        default_factory=PortableDirectories
    )
    ollama_url: str = Field(
        default_factory=lambda: _normalize_ollama_url(
            os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        )
    )
    update_manifest_url: str = Field(
        default_factory=lambda: str(os.getenv("HALAL_JORDAN_UPDATE_MANIFEST_URL", "") or "").strip()
    )
    update_check_enabled: bool = True
    update_check_on_startup: bool = True
    update_check_timeout_seconds: int = Field(default=3, ge=1, le=30)
    model_preference_order: list[str] = Field(
        default_factory=lambda: _default_model_order()
    )
    micro_fast_enabled: bool = True
    micro_fast_provider: Literal["rules", "local_gguf"] = "rules"
    micro_fast_name: str = "hj-fast-router-rules"
    micro_fast_assist_enabled: bool = True
    micro_fast_assist_confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    research_depth_default: Literal[
        "quick_source_check",
        "balanced_research",
        "deep_research",
    ] = "balanced_research"
    fast_path_enabled: bool = True
    fast_path_simple_top_k: int = Field(default=3, ge=1, le=10)
    fast_path_moderate_top_k: int = Field(default=5, ge=1, le=10)
    fast_path_simple_candidate_limit: int = Field(default=40, ge=5, le=500)
    fast_path_moderate_candidate_limit: int = Field(default=80, ge=5, le=500)
    fast_path_simple_max_tokens: int = Field(default=160, ge=32, le=512)
    fast_path_moderate_max_tokens: int = Field(default=240, ge=32, le=768)
    quick_research_simple_top_k: int = Field(default=2, ge=1, le=8)
    quick_research_moderate_top_k: int = Field(default=3, ge=1, le=8)
    quick_research_simple_candidate_limit: int = Field(default=24, ge=5, le=200)
    quick_research_moderate_candidate_limit: int = Field(default=48, ge=5, le=300)
    quick_research_simple_max_tokens: int = Field(default=120, ge=32, le=384)
    quick_research_moderate_max_tokens: int = Field(default=180, ge=32, le=512)
    retrieval_timeout_quick_ms: int = Field(default=8000, ge=1000, le=120000)
    retrieval_timeout_balanced_ms: int = Field(default=15000, ge=1000, le=120000)
    retrieval_timeout_deep_ms: int = Field(default=30000, ge=1000, le=180000)
    retrieval_warmup_timeout_ms: int = Field(default=45000, ge=1000, le=300000)
    retrieval_eager_warmup_enabled: bool = True
    fast_path_source_only_max_sources: int = Field(default=3, ge=1, le=10)
    fast_path_max_layer_count: int = Field(default=2, ge=1, le=6)
    deep_research_top_k: int = Field(default=8, ge=3, le=20)
    deep_research_candidate_limit: int = Field(default=900, ge=50, le=5000)
    deep_research_max_tokens: int = Field(default=420, ge=128, le=1200)
    micro_model_enabled: bool = True
    micro_model_provider: Literal["local_gguf", "ollama_name"] = "local_gguf"
    micro_model_path: str = "models/micro/qwen3-4b/Qwen3-4B-Q4_K_M.gguf"
    micro_model_name: str = "qwen3-4b-instruct-q4_k_m"
    micro_model_role: Literal["shadow", "assist", "final_synthesis"] = "shadow"
    micro_smart_enabled: bool = True
    micro_smart_provider: Literal["local_gguf", "ollama_name"] = "local_gguf"
    micro_smart_path: str = "models/micro/qwen3-4b/Qwen3-4B-Q4_K_M.gguf"
    micro_smart_name: str = "qwen3-4b-instruct-q4_k_m"
    micro_smart_role: Literal["shadow", "assist", "final_synthesis"] = "shadow"
    micro_smart_inline_enabled: bool = False
    micro_smart_admin_only: bool = True
    micro_shadow_n_ctx: int = Field(default=384, ge=128, le=40960)
    micro_shadow_n_threads: int = Field(
        default_factory=lambda: _default_micro_shadow_threads(),
        ge=1,
        le=64,
    )
    micro_shadow_n_batch: int = Field(default=256, ge=8, le=4096)
    micro_shadow_max_tokens: int = Field(default=56, ge=8, le=512)
    micro_shadow_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    micro_shadow_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    micro_shadow_repeat_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    micro_shadow_use_mlock: bool = False
    micro_shadow_use_mmap: bool = True
    micro_shadow_warmup_on_startup: bool = False
    micro_shadow_warmup_before_smoke: bool = True
    main_model_enabled: bool = False
    main_model_provider: Literal["local_gguf", "ollama_name"] = "local_gguf"
    main_model_path: str = ""
    main_model_name: str = ""
    model_role: Literal["shadow", "assist", "final_synthesis"] = "final_synthesis"
    launcher_model_ready_timeout_seconds: int = Field(default=120, ge=10, le=600)
    laptop_build: bool = False
    expected_ram_class: str = ""
    embedding_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    vector_index_path: str = "data/index/vector_index.npz"
    hybrid_weight_keyword: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_weight_semantic: float = Field(default=0.5, ge=0.0, le=1.0)
    tesseract_path: str = ""
    max_retrieval_candidates: int = Field(default=500, ge=10, le=5000)
    rerank_limit: int = Field(default=5, ge=1, le=50)
    citation_excerpt_length: int = Field(default=180, ge=60, le=500)
    location_tool_max_comparison_sites: int = Field(default=5, ge=2, le=12)
    session_continuity_ttl_minutes: int = Field(default=60, ge=1, le=720)
    admin_feature_toggles: AdminFeatureToggles = Field(
        default_factory=AdminFeatureToggles
    )
    log_retention_days: int = Field(default=30, ge=1, le=3650)
    reindex_behavior_flags: ReindexBehaviorFlags = Field(
        default_factory=ReindexBehaviorFlags
    )
    permission_mode: Literal["local_header_roles"] = "local_header_roles"

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_model_field_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        legacy_micro = str(payload.pop("micro_model_path_or_name", "") or "").strip()
        legacy_main = str(payload.pop("main_model_path_or_name", "") or "").strip()
        if legacy_micro:
            if _looks_like_model_path(legacy_micro):
                payload.setdefault("micro_model_path", legacy_micro)
            else:
                payload.setdefault("micro_model_name", legacy_micro)
                payload.setdefault("micro_model_provider", "ollama_name")
        if legacy_main:
            if _looks_like_model_path(legacy_main):
                payload.setdefault("main_model_path", legacy_main)
                payload.setdefault("main_model_provider", "local_gguf")
            else:
                payload.setdefault("main_model_name", legacy_main)
                payload.setdefault("main_model_provider", "ollama_name")
        if "micro_smart_enabled" not in payload and "micro_model_enabled" in payload:
            payload["micro_smart_enabled"] = payload["micro_model_enabled"]
        if "micro_smart_provider" not in payload and "micro_model_provider" in payload:
            payload["micro_smart_provider"] = payload["micro_model_provider"]
        if "micro_smart_path" not in payload and "micro_model_path" in payload:
            payload["micro_smart_path"] = payload["micro_model_path"]
        if "micro_smart_name" not in payload and "micro_model_name" in payload:
            payload["micro_smart_name"] = payload["micro_model_name"]
        if "micro_smart_role" not in payload and "micro_model_role" in payload:
            payload["micro_smart_role"] = payload["micro_model_role"]
        if "micro_model_enabled" not in payload and "micro_smart_enabled" in payload:
            payload["micro_model_enabled"] = payload["micro_smart_enabled"]
        if "micro_model_provider" not in payload and "micro_smart_provider" in payload:
            payload["micro_model_provider"] = payload["micro_smart_provider"]
        if "micro_model_path" not in payload and "micro_smart_path" in payload:
            payload["micro_model_path"] = payload["micro_smart_path"]
        if "micro_model_name" not in payload and "micro_smart_name" in payload:
            payload["micro_model_name"] = payload["micro_smart_name"]
        if "micro_model_role" not in payload and "micro_smart_role" in payload:
            payload["micro_model_role"] = payload["micro_smart_role"]
        return payload

    @model_validator(mode="after")
    def _validate_portable_model_paths(self) -> RuntimeConfig:
        self.ollama_url = _normalize_ollama_url(self.ollama_url)
        if self.portable_mode:
            for field_name in ("micro_model_path", "micro_smart_path", "main_model_path"):
                configured_path = str(getattr(self, field_name, "") or "").strip()
                if configured_path and Path(configured_path).expanduser().is_absolute():
                    raise ValueError(
                        f"{field_name} must be project-relative when portable_mode is enabled"
                    )
        self.micro_model_enabled = self.micro_smart_enabled
        self.micro_model_provider = self.micro_smart_provider
        self.micro_model_path = self.micro_smart_path
        self.micro_model_name = self.micro_smart_name
        self.micro_model_role = self.micro_smart_role
        return self

    @property
    def micro_model_target(self) -> str:
        return str(self.micro_model_path or self.micro_model_name or "").strip()

    @property
    def micro_smart_target(self) -> str:
        return str(self.micro_smart_path or self.micro_smart_name or "").strip()

    @property
    def main_model_target(self) -> str:
        return str(self.main_model_path or self.main_model_name or "").strip()


@dataclass(slots=True)
class ConfigUpdateResult:
    configured: dict[str, Any]
    defaults: dict[str, Any]
    effective: dict[str, Any]
    changed_fields: list[str]


class RuntimeConfigStore:
    """Load, validate, persist, and describe runtime config overrides."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def defaults_model(self) -> RuntimeConfig:
        return RuntimeConfig()

    def ensure_file_exists(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{}\n", encoding="utf-8")

    def load_overrides(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("runtime config file must contain an object")
        return payload

    def load_effective(self) -> RuntimeConfig:
        defaults = self.defaults_model().model_dump()
        merged = _deep_merge(defaults, self.load_overrides())
        return RuntimeConfig.model_validate(merged)

    def describe(self) -> dict[str, Any]:
        defaults = self.defaults_model().model_dump()
        configured = self.load_overrides()
        effective = self.load_effective().model_dump()
        return {
            "configured": configured,
            "defaults": defaults,
            "effective": effective,
        }

    def update(self, patch: dict[str, Any]) -> ConfigUpdateResult:
        if not isinstance(patch, dict):
            raise TypeError("config update patch must be an object")
        defaults = self.defaults_model().model_dump()
        current_overrides = self.load_overrides()
        current_effective = _deep_merge(defaults, current_overrides)
        merged_effective = _deep_merge(current_effective, patch)
        validated = RuntimeConfig.model_validate(merged_effective)
        new_effective = validated.model_dump()
        new_overrides = _diff_from_defaults(defaults, new_effective)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(new_overrides, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        changed_fields = sorted(
            set(_flatten_paths(current_effective)) | set(_flatten_paths(new_effective))
        )
        changed_fields = [
            field
            for field in changed_fields
            if _value_for_path(current_effective, field) != _value_for_path(new_effective, field)
        ]
        return ConfigUpdateResult(
            configured=new_overrides,
            defaults=defaults,
            effective=new_effective,
            changed_fields=changed_fields,
        )


def _default_model_order() -> list[str]:
    configured = os.getenv("HALAL_JORDAN_OLLAMA_MODEL")
    ordered = [configured] if configured else []
    ordered.extend(["qwen3:8b", "deepseek-r1:8b", "qwen2.5-coder:7b"])
    deduped: list[str] = []
    for item in ordered:
        if not item or item in deduped:
            continue
        deduped.append(item)
    return deduped


def _default_micro_shadow_threads() -> int:
    cpu_total = os.cpu_count() or 4
    return max(1, min(6, cpu_total))


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _looks_like_model_path(value: str) -> bool:
    candidate = Path(str(value or "").strip()).expanduser()
    return bool(
        candidate.is_absolute()
        or any(separator in str(value) for separator in ("\\", "/"))
        or candidate.suffix.lower() in {".gguf", ".bin", ".onnx"}
    )


def _normalize_ollama_url(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return "http://127.0.0.1:11434"
    if "://" not in normalized:
        return "http://" + normalized
    return normalized


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(dict(base[key]), value)
        else:
            merged[key] = value
    return merged


def _diff_from_defaults(defaults: dict[str, Any], effective: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, value in effective.items():
        default_value = defaults.get(key)
        if isinstance(value, dict) and isinstance(default_value, dict):
            nested = _diff_from_defaults(default_value, value)
            if nested:
                diff[key] = nested
        elif value != default_value:
            diff[key] = value
    return diff


def _flatten_paths(value: dict[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            paths.extend(_flatten_paths(item, path))
        else:
            paths.append(path)
    return paths


def _value_for_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
