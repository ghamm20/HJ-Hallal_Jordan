"""Portable USB-edition readiness checks and report rendering."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import shutil
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.backend.micro_shadow import MicroLLMShadowPlanner, shadow_backend_status
from app.backend.model_profiles import build_model_profile_status
from app.backend.portable_paths import (
    RuntimePathProfile,
    discover_config_path,
    resolve_runtime_path_profile,
)
from app.backend.runtime_selection import inspect_launcher_runtime
from app.backend.runtime_config import RuntimeConfigStore
from app.backend.update_center import UpdateCenter
from app.reasoning.local_gguf_client import (
    LocalGGUFMainModelRuntime,
    local_gguf_server_runtime_status,
)
from ingestion.parsers.pdf_parser import _discover_ocr_backend
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.persisted_index import inspect_persisted_retrieval_index
from app.retrieval.vector_store import inspect_vector_index

REQUIRED_PACKAGES = ("fastapi", "uvicorn", "pydantic", "requests", "pypdf")


def build_portable_readiness_report(
    repo_root: Path,
    *,
    port: int = 8000,
    runtime_dir_override: Path | None = None,
) -> dict[str, Any]:
    config_path = discover_config_path(
        repo_root,
        runtime_dir_override=runtime_dir_override,
    )
    config_store = RuntimeConfigStore(config_path)
    config_store.ensure_file_exists()
    config = config_store.load_effective()
    path_profile = resolve_runtime_path_profile(
        repo_root,
        config=config,
        runtime_dir_override=runtime_dir_override,
    )
    path_profile.ensure_base_directories()

    launcher_runtime = inspect_launcher_runtime(repo_root)
    package_status = _required_package_status()
    selected_runtime_packages = _selected_runtime_package_status(
        launcher_runtime,
        path_profile=path_profile,
    )
    disk_status = shutil.disk_usage(path_profile.root)
    ocr_backend = _discover_ocr_backend()
    shadow_backend = shadow_backend_status()
    model_status = build_model_profile_status(config, path_profile)
    main_runtime = LocalGGUFMainModelRuntime.status(
        config=config,
        path_profile=path_profile,
    )
    model_status["main"].update(
        {
            "available": main_runtime["ready"],
            "backend_available": main_runtime["backend_available"],
            "backend_name": main_runtime["backend_name"],
            "backend_version": main_runtime["backend_version"],
            "loaded": main_runtime["loaded"],
            "fallback_active": main_runtime["fallback_active"],
            "selected_for_production": main_runtime["selected_for_production"],
        }
    )
    index_status = inspect_persisted_retrieval_index(
        path_profile.index_dir,
        repo_root=repo_root,
        raw_root=path_profile.raw_corpus_dir,
        normalized_root=path_profile.normalized_corpus_dir,
    )
    vector_status = _embedding_runtime_status(config=config, path_profile=path_profile)
    update_center = UpdateCenter(path_profile)
    update_center.ensure_ready()
    update_status = update_center.status(config=config)
    llama_runtime = local_gguf_server_runtime_status(
        config=config,
        path_profile=path_profile,
    )
    shadow_inference = _shadow_inference_status(
        config=config,
        path_profile=path_profile,
        model_status=model_status,
    )
    folder_status = _folder_status(path_profile)
    port_status = _port_status(port)
    runtime_db_writable = _writable_status(path_profile.runtime_dir, "portable-runtime")
    logs_writable = _writable_status(path_profile.logs_dir, "portable-logs")
    config_writable = _writable_status(path_profile.config_dir, "portable-config")
    host_leakage = _host_leakage_status(
        launcher_runtime=launcher_runtime,
        llama_runtime=llama_runtime,
        model_status=model_status,
        folder_status=folder_status,
    )

    fatal_issues: list[str] = []
    warnings: list[str] = []
    missing_packages = [name for name, status in package_status.items() if not status["available"]]
    if missing_packages:
        fatal_issues.append(
            "Missing required Python packages: " + ", ".join(sorted(missing_packages))
        )
    if not folder_status["normalized"]["has_files"]:
        fatal_issues.append(
            "No normalized corpus files were found. Core retrieval-first operation requires a populated normalized corpus."
        )
    if not runtime_db_writable["writable"]:
        fatal_issues.append("Runtime directory is not writable for ops.db.")
    if not logs_writable["writable"]:
        fatal_issues.append("Logs directory is not writable.")
    if not config_writable["writable"]:
        fatal_issues.append("Config directory is not writable.")
    if not port_status["available"]:
        fatal_issues.append(f"Port {port} is not available for local launch.")
    if launcher_runtime["host_fallback_used"]:
        warnings.append(
            "Launcher would currently fall back to host Python because no bundled project-local runtime was found."
        )
    if launcher_runtime["bundled_python_present"] and not launcher_runtime["bundled_packages_present"]:
        warnings.append(
            "Bundled Python appears to be present, but no bundled package directory was detected."
        )
    if (
        launcher_runtime.get("selected")
        and launcher_runtime["selected"]["bundled"]
        and not selected_runtime_packages["all_required_available"]
    ):
        warnings.append(
            "Launcher-selected bundled runtime is missing required Python packages: "
            + ", ".join(selected_runtime_packages["missing"])
        )
    if not index_status["index_present"]:
        warnings.append(
            "Persistent retrieval index assets are missing under data/index. Retrieval can still fall back to the normalized corpus, but USB startup loses the persisted index layer."
        )
    elif not index_status["retrieval_index_loadable"]:
        warnings.append(
            "Persistent retrieval index exists, but retrieval would currently fall back to normalized-corpus scanning: "
            + str(index_status["retrieval_fallback_reason"] or "unknown_reason")
        )
    elif not index_status.get("prepared_search_loadable", False):
        warnings.append(
            "Prepared search metadata is missing or stale, so retrieval warmup would rebuild postings in memory: "
            + str(index_status.get("prepared_search_fallback_reason") or "unknown_reason")
        )
    if config.embedding_enabled and not vector_status["vector_index_present"]:
        warnings.append(
            "Hybrid retrieval is configured, but the vector index file is not present."
        )
    if config.embedding_enabled and vector_status["fallback_reason"]:
        warnings.append(
            "Hybrid retrieval would currently fall back to keyword-only: "
            + vector_status["fallback_reason"]
        )
    if not llama_runtime["server_binary_present"]:
        fatal_issues.append("Bundled llama.cpp runtime is missing at runtime/llama/llama-server.exe.")
    if not model_status["micro"]["available"]:
        warnings.append("Configured micro model is not currently available.")
    if model_status["main"]["enabled"] and not model_status["main"]["available"]:
        warnings.append("Configured main model is not currently available.")
    if (
        model_status["micro"]["enabled"]
        and model_status["micro"]["provider"] == "local_gguf"
        and model_status["micro"]["available"]
        and not shadow_backend["available"]
    ):
        warnings.append(
            "Micro model asset is present, but no local GGUF shadow runtime backend is available."
        )
    if not ocr_backend:
        warnings.append("No supported OCR backend is available on this machine.")
    if shadow_inference["attempted"] and not shadow_inference["working"]:
        warnings.append(
            "Local GGUF shadow inference is installed but the smoke probe did not complete successfully."
        )

    overall_status = "ready"
    if fatal_issues:
        overall_status = "blocked"
    elif warnings:
        overall_status = "warning"

    return {
        "generated_at": _utc_now(),
        "overall_status": overall_status,
        "portable_mode": config.portable_mode,
        "port": port,
        "path_profile": _serialize_path_profile(path_profile),
        "launcher_runtime": launcher_runtime,
        "selected_runtime_packages": selected_runtime_packages,
        "required_packages": package_status,
        "folders": folder_status,
        "index_assets": index_status,
        "embeddings": vector_status,
        "updates": update_status,
        "disk": {
            "root": str(path_profile.root),
            "total_bytes": disk_status.total,
            "used_bytes": disk_status.used,
            "free_bytes": disk_status.free,
        },
        "writable": {
            "runtime": runtime_db_writable,
            "logs": logs_writable,
            "config": config_writable,
        },
        "ocr": {
            "available": bool(ocr_backend),
            "backend_name": str(getattr(ocr_backend, "backend_name", "") or ""),
        },
        "shadow_backend": shadow_backend,
        "shadow_inference": shadow_inference,
        "models": model_status,
        "llama_runtime": llama_runtime,
        "host_leakage": host_leakage,
        "overall_usb_ready": not fatal_issues,
        "port_status": port_status,
        "fatal_issues": fatal_issues,
        "warnings": warnings,
    }


def write_portable_readiness_artifact(
    report: dict[str, Any],
    repo_root: Path,
) -> Path:
    readiness_dir = repo_root / "docs" / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    path = readiness_dir / "portable-readiness-report.md"
    path.write_text(render_portable_readiness_markdown(report), encoding="utf-8")
    return path


def render_portable_readiness_markdown(report: dict[str, Any]) -> str:
    root_path = Path(report["path_profile"]["root"])
    lines = [
        "# Portable Readiness Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"- Overall status: `{report['overall_status']}`",
        f"- overall_usb_ready: `{report['overall_usb_ready']}`",
        f"- Portable mode: `{report['portable_mode']}`",
        f"- Launch port: `{report['port']}`",
        "",
        "## Path Profile",
    ]
    for key, value in report["path_profile"].items():
        lines.append(f"- `{key}`: `{_display_report_path(value, root_path)}`")

    lines.extend(["", "## Storage", f"- Root: `{_display_report_path(report['disk']['root'], root_path)}`"])
    for key in ("total_bytes", "used_bytes", "free_bytes"):
        lines.append(f"- `{key}`: `{report['disk'][key]}`")

    lines.extend(["", "## Folders"])
    for key, payload in report["folders"].items():
        lines.append(
            f"- `{key}`: path=`{_display_report_path(payload['path'], root_path)}` files=`{payload['file_count']}` size_bytes=`{payload['size_bytes']}` has_files=`{payload['has_files']}`"
        )

    lines.extend(["", "## Retrieval Index"])
    index_assets = report["index_assets"]
    lines.append(f"- index_present: `{index_assets['index_present']}`")
    lines.append(f"- index_root: `{_display_report_path(index_assets['index_root'], root_path)}`")
    lines.append(f"- manifest_path: `{_display_report_path(index_assets['manifest_path'], root_path)}`")
    lines.append(f"- chunks_path: `{_display_report_path(index_assets['chunks_path'], root_path)}`")
    lines.append(f"- document_count: `{index_assets['document_count']}`")
    lines.append(f"- chunk_count: `{index_assets['chunk_count']}`")
    lines.append(f"- index_size_bytes: `{index_assets['index_size_bytes']}`")
    lines.append(f"- generated_at: `{index_assets['generated_at'] or 'none'}`")
    lines.append(f"- index_format: `{index_assets['index_format']}`")
    lines.append(
        f"- retrieval_bootstrap_source: `{index_assets['retrieval_bootstrap_source']}`"
    )
    lines.append(
        f"- retrieval_fallback_reason: `{index_assets['retrieval_fallback_reason'] or 'none'}`"
    )
    lines.append(
        f"- retrieval_index_loadable: `{index_assets['retrieval_index_loadable']}`"
    )
    lines.append(
        f"- prepared_search_present: `{index_assets.get('prepared_search_present', False)}`"
    )
    lines.append(
        f"- prepared_search_loadable: `{index_assets.get('prepared_search_loadable', False)}`"
    )
    lines.append(
        f"- prepared_search_fallback_reason: `{index_assets.get('prepared_search_fallback_reason') or 'none'}`"
    )
    lines.append(
        f"- prepared_search_manifest_path: `{_display_report_path(index_assets.get('prepared_search_manifest_path', ''), root_path)}`"
    )
    lines.append(
        f"- prepared_search_token_postings_path: `{_display_report_path(index_assets.get('prepared_search_token_postings_path', ''), root_path)}`"
    )
    lines.append(
        f"- prepared_search_generated_at: `{index_assets.get('prepared_search_generated_at') or 'none'}`"
    )
    lines.append(
        f"- prepared_search_token_count: `{index_assets.get('prepared_search_token_count', 0)}`"
    )
    admission_summary = index_assets.get("admission_summary", {})
    if admission_summary:
        lines.append(
            "- admission_summary: "
            + ", ".join(
                [
                    f"total_candidates=`{admission_summary.get('total_candidates', 0)}`",
                    f"admitted=`{admission_summary.get('admitted_count', 0)}`",
                    f"admitted_with_warnings=`{admission_summary.get('admitted_with_warnings_count', 0)}`",
                    f"deferred=`{admission_summary.get('deferred_count', 0)}`",
                    f"rejected=`{admission_summary.get('rejected_count', 0)}`",
                ]
            )
        )
    source_breakdown = index_assets.get("source_breakdown", {})
    if source_breakdown:
        for key, payload in source_breakdown.items():
            if not isinstance(payload, dict) or not payload:
                continue
            formatted = ", ".join(f"{label}={count}" for label, count in payload.items())
            lines.append(f"- `{key}`: {formatted}")

    lines.extend(["", "## Semantic Retrieval"])
    embeddings = report["embeddings"]
    lines.append(f"- embedding_enabled: `{embeddings['embedding_enabled']}`")
    lines.append(f"- embedding_model: `{embeddings['embedding_model']}`")
    lines.append(f"- embedding_device: `{embeddings['embedding_device']}`")
    lines.append(f"- vector_index_present: `{embeddings['vector_index_present']}`")
    lines.append(f"- vector_index_path: `{_display_report_path(embeddings['vector_index_path'], root_path)}`")
    lines.append(f"- embedding_model_loaded: `{embeddings['embedding_model_loaded']}`")
    lines.append(f"- hybrid_retrieval_active: `{embeddings['hybrid_retrieval_active']}`")
    lines.append(f"- fallback_reason: `{embeddings['fallback_reason'] or 'none'}`")

    lines.extend(["", "## Writable Checks"])
    for key, payload in report["writable"].items():
        lines.append(
            f"- `{key}`: writable=`{payload['writable']}` path=`{_display_report_path(payload['path'], root_path)}`"
        )
        if payload.get("error"):
            lines.append(f"  error=`{payload['error']}`")

    lines.extend(["", "## Required Packages"])
    for key, payload in report["required_packages"].items():
        lines.append(f"- `{key}`: available=`{payload['available']}`")

    lines.extend(["", "## Launcher Runtime"])
    launcher_runtime = report["launcher_runtime"]
    selected = launcher_runtime.get("selected")
    lines.append(f"- bundled_python_present: `{launcher_runtime['bundled_python_present']}`")
    lines.append(f"- bundled_packages_present: `{launcher_runtime['bundled_packages_present']}`")
    lines.append(f"- host_fallback_used: `{launcher_runtime['host_fallback_used']}`")
    if selected:
        lines.append(f"- selected_label: `{selected['label']}`")
        lines.append(f"- selected_source: `{selected['source']}`")
        lines.append(f"- selected_executable: `{_display_report_path(selected['executable'], root_path)}`")
    else:
        lines.append("- selected_label: `none`")
        lines.append("- selected_source: `none`")
        lines.append("- selected_executable: `none`")
    if launcher_runtime["bundled_python_paths"]:
        lines.append(
            "- bundled_python_paths: "
            + ", ".join(
                f"`{_display_report_path(item, root_path)}`"
                for item in launcher_runtime["bundled_python_paths"]
            )
        )
    else:
        lines.append("- bundled_python_paths: none")
    if launcher_runtime["bundled_package_paths"]:
        lines.append(
            "- bundled_package_paths: "
            + ", ".join(
                f"`{_display_report_path(item, root_path)}`"
                for item in launcher_runtime["bundled_package_paths"]
            )
        )
    else:
        lines.append("- bundled_package_paths: none")

    lines.extend(["", "## Bundled llama.cpp"])
    llama_runtime = report["llama_runtime"]
    lines.append(
        f"- bundled_llama_present: `{llama_runtime['server_binary_present']}`"
    )
    lines.append(
        f"- bundled_llama_path: `{_display_report_path(llama_runtime['server_binary_path'], root_path)}`"
    )
    lines.append(
        f"- llama_server_url: `{llama_runtime['server_url']}`"
    )
    lines.append(
        f"- llama_server_health_ok: `{llama_runtime['server_health_ok']}`"
    )
    lines.append(
        f"- llama_server_state_path: `{_display_report_path(llama_runtime['server_state_path'], root_path)}`"
    )
    state_payload = llama_runtime.get("server_state") or {}
    lines.append(
        f"- llama_server_pid: `{state_payload.get('pid') or 'none'}`"
    )
    lines.append(
        f"- llama_server_started_by: `{state_payload.get('started_by') or 'none'}`"
    )
    lines.append(
        f"- production_model_name: `{report['models']['main']['name'] or 'none'}`"
    )
    lines.append(
        f"- production_model_path: `{_display_report_path(report['models']['main']['resolved_path'] or report['models']['main']['path'] or 'none', root_path)}`"
    )
    visible_models = llama_runtime.get("server_visible_models") or []
    if visible_models:
        lines.append(
            "- llama_visible_models: "
            + ", ".join(f"`{item}`" for item in visible_models)
        )
    else:
        lines.append("- llama_visible_models: none")
    lines.append(f"- llama_error: `{llama_runtime.get('server_error') or 'none'}`")

    lines.extend(["", "## Internet Updates"])
    updates = report["updates"]
    lines.append(f"- internet_update_check_enabled: `{updates['internet_update_check_enabled']}`")
    lines.append(f"- update_manifest_url: `{updates['update_manifest_url'] or 'none'}`")
    lines.append(f"- last_update_check: `{updates['last_update_check'] or 'none'}`")
    lines.append(f"- last_update_check_status: `{updates['last_update_check_status']}`")
    lines.append(f"- last_update_check_error: `{updates['last_update_check_error'] or 'none'}`")
    lines.append(f"- pending_updates_count: `{updates['pending_updates_count']}`")
    lines.append(f"- downloaded_updates_count: `{updates['downloaded_updates_count']}`")
    lines.append(f"- verified_updates_count: `{updates['verified_updates_count']}`")
    lines.append(f"- index_rebuild_needed: `{updates['index_rebuild_needed']}`")
    lines.append(f"- offline_mode: `{updates['offline_mode']}`")
    lines.append(f"- updates_root: `{_display_report_path(updates['updates_root'], root_path)}`")
    lines.append(f"- inbox_dir: `{_display_report_path(updates['inbox_dir'], root_path)}`")
    lines.append(f"- processed_dir: `{_display_report_path(updates['processed_dir'], root_path)}`")
    lines.append(f"- failed_dir: `{_display_report_path(updates['failed_dir'], root_path)}`")
    lines.append(f"- quarantine_dir: `{_display_report_path(updates['quarantine_dir'], root_path)}`")
    lines.append(f"- manifests_dir: `{_display_report_path(updates['manifests_dir'], root_path)}`")

    lines.extend(["", "## Host Leakage Audit"])
    host_leakage = report["host_leakage"]
    lines.append(f"- host_python_fallback_used: `{host_leakage['host_python_fallback_used']}`")
    lines.append(
        f"- local_llama_runtime_dependency: `{host_leakage['local_llama_runtime_dependency']}`"
    )
    lines.append(
        f"- local_main_model_dependency: `{host_leakage['local_main_model_dependency']}`"
    )
    lines.append(
        f"- huggingface_cache_dependency: `{host_leakage['huggingface_cache_dependency']}`"
    )
    lines.append(
        f"- internet_dependency_normal_launch: `{host_leakage['internet_dependency_normal_launch']}`"
    )

    lines.extend(["", "## Selected Runtime Package Probe"])
    lines.append(
        f"- all_required_available: `{report['selected_runtime_packages']['all_required_available']}`"
    )
    lines.append(
        f"- available: `{', '.join(report['selected_runtime_packages']['available']) or 'none'}`"
    )
    lines.append(
        f"- missing: `{', '.join(report['selected_runtime_packages']['missing']) or 'none'}`"
    )
    lines.append(
        f"- probe_error: `{report['selected_runtime_packages']['probe_error'] or 'none'}`"
    )

    lines.extend(["", "## OCR"])
    lines.append(f"- available: `{report['ocr']['available']}`")
    lines.append(f"- backend_name: `{report['ocr']['backend_name'] or 'none'}`")

    lines.extend(["", "## Shadow Runtime"])
    lines.append(f"- available: `{report['shadow_backend']['available']}`")
    lines.append(f"- backend_name: `{report['shadow_backend']['backend_name'] or 'none'}`")
    lines.append(f"- backend_version: `{report['shadow_backend'].get('backend_version') or 'none'}`")

    lines.extend(["", "## Shadow Inference"])
    lines.append(f"- attempted: `{report['shadow_inference']['attempted']}`")
    lines.append(f"- working: `{report['shadow_inference']['working']}`")
    lines.append(f"- latency_ms: `{report['shadow_inference']['latency_ms']}`")
    lines.append(f"- error: `{report['shadow_inference']['error'] or 'none'}`")
    if report["shadow_inference"].get("sample"):
        sample = report["shadow_inference"]["sample"]
        lines.append(f"- predicted_intent: `{sample.get('micro_predicted_intent') or 'none'}`")
        lines.append(f"- suggested_mode: `{sample.get('micro_suggested_mode') or 'none'}`")

    lines.extend(["", "## Models"])
    for key in ("micro", "main"):
        payload = report["models"][key]
        lines.append(
            f"- `{key}`: enabled=`{payload['enabled']}` configured=`{payload['configured']}` provider=`{payload['provider']}` name=`{payload['name'] or 'none'}` role=`{payload['role']}` available=`{payload['available']}` source=`{payload['availability_source']}`"
        )
        if key == "main":
            lines.append(
                f"  backend_available=`{payload.get('backend_available')}` backend_name=`{payload.get('backend_name') or 'none'}` backend_version=`{payload.get('backend_version') or 'none'}`"
            )
            lines.append(
                f"  loaded=`{payload.get('loaded')}` selected_for_production=`{payload.get('selected_for_production')}` fallback_active=`{payload.get('fallback_active')}`"
            )
        lines.append(f"  configured_path=`{_display_report_path(payload['path'] or 'none', root_path)}`")
        lines.append(f"  resolved_path=`{_display_report_path(payload['resolved_path'] or 'none', root_path)}`")
        lines.append(f"  project_local=`{payload['project_local']}`")
        lines.append(f"  within_expected_directory=`{payload['within_expected_directory']}`")
        lines.append(f"  exists=`{payload['exists']}` is_file=`{payload['is_file']}`")
        lines.append(f"  extension=`{payload['extension'] or 'none'}` is_gguf=`{payload['is_gguf']}`")
        lines.append(f"  size_bytes=`{payload['size_bytes']}` size_over_1gb=`{payload['size_over_1gb']}`")
        if payload.get("error"):
            lines.append(f"  error=`{payload['error']}`")
    if report["models"].get("availability_error"):
        lines.append(f"- availability_error: `{report['models']['availability_error']}`")
    if report["llama_runtime"].get("server_visible_models"):
        lines.append(
            "- llama_visible_models: "
            + ", ".join(f"`{item}`" for item in report["llama_runtime"]["server_visible_models"])
        )
    else:
        lines.append("- llama_visible_models: none")

    lines.extend(["", "## Port Status"])
    lines.append(f"- available: `{report['port_status']['available']}`")
    lines.append(f"- host: `{report['port_status']['host']}`")
    lines.append(f"- port: `{report['port_status']['port']}`")
    if report["port_status"].get("error"):
        lines.append(f"- error: `{report['port_status']['error']}`")

    lines.extend(["", "## Fatal Issues"])
    if report["fatal_issues"]:
        lines.extend(f"- {item}" for item in report["fatal_issues"])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def _serialize_path_profile(path_profile: RuntimePathProfile) -> dict[str, Any]:
    return {
        "root": str(path_profile.root),
        "raw_corpus_dir": str(path_profile.raw_corpus_dir),
        "normalized_corpus_dir": str(path_profile.normalized_corpus_dir),
        "index_dir": str(path_profile.index_dir),
        "runtime_dir": str(path_profile.runtime_dir),
        "logs_dir": str(path_profile.logs_dir),
        "config_dir": str(path_profile.config_dir),
        "config_path": str(path_profile.config_path),
        "runtime_db_path": str(path_profile.runtime_db_path),
        "updates_root": str(path_profile.updates_root),
        "updates_inbox_dir": str(path_profile.updates_inbox_dir),
        "updates_processed_dir": str(path_profile.updates_processed_dir),
        "updates_failed_dir": str(path_profile.updates_failed_dir),
        "updates_quarantine_dir": str(path_profile.updates_quarantine_dir),
        "updates_manifests_dir": str(path_profile.updates_manifests_dir),
        "update_log_path": str(path_profile.update_log_path),
        "micro_models_dir": str(path_profile.micro_models_dir),
        "main_models_dir": str(path_profile.main_models_dir),
    }


def _display_report_path(value: Any, root_path: Path) -> str:
    text = str(value or "")
    if not text or text == "none":
        return text or "none"
    normalized = text.replace("/", "\\")
    try:
        candidate = Path(normalized)
    except OSError:
        return text
    try:
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(root_path.resolve()).as_posix() or "."
            except (OSError, ValueError):
                name = candidate.name or candidate.drive or "path"
                return f"external/{name}"
    except (OSError, ValueError):
        return text
    return text


def _required_package_status() -> dict[str, dict[str, Any]]:
    return {
        name: {"available": importlib.util.find_spec(name) is not None}
        for name in REQUIRED_PACKAGES
    }


def _selected_runtime_package_status(
    launcher_runtime: dict[str, Any],
    *,
    path_profile: RuntimePathProfile,
) -> dict[str, Any]:
    selected = launcher_runtime.get("selected")
    if not selected:
        return {
            "all_required_available": False,
            "available": [],
            "missing": list(REQUIRED_PACKAGES),
            "probe_error": "no_runtime_selected",
        }

    command = [
        selected["executable"],
        *selected.get("prefix", []),
        "-c",
        (
            "import importlib.util, json; "
            f"required={REQUIRED_PACKAGES!r}; "
            "available=[name for name in required if importlib.util.find_spec(name) is not None]; "
            "missing=[name for name in required if importlib.util.find_spec(name) is None]; "
            "print(json.dumps({'available': available, 'missing': missing}))"
        ),
    ]
    env = os.environ.copy()
    runtime_site_packages = path_profile.root / "runtime" / "site-packages"
    if runtime_site_packages.exists() and runtime_site_packages.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(runtime_site_packages)
            if not existing
            else str(runtime_site_packages) + os.pathsep + existing
        )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=20,
        )
    except OSError as exc:
        return {
            "all_required_available": False,
            "available": [],
            "missing": list(REQUIRED_PACKAGES),
            "probe_error": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired:
        return {
            "all_required_available": False,
            "available": [],
            "missing": list(REQUIRED_PACKAGES),
            "probe_error": "timeout",
        }

    if completed.returncode != 0:
        return {
            "all_required_available": False,
            "available": [],
            "missing": list(REQUIRED_PACKAGES),
            "probe_error": (completed.stderr or completed.stdout or "").strip() or "probe_failed",
        }

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "all_required_available": False,
            "available": [],
            "missing": list(REQUIRED_PACKAGES),
            "probe_error": f"json_decode_error: {exc}",
        }

    available = list(payload.get("available") or [])
    missing = list(payload.get("missing") or [])
    return {
        "all_required_available": not missing,
        "available": available,
        "missing": missing,
        "probe_error": None,
    }


def _folder_status(path_profile: RuntimePathProfile) -> dict[str, dict[str, Any]]:
    return {
        "raw": _scan_folder(path_profile.raw_corpus_dir),
        "normalized": _scan_folder(path_profile.normalized_corpus_dir),
        "index": _scan_folder(path_profile.index_dir),
        "updates_inbox": _scan_folder(path_profile.updates_inbox_dir),
        "updates_processed": _scan_folder(path_profile.updates_processed_dir),
        "updates_failed": _scan_folder(path_profile.updates_failed_dir),
        "updates_quarantine": _scan_folder(path_profile.updates_quarantine_dir),
        "updates_manifests": _scan_folder(path_profile.updates_manifests_dir),
        "models_micro": _scan_folder(path_profile.micro_models_dir),
        "models_main": _scan_folder(path_profile.main_models_dir),
    }


def _scan_folder(path: Path) -> dict[str, Any]:
    file_count = 0
    size_bytes = 0
    if path.exists():
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            file_count += 1
            try:
                size_bytes += candidate.stat().st_size
            except OSError:
                continue
    return {
        "path": str(path),
        "exists": path.exists(),
        "has_files": file_count > 0,
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _writable_status(path: Path, prefix: str) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".{prefix}-{datetime.now(UTC).timestamp():.0f}.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"writable": True, "path": str(path), "error": None}
    except OSError as exc:
        return {
            "writable": False,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _port_status(port: int, host: str = "127.0.0.1") -> dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return {"available": True, "host": host, "port": port, "error": None}
    except OSError as exc:
        return {
            "available": False,
            "host": host,
            "port": port,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sock.close()


def _embedding_runtime_status(
    *,
    config: Any,
    path_profile: RuntimePathProfile,
) -> dict[str, Any]:
    index_path = path_profile.root / str(config.vector_index_path or "data/index/vector_index.npz")
    vector_index = inspect_vector_index(index_path)
    embedder = LocalEmbedder(
        model_name=str(config.embedding_model or "").strip(),
        device=str(config.embedding_device or "cpu").strip(),
        enabled=bool(config.embedding_enabled),
    )
    embedder_status = embedder.status()
    fallback_reason = ""
    if not config.embedding_enabled:
        fallback_reason = "embedding_disabled"
    elif not vector_index["exists"]:
        fallback_reason = "vector_index_missing"
    elif not embedder_status["dependency_available"]:
        fallback_reason = embedder_status["reason"]
    elif not vector_index["loaded"]:
        fallback_reason = vector_index["reason"]
    return {
        "embedding_enabled": bool(config.embedding_enabled),
        "embedding_model": str(config.embedding_model or ""),
        "embedding_device": str(config.embedding_device or "cpu"),
        "vector_index_present": bool(vector_index["exists"]),
        "vector_index_path": str(index_path),
        "embedding_model_loaded": bool(embedder_status["model_loaded"]),
        "hybrid_retrieval_active": bool(
            config.embedding_enabled
            and vector_index["loaded"]
            and embedder_status["dependency_available"]
        ),
        "fallback_reason": fallback_reason,
        "embedder": embedder_status,
        "vector_index": vector_index,
    }


def _host_leakage_status(
    *,
    launcher_runtime: dict[str, Any],
    llama_runtime: dict[str, Any],
    model_status: dict[str, Any],
    folder_status: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    micro_model_path = str(
        model_status.get("micro", {}).get("resolved_path")
        or model_status.get("micro", {}).get("path")
        or ""
    ).lower()
    huggingface_cache_dependency = "huggingface" in micro_model_path
    local_launch_assets_ready = all(
        (
            not launcher_runtime["host_fallback_used"],
            launcher_runtime["bundled_packages_present"],
            bool(llama_runtime.get("server_binary_present")),
            bool(model_status.get("main", {}).get("available")),
            folder_status["normalized"]["has_files"],
        )
    )
    return {
        "host_python_fallback_used": bool(launcher_runtime["host_fallback_used"]),
        "local_llama_runtime_dependency": not bool(llama_runtime.get("server_binary_present")),
        "local_main_model_dependency": not bool(model_status.get("main", {}).get("available")),
        "huggingface_cache_dependency": huggingface_cache_dependency,
        "internet_dependency_normal_launch": not local_launch_assets_ready,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _shadow_inference_status(
    *,
    config: Any,
    path_profile: RuntimePathProfile,
    model_status: dict[str, Any],
) -> dict[str, Any]:
    micro_status = model_status.get("micro", {})
    if not micro_status.get("enabled"):
        return {
            "attempted": False,
            "working": False,
            "latency_ms": 0,
            "error": "micro_model_disabled",
            "sample": None,
        }
    if not micro_status.get("available"):
        return {
            "attempted": False,
            "working": False,
            "latency_ms": 0,
            "error": micro_status.get("error") or "micro_model_unavailable",
            "sample": None,
        }
    if not shadow_backend_status().get("available"):
        return {
            "attempted": False,
            "working": False,
            "latency_ms": 0,
            "error": "local_gguf_shadow_backend_unavailable",
            "sample": None,
        }
    planner = MicroLLMShadowPlanner(
        path_profile=path_profile,
        log_path=path_profile.runtime_dir / "micro_llm_shadow.jsonl",
    )
    result = planner.smoke_test(config=config)
    return {
        "attempted": True,
        "working": bool(result.get("success")),
        "latency_ms": int(result.get("latency_ms") or 0),
        "error": result.get("error"),
        "sample": {
            "micro_predicted_intent": result.get("micro_predicted_intent"),
            "micro_suggested_mode": result.get("micro_suggested_mode"),
            "micro_rewritten_query": result.get("micro_rewritten_query"),
            "confidence": result.get("confidence"),
        } if result.get("success") else None,
    }
