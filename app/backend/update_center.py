"""Project-local update inbox, verification, and review-stage apply helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from app.backend.portable_paths import RuntimePathProfile

UPDATE_TYPES = {
    "app",
    "corpus",
    "transcript",
    "model",
    "metadata",
    "prompt_pack",
}

SAFE_EXTENSIONS_BY_TYPE: dict[str, set[str]] = {
    "app": {".zip"},
    "corpus": {".zip", ".json", ".tar", ".tgz", ".gz"},
    "transcript": {".zip", ".json", ".tar", ".tgz", ".gz"},
    "model": {".gguf", ".zip", ".bin"},
    "metadata": {".json", ".zip"},
    "prompt_pack": {".json", ".zip", ".md", ".txt"},
}


@dataclass(slots=True)
class UpdateCenter:
    path_profile: RuntimePathProfile

    def ensure_ready(self) -> None:
        self.path_profile.ensure_base_directories()
        if not self.state_path.exists():
            self._write_state(self._default_state())
        if not self.update_log_path.exists():
            self.update_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.update_log_path.write_text("", encoding="utf-8")

    @property
    def updates_root(self) -> Path:
        return self.path_profile.updates_root

    @property
    def inbox_dir(self) -> Path:
        return self.path_profile.updates_inbox_dir

    @property
    def processed_dir(self) -> Path:
        return self.path_profile.updates_processed_dir

    @property
    def failed_dir(self) -> Path:
        return self.path_profile.updates_failed_dir

    @property
    def quarantine_dir(self) -> Path:
        return self.path_profile.updates_quarantine_dir

    @property
    def manifests_dir(self) -> Path:
        return self.path_profile.updates_manifests_dir

    @property
    def state_path(self) -> Path:
        return self.path_profile.update_state_path

    @property
    def update_log_path(self) -> Path:
        return self.path_profile.update_log_path

    @property
    def latest_manifest_path(self) -> Path:
        return self.manifests_dir / "latest_manifest.json"

    def status(self, *, config: Any) -> dict[str, Any]:
        self.ensure_ready()
        state = self._load_state()
        manifest_url = str(getattr(config, "update_manifest_url", "") or "").strip()
        update_check_enabled = bool(getattr(config, "update_check_enabled", True))
        enabled = bool(update_check_enabled and manifest_url)
        disabled_reason = None
        if not enabled:
            disabled_reason = (
                "disabled_by_config" if not update_check_enabled else "no_manifest_url_configured"
            )
        updates = [self._decorate_update_row(row) for row in state.get("updates", {}).values()]
        updates.sort(key=lambda item: (item["local_status_sort"], item["update_id"]))
        pending_updates = [item for item in updates if item["local_status"] not in {"review_staged"}]
        downloaded_updates = [
            item
            for item in updates
            if item["download"].get("status") in {"downloaded", "verified", "processed"}
        ]
        verified_updates = [
            item
            for item in updates
            if item["verification"].get("status") == "verified"
        ]
        index_rebuild_needed = any(
            bool(item["manifest"].get("requires_index_rebuild"))
            and item["apply"].get("status") == "review_staged"
            for item in updates
        )
        last_check_status = str(state.get("last_check_status") or "never_checked")
        if not enabled:
            last_check_status = "disabled"
        return {
            "enabled": enabled,
            "internet_update_check_enabled": enabled,
            "disabled_reason": disabled_reason,
            "update_manifest_url": manifest_url,
            "last_update_check": state.get("last_check_at"),
            "last_update_check_status": last_check_status,
            "last_update_check_error": state.get("last_check_error"),
            "last_update_check_duration_ms": int(state.get("last_check_duration_ms") or 0),
            "pending_updates_count": len(pending_updates),
            "downloaded_updates_count": len(downloaded_updates),
            "verified_updates_count": len(verified_updates),
            "index_rebuild_needed": index_rebuild_needed,
            "offline_mode": (not enabled) or last_check_status in {"unavailable", "error"},
            "updates_root": str(self.updates_root),
            "inbox_dir": str(self.inbox_dir),
            "processed_dir": str(self.processed_dir),
            "failed_dir": str(self.failed_dir),
            "quarantine_dir": str(self.quarantine_dir),
            "manifests_dir": str(self.manifests_dir),
            "update_log_path": str(self.update_log_path),
            "available_updates": updates,
            "history": self.history(limit=20),
        }

    def check_for_updates(
        self,
        *,
        config: Any,
        trigger: str,
    ) -> dict[str, Any]:
        self.ensure_ready()
        manifest_url = str(getattr(config, "update_manifest_url", "") or "").strip()
        timeout_seconds = int(getattr(config, "update_check_timeout_seconds", 3) or 3)
        if not getattr(config, "update_check_enabled", True) or not manifest_url:
            disabled_reason = (
                "disabled_by_config"
                if not getattr(config, "update_check_enabled", True)
                else "no_manifest_url_configured"
            )
            state = self._load_state()
            state["last_check_at"] = _utc_now()
            state["last_check_status"] = "disabled"
            state["last_check_error"] = disabled_reason
            state["last_check_duration_ms"] = 0
            self._write_state(state)
            self._log_event(
                action="check_manifest",
                status="disabled",
                details={"trigger": trigger, "reason": disabled_reason},
            )
            return self.status(config=config)

        started = datetime.now(UTC)
        state = self._load_state()
        try:
            response = requests.get(manifest_url, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            normalized_updates = _normalize_manifest_payload(payload)
        except requests.RequestException as exc:
            state["last_check_at"] = _utc_now()
            state["last_check_status"] = "unavailable"
            state["last_check_error"] = f"{type(exc).__name__}: {exc}"
            state["last_check_duration_ms"] = _duration_ms(started)
            self._write_state(state)
            self._log_event(
                action="check_manifest",
                status="update_check_unavailable",
                details={
                    "trigger": trigger,
                    "manifest_url": manifest_url,
                    "error": state["last_check_error"],
                },
            )
            return self.status(config=config)
        except ValueError as exc:
            state["last_check_at"] = _utc_now()
            state["last_check_status"] = "error"
            state["last_check_error"] = str(exc)
            state["last_check_duration_ms"] = _duration_ms(started)
            self._write_state(state)
            self._log_event(
                action="check_manifest",
                status="error",
                details={
                    "trigger": trigger,
                    "manifest_url": manifest_url,
                    "error": str(exc),
                },
            )
            return self.status(config=config)

        archived_path = self.manifests_dir / f"manifest-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        archived_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.latest_manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        existing_updates = state.get("updates", {})
        merged_updates: dict[str, Any] = {}
        for item in normalized_updates:
            update_id = item["update_id"]
            previous = existing_updates.get(update_id, {})
            preserved = {}
            previous_manifest = previous.get("manifest", {})
            if (
                previous_manifest.get("sha256") == item["sha256"]
                and previous_manifest.get("file_url") == item["file_url"]
            ):
                preserved = {
                    "download": previous.get("download", {}),
                    "verification": previous.get("verification", {}),
                    "apply": previous.get("apply", {}),
                }
            merged_updates[update_id] = {
                "update_id": update_id,
                "manifest": item,
                "download": preserved.get("download", {"status": "not_downloaded"}),
                "verification": preserved.get("verification", {"status": "not_verified"}),
                "apply": preserved.get("apply", {"status": "not_applied"}),
            }
        state["last_check_at"] = _utc_now()
        state["last_check_status"] = "ok"
        state["last_check_error"] = None
        state["last_check_duration_ms"] = _duration_ms(started)
        state["last_manifest_url"] = manifest_url
        state["last_manifest_archive_path"] = str(archived_path)
        state["updates"] = merged_updates
        self._write_state(state)
        self._log_event(
            action="check_manifest",
            status="ok",
            details={
                "trigger": trigger,
                "manifest_url": manifest_url,
                "available_update_count": len(normalized_updates),
                "archived_manifest_path": str(archived_path),
            },
        )
        return self.status(config=config)

    def download_update(self, *, update_id: str, config: Any) -> dict[str, Any]:
        self.ensure_ready()
        state = self._load_state()
        entry = self._get_update_entry(state, update_id)
        manifest = entry["manifest"]
        file_url = str(manifest.get("file_url") or "").strip()
        if not file_url:
            raise ValueError("update file_url is missing")
        parsed = urlparse(file_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("update file_url must use http or https")

        filename = _safe_filename_from_url(file_url, fallback=f"{update_id}.bin")
        inbox_path = self._unique_path(self.inbox_dir, filename)
        started = datetime.now(UTC)
        try:
            with requests.get(file_url, stream=True, timeout=max(int(getattr(config, "update_check_timeout_seconds", 3) or 3), 3)) as response:
                response.raise_for_status()
                with inbox_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        handle.write(chunk)
        except requests.RequestException as exc:
            entry["download"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "downloaded_at": _utc_now(),
            }
            self._write_state(state)
            self._log_event(
                action="download_update",
                status="failed",
                details={"update_id": update_id, "error": entry["download"]["error"]},
            )
            raise

        actual_size = inbox_path.stat().st_size
        expected_size = int(manifest.get("size_bytes") or 0)
        if expected_size > 0 and actual_size != expected_size:
            failed_path = self.failed_dir / inbox_path.name
            failed_path = self._move_with_unique_name(inbox_path, failed_path)
            entry["download"] = {
                "status": "failed",
                "error": "size_mismatch",
                "downloaded_at": _utc_now(),
                "local_path": str(failed_path),
                "size_bytes": actual_size,
                "expected_size_bytes": expected_size,
            }
            entry["verification"] = {"status": "not_verified"}
            self._write_state(state)
            self._log_event(
                action="download_update",
                status="failed",
                details={
                    "update_id": update_id,
                    "error": "size_mismatch",
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                },
            )
            return self._decorate_update_row(entry)

        if not _is_safe_extension(inbox_path.suffix.lower(), str(manifest.get("type") or "")):
            quarantined_path = self.quarantine_dir / inbox_path.name
            quarantined_path = self._move_with_unique_name(inbox_path, quarantined_path)
            entry["download"] = {
                "status": "quarantined",
                "error": "unsafe_file_extension",
                "downloaded_at": _utc_now(),
                "local_path": str(quarantined_path),
                "size_bytes": actual_size,
            }
            entry["verification"] = {"status": "not_verified"}
            self._write_state(state)
            self._log_event(
                action="download_update",
                status="quarantined",
                details={
                    "update_id": update_id,
                    "local_path": str(quarantined_path),
                    "extension": inbox_path.suffix.lower(),
                },
            )
            return self._decorate_update_row(entry)

        entry["download"] = {
            "status": "downloaded",
            "downloaded_at": _utc_now(),
            "local_path": str(inbox_path),
            "size_bytes": actual_size,
            "duration_ms": _duration_ms(started),
        }
        entry["verification"] = {"status": "not_verified"}
        self._write_state(state)
        self._log_event(
            action="download_update",
            status="downloaded",
            details={
                "update_id": update_id,
                "local_path": str(inbox_path),
                "size_bytes": actual_size,
            },
        )
        return self._decorate_update_row(entry)

    def verify_update(self, *, update_id: str) -> dict[str, Any]:
        self.ensure_ready()
        state = self._load_state()
        entry = self._get_update_entry(state, update_id)
        local_path = Path(str(entry.get("download", {}).get("local_path") or ""))
        if not local_path.exists() or not local_path.is_file():
            raise ValueError("downloaded update file is missing")
        expected_sha = str(entry["manifest"].get("sha256") or "").strip().lower()
        actual_sha = _sha256_for_path(local_path)
        if actual_sha != expected_sha:
            failed_path = self.failed_dir / local_path.name
            failed_path = self._move_with_unique_name(local_path, failed_path)
            entry["download"]["status"] = "failed"
            entry["download"]["local_path"] = str(failed_path)
            entry["verification"] = {
                "status": "checksum_mismatch",
                "verified_at": _utc_now(),
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
            self._write_state(state)
            self._log_event(
                action="verify_update",
                status="checksum_mismatch",
                details={
                    "update_id": update_id,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "moved_to": str(failed_path),
                },
            )
            return self._decorate_update_row(entry)

        entry["verification"] = {
            "status": "verified",
            "verified_at": _utc_now(),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
        }
        self._write_state(state)
        self._log_event(
            action="verify_update",
            status="verified",
            details={"update_id": update_id, "local_path": str(local_path)},
        )
        return self._decorate_update_row(entry)

    def apply_update(self, *, update_id: str) -> dict[str, Any]:
        self.ensure_ready()
        state = self._load_state()
        entry = self._get_update_entry(state, update_id)
        if entry.get("apply", {}).get("status") == "review_staged":
            return self._decorate_update_row(entry)
        if entry.get("verification", {}).get("status") != "verified":
            entry["apply"] = {
                "status": "blocked_verification_required",
                "applied_at": _utc_now(),
                "notes": "Checksum verification must succeed before review-stage apply.",
            }
            self._write_state(state)
            self._log_event(
                action="apply_update",
                status="blocked",
                details={"update_id": update_id, "reason": "verification_required"},
            )
            return self._decorate_update_row(entry)

        local_path = Path(str(entry.get("download", {}).get("local_path") or ""))
        if not local_path.exists() or not local_path.is_file():
            raise ValueError("verified update file is missing")

        review_dir = self.processed_dir / update_id
        review_dir.mkdir(parents=True, exist_ok=True)
        backup_dir: Path | None = None
        existing_files = [item for item in review_dir.iterdir() if item.is_file()]
        if existing_files:
            backup_dir = review_dir / f"backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for item in existing_files:
                shutil.move(str(item), str(backup_dir / item.name))

        staged_asset_path = review_dir / local_path.name
        shutil.move(str(local_path), str(staged_asset_path))
        manifest_path = review_dir / "update_manifest.json"
        manifest_path.write_text(
            json.dumps(entry["manifest"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        receipt_path = review_dir / "review_receipt.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "update_id": update_id,
                    "applied_at": _utc_now(),
                    "status": "review_staged",
                    "notes": "Update staged for explicit human review. No live corpus/app files were overwritten automatically.",
                    "requires_index_rebuild": bool(entry["manifest"].get("requires_index_rebuild")),
                    "backup_path": str(backup_dir) if backup_dir else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        entry["download"]["status"] = "processed"
        entry["download"]["local_path"] = str(staged_asset_path)
        entry["apply"] = {
            "status": "review_staged",
            "applied_at": _utc_now(),
            "processed_path": str(review_dir),
            "backup_path": str(backup_dir) if backup_dir else None,
            "notes": "Update staged for review. Manual inspection remains required before any live install.",
        }
        self._write_state(state)
        self._log_event(
            action="apply_update",
            status="review_staged",
            details={
                "update_id": update_id,
                "processed_path": str(review_dir),
                "requires_index_rebuild": bool(entry["manifest"].get("requires_index_rebuild")),
            },
        )
        return self._decorate_update_row(entry)

    def history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if not self.update_log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            lines = self.update_log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= limit:
                break
        return rows

    def _default_state(self) -> dict[str, Any]:
        return {
            "last_check_at": None,
            "last_check_status": "never_checked",
            "last_check_error": None,
            "last_check_duration_ms": 0,
            "last_manifest_url": None,
            "last_manifest_archive_path": None,
            "updates": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        if not isinstance(payload, dict):
            return self._default_state()
        payload.setdefault("updates", {})
        return payload

    def _write_state(self, payload: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_update_entry(self, state: dict[str, Any], update_id: str) -> dict[str, Any]:
        updates = state.setdefault("updates", {})
        entry = updates.get(update_id)
        if not isinstance(entry, dict):
            raise ValueError(f"unknown update_id: {update_id}")
        return entry

    def _decorate_update_row(self, entry: dict[str, Any]) -> dict[str, Any]:
        manifest = dict(entry.get("manifest") or {})
        download = dict(entry.get("download") or {})
        verification = dict(entry.get("verification") or {})
        apply_state = dict(entry.get("apply") or {})
        local_status = str(
            apply_state.get("status")
            or verification.get("status")
            or download.get("status")
            or "not_downloaded"
        )
        if local_status == "not_applied" and verification.get("status") == "verified":
            local_status = "verified"
        elif local_status == "not_applied" and download.get("status") == "downloaded":
            local_status = "downloaded"
        return {
            "update_id": str(entry.get("update_id") or manifest.get("update_id") or ""),
            "manifest": manifest,
            "download": download,
            "verification": verification,
            "apply": apply_state,
            "local_status": local_status,
            "local_status_sort": _local_status_sort_key(local_status),
        }

    def _unique_path(self, directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 2
        while True:
            alt = directory / f"{stem}-{counter}{suffix}"
            if not alt.exists():
                return alt
            counter += 1

    def _move_with_unique_name(self, source: Path, destination: Path) -> Path:
        target = self._unique_path(destination.parent, destination.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return target

    def _log_event(self, *, action: str, status: str, details: dict[str, Any]) -> None:
        self.update_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": _utc_now(),
            "action": action,
            "status": status,
            "details": details,
        }
        with self.update_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_manifest_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("updates")
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("update manifest must contain an 'updates' array or be a list")
    normalized: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("each update manifest entry must be an object")
        entry = {
            "update_id": _required_string(raw, "update_id"),
            "type": _required_string(raw, "type"),
            "version": _required_string(raw, "version"),
            "file_url": _required_string(raw, "file_url"),
            "sha256": _required_string(raw, "sha256").lower(),
            "size_bytes": _required_int(raw, "size_bytes"),
            "description": _required_string(raw, "description"),
            "requires_index_rebuild": bool(raw.get("requires_index_rebuild")),
        }
        if entry["type"] not in UPDATE_TYPES:
            raise ValueError(f"unsupported update type: {entry['type']}")
        if len(entry["sha256"]) != 64:
            raise ValueError(f"invalid sha256 for update {entry['update_id']}")
        parsed = urlparse(entry["file_url"])
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"file_url must use http or https for update {entry['update_id']}")
        normalized.append(entry)
    return normalized


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"update manifest field '{key}' is required")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"update manifest field '{key}' must be an integer") from exc
    if number < 0:
        raise ValueError(f"update manifest field '{key}' must be non-negative")
    return number


def _safe_filename_from_url(url: str, *, fallback: str) -> str:
    path = urlparse(url).path or ""
    candidate = Path(path).name or fallback
    return _sanitize_filename(candidate) or _sanitize_filename(fallback) or "update.bin"


def _sanitize_filename(value: str) -> str:
    safe = "".join(
        character
        for character in str(value or "")
        if character.isalnum() or character in {"-", "_", "."}
    )
    return safe[:180]


def _is_safe_extension(extension: str, update_type: str) -> bool:
    allowed = SAFE_EXTENSIONS_BY_TYPE.get(str(update_type or "").strip(), set())
    return extension in allowed


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 128), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _local_status_sort_key(status: str) -> int:
    order = {
        "review_staged": 0,
        "verified": 1,
        "downloaded": 2,
        "processed": 3,
        "blocked_verification_required": 4,
        "not_downloaded": 5,
        "not_verified": 6,
        "failed": 7,
        "checksum_mismatch": 8,
        "quarantined": 9,
    }
    return order.get(str(status or "").strip(), 99)


def _duration_ms(started: datetime) -> int:
    return int((datetime.now(UTC) - started).total_seconds() * 1000)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
