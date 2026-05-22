"""Tests for the bundled embedding model locator.

Pins the charter rule that Halal Jordan runs offline from a thumbdrive:
when a bundled snapshot is present, the embedder must use it directly
instead of going through the HuggingFace cache resolver (which fails
offline on Windows because symlinks aren't available).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.embedding_locator import (  # noqa: E402
    find_bundled_snapshot,
    hf_cache_root,
    repo_cache_dir,
    resolve_embedding_model_path,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_hf_cache_root_resolves_to_runtime_huggingface():
    root = hf_cache_root()
    assert root.name == "huggingface"
    assert root.parent.name == "runtime"


def test_repo_cache_dir_uses_hf_naming_convention():
    """HuggingFace's on-disk cache layout: owner/name becomes
    ``models--owner--name`` (no spaces, no slashes).
    """

    path = repo_cache_dir("sentence-transformers/all-MiniLM-L6-v2")
    assert path.name == "models--sentence-transformers--all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# find_bundled_snapshot
# ---------------------------------------------------------------------------


def test_returns_none_for_missing_cache(tmp_path):
    assert find_bundled_snapshot("nonexistent/model", repo_root=tmp_path) is None


def test_returns_none_when_snapshots_dir_missing(tmp_path):
    cache = repo_cache_dir("test/model", repo_root=tmp_path)
    cache.mkdir(parents=True)
    assert find_bundled_snapshot("test/model", repo_root=tmp_path) is None


def test_returns_none_when_snapshot_lacks_config_json(tmp_path):
    snap = repo_cache_dir("test/model", repo_root=tmp_path) / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    # Snapshot exists but no config.json -> not a usable model
    assert find_bundled_snapshot("test/model", repo_root=tmp_path) is None


def test_returns_snapshot_when_present(tmp_path):
    snap = repo_cache_dir("test/model", repo_root=tmp_path) / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "bert"}))
    result = find_bundled_snapshot("test/model", repo_root=tmp_path)
    assert result is not None
    assert result == snap.resolve()


def test_prefers_latest_snapshot_when_multiple(tmp_path):
    """When multiple commit snapshots exist, the lexicographically
    last one wins — commit hashes sort stable so this picks the latest.
    """

    cache = repo_cache_dir("test/model", repo_root=tmp_path)
    older = cache / "snapshots" / "aaa111"
    newer = cache / "snapshots" / "zzz999"
    for snap in (older, newer):
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
    result = find_bundled_snapshot("test/model", repo_root=tmp_path)
    assert result == newer.resolve()


# ---------------------------------------------------------------------------
# resolve_embedding_model_path — what the embedder calls
# ---------------------------------------------------------------------------


def test_resolve_returns_local_path_when_bundled(tmp_path):
    snap = repo_cache_dir("test/model", repo_root=tmp_path) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    result = resolve_embedding_model_path("test/model", repo_root=tmp_path)
    # Returns an absolute path that exists
    assert Path(result).is_dir()
    assert Path(result).resolve() == snap.resolve()


def test_resolve_falls_back_to_model_name_when_not_bundled(tmp_path):
    """When no bundled snapshot is available, return the model name
    unchanged so the HF library can resolve via its normal cache.
    """

    result = resolve_embedding_model_path("not/bundled", repo_root=tmp_path)
    assert result == "not/bundled"


# ---------------------------------------------------------------------------
# Real bundled model — must be present in the shipped repo
# ---------------------------------------------------------------------------


def test_shipped_minilm_snapshot_is_present():
    """The MiniLM-L6-v2 embedding model must ship with the project
    for true offline / thumbdrive operation. This test fails loudly if
    someone deletes runtime/huggingface/ from the bundle.
    """

    snapshot = find_bundled_snapshot("sentence-transformers/all-MiniLM-L6-v2")
    assert snapshot is not None, (
        "Bundled MiniLM model not found. The project must ship its own "
        "HuggingFace snapshot at runtime/huggingface/ for offline / "
        "thumbdrive use. Re-download or rebuild the bundle."
    )
    # And the snapshot has the expected weight files
    weight_files = list(snapshot.glob("model.safetensors")) + list(snapshot.glob("pytorch_model.bin"))
    assert weight_files, f"Snapshot at {snapshot} has no weight file"
