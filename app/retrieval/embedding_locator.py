"""Locate a bundled HuggingFace model snapshot for true offline use.

On Windows the HuggingFace cache stores model files directly in
``snapshots/<commit>/`` (no symlinks, because the OS doesn't grant
symlink permission to ordinary users). The cache layout still
*works* online, but offline cache lookups expect blob files and find
none — so SentenceTransformer happily tries to download.

This module detects the bundled snapshot directory and returns its
absolute path so we can hand it directly to SentenceTransformer,
bypassing the cache resolver entirely. Charter rule honored: the
system runs offline with zero cloud dependence.
"""

from __future__ import annotations

from pathlib import Path


def hf_cache_root(repo_root: Path | None = None) -> Path:
    """Project-bundled HF cache root."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "runtime" / "huggingface"


def repo_cache_dir(model_name: str, *, repo_root: Path | None = None) -> Path:
    """The ``models--owner--name`` directory inside the bundled cache."""

    safe = model_name.strip("/").replace("/", "--")
    return hf_cache_root(repo_root) / f"models--{safe}"


def find_bundled_snapshot(
    model_name: str,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Return the absolute path to a usable bundled snapshot, or None.

    A snapshot is "usable" when its directory exists AND contains
    ``config.json`` (the minimum HF identity file). When multiple
    snapshots are present, returns the lexicographically last one
    (HF commit hashes sort stable so this picks the latest).

    Returns None when no bundled snapshot is available — callers
    should fall back to the model name and let the HF library
    resolve from its normal cache.
    """

    cache_dir = repo_cache_dir(model_name, repo_root=repo_root)
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        reverse=True,
    )
    for snapshot in candidates:
        if (snapshot / "config.json").is_file():
            return snapshot.resolve()
    return None


def resolve_embedding_model_path(
    model_name: str,
    *,
    repo_root: Path | None = None,
) -> str:
    """Return either an absolute bundled-snapshot path or the original
    model name. SentenceTransformer accepts both — a local path skips
    network resolution entirely.
    """

    bundled = find_bundled_snapshot(model_name, repo_root=repo_root)
    if bundled is not None:
        return str(bundled)
    return model_name
