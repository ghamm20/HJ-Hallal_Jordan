"""Tests for the lightweight chunk metadata refresh script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.pipelines.refresh_chunk_metadata import refresh_chunks  # noqa: E402


def _write_chunks(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_chunks(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@pytest.fixture
def fake_index(tmp_path):
    repo = tmp_path
    index_dir = repo / "data" / "index"
    index_dir.mkdir(parents=True)
    chunks_path = index_dir / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {"chunk_id": 1, "collection": "Sahih al-Bukhari", "text": "..."},
            {"chunk_id": 2, "collection": "Sunan Abu Dawud", "text": "..."},
            {"chunk_id": 3, "collection": "Minhaj al-Abidin", "text": "..."},
            {"chunk_id": 4, "collection": "Random Unknown Book", "text": "..."},
            {"chunk_id": 5, "collection": "Sahih al-Bukhari", "hadith_grade": "daif", "text": "..."},
        ],
    )
    return repo, index_dir


def test_refresh_attaches_enrichment_to_bukhari_chunks(fake_index):
    repo, index_dir = fake_index
    refresh_chunks(repo_root=repo, index_root=index_dir)
    chunks = _read_chunks(index_dir / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    assert by_id[1]["hadith_grade"] == "sahih"
    assert by_id[1]["era"] == "classical"


def test_refresh_does_not_inflate_mixed_grade_collections(fake_index):
    repo, index_dir = fake_index
    refresh_chunks(repo_root=repo, index_root=index_dir)
    chunks = _read_chunks(index_dir / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    # Abu Dawud must NOT get a sahih default
    assert by_id[2].get("hadith_grade", "") == ""
    # But it does get era
    assert by_id[2]["era"] == "classical"


def test_refresh_preserves_explicit_metadata(fake_index):
    """Bukhari chunk that already carries an explicit daif grade keeps
    it — enrichment never overrides explicit values, even on refresh.
    """

    repo, index_dir = fake_index
    refresh_chunks(repo_root=repo, index_root=index_dir)
    chunks = _read_chunks(index_dir / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    assert by_id[5]["hadith_grade"] == "daif"


def test_refresh_leaves_unknown_collections_unmodified(fake_index):
    repo, index_dir = fake_index
    refresh_chunks(repo_root=repo, index_root=index_dir)
    chunks = _read_chunks(index_dir / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    assert "era" not in by_id[4] or not by_id[4].get("era")
    assert "collection_enrichment_applied" not in by_id[4]


def test_refresh_writes_backup(fake_index):
    repo, index_dir = fake_index
    refresh_chunks(repo_root=repo, index_root=index_dir)
    assert (index_dir / "chunks.jsonl.prev").exists()


def test_dry_run_does_not_modify_file(fake_index):
    repo, index_dir = fake_index
    original = (index_dir / "chunks.jsonl").read_text(encoding="utf-8")
    refresh_chunks(repo_root=repo, index_root=index_dir, dry_run=True)
    after = (index_dir / "chunks.jsonl").read_text(encoding="utf-8")
    assert original == after
    assert not (index_dir / "chunks.jsonl.prev").exists()


def test_refresh_passes_through_malformed_lines(fake_index, tmp_path):
    repo, index_dir = fake_index
    chunks_path = index_dir / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    # Must not raise
    refresh_chunks(repo_root=repo, index_root=index_dir)
    # Malformed line preserved verbatim
    content = chunks_path.read_text(encoding="utf-8")
    assert "{not valid json" in content
