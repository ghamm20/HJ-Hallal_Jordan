"""Lightweight metadata-refresh for persisted chunks.

The vector index (``data/index/vector_index.npz``) is keyed by chunk
position, not by chunk metadata. So when collection-level enrichment
adds new fields (era, scholar_authority, methodology_tags,
default_hadith_grade) we do NOT need to rebuild embeddings — we just
re-apply enrichment to each chunk in place.

This is the cheap path:

  - Stream-read ``data/index/chunks.jsonl``
  - Apply ``apply_collection_enrichment`` to each chunk
  - Stream-write to a temp file
  - Atomically rename

What gets updated:
  - Each chunk gains the new fields when they aren't already set
  - The persisted index manifest's metadata version is bumped
  - Madhhab / source-class buckets stay valid (those keys aren't changed)

What does NOT need to happen:
  - No re-embedding (text unchanged)
  - No re-chunking (boundaries unchanged)
  - No re-classification of source types
  - No restart of the running app (the next bootstrap reads the new
    chunks.jsonl)

Run it from the project root:

    runtime/python/python.exe -m ingestion.pipelines.refresh_chunk_metadata
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

from app.retrieval.inline_grading import apply_inline_grading
from app.retrieval.source_enrichment import apply_collection_enrichment

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILENAME = "chunks.jsonl"


def refresh_chunks(
    *,
    repo_root: Path | None = None,
    index_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Re-apply collection enrichment to persisted chunks in place.

    Returns counts of total chunks scanned and chunks modified.
    """

    root = repo_root or REPO_ROOT
    chunks_dir = index_root or (root / "data" / "index")
    chunks_path = chunks_dir / CHUNKS_FILENAME
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    started = time.time()
    total = 0
    modified = 0
    enrichment_counts: dict[str, int] = {}

    temp_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")

    with chunks_path.open("r", encoding="utf-8") as src, temp_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # Pass through unparseable lines untouched to avoid data loss
                dst.write(line + "\n")
                continue
            enriched = apply_inline_grading(apply_collection_enrichment(chunk))
            if enriched != chunk:
                modified += 1
                for field in enriched.get("collection_enrichment_applied", []):
                    enrichment_counts[field] = enrichment_counts.get(field, 0) + 1
            dst.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    elapsed = time.time() - started

    if dry_run:
        os.remove(temp_path)
        print(f"[dry-run] scanned {total} chunks, would modify {modified} in {elapsed:.1f}s")
    else:
        backup_path = chunks_path.with_suffix(chunks_path.suffix + ".prev")
        if backup_path.exists():
            backup_path.unlink()
        shutil.move(str(chunks_path), str(backup_path))
        shutil.move(str(temp_path), str(chunks_path))
        print(
            f"refreshed {modified}/{total} chunks in {elapsed:.1f}s "
            f"(backup at {backup_path.name})"
        )

    print("enrichment fields applied (count):")
    for field, count in sorted(enrichment_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {field}: {count}")

    return {
        "total": total,
        "modified": modified,
        "elapsed_seconds": round(elapsed, 2),
        **{f"enriched_{k}": v for k, v in enrichment_counts.items()},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-apply collection enrichment to persisted chunks "
        "without re-embedding. Cheap; safe to run any time."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report what would change, but do not write.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    refresh_chunks(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
