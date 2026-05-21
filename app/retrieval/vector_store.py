"""Lightweight local vector index with numpy and optional FAISS support."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.retrieval.embedder import LocalEmbedder


def vector_chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id", "") or "").strip()


@dataclass(slots=True)
class VectorStoreStatus:
    available: bool
    loaded: bool
    backend: str
    index_path: str
    chunk_count: int
    dimension: int
    reason: str


class LocalVectorStore:
    """Persist, load, rebuild, and query chunk vectors."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._numpy_path = (
            index_path
            if index_path.suffix.lower() == ".npz"
            else index_path.with_suffix(".npz")
        )
        self._faiss_path = self._numpy_path.with_suffix(".faiss")
        self._faiss_meta_path = self._faiss_path.with_suffix(".faiss.meta.json")
        self._chunk_ids: list[str] = []
        self._embeddings: Any | None = None
        self._index: Any | None = None
        self._numpy: Any | None = None
        self._faiss: Any | None = None
        self._backend = "numpy"
        self._status = VectorStoreStatus(
            available=False,
            loaded=False,
            backend="numpy",
            index_path=str(self._numpy_path),
            chunk_count=0,
            dimension=0,
            reason="uninitialized",
        )

    def status(self) -> dict[str, Any]:
        return {
            "available": self._status.available,
            "loaded": self._status.loaded,
            "backend": self._status.backend,
            "index_path": self._status.index_path,
            "chunk_count": self._status.chunk_count,
            "dimension": self._status.dimension,
            "reason": self._status.reason,
            "faiss_path": str(self._faiss_path),
            "numpy_path": str(self._numpy_path),
        }

    def load(self) -> bool:
        if self._status.loaded and self._status.available:
            return True
        if self._try_load_faiss():
            return True
        if self._try_load_numpy():
            return True
        return False

    def rebuild(
        self,
        *,
        chunks: list[dict[str, Any]],
        embedder: LocalEmbedder,
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("chunks must not be empty")
        embeddings = embedder.encode_chunks(chunks)
        chunk_ids = [vector_chunk_id(chunk) for chunk in chunks]
        if not all(chunk_ids):
            raise ValueError("all chunks must contain stable chunk_id values")
        numpy_module = self._require_numpy()
        embeddings = numpy_module.asarray(embeddings, dtype="float32")
        self._chunk_ids = chunk_ids
        self._embeddings = embeddings
        self._backend = "faiss" if self._faiss_available() else "numpy"
        if self._backend == "faiss":
            self._save_faiss()
        else:
            self._save_numpy()
        self._status = VectorStoreStatus(
            available=True,
            loaded=True,
            backend=self._backend,
            index_path=str(self._faiss_path if self._backend == "faiss" else self._numpy_path),
            chunk_count=len(self._chunk_ids),
            dimension=int(embeddings.shape[1]),
            reason="ready",
        )
        return self.status()

    def search(
        self,
        *,
        query_vector: Any,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self.load():
            raise RuntimeError(self._status.reason)
        if top_k <= 0 or not self._chunk_ids:
            return []
        numpy_module = self._require_numpy()
        query = numpy_module.asarray(query_vector, dtype="float32").reshape(1, -1)
        if self._backend == "faiss" and self._index is not None:
            scores, indices = self._index.search(query, min(top_k, len(self._chunk_ids)))
            return _collect_matches(
                chunk_ids=self._chunk_ids,
                scores=scores[0].tolist(),
                indices=indices[0].tolist(),
            )
        query = _normalize_rows(numpy_module, query)
        embeddings = _normalize_rows(numpy_module, self._embeddings)
        scores = numpy_module.matmul(embeddings, query.T).reshape(-1)
        limit = min(top_k, scores.shape[0])
        ranked = numpy_module.argsort(scores)[::-1][:limit]
        return [
            {
                "chunk_id": self._chunk_ids[int(index)],
                "score": float(scores[int(index)]),
            }
            for index in ranked
        ]

    def _save_numpy(self) -> None:
        numpy_module = self._require_numpy()
        self._numpy_path.parent.mkdir(parents=True, exist_ok=True)
        numpy_module.savez_compressed(
            self._numpy_path,
            chunk_ids=numpy_module.asarray(self._chunk_ids, dtype=object),
            embeddings=numpy_module.asarray(self._embeddings, dtype="float32"),
            backend="numpy",
            generated_at=_utc_now(),
        )

    def _try_load_numpy(self) -> bool:
        if not self._numpy_path.exists():
            self._status = VectorStoreStatus(
                available=False,
                loaded=False,
                backend="numpy",
                index_path=str(self._numpy_path),
                chunk_count=0,
                dimension=0,
                reason="vector_index_missing",
            )
            return False
        numpy_module = self._require_numpy()
        try:
            payload = numpy_module.load(self._numpy_path, allow_pickle=True)
            chunk_ids = [str(item) for item in payload["chunk_ids"].tolist()]
            embeddings = numpy_module.asarray(payload["embeddings"], dtype="float32")
        except Exception:
            self._status = VectorStoreStatus(
                available=False,
                loaded=False,
                backend="numpy",
                index_path=str(self._numpy_path),
                chunk_count=0,
                dimension=0,
                reason="vector_index_load_failed",
            )
            return False
        self._backend = "numpy"
        self._chunk_ids = chunk_ids
        self._embeddings = embeddings
        self._status = VectorStoreStatus(
            available=True,
            loaded=True,
            backend="numpy",
            index_path=str(self._numpy_path),
            chunk_count=len(chunk_ids),
            dimension=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            reason="ready",
        )
        return True

    def _save_faiss(self) -> None:
        numpy_module = self._require_numpy()
        faiss_module = self._require_faiss()
        embeddings = _normalize_rows(
            numpy_module,
            numpy_module.asarray(self._embeddings, dtype="float32"),
        )
        index = faiss_module.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        self._faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss_module.write_index(index, str(self._faiss_path))
        self._faiss_meta_path.write_text(
            json.dumps(
                {
                    "chunk_ids": self._chunk_ids,
                    "generated_at": _utc_now(),
                    "dimension": int(embeddings.shape[1]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._index = index

    def _try_load_faiss(self) -> bool:
        if not self._faiss_available():
            return False
        if not (self._faiss_path.exists() and self._faiss_meta_path.exists()):
            return False
        faiss_module = self._require_faiss()
        try:
            meta = json.loads(self._faiss_meta_path.read_text(encoding="utf-8"))
            chunk_ids = [str(item) for item in meta.get("chunk_ids", [])]
            index = faiss_module.read_index(str(self._faiss_path))
        except Exception:
            return False
        self._backend = "faiss"
        self._chunk_ids = chunk_ids
        self._index = index
        self._status = VectorStoreStatus(
            available=True,
            loaded=True,
            backend="faiss",
            index_path=str(self._faiss_path),
            chunk_count=len(chunk_ids),
            dimension=int(index.d),
            reason="ready",
        )
        return True

    def _faiss_available(self) -> bool:
        return importlib.util.find_spec("faiss") is not None

    def _require_numpy(self) -> Any:
        if self._numpy is not None:
            return self._numpy
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError("numpy_missing")
        import numpy

        self._numpy = numpy
        return numpy

    def _require_faiss(self) -> Any:
        if self._faiss is not None:
            return self._faiss
        if not self._faiss_available():
            raise RuntimeError("faiss_missing")
        import faiss

        self._faiss = faiss
        return faiss


def inspect_vector_index(index_path: Path) -> dict[str, Any]:
    store = LocalVectorStore(index_path)
    loaded = store.load()
    return {
        **store.status(),
        "exists": bool(store._numpy_path.exists() or store._faiss_path.exists()),
        "loaded": loaded,
    }


def build_vector_index(
    *,
    chunks: list[dict[str, Any]],
    embedder: LocalEmbedder,
    index_path: Path,
) -> dict[str, Any]:
    started = perf_counter()
    store = LocalVectorStore(index_path)
    status = store.rebuild(chunks=chunks, embedder=embedder)
    elapsed = perf_counter() - started
    resolved_path = Path(str(status["index_path"]))
    size_bytes = resolved_path.stat().st_size if resolved_path.exists() else 0
    return {
        "chunk_count": len(chunks),
        "embedding_time_seconds": round(elapsed, 4),
        "index_size_bytes": size_bytes,
        "backend": status["backend"],
        "index_path": status["index_path"],
        "model_name": embedder.model_name,
        "device": embedder.device,
    }


def _normalize_rows(numpy_module: Any, matrix: Any) -> Any:
    norms = numpy_module.linalg.norm(matrix, axis=1, keepdims=True)
    norms = numpy_module.where(norms == 0, 1.0, norms)
    return matrix / norms


def _collect_matches(
    *,
    chunk_ids: list[str],
    scores: list[float],
    indices: list[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, score in zip(indices, scores, strict=False):
        if index < 0 or index >= len(chunk_ids):
            continue
        results.append({"chunk_id": chunk_ids[index], "score": float(score)})
    return results


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
