"""Lazy local embedding support for semantic retrieval."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


def build_embedding_text(chunk: dict[str, Any]) -> str:
    parts = [
        str(chunk.get("title", "") or ""),
        str(chunk.get("collection", "") or ""),
        str(chunk.get("author", "") or ""),
        str(chunk.get("madhhab", "") or ""),
        str(chunk.get("reference", "") or ""),
        str(chunk.get("section_label", "") or ""),
        str(chunk.get("text", "") or chunk.get("quote", "") or ""),
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class EmbeddingBackendStatus:
    enabled: bool
    model_name: str
    device: str
    backend: str
    dependency_available: bool
    model_loaded: bool
    available: bool
    reason: str


class LocalEmbedder:
    """Wrap SentenceTransformer with honest lazy loading and failure reporting."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        enabled: bool = True,
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.device = str(device or "cpu").strip().lower() or "cpu"
        self.enabled = bool(enabled)
        self._model: Any | None = None
        self._numpy: Any | None = None
        self._status = EmbeddingBackendStatus(
            enabled=self.enabled,
            model_name=self.model_name,
            device=self.device,
            backend="sentence_transformers",
            dependency_available=False,
            model_loaded=False,
            available=False,
            reason="embedding_disabled" if not self.enabled else "uninitialized",
        )

    def status(self) -> dict[str, Any]:
        payload = asdict(self._status)
        payload["model_name"] = self.model_name
        payload["device"] = self.device
        payload["enabled"] = self.enabled
        if self.enabled and not payload["model_loaded"]:
            payload["dependency_available"] = self._dependency_available()
            if (
                payload["reason"] == "uninitialized"
                and not payload["dependency_available"]
            ):
                payload["reason"] = "sentence_transformers_missing"
        return payload

    def encode_chunks(self, chunks: list[dict[str, Any]]) -> Any:
        return self.encode_texts([build_embedding_text(chunk) for chunk in chunks])

    def encode_query(self, question: str) -> Any:
        vector = self.encode_texts([str(question or "").strip()])
        return vector[0]

    def encode_texts(self, texts: list[str]) -> Any:
        if not texts:
            numpy_module = self._require_numpy()
            return numpy_module.zeros((0, 0), dtype="float32")
        model = self._require_model()
        embeddings = model.encode(
            texts,
            batch_size=min(32, max(1, len(texts))),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        numpy_module = self._require_numpy()
        return numpy_module.asarray(embeddings, dtype="float32")

    def _require_model(self) -> Any:
        if not self.enabled:
            self._status = EmbeddingBackendStatus(
                enabled=False,
                model_name=self.model_name,
                device=self.device,
                backend="sentence_transformers",
                dependency_available=False,
                model_loaded=False,
                available=False,
                reason="embedding_disabled",
            )
            raise RuntimeError("embedding_disabled")
        if self._model is not None:
            return self._model
        numpy_module = self._require_numpy()
        if not self._dependency_available():
            self._status = EmbeddingBackendStatus(
                enabled=True,
                model_name=self.model_name,
                device=self.device,
                backend="sentence_transformers",
                dependency_available=False,
                model_loaded=False,
                available=False,
                reason="sentence_transformers_missing",
            )
            raise RuntimeError("sentence_transformers_missing")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - import error path
            self._status = EmbeddingBackendStatus(
                enabled=True,
                model_name=self.model_name,
                device=self.device,
                backend="sentence_transformers",
                dependency_available=False,
                model_loaded=False,
                available=False,
                reason=f"sentence_transformers_import_failed:{exc.__class__.__name__}",
            )
            raise RuntimeError(self._status.reason) from exc
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            self._status = EmbeddingBackendStatus(
                enabled=True,
                model_name=self.model_name,
                device=self.device,
                backend="sentence_transformers",
                dependency_available=True,
                model_loaded=False,
                available=False,
                reason=f"embedding_model_load_failed:{exc.__class__.__name__}",
            )
            raise RuntimeError(self._status.reason) from exc
        self._numpy = numpy_module
        self._status = EmbeddingBackendStatus(
            enabled=True,
            model_name=self.model_name,
            device=self.device,
            backend="sentence_transformers",
            dependency_available=True,
            model_loaded=True,
            available=True,
            reason="ready",
        )
        return self._model

    def _require_numpy(self) -> Any:
        if self._numpy is not None:
            return self._numpy
        if importlib.util.find_spec("numpy") is None:
            self._status = EmbeddingBackendStatus(
                enabled=self.enabled,
                model_name=self.model_name,
                device=self.device,
                backend="sentence_transformers",
                dependency_available=False,
                model_loaded=False,
                available=False,
                reason="numpy_missing",
            )
            raise RuntimeError("numpy_missing")
        import numpy

        self._numpy = numpy
        return numpy

    def _dependency_available(self) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None
