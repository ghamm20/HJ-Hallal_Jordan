"""Minimal Ollama client for schema-guided chat output."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import requests


class OllamaClientProtocol(Protocol):
    def resolve_model(self, requested_model: str | None = None) -> str:
        """Resolve the final model name."""

    def chat_json(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON object from the model."""


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        timeout_seconds: int = 90,
    ) -> None:
        configured = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.host = _normalize_ollama_host(configured)
        self.timeout_seconds = timeout_seconds

    def resolve_model(self, requested_model: str | None = None) -> str:
        if requested_model:
            return requested_model
        env_model = os.getenv("HALAL_JORDAN_OLLAMA_MODEL")
        if env_model:
            return env_model
        models = self._list_models()
        for candidate in ("qwen3:8b", "deepseek-r1:8b", "qwen2.5-coder:7b"):
            if candidate in models:
                return candidate
        if not models:
            raise RuntimeError("no local Ollama models are available")
        return models[0]

    def chat_json(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_options = {"temperature": 0.2}
        if options:
            request_options.update(options)
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": request_options,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["message"]["content"]
        return _parse_json_object(content)

    def _list_models(self) -> list[str]:
        response = requests.get(
            f"{self.host}/api/tags",
            timeout=min(self.timeout_seconds, 10),
        )
        response.raise_for_status()
        payload = response.json()
        return [item["name"] for item in payload.get("models", [])]


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("expected model output to be a JSON object")
    return parsed


def _normalize_ollama_host(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return "http://127.0.0.1:11434"
    if "://" not in normalized:
        return "http://" + normalized
    return normalized
