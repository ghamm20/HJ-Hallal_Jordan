"""Compatibility alias for the canonical backend entrypoint."""

from app.backend.main import app, create_app

__all__ = ["app", "create_app"]
