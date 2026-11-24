"""Immutable FastAPI application state container and dependency provider (D-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from backend.app.api.services.preset_service import PresetStore
from backend.app.pdi.config import PDIConfig


@dataclass(frozen=True, slots=True)
class AppState:
    """Startup-cached read-only runtime state shared across FastAPI route handlers."""

    bundle: dict[str, Any]
    pdi_config: PDIConfig
    pdi_config_hash: str
    preset_store: PresetStore
    metrics_payload: dict[str, Any]


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency returning the initialized ``AppState`` from ``app.state.context``."""
    state = getattr(request.app.state, "context", None)
    if not isinstance(state, AppState):
        raise RuntimeError("Application state context is not initialized.")
    return state


__all__ = [
    "AppState",
    "get_app_state",
]
