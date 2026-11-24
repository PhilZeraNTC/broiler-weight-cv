"""FastAPI application factory, lifespan boot, localhost CORS & guarded SPA fallback (D-01, D-05).

Launch locally via::

    uv run uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.dependencies import AppState
from backend.app.api.routers import (
    health_router,
    metrics_router,
    predict_router,
    presets_router,
    watershed_router,
)
from backend.app.api.services.metrics_service import load_metrics_payload
from backend.app.api.services.preset_service import build_preset_store
from backend.app.config import FRONTEND_DIST_DIR, MODEL_BUNDLE_PATH
from backend.app.modeling.artifacts import load_model_bundle
from backend.app.pdi.config import PDIConfig

logger = logging.getLogger(__name__)

LOCALHOST_CORS_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)

_ERR_INTERNAL_SANITIZED: str = (
    "Não foi possível concluir a análise devido a um erro interno local. "
    "Verifique se o servidor Uvicorn está ativo no terminal e clique em "
    "'Tentar Análise Novamente'."
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load locked model bundle, audit JSONs, preset allowlist, and thumbnails once at boot."""
    cv2.setNumThreads(2)
    expected_hash = PDIConfig().config_hash()
    bundle = load_model_bundle(MODEL_BUNDLE_PATH, expected_pdi_config_hash=expected_hash)
    pdi_config = PDIConfig(**bundle["pdi_config"])
    preset_store = build_preset_store(expected_pdi_config_hash=expected_hash)
    metrics_payload = load_metrics_payload(expected_pdi_config_hash=expected_hash)

    application.state.context = AppState(
        bundle=bundle,
        pdi_config=pdi_config,
        pdi_config_hash=expected_hash,
        preset_store=preset_store,
        metrics_payload=metrics_payload,
    )
    yield


def create_app(frontend_dist_dir: Path | None = None) -> FastAPI:
    """Create the configured AvianScale CV FastAPI application instance."""
    application = FastAPI(
        title="AvianScale CV API",
        version="1.0.0",
        description=(
            "API local de PDI clássico calibrado por óptica EXIF e regressão tabular "
            "para estimativa não invasiva de peso vivo de frangos de corte."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(LOCALHOST_CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        # Log only sanitized class & path prefix — never raw filenames or stack traces in response.
        logger.error(
            "Unhandled API error (%s) on route path %s",
            type(exc).__name__,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": _ERR_INTERNAL_SANITIZED},
        )

    # Register all /api/* routers strictly BEFORE optional static SPA fallback (D-05).
    application.include_router(health_router)
    application.include_router(presets_router)
    application.include_router(predict_router)
    application.include_router(watershed_router)
    application.include_router(metrics_router)

    dist_dir = Path(frontend_dist_dir) if frontend_dist_dir is not None else FRONTEND_DIST_DIR
    index_html = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if index_html.is_file():
        if assets_dir.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse:
            normalized = full_path.lstrip("/")
            if normalized == "api" or normalized.startswith("api/"):
                raise HTTPException(
                    status_code=404,
                    detail="Endpoint da API não encontrado.",
                )
            return FileResponse(index_html)

    return application


app: FastAPI = create_app()

__all__ = [
    "LOCALHOST_CORS_ORIGINS",
    "app",
    "create_app",
]
