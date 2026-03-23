from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import ServiceError
from app.inference.factory import build_inference_backend
from app.services.extraction import ExtractionService
from app.services.storage import StorageService


LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    storage = StorageService(settings)
    backend = build_inference_backend(settings)
    extraction_service = ExtractionService(settings, storage, backend)

    app.state.settings = settings
    app.state.storage = storage
    app.state.inference_backend = backend
    app.state.extraction_service = extraction_service

    if settings.preload_model:
        try:
            backend.warmup()
        except Exception:
            LOGGER.exception("Model warmup failed during startup")

    try:
        yield
    finally:
        backend.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(ServiceError)
    async def handle_service_error(
        _request: Request,
        exc: ServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    return app


app = create_app()