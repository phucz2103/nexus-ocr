from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import Settings
from app.core.errors import InvalidUploadError
from app.inference.base import InferenceBackend
from app.models.api import (
    ExtractJobResponse,
    ExtractResponse,
    HealthResponse,
    ReadyResponse,
)
from app.services.extraction import CompletedExtraction, ExtractionService
from app.services.jobs import AsyncJobService


router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_extraction_service(request: Request) -> ExtractionService:
    return request.app.state.extraction_service


def get_job_service(request: Request) -> AsyncJobService:
    return request.app.state.job_service


def get_inference_backend(request: Request) -> InferenceBackend:
    return request.app.state.inference_backend


def _build_file_response(path: Path, media_type: str | None = None) -> FileResponse:
    resolved_media_type = media_type or mimetypes.guess_type(path.name)[0]
    return FileResponse(
        path,
        media_type=resolved_media_type or "application/octet-stream",
        filename=path.name,
    )


def _parse_optional_page_value(value: str | None, field_name: str) -> int | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    try:
        return int(normalized)
    except ValueError as exc:
        raise InvalidUploadError(f"`{field_name}` must be an integer.") from exc


def _to_extract_response(result: CompletedExtraction) -> ExtractResponse:
    return ExtractResponse(
        request_id=result.request_id,
        status="succeeded",
        backend=result.backend,
        artifacts={
            "json_url": result.json_url,
            "markdown_url": result.markdown_url,
            "input_url": result.input_url,
        },
        summary=result.summary,
        data_info=result.data_info,
        page_count=result.page_count,
        raw_text=result.raw_text,
        processing_duration_ms=result.processing_duration_ms,
        detector_confidence=result.detector_confidence,
        engine_version=result.engine_version,
        page_metrics=result.page_metrics,
        page_selection=result.page_selection,
        table_detection=result.table_detection,
        stage_timings=result.stage_timings,
    )


def _to_job_response(result: dict) -> ExtractJobResponse:
    return ExtractJobResponse(**result)


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    backend: InferenceBackend = Depends(get_inference_backend),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        backend=backend.name,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(
    backend: InferenceBackend = Depends(get_inference_backend),
) -> ReadyResponse:
    status = backend.get_status()
    return ReadyResponse(
        status="ready" if status.ready else "not_ready",
        backend=status.backend,
        ready=status.ready,
        initialized=status.initialized,
        last_error=status.last_error,
    )


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    page_start: str | None = Form(None),
    page_end: str | None = Form(None),
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractResponse:
    result = await service.extract(
        file,
        page_start=_parse_optional_page_value(page_start, "page_start"),
        page_end=_parse_optional_page_value(page_end, "page_end"),
    )
    return _to_extract_response(result)


@router.post(
    "/v1/extract/jobs",
    response_model=ExtractJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_extract_job(
    file: UploadFile = File(...),
    page_start: str | None = Form(None),
    page_end: str | None = Form(None),
    service: AsyncJobService = Depends(get_job_service),
) -> ExtractJobResponse:
    payload = await file.read()
    result = service.create_job(
        payload=payload,
        filename=file.filename,
        content_type=file.content_type,
        page_start=_parse_optional_page_value(page_start, "page_start"),
        page_end=_parse_optional_page_value(page_end, "page_end"),
    )
    service.start_job(result["job_id"])
    return _to_job_response(result)


@router.get("/v1/extract/jobs/{job_id}", response_model=ExtractJobResponse)
def get_extract_job(
    job_id: str,
    service: AsyncJobService = Depends(get_job_service),
) -> ExtractJobResponse:
    return _to_job_response(service.get_job(job_id))


@router.get("/v1/results/{request_id}/json")
def get_result_json(
    request_id: str,
    service: ExtractionService = Depends(get_extraction_service),
) -> JSONResponse:
    return JSONResponse(content=service.load_result_json(request_id))


@router.get("/v1/results/{request_id}/markdown")
def get_result_markdown(
    request_id: str,
    service: ExtractionService = Depends(get_extraction_service),
) -> FileResponse:
    return _build_file_response(
        service.resolve_result_markdown(request_id),
        media_type="text/markdown",
    )


@router.get("/v1/results/{request_id}/input")
def get_input_image(
    request_id: str,
    service: ExtractionService = Depends(get_extraction_service),
) -> FileResponse:
    return _build_file_response(service.resolve_input_file(request_id))


@router.get("/v1/results/{request_id}/artifacts/{artifact_name}")
def get_result_artifact(
    request_id: str,
    artifact_name: str,
    service: ExtractionService = Depends(get_extraction_service),
) -> FileResponse:
    return _build_file_response(service.resolve_artifact(request_id, artifact_name))
