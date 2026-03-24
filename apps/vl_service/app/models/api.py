from __future__ import annotations

from pydantic import BaseModel


class ArtifactLinks(BaseModel):
    json_url: str
    markdown_url: str
    input_url: str


class ExtractionSummary(BaseModel):
    pages: int
    blocks: int
    tables: int


class DataPageInfo(BaseModel):
    page_index: int
    width: int | None
    height: int | None


class DataInfo(BaseModel):
    width: int | None
    height: int | None
    type: str
    page_count: int | None = None
    pages: list[DataPageInfo] | None = None


class PageMetric(BaseModel):
    page_index: int
    processing_duration_ms: int | None = None
    detector_confidence: float | None = None
    block_count: int | None = None


class PageSelection(BaseModel):
    page_start: int
    page_end: int
    selected_page_count: int
    total_page_count: int


class TableDetectionPage(BaseModel):
    page_index: int
    has_table: bool | None = None
    score: float | None = None
    label: str | None = None


class TableDetectionResult(BaseModel):
    backend: str
    enabled: bool
    status: str
    model_name: str
    threshold: float | None = None
    document_has_table: bool | None = None
    processing_duration_ms: int | None = None
    recommended_route: str | None = None
    actual_route: str
    error: str | None = None
    pages: list[TableDetectionPage]


class ExtractResponse(BaseModel):
    request_id: str
    status: str
    backend: str
    artifacts: ArtifactLinks
    summary: ExtractionSummary
    data_info: DataInfo
    page_count: int
    raw_text: str
    processing_duration_ms: int | None = None
    detector_confidence: float | None = None
    engine_version: str
    page_metrics: list[PageMetric]
    page_selection: PageSelection | None = None
    table_detection: TableDetectionResult
    stage_timings: dict[str, int | None]


class ExtractJobResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    created_at: str
    updated_at: str
    request_id: str | None = None
    backend: str | None = None
    summary: ExtractionSummary | None = None
    data_info: DataInfo | None = None
    page_count: int | None = None
    raw_text: str | None = None
    processing_duration_ms: int | None = None
    detector_confidence: float | None = None
    engine_version: str | None = None
    page_metrics: list[PageMetric] | None = None
    page_selection: PageSelection | None = None
    table_detection: TableDetectionResult | None = None
    stage_timings: dict[str, int | None] | None = None
    artifacts: ArtifactLinks | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    backend: str


class ReadyResponse(BaseModel):
    status: str
    backend: str
    ready: bool
    initialized: bool
    last_error: str | None = None
