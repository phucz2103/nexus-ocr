from __future__ import annotations

from pydantic import BaseModel


class ArtifactLinks(BaseModel):
    json_url: str
    input_url: str


class ExtractionSummary(BaseModel):
    pages: int
    lines: int
    words: int


class DataPageInfo(BaseModel):
    page_index: int
    width: int | None
    height: int | None


class DataInfo(BaseModel):
    width: int | None
    height: int | None
    type: str
    page_count: int
    pages: list[DataPageInfo]


class PageSelection(BaseModel):
    page_start: int
    page_end: int
    selected_page_count: int
    total_page_count: int


class ExtractResponse(BaseModel):
    request_id: str
    status: str
    backend: str
    artifacts: ArtifactLinks
    summary: ExtractionSummary
    data_info: DataInfo
    page_count: int
    page_selection: PageSelection | None = None


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
    page_selection: PageSelection | None = None
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
