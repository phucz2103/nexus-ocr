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


class DataInfo(BaseModel):
    width: int | None
    height: int | None
    type: str


class ExtractResponse(BaseModel):
    request_id: str
    status: str
    backend: str
    artifacts: ArtifactLinks
    summary: ExtractionSummary
    data_info: DataInfo


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
