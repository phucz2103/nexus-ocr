from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass(slots=True)
class TableDetectionPage:
    page_index: int
    has_table: bool | None
    score: float | None = None
    label: str | None = None


@dataclass(slots=True)
class TableDetectionResult:
    backend: str
    enabled: bool
    status: str
    model_name: str
    threshold: float | None
    document_has_table: bool | None
    processing_duration_ms: int | None = None
    pages: list[TableDetectionPage] = field(default_factory=list)
    recommended_route: str | None = None
    actual_route: str = "router_service"
    error: str | None = None


class TableDetector(ABC):
    name: str

    @abstractmethod
    def detect(
        self,
        page_images: list[Image.Image],
        page_indices: list[int],
    ) -> TableDetectionResult:
        raise NotImplementedError

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None
