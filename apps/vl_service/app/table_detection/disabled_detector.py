from __future__ import annotations

from PIL import Image

from app.core.config import Settings
from app.table_detection.base import TableDetectionPage, TableDetectionResult, TableDetector


class DisabledTableDetector(TableDetector):
    name = "disabled"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def detect(
        self,
        page_images: list[Image.Image],
        page_indices: list[int],
    ) -> TableDetectionResult:
        pages = [
            TableDetectionPage(
                page_index=page_index,
                has_table=None,
                score=None,
                label=None,
            )
            for page_index in page_indices
        ]
        return TableDetectionResult(
            backend=self.name,
            enabled=False,
            status="disabled",
            model_name=self._settings.table_detector_model_name,
            threshold=self._settings.table_detector_threshold,
            document_has_table=None,
            processing_duration_ms=0,
            pages=pages,
            recommended_route=None,
            actual_route="ocr_vl_service",
            error=None,
        )
