from __future__ import annotations

from time import perf_counter

from PIL import Image

from app.core.config import Settings
from app.table_detection.base import TableDetectionPage, TableDetectionResult, TableDetector


class MockTableDetector(TableDetector):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def detect(
        self,
        page_images: list[Image.Image],
        page_indices: list[int],
    ) -> TableDetectionResult:
        started_at = perf_counter()
        has_table = bool(self._settings.table_detector_mock_document_has_table)
        score = (
            float(self._settings.table_detector_mock_table_score)
            if has_table
            else float(self._settings.table_detector_mock_no_table_score)
        )
        label = "table" if has_table else "no_table"
        pages = [
            TableDetectionPage(
                page_index=page_index,
                has_table=has_table,
                score=round(score, 4),
                label=label,
            )
            for page_index in page_indices
        ]
        return TableDetectionResult(
            backend=self.name,
            enabled=True,
            status="succeeded",
            model_name=self._settings.table_detector_model_name,
            threshold=self._settings.table_detector_threshold,
            document_has_table=has_table,
            processing_duration_ms=max(
                1,
                int(round((perf_counter() - started_at) * 1000)),
            ),
            pages=pages,
            recommended_route="ocr_vl_service" if has_table else "ocr_service",
            actual_route="router_service",
            error=None,
        )
