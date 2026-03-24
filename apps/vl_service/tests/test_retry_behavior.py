from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.core.config import Settings
from app.core.errors import BackendUnavailableError, InferenceFailedError
from app.inference.base import BackendStatus, InferenceBackend, InferencePage, InferenceRunResult
from app.services.extraction import ExtractionService
from app.services.storage import StorageService
from app.table_detection.disabled_detector import DisabledTableDetector


class FlakyBackend(InferenceBackend):
    name = "fake"

    def __init__(self, failures: int, error_cls: type[Exception]) -> None:
        self.failures = failures
        self.error_cls = error_cls
        self.calls = 0
        self.close_calls = 0

    def warmup(self) -> None:
        return None

    def get_status(self) -> BackendStatus:
        return BackendStatus(
            backend=self.name,
            ready=True,
            initialized=True,
            last_error=None,
        )

    def extract(self, input_path: Path) -> InferenceRunResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error_cls("temporary backend failure")

        page = InferencePage(
            pruned_result={
                "page_count": None,
                "width": 160,
                "height": 80,
                "model_settings": {},
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Recovered text after retry.",
                    }
                ],
                "layout_det_res": {
                    "boxes": [{"score": 0.91}],
                },
            },
            markdown={"text": "Recovered text after retry.", "images": {}},
            images={},
            metrics={
                "page_index": 0,
                "processing_duration_ms": 7,
                "detector_confidence": 0.91,
                "block_count": 1,
            },
        )
        return InferenceRunResult(
            backend=self.name,
            pages=[page],
            processing_duration_ms=7,
            engine_version="fake/v1.5",
        )

    def close(self) -> None:
        self.close_calls += 1


class RetryBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _build_settings(self, retry_attempts: int = 1) -> Settings:
        return Settings(
            backend="mock",
            artifact_root=Path(self._temp_dir.name) / "artifacts",
            retry_attempts=retry_attempts,
            retry_backoff_ms=0,
            table_detector_backend="disabled",
        )

    def _build_upload(self) -> UploadFile:
        image_bytes = io.BytesIO()
        Image.new("RGB", (160, 80), color="white").save(image_bytes, format="PNG")
        image_bytes.seek(0)
        return UploadFile(
            file=image_bytes,
            filename="sample.png",
            headers=Headers({"content-type": "image/png"}),
        )

    async def test_extract_retries_once_and_recovers(self) -> None:
        settings = self._build_settings(retry_attempts=1)
        storage = StorageService(settings)
        backend = FlakyBackend(failures=1, error_cls=InferenceFailedError)
        table_detector = DisabledTableDetector(settings)
        service = ExtractionService(settings, storage, backend, table_detector)

        result = await service.extract(self._build_upload())

        self.assertEqual(result.backend, "fake")
        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.raw_text, "Recovered text after retry.")
        self.assertEqual(result.processing_duration_ms, 7)
        self.assertEqual(result.detector_confidence, 0.91)
        self.assertEqual(len(result.page_metrics), 1)
        self.assertEqual(result.page_metrics[0]["page_index"], 0)
        self.assertEqual(result.table_detection["status"], "disabled")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.close_calls, 1)

    async def test_extract_retries_backend_init_failures(self) -> None:
        settings = self._build_settings(retry_attempts=1)
        storage = StorageService(settings)
        backend = FlakyBackend(failures=1, error_cls=BackendUnavailableError)
        table_detector = DisabledTableDetector(settings)
        service = ExtractionService(settings, storage, backend, table_detector)

        result = await service.extract(self._build_upload())

        self.assertEqual(result.backend, "fake")
        self.assertEqual(result.table_detection["status"], "disabled")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.close_calls, 1)

    async def test_extract_raises_after_retry_budget_is_exhausted(self) -> None:
        settings = self._build_settings(retry_attempts=1)
        storage = StorageService(settings)
        backend = FlakyBackend(failures=2, error_cls=InferenceFailedError)
        table_detector = DisabledTableDetector(settings)
        service = ExtractionService(settings, storage, backend, table_detector)

        with self.assertRaises(InferenceFailedError):
            await service.extract(self._build_upload())

        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
