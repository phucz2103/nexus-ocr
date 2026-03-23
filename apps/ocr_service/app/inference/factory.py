from __future__ import annotations

from app.core.config import Settings
from app.core.errors import BackendUnavailableError
from app.inference.base import InferenceBackend
from app.inference.mock_backend import MockOCRBackend
from app.inference.paddle_ocr_backend import PaddleOCRBackend


def build_inference_backend(settings: Settings) -> InferenceBackend:
    if settings.backend == "mock":
        return MockOCRBackend(settings)
    if settings.backend == "paddleocr":
        return PaddleOCRBackend(settings)
    raise BackendUnavailableError(f"Unsupported backend: {settings.backend}")