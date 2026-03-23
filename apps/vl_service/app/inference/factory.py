from __future__ import annotations

from app.core.config import Settings
from app.core.errors import BackendUnavailableError
from app.inference.base import InferenceBackend
from app.inference.mock_backend import MockVLBackend
from app.inference.paddle_vl_backend import PaddleOCRVLBackend


def build_inference_backend(settings: Settings) -> InferenceBackend:
    if settings.backend == "mock":
        return MockVLBackend(settings)
    if settings.backend == "paddleocr_vl":
        return PaddleOCRVLBackend(settings)
    raise BackendUnavailableError(f"Unsupported backend: {settings.backend}")
