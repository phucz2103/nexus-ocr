from __future__ import annotations

from app.core.config import Settings
from app.core.errors import BackendUnavailableError
from app.table_detection.base import TableDetector
from app.table_detection.disabled_detector import DisabledTableDetector
from app.table_detection.mock_detector import MockTableDetector
from app.table_detection.protonx_detector import ProtonXTableDetector



def build_table_detector(settings: Settings) -> TableDetector:
    if settings.table_detector_backend == "disabled":
        return DisabledTableDetector(settings)
    if settings.table_detector_backend == "mock":
        return MockTableDetector(settings)
    if settings.table_detector_backend == "protonx":
        return ProtonXTableDetector(settings)
    raise BackendUnavailableError(
        f"Unsupported table detector backend: {settings.table_detector_backend}"
    )
