from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BackendStatus:
    backend: str
    ready: bool
    initialized: bool
    last_error: str | None = None


@dataclass(slots=True)
class InferencePage:
    pruned_result: dict[str, Any]
    markdown: dict[str, Any]
    images: dict[str, Any]


@dataclass(slots=True)
class InferenceRunResult:
    backend: str
    pages: list[InferencePage]


def build_model_settings_snapshot(settings: Any) -> dict[str, Any]:
    return {
        "use_doc_preprocessor": bool(settings.use_doc_preprocessor),
        "use_layout_detection": bool(settings.use_layout_detection),
        "use_chart_recognition": bool(settings.use_chart_recognition),
        "use_seal_recognition": bool(settings.use_seal_recognition),
        "use_ocr_for_image_block": bool(settings.use_ocr_for_image_block),
        "format_block_content": bool(settings.format_block_content),
        "merge_layout_blocks": bool(settings.merge_layout_blocks),
        "markdown_ignore_labels": list(settings.markdown_ignore_labels),
        "return_layout_polygon_points": True,
    }


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> BackendStatus:
        raise NotImplementedError

    @abstractmethod
    def extract(self, input_path: Path) -> InferenceRunResult:
        raise NotImplementedError

    def close(self) -> None:
        return None
