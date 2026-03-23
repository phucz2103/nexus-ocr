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
        "use_textline_orientation": bool(settings.use_textline_orientation),
        "ocr_version": settings.ocr_version,
        "preferred_lang": settings.preferred_lang,
        "text_detection_model_name": settings.text_detection_model_name,
        "text_recognition_model_name": settings.text_recognition_model_name,
        "text_det_limit_side_len": settings.text_det_limit_side_len,
        "text_det_limit_type": settings.text_det_limit_type,
        "text_rec_score_thresh": settings.text_rec_score_thresh,
        "return_word_box": settings.return_word_box,
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