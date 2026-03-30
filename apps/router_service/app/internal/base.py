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
    metrics: dict[str, Any] | None = None


@dataclass(slots=True)
class InferenceRunResult:
    backend: str
    pages: list[InferencePage]
    processing_duration_ms: int | None = None
    engine_version: str | None = None


def build_model_settings_snapshot(settings: Any) -> dict[str, Any]:
    return {
        "backend": getattr(settings, "backend", None),
        "use_doc_preprocessor": bool(getattr(settings, "use_doc_preprocessor", False)),
        "use_textline_orientation": bool(getattr(settings, "use_textline_orientation", False)),
        "ocr_version": getattr(settings, "ocr_version", None),
        "preferred_lang": getattr(settings, "preferred_lang", None),
        "text_detection_model_name": getattr(settings, "text_detection_model_name", None),
        "text_recognition_model_name": getattr(settings, "text_recognition_model_name", None),
        "text_det_limit_side_len": getattr(settings, "text_det_limit_side_len", None),
        "text_det_limit_type": getattr(settings, "text_det_limit_type", None),
        "text_rec_score_thresh": getattr(settings, "text_rec_score_thresh", None),
        "return_word_box": getattr(settings, "return_word_box", None),
        "pipeline_version": getattr(settings, "pipeline_version", None),
        "use_layout_detection": getattr(settings, "use_layout_detection", None),
        "use_chart_recognition": getattr(settings, "use_chart_recognition", None),
        "use_seal_recognition": getattr(settings, "use_seal_recognition", None),
        "use_ocr_for_image_block": getattr(settings, "use_ocr_for_image_block", None),
        "format_block_content": getattr(settings, "format_block_content", None),
        "merge_layout_blocks": getattr(settings, "merge_layout_blocks", None),
        "vietocr_config_name": getattr(settings, "vietocr_config_name", None),
        "vietocr_weights": (
            str(settings.vietocr_weights)
            if getattr(settings, "vietocr_weights", None) is not None
            else None
        ),
        "vietocr_device": getattr(settings, "vietocr_device", None),
        "vietocr_beamsearch": getattr(settings, "vietocr_beamsearch", None),
        "vietocr_fallback_to_paddle_recognition": getattr(
            settings,
            "vietocr_fallback_to_paddle_recognition",
            None,
        ),
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
