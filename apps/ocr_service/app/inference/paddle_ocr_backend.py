from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.core.errors import BackendUnavailableError, InferenceFailedError
from app.inference.base import (
    BackendStatus,
    InferenceBackend,
    InferencePage,
    InferenceRunResult,
    build_model_settings_snapshot,
)


LOGGER = logging.getLogger(__name__)


class PaddleOCRBackend(InferenceBackend):
    name = "paddleocr"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Any | None = None
        self._last_error: str | None = None
        self._lock = Lock()

    def warmup(self) -> None:
        self._ensure_pipeline()

    def get_status(self) -> BackendStatus:
        return BackendStatus(
            backend=self.name,
            ready=self._last_error is None,
            initialized=self._pipeline is not None,
            last_error=self._last_error,
        )

    def extract(self, input_path: Path) -> InferenceRunResult:
        pipeline = self._ensure_pipeline()
        width, height = self._read_image_size(input_path)
        try:
            raw_results = list(
                pipeline.predict(str(input_path), **self._settings.predict_kwargs())
            )
        except Exception as exc:
            self._last_error = str(exc)
            LOGGER.exception("PaddleOCR inference failed for %s", input_path)
            raise InferenceFailedError(
                f"PaddleOCR failed to process `{input_path.name}`: {exc}"
            ) from exc

        if not raw_results:
            raise InferenceFailedError("PaddleOCR returned no OCR results.")

        pages: list[InferencePage] = []
        for raw_result in raw_results:
            raw_json = raw_result.json.get("res", {})
            images = dict(getattr(raw_result, "img", {}) or {})
            doc_preprocessor_res = raw_result.get("doc_preprocessor_res") or {}
            output_img = doc_preprocessor_res.get("output_img")
            if output_img is not None and "preprocessed_img" not in images:
                images["preprocessed_img"] = output_img

            markdown_text = self._build_markdown_text(raw_json.get("rec_texts", []))
            pages.append(
                InferencePage(
                    pruned_result=self._normalize_page_result(raw_json, width, height),
                    markdown={
                        "text": markdown_text,
                        "images": {},
                    },
                    images=images,
                )
            )

        self._last_error = None
        return InferenceRunResult(backend=self.name, pages=pages)

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None and hasattr(pipeline, "close"):
            pipeline.close()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                if self._settings.disable_model_source_check:
                    os.environ.setdefault(
                        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
                        "True",
                    )
                from paddleocr import PaddleOCR

                self._pipeline = PaddleOCR(**self._settings.pipeline_kwargs())
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("Failed to initialize PaddleOCR pipeline")
                raise BackendUnavailableError(
                    f"Unable to initialize PaddleOCR: {exc}"
                ) from exc

        return self._pipeline

    def _normalize_page_result(
        self,
        page_data: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        rec_texts = page_data.get("rec_texts", []) or []
        rec_scores = page_data.get("rec_scores", []) or []
        dt_polys = page_data.get("dt_polys", []) or []
        rec_polys = page_data.get("rec_polys", []) or []
        rec_boxes = page_data.get("rec_boxes", []) or []

        ocr_res_list: list[dict[str, Any]] = []
        for index, text in enumerate(rec_texts):
            ocr_res_list.append(
                {
                    "text": text,
                    "rec_score": rec_scores[index] if index < len(rec_scores) else None,
                    "dt_poly": dt_polys[index] if index < len(dt_polys) else [],
                    "rec_poly": rec_polys[index] if index < len(rec_polys) else [],
                    "rec_box": rec_boxes[index] if index < len(rec_boxes) else [],
                    "line_id": index,
                    "line_order": index + 1,
                }
            )

        return {
            "page_count": page_data.get("page_index"),
            "page_index": page_data.get("page_index"),
            "width": width,
            "height": height,
            "model_settings": build_model_settings_snapshot(self._settings),
            "full_text": "\n".join(text.strip() for text in rec_texts if text and text.strip()),
            "ocr_res_list": ocr_res_list,
            "ocr_res": page_data,
        }

    def _build_markdown_text(self, rec_texts: list[str]) -> str:
        lines = [text.strip() for text in rec_texts if text and text.strip()]
        return "\n\n".join(lines)

    def _read_image_size(self, input_path: Path) -> tuple[int, int]:
        with Image.open(input_path) as image:
            image.load()
            return image.size