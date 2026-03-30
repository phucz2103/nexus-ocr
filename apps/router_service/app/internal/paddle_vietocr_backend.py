from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.core.errors import BackendUnavailableError, InferenceFailedError
from app.internal.base import (
    BackendStatus,
    InferenceBackend,
    InferencePage,
    InferenceRunResult,
    build_model_settings_snapshot,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected alongside paddleocr
    np = None


LOGGER = logging.getLogger(__name__)


class PaddleVietOCRBackend(InferenceBackend):
    name = "paddleocr_vietocr"

    def __init__(
        self,
        settings: Settings,
        *,
        pipeline: Any | None = None,
        recognizer: Any | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline: Any | None = pipeline
        self._recognizer: Any | None = recognizer
        self._last_error: str | None = None
        self._lock = Lock()

    def warmup(self) -> None:
        self._ensure_pipeline()
        self._ensure_recognizer()

    def get_status(self) -> BackendStatus:
        initialized = self._pipeline is not None and self._recognizer is not None
        return BackendStatus(
            backend=self.name,
            ready=self._last_error is None,
            initialized=initialized,
            last_error=self._last_error,
        )

    def extract(self, input_path: Path) -> InferenceRunResult:
        pipeline = self._ensure_pipeline()
        recognizer = self._ensure_recognizer()
        width, height = self._read_image_size(input_path)
        fallback_image = self._load_rgb_image(input_path)

        try:
            raw_results = list(
                pipeline.predict(str(input_path), **self._settings.predict_kwargs())
            )
        except Exception as exc:
            self._last_error = str(exc)
            LOGGER.exception("Hybrid PaddleOCR/VietOCR inference failed for %s", input_path)
            raise InferenceFailedError(
                f"Hybrid PaddleOCR/VietOCR failed to process `{input_path.name}`: {exc}"
            ) from exc

        if not raw_results:
            raise InferenceFailedError("Hybrid PaddleOCR/VietOCR returned no OCR results.")

        pages: list[InferencePage] = []
        for raw_result in raw_results:
            raw_json = dict(raw_result.json.get("res", {}) or {})
            images = dict(getattr(raw_result, "img", {}) or {})
            doc_preprocessor_res = raw_result.get("doc_preprocessor_res") or {}
            output_img = doc_preprocessor_res.get("output_img")
            if output_img is not None and "preprocessed_img" not in images:
                images["preprocessed_img"] = output_img

            crop_source = self._resolve_crop_source(images, fallback_image)
            ocr_res_list, full_text = self._recognize_page(
                recognizer,
                crop_source,
                raw_json,
            )
            raw_json["rec_texts"] = [item["text"] for item in ocr_res_list]
            raw_json["rec_scores"] = [item["rec_score"] for item in ocr_res_list]

            pages.append(
                InferencePage(
                    pruned_result=self._normalize_page_result(
                        raw_json,
                        width,
                        height,
                        ocr_res_list,
                        full_text,
                    ),
                    markdown={
                        "text": self._build_markdown_text([item["text"] for item in ocr_res_list]),
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
        self._recognizer = None
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
                LOGGER.exception("Failed to initialize PaddleOCR pipeline for hybrid backend")
                raise BackendUnavailableError(
                    f"Unable to initialize PaddleOCR for hybrid backend: {exc}"
                ) from exc

        return self._pipeline

    def _ensure_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer

        with self._lock:
            if self._recognizer is not None:
                return self._recognizer

            try:
                from vietocr.tool.config import Cfg
                from vietocr.tool.predictor import Predictor

                config = Cfg.load_config_from_name(self._settings.vietocr_config_name)
                if self._settings.vietocr_weights is not None:
                    config["weights"] = str(self._settings.vietocr_weights)
                config["device"] = self._settings.vietocr_device

                predictor_config = config.get("predictor")
                if isinstance(predictor_config, dict):
                    predictor_config["beamsearch"] = self._settings.vietocr_beamsearch

                cnn_config = config.get("cnn")
                if isinstance(cnn_config, dict) and "pretrained" in cnn_config:
                    cnn_config["pretrained"] = False

                self._recognizer = Predictor(config)
                self._last_error = None
            except ModuleNotFoundError as exc:
                self._last_error = str(exc)
                raise BackendUnavailableError(
                    "VietOCR is not installed. Install it with `pip install vietocr` "
                    "or switch OCR_SERVICE_BACKEND back to `paddleocr`."
                ) from exc
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("Failed to initialize VietOCR recognizer")
                raise BackendUnavailableError(
                    f"Unable to initialize VietOCR recognizer: {exc}"
                ) from exc

        return self._recognizer

    def _recognize_page(
        self,
        recognizer: Any,
        image: Image.Image,
        page_data: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        det_polys = page_data.get("dt_polys") or page_data.get("rec_polys") or []
        fallback_texts = page_data.get("rec_texts", []) or []

        if not det_polys:
            fallback_items = self._build_fallback_items(page_data)
            full_text = "\n".join(
                item["text"].strip() for item in fallback_items if item["text"].strip()
            )
            return fallback_items, full_text

        ocr_res_list: list[dict[str, Any]] = []
        text_lines: list[str] = []
        for index, polygon in enumerate(det_polys):
            crop = self._crop_polygon_region(image, polygon)
            text, score = self._predict_crop_text(recognizer, crop)
            if (
                not text
                and self._settings.vietocr_fallback_to_paddle_recognition
                and index < len(fallback_texts)
            ):
                text = str(fallback_texts[index] or "").strip()

            box = self._bounding_box(polygon)
            ocr_res_list.append(
                {
                    "text": text,
                    "rec_score": score,
                    "dt_poly": polygon,
                    "rec_poly": polygon,
                    "rec_box": box,
                    "line_id": index,
                    "line_order": index + 1,
                }
            )
            if text:
                text_lines.append(text)

        return ocr_res_list, "\n".join(text_lines)

    def _build_fallback_items(self, page_data: dict[str, Any]) -> list[dict[str, Any]]:
        rec_texts = page_data.get("rec_texts", []) or []
        rec_scores = page_data.get("rec_scores", []) or []
        dt_polys = page_data.get("dt_polys", []) or []
        rec_polys = page_data.get("rec_polys", []) or []
        rec_boxes = page_data.get("rec_boxes", []) or []

        fallback_items: list[dict[str, Any]] = []
        for index, text in enumerate(rec_texts):
            fallback_items.append(
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
        return fallback_items

    def _predict_crop_text(self, recognizer: Any, crop: Image.Image) -> tuple[str, float | None]:
        try:
            prediction = recognizer.predict(crop)
        except Exception as exc:
            LOGGER.warning("VietOCR crop prediction failed: %s", exc)
            return "", None

        if isinstance(prediction, tuple):
            text = str(prediction[0] or "") if prediction else ""
            score = prediction[1] if len(prediction) > 1 else None
        elif isinstance(prediction, dict):
            text = str(
                prediction.get("text")
                or prediction.get("pred")
                or prediction.get("prediction")
                or ""
            )
            score = prediction.get("score") or prediction.get("confidence")
        else:
            text = str(prediction or "")
            score = None

        normalized_score = float(score) if isinstance(score, (int, float)) else None
        return text.strip(), normalized_score

    def _normalize_page_result(
        self,
        page_data: dict[str, Any],
        width: int,
        height: int,
        ocr_res_list: list[dict[str, Any]],
        full_text: str,
    ) -> dict[str, Any]:
        normalized_page_data = dict(page_data)
        normalized_page_data["rec_texts"] = [item["text"] for item in ocr_res_list]
        normalized_page_data["rec_scores"] = [item["rec_score"] for item in ocr_res_list]
        normalized_page_data["rec_polys"] = [item["rec_poly"] for item in ocr_res_list]
        normalized_page_data["rec_boxes"] = [item["rec_box"] for item in ocr_res_list]

        return {
            "page_count": page_data.get("page_index"),
            "page_index": page_data.get("page_index"),
            "width": width,
            "height": height,
            "model_settings": build_model_settings_snapshot(self._settings),
            "full_text": full_text,
            "ocr_res_list": ocr_res_list,
            "ocr_res": normalized_page_data,
        }

    def _build_markdown_text(self, rec_texts: list[str]) -> str:
        lines = [text.strip() for text in rec_texts if text and text.strip()]
        return "\n\n".join(lines)

    def _resolve_crop_source(
        self,
        images: dict[str, Any],
        fallback_image: Image.Image,
    ) -> Image.Image:
        for key in ("preprocessed_img", "output_img", "input_img"):
            if key not in images:
                continue
            resolved = self._to_pil_image(images[key])
            if resolved is not None:
                return resolved
        return fallback_image.copy()

    def _to_pil_image(self, image_value: Any) -> Image.Image | None:
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if np is not None and isinstance(image_value, np.ndarray):
            array = image_value
            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")
            if array.ndim == 3 and array.shape[2] > 3:
                array = array[:, :, :3]
            return Image.fromarray(array).convert("RGB")
        return None

    def _crop_polygon_region(self, image: Image.Image, polygon: Any) -> Image.Image:
        box = self._bounding_box(polygon)
        if not box:
            return image.copy()

        left, top, right, bottom = box
        padding = max(int(self._settings.vietocr_crop_padding), 0)
        left = max(left - padding, 0)
        top = max(top - padding, 0)
        right = min(right + padding, image.width)
        bottom = min(bottom + padding, image.height)

        if right <= left or bottom <= top:
            return image.copy()
        return image.crop((left, top, right, bottom)).convert("RGB")

    def _bounding_box(self, polygon: Any) -> list[int]:
        points = self._normalize_polygon(polygon)
        if not points:
            return []
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        return [
            max(int(min(x_values)), 0),
            max(int(min(y_values)), 0),
            max(int(max(x_values)), 0),
            max(int(max(y_values)), 0),
        ]

    def _normalize_polygon(self, polygon: Any) -> list[tuple[float, float]]:
        if polygon is None:
            return []

        points: list[tuple[float, float]] = []
        for point in polygon:
            if point is None or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            points.append((x, y))
        return points

    def _read_image_size(self, input_path: Path) -> tuple[int, int]:
        with Image.open(input_path) as image:
            image.load()
            return image.size

    def _load_rgb_image(self, input_path: Path) -> Image.Image:
        with Image.open(input_path) as image:
            image.load()
            return image.convert("RGB")

