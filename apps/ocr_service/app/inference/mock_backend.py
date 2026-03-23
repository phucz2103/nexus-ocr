from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from app.core.config import Settings
from app.inference.base import (
    BackendStatus,
    InferenceBackend,
    InferencePage,
    InferenceRunResult,
    build_model_settings_snapshot,
)


class MockOCRBackend(InferenceBackend):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

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
        with Image.open(input_path) as image:
            image.load()
            base_image = image.convert("RGB")

        width, height = base_image.size
        lines = [
            "Mock OCR extraction for document text.",
            "72 DPI phu hop cho web va man hinh.",
            "300 DPI phu hop cho in an chat luong cao.",
        ]
        line_height = max(36, height // 8)
        top = max(16, height // 12)
        left = max(16, width // 14)
        right = min(width - 16, width - width // 12)

        preview = base_image.copy()
        drawer = ImageDraw.Draw(preview)

        dt_polys: list[list[list[int]]] = []
        rec_polys: list[list[list[int]]] = []
        rec_boxes: list[list[int]] = []
        rec_scores: list[float] = []
        ocr_res_list: list[dict[str, object]] = []

        for index, text in enumerate(lines):
            y0 = min(height - 20, top + index * line_height)
            y1 = min(height - 8, y0 + max(28, line_height - 8))
            poly = [[left, y0], [right, y0], [right, y1], [left, y1]]
            box = [left, y0, right, y1]
            score = round(0.99 - index * 0.02, 4)

            drawer.rectangle([left, y0, right, y1], outline="red", width=max(2, width // 320))

            dt_polys.append(poly)
            rec_polys.append(poly)
            rec_boxes.append(box)
            rec_scores.append(score)
            ocr_res_list.append(
                {
                    "text": text,
                    "rec_score": score,
                    "dt_poly": poly,
                    "rec_poly": poly,
                    "rec_box": box,
                    "line_id": index,
                    "line_order": index + 1,
                }
            )

        full_text = "\n".join(lines)
        page = InferencePage(
            pruned_result={
                "page_count": None,
                "page_index": None,
                "width": width,
                "height": height,
                "model_settings": build_model_settings_snapshot(self._settings),
                "full_text": full_text,
                "ocr_res_list": ocr_res_list,
                "ocr_res": {
                    "input_path": str(input_path),
                    "page_index": None,
                    "model_settings": {
                        "use_doc_preprocessor": bool(self._settings.use_doc_preprocessor),
                        "use_textline_orientation": bool(self._settings.use_textline_orientation),
                    },
                    "dt_polys": dt_polys,
                    "text_det_params": {
                        "limit_side_len": self._settings.text_det_limit_side_len,
                        "limit_type": self._settings.text_det_limit_type,
                    },
                    "text_type": "general",
                    "textline_orientation_angles": [],
                    "text_rec_score_thresh": self._settings.text_rec_score_thresh,
                    "return_word_box": self._settings.return_word_box,
                    "rec_texts": lines,
                    "rec_scores": rec_scores,
                    "rec_polys": rec_polys,
                    "rec_boxes": rec_boxes,
                },
            },
            markdown={
                "text": "\n\n".join(lines),
                "images": {},
            },
            images={
                "ocr_res_img": preview,
            },
        )
        return InferenceRunResult(backend=self.name, pages=[page])