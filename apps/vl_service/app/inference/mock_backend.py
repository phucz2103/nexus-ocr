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


class MockVLBackend(InferenceBackend):
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
        x0 = max(12, width // 20)
        y0 = max(12, height // 20)
        x1 = min(width - 12, width - width // 10)
        y1 = min(height - 12, y0 + max(40, height // 6))
        polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        layout_preview = base_image.copy()
        drawer = ImageDraw.Draw(layout_preview)
        drawer.rectangle([x0, y0, x1, y1], outline="red", width=max(2, width // 320))

        content = (
            f"Mock extraction for `{input_path.name}`. "
            "Set `VL_SERVICE_BACKEND=paddleocr_vl` to run the real PaddleOCR-VL pipeline."
        )
        block = {
            "block_label": "paragraph_title",
            "block_content": content,
            "block_bbox": [x0, y0, x1, y1],
            "block_id": 0,
            "block_order": 1,
            "group_id": 0,
            "global_block_id": 0,
            "global_group_id": 0,
            "block_polygon_points": polygon,
        }
        pruned_result = {
            "page_count": None,
            "width": width,
            "height": height,
            "model_settings": build_model_settings_snapshot(self._settings),
            "parsing_res_list": [block],
            "layout_det_res": {
                "boxes": [
                    {
                        "cls_id": 17,
                        "label": "paragraph_title",
                        "score": 1.0,
                        "coordinate": [x0, y0, x1, y1],
                        "order": 1,
                        "polygon_points": polygon,
                    }
                ]
            },
        }
        markdown = {
            "text": f"## Mock extraction for {input_path.name}\n\n{content}",
            "images": {},
        }
        page = InferencePage(
            pruned_result=pruned_result,
            markdown=markdown,
            images={
                "input_img": base_image.copy(),
                "preprocessed_img": base_image.copy(),
                "layout_det_res": layout_preview,
            },
        )
        return InferenceRunResult(backend=self.name, pages=[page])
