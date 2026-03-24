from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
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
        pages = self._load_pages(input_path)
        total_pages = len(pages)
        results: list[InferencePage] = []

        for page_index, base_image in enumerate(pages):
            width, height = base_image.size
            x0 = max(12, width // 20)
            y0 = max(12, height // 20)
            x1 = min(width - 12, width - width // 10)
            y1 = min(height - 12, y0 + max(40, height // 6))
            polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

            layout_preview = base_image.copy()
            drawer = ImageDraw.Draw(layout_preview)
            drawer.rectangle(
                [x0, y0, x1, y1],
                outline="red",
                width=max(2, width // 320),
            )

            page_label = f"page {page_index + 1}/{total_pages}"
            content = (
                f"Mock extraction for `{input_path.name}` ({page_label}). "
                "Set `VL_SERVICE_BACKEND=paddleocr_vl` to run the real PaddleOCR-VL pipeline."
            )
            block = {
                "block_label": "paragraph_title",
                "block_content": content,
                "block_bbox": [x0, y0, x1, y1],
                "block_id": 0,
                "block_order": 1,
                "group_id": 0,
                "global_block_id": page_index,
                "global_group_id": page_index,
                "block_polygon_points": polygon,
            }
            pruned_result = {
                "page_count": total_pages if input_path.suffix.lower() == ".pdf" else None,
                "page_index": page_index if input_path.suffix.lower() == ".pdf" else None,
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
                "text": f"## Mock extraction for {input_path.name} ({page_label})\n\n{content}",
                "images": {},
            }
            results.append(
                InferencePage(
                    pruned_result=pruned_result,
                    markdown=markdown,
                    images={
                        "input_img": base_image.copy(),
                        "preprocessed_img": base_image.copy(),
                        "layout_det_res": layout_preview,
                    },
                    metrics={
                        "page_index": page_index,
                        "processing_duration_ms": 5,
                        "detector_confidence": 1.0,
                        "block_count": 1,
                    },
                )
            )

        return InferenceRunResult(
            backend=self.name,
            pages=results,
            processing_duration_ms=max(1, total_pages * 5),
            engine_version=f"mock/{self._settings.pipeline_version}",
        )

    def _load_pages(self, input_path: Path) -> list[Image.Image]:
        if input_path.suffix.lower() != ".pdf":
            with Image.open(input_path) as image:
                image.load()
                return [image.convert("RGB")]

        document = pdfium.PdfDocument(str(input_path))
        pages: list[Image.Image] = []
        try:
            for page_index in range(len(document)):
                page = document[page_index]
                try:
                    bitmap = page.render(scale=2.0)
                    pages.append(bitmap.to_pil().convert("RGB"))
                finally:
                    page.close()
        finally:
            document.close()
        return pages