from __future__ import annotations

import html
import logging
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from uuid import uuid4

import pypdfium2 as pdfium
from fastapi import UploadFile
from PIL import Image

from app.core.config import Settings
from app.core.errors import (
    InferenceFailedError,
    InvalidUploadError,
)
from app.internal.runtime import LocalEngineResult, LocalOcrRuntime, LocalVlRuntime
from app.services.storage import StorageService
from app.table_detection.base import TableDetectionResult, TableDetector


LOGGER = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ALLOWED_DOCUMENT_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | {".pdf"}
@dataclass(slots=True)
class CompletedExtraction:
    request_id: str
    backend: str
    summary: dict[str, int | None]
    data_info: dict[str, Any]
    json_url: str
    page_count: int
    raw_text: str
    raw_text_plain: str
    mapping_text: str
    tables: list[dict[str, Any]]
    processing_duration_ms: int | None
    detector_confidence: float | None
    engine_version: str
    page_metrics: list[dict[str, Any]]
    page_selection: dict[str, Any] | None
    table_detection: dict[str, Any]
    stage_timings: dict[str, int | None]


class ExtractionService:
    def __init__(
        self,
        settings: Settings,
        storage: StorageService,
        table_detector: TableDetector,
        ocr_runtime: LocalOcrRuntime,
        vl_runtime: LocalVlRuntime,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._table_detector = table_detector
        self._ocr_runtime = ocr_runtime
        self._vl_runtime = vl_runtime

    async def extract(
        self,
        file: UploadFile,
        page_start: int | None = None,
        page_end: int | None = None,
        enforce_sync_limit: bool = True,
    ) -> CompletedExtraction:
        payload = await file.read()
        if not payload:
            raise InvalidUploadError("Uploaded file is empty.")

        max_bytes = self._settings.max_upload_size_mb * 1024 * 1024
        if len(payload) > max_bytes:
            raise InvalidUploadError(
                f"Uploaded file exceeds the {self._settings.max_upload_size_mb} MB limit."
            )

        input_prepare_started_at = perf_counter()
        resolved_filename = file.filename or "upload"
        input_suffix = self._resolve_input_suffix(resolved_filename, file.content_type)
        request_id = self._generate_request_id()
        paths = self._storage.prepare_request(request_id)

        with TemporaryDirectory() as temp_dir:
            temp_input_path = Path(temp_dir) / f"input{input_suffix}"
            temp_input_path.write_bytes(payload)

            data_info = self._read_input_info(temp_input_path)
            page_selection = self._normalize_page_selection(
                temp_input_path,
                data_info,
                page_start,
                page_end,
            )
            if enforce_sync_limit:
                self._validate_sync_input_limits(temp_input_path, data_info, page_selection)
            input_prepare_ms = max(
                1,
                int(round((perf_counter() - input_prepare_started_at) * 1000)),
            )

            selected_page_indices = (
                list(page_selection["selected_page_indices"])
                if page_selection is not None
                else None
            )

            should_render_pdf_previews = (
                data_info["type"] == "pdf"
                and self._settings.table_detector_backend != "disabled"
            )
            pdf_render_ms: int | None = None
            input_page_images: list[Image.Image] | None = None
            if should_render_pdf_previews:
                pdf_render_started_at = perf_counter()
                input_page_images = self._render_pdf_previews(temp_input_path, selected_page_indices)
                pdf_render_ms = max(
                    1,
                    int(round((perf_counter() - pdf_render_started_at) * 1000)),
                )

            table_detection_started_at = perf_counter()
            table_detection_page_indices = (
                list(selected_page_indices)
                if selected_page_indices is not None
                else [0]
            )
            if self._settings.table_detector_backend == "disabled":
                table_detection_images: list[Image.Image] = []
            elif input_page_images is not None:
                table_detection_images = [image.copy() for image in input_page_images]
            else:
                table_detection_images = self._load_image_pages(temp_input_path)

            table_detection = self._detect_tables(
                page_images=table_detection_images,
                page_indices=table_detection_page_indices,
            )
            table_detection_ms = table_detection.processing_duration_ms
            if table_detection_ms is None:
                table_detection_ms = max(
                    1,
                    int(round((perf_counter() - table_detection_started_at) * 1000)),
                )

            route_name, runtime = self._resolve_route(table_detection)
            downstream_started_at = perf_counter()
            engine_result = await self._run_local_engine(
                runtime=runtime,
                payload=payload,
                filename=resolved_filename,
                content_type=file.content_type,
                page_selection=page_selection,
            )
            downstream_duration_ms = max(
                1,
                int(round((perf_counter() - downstream_started_at) * 1000)),
            )

            result_assembly_started_at = perf_counter()
            normalized_table_detection = self._apply_actual_route(table_detection, route_name)
            stage_timings = self._build_stage_timings(
                engine_result.stage_timings,
                input_prepare_ms=input_prepare_ms,
                pdf_render_ms=pdf_render_ms,
                table_detection_ms=table_detection_ms,
                ocr_ms=engine_result.processing_duration_ms,
            )

            result_json = self._build_result_json(
                request_id=request_id,
                engine_result=engine_result,
                table_detection=normalized_table_detection,
                stage_timings=stage_timings,
            )

            stage_timings["result_assembly_ms"] = max(
                1,
                int(round((perf_counter() - result_assembly_started_at) * 1000)),
            )
            result_json["stageTimings"] = self._to_json_stage_timings(stage_timings)

        self._storage.write_json(paths.result_json, result_json)

        processing_duration_ms = result_json.get("processingDurationMs")
        if processing_duration_ms is None:
            processing_duration_ms = downstream_duration_ms

        return CompletedExtraction(
            request_id=request_id,
            backend=engine_result.backend,
            summary=self._build_summary(result_json),
            data_info=engine_result.data_info or result_json.get("dataInfo") or data_info,
            json_url=self._result_json_url(request_id),
            page_count=int(result_json.get("pageCount") or 0),
            raw_text=str(result_json.get("rawText") or ""),
            raw_text_plain=str(result_json.get("rawTextPlain") or ""),
            mapping_text=str(result_json.get("mappingText") or ""),
            tables=self._to_response_tables(list(result_json.get("tables") or [])),
            processing_duration_ms=processing_duration_ms,
            detector_confidence=result_json.get("detectorConfidence"),
            engine_version=str(result_json.get("engineVersion") or ""),
            page_metrics=self._to_response_page_metrics(list(result_json.get("pageMetrics") or [])),
            page_selection=self._to_response_page_selection(page_selection),
            table_detection=self._to_response_table_detection(normalized_table_detection),
            stage_timings=self._to_response_stage_timings(stage_timings),
        )

    def load_result_json(self, request_id: str) -> dict[str, Any]:
        return self._storage.read_json(self._storage.resolve_result_json(request_id))

    def resolve_artifact(self, request_id: str, artifact_name: str) -> Path:
        return self._storage.resolve_image_artifact(request_id, artifact_name)

    def _build_result_json(
        self,
        request_id: str,
        engine_result: LocalEngineResult,
        table_detection: TableDetectionResult,
        stage_timings: dict[str, int | None],
    ) -> dict[str, Any]:
        result_json = dict(engine_result.result_json)
        result_json = self._apply_normalized_output(result_json)
        result_json["tableDetection"] = self._to_json_table_detection(table_detection)
        result_json["stageTimings"] = self._to_json_stage_timings(stage_timings)
        result_json.setdefault("dataInfo", engine_result.data_info)
        result_json.setdefault("pageCount", engine_result.page_count)
        result_json.setdefault("rawText", engine_result.raw_text)
        result_json.setdefault("processingDurationMs", engine_result.processing_duration_ms)
        result_json.setdefault("detectorConfidence", engine_result.detector_confidence)
        result_json.setdefault("engineVersion", engine_result.engine_version)
        result_json.setdefault("pageMetrics", engine_result.page_metrics or [])
        if result_json.get("pageSelection") is None:
            result_json["pageSelection"] = self._to_json_page_selection_from_response(engine_result.page_selection)
        result_json.setdefault("artifacts", {})
        result_json["artifacts"]["json_url"] = self._result_json_url(request_id)
        return result_json

    def _apply_normalized_output(self, result_json: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(result_json)
        original_raw_text = str(normalized.get("rawText") or "")
        layout_blocks = self._collect_layout_blocks(normalized)

        if layout_blocks:
            tables = self._extract_tables(layout_blocks)
            raw_text_plain = self._build_text_from_blocks(
                layout_blocks,
                excluded_labels=None,
            )
            mapping_text = self._build_text_from_blocks(
                layout_blocks,
                excluded_labels={label.lower() for label in self._settings.markdown_ignore_labels},
            )
        else:
            tables = []
            raw_text_plain = self._strip_html_to_text(original_raw_text)
            mapping_text = raw_text_plain

        normalized["rawTextRaw"] = original_raw_text
        normalized["rawTextPlain"] = raw_text_plain
        normalized["mappingText"] = mapping_text
        normalized["tables"] = tables
        normalized["rawText"] = mapping_text or raw_text_plain or original_raw_text
        return normalized

    def _collect_layout_blocks(self, result_json: dict[str, Any]) -> list[dict[str, Any]]:
        layout_pages = list(result_json.get("layoutParsingResults") or [])
        blocks: list[dict[str, Any]] = []
        for page_index, page in enumerate(layout_pages):
            pruned_result = page.get("prunedResult") or {}
            parsing_res_list = list(pruned_result.get("parsing_res_list") or [])
            sortable_blocks: list[tuple[tuple[int, int], dict[str, Any]]] = []
            for fallback_index, block in enumerate(parsing_res_list):
                order = block.get("block_order")
                sort_order = (
                    int(order) if isinstance(order, int) or (isinstance(order, str) and order.isdigit()) else 10_000 + fallback_index
                )
                sortable_blocks.append(((sort_order, fallback_index), block))
            for _, block in sorted(sortable_blocks, key=lambda item: item[0]):
                blocks.append(
                    {
                        "page_index": page_index,
                        "block_label": str(block.get("block_label") or ""),
                        "block_content": str(block.get("block_content") or ""),
                        "block_id": block.get("block_id"),
                        "block_order": block.get("block_order"),
                        "block_bbox": list(block.get("block_bbox") or []),
                        "block_polygon_points": list(block.get("block_polygon_points") or []),
                    }
                )
        return blocks

    def _extract_tables(self, layout_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tables: list[dict[str, Any]] = []
        for block in layout_blocks:
            if block["block_label"].lower() != "table":
                continue
            table_html = block["block_content"].strip()
            tables.append(
                {
                    "pageIndex": block["page_index"],
                    "blockId": block.get("block_id"),
                    "blockOrder": block.get("block_order"),
                    "bbox": list(block.get("block_bbox") or []),
                    "polygonPoints": list(block.get("block_polygon_points") or []),
                    "html": table_html,
                    "text": self._html_table_to_plain_text(table_html),
                }
            )
        return tables

    def _build_text_from_blocks(
        self,
        layout_blocks: list[dict[str, Any]],
        excluded_labels: set[str] | None,
    ) -> str:
        parts: list[str] = []
        for block in layout_blocks:
            label = block["block_label"].lower()
            if excluded_labels is not None and label in excluded_labels:
                continue
            content = block["block_content"]
            if not content.strip():
                continue
            if label == "table":
                normalized_text = self._html_table_to_plain_text(content)
            else:
                normalized_text = self._normalize_plain_text(content)
            if normalized_text:
                parts.append(normalized_text)
        return "\n\n".join(parts).strip()

    def _html_table_to_plain_text(self, table_html: str) -> str:
        if not table_html.strip():
            return ""
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
        plain_rows: list[str] = []
        for row_html in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
            normalized_cells = [self._normalize_plain_text(cell) for cell in cells]
            normalized_cells = [cell for cell in normalized_cells if cell]
            if normalized_cells:
                plain_rows.append(" | ".join(normalized_cells))
        if plain_rows:
            return "\n".join(plain_rows)
        return self._strip_html_to_text(table_html)

    def _strip_html_to_text(self, value: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|tr|li|table|thead|tbody|tfoot|ul|ol)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return self._normalize_plain_text(text)

    def _normalize_plain_text(self, value: str) -> str:
        text = html.unescape(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _resolve_route(
        self,
        table_detection: TableDetectionResult,
    ) -> tuple[str, LocalOcrRuntime | LocalVlRuntime]:
        recommended_route = table_detection.recommended_route or "ocr_vl_service"
        if recommended_route == "ocr_service":
            return "ocr_service", self._ocr_runtime
        return "ocr_vl_service", self._vl_runtime

    async def _run_local_engine(
        self,
        runtime: LocalOcrRuntime | LocalVlRuntime,
        payload: bytes,
        filename: str,
        content_type: str | None,
        page_selection: dict[str, Any] | None,
    ) -> LocalEngineResult:
        from io import BytesIO

        from starlette.datastructures import Headers

        upload = UploadFile(
            file=BytesIO(payload),
            filename=filename,
            headers=Headers({"content-type": content_type or "application/octet-stream"}),
        )
        try:
            self._release_inactive_runtime(runtime)
            return await runtime.extract(
                upload,
                page_start=(page_selection or {}).get("page_start"),
                page_end=(page_selection or {}).get("page_end"),
                enforce_sync_limit=False,
            )
        finally:
            try:
                upload.file.close()
            except Exception:
                pass

    def _release_inactive_runtime(
        self,
        selected_runtime: LocalOcrRuntime | LocalVlRuntime,
    ) -> None:
        if selected_runtime is self._ocr_runtime:
            return
        self._ocr_runtime.close()

    def _detect_tables(
        self,
        page_images: list[Image.Image],
        page_indices: list[int],
    ) -> TableDetectionResult:
        try:
            return self._table_detector.detect(page_images, page_indices)
        except Exception as exc:
            LOGGER.warning("Table detection failed, using VL route fallback: %s", exc)
            if not self._settings.table_detector_fail_open:
                raise
            return TableDetectionResult(
                backend=getattr(self._table_detector, "name", "unknown"),
                enabled=True,
                status="failed",
                model_name=self._settings.table_detector_model_name,
                threshold=self._settings.table_detector_threshold,
                document_has_table=None,
                processing_duration_ms=None,
                pages=[],
                recommended_route="ocr_vl_service",
                actual_route="router_service",
                error=str(exc),
            )

    def _apply_actual_route(
        self,
        table_detection: TableDetectionResult,
        route_name: str,
    ) -> TableDetectionResult:
        return TableDetectionResult(
            backend=table_detection.backend,
            enabled=table_detection.enabled,
            status=table_detection.status,
            model_name=table_detection.model_name,
            threshold=table_detection.threshold,
            document_has_table=table_detection.document_has_table,
            processing_duration_ms=table_detection.processing_duration_ms,
            pages=list(table_detection.pages),
            recommended_route=table_detection.recommended_route,
            actual_route=route_name,
            error=table_detection.error,
        )

    def _build_stage_timings(
        self,
        downstream_stage_timings: dict[str, Any] | None,
        *,
        input_prepare_ms: int,
        pdf_render_ms: int | None,
        table_detection_ms: int | None,
        ocr_ms: int | None,
    ) -> dict[str, int | None]:
        if downstream_stage_timings:
            return {
                "input_prepare_ms": input_prepare_ms,
                "pdf_render_ms": pdf_render_ms,
                "table_detection_ms": table_detection_ms,
                "ocr_ms": self._extract_stage_value(downstream_stage_timings, "ocr_ms", "ocrMs", ocr_ms),
                "result_assembly_ms": self._extract_stage_value(
                    downstream_stage_timings,
                    "result_assembly_ms",
                    "resultAssemblyMs",
                    None,
                ),
            }
        return {
            "input_prepare_ms": input_prepare_ms,
            "pdf_render_ms": pdf_render_ms,
            "table_detection_ms": table_detection_ms,
            "ocr_ms": ocr_ms,
            "result_assembly_ms": None,
        }

    def _extract_stage_value(
        self,
        source: dict[str, Any],
        snake_key: str,
        camel_key: str,
        default: int | None,
    ) -> int | None:
        value = source.get(snake_key)
        if value is None:
            value = source.get(camel_key, default)
        return value

    def _build_summary(self, result_json: dict[str, Any]) -> dict[str, int | None]:
        summary = result_json.get("summary")
        if isinstance(summary, dict):
            return {
                "pages": self._safe_int(summary.get("pages"), result_json.get("pageCount")),
                "blocks": self._safe_int(summary.get("blocks"), 0),
                "tables": self._safe_int(summary.get("tables"), 0),
                "lines": self._safe_int(summary.get("lines"), None),
                "words": self._safe_int(summary.get("words"), None),
            }

        layout_pages = result_json.get("layoutParsingResults", []) or []
        pages = len(layout_pages)
        blocks = 0
        tables = 0
        lines = 0
        words = 0

        for page in layout_pages:
            pruned_result = page.get("prunedResult", {}) or {}
            parsing_res_list = pruned_result.get("parsing_res_list", []) or []
            ocr_res_list = pruned_result.get("ocr_res_list", []) or []
            blocks += len(parsing_res_list) or len(ocr_res_list)
            tables += sum(
                1 for item in parsing_res_list if item.get("block_label") == "table"
            )
            lines += len(ocr_res_list)
            words += sum(len((item.get("text") or "").split()) for item in ocr_res_list)

        return {
            "pages": pages,
            "blocks": blocks,
            "tables": tables,
            "lines": lines or None,
            "words": words or None,
        }

    def _safe_int(self, value: Any, default: int | None) -> int | None:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _read_input_info(self, input_path: Path) -> dict[str, Any]:
        if input_path.suffix.lower() == ".pdf":
            return self._read_pdf_info(input_path)
        return self._read_image_info(input_path)

    def _load_image_pages(self, input_path: Path) -> list[Image.Image]:
        try:
            with Image.open(input_path) as image:
                image.load()
                return [image.convert("RGB").copy()]
        except Exception as exc:
            raise InvalidUploadError(
                f"Unable to open `{input_path.name}` as an image: {exc}"
            ) from exc

    def _read_image_info(self, image_path: Path) -> dict[str, Any]:
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
        except Exception as exc:
            raise InvalidUploadError(
                f"Unable to open `{image_path.name}` as an image: {exc}"
            ) from exc
        return {
            "width": width,
            "height": height,
            "type": "image",
            "page_count": 1,
            "pages": [{"page_index": 0, "width": width, "height": height}],
        }

    def _read_pdf_info(self, pdf_path: Path) -> dict[str, Any]:
        try:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                pages: list[dict[str, int | None]] = []
                for page_index in range(len(document)):
                    page = document[page_index]
                    try:
                        width, height = page.get_size()
                        pages.append(
                            {
                                "page_index": page_index,
                                "width": int(round(width)),
                                "height": int(round(height)),
                            }
                        )
                    finally:
                        page.close()
            finally:
                document.close()
        except Exception as exc:
            raise InvalidUploadError(
                f"Unable to open `{pdf_path.name}` as a PDF: {exc}"
            ) from exc

        return {
            "width": None,
            "height": None,
            "type": "pdf",
            "page_count": len(pages),
            "pages": pages,
        }

    def _normalize_page_selection(
        self,
        input_path: Path,
        data_info: dict[str, Any],
        page_start: int | None,
        page_end: int | None,
    ) -> dict[str, Any] | None:
        is_pdf = input_path.suffix.lower() == ".pdf"
        if not is_pdf:
            if page_start is not None or page_end is not None:
                raise InvalidUploadError(
                    "`page_start` and `page_end` can only be used with PDF uploads."
                )
            return None

        total_page_count = int(data_info.get("page_count") or 0)
        if total_page_count <= 0:
            raise InvalidUploadError(f"`{input_path.name}` does not contain any PDF pages.")

        if page_end is not None and page_start is None:
            raise InvalidUploadError("`page_start` is required when `page_end` is provided.")

        if page_start is None:
            page_start = 1
            page_end = total_page_count
        elif page_end is None:
            page_end = page_start

        if page_start < 1:
            raise InvalidUploadError("`page_start` must be greater than or equal to 1.")
        if page_end < page_start:
            raise InvalidUploadError("`page_end` must be greater than or equal to `page_start`.")
        if page_end > total_page_count:
            raise InvalidUploadError(
                f"`page_end` must be less than or equal to the PDF page count ({total_page_count})."
            )

        return {
            "page_start": page_start,
            "page_end": page_end,
            "selected_page_count": page_end - page_start + 1,
            "total_page_count": total_page_count,
            "selected_page_indices": list(range(page_start - 1, page_end)),
        }

    def _validate_sync_input_limits(
        self,
        input_path: Path,
        data_info: dict[str, Any],
        page_selection: dict[str, Any] | None,
    ) -> None:
        if input_path.suffix.lower() != ".pdf":
            return

        page_count = int(data_info.get("page_count") or 0)
        if page_count <= 0:
            raise InvalidUploadError(f"`{input_path.name}` does not contain any PDF pages.")

        selected_page_count = int((page_selection or {}).get("selected_page_count") or page_count)
        if selected_page_count > self._settings.max_sync_pdf_pages:
            page_start = (page_selection or {}).get("page_start", 1)
            page_end = (page_selection or {}).get("page_end", page_count)
            raise InvalidUploadError(
                f"`{input_path.name}` selected pages {page_start}-{page_end} ({selected_page_count} pages), which exceeds the synchronous PDF limit of {self._settings.max_sync_pdf_pages} pages. Choose a smaller `page_start`/`page_end` range or move this flow to async processing."
            )

    def _render_pdf_previews(
        self,
        pdf_path: Path,
        selected_page_indices: list[int] | None = None,
    ) -> list[Image.Image]:
        document = pdfium.PdfDocument(str(pdf_path))
        previews: list[Image.Image] = []
        try:
            page_indices = selected_page_indices or list(range(len(document)))
            for page_index in page_indices:
                page = document[page_index]
                try:
                    bitmap = page.render(scale=2.0)
                    previews.append(bitmap.to_pil().convert("RGB").copy())
                finally:
                    page.close()
        finally:
            document.close()
        return previews

    def _resolve_input_suffix(self, filename: str, content_type: str | None) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in ALLOWED_DOCUMENT_EXTENSIONS:
            return suffix
        guessed_suffix = mimetypes.guess_extension(content_type or "")
        if guessed_suffix in ALLOWED_DOCUMENT_EXTENSIONS:
            return guessed_suffix
        guessed_from_name = mimetypes.guess_type(filename)[0]
        guessed_suffix = mimetypes.guess_extension(guessed_from_name or "")
        if guessed_suffix in ALLOWED_DOCUMENT_EXTENSIONS:
            return guessed_suffix
        raise InvalidUploadError("Only image and PDF uploads are supported in this version.")

    def _generate_request_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{self._settings.request_id_prefix}_{timestamp}_{uuid4().hex[:8]}"

    def _result_json_url(self, request_id: str) -> str:
        return f"{self._settings.api_prefix}/results/{request_id}/json"

    def _to_response_page_selection(
        self,
        page_selection: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if page_selection is None:
            return None
        return {
            "page_start": page_selection["page_start"],
            "page_end": page_selection["page_end"],
            "selected_page_count": page_selection["selected_page_count"],
            "total_page_count": page_selection["total_page_count"],
        }

    def _to_json_page_selection_from_response(
        self,
        page_selection: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if page_selection is None:
            return None
        if "pageStart" in page_selection:
            return page_selection
        return {
            "pageStart": page_selection.get("page_start"),
            "pageEnd": page_selection.get("page_end"),
            "selectedPageCount": page_selection.get("selected_page_count"),
            "totalPageCount": page_selection.get("total_page_count"),
        }

    def _to_response_stage_timings(
        self,
        stage_timings: dict[str, int | None],
    ) -> dict[str, int | None]:
        return {
            "input_prepare_ms": stage_timings.get("input_prepare_ms"),
            "pdf_render_ms": stage_timings.get("pdf_render_ms"),
            "table_detection_ms": stage_timings.get("table_detection_ms"),
            "ocr_ms": stage_timings.get("ocr_ms"),
            "result_assembly_ms": stage_timings.get("result_assembly_ms"),
        }

    def _to_response_page_metrics(
        self,
        page_metrics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for metric in page_metrics:
            normalized.append(
                {
                    "page_index": metric.get("page_index", metric.get("pageIndex")),
                    "processing_duration_ms": metric.get(
                        "processing_duration_ms",
                        metric.get("processingDurationMs"),
                    ),
                    "detector_confidence": metric.get(
                        "detector_confidence",
                        metric.get("detectorConfidence"),
                    ),
                    "block_count": metric.get("block_count", metric.get("blockCount")),
                    "line_count": metric.get("line_count", metric.get("lineCount")),
                    "word_count": metric.get("word_count", metric.get("wordCount")),
                }
            )
        return normalized

    def _to_response_tables(
        self,
        tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for table in tables:
            normalized.append(
                {
                    "page_index": table.get("page_index", table.get("pageIndex")),
                    "block_id": table.get("block_id", table.get("blockId")),
                    "block_order": table.get("block_order", table.get("blockOrder")),
                    "bbox": table.get("bbox"),
                    "polygon_points": table.get(
                        "polygon_points",
                        table.get("polygonPoints"),
                    ),
                    "html": table.get("html", ""),
                    "text": table.get("text", ""),
                }
            )
        return normalized

    def _to_json_stage_timings(
        self,
        stage_timings: dict[str, int | None],
    ) -> dict[str, int | None]:
        return {
            "inputPrepareMs": stage_timings.get("input_prepare_ms"),
            "pdfRenderMs": stage_timings.get("pdf_render_ms"),
            "tableDetectionMs": stage_timings.get("table_detection_ms"),
            "ocrMs": stage_timings.get("ocr_ms"),
            "resultAssemblyMs": stage_timings.get("result_assembly_ms"),
        }

    def _to_response_table_detection(
        self,
        table_detection: TableDetectionResult,
    ) -> dict[str, Any]:
        return {
            "backend": table_detection.backend,
            "enabled": table_detection.enabled,
            "status": table_detection.status,
            "model_name": table_detection.model_name,
            "threshold": table_detection.threshold,
            "document_has_table": table_detection.document_has_table,
            "processing_duration_ms": table_detection.processing_duration_ms,
            "recommended_route": table_detection.recommended_route,
            "actual_route": table_detection.actual_route,
            "error": table_detection.error,
            "pages": [
                {
                    "page_index": page.page_index,
                    "has_table": page.has_table,
                    "score": page.score,
                    "label": page.label,
                }
                for page in table_detection.pages
            ],
        }

    def _to_json_table_detection(
        self,
        table_detection: TableDetectionResult,
    ) -> dict[str, Any]:
        return {
            "backend": table_detection.backend,
            "enabled": table_detection.enabled,
            "status": table_detection.status,
            "modelName": table_detection.model_name,
            "threshold": table_detection.threshold,
            "documentHasTable": table_detection.document_has_table,
            "processingDurationMs": table_detection.processing_duration_ms,
            "recommendedRoute": table_detection.recommended_route,
            "actualRoute": table_detection.actual_route,
            "error": table_detection.error,
            "pages": [
                {
                    "pageIndex": page.page_index,
                    "hasTable": page.has_table,
                    "score": page.score,
                    "label": page.label,
                }
                for page in table_detection.pages
            ],
        }
