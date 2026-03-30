from __future__ import annotations

import mimetypes
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
from app.core.errors import InferenceFailedError, InvalidUploadError
from app.internal.base import InferenceBackend, InferencePage, InferenceRunResult
from app.services.storage import StorageService


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
    markdown_url: str
    input_url: str
    page_count: int
    raw_text: str
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
        backend: InferenceBackend,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._backend = backend

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
        input_suffix = self._resolve_input_suffix(file)
        request_id = self._generate_request_id()
        paths = self._storage.prepare_request(request_id, input_suffix)
        self._storage.write_bytes(paths.input_file, payload)

        data_info = self._read_input_info(paths.input_file)
        page_selection = self._normalize_page_selection(
            paths.input_file,
            data_info,
            page_start,
            page_end,
        )
        if enforce_sync_limit:
            self._validate_sync_input_limits(paths.input_file, data_info, page_selection)
        input_prepare_ms = max(1, int(round((perf_counter() - input_prepare_started_at) * 1000)))

        selected_page_indices = (
            list(page_selection["selected_page_indices"])
            if page_selection is not None
            else None
        )

        should_render_pdf_previews = (
            data_info["type"] == "pdf"
            and (
                self._settings.save_artifact_images
                or (
                    page_selection is not None
                    and page_selection["selected_page_count"] != page_selection["total_page_count"]
                )
            )
        )

        pdf_render_ms: int | None = None
        input_page_images: list[Image.Image] | None = None
        if should_render_pdf_previews:
            pdf_render_started_at = perf_counter()
            input_page_images = self._render_pdf_previews(paths.input_file, selected_page_indices)
            pdf_render_ms = max(1, int(round((perf_counter() - pdf_render_started_at) * 1000)))

        ocr_started_at = perf_counter()
        inference_result, result_page_indices = self._extract_document(
            paths.input_file,
            data_info,
            page_selection,
            input_page_images,
        )
        processing_duration_ms = inference_result.processing_duration_ms
        if processing_duration_ms is None:
            processing_duration_ms = max(1, int(round((perf_counter() - ocr_started_at) * 1000)))
        engine_version = inference_result.engine_version or self._settings.engine_version

        result_assembly_started_at = perf_counter()
        stage_timings = {
            "input_prepare_ms": input_prepare_ms,
            "pdf_render_ms": pdf_render_ms,
            "table_detection_ms": None,
            "ocr_ms": processing_duration_ms,
            "result_assembly_ms": None,
        }
        (
            result_json,
            result_markdown,
            raw_text,
            detector_confidence,
            page_metrics,
        ) = self._assemble_result(
            request_id=request_id,
            data_info=data_info,
            pages=inference_result.pages,
            input_page_images=input_page_images,
            processing_duration_ms=processing_duration_ms,
            engine_version=engine_version,
            result_page_indices=result_page_indices,
            page_selection=page_selection,
            stage_timings=stage_timings,
        )
        stage_timings["result_assembly_ms"] = max(
            1,
            int(round((perf_counter() - result_assembly_started_at) * 1000)),
        )
        result_json["stageTimings"] = self._to_json_stage_timings(stage_timings)

        self._storage.write_json(paths.result_json, result_json)
        self._storage.write_text(paths.result_markdown, result_markdown)

        table_detection = self._build_disabled_table_detection()
        return CompletedExtraction(
            request_id=request_id,
            backend=inference_result.backend,
            summary=self._build_summary(result_json),
            data_info=data_info,
            json_url=self._result_json_url(request_id),
            markdown_url=self._result_markdown_url(request_id),
            input_url=self._input_url(request_id),
            page_count=result_json["pageCount"],
            raw_text=raw_text,
            processing_duration_ms=processing_duration_ms,
            detector_confidence=detector_confidence,
            engine_version=engine_version,
            page_metrics=page_metrics,
            page_selection=(
                self._to_response_page_selection(page_selection)
                if page_selection is not None
                else None
            ),
            table_detection=table_detection,
            stage_timings=self._to_response_stage_timings(stage_timings),
        )

    def load_result_json(self, request_id: str) -> dict[str, Any]:
        return self._storage.read_json(self._storage.resolve_result_json(request_id))

    def resolve_result_markdown(self, request_id: str) -> Path:
        return self._storage.resolve_result_markdown(request_id)

    def resolve_input_file(self, request_id: str) -> Path:
        return self._storage.resolve_input_file(request_id)

    def resolve_artifact(self, request_id: str, artifact_name: str) -> Path:
        return self._storage.resolve_image_artifact(request_id, artifact_name)

    def _extract_document(
        self,
        input_path: Path,
        data_info: dict[str, Any],
        page_selection: dict[str, Any] | None,
        input_page_images: list[Image.Image] | None,
    ) -> tuple[InferenceRunResult, list[int] | None]:
        if data_info.get("type") != "pdf":
            return self._backend.extract(input_path), None

        selected_page_indices = list((page_selection or {}).get("selected_page_indices") or [])
        if not selected_page_indices:
            selected_page_indices = list(range(int(data_info.get("page_count") or 0)))

        page_images = input_page_images or self._render_pdf_previews(input_path, selected_page_indices)
        all_pages: list[InferencePage] = []
        result_page_indices: list[int] = []
        backend_name: str | None = None
        engine_version: str | None = None
        total_duration_ms = 0
        has_duration = False

        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for original_page_index, page_image in zip(selected_page_indices, page_images):
                page_path = temp_root / f"page_{original_page_index + 1}.png"
                page_image.save(page_path)
                page_result = self._backend.extract(page_path)
                backend_name = backend_name or page_result.backend
                engine_version = engine_version or page_result.engine_version
                if page_result.processing_duration_ms is not None:
                    total_duration_ms += page_result.processing_duration_ms
                    has_duration = True
                all_pages.extend(page_result.pages)
                result_page_indices.extend([original_page_index] * len(page_result.pages))

        return (
            InferenceRunResult(
                backend=backend_name or getattr(self._backend, "name", self._settings.backend),
                pages=all_pages,
                processing_duration_ms=(total_duration_ms if has_duration else None),
                engine_version=engine_version or self._settings.engine_version,
            ),
            result_page_indices,
        )

    def _assemble_result(
        self,
        request_id: str,
        data_info: dict[str, Any],
        pages: list[InferencePage],
        input_page_images: list[Image.Image] | None,
        processing_duration_ms: int,
        engine_version: str,
        result_page_indices: list[int] | None,
        page_selection: dict[str, Any] | None,
        stage_timings: dict[str, int | None],
    ) -> tuple[dict[str, Any], str, str, float | None, list[dict[str, Any]]]:
        if not pages:
            raise InferenceFailedError("Inference produced no pages.")

        layout_results: list[dict[str, Any]] = []
        preprocessed_images: list[str] = []
        markdown_pages: list[str] = []
        raw_text_pages: list[str] = []
        page_metrics: list[dict[str, Any]] = []
        page_count = len(pages)
        is_pdf = data_info.get("type") == "pdf"

        for local_page_index, page in enumerate(pages):
            page_index = (
                result_page_indices[local_page_index]
                if result_page_indices is not None and local_page_index < len(result_page_indices)
                else local_page_index
            )
            page_pruned_result = dict(page.pruned_result)
            page_pruned_result["page_count"] = page_count
            page_pruned_result["page_index"] = page_index

            input_image_url, page_output_images, page_preprocessed_images = self._persist_page_artifacts(
                request_id=request_id,
                artifact_page_index=page_index,
                page=page,
                input_page_image=(
                    input_page_images[local_page_index]
                    if input_page_images is not None and local_page_index < len(input_page_images)
                    else None
                ),
                prefer_original_input=not is_pdf,
            )

            page_markdown = {
                "text": page.markdown.get("text", "") if self._settings.generate_markdown else "",
                "images": {},
            }
            layout_page = {
                "prunedResult": page_pruned_result,
                "markdown": page_markdown,
                "outputImages": page_output_images,
                "inputImage": input_image_url,
            }
            layout_results.append(layout_page)
            preprocessed_images.extend(page_preprocessed_images)

            markdown_text = page_markdown["text"].strip()
            if markdown_text:
                markdown_pages.append(markdown_text)

            page_raw_text = self._extract_page_raw_text(page_pruned_result)
            if page_raw_text:
                raw_text_pages.append(page_raw_text)

            page_metric = self._normalize_page_metric(page_index, page, page_pruned_result)
            page_metrics.append(page_metric)

        result_markdown = ""
        if self._settings.generate_markdown:
            result_markdown = "\n\n---\n\n".join(markdown_pages).strip()
            if result_markdown:
                result_markdown += "\n"

        raw_text = "\n\n".join(raw_text_pages).strip()
        detector_confidence = self._aggregate_detector_confidence(page_metrics)
        table_detection = self._build_disabled_table_detection_json()

        result_json = {
            "layoutParsingResults": layout_results,
            "ocrParsingResults": layout_results,
            "dataInfo": data_info,
            "preprocessedImages": preprocessed_images,
            "pageCount": page_count,
            "rawText": raw_text,
            "processingDurationMs": processing_duration_ms,
            "detectorConfidence": detector_confidence,
            "engineVersion": engine_version,
            "pageMetrics": [self._to_json_page_metric(item) for item in page_metrics],
            "pageSelection": (
                self._to_json_page_selection(page_selection)
                if page_selection is not None
                else None
            ),
            "tableDetection": table_detection,
            "stageTimings": self._to_json_stage_timings(stage_timings),
        }

        return result_json, result_markdown, raw_text, detector_confidence, page_metrics

    def _persist_page_artifacts(
        self,
        request_id: str,
        artifact_page_index: int,
        page: InferencePage,
        input_page_image: Image.Image | None,
        prefer_original_input: bool,
    ) -> tuple[str, dict[str, str], list[str]]:
        output_images: dict[str, str] = {}
        preprocessed_images: list[str] = []
        input_image_url = self._input_url(request_id)

        if not self._settings.save_artifact_images:
            return input_image_url, output_images, preprocessed_images

        if input_page_image is not None:
            artifact_name = self._storage.save_image_artifact(
                request_id,
                f"page_{artifact_page_index}_input_img",
                input_page_image,
            )
            input_image_url = self._artifact_url(request_id, artifact_name)

        for image_key, image_value in page.images.items():
            if image_key.startswith("input_img"):
                if prefer_original_input or input_page_image is not None:
                    continue
                artifact_name = self._storage.save_image_artifact(
                    request_id,
                    f"page_{artifact_page_index}_{image_key}",
                    image_value,
                )
                input_image_url = self._artifact_url(request_id, artifact_name)
                continue

            artifact_name = self._storage.save_image_artifact(
                request_id,
                f"page_{artifact_page_index}_{image_key}",
                image_value,
            )
            artifact_url = self._artifact_url(request_id, artifact_name)
            if image_key.startswith("preprocessed_img"):
                preprocessed_images.append(artifact_url)
            else:
                output_images[image_key] = artifact_url

        return input_image_url, output_images, preprocessed_images

    def _build_summary(self, result_json: dict[str, Any]) -> dict[str, int | None]:
        pages = result_json.get("layoutParsingResults", [])
        lines = 0
        words = 0

        for page in pages:
            ocr_res_list = page.get("prunedResult", {}).get("ocr_res_list", []) or []
            lines += len(ocr_res_list)
            words += sum(len((item.get("text") or "").split()) for item in ocr_res_list)

        return {
            "pages": len(pages),
            "blocks": lines,
            "tables": 0,
            "lines": lines,
            "words": words,
        }

    def _extract_page_raw_text(self, page_pruned_result: dict[str, Any]) -> str:
        full_text = page_pruned_result.get("full_text") or ""
        if isinstance(full_text, str) and full_text.strip():
            return full_text.strip()
        fragments = [
            (item.get("text") or "").strip()
            for item in (page_pruned_result.get("ocr_res_list") or [])
            if (item.get("text") or "").strip()
        ]
        return "\n".join(fragments)

    def _normalize_page_metric(
        self,
        page_index: int,
        page: InferencePage,
        page_pruned_result: dict[str, Any],
    ) -> dict[str, Any]:
        metric = dict(getattr(page, "metrics", None) or {})
        line_items = page_pruned_result.get("ocr_res_list", []) or []
        line_count = len(line_items)
        word_count = sum(len((item.get("text") or "").split()) for item in line_items)
        metric["page_index"] = page_pruned_result.get("page_index", page_index)
        metric.setdefault("processing_duration_ms", None)
        metric.setdefault("detector_confidence", self._compute_detector_confidence(page_pruned_result))
        metric.setdefault("block_count", line_count)
        metric.setdefault("line_count", line_count)
        metric.setdefault("word_count", word_count)
        return metric

    def _compute_detector_confidence(self, page_pruned_result: dict[str, Any]) -> float | None:
        scores = [
            float(item.get("rec_score"))
            for item in (page_pruned_result.get("ocr_res_list") or [])
            if isinstance(item.get("rec_score"), (int, float))
        ]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    def _aggregate_detector_confidence(self, page_metrics: list[dict[str, Any]]) -> float | None:
        scores = [
            float(item["detector_confidence"])
            for item in page_metrics
            if item.get("detector_confidence") is not None
        ]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    def _build_disabled_table_detection(self) -> dict[str, Any]:
        return {
            "backend": "disabled",
            "enabled": False,
            "status": "disabled",
            "model_name": "disabled",
            "threshold": None,
            "document_has_table": None,
            "processing_duration_ms": None,
            "recommended_route": None,
            "actual_route": "ocr_service",
            "error": None,
            "pages": [],
        }

    def _build_disabled_table_detection_json(self) -> dict[str, Any]:
        return {
            "backend": "disabled",
            "enabled": False,
            "status": "disabled",
            "modelName": "disabled",
            "threshold": None,
            "documentHasTable": None,
            "processingDurationMs": None,
            "recommendedRoute": None,
            "actualRoute": "ocr_service",
            "error": None,
            "pages": [],
        }

    def _to_json_page_metric(self, metric: dict[str, Any]) -> dict[str, Any]:
        return {
            "pageIndex": metric.get("page_index"),
            "processingDurationMs": metric.get("processing_duration_ms"),
            "detectorConfidence": metric.get("detector_confidence"),
            "blockCount": metric.get("block_count"),
            "lineCount": metric.get("line_count"),
            "wordCount": metric.get("word_count"),
        }

    def _to_response_page_selection(self, page_selection: dict[str, Any]) -> dict[str, Any]:
        return {
            "page_start": page_selection["page_start"],
            "page_end": page_selection["page_end"],
            "selected_page_count": page_selection["selected_page_count"],
            "total_page_count": page_selection["total_page_count"],
        }

    def _to_json_page_selection(self, page_selection: dict[str, Any]) -> dict[str, Any]:
        return {
            "pageStart": page_selection["page_start"],
            "pageEnd": page_selection["page_end"],
            "selectedPageCount": page_selection["selected_page_count"],
            "totalPageCount": page_selection["total_page_count"],
        }

    def _to_response_stage_timings(self, stage_timings: dict[str, int | None]) -> dict[str, int | None]:
        return {
            "input_prepare_ms": stage_timings.get("input_prepare_ms"),
            "pdf_render_ms": stage_timings.get("pdf_render_ms"),
            "table_detection_ms": stage_timings.get("table_detection_ms"),
            "ocr_ms": stage_timings.get("ocr_ms"),
            "result_assembly_ms": stage_timings.get("result_assembly_ms"),
        }

    def _to_json_stage_timings(self, stage_timings: dict[str, int | None]) -> dict[str, int | None]:
        return {
            "inputPrepareMs": stage_timings.get("input_prepare_ms"),
            "pdfRenderMs": stage_timings.get("pdf_render_ms"),
            "tableDetectionMs": stage_timings.get("table_detection_ms"),
            "ocrMs": stage_timings.get("ocr_ms"),
            "resultAssemblyMs": stage_timings.get("result_assembly_ms"),
        }

    def _read_input_info(self, input_path: Path) -> dict[str, Any]:
        if input_path.suffix.lower() == ".pdf":
            return self._read_pdf_info(input_path)
        return self._read_image_info(input_path)

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
            raise InvalidUploadError(
                "`page_start` is required when `page_end` is provided."
            )

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

        selected_page_count = page_end - page_start + 1
        return {
            "page_start": page_start,
            "page_end": page_end,
            "selected_page_count": selected_page_count,
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
                previews.append(self._render_pdf_page(document, page_index))
        finally:
            document.close()
        return previews

    def _render_pdf_page(self, document: pdfium.PdfDocument, page_index: int) -> Image.Image:
        page = document[page_index]
        try:
            bitmap = page.render(scale=self._settings.pdf_render_scale)
            return bitmap.to_pil().convert("RGB").copy()
        finally:
            page.close()

    def _resolve_input_suffix(self, file: UploadFile) -> str:
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix in ALLOWED_DOCUMENT_EXTENSIONS:
            return suffix

        content_type = file.content_type or mimetypes.guess_type(filename)[0]
        guessed_suffix = mimetypes.guess_extension(content_type or "")
        if guessed_suffix in ALLOWED_DOCUMENT_EXTENSIONS:
            return guessed_suffix

        raise InvalidUploadError("Only image and PDF uploads are supported in this version.")

    def _generate_request_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{self._settings.request_id_prefix}_{timestamp}_{uuid4().hex[:8]}"

    def _result_json_url(self, request_id: str) -> str:
        return f"{self._settings.api_prefix}/results/{request_id}/json"

    def _result_markdown_url(self, request_id: str) -> str:
        return f"{self._settings.api_prefix}/results/{request_id}/markdown"

    def _input_url(self, request_id: str) -> str:
        return f"{self._settings.api_prefix}/results/{request_id}/input"

    def _artifact_url(self, request_id: str, artifact_name: str) -> str:
        return f"{self._settings.api_prefix}/results/{request_id}/artifacts/{artifact_name}"

