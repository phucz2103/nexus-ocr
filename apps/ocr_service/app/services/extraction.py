from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import pypdfium2 as pdfium
from fastapi import UploadFile
from PIL import Image

from app.core.config import Settings
from app.core.errors import InferenceFailedError, InvalidUploadError
from app.inference.base import InferenceBackend, InferencePage
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
    summary: dict[str, int]
    data_info: dict[str, Any]
    json_url: str
    input_url: str
    page_count: int
    page_selection: dict[str, Any] | None


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

        pages, result_page_indices, backend_name = self._extract_document(
            paths.input_file,
            data_info,
            page_selection,
        )
        result_json = self._assemble_result(
            request_id=request_id,
            data_info=data_info,
            pages=pages,
            result_page_indices=result_page_indices,
        )

        self._storage.write_json(paths.result_json, result_json)

        return CompletedExtraction(
            request_id=request_id,
            backend=backend_name,
            summary=self._build_summary(result_json),
            data_info=data_info,
            json_url=self._result_json_url(request_id),
            input_url=self._input_url(request_id),
            page_count=result_json["pageCount"],
            page_selection=(
                self._to_response_page_selection(page_selection)
                if page_selection is not None
                else None
            ),
        )

    def load_result_json(self, request_id: str) -> dict[str, Any]:
        return self._storage.read_json(self._storage.resolve_result_json(request_id))

    def resolve_input_file(self, request_id: str) -> Path:
        return self._storage.resolve_input_file(request_id)

    def _extract_document(
        self,
        input_path: Path,
        data_info: dict[str, Any],
        page_selection: dict[str, Any] | None,
    ) -> tuple[list[InferencePage], list[int] | None, str]:
        if data_info.get("type") != "pdf":
            inference_result = self._backend.extract(input_path)
            return inference_result.pages, None, inference_result.backend

        selected_page_indices = list((page_selection or {}).get("selected_page_indices") or [])
        if not selected_page_indices:
            selected_page_indices = list(range(int(data_info.get("page_count") or 0)))

        page_images = self._render_pdf_pages(input_path, selected_page_indices)
        all_pages: list[InferencePage] = []
        result_page_indices: list[int] = []
        backend_name: str | None = None

        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for original_page_index, page_image in zip(selected_page_indices, page_images):
                page_path = temp_root / f"page_{original_page_index + 1}.png"
                page_image.save(page_path)
                page_result = self._backend.extract(page_path)
                backend_name = backend_name or page_result.backend
                all_pages.extend(page_result.pages)
                result_page_indices.extend([original_page_index] * len(page_result.pages))

        return all_pages, result_page_indices, backend_name or getattr(self._backend, 'name', self._settings.backend)

    def _assemble_result(
        self,
        request_id: str,
        data_info: dict[str, Any],
        pages: list[InferencePage],
        result_page_indices: list[int] | None,
    ) -> dict[str, Any]:
        if not pages:
            raise InferenceFailedError("Inference produced no pages.")

        ocr_results: list[dict[str, Any]] = []
        preprocessed_images: list[str] = []
        page_count = len(pages)

        for local_page_index, page in enumerate(pages):
            page_index = (
                result_page_indices[local_page_index]
                if result_page_indices is not None and local_page_index < len(result_page_indices)
                else local_page_index
            )
            page_pruned_result = dict(page.pruned_result)
            page_pruned_result["page_count"] = page_count
            page_pruned_result["page_index"] = page_index

            input_image_url, page_output_images, page_preprocessed_images = (
                self._persist_page_artifacts(request_id, page_index, page)
            )

            page_markdown = {
                "text": page.markdown.get("text", "") if self._settings.generate_markdown else "",
                "images": {},
            }
            ocr_results.append(
                {
                    "prunedResult": page_pruned_result,
                    "markdown": page_markdown,
                    "outputImages": page_output_images,
                    "inputImage": input_image_url,
                }
            )
            preprocessed_images.extend(page_preprocessed_images)

        result_json = {
            "ocrParsingResults": ocr_results,
            "dataInfo": data_info,
            "preprocessedImages": preprocessed_images,
            "pageCount": page_count,
        }

        return result_json

    def _persist_page_artifacts(
        self,
        request_id: str,
        page_index: int,
        page: InferencePage,
    ) -> tuple[str, dict[str, str], list[str]]:
        output_images: dict[str, str] = {}
        preprocessed_images: list[str] = []
        input_image_url = self._input_url(request_id)
        if not self._settings.save_artifact_images:
            return input_image_url, output_images, preprocessed_images

        for image_key, image_value in page.images.items():
            if image_key.startswith("input_img"):
                continue

            artifact_name = self._storage.save_image_artifact(
                request_id,
                f"page_{page_index}_{image_key}",
                image_value,
            )
            artifact_url = self._artifact_url(request_id, artifact_name)

            if image_key.startswith("preprocessed_img"):
                preprocessed_images.append(artifact_url)
            else:
                output_images[image_key] = artifact_url
        return input_image_url, output_images, preprocessed_images

    def _build_summary(self, result_json: dict[str, Any]) -> dict[str, int]:
        pages = result_json.get("ocrParsingResults", [])
        lines = 0
        words = 0

        for page in pages:
            ocr_res_list = page.get("prunedResult", {}).get("ocr_res_list", []) or []
            lines += len(ocr_res_list)
            words += sum(
                len((item.get("text") or "").split())
                for item in ocr_res_list
            )

        return {
            "pages": len(pages),
            "lines": lines,
            "words": words,
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

    def _to_response_page_selection(
        self,
        page_selection: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "page_start": page_selection["page_start"],
            "page_end": page_selection["page_end"],
            "selected_page_count": page_selection["selected_page_count"],
            "total_page_count": page_selection["total_page_count"],
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

    def _render_pdf_pages(
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
                    bitmap = page.render(scale=self._settings.pdf_render_scale)
                    previews.append(bitmap.to_pil().convert("RGB").copy())
                finally:
                    page.close()
        finally:
            document.close()
        return previews

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

    def _input_url(self, request_id: str) -> str:
        return f"{self._settings.api_prefix}/results/{request_id}/input"

