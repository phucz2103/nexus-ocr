from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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


@dataclass(slots=True)
class CompletedExtraction:
    request_id: str
    backend: str
    summary: dict[str, int]
    data_info: dict[str, Any]
    json_url: str
    markdown_url: str
    input_url: str


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

    async def extract(self, file: UploadFile) -> CompletedExtraction:
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

        data_info = self._read_image_info(paths.input_file)
        inference_result = self._backend.extract(paths.input_file)
        result_json, result_markdown = self._assemble_result(
            request_id=request_id,
            data_info=data_info,
            pages=inference_result.pages,
        )

        self._storage.write_json(paths.result_json, result_json)
        self._storage.write_text(paths.result_markdown, result_markdown)

        return CompletedExtraction(
            request_id=request_id,
            backend=inference_result.backend,
            summary=self._build_summary(result_json),
            data_info=data_info,
            json_url=self._result_json_url(request_id),
            markdown_url=self._result_markdown_url(request_id),
            input_url=self._input_url(request_id),
        )

    def load_result_json(self, request_id: str) -> dict[str, Any]:
        return self._storage.read_json(self._storage.resolve_result_json(request_id))

    def resolve_result_markdown(self, request_id: str) -> Path:
        return self._storage.resolve_result_markdown(request_id)

    def resolve_input_file(self, request_id: str) -> Path:
        return self._storage.resolve_input_file(request_id)

    def resolve_artifact(self, request_id: str, artifact_name: str) -> Path:
        return self._storage.resolve_image_artifact(request_id, artifact_name)

    def _assemble_result(
        self,
        request_id: str,
        data_info: dict[str, Any],
        pages: list[InferencePage],
    ) -> tuple[dict[str, Any], str]:
        if not pages:
            raise InferenceFailedError("Inference produced no pages.")

        ocr_results: list[dict[str, Any]] = []
        preprocessed_images: list[str] = []
        markdown_pages: list[str] = []

        for page_index, page in enumerate(pages):
            (
                input_image_url,
                page_output_images,
                page_preprocessed_images,
                markdown_image_urls,
            ) = self._persist_page_artifacts(request_id, page_index, page)

            page_markdown = {
                "text": page.markdown.get("text", ""),
                "images": markdown_image_urls,
            }
            ocr_results.append(
                {
                    "prunedResult": page.pruned_result,
                    "markdown": page_markdown,
                    "outputImages": page_output_images,
                    "inputImage": input_image_url,
                }
            )
            preprocessed_images.extend(page_preprocessed_images)

            markdown_text = page_markdown["text"].strip()
            if markdown_text:
                markdown_pages.append(markdown_text)

        result_json = {
            "ocrParsingResults": ocr_results,
            "dataInfo": data_info,
            "preprocessedImages": preprocessed_images,
        }

        result_markdown = "\n\n---\n\n".join(markdown_pages).strip()
        if result_markdown:
            result_markdown += "\n"

        return result_json, result_markdown

    def _persist_page_artifacts(
        self,
        request_id: str,
        page_index: int,
        page: InferencePage,
    ) -> tuple[str, dict[str, str], list[str], dict[str, str]]:
        output_images: dict[str, str] = {}
        preprocessed_images: list[str] = []
        input_image_url = self._input_url(request_id)

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

        markdown_image_urls: dict[str, str] = {}
        for markdown_key, markdown_value in page.markdown.get("images", {}).items():
            artifact_name = self._storage.save_image_artifact(
                request_id,
                f"page_{page_index}_markdown_{Path(markdown_key).stem}",
                markdown_value,
            )
            markdown_image_urls[markdown_key] = self._artifact_url(
                request_id,
                artifact_name,
            )

        return input_image_url, output_images, preprocessed_images, markdown_image_urls

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
        }

    def _resolve_input_suffix(self, file: UploadFile) -> str:
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix in ALLOWED_IMAGE_EXTENSIONS:
            return suffix

        content_type = file.content_type or mimetypes.guess_type(filename)[0]
        guessed_suffix = mimetypes.guess_extension(content_type or "")
        if guessed_suffix in ALLOWED_IMAGE_EXTENSIONS:
            return guessed_suffix

        raise InvalidUploadError("Only image uploads are supported in this version.")

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
        return (
            f"{self._settings.api_prefix}/results/"
            f"{request_id}/artifacts/{artifact_name}"
        )