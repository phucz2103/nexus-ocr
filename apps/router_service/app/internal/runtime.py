from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import UploadFile

from app.core.config import Settings
from app.services.storage import StorageService
from app.table_detection.disabled_detector import DisabledTableDetector

from app.internal.ocr_engine import ExtractionService as OcrExtractionService
from app.internal.ocr_mock_backend import MockOCRBackend
from app.internal.paddle_ocr_backend import PaddleOCRBackend
from app.internal.paddle_vietocr_backend import PaddleVietOCRBackend
from app.internal.paddle_vl_backend import PaddleOCRVLBackend
from app.internal.vl_engine import ExtractionService as VlExtractionService
from app.internal.vl_mock_backend import MockVLBackend


@dataclass(slots=True)
class LocalEngineResult:
    backend: str
    summary: dict[str, Any]
    data_info: dict[str, Any]
    page_count: int
    raw_text: str
    processing_duration_ms: int | None
    detector_confidence: float | None
    engine_version: str
    page_metrics: list[dict[str, Any]]
    page_selection: dict[str, Any] | None
    stage_timings: dict[str, int | None] | None
    result_json: dict[str, Any]


class OcrSettingsView(SimpleNamespace):
    def __init__(self, settings: Settings, artifact_root: Path) -> None:
        super().__init__(
            api_prefix=settings.api_prefix,
            backend=settings.ocr_backend,
            ocr_backend=settings.ocr_backend,
            ocr_device=settings.ocr_device,
            disable_model_source_check=settings.disable_model_source_check,
            artifact_root=artifact_root,
            max_upload_size_mb=settings.max_upload_size_mb,
            max_sync_pdf_pages=settings.max_sync_pdf_pages,
            request_id_prefix=f"{settings.request_id_prefix}_ocr",
            save_artifact_images=settings.save_artifact_images,
            generate_markdown=settings.generate_markdown,
            ocr_preferred_lang=settings.ocr_preferred_lang,
            preferred_lang=settings.ocr_preferred_lang,
            ocr_version=settings.ocr_version,
            pdf_render_scale=settings.pdf_render_scale,
            text_detection_model_name=settings.text_detection_model_name,
            text_detection_model_dir=settings.text_detection_model_dir,
            text_recognition_model_name=settings.text_recognition_model_name,
            text_recognition_model_dir=settings.text_recognition_model_dir,
            doc_orientation_classify_model_name=settings.doc_orientation_classify_model_name,
            doc_orientation_classify_model_dir=settings.doc_orientation_classify_model_dir,
            doc_unwarping_model_name=settings.doc_unwarping_model_name,
            doc_unwarping_model_dir=settings.doc_unwarping_model_dir,
            textline_orientation_model_name=settings.textline_orientation_model_name,
            textline_orientation_model_dir=settings.textline_orientation_model_dir,
            textline_orientation_batch_size=settings.textline_orientation_batch_size,
            text_recognition_batch_size=settings.text_recognition_batch_size,
            use_doc_orientation_classify=settings.use_doc_orientation_classify,
            use_doc_unwarping=settings.use_doc_unwarping,
            use_textline_orientation=settings.use_textline_orientation,
            text_det_limit_side_len=settings.text_det_limit_side_len,
            text_det_limit_type=settings.text_det_limit_type,
            text_det_thresh=settings.text_det_thresh,
            text_det_box_thresh=settings.text_det_box_thresh,
            text_det_unclip_ratio=settings.text_det_unclip_ratio,
            text_det_input_shape=settings.text_det_input_shape,
            text_rec_score_thresh=settings.text_rec_score_thresh,
            return_word_box=settings.return_word_box,
            text_rec_input_shape=settings.text_rec_input_shape,
            vietocr_config_name=settings.vietocr_config_name,
            vietocr_weights=settings.vietocr_weights,
            vietocr_device=settings.vietocr_device,
            vietocr_beamsearch=settings.vietocr_beamsearch,
            vietocr_fallback_to_paddle_recognition=settings.vietocr_fallback_to_paddle_recognition,
            vietocr_crop_padding=settings.vietocr_crop_padding,
            engine_version=settings.ocr_engine_version,
            use_doc_preprocessor=settings.use_doc_preprocessor,
        )

    def pipeline_kwargs(self) -> dict[str, object]:
        return Settings.ocr_pipeline_kwargs(self)

    def predict_kwargs(self) -> dict[str, object]:
        return Settings.ocr_predict_kwargs(self)


class VlSettingsView(SimpleNamespace):
    def __init__(self, settings: Settings, artifact_root: Path) -> None:
        super().__init__(
            api_prefix=settings.api_prefix,
            backend=settings.vl_backend,
            vl_backend=settings.vl_backend,
            vl_device=settings.vl_device,
            disable_model_source_check=settings.disable_model_source_check,
            artifact_root=artifact_root,
            max_upload_size_mb=settings.max_upload_size_mb,
            max_sync_pdf_pages=settings.max_sync_pdf_pages,
            retry_attempts=settings.vl_retry_attempts,
            retry_backoff_ms=settings.vl_retry_backoff_ms,
            request_id_prefix=f"{settings.request_id_prefix}_vl",
            save_artifact_images=settings.save_artifact_images,
            generate_markdown=settings.generate_markdown,
            table_detector_backend="disabled",
            table_detector_model_name=settings.table_detector_model_name,
            table_detector_threshold=settings.table_detector_threshold,
            table_detector_fail_open=settings.table_detector_fail_open,
            pipeline_version=settings.pipeline_version,
            layout_detection_model_name=settings.layout_detection_model_name,
            layout_detection_model_dir=settings.layout_detection_model_dir,
            vl_rec_model_name=settings.vl_rec_model_name,
            vl_rec_model_dir=settings.vl_rec_model_dir,
            use_doc_orientation_classify=settings.use_doc_orientation_classify,
            use_doc_unwarping=settings.use_doc_unwarping,
            use_layout_detection=settings.use_layout_detection,
            use_chart_recognition=settings.use_chart_recognition,
            use_seal_recognition=settings.use_seal_recognition,
            use_ocr_for_image_block=settings.use_ocr_for_image_block,
            format_block_content=settings.format_block_content,
            merge_layout_blocks=settings.merge_layout_blocks,
            markdown_ignore_labels=settings.markdown_ignore_labels,
            pretty_markdown=settings.pretty_markdown,
            show_formula_number=settings.show_formula_number,
            vl_use_queues=settings.vl_use_queues,
            use_queues=settings.vl_use_queues,
            engine_version=settings.vl_engine_version,
            use_doc_preprocessor=settings.use_doc_preprocessor,
        )

    def pipeline_kwargs(self) -> dict[str, object]:
        return Settings.vl_pipeline_kwargs(self)

    def predict_kwargs(self) -> dict[str, object]:
        return Settings.vl_predict_kwargs(self)


class LocalOcrRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runtime_root = settings.artifact_root / ".internal" / "ocr"
        self._runtime_settings = OcrSettingsView(settings, self._runtime_root)
        self._storage = StorageService(self._runtime_settings)
        self._backend = self._build_backend()
        self._service = OcrExtractionService(self._runtime_settings, self._storage, self._backend)

    def warmup(self) -> None:
        self._backend.warmup()

    def close(self) -> None:
        self._backend.close()
        self._release_process_memory()

    async def extract(
        self,
        file: UploadFile,
        page_start: int | None = None,
        page_end: int | None = None,
        enforce_sync_limit: bool = True,
    ) -> LocalEngineResult:
        completed = await self._service.extract(
            file,
            page_start=page_start,
            page_end=page_end,
            enforce_sync_limit=enforce_sync_limit,
        )
        result_json = self._service.load_result_json(completed.request_id)
        self._cleanup_request(completed.request_id)
        return LocalEngineResult(
            backend=completed.backend,
            summary=completed.summary,
            data_info=completed.data_info,
            page_count=completed.page_count,
            raw_text=completed.raw_text,
            processing_duration_ms=completed.processing_duration_ms,
            detector_confidence=completed.detector_confidence,
            engine_version=completed.engine_version,
            page_metrics=completed.page_metrics,
            page_selection=completed.page_selection,
            stage_timings=completed.stage_timings,
            result_json=result_json,
        )

    def _build_backend(self):
        if self._runtime_settings.backend == "mock":
            return MockOCRBackend(self._runtime_settings)
        if self._runtime_settings.backend == "paddleocr":
            return PaddleOCRBackend(self._runtime_settings)
        return PaddleVietOCRBackend(self._runtime_settings)

    def _cleanup_request(self, request_id: str) -> None:
        request_dir = self._runtime_root / request_id
        if request_dir.exists():
            shutil.rmtree(request_dir, ignore_errors=True)

    def _release_process_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import paddle

            if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "empty_cache"):
                paddle.device.cuda.empty_cache()
        except Exception:
            pass


class LocalVlRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runtime_root = settings.artifact_root / ".internal" / "vl"
        self._runtime_settings = VlSettingsView(settings, self._runtime_root)
        self._storage = StorageService(self._runtime_settings)
        self._backend = self._build_backend()
        self._table_detector = DisabledTableDetector(settings)
        self._service = VlExtractionService(
            self._runtime_settings,
            self._storage,
            self._backend,
            self._table_detector,
        )

    def warmup(self) -> None:
        self._backend.warmup()

    def close(self) -> None:
        self._backend.close()
        self._release_process_memory()

    async def extract(
        self,
        file: UploadFile,
        page_start: int | None = None,
        page_end: int | None = None,
        enforce_sync_limit: bool = True,
    ) -> LocalEngineResult:
        completed = await self._service.extract(
            file,
            page_start=page_start,
            page_end=page_end,
            enforce_sync_limit=enforce_sync_limit,
        )
        result_json = self._service.load_result_json(completed.request_id)
        self._cleanup_request(completed.request_id)
        return LocalEngineResult(
            backend=completed.backend,
            summary=completed.summary,
            data_info=completed.data_info,
            page_count=completed.page_count,
            raw_text=completed.raw_text,
            processing_duration_ms=completed.processing_duration_ms,
            detector_confidence=completed.detector_confidence,
            engine_version=completed.engine_version,
            page_metrics=completed.page_metrics,
            page_selection=completed.page_selection,
            stage_timings=completed.stage_timings,
            result_json=result_json,
        )

    def _build_backend(self):
        if self._runtime_settings.backend == "mock":
            return MockVLBackend(self._runtime_settings)
        return PaddleOCRVLBackend(self._runtime_settings)

    def _cleanup_request(self, request_id: str) -> None:
        request_dir = self._runtime_root / request_id
        if request_dir.exists():
            shutil.rmtree(request_dir, ignore_errors=True)

    def _release_process_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import paddle

            if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "empty_cache"):
                paddle.device.cuda.empty_cache()
        except Exception:
            pass
