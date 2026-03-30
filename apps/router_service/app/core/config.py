from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_ROOT = Path(__file__).resolve().parents[2]


def resolve_workspace_root() -> Path:
    for candidate in SERVICE_ROOT.parents:
        if (candidate / "requirements").exists() and (candidate / "apps").exists():
            return candidate
    return SERVICE_ROOT


WORKSPACE_ROOT = resolve_workspace_root()
DEFAULT_ARTIFACT_ROOT = WORKSPACE_ROOT / "results"
DEFAULT_VL_VENDOR_ROOT = WORKSPACE_ROOT / "vendor" / "PaddleOCR-VL"
DEFAULT_VL_LAYOUT_MODEL_DIR = DEFAULT_VL_VENDOR_ROOT / "PP-DocLayoutV2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTER_SERVICE_",
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        enable_decoding=False,
    )

    app_name: str = "router-service"
    api_prefix: str = "/v1"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8300
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False

    backend: str = "router"
    preload_model: bool = False
    disable_model_source_check: bool = True

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    max_upload_size_mb: int = 25
    max_sync_pdf_pages: int = 5
    request_id_prefix: str = "req"

    save_artifact_images: bool = False
    generate_markdown: bool = False
    pdf_render_scale: float = 2.5

    table_detector_backend: str = "protonx"
    table_detector_model_name: str = "protonx-models/protonx-table-detector"
    table_detector_threshold: float = 0.8
    table_detector_device: str = "auto"
    table_detector_fail_open: bool = True
    table_detector_mock_document_has_table: bool = False
    table_detector_mock_table_score: float = 0.95
    table_detector_mock_no_table_score: float = 0.05

    ocr_backend: str = "paddleocr_vietocr"
    ocr_device: str | None = None
    ocr_preferred_lang: str = "vi"
    ocr_version: str = "PP-OCRv5"
    text_detection_model_name: str | None = None
    text_detection_model_dir: Path | None = None
    text_recognition_model_name: str | None = None
    text_recognition_model_dir: Path | None = None
    doc_orientation_classify_model_name: str | None = None
    doc_orientation_classify_model_dir: Path | None = None
    doc_unwarping_model_name: str | None = None
    doc_unwarping_model_dir: Path | None = None
    textline_orientation_model_name: str | None = None
    textline_orientation_model_dir: Path | None = None
    textline_orientation_batch_size: int | None = None
    text_recognition_batch_size: int | None = None
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    text_det_limit_side_len: int | None = None
    text_det_limit_type: str | None = None
    text_det_thresh: float | None = None
    text_det_box_thresh: float | None = None
    text_det_unclip_ratio: float | None = None
    text_det_input_shape: str | None = None
    text_rec_score_thresh: float = 0.0
    return_word_box: bool = False
    text_rec_input_shape: str | None = None
    vietocr_config_name: str = "vgg_transformer"
    vietocr_weights: Path | None = None
    vietocr_device: str = "cpu"
    vietocr_beamsearch: bool = False
    vietocr_fallback_to_paddle_recognition: bool = True
    vietocr_crop_padding: int = 2

    vl_backend: str = "paddleocr_vl"
    vl_device: str | None = None
    vl_retry_attempts: int = 1
    vl_retry_backoff_ms: int = 0
    vl_use_queues: bool = False
    pipeline_version: str = "v1.5"
    layout_detection_model_name: str | None = None
    layout_detection_model_dir: Path | None = None
    vl_rec_model_name: str | None = None
    vl_rec_model_dir: Path | None = None
    use_layout_detection: bool = True
    use_chart_recognition: bool = False
    use_seal_recognition: bool = False
    use_ocr_for_image_block: bool = False
    format_block_content: bool = True
    merge_layout_blocks: bool = True
    markdown_ignore_labels: list[str] = Field(
        default_factory=lambda: [
            "number",
            "footnote",
            "header",
            "header_image",
            "footer",
            "footer_image",
            "aside_text",
        ]
    )
    pretty_markdown: bool = True
    show_formula_number: bool = False

    @field_validator(
        "artifact_root",
        "text_detection_model_dir",
        "text_recognition_model_dir",
        "doc_orientation_classify_model_dir",
        "doc_unwarping_model_dir",
        "textline_orientation_model_dir",
        "vietocr_weights",
        "layout_detection_model_dir",
        "vl_rec_model_dir",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        return Path(value)

    @field_validator(
        "cors_allow_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "markdown_ignore_labels",
        mode="before",
    )
    @classmethod
    def normalize_csv_list(
        cls, value: str | list[str] | tuple[str, ...]
    ) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @property
    def use_doc_preprocessor(self) -> bool:
        return self.use_doc_orientation_classify or self.use_doc_unwarping

    @property
    def ocr_engine_version(self) -> str:
        if self.ocr_backend == "paddleocr_vietocr":
            return f"PaddleOCR+VietOCR/{self.ocr_version}"
        if self.ocr_backend == "paddleocr":
            return f"PaddleOCR/{self.ocr_version}"
        return self.ocr_backend

    @property
    def vl_engine_version(self) -> str:
        return f"PaddleOCR-VL/{self.pipeline_version}"

    def ocr_pipeline_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "text_rec_score_thresh": self.text_rec_score_thresh,
            "return_word_box": self.return_word_box,
        }
        if self.ocr_device:
            kwargs["device"] = self.ocr_device
        optional_values = {
            "doc_orientation_classify_model_name": self.doc_orientation_classify_model_name,
            "doc_orientation_classify_model_dir": self.doc_orientation_classify_model_dir,
            "doc_unwarping_model_name": self.doc_unwarping_model_name,
            "doc_unwarping_model_dir": self.doc_unwarping_model_dir,
            "text_detection_model_name": self.text_detection_model_name,
            "text_detection_model_dir": self.text_detection_model_dir,
            "textline_orientation_model_name": self.textline_orientation_model_name,
            "textline_orientation_model_dir": self.textline_orientation_model_dir,
            "textline_orientation_batch_size": self.textline_orientation_batch_size,
            "text_recognition_model_name": self.text_recognition_model_name,
            "text_recognition_model_dir": self.text_recognition_model_dir,
            "text_recognition_batch_size": self.text_recognition_batch_size,
            "text_det_limit_side_len": self.text_det_limit_side_len,
            "text_det_limit_type": self.text_det_limit_type,
            "text_det_thresh": self.text_det_thresh,
            "text_det_box_thresh": self.text_det_box_thresh,
            "text_det_unclip_ratio": self.text_det_unclip_ratio,
            "text_det_input_shape": self.text_det_input_shape,
            "text_rec_input_shape": self.text_rec_input_shape,
        }
        for key, value in optional_values.items():
            if value is not None:
                kwargs[key] = str(value) if isinstance(value, Path) else value

        has_explicit_text_models = any(
            value is not None
            for value in (
                self.text_detection_model_name,
                self.text_detection_model_dir,
                self.text_recognition_model_name,
                self.text_recognition_model_dir,
            )
        )
        if not has_explicit_text_models:
            kwargs["lang"] = self.ocr_preferred_lang
            kwargs["ocr_version"] = self.ocr_version

        return kwargs

    def ocr_predict_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "text_rec_score_thresh": self.text_rec_score_thresh,
            "return_word_box": self.return_word_box,
        }
        optional_values = {
            "text_det_limit_side_len": self.text_det_limit_side_len,
            "text_det_limit_type": self.text_det_limit_type,
            "text_det_thresh": self.text_det_thresh,
            "text_det_box_thresh": self.text_det_box_thresh,
            "text_det_unclip_ratio": self.text_det_unclip_ratio,
        }
        for key, value in optional_values.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    def vl_pipeline_kwargs(self) -> dict[str, object]:
        if self.pipeline_version == "v1":
            if self.vl_rec_model_dir is None and DEFAULT_VL_VENDOR_ROOT.exists():
                self.vl_rec_model_dir = DEFAULT_VL_VENDOR_ROOT
            if (
                self.layout_detection_model_dir is None
                and DEFAULT_VL_LAYOUT_MODEL_DIR.exists()
            ):
                self.layout_detection_model_dir = DEFAULT_VL_LAYOUT_MODEL_DIR
            if self.vl_rec_model_name is None and self.vl_rec_model_dir is not None:
                self.vl_rec_model_name = "PaddleOCR-VL-0.9B"
            if (
                self.layout_detection_model_name is None
                and self.layout_detection_model_dir is not None
            ):
                self.layout_detection_model_name = "PP-DocLayoutV2"

        kwargs: dict[str, object] = {
            "pipeline_version": self.pipeline_version,
            "device": self.vl_device,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_layout_detection": self.use_layout_detection,
            "use_chart_recognition": self.use_chart_recognition,
            "use_seal_recognition": self.use_seal_recognition,
            "use_ocr_for_image_block": self.use_ocr_for_image_block,
            "format_block_content": self.format_block_content,
            "merge_layout_blocks": self.merge_layout_blocks,
            "markdown_ignore_labels": self.markdown_ignore_labels,
            "use_queues": self.vl_use_queues,
        }
        if self.layout_detection_model_name:
            kwargs["layout_detection_model_name"] = self.layout_detection_model_name
        if self.layout_detection_model_dir:
            kwargs["layout_detection_model_dir"] = str(self.layout_detection_model_dir)
        if self.vl_rec_model_name:
            kwargs["vl_rec_model_name"] = self.vl_rec_model_name
        if self.vl_rec_model_dir:
            kwargs["vl_rec_model_dir"] = str(self.vl_rec_model_dir)
        if kwargs["device"] in (None, ""):
            kwargs.pop("device")
        return kwargs

    def vl_predict_kwargs(self) -> dict[str, object]:
        return {
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_layout_detection": self.use_layout_detection,
            "use_chart_recognition": self.use_chart_recognition,
            "use_seal_recognition": self.use_seal_recognition,
            "use_ocr_for_image_block": self.use_ocr_for_image_block,
            "format_block_content": self.format_block_content,
            "merge_layout_blocks": self.merge_layout_blocks,
            "markdown_ignore_labels": self.markdown_ignore_labels,
            "use_queues": self.vl_use_queues,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
