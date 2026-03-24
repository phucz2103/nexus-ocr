from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts" / "vl_service"
DEFAULT_V1_VENDOR_ROOT = WORKSPACE_ROOT / "vendor" / "PaddleOCR-VL"
DEFAULT_V1_LAYOUT_MODEL_DIR = DEFAULT_V1_VENDOR_ROOT / "PP-DocLayoutV2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VL_SERVICE_",
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        enable_decoding=False,
    )

    app_name: str = "vl-service"
    api_prefix: str = "/v1"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8100

    backend: str = "paddleocr_vl"
    preload_model: bool = False
    disable_model_source_check: bool = True
    use_queues: bool = False

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    max_upload_size_mb: int = 25
    max_sync_pdf_pages: int = 5
    retry_attempts: int = 1
    retry_backoff_ms: int = 0
    request_id_prefix: str = "req"

    table_detector_backend: str = "disabled"
    table_detector_model_name: str = "protonx-models/protonx-table-detector"
    table_detector_threshold: float = 0.8
    table_detector_device: str = "auto"
    table_detector_fail_open: bool = True
    table_detector_mock_document_has_table: bool = False
    table_detector_mock_table_score: float = 0.95
    table_detector_mock_no_table_score: float = 0.05

    pipeline_version: str = "v1.5"
    layout_detection_model_name: str | None = None
    layout_detection_model_dir: Path | None = None
    vl_rec_model_name: str | None = None
    vl_rec_model_dir: Path | None = None

    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
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
        "layout_detection_model_dir",
        "vl_rec_model_dir",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        return Path(value)

    @field_validator("markdown_ignore_labels", mode="before")
    @classmethod
    def normalize_markdown_ignore_labels(
        cls, value: str | list[str] | tuple[str, ...]
    ) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @property
    def use_doc_preprocessor(self) -> bool:
        return self.use_doc_orientation_classify or self.use_doc_unwarping

    @property
    def engine_version(self) -> str:
        return f"PaddleOCR-VL/{self.pipeline_version}"

    @model_validator(mode="after")
    def apply_local_model_defaults(self) -> "Settings":
        if self.pipeline_version != "v1":
            return self

        if self.vl_rec_model_dir is None and DEFAULT_V1_VENDOR_ROOT.exists():
            self.vl_rec_model_dir = DEFAULT_V1_VENDOR_ROOT
        if (
            self.layout_detection_model_dir is None
            and DEFAULT_V1_LAYOUT_MODEL_DIR.exists()
        ):
            self.layout_detection_model_dir = DEFAULT_V1_LAYOUT_MODEL_DIR

        if self.vl_rec_model_name is None and self.vl_rec_model_dir is not None:
            self.vl_rec_model_name = "PaddleOCR-VL-0.9B"
        if (
            self.layout_detection_model_name is None
            and self.layout_detection_model_dir is not None
        ):
            self.layout_detection_model_name = "PP-DocLayoutV2"

        return self

    def pipeline_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "pipeline_version": self.pipeline_version,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_layout_detection": self.use_layout_detection,
            "use_chart_recognition": self.use_chart_recognition,
            "use_seal_recognition": self.use_seal_recognition,
            "use_ocr_for_image_block": self.use_ocr_for_image_block,
            "format_block_content": self.format_block_content,
            "merge_layout_blocks": self.merge_layout_blocks,
            "markdown_ignore_labels": self.markdown_ignore_labels,
            "use_queues": self.use_queues,
        }
        if self.layout_detection_model_name:
            kwargs["layout_detection_model_name"] = self.layout_detection_model_name
        if self.layout_detection_model_dir:
            kwargs["layout_detection_model_dir"] = str(self.layout_detection_model_dir)
        if self.vl_rec_model_name:
            kwargs["vl_rec_model_name"] = self.vl_rec_model_name
        if self.vl_rec_model_dir:
            kwargs["vl_rec_model_dir"] = str(self.vl_rec_model_dir)
        return kwargs

    def predict_kwargs(self) -> dict[str, object]:
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
            "use_queues": self.use_queues,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
