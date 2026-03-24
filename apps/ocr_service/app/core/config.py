from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_ROOT = Path(__file__).resolve().parents[2]


def resolve_workspace_root() -> Path:
    for candidate in SERVICE_ROOT.parents:
        if (candidate / "requirements").exists() and (candidate / "apps").exists():
            return candidate
    return SERVICE_ROOT


WORKSPACE_ROOT = resolve_workspace_root()
DEFAULT_ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts" / "ocr_service"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_SERVICE_",
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        enable_decoding=False,
    )

    app_name: str = "ocr-service"
    api_prefix: str = "/v1"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8200

    backend: str = "paddleocr"
    preload_model: bool = False
    disable_model_source_check: bool = True

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    max_upload_size_mb: int = 25
    request_id_prefix: str = "req"

    preferred_lang: str = "vi"
    ocr_version: str = "PP-OCRv5"
    text_detection_model_name: str | None = "PP-OCRv5_server_det"
    text_detection_model_dir: Path | None = None
    text_recognition_model_name: str | None = "latin_PP-OCRv5_mobile_rec"
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

    @field_validator(
        "artifact_root",
        "text_detection_model_dir",
        "text_recognition_model_dir",
        "doc_orientation_classify_model_dir",
        "doc_unwarping_model_dir",
        "textline_orientation_model_dir",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        return Path(value)

    @property
    def use_doc_preprocessor(self) -> bool:
        return self.use_doc_orientation_classify or self.use_doc_unwarping

    def pipeline_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "text_rec_score_thresh": self.text_rec_score_thresh,
            "return_word_box": self.return_word_box,
        }

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
            kwargs["lang"] = self.preferred_lang
            kwargs["ocr_version"] = self.ocr_version

        return kwargs

    def predict_kwargs(self) -> dict[str, object]:
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
