from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import ModuleType
from typing import Any

from app.core.config import Settings
from app.core.errors import BackendUnavailableError, InferenceFailedError
from app.inference.base import (
    BackendStatus,
    InferenceBackend,
    InferencePage,
    InferenceRunResult,
    build_model_settings_snapshot,
)


LOGGER = logging.getLogger(__name__)


def _install_paddlex_official_models_shim() -> None:
    try:
        from paddlex.inference.utils.official_models import official_models  # noqa: F401

        LOGGER.info("Using PaddleX official_models registry")
        return
    except Exception as exc:
        module_name = "paddlex.inference.utils.official_models"
        if module_name in sys.modules:
            return

        shim = ModuleType(module_name)
        shim.official_models = {}
        sys.modules[module_name] = shim
        LOGGER.warning(
            "Installed PaddleX official_models compatibility shim because the real registry could not be imported: %s",
            exc,
        )


def _install_paddle_inference_compat_shim() -> None:
    import paddle
    from paddle import inference
    from paddle.incubate.nn import functional as incubate_nn_functional
    from paddle.incubate.tensor import manipulation as tensor_manipulation
    from paddle.nn.functional import flash_attention as flash_attention_module

    patched = False

    if not hasattr(inference.Config, "set_optimization_level"):
        setattr(inference.Config, "set_optimization_level", lambda self, level: None)
        patched = True

    if not hasattr(tensor_manipulation, "create_async_load"):
        class _NoOpAsyncTask:
            def cpu_wait(self) -> None:
                return None

            def cuda_wait(self) -> None:
                return None

        class _NoOpAsyncLoader:
            def offload(self, *args, **kwargs):
                return _NoOpAsyncTask()

            def reload(self, *args, **kwargs):
                return _NoOpAsyncTask()

        setattr(
            tensor_manipulation,
            "create_async_load",
            lambda *args, **kwargs: _NoOpAsyncLoader(),
        )
        patched = True

    if not hasattr(incubate_nn_functional, "fused_rms_norm_ext"):
        def _fused_rms_norm_ext(x, weight, epsilon, *args, **kwargs):
            with paddle.amp.auto_cast(False):
                hidden_states = x.astype("float32")
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                normalized = paddle.rsqrt(variance + epsilon) * hidden_states
            return normalized.astype(weight.dtype) * weight, None

        setattr(incubate_nn_functional, "fused_rms_norm_ext", _fused_rms_norm_ext)
        patched = True

    if not hasattr(incubate_nn_functional, "cal_aux_loss"):
        def _cal_aux_loss(*args, **kwargs):
            gate_prob = args[0] if args else paddle.to_tensor(0.0, dtype="float32")
            dtype = getattr(gate_prob, "dtype", "float32")
            return paddle.zeros([], dtype=dtype), None, None

        setattr(incubate_nn_functional, "cal_aux_loss", _cal_aux_loss)
        patched = True

    if not hasattr(flash_attention_module, "flashmask_attention") and hasattr(
        flash_attention_module, "flash_attention_with_sparse_mask"
    ):
        setattr(
            flash_attention_module,
            "flashmask_attention",
            flash_attention_module.flash_attention_with_sparse_mask,
        )
        patched = True

    if patched:
        LOGGER.info("Installed Paddle inference compatibility shim")


def _patch_processor_instance(processor: Any) -> None:
    import copy
    from types import MethodType

    import numpy as np
    import paddle
    from paddlex.inference.models.doc_vlm.processors.common import (
        BatchFeature,
        fetch_image,
    )

    if getattr(processor, "_codex_placeholder_patch", False):
        return

    def patched_preprocess(
        self,
        input_dicts,
        min_pixels=None,
        max_pixels=None,
    ):
        images = [fetch_image(input_dict["image"]) for input_dict in input_dicts]

        text = []
        for input_dict in input_dicts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": input_dict["query"]},
                    ],
                }
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            text.append(prompt)

        output_kwargs = {
            "tokenizer_init_kwargs": self.tokenizer.init_kwargs,
            "text_kwargs": copy.deepcopy(self._DEFAULT_TEXT_KWARGS),
            "video_kwargs": copy.deepcopy(self._DEFAULT_VIDEO_KWARGS),
        }

        if min_pixels is not None or max_pixels is not None:
            size = {
                "min_pixels": min_pixels or self.image_processor.min_pixels,
                "max_pixels": max_pixels or self.image_processor.max_pixels,
            }
        else:
            size = None

        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                size=size,
                return_tensors="pd",
            )
            image_inputs["pixel_values"] = image_inputs["pixel_values"]
            image_grid_thw = image_inputs["image_grid_thw"]
            if paddle.is_tensor(image_grid_thw):
                image_grid_for_text = image_grid_thw.detach().cpu().numpy()
            else:
                image_grid_for_text = np.asarray(image_grid_thw)
        else:
            image_inputs = {}
            image_grid_thw = None
            image_grid_for_text = None

        videos_inputs = {}
        video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            index = 0
            merge_size = int(np.asarray(self.image_processor.merge_size).reshape(-1)[0])
            for i in range(len(text)):
                while self.image_token in text[i]:
                    grid = np.asarray(image_grid_for_text[index]).reshape(-1)
                    placeholder_count = int(np.prod(grid))
                    placeholder_count = placeholder_count // merge_size // merge_size
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * placeholder_count,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>",
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    processor.preprocess = MethodType(patched_preprocess, processor)
    processor._codex_placeholder_patch = True
    LOGGER.info("Applied PaddleOCR-VL processor compatibility patch")


def _patch_projector_instance(projector: Any) -> None:
    from types import MethodType

    import paddle
    from einops import rearrange

    if getattr(projector, "_codex_grid_patch", False):
        return

    def patched_forward(self, image_features, image_grid_thw):
        m1, m2 = self.merge_kernel_size
        if isinstance(image_features, (list, tuple)):
            processed_features = []
            for image_feature, image_grid in zip(image_features, image_grid_thw):
                image_feature = self.pre_norm(image_feature)
                grid = paddle.reshape(image_grid, [-1])
                if grid.shape[0] < 3:
                    raise ValueError(f"Unexpected image_grid_thw shape: {image_grid}")
                t = int(grid[0].item())
                h = int(grid[1].item())
                w = int(grid[2].item())
                image_feature = rearrange(
                    image_feature,
                    "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                    t=t,
                    h=h // int(m1),
                    p1=int(m1),
                    w=w // int(m2),
                    p2=int(m2),
                )
                hidden_states = self.linear_1(image_feature)
                hidden_states = self.act(hidden_states)
                hidden_states = self.linear_2(hidden_states)
                processed_features.append(hidden_states)
            return processed_features

        dims = image_features.shape[:-1]
        dim = image_features.shape[-1]
        image_features = paddle.reshape(image_features, [-1, dim])
        hidden_states = self.pre_norm(image_features)
        hidden_states = paddle.reshape(hidden_states, [-1, self.hidden_size])
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return paddle.reshape(hidden_states, [*dims, -1])

    projector.forward = MethodType(patched_forward, projector)
    projector._codex_grid_patch = True
    LOGGER.info("Applied PaddleOCR-VL projector compatibility patch")


class PaddleOCRVLBackend(InferenceBackend):
    name = "paddleocr_vl"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Any | None = None
        self._last_error: str | None = None
        self._lock = Lock()

    def warmup(self) -> None:
        self._ensure_pipeline()

    def get_status(self) -> BackendStatus:
        return BackendStatus(
            backend=self.name,
            ready=self._last_error is None,
            initialized=self._pipeline is not None,
            last_error=self._last_error,
        )

    def extract(self, input_path: Path) -> InferenceRunResult:
        pipeline = self._ensure_pipeline()
        started_at = perf_counter()
        last_page_checkpoint = started_at
        raw_results: list[Any] = []
        try:
            for raw_result in pipeline.predict(
                str(input_path),
                **self._settings.predict_kwargs(),
            ):
                raw_results.append(raw_result)
        except Exception as exc:
            self._last_error = str(exc)
            LOGGER.exception("PaddleOCR-VL inference failed for %s", input_path)
            raise InferenceFailedError(
                f"PaddleOCR-VL failed to process `{input_path.name}`: {exc}"
            ) from exc

        if not raw_results:
            raise InferenceFailedError("PaddleOCR-VL returned no parsing results.")

        pages: list[InferencePage] = []
        for page_index, raw_result in enumerate(raw_results):
            current_checkpoint = perf_counter()
            page_duration_ms = max(
                1,
                int(round((current_checkpoint - last_page_checkpoint) * 1000)),
            )
            last_page_checkpoint = current_checkpoint

            raw_json = raw_result.json.get("res", {})
            raw_markdown = self._extract_markdown_payload(raw_result)
            page = InferencePage(
                pruned_result=self._normalize_page_result(raw_json, page_index),
                markdown=raw_markdown,
                images=dict(getattr(raw_result, "img", {}) or {}),
                metrics=self._build_page_metrics(
                    page_index=page_index,
                    page_data=raw_json,
                    processing_duration_ms=page_duration_ms,
                ),
            )
            pages.append(page)

        self._last_error = None
        return InferenceRunResult(
            backend=self.name,
            pages=pages,
            processing_duration_ms=max(
                1,
                int(round((perf_counter() - started_at) * 1000)),
            ),
            engine_version=self._settings.engine_version,
        )

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None and hasattr(pipeline, "close"):
            pipeline.close()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                if self._settings.disable_model_source_check:
                    os.environ.setdefault(
                        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True"
                    )

                _install_paddle_inference_compat_shim()

                from paddleocr import PaddleOCRVL

                self._pipeline = PaddleOCRVL(**self._settings.pipeline_kwargs())

                predictor = self._pipeline.paddlex_pipeline.vl_rec_model
                _patch_processor_instance(predictor.processor)
                _patch_projector_instance(predictor.infer.mlp_AR)

                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("Failed to initialize PaddleOCR-VL pipeline")
                raise BackendUnavailableError(
                    f"Unable to initialize PaddleOCR-VL: {exc}"
                ) from exc

        return self._pipeline

    def _extract_markdown_payload(self, raw_result: Any) -> dict[str, Any]:
        if hasattr(raw_result, "_to_markdown"):
            raw_markdown = raw_result._to_markdown(
                pretty=self._settings.pretty_markdown,
                show_formula_number=self._settings.show_formula_number,
            ) or {}
        else:
            raw_markdown = getattr(raw_result, "markdown", {}) or {}

        return {
            "text": raw_markdown.get("text")
            or raw_markdown.get("markdown_texts", ""),
            "images": raw_markdown.get("images")
            or raw_markdown.get("markdown_images", {}),
        }

    def _normalize_page_result(
        self,
        page_data: dict[str, Any],
        page_index: int,
    ) -> dict[str, Any]:
        normalized = {
            "page_count": page_data.get("page_count"),
            "page_index": page_data.get("page_index", page_index),
            "width": page_data.get("width"),
            "height": page_data.get("height"),
            "model_settings": build_model_settings_snapshot(self._settings),
            "parsing_res_list": page_data.get("parsing_res_list", []),
        }
        for optional_key in ("layout_det_res", "spotting_res", "doc_preprocessor_res"):
            if optional_key in page_data and page_data[optional_key] is not None:
                normalized[optional_key] = page_data[optional_key]
        return normalized

    def _build_page_metrics(
        self,
        page_index: int,
        page_data: dict[str, Any],
        processing_duration_ms: int,
    ) -> dict[str, Any]:
        layout_boxes = (page_data.get("layout_det_res") or {}).get("boxes") or []
        scores = [
            float(box.get("score"))
            for box in layout_boxes
            if isinstance(box.get("score"), (int, float))
        ]
        detector_confidence = (
            round(sum(scores) / len(scores), 4)
            if scores
            else None
        )
        return {
            "page_index": page_data.get("page_index", page_index),
            "processing_duration_ms": processing_duration_ms,
            "detector_confidence": detector_confidence,
            "block_count": len(page_data.get("parsing_res_list", []) or []),
        }

