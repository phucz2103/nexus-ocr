from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.table_detection.base import TableDetectionPage, TableDetectionResult, TableDetector


class ProtonXTableDetector(TableDetector):
    name = "protonx"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._model: Any | None = None
        self._transform: Any | None = None
        self._device: Any | None = None

    def warmup(self) -> None:
        self._ensure_model()

    def detect(
        self,
        page_images: list[Image.Image],
        page_indices: list[int],
    ) -> TableDetectionResult:
        torch = self._ensure_model()
        started_at = perf_counter()

        if not page_images:
            return TableDetectionResult(
                backend=self.name,
                enabled=True,
                status="succeeded",
                model_name=self._settings.table_detector_model_name,
                threshold=self._settings.table_detector_threshold,
                document_has_table=False,
                processing_duration_ms=0,
                pages=[],
                recommended_route="ocr_service",
                actual_route="ocr_vl_service",
                error=None,
            )

        batch = torch.stack(
            [self._transform(image.convert("RGB")) for image in page_images]
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            scores, predictions = torch.max(probabilities, dim=1)

        pages: list[TableDetectionPage] = []
        document_has_table = False
        for page_index, score_tensor, prediction_tensor in zip(
            page_indices,
            scores,
            predictions,
        ):
            predicted_label = int(prediction_tensor.item())
            score = round(float(score_tensor.item()), 4)
            has_table = (
                predicted_label == 1
                and score >= self._settings.table_detector_threshold
            )
            pages.append(
                TableDetectionPage(
                    page_index=page_index,
                    has_table=has_table,
                    score=score,
                    label="table" if predicted_label == 1 else "no_table",
                )
            )
            document_has_table = document_has_table or has_table

        return TableDetectionResult(
            backend=self.name,
            enabled=True,
            status="succeeded",
            model_name=self._settings.table_detector_model_name,
            threshold=self._settings.table_detector_threshold,
            document_has_table=document_has_table,
            processing_duration_ms=max(
                1,
                int(round((perf_counter() - started_at) * 1000)),
            ),
            pages=pages,
            recommended_route="ocr_vl_service" if document_has_table else "ocr_service",
            actual_route="ocr_vl_service",
            error=None,
        )

    def _ensure_model(self):
        if self._model is not None:
            import torch

            return torch

        with self._lock:
            if self._model is not None:
                import torch

                return torch

            import torch
            import torch.nn as nn
            from huggingface_hub import hf_hub_download
            from torchvision import models as pretrained_models
            from torchvision import transforms

            device_name = self._settings.table_detector_device
            if device_name == "auto":
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(device_name)

            model_path = hf_hub_download(
                repo_id=self._settings.table_detector_model_name,
                filename="model/table_detector.pth",
            )

            model = pretrained_models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(
                in_features=model.classifier[1].in_features,
                out_features=2,
            )
            state_dict = torch.load(Path(model_path), map_location=self._device)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._model = model
            self._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        return torch
