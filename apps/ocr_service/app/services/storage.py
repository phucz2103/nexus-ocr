from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.errors import ArtifactNotFoundError


@dataclass(slots=True)
class RequestPaths:
    request_id: str
    request_dir: Path
    images_dir: Path
    input_file: Path
    result_json: Path
    result_markdown: Path


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self._root = settings.artifact_root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def prepare_request(self, request_id: str, input_suffix: str) -> RequestPaths:
        request_dir = self._root / request_id
        images_dir = request_dir / "images"
        request_dir.mkdir(parents=True, exist_ok=True)
        return RequestPaths(
            request_id=request_id,
            request_dir=request_dir,
            images_dir=images_dir,
            input_file=request_dir / f"input{input_suffix}",
            result_json=request_dir / "result.json",
            result_markdown=request_dir / "result.md",
        )

    def write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_json(self, path: Path, content: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def save_image_artifact(
        self,
        request_id: str,
        artifact_name: str,
        image_data: Any,
    ) -> str:
        safe_name = self._sanitize_artifact_name(artifact_name)
        artifact_file = self._root / request_id / "images" / f"{safe_name}.png"
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        image = self._coerce_image(image_data)
        image.save(artifact_file)
        return artifact_file.name

    def resolve_result_json(self, request_id: str) -> Path:
        return self._require_existing(self._root / request_id / "result.json")

    def resolve_result_markdown(self, request_id: str) -> Path:
        return self._require_existing(self._root / request_id / "result.md")

    def resolve_input_file(self, request_id: str) -> Path:
        request_dir = self._root / request_id
        matches = sorted(request_dir.glob("input.*"))
        if not matches:
            raise ArtifactNotFoundError(f"Input artifact not found for `{request_id}`.")
        return self._require_existing(matches[0])

    def resolve_image_artifact(self, request_id: str, artifact_name: str) -> Path:
        if Path(artifact_name).name != artifact_name:
            raise ArtifactNotFoundError("Invalid artifact name.")
        return self._require_existing(
            self._root / request_id / "images" / artifact_name
        )

    def _require_existing(self, path: Path) -> Path:
        if not path.exists():
            raise ArtifactNotFoundError(f"Artifact not found: `{path.name}`.")
        return path

    def _sanitize_artifact_name(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
        return cleaned or "artifact"

    def _coerce_image(self, image_data: Any) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        array = np.asarray(image_data)
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L").convert("RGB")
        return Image.fromarray(array).convert("RGB")
