from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


class ExtractApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        os.environ["VL_SERVICE_BACKEND"] = "mock"
        os.environ["VL_SERVICE_ARTIFACT_ROOT"] = str(
            Path(self._temp_dir.name) / "artifacts"
        )

        from app.core.config import get_settings

        get_settings.cache_clear()

        from app.main import create_app

        self._client_context = TestClient(create_app())
        self.client = self._client_context.__enter__()

    def tearDown(self) -> None:
        self._client_context.__exit__(None, None, None)
        self._temp_dir.cleanup()
        os.environ.pop("VL_SERVICE_BACKEND", None)
        os.environ.pop("VL_SERVICE_ARTIFACT_ROOT", None)

        from app.core.config import get_settings

        get_settings.cache_clear()

    def test_extract_creates_json_and_markdown_artifacts(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (320, 180), color="white").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["summary"]["pages"], 1)

        request_id = payload["request_id"]

        json_response = self.client.get(payload["artifacts"]["json_url"])
        self.assertEqual(json_response.status_code, 200, json_response.text)
        result_json = json_response.json()
        self.assertIn("layoutParsingResults", result_json)
        self.assertIn("dataInfo", result_json)

        markdown_response = self.client.get(payload["artifacts"]["markdown_url"])
        self.assertEqual(markdown_response.status_code, 200, markdown_response.text)
        self.assertIn("Mock extraction", markdown_response.text)

        artifact_root = Path(os.environ["VL_SERVICE_ARTIFACT_ROOT"]) / request_id
        self.assertTrue((artifact_root / "result.json").exists())
        self.assertTrue((artifact_root / "result.md").exists())
        self.assertTrue((artifact_root / "images").exists())


if __name__ == "__main__":
    unittest.main()
