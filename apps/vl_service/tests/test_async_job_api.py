from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


class AsyncJobApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        os.environ["VL_SERVICE_BACKEND"] = "mock"
        os.environ["VL_SERVICE_ARTIFACT_ROOT"] = str(
            Path(self._temp_dir.name) / "artifacts"
        )
        os.environ["VL_SERVICE_TABLE_DETECTOR_BACKEND"] = "mock"
        os.environ["VL_SERVICE_TABLE_DETECTOR_MOCK_DOCUMENT_HAS_TABLE"] = "false"
        os.environ.pop("VL_SERVICE_MAX_SYNC_PDF_PAGES", None)

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
        os.environ.pop("VL_SERVICE_TABLE_DETECTOR_BACKEND", None)
        os.environ.pop("VL_SERVICE_TABLE_DETECTOR_MOCK_DOCUMENT_HAS_TABLE", None)
        os.environ.pop("VL_SERVICE_MAX_SYNC_PDF_PAGES", None)

        from app.core.config import get_settings

        get_settings.cache_clear()

    def _build_pdf(self, page_count: int) -> io.BytesIO:
        pdf_bytes = io.BytesIO()
        pages = [
            Image.new("RGB", (320, 180), color=color)
            for color in ("white", "lightgray", "gainsboro")
        ][:page_count]
        pages[0].save(pdf_bytes, format="PDF", save_all=True, append_images=pages[1:])
        pdf_bytes.seek(0)
        return pdf_bytes

    def _poll_job(self, status_url: str, timeout_s: float = 3.0) -> dict:
        deadline = time.time() + timeout_s
        last_payload = None
        while time.time() < deadline:
            response = self.client.get(status_url)
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            last_payload = payload
            if payload["status"] in {"succeeded", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail(f"Timed out waiting for async job completion. Last payload: {last_payload}")

    def test_async_job_can_process_image_and_expose_artifacts(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (320, 180), color="white").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/v1/extract/jobs",
            files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
        )

        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertTrue(payload["job_id"].startswith("job_"))
        self.assertTrue(payload["status_url"].endswith(payload["job_id"]))

        job_payload = self._poll_job(payload["status_url"])
        self.assertEqual(job_payload["status"], "succeeded")
        self.assertIsNotNone(job_payload["request_id"])
        self.assertEqual(job_payload["summary"]["pages"], 1)
        self.assertEqual(job_payload["artifacts"]["json_url"].split("/")[-2], job_payload["request_id"])

        result_json_response = self.client.get(job_payload["artifacts"]["json_url"])
        self.assertEqual(result_json_response.status_code, 200, result_json_response.text)
        result_json = result_json_response.json()
        self.assertEqual(result_json["dataInfo"]["type"], "image")
        self.assertIn("tableDetection", result_json)

    def test_async_job_can_process_pdf_page_range(self) -> None:
        response = self.client.post(
            "/v1/extract/jobs",
            data={"page_start": "2", "page_end": "2"},
            files={"file": ("sample.pdf", self._build_pdf(3).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        job_payload = self._poll_job(payload["status_url"])
        self.assertEqual(job_payload["status"], "succeeded")
        self.assertEqual(job_payload["page_count"], 1)
        self.assertEqual(job_payload["page_selection"]["page_start"], 2)
        self.assertEqual(job_payload["page_selection"]["page_end"], 2)

        result_json_response = self.client.get(job_payload["artifacts"]["json_url"])
        self.assertEqual(result_json_response.status_code, 200, result_json_response.text)
        result_json = result_json_response.json()
        self.assertEqual(result_json["pageSelection"]["pageStart"], 2)
        self.assertEqual(result_json["pageSelection"]["pageEnd"], 2)

    def test_async_job_bypasses_sync_pdf_page_limit(self) -> None:
        os.environ["VL_SERVICE_MAX_SYNC_PDF_PAGES"] = "1"
        self._client_context.__exit__(None, None, None)

        from app.core.config import get_settings

        get_settings.cache_clear()

        from app.main import create_app

        self._client_context = TestClient(create_app())
        self.client = self._client_context.__enter__()

        response = self.client.post(
            "/v1/extract/jobs",
            files={"file": ("sample.pdf", self._build_pdf(2).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        job_payload = self._poll_job(payload["status_url"])
        self.assertEqual(job_payload["status"], "succeeded")
        self.assertEqual(job_payload["page_count"], 2)


if __name__ == "__main__":
    unittest.main()
