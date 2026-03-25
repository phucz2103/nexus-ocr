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


class ExtractApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        os.environ["OCR_SERVICE_BACKEND"] = "mock"
        os.environ["OCR_SERVICE_ARTIFACT_ROOT"] = str(
            Path(self._temp_dir.name) / "artifacts"
        )
        os.environ["OCR_SERVICE_MAX_SYNC_PDF_PAGES"] = "5"

        from app.core.config import get_settings

        get_settings.cache_clear()

        from app.main import create_app

        self._client_context = TestClient(create_app())
        self.client = self._client_context.__enter__()

    def tearDown(self) -> None:
        self._client_context.__exit__(None, None, None)
        self._temp_dir.cleanup()
        os.environ.pop("OCR_SERVICE_BACKEND", None)
        os.environ.pop("OCR_SERVICE_ARTIFACT_ROOT", None)
        os.environ.pop("OCR_SERVICE_MAX_SYNC_PDF_PAGES", None)

        from app.core.config import get_settings

        get_settings.cache_clear()

    def test_extract_creates_json_only_artifact(self) -> None:
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
        self.assertEqual(payload["page_count"], 1)
        self.assertIsNone(payload["page_selection"])

        request_id = payload["request_id"]

        json_response = self.client.get(payload["artifacts"]["json_url"])
        self.assertEqual(json_response.status_code, 200, json_response.text)
        result_json = json_response.json()
        self.assertIn("ocrParsingResults", result_json)
        self.assertIn("dataInfo", result_json)
        self.assertEqual(result_json["pageCount"], 1)

        artifact_root = Path(os.environ["OCR_SERVICE_ARTIFACT_ROOT"]) / request_id
        self.assertTrue((artifact_root / "result.json").exists())
        self.assertFalse((artifact_root / "result.md").exists())
        self.assertFalse((artifact_root / "images").exists())

    def test_extract_pdf_with_page_range(self) -> None:
        pdf_bytes = self._build_pdf([(320, 180), (400, 240), (500, 300)])

        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
            data={"page_start": "2", "page_end": "3"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["page_count"], 2)
        self.assertEqual(payload["page_selection"]["page_start"], 2)
        self.assertEqual(payload["page_selection"]["page_end"], 3)
        self.assertEqual(payload["page_selection"]["selected_page_count"], 2)
        self.assertEqual(payload["data_info"]["type"], "pdf")
        self.assertEqual(payload["data_info"]["page_count"], 3)

        result_json = self.client.get(payload["artifacts"]["json_url"]).json()
        self.assertEqual(result_json["pageCount"], 2)
        self.assertEqual(result_json["dataInfo"]["page_count"], 3)
        self.assertEqual(
            [item["prunedResult"]["page_index"] for item in result_json["ocrParsingResults"]],
            [1, 2],
        )

    def test_extract_pdf_requires_page_start_when_page_end_is_provided(self) -> None:
        pdf_bytes = self._build_pdf([(320, 180), (400, 240)])

        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
            data={"page_end": "2"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("`page_start` is required", response.text)

    def test_create_async_job_for_pdf(self) -> None:
        pdf_bytes = self._build_pdf([(320, 180), (400, 240), (500, 300)])

        create_response = self.client.post(
            "/v1/extract/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
            data={"page_start": "1", "page_end": "2"},
        )

        self.assertEqual(create_response.status_code, 202, create_response.text)
        job_payload = create_response.json()
        self.assertEqual(job_payload["status"], "queued")
        self.assertIsNotNone(job_payload["job_id"])

        final_payload = None
        for _ in range(50):
            status_response = self.client.get(job_payload["status_url"])
            self.assertEqual(status_response.status_code, 200, status_response.text)
            final_payload = status_response.json()
            if final_payload["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

        self.assertIsNotNone(final_payload)
        self.assertEqual(final_payload["status"], "succeeded", final_payload)
        self.assertEqual(final_payload["page_count"], 2)
        self.assertEqual(final_payload["page_selection"]["selected_page_count"], 2)
        self.assertIsNotNone(final_payload["request_id"])
        self.assertIsNotNone(final_payload["artifacts"])

    def _build_pdf(self, page_sizes: list[tuple[int, int]]) -> bytes:
        pages = [Image.new("RGB", size, color="white") for size in page_sizes]
        buffer = io.BytesIO()
        pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
