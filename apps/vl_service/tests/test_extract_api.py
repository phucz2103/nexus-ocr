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
        os.environ["VL_SERVICE_TABLE_DETECTOR_BACKEND"] = "mock"
        os.environ["VL_SERVICE_TABLE_DETECTOR_MOCK_DOCUMENT_HAS_TABLE"] = "false"
        os.environ["VL_SERVICE_SAVE_ARTIFACT_IMAGES"] = "true"
        os.environ["VL_SERVICE_GENERATE_MARKDOWN"] = "true"
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
        os.environ.pop("VL_SERVICE_SAVE_ARTIFACT_IMAGES", None)
        os.environ.pop("VL_SERVICE_GENERATE_MARKDOWN", None)

        from app.core.config import get_settings

        get_settings.cache_clear()

    def _recreate_client(self) -> None:
        self._client_context.__exit__(None, None, None)

        from app.core.config import get_settings

        get_settings.cache_clear()

        from app.main import create_app

        self._client_context = TestClient(create_app())
        self.client = self._client_context.__enter__()

    def _build_pdf(self, page_count: int) -> io.BytesIO:
        pdf_bytes = io.BytesIO()
        pages = [
            Image.new("RGB", (320, 180), color=color)
            for color in ("white", "lightgray", "gainsboro", "whitesmoke", "snow")
        ][:page_count]
        pages[0].save(pdf_bytes, format="PDF", save_all=True, append_images=pages[1:])
        pdf_bytes.seek(0)
        return pdf_bytes

    def test_cors_preflight_allows_cross_origin_post(self) -> None:
        response = self.client.options(
            "/v1/extract",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
        self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))

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
        self.assertEqual(payload["page_count"], 1)
        self.assertIsNone(payload["page_selection"])
        self.assertIn("Mock extraction", payload["raw_text"])
        self.assertIsNotNone(payload["processing_duration_ms"])
        self.assertIsNotNone(payload["engine_version"])
        self.assertEqual(len(payload["page_metrics"]), 1)
        self.assertIn("stage_timings", payload)
        self.assertIn("ocr_ms", payload["stage_timings"])
        self.assertEqual(payload["table_detection"]["backend"], "mock")
        self.assertFalse(payload["table_detection"]["document_has_table"])
        self.assertEqual(payload["table_detection"]["recommended_route"], "ocr_service")
        self.assertEqual(payload["table_detection"]["actual_route"], "ocr_vl_service")
        self.assertEqual(len(payload["table_detection"]["pages"]), 1)

        request_id = payload["request_id"]

        json_response = self.client.get(payload["artifacts"]["json_url"])
        self.assertEqual(json_response.status_code, 200, json_response.text)
        result_json = json_response.json()
        self.assertIn("layoutParsingResults", result_json)
        self.assertIn("dataInfo", result_json)
        self.assertEqual(result_json["dataInfo"]["type"], "image")
        self.assertEqual(result_json["pageCount"], 1)
        self.assertIsNone(result_json["pageSelection"])
        self.assertIn("rawText", result_json)
        self.assertIn("processingDurationMs", result_json)
        self.assertIn("engineVersion", result_json)
        self.assertEqual(len(result_json["pageMetrics"]), 1)
        self.assertIn("stageTimings", result_json)
        self.assertIn("ocrMs", result_json["stageTimings"])
        self.assertIn("tableDetection", result_json)
        self.assertFalse(result_json["tableDetection"]["documentHasTable"])
        self.assertEqual(result_json["tableDetection"]["recommendedRoute"], "ocr_service")

        markdown_response = self.client.get(payload["artifacts"]["markdown_url"])
        self.assertEqual(markdown_response.status_code, 200, markdown_response.text)
        self.assertIn("Mock extraction", markdown_response.text)

        artifact_root = Path(os.environ["VL_SERVICE_ARTIFACT_ROOT"]) / request_id
        self.assertTrue((artifact_root / "result.json").exists())
        self.assertTrue((artifact_root / "result.md").exists())
        self.assertTrue((artifact_root / "images").exists())

    def test_extract_pdf_creates_multi_page_results(self) -> None:
        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.pdf", self._build_pdf(2).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["summary"]["pages"], 2)
        self.assertEqual(payload["page_count"], 2)
        self.assertEqual(len(payload["page_metrics"]), 2)
        self.assertEqual(payload["page_selection"]["page_start"], 1)
        self.assertEqual(payload["page_selection"]["page_end"], 2)
        self.assertEqual(payload["page_selection"]["selected_page_count"], 2)
        self.assertFalse(payload["table_detection"]["document_has_table"])
        self.assertEqual(len(payload["table_detection"]["pages"]), 2)

        result_json = self.client.get(payload["artifacts"]["json_url"]).json()
        self.assertEqual(result_json["dataInfo"]["type"], "pdf")
        self.assertEqual(result_json["dataInfo"]["page_count"], 2)
        self.assertEqual(len(result_json["layoutParsingResults"]), 2)
        self.assertEqual(result_json["pageCount"], 2)
        self.assertEqual(len(result_json["pageMetrics"]), 2)
        self.assertEqual(result_json["pageSelection"]["pageStart"], 1)
        self.assertEqual(result_json["pageSelection"]["pageEnd"], 2)
        self.assertIn("tableDetection", result_json)
        self.assertFalse(result_json["tableDetection"]["documentHasTable"])
        self.assertEqual(len(result_json["tableDetection"]["pages"]), 2)
        self.assertTrue(
            result_json["layoutParsingResults"][0]["inputImage"].endswith("page_0_input_img.png")
        )
        self.assertTrue(
            result_json["layoutParsingResults"][1]["inputImage"].endswith("page_1_input_img.png")
        )

        markdown_response = self.client.get(payload["artifacts"]["markdown_url"])
        self.assertEqual(markdown_response.status_code, 200, markdown_response.text)
        self.assertIn("page 1/2", markdown_response.text)
        self.assertIn("page 2/2", markdown_response.text)

    def test_extract_pdf_with_blank_page_fields_processes_full_document(self) -> None:
        response = self.client.post(
            "/v1/extract",
            data={"page_start": "", "page_end": ""},
            files={"file": ("sample.pdf", self._build_pdf(2).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["page_count"], 2)
        self.assertEqual(payload["page_selection"]["page_start"], 1)
        self.assertEqual(payload["page_selection"]["page_end"], 2)

    def test_extract_pdf_page_range_processes_selected_pages_only(self) -> None:
        response = self.client.post(
            "/v1/extract",
            data={"page_start": "2", "page_end": "2"},
            files={"file": ("sample.pdf", self._build_pdf(3).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["page_count"], 1)
        self.assertEqual(payload["summary"]["pages"], 1)
        self.assertEqual(payload["page_selection"]["page_start"], 2)
        self.assertEqual(payload["page_selection"]["page_end"], 2)
        self.assertEqual(payload["page_selection"]["selected_page_count"], 1)
        self.assertEqual(payload["page_selection"]["total_page_count"], 3)

        result_json = self.client.get(payload["artifacts"]["json_url"]).json()
        self.assertEqual(result_json["dataInfo"]["page_count"], 3)
        self.assertEqual(result_json["pageCount"], 1)
        self.assertEqual(result_json["pageSelection"]["pageStart"], 2)
        self.assertEqual(result_json["pageSelection"]["pageEnd"], 2)
        self.assertEqual(result_json["pageSelection"]["selectedPageCount"], 1)
        self.assertEqual(result_json["pageSelection"]["totalPageCount"], 3)
        self.assertEqual(len(result_json["layoutParsingResults"]), 1)
        self.assertEqual(
            result_json["layoutParsingResults"][0]["prunedResult"]["page_index"],
            1,
        )
        self.assertEqual(result_json["pageMetrics"][0]["pageIndex"], 1)
        self.assertEqual(result_json["tableDetection"]["pages"][0]["pageIndex"], 1)
        self.assertTrue(
            result_json["layoutParsingResults"][0]["inputImage"].endswith("page_1_input_img.png")
        )

    def test_extract_rejects_page_range_for_images(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (320, 180), color="white").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/v1/extract",
            data={"page_start": "1"},
            files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("can only be used with PDF uploads", response.json()["detail"])

    def test_extract_rejects_page_end_without_page_start(self) -> None:
        response = self.client.post(
            "/v1/extract",
            data={"page_end": "2"},
            files={"file": ("sample.pdf", self._build_pdf(3).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("`page_start` is required", response.json()["detail"])

    def test_extract_rejects_pdf_over_sync_page_limit(self) -> None:
        os.environ["VL_SERVICE_MAX_SYNC_PDF_PAGES"] = "1"
        self._recreate_client()

        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.pdf", self._build_pdf(2).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("exceeds the synchronous PDF limit of 1 pages", response.json()["detail"])

    def test_extract_pdf_page_range_can_fit_within_sync_limit(self) -> None:
        os.environ["VL_SERVICE_MAX_SYNC_PDF_PAGES"] = "1"
        self._recreate_client()

        response = self.client.post(
            "/v1/extract",
            data={"page_start": "2", "page_end": "2"},
            files={"file": ("sample.pdf", self._build_pdf(2).getvalue(), "application/pdf")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["page_count"], 1)
        self.assertEqual(payload["page_selection"]["selected_page_count"], 1)

    def test_extract_can_report_table_presence_without_changing_actual_route(self) -> None:
        os.environ["VL_SERVICE_TABLE_DETECTOR_MOCK_DOCUMENT_HAS_TABLE"] = "true"
        self._recreate_client()

        image_bytes = io.BytesIO()
        Image.new("RGB", (320, 180), color="white").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["table_detection"]["document_has_table"])
        self.assertEqual(payload["table_detection"]["recommended_route"], "ocr_vl_service")
        self.assertEqual(payload["table_detection"]["actual_route"], "ocr_vl_service")

    def test_extract_can_disable_markdown_and_artifact_images(self) -> None:
        os.environ["VL_SERVICE_SAVE_ARTIFACT_IMAGES"] = "false"
        os.environ["VL_SERVICE_GENERATE_MARKDOWN"] = "false"
        self._recreate_client()

        image_bytes = io.BytesIO()
        Image.new("RGB", (320, 180), color="white").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/v1/extract",
            files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        request_id = payload["request_id"]
        self.assertIsNone(payload["stage_timings"]["pdf_render_ms"])
        self.assertIn("result_assembly_ms", payload["stage_timings"])
        self.assertIn("Mock extraction", payload["raw_text"])

        result_json = self.client.get(payload["artifacts"]["json_url"]).json()
        self.assertEqual(result_json["layoutParsingResults"][0]["outputImages"], {})
        self.assertEqual(result_json["layoutParsingResults"][0]["markdown"]["text"], "")
        self.assertEqual(result_json["layoutParsingResults"][0]["markdown"]["images"], {})
        self.assertEqual(result_json["preprocessedImages"], [])
        self.assertEqual(
            result_json["layoutParsingResults"][0]["inputImage"],
            f"/v1/results/{request_id}/input",
        )
        self.assertIn("stageTimings", result_json)
        self.assertIn("resultAssemblyMs", result_json["stageTimings"])

        markdown_response = self.client.get(payload["artifacts"]["markdown_url"])
        self.assertEqual(markdown_response.status_code, 200, markdown_response.text)
        self.assertEqual(markdown_response.text, "")

        artifact_root = Path(os.environ["VL_SERVICE_ARTIFACT_ROOT"]) / request_id / "images"
        self.assertEqual(list(artifact_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
