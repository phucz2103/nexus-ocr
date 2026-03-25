# vl-service

Standalone API for:

- uploading one image or PDF
- running PaddleOCR-VL or the mock backend
- generating `result.json`
- generating `result.md`
- serving generated results over HTTP
- creating async OCR jobs for long-running processing

## Install

```powershell
python -m pip install -r requirements/vl_service.txt
```

## Local Config

- `.env` is set to the real local PaddleOCR-VL setup on this machine.
- `.env.example` shows the minimum config needed to run the service again.
- synchronous PDF uploads are limited by `VL_SERVICE_MAX_SYNC_PDF_PAGES` and default to `5` pages.
- transient OCR-VL failures are retried `1` extra time by default through `VL_SERVICE_RETRY_ATTEMPTS`.
- `VL_SERVICE_RETRY_BACKOFF_MS` lets you add a small delay between retries if the runtime is unstable.
- optional table detection can run before OCR-VL. The current routing still always executes OCR-VL, but the result includes `tableDetection` metadata so you can switch to `ocr-service` later without changing the contract.
- `VL_SERVICE_SAVE_ARTIFACT_IMAGES=false` skips saving preview/layout artifact images when you only need structured JSON.
- `VL_SERVICE_GENERATE_MARKDOWN=false` skips markdown generation work and keeps `result.md` empty while preserving the API contract.
- every extraction now includes `stage_timings` / `stageTimings` for the heavy stages so you can see whether time is going into input prep, PDF rendering, table detection, OCR, or result assembly.
- CORS is enabled through `VL_SERVICE_CORS_ALLOW_*` settings. By default the service allows all origins, methods, and headers for local integration work.

## Run

```powershell
cd E:\ocr\local-ocr-platform\apps\vl_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

## Sync Contract

`POST /v1/extract` accepts both image files and PDF files.

Optional multipart form fields for PDF uploads:

- `page_start`: 1-based start page
- `page_end`: 1-based end page

If `page_start` is provided and `page_end` is omitted, the service processes a single PDF page.

The JSON artifact includes:

- `pageCount`
- `rawText`
- `processingDurationMs`
- `detectorConfidence`
- `engineVersion`
- `pageMetrics`
- `pageSelection`
- `tableDetection`
- `stageTimings`

For PDF uploads:

- `dataInfo` includes `page_count` and per-page sizes
- `pageSelection` shows which original pages were processed
- each page in `layoutParsingResults` gets its own `inputImage` preview artifact when artifact saving is enabled
- PDFs above the sync page limit are rejected early with a clear error message
- large PDFs can still run synchronously when `page_start` and `page_end` narrow the request to a small enough slice
- larger PDFs should move to the async job flow instead of the sync endpoint

## Async Job Flow

Use async jobs for long-running OCR-VL work, especially larger PDFs. Async jobs are not limited by `VL_SERVICE_MAX_SYNC_PDF_PAGES`.

1. Create a job:

```http
POST /v1/extract/jobs
```

Response:

```json
{
  "job_id": "job_20260324T010101_ab12cd34",
  "status": "queued",
  "status_url": "/v1/extract/jobs/job_20260324T010101_ab12cd34"
}
```

2. Poll the job:

```http
GET /v1/extract/jobs/{job_id}
```

Job statuses:

- `queued`
- `processing`
- `succeeded`
- `failed`

When the job succeeds, the status response includes:

- `request_id`
- `artifacts.json_url`
- `artifacts.markdown_url`
- `artifacts.input_url`
- extraction summary and metadata
- `stage_timings`

## Retry Behavior

- only transient backend errors are retried
- invalid input errors are not retried
- each retry resets the backend before trying again

## Table Detection

- `VL_SERVICE_TABLE_DETECTOR_BACKEND=disabled` keeps the current behavior unchanged
- `VL_SERVICE_TABLE_DETECTOR_BACKEND=mock` is useful for tests
- `VL_SERVICE_TABLE_DETECTOR_BACKEND=protonx` enables the ProtonX table classifier
- the current OCR route is still forced to `ocr_vl_service`
- `tableDetection.recommendedRoute` tells you what the future router should choose:
  - `ocr_service` when no table is detected
  - `ocr_vl_service` when a table is detected
- the ProtonX backend uses the Hugging Face model `protonx-models/protonx-table-detector`
- ProtonX requires a local `torch` + `torchvision` install that matches your machine

## Notes

- The current async job implementation is a local in-process worker suitable for MVP and internal use.
- If the service process restarts, queued or processing jobs are not resumed automatically.
- Very large PDFs should move to a real external queue later instead of the synchronous API or in-process worker.
- The real backend still depends on the local `paddle` and `paddlex` native stack behaving correctly on this machine.

