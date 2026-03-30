# router-service

Standalone OCR API for:

- uploading image or PDF
- running ProtonX table detection before OCR
- choosing and running the OCR or VL engine inside the same process
- preserving one sync and async OCR contract for FE/BE
- generating local `result.json` artifacts

## Install

```powershell
python -m pip install -r requirements/router_service.txt
```

## Local Config

- `.env.example` shows the minimum config needed to run the service.
- `ROUTER_SERVICE_TABLE_DETECTOR_BACKEND=protonx` enables ProtonX routing.
- `ROUTER_SERVICE_TABLE_DETECTOR_FAIL_OPEN=true` falls back to the VL engine when detection fails.
- `ROUTER_SERVICE_OCR_BACKEND` selects the plain OCR engine.
- `ROUTER_SERVICE_VL_BACKEND` selects the layout-aware VL engine.
- Leave `ROUTER_SERVICE_VL_DEVICE` empty to let PaddleOCR-VL use GPU automatically.
- Set `ROUTER_SERVICE_VL_DEVICE=cpu` only when you need a stability fallback for low-VRAM machines.
- synchronous PDF uploads are limited by `ROUTER_SERVICE_MAX_SYNC_PDF_PAGES` and default to `5` pages.

## Contract

`router-service` returns:

- `artifacts.json_url`
- `summary`
- `data_info`
- `page_count`
- `raw_text`
- `raw_text_plain`
- `mapping_text`
- `tables`
- `processing_duration_ms`
- `detector_confidence`
- `engine_version`
- `page_metrics`
- `page_selection`
- `table_detection`
- `stage_timings`

## Run

```powershell
cd E:\Nexus project\ocr\apps\router_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8300
```

Or run the single deployable stack from the OCR workspace root:

```powershell
cd E:\Nexus project\ocr
docker compose up --build
```

## Sync Contract

`POST /v1/extract` accepts both image files and PDF files.

Optional multipart form fields for PDF uploads:

- `page_start`: 1-based start page
- `page_end`: 1-based end page

If `page_start` is provided and `page_end` is omitted, the service processes a single PDF page.

## Async Job Flow

Use async jobs for long-running OCR work, especially larger PDFs. Async jobs are not limited by `ROUTER_SERVICE_MAX_SYNC_PDF_PAGES`.

1. Create a job:

```http
POST /v1/extract/jobs
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

## Routing Logic

- ProtonX checks selected pages first.
- `tableDetection.recommendedRoute=ocr_service` when no table is detected.
- `tableDetection.recommendedRoute=ocr_vl_service` when a table is detected.
- `tableDetection.actualRoute` reports the engine path that router actually executed.
- Router executes exactly one internal engine path for each request.

## Notes

- Current async jobs are still local in-process workers, suitable for MVP/internal use.
- Router now persists only `result.json`; it does not keep request input files or `result.md`.
- Router is now the only service you need to deploy for the OCR flow.
- The legacy `ocr_service` and `vl_service` apps have been removed from the repo; their logic now lives inside `router_service`.
