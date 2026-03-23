# vl-service

Standalone API for:

- uploading one image or PDF
- running PaddleOCR-VL or the mock backend
- generating `result.json`
- generating `result.md`
- serving generated artifacts over HTTP

## Install

```powershell
python -m pip install -r requirements/vl_service.txt
```

## Local Config

- `.env` is set to the real local PaddleOCR-VL setup on this machine.
- `.env.example` shows the minimum config needed to run the service again.

## Run

```powershell
cd E:\ocr\local-ocr-platform\apps\vl_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

## Notes

- `POST /v1/extract` now accepts both image files and PDF files.
- For PDF uploads, `dataInfo` includes `page_count` and per-page sizes.
- Each page in `layoutParsingResults` gets its own `inputImage` preview artifact.
- The real backend still depends on the local `paddle` and `paddlex` native stack behaving correctly on this machine.