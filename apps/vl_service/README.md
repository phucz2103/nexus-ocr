# vl-service

Standalone API for:

- uploading one image
- running PaddleOCR-VL or the mock backend
- generating `result.json`
- generating `result.md`
- serving generated artifacts over HTTP

## Install

```powershell
python -m pip install -r requirements/vl_service.txt
```

## Local Config

- `.env` is set to `mock` so the API contract can be tested immediately.
- `.env.real` contains the local PaddleOCR-VL `v1` setup for `vendor/PaddleOCR-VL`.
- `.env.example` keeps the same `v1` layout if you want to recreate the real config.

## Run

```powershell
cd E:\ocr\local-ocr-platform\apps\vl_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

## Notes

- The vendored OCR-VL bundle matches PaddleOCR-VL `v1`.
- The real backend still depends on the local `paddle` and `paddlex` native stack behaving correctly on this machine.
