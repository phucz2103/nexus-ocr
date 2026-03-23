# ocr-service

Standalone API for:

- uploading one image
- running PaddleOCR or the mock backend
- generating `result.json`
- generating `result.md`
- serving generated artifacts over HTTP

## Model Choice

The default real backend is tuned to stay on the PP-OCRv5 family while favoring document accuracy:

- detection: `PP-OCRv5_server_det`
- recognition: `latin_PP-OCRv5_mobile_rec`

This keeps the higher-accuracy server detector while pairing it with the PP-OCRv5 Latin-script multilingual recognizer, which is the closest PP-OCRv5 fit for Vietnamese and English text documents.

## Install

```powershell
python -m pip install -r requirements/ocr_service.txt
```

## Run

```powershell
cd E:\ocr\local-ocr-platform\apps\ocr_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8200
```

## Mock Test

```powershell
$env:OCR_SERVICE_BACKEND='mock'
python -m unittest discover -s apps/ocr_service/tests -p "test_*.py"
```