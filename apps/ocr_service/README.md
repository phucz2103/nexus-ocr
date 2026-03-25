# ocr-service

Standalone API for:

- uploading one image or PDF
- running PaddleOCR or the mock backend
- generating `result.json`
- generating `result.md`
- serving generated results over HTTP
- synchronous and async extraction flows

## Quality Notes

The safest default for Vietnamese text quality is to let PaddleOCR choose models from:

- `OCR_SERVICE_PREFERRED_LANG=vi`
- `OCR_SERVICE_OCR_VERSION=PP-OCRv5`

Avoid forcing `OCR_SERVICE_TEXT_RECOGNITION_MODEL_NAME=latin_PP-OCRv5_mobile_rec` unless you specifically want a Latin-only recognition path, because it can degrade Vietnamese output quality.

Recommended quality-related defaults:

- `OCR_SERVICE_USE_TEXTLINE_ORIENTATION=true`
- `OCR_SERVICE_PDF_RENDER_SCALE=2.5`
- keep explicit text model names unset unless you have validated a better pair

## Install

```powershell
python -m pip install -r requirements/ocr_service.txt
```

## Run

```powershell
cd E:\Nexus project\ocr\apps\ocr_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8200
```

## Mock Test

```powershell
$env:OCR_SERVICE_BACKEND='mock'
python -m unittest discover -s apps/ocr_service/tests -p "test_*.py"
```

