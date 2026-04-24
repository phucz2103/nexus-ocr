# router-service

OCR API gom 1 service duy nhat cho upload anh/PDF, detect bang va route sang OCR/VL.

## Yeu cau may

- Python 3.11
- Khuyen nghi con trong o C: toi thieu 20 GB
- Windows pagefile nen de `System managed` hoac >= 32 GB neu dung PaddleOCR-VL

## Cai dat

```powershell
cd E:\Nexus project\ocr
.\.venv\Scripts\python.exe -m pip install -r requirements\router_service.txt
```

## Chay local

```powershell
cd E:\Nexus project\ocr\apps\router_service
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8300
```

Kiem tra:

```powershell
curl.exe http://127.0.0.1:8300/health
```

## Chay Docker

```powershell
cd E:\Nexus project\ocr
docker compose up --build
```

## API chinh

- `POST /v1/extract`
- `POST /v1/extract/jobs`
- `GET /v1/extract/jobs/{job_id}`
- `GET /health`
- `GET /ready`

## Bien moi truong can quan tam

- `ROUTER_SERVICE_TABLE_DETECTOR_BACKEND=protonx`
- `ROUTER_SERVICE_TABLE_DETECTOR_FAIL_OPEN=true`
- `ROUTER_SERVICE_OCR_BACKEND=paddleocr_vietocr`
- `ROUTER_SERVICE_OCR_PREFERRED_LANG=vi`
- `ROUTER_SERVICE_OCR_VERSION=PP-OCRv5`
- `ROUTER_SERVICE_VL_BACKEND=paddleocr_vl`
- `ROUTER_SERVICE_PIPELINE_VERSION=v1.5`

File mau: `apps/router_service/.env.example`.

## Ghi chu

- `results/` la output runtime (co the xoa noi dung ben trong khi can don dung luong).
- Neu gap loi `os error 1455`, nguyen nhan la thieu virtual memory/pagefile.
