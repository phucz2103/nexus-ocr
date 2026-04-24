# OCR Workspace

## Thu muc nao can giu

- `results/`: CAN GIU thu muc (de service ghi ket qua). Co the xoa NOI DUNG ben trong bat ky luc nao.
- `scripts/`: KHONG CAN (da xoa vi dang rong).
- `vendor/`: KHONG CAN voi cau hinh hien tai (`ROUTER_SERVICE_PIPELINE_VERSION=v1.5`) nen da xoa.

## Chay nhanh

```powershell
cd E:\Nexus project\ocr
.\.venv\Scripts\python.exe -m pip install -r requirements\router_service.txt
cd .\apps\router_service
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8300
```

Kiem tra:

```powershell
curl.exe http://127.0.0.1:8300/health
```

## Docker

```powershell
cd E:\Nexus project\ocr
docker compose up --build
```

Cau hinh Docker da dong bo voi route hien tai: `protonx + paddleocr_vietocr + paddleocr_vl (v1.5)`.
