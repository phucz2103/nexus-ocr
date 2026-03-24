FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ocr_service.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install paddlepaddle==3.2.1 && \
    python -m pip install -r /tmp/requirements.txt

COPY apps/ocr_service /app

EXPOSE 8200

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8200"]
