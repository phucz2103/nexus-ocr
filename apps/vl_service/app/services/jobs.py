from __future__ import annotations

import asyncio
import io
import mimetypes
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.core.config import Settings
from app.core.errors import ArtifactNotFoundError, InvalidUploadError
from app.services.extraction import ExtractionService
from app.services.storage import StorageService


class AsyncJobService:
    def __init__(
        self,
        settings: Settings,
        storage: StorageService,
        extraction_service: ExtractionService,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._extraction_service = extraction_service
        self._jobs_root = self._storage.root / "jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._record_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def create_job(
        self,
        payload: bytes,
        filename: str | None,
        content_type: str | None,
        page_start: int | None,
        page_end: int | None,
    ) -> dict[str, Any]:
        if not payload:
            raise InvalidUploadError("Uploaded file is empty.")

        max_bytes = self._settings.max_upload_size_mb * 1024 * 1024
        if len(payload) > max_bytes:
            raise InvalidUploadError(
                f"Uploaded file exceeds the {self._settings.max_upload_size_mb} MB limit."
            )

        suffix = self._resolve_input_suffix(filename, content_type)
        job_id = self._generate_job_id()
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        input_path = job_dir / f"input{suffix}"
        self._storage.write_bytes(input_path, payload)

        now = self._now_iso()
        record = {
            "job_id": job_id,
            "status": "queued",
            "status_url": self._status_url(job_id),
            "created_at": now,
            "updated_at": now,
            "request_id": None,
            "backend": None,
            "original_filename": filename or f"input{suffix}",
            "content_type": content_type,
            "page_start": page_start,
            "page_end": page_end,
            "summary": None,
            "data_info": None,
            "page_count": None,
            "raw_text": None,
            "processing_duration_ms": None,
            "detector_confidence": None,
            "engine_version": None,
            "page_metrics": None,
            "page_selection": None,
            "table_detection": None,
            "artifacts": None,
            "error": None,
        }
        self._write_job_record(job_id, record)
        return record

    def start_job(self, job_id: str) -> None:
        worker = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True,
            name=f"vl-service-job-{job_id}",
        )
        self._threads[job_id] = worker
        worker.start()

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._read_job_record(job_id)

    def close(self) -> None:
        return None

    def _run_job(self, job_id: str) -> None:
        with self._worker_lock:
            record = self._read_job_record(job_id)
            if record["status"] not in {"queued", "failed"}:
                return

            self._update_job_record(job_id, status="processing", error=None)
            record = self._read_job_record(job_id)
            input_path = self._resolve_job_input_path(job_id)
            payload = input_path.read_bytes()

            upload = UploadFile(
                file=io.BytesIO(payload),
                filename=record["original_filename"],
                headers=Headers(
                    {
                        "content-type": (
                            record.get("content_type")
                            or mimetypes.guess_type(record["original_filename"])[0]
                            or "application/octet-stream"
                        )
                    }
                ),
            )

            try:
                result = asyncio.run(
                    self._extraction_service.extract(
                        upload,
                        page_start=record.get("page_start"),
                        page_end=record.get("page_end"),
                        enforce_sync_limit=False,
                    )
                )
                self._update_job_record(
                    job_id,
                    status="succeeded",
                    request_id=result.request_id,
                    backend=result.backend,
                    summary=result.summary,
                    data_info=result.data_info,
                    page_count=result.page_count,
                    raw_text=result.raw_text,
                    processing_duration_ms=result.processing_duration_ms,
                    detector_confidence=result.detector_confidence,
                    engine_version=result.engine_version,
                    page_metrics=result.page_metrics,
                    page_selection=result.page_selection,
                    table_detection=result.table_detection,
                    artifacts={
                        "json_url": result.json_url,
                        "markdown_url": result.markdown_url,
                        "input_url": result.input_url,
                    },
                    error=None,
                )
            except Exception as exc:
                detail = getattr(exc, "detail", str(exc))
                self._update_job_record(
                    job_id,
                    status="failed",
                    error=detail,
                )
            finally:
                try:
                    upload.file.close()
                except Exception:
                    pass
                self._threads.pop(job_id, None)

    def _resolve_input_suffix(
        self,
        filename: str | None,
        content_type: str | None,
    ) -> str:
        resolved_filename = filename or "upload"
        suffix = Path(resolved_filename).suffix.lower()
        allowed_extensions = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".pdf"}
        if suffix in allowed_extensions:
            return suffix

        guessed_suffix = mimetypes.guess_extension(content_type or "")
        if guessed_suffix in allowed_extensions:
            return guessed_suffix

        guessed_from_name = mimetypes.guess_type(resolved_filename)[0]
        guessed_suffix = mimetypes.guess_extension(guessed_from_name or "")
        if guessed_suffix in allowed_extensions:
            return guessed_suffix

        raise InvalidUploadError("Only image and PDF uploads are supported in this version.")

    def _generate_job_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"job_{timestamp}_{uuid4().hex[:8]}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _status_url(self, job_id: str) -> str:
        return f"{self._settings.api_prefix}/extract/jobs/{job_id}"

    def _job_dir(self, job_id: str) -> Path:
        return self._jobs_root / job_id

    def _job_record_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _resolve_job_input_path(self, job_id: str) -> Path:
        matches = sorted(self._job_dir(job_id).glob("input.*"))
        if not matches:
            raise ArtifactNotFoundError(f"Input artifact not found for job `{job_id}`.")
        return matches[0]

    def _read_job_record(self, job_id: str) -> dict[str, Any]:
        path = self._job_record_path(job_id)
        if not path.exists():
            raise ArtifactNotFoundError(f"Job not found: `{job_id}`.")
        return self._storage.read_json(path)

    def _write_job_record(self, job_id: str, record: dict[str, Any]) -> None:
        with self._record_lock:
            self._storage.write_json(self._job_record_path(job_id), record)

    def _update_job_record(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._record_lock:
            record = self._read_job_record(job_id)
            record.update(changes)
            record["updated_at"] = self._now_iso()
            self._storage.write_json(self._job_record_path(job_id), record)
            return record
