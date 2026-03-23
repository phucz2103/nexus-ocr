from __future__ import annotations


class ServiceError(Exception):
    status_code = 500

    def __init__(self, detail: str, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class InvalidUploadError(ServiceError):
    status_code = 400


class ArtifactNotFoundError(ServiceError):
    status_code = 404


class BackendUnavailableError(ServiceError):
    status_code = 503


class InferenceFailedError(ServiceError):
    status_code = 500