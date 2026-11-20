"""Centralized exception classes and FastAPI error handlers.

Guarantees predictable, standardized error payloads across all endpoints
and prevents internal stack trace or credential leakage to production clients.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("perceptguard.api.exceptions")


class PerceptGuardAPIError(Exception):
    """Base exception for application-level REST API errors."""

    def __init__(
        self,
        message: str,
        code: str = "API_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class ResourceNotFoundError(PerceptGuardAPIError):
    """Raised when a requested resource (alert, sequence, checkpoint) is not found."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class InvalidInputError(PerceptGuardAPIError):
    """Raised when request payload or query parameter fails business logic constraints."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="INVALID_INPUT",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class InvalidVideoError(PerceptGuardAPIError):
    """Raised when an uploaded video file is corrupt, unsupported, or exceeds size limits."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="INVALID_VIDEO_FILE",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class ModelInferenceError(PerceptGuardAPIError):
    """Raised when an ML model fails during evaluation or outputs non-finite values."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="MODEL_INFERENCE_ERROR",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
        )


class ServiceUnavailableError(PerceptGuardAPIError):
    """Raised when a core subsystem required for operation is uninitialized or degraded."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


def _format_error_envelope(
    code: str,
    message: str,
    details: Optional[Any] = None,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    """Build standardized API error envelope matching production schema."""
    req_id = getattr(request.state, "request_id", "unknown") if request and hasattr(request, "state") else "unknown"
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
        "request_id": req_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register uniform exception handlers on the FastAPI application instance."""

    @app.exception_handler(PerceptGuardAPIError)
    async def handle_perceptguard_api_error(
        request: Request, exc: PerceptGuardAPIError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            f"API Business Error: code={exc.code} status={exc.status_code} "
            f"message={exc.message} request_id={req_id}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_format_error_envelope(
                code=exc.code,
                message=exc.message,
                details=exc.details,
                request=request,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        logger.info(f"Request validation failure: {exc.errors()} request_id={req_id}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_format_error_envelope(
                code="VALIDATION_ERROR",
                message="Request payload or parameters failed validation schema.",
                details=exc.errors(),
                request=request,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            429: "TOO_MANY_REQUESTS",
            500: "INTERNAL_SERVER_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=_format_error_envelope(
                code=code,
                message=str(exc.detail),
                details=None,
                request=request,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        # Log complete stack trace internally to server logs, never exposed to user
        logger.exception(
            f"Unhandled exception during request processing: {exc} request_id={req_id}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_format_error_envelope(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred. Diagnostics logged under request_id.",
                details=None,
                request=request,
            ),
        )
