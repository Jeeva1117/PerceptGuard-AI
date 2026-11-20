"""Production middleware components for request tracking, security, and logging.

Enforces request ID propagation, request timing, security headers,
and structured observability without leaking sensitive payloads.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("perceptguard.api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Generates request tracing IDs, measures execution duration, and writes structured logs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Resolve or generate X-Request-ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in request.state for downstream handlers
        request.state.request_id = request_id

        start_time = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach tracing ID to response headers
            response.headers["X-Request-ID"] = request_id

            logger.info(
                f"HTTP {request.method} {request.url.path} "
                f"status={response.status_code} "
                f"duration_ms={duration_ms:.2f} "
                f"client={client_host} "
                f"request_id={request_id}"
            )
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"HTTP {request.method} {request.url.path} "
                f"EXCEPTION={type(exc).__name__} "
                f"duration_ms={duration_ms:.2f} "
                f"client={client_host} "
                f"request_id={request_id}"
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects standard defensive HTTP security headers into all responses."""

    def __init__(self, app, is_production: bool = False):
        super().__init__(app)
        self.is_production = is_production

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if self.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
