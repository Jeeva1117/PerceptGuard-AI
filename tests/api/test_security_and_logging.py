"""Automated tests for security hardening, request tracing, and error masking."""

from __future__ import annotations

import re
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config.system_settings import SystemSettings


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_request_id_generation_and_propagation(client: TestClient):
    """Verify X-Request-ID is generated or propagated across requests."""
    # 1. Server generates UUID when omitted
    res = client.get("/health/live")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    req_id = res.headers["X-Request-ID"]
    assert len(req_id) >= 16

    # 2. Server propagates client-supplied X-Request-ID
    custom_id = "trace-client-req-987654"
    res_custom = client.get("/health/live", headers={"X-Request-ID": custom_id})
    assert res_custom.status_code == 200
    assert res_custom.headers["X-Request-ID"] == custom_id


def test_security_headers_present(client: TestClient):
    """Verify standard defensive security headers are returned on all endpoints."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"


def test_cors_headers_handling(client: TestClient):
    """Verify CORS origin validation headers."""
    headers = {"Origin": "http://localhost:8501"}
    res = client.get("/health/live", headers=headers)
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_standardized_error_envelope_and_no_stack_traces():
    """Verify uncaught exceptions do NOT leak stack traces to response body."""
    app = create_app()

    # Intentionally register a route that throws an unhandled exception
    @app.get("/test-internal-error")
    def trigger_crash():
        raise RuntimeError("Sensitive internal database connection failed: secret_user:secret_pass")

    with TestClient(app, raise_server_exceptions=False) as client_crash:
        res = client_crash.get("/test-internal-error")
        assert res.status_code == 500
        data = res.json()

        # Verify standardized envelope format
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert "request_id" in data
        assert "timestamp" in data

        # Verify sensitive error detail and traceback are masked
        assert "secret_pass" not in res.text
        assert "Traceback" not in res.text
        assert "RuntimeError" not in res.text
