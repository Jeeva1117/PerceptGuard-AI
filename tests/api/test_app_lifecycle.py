"""Automated tests for FastAPI application lifecycle (startup, shutdown, resource allocation)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config.system_settings import SystemSettings


def test_app_startup_and_shutdown_lifecycle():
    """Verify application startup initializes services and shutdown cleans up resources."""
    app = create_app()

    with TestClient(app) as client:
        # Check that lifecycle startup executed
        assert hasattr(app.state, "integrator")
        assert app.state.integrator is not None
        assert hasattr(app.state, "dashboard_service")
        assert app.state.dashboard_service is not None
        assert hasattr(app.state, "startup_time")

        # Test root endpoint returns active status
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ONLINE"
        assert data["name"] == "PerceptGuard AI"

    # After exit of TestClient context, shutdown has completed
    assert app.state.integrator is None
    assert app.state.dashboard_service is None


def test_app_startup_with_custom_settings():
    """Verify app initializes correctly with custom isolated configuration."""
    custom_settings = SystemSettings.load(
        env_override={
            "APP_ENV": "testing",
            "DEBUG": "false",
            "API_PREFIX": "/api/v2",
        }
    )
    app = create_app(settings=custom_settings)

    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["environment"] == "testing"
        assert data["api_prefix"] == "/api/v2"
