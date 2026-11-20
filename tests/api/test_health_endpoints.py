"""Automated tests for health, liveness, and readiness endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_endpoint(client: TestClient):
    """Verify /health/live and /api/v1/health/live respond with alive status."""
    res_root = client.get("/health/live")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "alive"

    res_prefixed = client.get("/api/v1/health/live")
    assert res_prefixed.status_code == 200
    assert res_prefixed.json()["status"] == "alive"


def test_readiness_endpoint(client: TestClient):
    """Verify readiness probe returns operational subsystem statuses."""
    res = client.get("/api/v1/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("ready", "degraded")
    assert "subsystems" in data
    assert "dashboard_service" in data["subsystems"]
    assert data["subsystems"]["dashboard_service"] == "READY"


def test_comprehensive_health_endpoint(client: TestClient):
    """Verify /health returns structured operational health indicators."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["project_name"] == "PerceptGuard AI"
    assert "pipeline_status" in data
    assert "health_score" in data
    assert "calibration_gate_status" in data
