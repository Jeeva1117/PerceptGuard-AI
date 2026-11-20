"""Automated tests for telemetry and risk monitoring endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_telemetry_health(client: TestClient):
    """Verify /telemetry/health returns system operational metrics."""
    res = client.get("/api/v1/telemetry/health")
    assert res.status_code == 200
    data = res.json()
    assert data["project_name"] == "PerceptGuard AI"
    assert data["health_score"] >= 0.0


def test_pipeline_status(client: TestClient):
    """Verify /telemetry/pipeline-status returns status of all 8 perception stages."""
    res = client.get("/api/v1/telemetry/pipeline-status")
    assert res.status_code == 200
    data = res.json()
    assert "components" in data
    assert "stage1_video_processing" in data["components"]
    assert "stage6_lstm_forecasting" in data["components"]
    assert "stage8_early_warning" in data["components"]


def test_risk_overview(client: TestClient):
    """Verify /telemetry/risk-overview returns H1, H3, and H5 horizons."""
    res = client.get("/api/v1/telemetry/risk-overview")
    assert res.status_code == 200
    data = res.json()
    assert "horizons" in data
    assert "h1" in data["horizons"]
    assert "h3" in data["horizons"]
    assert "h5" in data["horizons"]
    assert data["calibration_status"] == "PROVISIONAL"


def test_temporal_monitoring_valid(client: TestClient):
    """Verify /telemetry/temporal-monitoring returns trajectories and state transitions."""
    res = client.get("/api/v1/telemetry/temporal-monitoring?limit=25")
    assert res.status_code == 200
    data = res.json()
    assert "horizons" in data
    assert "state_transitions" in data
    assert data["limit"] == 25


def test_temporal_monitoring_invalid_horizon(client: TestClient):
    """Verify /telemetry/temporal-monitoring rejects unsupported horizons with 400."""
    res = client.get("/api/v1/telemetry/temporal-monitoring?horizon=7")
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_INPUT"
    assert "7" in data["error"]["message"]


def test_detection_tracking_context(client: TestClient):
    """Verify /telemetry/detection-tracking returns YOLO11n and ByteTrack context."""
    res = client.get("/api/v1/telemetry/detection-tracking")
    assert res.status_code == 200
    data = res.json()
    assert "tracker_type" in data
    assert data["tracker_type"] == "bytetrack"
    assert "total_detections" in data
    assert "mean_confidence_overall" in data
