"""Automated tests for alert management and incident lifecycle endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_get_alerts_pagination(client: TestClient):
    """Verify /alerts returns paginated response structure."""
    res = client.get("/api/v1/alerts?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert "total_count" in data
    assert "alerts" in data
    assert data["page"] == 1
    assert data["page_size"] == 10


def test_get_alerts_invalid_severity(client: TestClient):
    """Verify /alerts rejects invalid severity values with 400 Bad Request."""
    res = client.get("/api/v1/alerts?severity=EXTREME")
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_INPUT"
    assert "EXTREME" in data["error"]["message"]


def test_get_alert_detail_not_found(client: TestClient):
    """Verify /alerts/{alert_id} returns 404 for missing alerts."""
    res = client.get("/api/v1/alerts/non_existent_alert_id_99999")
    assert res.status_code == 404
    data = res.json()
    assert data["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert "non_existent_alert_id_99999" in data["error"]["message"]


def test_acknowledge_alert_not_found(client: TestClient):
    """Verify acknowledging a non-existent alert returns 404."""
    res = client.post(
        "/api/v1/alerts/non_existent_alert_id_99999/acknowledge",
        json={"operator_id": "test_operator"},
    )
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_resolve_alert_validation_error(client: TestClient):
    """Verify resolving alert with missing or too-short reason returns 422."""
    res = client.post(
        "/api/v1/alerts/alert_123/resolve",
        json={"reason": "no"},  # Reason must be at least 3 chars
    )
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


def test_alert_lifecycle_flow(client: TestClient):
    """Verify end-to-end alert query, detail, acknowledge, and resolve flow."""
    from src.early_warning.interfaces import (
        AlertLifecycleState,
        AlertSeverity,
        AlertType,
        EarlyWarningAlert,
        RiskLevel,
    )

    test_alert = EarlyWarningAlert(
        alert_id="alert_lifecycle_test_001",
        sequence_id="seq_test_lifecycle",
        risk_level=RiskLevel.HIGH,
        risk_score=0.88,
        trigger_horizon=1,
        source_frame_id="frame_0001.jpg",
        target_frame_id="frame_0002.jpg",
        target_timestamp_seconds=10.0,
        warning_message="Lifecycle test warning",
        recommended_action="Prepare for manual intervention",
        is_suppressed=False,
        severity=AlertSeverity.CRITICAL,
        alert_type=AlertType.THRESHOLD_BREACH,
        lifecycle_state=AlertLifecycleState.ACTIVE,
    )
    engine = client.app.state.dashboard_service.engine
    engine.active_alerts[test_alert.alert_id] = test_alert
    engine.alert_history.append(test_alert)

    # 1. Query alert detail
    res_detail = client.get(f"/api/v1/alerts/{test_alert.alert_id}")
    assert res_detail.status_code == 200
    assert res_detail.json()["alert_id"] == test_alert.alert_id

    # 2. Acknowledge alert
    res_ack = client.post(
        f"/api/v1/alerts/{test_alert.alert_id}/acknowledge",
        json={"operator_id": "auditor_ops"},
    )
    assert res_ack.status_code == 200
    assert res_ack.json()["new_state"] == "ACKNOWLEDGED"

    # 3. Resolve alert
    res_res = client.post(
        f"/api/v1/alerts/{test_alert.alert_id}/resolve",
        json={"reason": "Audit verification resolution", "operator_id": "auditor_ops"},
    )
    assert res_res.status_code == 200
    assert res_res.json()["new_state"] == "RESOLVED"
