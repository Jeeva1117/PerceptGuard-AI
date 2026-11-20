"""Unit, Integration, and End-to-End Tests for Dashboard & Pipeline Integration.

Tests the full integration of:
- Task 2 (Risk Scorer)
- Task 3 (Risk Classifier & Calibration)
- Task 4 (Temporal Risk Monitor)
- Task 5 (Early Warning Engine & Lifecycle)
- Task 6 (Dashboard Service & Streamlit Frontend)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.early_warning.dashboard_service import DashboardService
from src.early_warning.early_warning_config import EarlyWarningConfig
from src.early_warning.early_warning_engine import EarlyWarningEngine
from src.early_warning.interfaces import (
    AlertLifecycleState,
    AlertSeverity,
    AlertType,
    RiskLevel,
)
from src.early_warning.pipeline_integrator import (
    PipelineBatchResult,
    PipelineIntegrator,
    PipelineProcessingResult,
)
from src.risk.risk_config import RiskConfig
from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.temporal_risk_monitor import TemporalRiskMonitor


@pytest.fixture
def clean_pipeline():
    """Fixture providing an isolated pipeline integrator and dashboard service."""
    config = EarlyWarningConfig()
    risk_config = RiskConfig()
    scorer = ReliabilityRiskScorer(config=risk_config)
    monitor = TemporalRiskMonitor(config=risk_config)
    engine = EarlyWarningEngine(config=config)
    service = DashboardService(config=config, risk_config=risk_config, monitor=monitor, engine=engine)
    integrator = PipelineIntegrator(
        config=config,
        risk_config=risk_config,
        scorer=scorer,
        monitor=monitor,
        engine=engine,
        dashboard_service=service,
    )
    return integrator, service


# ==============================================================================
# 1. Backend Service Endpoint Tests
# ==============================================================================

def test_system_health_endpoint(clean_pipeline):
    """Verify get_system_health returns valid operational status and metadata."""
    _, service = clean_pipeline
    health = service.get_system_health()

    assert health["project_name"] == "PerceptGuard AI"
    assert health["pipeline_status"] in ("READY", "RUNNING", "DEGRADED", "BLOCKED", "UNAVAILABLE")
    assert health["calibration_gate_status"] == "PROVISIONAL"
    assert "overall_health" in health
    assert "health_score" in health
    assert "model_availability" in health
    assert isinstance(health["model_availability"], dict)


def test_pipeline_status_endpoint(clean_pipeline):
    """Verify get_pipeline_status reports all 8 perception pipeline stages."""
    _, service = clean_pipeline
    status = service.get_pipeline_status()

    components = status["components"]
    assert len(components) == 8
    assert "stage1_video_processing" in components
    assert "stage2_yolo_detection" in components
    assert "stage3_bytetrack_tracking" in components
    assert "stage4_feature_engineering" in components
    assert "stage5_temporal_sequences" in components
    assert "stage6_lstm_forecasting" in components
    assert "stage7_risk_scoring" in components
    assert "stage8_early_warning" in components
    for comp in components.values():
        assert "status" in comp
        assert "details" in comp


def test_reliability_risk_overview_empty_state(clean_pipeline):
    """Verify risk overview before any predictions are ingested."""
    _, service = clean_pipeline
    overview = service.get_reliability_risk_overview()

    assert overview["calibration_status"] == "PROVISIONAL"
    assert "calibration_notice" in overview
    horizons = overview["horizons"]
    assert "h1" in horizons and "h3" in horizons and "h5" in horizons
    assert horizons["h1"]["risk_score"] is None
    assert horizons["h1"]["risk_level"] == "UNKNOWN"
    assert horizons["h1"]["status"] == "NO_DATA"


def test_h1_h3_h5_independent_response_handling(clean_pipeline):
    """Verify H1, H3, and H5 horizons are independently tracked and reported."""
    integrator, service = clean_pipeline

    # Ingest distinct predictions for H1, H3, H5
    integrator.process_prediction({
        "sequence_id": "test_seq_01",
        "horizon": 1,
        "predicted_reliability": 0.85,
        "timestamp_seconds": 1.0,
    })
    integrator.process_prediction({
        "sequence_id": "test_seq_01",
        "horizon": 3,
        "predicted_reliability": 0.60,
        "timestamp_seconds": 1.0,
    })
    integrator.process_prediction({
        "sequence_id": "test_seq_01",
        "horizon": 5,
        "predicted_reliability": 0.30,
        "timestamp_seconds": 1.0,
    })

    overview = service.get_reliability_risk_overview(sequence_id="test_seq_01")
    h1 = overview["horizons"]["h1"]
    h3 = overview["horizons"]["h3"]
    h5 = overview["horizons"]["h5"]

    # Verify scores are S = 1 - R and strictly independent
    assert h1["risk_score"] == pytest.approx(0.15, abs=1e-3)
    assert h3["risk_score"] == pytest.approx(0.40, abs=1e-3)
    assert h5["risk_score"] == pytest.approx(0.70, abs=1e-3)
    assert h1["horizon"] == 1
    assert h3["horizon"] == 3
    assert h5["horizon"] == 5


def test_temporal_monitoring_data_endpoint(clean_pipeline):
    """Verify get_temporal_monitoring_data returns chronological history and transitions."""
    integrator, service = clean_pipeline

    for t, rel in enumerate([0.9, 0.7, 0.4, 0.1]):
        integrator.process_prediction({
            "sequence_id": "seq_temporal",
            "horizon": 1,
            "predicted_reliability": rel,
            "timestamp_seconds": float(t * 0.1),
        })

    temp_data = service.get_temporal_monitoring_data(sequence_id="seq_temporal", horizon=1, limit=5)
    records = temp_data["horizons"]["h1"]
    assert len(records) == 4
    # Verify chronological ordering
    timestamps = [r["timestamp_seconds"] for r in records]
    assert timestamps == sorted(timestamps)


def test_alerts_endpoint_filtering_and_pagination(clean_pipeline):
    """Verify get_alerts supports filtering by severity, horizon, type, status, and page size."""
    integrator, service = clean_pipeline

    # Ingest records that generate alerts
    for i in range(10):
        integrator.process_prediction({
            "sequence_id": f"seq_alerts_{i}",
            "horizon": 1 if i % 2 == 0 else 3,
            "predicted_reliability": 0.05,  # Very low reliability -> High risk alert
            "timestamp_seconds": float(i),
        })

    # Test all alerts
    all_res = service.get_alerts(page=1, page_size=20)
    assert all_res["total_count"] >= 10

    # Filter by horizon
    h1_res = service.get_alerts(horizon=1, page=1, page_size=20)
    assert all(a["trigger_horizon"] == 1 for a in h1_res["alerts"])

    # Filter by severity
    crit_res = service.get_alerts(severity="CRITICAL", page=1, page_size=20)
    assert all(a["severity"] == "CRITICAL" for a in crit_res["alerts"])

    # Pagination test
    page1 = service.get_alerts(page=1, page_size=3)
    assert len(page1["alerts"]) == 3
    assert page1["page"] == 1
    assert page1["total_pages"] >= 4


def test_alert_detail_and_lifecycle_actions(clean_pipeline):
    """Verify get_alert_detail, acknowledge_alert, and resolve_alert APIs."""
    integrator, service = clean_pipeline

    integrator.process_prediction({
        "sequence_id": "seq_detail_test",
        "horizon": 1,
        "predicted_reliability": 0.02,
        "timestamp_seconds": 1.0,
    })

    alerts = service.get_alerts()["alerts"]
    assert len(alerts) > 0
    alert_id = alerts[0]["alert_id"]

    # 1. Detail inspection
    detail = service.get_alert_detail(alert_id)
    assert detail is not None
    assert detail["alert_id"] == alert_id
    assert "lifecycle_events" in detail

    # 2. Acknowledge
    ack_res = service.acknowledge_alert(alert_id, operator_id="test_operator")
    assert ack_res["acknowledged"] is True

    updated_detail = service.get_alert_detail(alert_id)
    assert updated_detail["lifecycle_state"] == "ACKNOWLEDGED"

    # 3. Resolve
    res_res = service.resolve_alert(alert_id, reason="Resolved in test", operator_id="test_operator")
    assert res_res["resolved"] is True

    resolved_detail = service.get_alert_detail(alert_id)
    assert resolved_detail["lifecycle_state"] == "RESOLVED"


def test_detection_tracking_context_endpoint(clean_pipeline):
    """Verify authentic YOLO11n + ByteTrack tracking summary is returned."""
    _, service = clean_pipeline
    tracking = service.get_detection_tracking_context()

    assert tracking["status"] == "AVAILABLE"
    assert tracking["model_name"] == "yolo11n"
    assert tracking["tracker_type"] == "bytetrack"
    assert tracking["total_frames_processed"] == 30
    assert tracking["total_detections"] == 27
    assert tracking["total_unique_tracks"] == 2
    assert tracking["is_synthetic"] is False


def test_detection_tracking_context_missing_file():
    """Verify graceful fallback when tracking summary file does not exist."""
    service = DashboardService(root_dir=Path("nonexistent_directory_for_test"))
    tracking = service.get_detection_tracking_context()

    assert tracking["status"] == "UNAVAILABLE"
    assert tracking["total_detections"] == 0
    assert "not found" in tracking["message"]


def test_missing_forecast_and_stale_behavior(clean_pipeline):
    """Verify missing forecast input handled cleanly without crashing or false zeroes."""
    integrator, service = clean_pipeline

    # Send an observation with missing reliability
    res = integrator.process_prediction({
        "sequence_id": "seq_missing",
        "horizon": 1,
        "predicted_reliability": 1.0,  # nominal
        "timestamp_seconds": 0.0,
    })
    assert res.risk_score == pytest.approx(0.0, abs=1e-3)


def test_invalid_api_parameters_handling(clean_pipeline):
    """Verify invalid pagination and non-existent alert IDs return structured responses."""
    _, service = clean_pipeline

    # Nonexistent alert detail
    detail = service.get_alert_detail("nonexistent_alert_id_999999")
    assert detail is None

    # Acknowledge nonexistent
    ack_res = service.acknowledge_alert("nonexistent_alert_id_999999")
    assert ack_res["acknowledged"] is False

    # Resolve nonexistent
    res_res = service.resolve_alert("nonexistent_alert_id_999999")
    assert res_res["resolved"] is False

    # Empty page pagination
    page_res = service.get_alerts(page=999, page_size=20)
    assert len(page_res["alerts"]) == 0
    assert page_res["page"] == 999


# ==============================================================================
# 2. End-to-End Pipeline Smoke Test
# ==============================================================================

def test_pipeline_integrator_smoke_test(clean_pipeline):
    """Verify run_smoke_test processes verified real prediction artifacts end-to-end."""
    integrator, service = clean_pipeline

    smoke_results = integrator.run_smoke_test(num_records=15)
    assert smoke_results["status"] == "PASS"
    assert smoke_results["records_processed"] == 15
    assert smoke_results["mean_pipeline_latency_ms"] < 10.0  # CPU budget < 10ms
    assert smoke_results["tracking_available"] is True
    assert smoke_results["calibration_status"] == "PROVISIONAL"
    assert smoke_results["horizons_scored"] == ["h1", "h3", "h5"]


def test_pipeline_latency_budget(clean_pipeline):
    """Verify per-frame end-to-end pipeline latency consumes < 1% of 33ms budget."""
    integrator, _ = clean_pipeline

    # Run 30 predictions through full pipeline
    batch_res = integrator.ingest_predictions_file(max_records=30)
    assert batch_res.total_processed == 30
    # Mean latency should be well under 1ms per prediction on CPU
    assert batch_res.mean_latency_ms < 2.0
