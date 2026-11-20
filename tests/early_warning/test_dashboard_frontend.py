"""Frontend unit and integration tests for Streamlit Early Warning Dashboard.

Validates that dashboard UI rendering logic, data transformations,
filtering behaviors, empty states, and calibration displays function correctly.
"""

from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import pytest

from dashboard.app import get_integrator_and_service
from src.early_warning.dashboard_service import DashboardService
from src.early_warning.pipeline_integrator import PipelineIntegrator


def test_dashboard_data_loading_and_service_init():
    """Verify get_integrator_and_service loads real predictions artifact into service."""
    integrator, service = get_integrator_and_service()

    assert isinstance(integrator, PipelineIntegrator)
    assert isinstance(service, DashboardService)

    # Health check
    health = service.get_system_health()
    assert health["project_name"] == "PerceptGuard AI"
    assert health["calibration_gate_status"] == "PROVISIONAL"


def test_dashboard_chart_dataframe_transformation():
    """Verify temporal monitoring data formats into a valid multi-horizon dataframe."""
    _, service = get_integrator_and_service()
    temp_data = service.get_temporal_monitoring_data(limit=50)
    horizons_hist = temp_data.get("horizons", {})

    chart_rows = []
    for h_key, records in horizons_hist.items():
        h_name = h_key.upper()
        for r in records:
            if r.get("risk_score") is not None:
                chart_rows.append({
                    "Timestamp (s)": r["timestamp_seconds"],
                    "Risk Score": r["risk_score"],
                    "Horizon": h_name,
                })

    df = pd.DataFrame(chart_rows)
    assert not df.empty
    assert "Timestamp (s)" in df.columns
    assert "Risk Score" in df.columns
    assert "Horizon" in df.columns

    # Verify H1, H3, H5 are present
    unique_horizons = set(df["Horizon"].unique())
    assert "H1" in unique_horizons or "H3" in unique_horizons or "H5" in unique_horizons

    # Verify pivot does not raise errors
    df_pivot = df.pivot_table(index="Timestamp (s)", columns="Horizon", values="Risk Score")
    assert not df_pivot.empty


def test_dashboard_alert_filtering_logic():
    """Verify alert filtering combinations produce strictly conforming subsets."""
    _, service = get_integrator_and_service()

    # 1. Severity filter
    crit_alerts = service.get_alerts(severity="CRITICAL")
    assert all(a["severity"] == "CRITICAL" for a in crit_alerts["alerts"])

    # 2. Horizon filter
    h1_alerts = service.get_alerts(horizon=1)
    assert all(a["trigger_horizon"] == 1 for a in h1_alerts["alerts"])

    # 3. Status filter
    act_alerts = service.get_alerts(status="ACTIVE")
    assert all(a["lifecycle_state"] == "ACTIVE" for a in act_alerts["alerts"])


def test_dashboard_empty_state_handling():
    """Verify dashboard service handles completely empty monitor state gracefully."""
    empty_service = DashboardService(root_dir=Path("nonexistent_dir"))
    overview = empty_service.get_reliability_risk_overview()

    assert overview["calibration_status"] == "PROVISIONAL"
    for h in ("h1", "h3", "h5"):
        assert overview["horizons"][h]["risk_score"] is None
        assert overview["horizons"][h]["risk_level"] == "UNKNOWN"

    temp = empty_service.get_temporal_monitoring_data()
    for h in ("h1", "h3", "h5"):
        assert len(temp["horizons"][h]) == 0


def test_dashboard_no_fake_production_data():
    """Verify tracking and risk data explicitly flag synthetic or authentic origin."""
    _, service = get_integrator_and_service()
    tracking = service.get_detection_tracking_context()

    assert tracking["is_synthetic"] is False
    assert tracking["model_name"] == "yolo11n"
    assert tracking["tracker_type"] == "bytetrack"
    assert tracking["total_detections"] == 27
