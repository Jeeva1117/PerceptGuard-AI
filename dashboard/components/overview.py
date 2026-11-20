"""Section A: System Overview and Operational Health component."""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_error_boundary, render_stale_warning


def render_system_overview(client: DashboardApiClient) -> None:
    """Render high-level operational overview, perception health, and model availability."""
    st.subheader("Section A — System Operational Overview")

    try:
        health = client.get_system_health()
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load System Operational Health",
            error_message=f"Backend communication error: {err.message}",
            retry_callback=st.rerun,
            key_suffix="health",
        )
        return

    status = health.get("pipeline_status", "UNKNOWN")
    overall = health.get("overall_health", "UNKNOWN")
    score = health.get("health_score", 0.0)
    freshness = health.get("data_freshness", "UNKNOWN")
    seq_id = health.get("current_sequence_id", "N/A")
    last_ts = health.get("last_prediction_timestamp_seconds", 0.0)
    models = health.get("model_availability", {})

    # Staleness warning if data expired
    render_stale_warning(
        last_sync=client.last_sync_time,
        freshness=freshness,
        ttl_seconds=client.settings.dashboard.telemetry_freshness_ttl_seconds,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Pipeline Status",
            value=status,
            help="Status values: READY, RUNNING, DEGRADED, BLOCKED, UNAVAILABLE",
        )
    with col2:
        st.metric(
            label="Overall Perception Health",
            value=overall,
            delta=f"{score:.1f}% Score" if score > 0 else None,
        )
    with col3:
        st.metric(
            label="Data Freshness",
            value=freshness,
            help="CURRENT if within freshness window, STALE if expired",
        )
    with col4:
        st.metric(
            label="Sequence ID",
            value=seq_id.replace("seq_", "")[:18] if seq_id != "N/A" else "N/A",
            help=seq_id,
        )
    with col5:
        st.metric(
            label="Last Forecast Time",
            value=f"{last_ts:.3f}s" if last_ts > 0 else "0.000s",
            help="Telemetry timestamp of most recent multi-horizon prediction",
        )

    # Model availability pills & Calibration status
    st.markdown(
        f"""
        <div style="background: #1e1e24; padding: 10px 16px; border-radius: 6px; font-size: 13px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong>Model Availability:</strong>
                <span style="margin-left: 8px; color: {'#52b788' if models.get('lstm_forecaster') == 'AVAILABLE' else '#f4a261'};">LSTM Forecaster: {models.get('lstm_forecaster', 'UNAVAILABLE')}</span> |
                <span style="margin-left: 8px; color: {'#52b788' if models.get('yolo_detector') == 'AVAILABLE' else '#f4a261'};">YOLO11n Detector: {models.get('yolo_detector', 'UNAVAILABLE')}</span> |
                <span style="margin-left: 8px; color: {'#52b788' if models.get('bytetrack_tracker') == 'AVAILABLE' else '#f4a261'};">ByteTrack Tracker: {models.get('bytetrack_tracker', 'UNAVAILABLE')}</span>
            </div>
            <div>
                <span style="background: #9b5de5; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;">
                    CALIBRATION: PROVISIONAL
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
