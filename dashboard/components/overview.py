"""Executive Overview and Operational KPI Dashboard component."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_error_boundary, render_stale_warning


def render_system_overview(client: DashboardApiClient) -> None:
    """Render executive operational overview, high-level KPIs, and pipeline health."""
    st.markdown("### 📊 Executive Operational Overview")
    st.caption("Live enterprise telemetry, perception pipeline health, model status, and processing summaries.")

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

    # Fetch jobs history for high-level statistics
    try:
        jobs = client.list_video_jobs(limit=50)
    except Exception:
        jobs = []

    total_jobs = len(jobs)
    completed_jobs = sum(1 for j in jobs if j.get("status") == "COMPLETED")
    failed_jobs = sum(1 for j in jobs if j.get("status") == "FAILED")
    in_flight_jobs = sum(1 for j in jobs if j.get("status") in ("QUEUED", "PROCESSING", "VALIDATING"))
    total_detections_all = sum(j.get("total_detections", 0) for j in jobs)
    unique_tracks_all = sum(j.get("unique_tracks", 0) for j in jobs)

    # Telemetry metrics from health
    status = health.get("pipeline_status", "UNKNOWN")
    overall = health.get("overall_health", "UNKNOWN")
    score = health.get("health_score", 0.0)
    freshness = health.get("data_freshness", "UNKNOWN")
    seq_id = health.get("current_sequence_id", "N/A")
    last_ts = health.get("last_prediction_timestamp_seconds", 0.0)
    models = health.get("model_availability", {})

    # Fetch current risk overview
    try:
        risk_data = client.get_risk_overview()
        h1_data = risk_data.get("horizons", {}).get("h1", {})
        current_risk_score = h1_data.get("risk_score")
        current_risk_level = h1_data.get("risk_level", "UNKNOWN")
    except Exception:
        current_risk_score = None
        current_risk_level = "UNKNOWN"

    # Fetch active alerts count
    try:
        alerts_data = client.get_alerts(status="ACTIVE", page_size=1)
        active_alerts_count = alerts_data.get("total_alerts", 0)
    except Exception:
        active_alerts_count = 0

    # Staleness warning if data expired
    render_stale_warning(
        last_sync=client.last_sync_time,
        freshness=freshness,
        ttl_seconds=client.settings.dashboard.telemetry_freshness_ttl_seconds,
    )

    # --------------------------------------------------------------------------
    # Tier 1: System Health & Pipeline Operations KPIs
    # --------------------------------------------------------------------------
    st.markdown("#### System Health & Telemetry Status")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(
            label="Pipeline Status",
            value=status,
            help="Status values: READY, RUNNING, DEGRADED, BLOCKED, UNAVAILABLE",
        )
    with kpi2:
        st.metric(
            label="Overall Perception Health",
            value=overall,
            delta=f"{score:.1f}% Score" if score > 0 else None,
        )
    with kpi3:
        st.metric(
            label="Data Freshness",
            value=freshness,
            help="CURRENT if within freshness window, STALE if expired",
        )
    with kpi4:
        st.metric(
            label="Active Alert Incidents",
            value=active_alerts_count,
            delta=f"{active_alerts_count} Active" if active_alerts_count > 0 else "Nominal",
            delta_color="inverse" if active_alerts_count > 0 else "normal",
        )
    with kpi5:
        risk_val_str = f"{current_risk_score:.3f}" if current_risk_score is not None else "N/A"
        st.metric(
            label="Current H1 Risk",
            value=risk_val_str,
            delta=current_risk_level if current_risk_level != "UNKNOWN" else None,
            delta_color="inverse" if current_risk_level in ("HIGH", "CRITICAL") else "normal",
        )

    # --------------------------------------------------------------------------
    # Tier 2: Video Processing Throughput KPIs
    # --------------------------------------------------------------------------
    st.markdown("#### Video Processing & Perception Throughput")
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    with pk1:
        st.metric(label="Total Processed Videos", value=total_jobs)
    with pk2:
        st.metric(label="Completed Jobs", value=completed_jobs)
    with pk3:
        st.metric(label="Failed / Rejected Jobs", value=failed_jobs)
    with pk4:
        st.metric(label="Total Objects Detected", value=total_detections_all)
    with pk5:
        st.metric(label="Unique Object Tracks", value=unique_tracks_all)

    # --------------------------------------------------------------------------
    # Model Availability Banner & Calibration Notice
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: #181926; border: 1px solid #313244; padding: 12px 18px; border-radius: 8px; font-size: 13px; margin: 16px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <strong>Model Checkpoint Status:</strong>
                <span style="margin-left: 10px; color: {'#52b788' if models.get('lstm_forecaster') == 'AVAILABLE' else '#f4a261'}; font-weight: 600;">
                    LSTM Forecaster: {models.get('lstm_forecaster', 'UNAVAILABLE')}
                </span> |
                <span style="margin-left: 10px; color: {'#52b788' if models.get('yolo_detector') == 'AVAILABLE' else '#f4a261'}; font-weight: 600;">
                    YOLO11n Detector: {models.get('yolo_detector', 'UNAVAILABLE')}
                </span> |
                <span style="margin-left: 10px; color: {'#52b788' if models.get('bytetrack_tracker') == 'AVAILABLE' else '#f4a261'}; font-weight: 600;">
                    ByteTrack Tracker: {models.get('bytetrack_tracker', 'UNAVAILABLE')}
                </span>
            </div>
            <div>
                <span style="background: #9b5de5; color: #ffffff; padding: 3px 10px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.5px;">
                    CALIBRATION: PROVISIONAL
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Active Jobs Quick Table
    # --------------------------------------------------------------------------
    if jobs:
        st.markdown("#### Recent Job Submissions")
        table_rows = []
        for j in jobs[:5]:
            table_rows.append({
                "Job ID": j.get("job_id"),
                "Video Filename": j.get("filename"),
                "Status": j.get("status"),
                "Frames": j.get("frames_processed", 0),
                "Detections": j.get("total_detections", 0),
                "Tracks": j.get("unique_tracks", 0),
                "Alerts": j.get("alerts_count", 0),
                "Submitted At": j.get("created_at", "")[:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
