"""PerceptGuard AI — Early Warning & Perception Reliability Dashboard.

Streamlit interactive application providing real-time and historical perception
reliability monitoring, multi-horizon failure risk forecasting (H1, H3, H5),
temporal risk telemetry, early-warning alert management, and authentic
detection/tracking context.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.early_warning.dashboard_service import DashboardService
from src.early_warning.pipeline_integrator import PipelineIntegrator


# Helper for caching service instance across Streamlit reruns
@st.cache_resource(show_spinner=False)
def get_integrator_and_service() -> tuple[PipelineIntegrator, DashboardService]:
    """Initialize singleton PipelineIntegrator and DashboardService with ingested data."""
    integrator = PipelineIntegrator(root_dir=PROJECT_ROOT)
    # Automatically ingest available verified predictions artifact on startup
    preds_file = (
        PROJECT_ROOT
        / "experiments"
        / "forecasting"
        / "multi_horizon"
        / "official_multi_horizon_run"
        / "predictions.json"
    )
    if preds_file.exists():
        integrator.ingest_predictions_file(preds_file)
    return integrator, integrator.dashboard_service


def render_header():
    """Render top branding header, mission statement, and system banner."""
    settings = get_settings()
    env_str = settings.app.environment.value.upper()
    version_str = settings.app.version
    gate_str = settings.dashboard.calibration_status_label

    st.set_page_config(
        page_title="PerceptGuard AI | Early Warning Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #0d1b2a 0%, #1b263b 100%); padding: 18px 24px; border-radius: 8px; border-left: 6px solid #415a77; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="color: #e0e1dd; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.5px;">
                        🛡️ PerceptGuard AI <span style="font-size: 14px; background: #415a77; color: #fff; padding: 2px 8px; border-radius: 4px; margin-left: 10px;">v{version_str}</span>
                        <span style="font-size: 12px; background: #2a9d8f; color: #fff; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">{env_str}</span>
                    </h1>
                    <p style="color: #778da9; margin: 4px 0 0 0; font-size: 13px;">
                        Autonomous Driving Perception Failure Early Warning & Multi-Horizon Reliability Forecasting System
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 11px; color: #a3b18a; font-family: monospace;">Device: {settings.model.device.upper()} (Windows 11 / Intel Core i3 / 8GB)</span><br/>
                    <span style="font-size: 11px; color: #e0a96d; font-family: monospace;">Gate: {gate_str}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_a_system_overview(service: DashboardService):
    """Section A: System Overview and Operational Health."""
    health = service.get_system_health()
    status = health.get("pipeline_status", "UNKNOWN")
    overall = health.get("overall_health", "UNKNOWN")
    score = health.get("health_score", 0.0)
    freshness = health.get("data_freshness", "UNKNOWN")
    seq_id = health.get("current_sequence_id", "N/A")
    last_ts = health.get("last_prediction_timestamp_seconds", 0.0)
    models = health.get("model_availability", {})

    # Status color mapping
    status_colors = {
        "READY": "#2a9d8f",
        "RUNNING": "#3a86ff",
        "DEGRADED": "#e76f51",
        "BLOCKED": "#d62828",
        "UNAVAILABLE": "#6c757d",
    }
    badge_color = status_colors.get(status, "#6c757d")

    st.subheader("Section A — System Operational Overview")

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
            delta=f"{score:.1f}% Score",
        )
    with col3:
        st.metric(
            label="Data Freshness",
            value=freshness,
            help="CURRENT if within telemetry freshness window, STALE if expired",
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
            value=f"{last_ts:.3f}s",
            help="Telemetry timestamp of most recent multi-horizon prediction",
        )

    # Model availability pills & Calibration status
    st.markdown(
        f"""
        <div style="background: #1e1e24; padding: 10px 16px; border-radius: 6px; font-size: 13px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
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


def render_section_b_risk_overview(service: DashboardService):
    """Section B: Multi-Horizon Reliability Risk Overview (H1, H3, H5)."""
    st.subheader("Section B — Multi-Horizon Reliability Risk Overview")

    overview = service.get_reliability_risk_overview()
    horizons = overview.get("horizons", {})

    st.markdown(
        r"""
        > ℹ️ **Notice:** Multi-horizon risk scores $S \in [0.0, 1.0]$ represent derived perception failure risk ($S = 1.0 - R$).
        > Thresholds are currently designated as **PROVISIONAL** due to zero-inflation in single-clip driving data.
        """
    )

    col1, col2, col3 = st.columns(3)

    def _render_horizon_card(col, h_key: str, h_label: str, horizon_int: int):
        data = horizons.get(h_key, {})
        score = data.get("risk_score")
        level = data.get("risk_level", "UNKNOWN")
        trend = data.get("trend_direction", "UNKNOWN")
        rolling_mean = data.get("rolling_mean")
        volatility = data.get("volatility")
        obs_count = data.get("observation_count", 0)

        # Level color badge
        level_colors = {
            "LOW": "#2a9d8f",
            "MEDIUM": "#e9c46a",
            "HIGH": "#f4a261",
            "CRITICAL": "#e76f51",
            "UNKNOWN": "#6c757d",
        }
        l_color = level_colors.get(level, "#6c757d")

        trend_symbols = {
            "INCREASING": "↗ INCREASING",
            "DECREASING": "↘ DECREASING",
            "STABLE": "→ STABLE",
            "VOLATILE": "↕ VOLATILE",
            "UNKNOWN": "— UNKNOWN",
        }
        t_sym = trend_symbols.get(trend, trend)

        with col:
            st.markdown(
                f"""
                <div style="background: #181926; border: 1px solid #313244; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #cdd6f4;">{h_label} (H{horizon_int})</h4>
                        <span style="background: {l_color}; color: #11111b; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px;">
                            {level}
                        </span>
                    </div>
                    <div style="font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                        {f'{score:.4f}' if score is not None else 'N/A'}
                    </div>
                    <div style="font-size: 12px; color: #a6adc8; margin-bottom: 8px;">
                        Trend: <strong>{t_sym}</strong>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #313244; margin: 8px 0;"/>
                    <div style="font-size: 11px; color: #bac2de; line-height: 1.6;">
                        <strong>Rolling Mean:</strong> {f'{rolling_mean:.4f}' if rolling_mean is not None else 'N/A'}<br/>
                        <strong>Volatility (σ):</strong> {f'{volatility:.4f}' if volatility is not None else 'N/A'}<br/>
                        <strong>Observations:</strong> {obs_count} frames
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _render_horizon_card(col1, "h1", "Immediate Horizon (t+1)", 1)
    _render_horizon_card(col2, "h3", "Intermediate Horizon (t+3)", 3)
    _render_horizon_card(col3, "h5", "Extended Horizon (t+5)", 5)


def render_section_c_temporal_monitoring(service: DashboardService):
    """Section C: Temporal Risk Monitoring & History Visualization."""
    st.subheader("Section C — Temporal Risk Monitoring")

    temp_data = service.get_temporal_monitoring_data(limit=100)
    horizons_hist = temp_data.get("horizons", {})
    transitions = temp_data.get("state_transitions", [])

    # Check if we have history
    has_history = any(len(records) > 0 for records in horizons_hist.values())

    if not has_history:
        st.info("ℹ️ No temporal monitoring history recorded yet. Ingest predictions to populate trajectory charts.")
        return

    # Prepare DataFrame for line chart
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

    if chart_rows:
        df_chart = pd.DataFrame(chart_rows)
        # Pivot table for Streamlit line chart: index=Timestamp, columns=Horizon, values=Risk Score
        df_pivot = df_chart.pivot_table(
            index="Timestamp (s)", columns="Horizon", values="Risk Score"
        )
        st.line_chart(df_pivot, use_container_width=True)
    else:
        st.warning("⚠️ No valid non-null risk observations available for plotting.")

    # State transitions expander
    with st.expander(f"Operational State Transitions ({len(transitions)} events)", expanded=False):
        if transitions:
            df_trans = pd.DataFrame([
                {
                    "Timestamp (s)": f"{t.get('timestamp_seconds', 0.0):.3f}",
                    "Horizon": f"H{t.get('horizon')}",
                    "From Level": t.get("previous_level"),
                    "To Level": t.get("current_level"),
                    "Risk Score": f"{t.get('risk_score', 0.0):.4f}",
                    "Genuine": "Yes" if t.get("is_genuine_transition", True) else "No (Jitter)",
                    "Notes": t.get("notes", ""),
                }
                for t in reversed(transitions[-20:])
            ])
            st.dataframe(df_trans, use_container_width=True)
        else:
            st.caption("No operational state transitions recorded.")


def render_section_d_early_warning_alerts(service: DashboardService):
    """Section D: Early Warning Alerts Feed, Filtering, and Lifecycle Management."""
    st.subheader("Section D — Early Warning Alerts & Incident Feed")

    # Filter controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sel_sev = st.selectbox(
            "Filter Severity",
            options=["ALL", "CRITICAL", "WARNING", "CAUTION", "INFO", "DATA_QUALITY_WARNING"],
            index=0,
        )
    with col_f2:
        sel_horiz = st.selectbox(
            "Filter Horizon",
            options=["ALL", "H1 (t+1)", "H3 (t+3)", "H5 (t+5)"],
            index=0,
        )
    with col_f3:
        sel_type = st.selectbox(
            "Filter Alert Type",
            options=[
                "ALL",
                "RISK_LEVEL_THRESHOLD",
                "RAPID_RISK_INCREASE",
                "VOLATILITY_SPIKE",
                "CONSECUTIVE_DEGRADATION",
                "HORIZON_DIVERGENCE",
                "DATA_QUALITY",
            ],
            index=0,
        )
    with col_f4:
        sel_status = st.selectbox(
            "Filter Lifecycle State",
            options=["ALL", "ACTIVE", "ACKNOWLEDGED", "RESOLVED", "EXPIRED"],
            index=0,
        )

    # Parse filters
    sev_filter = None if sel_sev == "ALL" else sel_sev
    horizon_filter = None
    if sel_horiz == "H1 (t+1)":
        horizon_filter = 1
    elif sel_horiz == "H3 (t+3)":
        horizon_filter = 3
    elif sel_horiz == "H5 (t+5)":
        horizon_filter = 5
    type_filter = None if sel_type == "ALL" else sel_type
    status_filter = None if sel_status == "ALL" else sel_status

    alerts_res = service.get_alerts(
        severity=sev_filter,
        horizon=horizon_filter,
        alert_type=type_filter,
        status=status_filter,
        page=1,
        page_size=50,
    )

    alerts = alerts_res.get("alerts", [])
    total_count = alerts_res.get("total_count", 0)
    active_count = alerts_res.get("active_alerts_count", 0)

    st.markdown(
        f"**Showing {len(alerts)} of {total_count} matching alerts** (Active: `{active_count}`)"
    )

    if not alerts:
        st.info("No alerts match the selected criteria.")
    else:
        # Table view
        table_rows = []
        for a in alerts:
            table_rows.append({
                "Alert ID": a["alert_id"],
                "Severity": a["severity"],
                "Type": a["alert_type"],
                "Horizon": f"H{a['trigger_horizon']}",
                "Score": f"{a['risk_score']:.4f}",
                "State": a["lifecycle_state"],
                "Time (s)": f"{a['timestamp_seconds']:.3f}",
                "Message": a["warning_message"][:60] + "...",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Detail Inspection & Management
        st.markdown("#### Alert Detail Inspection & Operator Actions")
        alert_ids = [a["alert_id"] for a in alerts]
        selected_alert_id = st.selectbox("Select Alert to Inspect / Manage", options=alert_ids)

        if selected_alert_id:
            detail = service.get_alert_detail(selected_alert_id)
            if detail:
                col_d1, col_d2 = st.columns([2, 1])
                with col_d1:
                    st.markdown(
                        f"""
                        **Alert ID:** `{detail['alert_id']}`<br/>
                        **Rule Triggered:** `{detail['rule_id']}` | **Type:** `{detail['alert_type']}`<br/>
                        **Severity:** `{detail['severity']}` | **Status:** `{detail['lifecycle_state']}`<br/>
                        **Message:** {detail['warning_message']}<br/>
                        **Trigger Rationale:** {detail.get('trigger_rationale', 'N/A')}<br/>
                        **Evidence:** Frame `{detail.get('source_frame_id')}` &rarr; Target `{detail.get('target_frame_id')}` | Model: `{detail.get('model_version')}`
                        """,
                        unsafe_allow_html=True,
                    )
                with col_d2:
                    st.markdown("**Operator Lifecycle Actions**")
                    curr_state = detail["lifecycle_state"]
                    can_ack = curr_state == "ACTIVE"
                    can_resolve = curr_state in ("ACTIVE", "ACKNOWLEDGED")

                    c_act1, c_act2 = st.columns(2)
                    with c_act1:
                        if st.button("Acknowledge", disabled=not can_ack, key=f"btn_ack_{selected_alert_id}"):
                            ack_res = service.acknowledge_alert(selected_alert_id, operator_id="dashboard_operator")
                            if ack_res["acknowledged"]:
                                st.success("Alert Acknowledged!")
                                st.rerun()
                            else:
                                st.error("Failed to acknowledge.")
                    with c_act2:
                        if st.button("Resolve", disabled=not can_resolve, key=f"btn_res_{selected_alert_id}"):
                            res_res = service.resolve_alert(
                                selected_alert_id, reason="Resolved from dashboard", operator_id="dashboard_operator"
                            )
                            if res_res["resolved"]:
                                st.success("Alert Resolved!")
                                st.rerun()
                            else:
                                st.error("Failed to resolve.")

                # Lifecycle history audit trail
                events = detail.get("lifecycle_events", [])
                if events:
                    with st.expander(f"Audit Trail ({len(events)} lifecycle events)", expanded=False):
                        st.dataframe(pd.DataFrame(events), use_container_width=True)


def render_section_e_detection_tracking(service: DashboardService):
    """Section E: Real Detection & Tracking Context from YOLO11n + ByteTrack."""
    st.subheader("Section E — Detection & Tracking Telemetry Context")

    tracking = service.get_detection_tracking_context()
    status = tracking.get("status", "UNAVAILABLE")

    if status != "AVAILABLE":
        st.warning(f"⚠️ Tracking Context: {tracking.get('message', 'Artifacts unavailable.')}")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Detection Engine",
            value=tracking.get("model_name", "YOLO11n").upper(),
            help="Pretrained YOLO11 nano model on CPU",
        )
    with col2:
        st.metric(
            label="Multi-Object Tracker",
            value=tracking.get("tracker_type", "ByteTrack").capitalize(),
            help="ByteTrack two-stage Kalman filter data association",
        )
    with col3:
        st.metric(
            label="Processed Frames",
            value=f"{tracking.get('total_frames_processed', 0)} frames",
        )
    with col4:
        st.metric(
            label="Total Detections",
            value=f"{tracking.get('total_detections', 0)} bboxes",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            label="Unique Tracks",
            value=f"{tracking.get('total_unique_tracks', 0)} tracks",
        )
    with col6:
        st.metric(
            label="Max Active Simultaneous",
            value=f"{tracking.get('max_simultaneous_active_tracks', 0)} vehicles",
        )
    with col7:
        mean_conf = tracking.get("mean_confidence_overall", 0.0)
        st.metric(
            label="Mean Confidence",
            value=f"{mean_conf * 100:.1f}%",
        )
    with col8:
        lat = tracking.get("mean_processing_latency_ms", 0.0)
        st.metric(
            label="Detection Latency",
            value=f"{lat:.1f} ms/frame",
        )


def render_pipeline_status_accordion(service: DashboardService):
    """Render 8-stage perception pipeline component status accordion."""
    status_dict = service.get_pipeline_status()
    components = status_dict.get("components", {})

    with st.expander("PerceptGuard AI 8-Stage Architecture Pipeline Status", expanded=False):
        stage_rows = []
        for stage, comp in components.items():
            stage_rows.append({
                "Stage Name": stage.replace("_", " ").title(),
                "Operational Status": comp.get("status"),
                "Details": comp.get("details"),
            })
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)


def main():
    """Main dashboard entry point."""
    render_header()
    integrator, service = get_integrator_and_service()

    # Sidebar controls
    settings = get_settings()
    with st.sidebar:
        st.header("⚙️ Controls & Ingestion")
        st.markdown(
            f"""
            **Environment:** `{settings.app.environment.value}`<br/>
            **Backend API:** `{settings.dashboard.api_base_url}`<br/>
            **Dataset:** `driving_sample` (720p 30fps)<br/>
            **Multi-Horizon Run:** `official_multi_horizon_run`<br/>
            **Forecasting Model:** `LSTMForecaster` (H1, H3, H5)
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if st.button("🔄 Refresh Telemetry"):
            st.rerun()

        st.markdown("---")
        st.caption("PerceptGuard AI — Day 11 Hardened Config Foundation")

    # Render main 5 sections
    render_section_a_system_overview(service)
    st.markdown("---")
    render_section_b_risk_overview(service)
    st.markdown("---")
    render_section_c_temporal_monitoring(service)
    st.markdown("---")
    render_section_d_early_warning_alerts(service)
    st.markdown("---")
    render_section_e_detection_tracking(service)
    st.markdown("---")
    render_pipeline_status_accordion(service)


if __name__ == "__main__":
    main()
