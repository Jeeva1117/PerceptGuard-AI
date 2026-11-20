"""PerceptGuard AI — Early Warning & Perception Reliability Dashboard.

Streamlit interactive application providing real-time and historical perception
reliability monitoring, multi-horizon failure risk forecasting (H1, H3, H5),
temporal risk telemetry, early-warning alert management, and authentic
detection/tracking context. Hardened for production deployment with dual
REST API / In-Process fallback architectures.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Streamlit page config MUST be called before any other Streamlit UI command
st.set_page_config(
    page_title="PerceptGuard AI | Early Warning Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import get_settings
from src.early_warning.dashboard_service import DashboardService
from src.early_warning.pipeline_integrator import PipelineIntegrator
from dashboard.api_client import DashboardApiClient
from dashboard.components import (
    render_header,
    render_system_overview,
    render_risk_overview,
    render_temporal_monitoring,
    render_early_warning_alerts,
    render_detection_tracking,
    render_pipeline_status_accordion,
    render_video_processing_view,
)


@st.cache_resource(show_spinner=False)
def get_integrator_and_service() -> tuple[PipelineIntegrator, DashboardService]:
    """Initialize singleton PipelineIntegrator and DashboardService with ingested data."""
    integrator = PipelineIntegrator(root_dir=PROJECT_ROOT)
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


@st.cache_resource(show_spinner=False)
def get_dashboard_client() -> DashboardApiClient:
    """Initialize singleton DashboardApiClient with automatic backend detection & in-process fallback."""
    settings = get_settings()
    client = DashboardApiClient(
        base_url=settings.dashboard.api_base_url,
        timeout_seconds=settings.dashboard.api_request_timeout_seconds,
        settings=settings,
    )
    # Check if backend is reachable; if not, initialize in-process fallback
    is_live = client.probe_backend()
    if not is_live:
        client.initialize_in_process_service(root_dir=PROJECT_ROOT)
    return client


def main() -> None:
    """Main dashboard entry point."""
    settings = get_settings()
    client = get_dashboard_client()

    # Re-probe on run to ensure transport mode is up to date
    client.probe_backend()
    if client.transport_mode == "IN_PROCESS" and not client.is_in_process_ready:
        client.initialize_in_process_service(root_dir=PROJECT_ROOT)

    calibration_label = settings.dashboard.calibration_status_label

    # Render header banner
    render_header(
        transport_mode=client.transport_mode,
        calibration_status=calibration_label,
    )

    # Sidebar configuration and diagnostics
    with st.sidebar:
        st.header("⚙️ Controls & Navigation")
        nav_selection = st.radio(
            "Navigation Menu:",
            [
                "📊 Operations Overview",
                "🎥 Video Pipeline & Ingestion",
                "📈 Temporal Risk & Tracking",
                "🚨 Early-Warning Alerts",
                "🔍 Diagnostics & Architecture",
            ],
            index=0,
        )
        st.markdown("---")

        mode_badge_color = "#2a9d8f" if client.transport_mode == "REST_API" else "#e76f51"
        st.markdown(
            f"""
            <div style="background: #181926; border: 1px solid #313244; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; font-size: 12px;">
                <strong>Transport Mode:</strong>
                <span style="background: {mode_badge_color}; color: #ffffff; padding: 1px 6px; border-radius: 3px; font-weight: 700; font-size: 10px; margin-left: 6px;">
                    {client.transport_mode}
                </span><br/>
                <strong>Backend URL:</strong> <code>{client.base_url}</code><br/>
                <strong>Environment:</strong> <code>{settings.app.environment.value}</code><br/>
                <strong>Freshness TTL:</strong> <code>{settings.dashboard.telemetry_freshness_ttl_seconds}s</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            **Dataset:** `driving_sample` (720p 30fps)<br/>
            **Multi-Horizon Run:** `official_multi_horizon_run`<br/>
            **Forecasting Model:** `LSTMForecaster` (H1, H3, H5)<br/>
            **Gate:** `{calibration_label}`
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.button("🔄 Refresh Telemetry", use_container_width=True):
            client.probe_backend()
            st.rerun()

        st.markdown("---")
        st.caption("PerceptGuard AI — Day 13 Integrated Dashboard")

    # Render Active Navigation View
    if nav_selection == "📊 Operations Overview":
        render_system_overview(client)
        st.markdown("---")
        render_risk_overview(client)
        st.markdown("---")
        render_temporal_monitoring(client)
        st.markdown("---")
        render_early_warning_alerts(client)
        st.markdown("---")
        render_detection_tracking(client)
        st.markdown("---")
        render_pipeline_status_accordion(client)

    elif nav_selection == "🎥 Video Pipeline & Ingestion":
        render_video_processing_view(client)

    elif nav_selection == "📈 Temporal Risk & Tracking":
        render_risk_overview(client)
        st.markdown("---")
        render_temporal_monitoring(client)
        st.markdown("---")
        render_detection_tracking(client)

    elif nav_selection == "🚨 Early-Warning Alerts":
        render_early_warning_alerts(client)

    elif nav_selection == "🔍 Diagnostics & Architecture":
        render_pipeline_status_accordion(client)
        st.markdown("---")
        render_system_overview(client)



if __name__ == "__main__":
    main()

# Backward compatibility function aliases
render_section_a_system_overview = render_system_overview
render_section_b_risk_overview = render_risk_overview
render_section_c_temporal_monitoring = render_temporal_monitoring
render_section_d_early_warning_alerts = render_early_warning_alerts
render_section_e_detection_tracking = render_detection_tracking

