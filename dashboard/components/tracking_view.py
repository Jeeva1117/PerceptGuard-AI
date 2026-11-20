"""Section E: Real Detection & Tracking Context component (YOLO11n + ByteTrack)."""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_detection_tracking(client: DashboardApiClient) -> None:
    """Section E: Real Detection & Tracking Context from YOLO11n + ByteTrack."""
    st.subheader("Section E — Detection & Tracking Telemetry Context")

    try:
        tracking = client.get_tracking_context()
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load Detection & Tracking Telemetry",
            error_message=f"Backend communication error: {err.message}",
            retry_callback=st.rerun,
            key_suffix="tracking_context",
        )
        return

    status = tracking.get("status", "UNAVAILABLE")

    if status != "AVAILABLE":
        render_empty_state(
            title="Tracking Context Unavailable",
            message=tracking.get(
                "message",
                "YOLO11n + ByteTrack detection artifacts are not available or not loaded.",
            ),
            icon="🎯",
        )
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
