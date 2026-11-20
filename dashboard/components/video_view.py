"""Video ingestion, pipeline processing, and annotated frame inspection component.

Allows operators to upload driving videos, monitor real-time background processing stages,
inspect multi-horizon failure forecasts, and view annotated frames with bounding boxes,
ByteTrack identities, and telemetry risk indicators.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.api_client import DashboardApiClient
from dashboard.components.state_views import (
    render_empty_state,
    render_error_boundary,
    render_stale_warning,
)


def render_video_processing_view(client: DashboardApiClient) -> None:
    """Render complete video upload, processing, and visualization console."""
    st.markdown("### 🎥 Video Ingestion & Perception Reliability Pipeline")
    st.caption(
        "Upload an autonomous driving camera video to execute the full pipeline: "
        "YOLO Detection ➔ ByteTrack Tracking ➔ 27 Multi-Modal Features ➔ "
        "LSTM Multi-Horizon Forecasting (H1, H3, H5) ➔ Reliability Risk Scoring ➔ Early Warning Alerts."
    )

    col_upload, col_preview = st.columns([1, 1], gap="medium")

    with col_upload:
        st.markdown("#### 1. Ingest Video File")
        uploaded_file = st.file_uploader(
            "Select driving sequence (MP4, AVI, MOV):",
            type=["mp4", "avi", "mov", "mkv"],
            help="Maximum upload ceiling: 100 MB. Optimized for CPU execution.",
        )

        max_frames = st.slider(
            "Analysis Horizon (Max Frames on CPU):",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Limits processing length to respect CPU and memory limits (Intel Core i3 / 8 GB RAM).",
        )

        submit_btn = st.button("🚀 Process Video Pipeline", type="primary", use_container_width=True)

    with col_preview:
        st.markdown("#### 2. Video Preview & Metadata")
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            size_mb = len(file_bytes) / (1024 * 1024)
            st.video(file_bytes)
            st.caption(f"**Filename:** `{uploaded_file.name}` | **Size:** `{size_mb:.2f} MB`")
        else:
            st.info("No video loaded. Upload a video file above to view container preview and properties.")

    # --------------------------------------------------------------------------
    # Job Submission Handling
    # --------------------------------------------------------------------------
    if submit_btn and uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        with st.spinner("Submitting video to processing pipeline..."):
            try:
                sub_res = client.submit_video_job(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    max_frames=max_frames,
                )
                st.session_state["active_job_id"] = sub_res["job_id"]
                st.success(f"Job dispatched successfully! Job ID: `{sub_res['job_id']}`")
            except Exception as e:
                st.error(f"Failed to submit video: {e}")

    # --------------------------------------------------------------------------
    # Active Job Polling & Progress HUD
    # --------------------------------------------------------------------------
    active_job_id = st.session_state.get("active_job_id")
    if active_job_id:
        st.markdown("---")
        st.markdown(f"#### ⚡ Active Job Execution: `{active_job_id}`")

        job_info = client.get_video_job_status(active_job_id)
        job_status = job_info.get("status", "UNKNOWN")
        stage = job_info.get("stage", "UNKNOWN")
        progress = float(job_info.get("progress_percentage", 0.0))
        msg = job_info.get("message", "")

        # Progress bar
        st.progress(min(1.0, max(0.0, progress / 100.0)))

        badge_color = (
            "#2a9d8f" if job_status == "COMPLETED" else
            "#e76f51" if job_status == "FAILED" else
            "#457b9d"
        )
        st.markdown(
            f"""
            <div style="background: #1e1e2e; border: 1px solid #313244; border-radius: 8px; padding: 10px 16px; margin: 8px 0 16px 0;">
                <strong>Status:</strong>
                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right: 12px;">
                    {job_status}
                </span>
                <strong>Stage:</strong> <code>{stage}</code> &nbsp;|&nbsp;
                <strong>Progress:</strong> <code>{progress:.1f}%</code> &nbsp;|&nbsp;
                <strong>Activity:</strong> {msg}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Polling trigger if processing
        if job_status in ("QUEUED", "VALIDATING", "PROCESSING"):
            time.sleep(1.0)
            st.rerun()

        # If completed, render full results
        if job_status == "COMPLETED":
            results = client.get_video_job_results(active_job_id)
            if results:
                render_job_results(results, client, active_job_id)

    # --------------------------------------------------------------------------
    # Job History Table
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📋 Recent Video Processing Jobs")
    try:
        jobs = client.list_video_jobs(limit=10)
        if jobs:
            table_rows = []
            for j in jobs:
                table_rows.append({
                    "Job ID": j.get("job_id"),
                    "Filename": j.get("filename"),
                    "Status": j.get("status"),
                    "Stage": j.get("stage"),
                    "Frames": j.get("frames_processed"),
                    "Detections": j.get("total_detections"),
                    "Tracks": j.get("unique_tracks"),
                    "Alerts": j.get("alerts_count"),
                    "Created": j.get("created_at")[:19].replace("T", " ") if j.get("created_at") else "-",
                })
            st.dataframe(table_rows, use_container_width=True)
        else:
            st.caption("No historical video jobs recorded in this session.")
    except Exception as e:
        st.caption(f"Could not load jobs history: {e}")


def render_job_results(results: Dict[str, Any], client: DashboardApiClient, job_id: str) -> None:
    """Render detailed visualization of a completed video processing job."""
    st.markdown("### 📊 Pipeline Analysis Results")

    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Frames Analyzed", results.get("frames_processed", 0))
    kpi2.metric("Objects Detected", results.get("total_detections", 0))
    kpi3.metric("Unique Tracks", results.get("unique_tracks", 0))
    kpi4.metric("Alerts Triggered", results.get("total_alerts", 0))
    kpi5.metric("Latency", f"{results.get('total_duration_seconds', 0.0):.2f}s")

    # Horizon Risk Scorecards
    st.markdown("#### Multi-Horizon Risk Assessment")
    per_h = results.get("per_horizon_risk", {})
    rh1, rh3, rh5 = st.columns(3)

    for col, h_name, h_key in [(rh1, "Horizon 1 (t+1)", "h1"), (rh3, "Horizon 3 (t+3)", "h3"), (rh5, "Horizon 5 (t+5)", "h5")]:
        h_info = per_h.get(h_key, {})
        latest_risk = h_info.get("latest_risk", 0.0)
        category = h_info.get("category", "UNKNOWN")
        bg_color = (
            "#e63946" if category == "CRITICAL" else
            "#f4a261" if category == "HIGH" else
            "#e9c46a" if category == "MEDIUM" else
            "#2a9d8f" if category == "LOW" else "#6c757d"
        )
        with col:
            st.markdown(
                f"""
                <div style="background: #181926; border: 1px solid #313244; border-radius: 8px; padding: 14px; text-align: center;">
                    <div style="font-size: 13px; color: #a6adc8; margin-bottom: 4px;">{h_name}</div>
                    <div style="font-size: 26px; font-weight: bold; color: {bg_color}; margin-bottom: 4px;">
                        {latest_risk:.3f}
                    </div>
                    <span style="background: {bg_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                        {category}
                    </span>
                    <div style="font-size: 11px; color: #6c7086; margin-top: 6px;">
                        Mean: {h_info.get('mean_risk', 0.0):.3f} | Max: {h_info.get('max_risk', 0.0):.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Annotated Frame Inspection
    frames = results.get("frames", [])
    if frames:
        st.markdown("---")
        st.markdown("#### 🖼️ Annotated Frame Inspector")
        st.caption("Browse frame-by-frame perception results with YOLO bounding boxes, ByteTrack IDs, and HUD risk telemetry.")

        frame_idx_select = st.slider(
            "Select Frame Index:",
            min_value=0,
            max_value=len(frames) - 1,
            value=0,
            step=1,
        )

        selected_frame = frames[frame_idx_select]
        annotated_path = selected_frame.get("annotated_frame_path", "")

        col_img, col_frame_info = st.columns([3, 2], gap="medium")

        with col_img:
            if Path(annotated_path).exists():
                st.image(str(annotated_path), caption=f"Frame {selected_frame['frame_index']} (Timestamp: {selected_frame['timestamp_seconds']}s)", use_container_width=True)
            else:
                st.info(f"Frame image not found at `{annotated_path}`.")

        with col_frame_info:
            st.markdown(f"**Frame {selected_frame['frame_index']} Diagnostics**")
            st.markdown(f"- **Timestamp:** `{selected_frame['timestamp_seconds']}s`")
            st.markdown(f"- **Detections Count:** `{selected_frame['detections_count']}`")
            st.markdown(f"- **Active Tracks:** `{selected_frame['tracks_count']}`")
            st.markdown(f"- **Laplacian Blur:** `{selected_frame['visual_quality'].get('blur_laplacian', 0.0)}`")
            st.markdown(f"- **Mean Brightness:** `{selected_frame['visual_quality'].get('brightness', 0.0)}`")
            st.markdown(f"- **RMS Contrast:** `{selected_frame['visual_quality'].get('contrast', 0.0)}`")
            st.markdown(f"- **Frame Latency:** `{selected_frame['latency_ms']}ms`")

            # Objects table
            dets = selected_frame.get("detections", [])
            if dets:
                st.markdown("**Detected Objects:**")
                det_table = [{"Class": d["class_name"], "Confidence": f"{d['confidence']:.2f}", "BBox": str(d["bbox_xyxy"])} for d in dets]
                st.dataframe(det_table, use_container_width=True, height=140)
            else:
                st.caption("No objects detected in this frame.")

    # Generated Alerts Table
    st.markdown("---")
    st.markdown("#### 🚨 Early-Warning Alerts Triggered in this Video")
    video_alerts = []
    for f in frames:
        for a in f.get("alerts", []):
            video_alerts.append({
                "Alert ID": a.get("alert_id"),
                "Horizon": f"H{a.get('horizon')}",
                "Severity": a.get("severity"),
                "Risk Score": f"{a.get('risk_score', 0.0):.3f}",
                "Reason": a.get("trigger_reason"),
                "Time": f"{a.get('source_timestamp_seconds', 0.0):.2f}s",
            })

    if video_alerts:
        st.dataframe(video_alerts, use_container_width=True)
    else:
        st.success("No critical failure alerts triggered. Perception reliability remained within nominal thresholds.")
