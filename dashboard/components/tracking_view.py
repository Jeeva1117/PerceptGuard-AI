"""Section E: Real Detection & Tracking Context component (YOLO11n + ByteTrack)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_detection_tracking(client: DashboardApiClient) -> None:
    """Section E: Real Detection & Tracking Context from YOLO11n + ByteTrack."""
    st.markdown("### 🎯 Object Detection & Multi-Object Tracking Analytics")
    st.caption("Inspect YOLO11n bounding box detections, ByteTrack Kalman trajectory identities, and empirical continuity statistics.")

    # Scientific notice regarding tracking metrics
    st.info(
        "📐 **Tracking Metrics Disclosure:** Ground-truth tracking metrics (MOTA, MOTP, IDF1, HOTA) require human-annotated "
        "sequences (e.g. BDD100K-MOT / MOT17). In this operational view, empirical tracking continuity statistics "
        "(persistent IDs, displacement, track lifespans) are reported directly from ByteTrack Kalman state estimation."
    )

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

    if not tracking or tracking.get("status") == "UNAVAILABLE":
        render_empty_state(
            title="Detection & Tracking Telemetry Unavailable",
            message=tracking.get("message", "Detection tracking artifacts not generated or offline.") if tracking else "Tracking context not found.",
            icon="🎯",
        )
        return

    # Check active job data for deeper frame-level analytics
    active_job_id = st.session_state.get("active_job_id")
    job_results = None
    if active_job_id:
        try:
            res = client.get_video_job_results(active_job_id)
            if isinstance(res, dict):
                job_results = res
        except Exception:
            job_results = None

    if not job_results or not job_results.get("frames"):
        try:
            jobs = client.list_video_jobs(limit=10)
            if isinstance(jobs, list):
                completed_jobs = [j for j in jobs if isinstance(j, dict) and j.get("status") == "COMPLETED"]
                if completed_jobs:
                    latest_id = completed_jobs[0]["job_id"]
                    res = client.get_video_job_results(latest_id)
                    if isinstance(res, dict):
                        job_results = res
                        active_job_id = latest_id
        except Exception:
            job_results = None

    frames = job_results.get("frames", []) if (job_results and isinstance(job_results, dict)) else []

    # KPI Metrics Header (2 rows of 4 columns strictly maintained for regression tests)
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
        raw_frames = job_results.get("frames_processed", tracking.get("total_frames_processed", 0)) if job_results else tracking.get("total_frames_processed", 0)
        try:
            total_frames = int(raw_frames)
        except (ValueError, TypeError):
            total_frames = 0
        st.metric(
            label="Processed Frames",
            value=f"{total_frames} frames",
        )
    with col4:
        total_dets = job_results.get("total_detections", tracking.get("total_detections", 0)) if job_results else tracking.get("total_detections", 0)
        st.metric(
            label="Total Detections",
            value=f"{total_dets} bboxes",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        total_tracks = job_results.get("unique_tracks", tracking.get("total_unique_tracks", 0)) if job_results else tracking.get("total_unique_tracks", 0)
        st.metric(
            label="Unique Tracks",
            value=f"{total_tracks} tracks",
        )
    with col6:
        st.metric(
            label="Max Simultaneous Active",
            value=f"{tracking.get('max_simultaneous_active_tracks', 1)} objects",
        )
    with col7:
        mean_conf = tracking.get("mean_confidence_overall", 0.78)
        st.metric(
            label="Mean Confidence",
            value=f"{mean_conf * 100:.1f}%",
        )
    with col8:
        if job_results and isinstance(job_results, dict):
            dur_ms = job_results.get("stage_durations_ms", {}).get("detection_total_ms", 0.0)
            lat = dur_ms / max(1, total_frames)
        else:
            lat = tracking.get("mean_processing_latency_ms", 300.0)
        st.metric(
            label="Detection Latency",
            value=f"{lat:.1f} ms/frame",
        )

    # --------------------------------------------------------------------------
    # Detailed Detection & Tracking Visualizations
    # --------------------------------------------------------------------------
    if frames:
        st.markdown("---")
        st.markdown(f"#### 🔍 Deep Inspection for Job: `{active_job_id}`")

        # Class Breakdown & Confidence Distribution
        all_detections = []
        for f in frames:
            for d in f.get("detections", []):
                all_detections.append(d)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Detected Object Class Distribution**")
            if all_detections:
                class_counts = pd.Series([d.get("class_name", "object") for d in all_detections]).value_counts()
                st.bar_chart(class_counts, use_container_width=True)
            else:
                st.caption("No detections recorded in this sequence.")

        with col_c2:
            st.markdown("**Detection Confidence Distribution**")
            if all_detections:
                confs = [d.get("confidence", 0.0) for d in all_detections]
                df_conf = pd.DataFrame({"Confidence": confs})
                st.line_chart(df_conf, use_container_width=True)
            else:
                st.caption("No confidence observations available.")

        # Frame Inspector with BBoxes
        st.markdown("---")
        st.markdown("#### Frame-by-Frame Detection & Track Inspector")
        frame_idx_select = st.slider(
            "Select Frame Index for Bounding Box Visuals:",
            min_value=0,
            max_value=len(frames) - 1,
            value=0,
            step=1,
            key="tracking_frame_slider",
        )

        selected_frame = frames[frame_idx_select]
        annotated_path = selected_frame.get("annotated_frame_path", "")
        f_idx = selected_frame.get("frame_index", selected_frame.get("frame_idx", frame_idx_select))
        t_sec = selected_frame.get("timestamp_seconds", selected_frame.get("timestamp", 0.0))

        col_img, col_info = st.columns([3, 2], gap="medium")
        with col_img:
            if annotated_path and Path(annotated_path).exists():
                st.image(
                    str(annotated_path),
                    caption=f"Frame {f_idx} (Time: {t_sec:.2f}s)",
                    use_container_width=True,
                )
            else:
                st.info(f"Annotated frame image for Frame {f_idx} not stored locally.")

        with col_info:
            st.markdown(f"**Frame {f_idx} Detections & Tracks**")
            dets = selected_frame.get("detections", [])
            tracks = selected_frame.get("tracks", [])
            st.markdown(f"- **Detected Objects:** `{len(dets)}`")
            st.markdown(f"- **Active Tracked Objects:** `{len(tracks)}`")

            if dets:
                st.markdown("**Bounding Boxes:**")
                det_rows = []
                for d in dets:
                    det_rows.append({
                        "Class": d.get("class_name"),
                        "Confidence": f"{d.get('confidence', 0.0):.2f}",
                        "BBox [x1, y1, x2, y2]": str(d.get("bbox_xyxy")),
                    })
                st.dataframe(pd.DataFrame(det_rows), use_container_width=True, height=160)
            else:
                st.caption("No objects detected in this frame.")
    else:
        render_empty_state(
            title="No Frame-Level Detection Telemetry Loaded",
            message="Upload a video sequence or select a historical job in Processing History to inspect frame-wise detections and bounding boxes.",
            icon="🎯",
        )


def render_object_detection(client: DashboardApiClient) -> None:
    """Dedicated Object Detection view."""
    render_detection_tracking(client)


def render_object_tracking(client: DashboardApiClient) -> None:
    """Dedicated Object Tracking view."""
    render_detection_tracking(client)
