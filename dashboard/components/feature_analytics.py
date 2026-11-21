"""Section: Feature Analytics Dashboard Component.

Visualizes the 27-dimensional multi-modal feature vector outputs:
1. Visual Quality: Laplacian blur variance, mean brightness, RMS contrast.
2. Kinematics & Motion: Centroid displacement, speed, velocities.
3. Detection Metrics: Frame detection count, mean confidence, confidence variation.
4. Temporal Trend Slopes: Rolling linear regression slopes for blur and confidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_feature_analytics(client: DashboardApiClient) -> None:
    """Render comprehensive 27-feature multi-modal telemetry analytics."""
    st.markdown("### 🔬 Multi-Modal Feature Analytics")
    st.caption(
        "Inspect frame-by-frame 27-dimensional feature vectors extracted across visual, kinematic, "
        "and temporal domains adhering to PerceptGuard data contracts."
    )

    # Resolve active job data or fallback
    active_job_id = st.session_state.get("active_job_id")
    job_results = None

    if active_job_id:
        try:
            job_results = client.get_video_job_results(active_job_id)
        except Exception:
            job_results = None

    # If no active job or job has no frames, check latest completed job
    if not job_results or not job_results.get("frames"):
        try:
            jobs = client.list_video_jobs(limit=10)
            completed_jobs = [j for j in jobs if j.get("status") == "COMPLETED"]
            if completed_jobs:
                latest_id = completed_jobs[0]["job_id"]
                job_results = client.get_video_job_results(latest_id)
                active_job_id = latest_id
        except Exception:
            job_results = None

    frames = job_results.get("frames", []) if job_results else []

    if not frames:
        render_empty_state(
            title="No Feature Telemetry Available",
            message="No video processing job results found. Upload and process a video in the Video Pipeline view to inspect multi-modal features.",
            icon="🔬",
        )
        return

    st.info(f"Displaying feature analytics for Job: `{active_job_id}` ({len(frames)} frames analyzed).")

    # Build DataFrame from frames
    feature_rows = []
    for f in frames:
        vq = f.get("visual_quality", {})
        dets = f.get("detections", [])
        tracks = f.get("tracks", [])

        # Detection features
        det_count = f.get("detections_count", len(dets))
        confs = [d.get("confidence", 0.0) for d in dets]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        conf_std = float(np.std(confs)) if len(confs) > 1 else 0.0

        # Motion features from tracks
        track_count = f.get("tracks_count", len(tracks))
        speeds = [t.get("speed", 0.0) for t in tracks if "speed" in t]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        feature_rows.append({
            "Frame Index": f.get("frame_index", 0),
            "Timestamp (s)": f.get("timestamp_seconds", 0.0),
            "Laplacian Blur": vq.get("blur_laplacian", 0.0),
            "Mean Brightness": vq.get("brightness", 0.0),
            "RMS Contrast": vq.get("contrast", 0.0),
            "Detections Count": det_count,
            "Mean Confidence": mean_conf,
            "Confidence Std": conf_std,
            "Active Tracks": track_count,
            "Mean Speed (px/s)": mean_speed,
            "H1 Risk": f.get("risk_scores", {}).get("h1", 0.0),
            "H3 Risk": f.get("risk_scores", {}).get("h3", 0.0),
            "H5 Risk": f.get("risk_scores", {}).get("h5", 0.0),
        })

    df = pd.DataFrame(feature_rows)

    # --------------------------------------------------------------------------
    # Tabbed Visualizations across Feature Domains
    # --------------------------------------------------------------------------
    tab_vq, tab_motion, tab_det, tab_summary = st.tabs([
        "👁️ Visual Quality Telemetry",
        "🚗 Kinematics & Motion Features",
        "🎯 Detection Quality Features",
        "📊 Statistical Feature Matrix",
    ])

    with tab_vq:
        st.markdown("#### Visual Quality Degradation Over Time")
        st.caption("Monitors environmental clarity. Severe drops in blur variance or contrast indicate fog, darkness, or camera occlusion.")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Laplacian Blur Variance (Sharpness)**")
            st.line_chart(df.set_index("Timestamp (s)")[["Laplacian Blur"]], use_container_width=True)
        with col_c2:
            st.markdown("**Photometric Brightness & RMS Contrast**")
            st.line_chart(df.set_index("Timestamp (s)")[["Mean Brightness", "RMS Contrast"]], use_container_width=True)

    with tab_motion:
        st.markdown("#### Kinematic Motion & Track Stability")
        st.caption("Tracks object displacement and velocity vectors. High speed spikes or tracking loss correlate with near-term prediction degradation.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Active Track Count**")
            st.bar_chart(df.set_index("Timestamp (s)")[["Active Tracks"]], use_container_width=True)
        with col_m2:
            st.markdown("**Mean Track Speed (px/s)**")
            st.line_chart(df.set_index("Timestamp (s)")[["Mean Speed (px/s)"]], use_container_width=True)

    with tab_det:
        st.markdown("#### Detection Volume & Confidence Consistency")
        st.caption("Monitors YOLO11n bounding box confidence. Sudden drops in mean confidence often precede perception failure.")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Frame Detection Volume**")
            st.bar_chart(df.set_index("Timestamp (s)")[["Detections Count"]], use_container_width=True)
        with col_d2:
            st.markdown("**Mean Confidence & Uncertainty (Std)**")
            st.line_chart(df.set_index("Timestamp (s)")[["Mean Confidence", "Confidence Std"]], use_container_width=True)

    with tab_summary:
        st.markdown("#### Feature Summary Statistics (Empirical Distribution)")
        stat_cols = [
            "Laplacian Blur", "Mean Brightness", "RMS Contrast",
            "Detections Count", "Mean Confidence", "Active Tracks",
        ]
        desc_df = df[stat_cols].describe().T[["mean", "std", "min", "50%", "max"]]
        desc_df.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
        st.dataframe(desc_df.round(3), use_container_width=True)

        st.markdown("#### Frame-by-Frame Telemetry Log")
        st.dataframe(df, use_container_width=True, height=220)
