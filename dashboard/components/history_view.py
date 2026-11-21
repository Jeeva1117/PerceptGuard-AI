"""Section: Video Processing History & Job Audit Log component."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_processing_history(client: DashboardApiClient) -> None:
    """Render historical video processing job records with filtering and deep inspection."""
    st.markdown("### 📋 Video Processing History & Job Records")
    st.caption("Audit log of all video processing jobs submitted to the background pipeline manager.")

    # Filter controls
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        status_filter = st.selectbox(
            "Filter by Status:",
            ["ALL", "COMPLETED", "FAILED", "PROCESSING", "QUEUED", "CANCELLED"],
            index=0,
        )

    try:
        jobs = client.list_video_jobs(limit=50)
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load Job History",
            error_message=f"Backend communication error: {err.message}",
            retry_callback=st.rerun,
            key_suffix="history",
        )
        return

    if status_filter != "ALL":
        jobs = [j for j in jobs if j.get("status") == status_filter]

    if not jobs:
        render_empty_state(
            title="No Matching Processing Jobs Found",
            message=f"No jobs match the selected filter criteria ('{status_filter}').",
            icon="📋",
        )
        return

    # Summary table
    table_rows = []
    for j in jobs:
        created_str = j.get("created_at", "")[:19].replace("T", " ") if j.get("created_at") else "-"
        completed_str = j.get("completed_at", "")[:19].replace("T", " ") if j.get("completed_at") else "-"
        dur = j.get("result_summary", {}).get("total_duration_seconds")
        dur_str = f"{dur:.2f}s" if dur is not None else "-"
        table_rows.append({
            "Job ID": j.get("job_id"),
            "Video Filename": j.get("filename"),
            "Status": j.get("status"),
            "Stage": j.get("stage"),
            "Frames": j.get("frames_processed", 0),
            "Detections": j.get("total_detections", 0),
            "Tracks": j.get("unique_tracks", 0),
            "Alerts": j.get("alerts_count", 0),
            "Duration": dur_str,
            "Submitted At": created_str,
            "Completed At": completed_str,
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Job selection and inspection
    st.markdown("---")
    st.markdown("#### 🔍 Deep Job Artifact Inspection")
    job_ids = [j.get("job_id") for j in jobs]
    selected_job_id = st.selectbox("Select Job to Inspect or Set as Active:", job_ids)

    if selected_job_id:
        selected_job = next((j for j in jobs if j.get("job_id") == selected_job_id), None)
        if selected_job:
            col_b1, col_b2 = st.columns([1, 3])
            with col_b1:
                if st.button("📌 Set as Active Dashboard Job", type="primary", use_container_width=True):
                    st.session_state["active_job_id"] = selected_job_id
                    st.success(f"Job `{selected_job_id}` is now active across all dashboard tabs!")

            with col_b2:
                st.caption(f"Status: **{selected_job.get('status')}** | Activity: {selected_job.get('message', '')}")

            # Failure details if failed
            if selected_job.get("status") == "FAILED":
                err_msg = selected_job.get("error_message") or selected_job.get("message")
                st.error(f"Execution Error Details: {err_msg}")

            # Raw results JSON
            try:
                results_json = client.get_video_job_results(selected_job_id)
                if results_json:
                    with st.expander(f"Structured Artifact: results.json for {selected_job_id}", expanded=False):
                        st.json(results_json)
            except Exception as e:
                st.caption(f"Could not load full results JSON: {e}")
