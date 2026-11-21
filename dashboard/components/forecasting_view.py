"""Section: Predictive Reliability Forecasting Component (H1, H3, H5).

Visualizes multi-horizon reliability forecasts from the PyTorch LSTM model,
horizon alignment definitions, evaluation metrics from official benchmark runs,
and strict scientific distinction between predicted reliability and ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_reliability_forecasting(client: DashboardApiClient) -> None:
    """Render multi-horizon reliability forecasting telemetry and evaluation metrics."""
    st.markdown("### 🔮 Multi-Horizon Reliability Forecasting (H1, H3, H5)")
    st.caption(
        "Anticipating object detection failure before it occurs. The PyTorch LSTM forecaster evaluates "
        "27-dimensional feature sequences to predict bounding-box reliability across future horizons: "
        "H1 (t+1), H3 (t+3), and H5 (t+5)."
    )

    # Scientific Integrity Notice
    st.info(
        r"🔬 **Scientific Data Contract Disclosure:** Predicted reliability $R \in [0.0, 1.0]$ represents "
        r"the model's forecast of future object detection quality. It is strictly distinguished from detection "
        r"confidence ($P(\text{class}|\text{box})$). Ground-truth reliability requires IoU-based matching against "
        r"human ground-truth annotations ($R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$)."
    )

    # Model Architecture & Deployment Card
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Forecasting Architecture", "LSTM (PyTorch CPU)")
    with col_m2:
        st.metric("Input Sequence Length", "10 Frames")
    with col_m3:
        st.metric("Feature Dimensions", "27 Features")
    with col_m4:
        st.metric("Forecast Horizons", "H1, H3, H5")

    # Load active job data or fallback
    active_job_id = st.session_state.get("active_job_id")
    job_results = None
    if active_job_id:
        try:
            job_results = client.get_video_job_results(active_job_id)
        except Exception:
            job_results = None

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

    # --------------------------------------------------------------------------
    # Forecasting Trajectory Time-Series
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Predicted Multi-Horizon Reliability Trajectories")

    if frames:
        st.caption(f"Forecasting trajectories from Job: `{active_job_id}`.")
        chart_rows = []
        for f in frames:
            rel = f.get("predicted_reliability", {})
            ts = f.get("timestamp_seconds", 0.0)
            chart_rows.append({
                "Timestamp (s)": ts,
                "H1 (t+1) Reliability": rel.get("h1", 0.0),
                "H3 (t+3) Reliability": rel.get("h3", 0.0),
                "H5 (t+5) Reliability": rel.get("h5", 0.0),
            })

        df_rel = pd.DataFrame(chart_rows).set_index("Timestamp (s)")
        st.line_chart(df_rel, use_container_width=True)

        # Horizon Summary Scorecards
        per_h = job_results.get("per_horizon_risk", {})
        h_col1, h_col2, h_col3 = st.columns(3)
        for col, h_name, h_key in [(h_col1, "Horizon 1 (t+1)", "h1"), (h_col2, "Horizon 3 (t+3)", "h3"), (h_col3, "Horizon 5 (t+5)", "h5")]:
            with col:
                h_info = per_h.get(h_key, {})
                latest_r = 1.0 - h_info.get("latest_risk", 0.5)
                st.markdown(
                    f"""
                    <div style="background: #181926; border: 1px solid #313244; border-radius: 8px; padding: 12px; text-align: center;">
                        <div style="font-size: 12px; color: #a6adc8;">{h_name}</div>
                        <div style="font-size: 24px; font-weight: bold; color: #52b788; margin: 4px 0;">
                            {latest_r:.3f}
                        </div>
                        <div style="font-size: 11px; color: #89b4fa;">Predicted Reliability</div>
                        <div style="font-size: 11px; color: #6c7086; margin-top: 4px;">
                            Corresponding Failure Risk: <strong>{h_info.get('latest_risk', 0.0):.3f}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        render_empty_state(
            title="No Active Job Forecasting Trajectories",
            message="Upload a video sequence to inspect real-time multi-horizon forecast trajectories.",
            icon="🔮",
        )

    # --------------------------------------------------------------------------
    # Official Model Evaluation Benchmark Metrics
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Official Model Training & Evaluation Benchmark")
    st.caption("Empirical training convergence metrics from the official multi-horizon LSTM evaluation benchmark.")

    summary_path = Path("experiments/forecasting/lstm_training/official_lstm_training/training_summary.json")
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                tsum = json.load(f)
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Benchmark Run ID", tsum.get("run_id", "N/A"))
            b2.metric("Validation MSE Loss", f"{tsum.get('best_val_loss', 0.0):.6f}")
            b3.metric("Training Epochs", tsum.get("total_epochs", 0))
            b4.metric("Convergence Status", tsum.get("status", "N/A"))
        except Exception as e:
            st.caption(f"Could not load official training summary: {e}")
    else:
        st.caption("Official model training summary file not located.")

    # --------------------------------------------------------------------------
    # Ingested Multi-Horizon Dataset Predictions Explorer
    # --------------------------------------------------------------------------
    preds_file = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")
    if preds_file.exists():
        with st.expander("📚 Ingested Dataset Multi-Horizon Predictions Explorer (414 Official Records)", expanded=False):
            try:
                with open(preds_file, "r", encoding="utf-8") as f:
                    preds_list = json.load(f)
                if isinstance(preds_list, list) and preds_list:
                    p_df = pd.DataFrame(preds_list)[
                        ["sequence_id", "horizon", "target_timestamp_seconds", "predicted_reliability", "inference_time_ms", "prediction_status"]
                    ]
                    p_df.columns = ["Sequence ID", "Horizon", "Target Time (s)", "Predicted Reliability", "Inference Latency (ms)", "Status"]
                    st.dataframe(p_df, use_container_width=True, height=220)
            except Exception as e:
                st.caption(f"Could not load predictions dataset: {e}")
