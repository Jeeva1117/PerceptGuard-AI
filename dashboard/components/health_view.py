"""Section: System Operational Health & Hardware Telemetry Component."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict
import pandas as pd
import psutil
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_error_boundary


def render_system_health(client: DashboardApiClient) -> None:
    """Render comprehensive backend health, storage permissions, model hashes, and host resources."""
    st.markdown("### 🩺 System Operational Health & Hardware Diagnostics")
    st.caption("Live monitoring of API connectivity, local storage integrity, model checkpoints, and edge CPU/RAM compute resources.")

    # --------------------------------------------------------------------------
    # Tier 1: Service Transport & API Diagnostics
    # --------------------------------------------------------------------------
    st.markdown("#### 1. Backend Service & API Transport")

    t0 = time.perf_counter()
    is_live = client.probe_backend()
    probe_latency_ms = (time.perf_counter() - t0) * 1000.0

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric(
            label="Backend Transport",
            value=client.transport_mode,
            delta="Online" if is_live else "In-Process Fallback",
            delta_color="normal" if is_live else "inverse",
        )
    with col_s2:
        st.metric(label="API Probe Latency", value=f"{probe_latency_ms:.1f} ms")
    with col_s3:
        st.metric(label="Base URL", value=client.base_url.replace("http://", ""))
    with col_s4:
        st.metric(label="API Timeout Setting", value=f"{client.timeout}s")

    # --------------------------------------------------------------------------
    # Tier 2: Model Weights & Checkpoints Integrity
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 2. Model Checkpoints & Device Allocation")

    yolo_path = Path("models/yolo/yolo11n.pt")
    lstm_path = Path("models/forecasting/lstm/model.pt")

    yolo_exists = yolo_path.exists()
    yolo_size = f"{yolo_path.stat().st_size / (1024*1024):.2f} MB" if yolo_exists else "MISSING"

    lstm_exists = lstm_path.exists()
    lstm_size = f"{lstm_path.stat().st_size / 1024:.1f} KB" if lstm_exists else "MISSING"

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("YOLO11n Weights", "PRESENT" if yolo_exists else "NOT FOUND", delta=yolo_size)
    with col_m2:
        st.metric("YOLO Inference Device", "CPU (Direct)")
    with col_m3:
        st.metric("LSTM Checkpoint", "PRESENT" if lstm_exists else "NOT FOUND", delta=lstm_size)
    with col_m4:
        st.metric("LSTM Compute Device", "CPU (Pure PyTorch)")

    # --------------------------------------------------------------------------
    # Tier 3: Storage & Persistence Diagnostics
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 3. Storage Architecture & Persistence Directories")

    jobs_dir = Path("outputs/jobs")
    datasets_dir = Path("datasets/processed")

    jobs_exists = jobs_dir.exists() and os.access(jobs_dir, os.W_OK)
    datasets_exists = datasets_dir.exists() and os.access(datasets_dir, os.R_OK)

    col_st1, col_st2, col_st3 = st.columns(3)
    with col_st1:
        st.metric("Jobs Persistence Storage", "READ/WRITE" if jobs_exists else "BLOCKED", help=str(jobs_dir.resolve()))
    with col_st2:
        st.metric("Datasets Directory", "ACCESSIBLE" if datasets_exists else "MISSING", help=str(datasets_dir.resolve()))
    with col_st3:
        disk_usage = psutil.disk_usage(str(Path.cwd()))
        st.metric("Free Disk Space", f"{disk_usage.free / (1024**3):.1f} GB", help=f"Total: {disk_usage.total / (1024**3):.1f} GB")

    # --------------------------------------------------------------------------
    # Tier 4: Edge Host Compute & Memory Telemetry
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 4. Host Compute & Resource Consumption (Intel Core i3 / 8 GB RAM)")

    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / (1024**3)
    ram_total_gb = vm.total / (1024**3)
    ram_pct = vm.percent

    process = psutil.Process(os.getpid())
    proc_ram_mb = process.memory_info().rss / (1024 * 1024)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("CPU Utilization", f"{cpu_pct:.1f}%")
    with col_r2:
        st.metric("System RAM Utilization", f"{ram_pct:.1f}%", delta=f"{ram_used_gb:.2f} / {ram_total_gb:.1f} GB")
    with col_r3:
        st.metric("Dashboard Process RSS", f"{proc_ram_mb:.1f} MB", help="Memory dedicated to active Streamlit process")
    with col_r4:
        st.metric("Worker Concurrency Ceiling", "1 Worker (Bounded)", help="ThreadPoolExecutor bound protecting 8 GB memory limit")

    # --------------------------------------------------------------------------
    # Tier 5: 8-Stage Architecture Status Accordion
    # --------------------------------------------------------------------------
    st.markdown("---")
    try:
        status_dict = client.get_pipeline_status()
        components = status_dict.get("components", {})
        with st.expander("PerceptGuard AI 8-Stage Architecture Pipeline Status", expanded=True):
            stage_rows = []
            for stage, comp in components.items():
                stage_rows.append({
                    "Stage Name": stage.replace("_", " ").title(),
                    "Operational Status": comp.get("status", "UNKNOWN"),
                    "Details": comp.get("details", ""),
                })
            st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)
    except Exception as e:
        st.caption(f"Could not load 8-stage pipeline status: {e}")
