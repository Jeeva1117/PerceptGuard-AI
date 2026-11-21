# PerceptGuard AI — Professional Dashboard Architecture & Specification

## 1. Executive Summary

The **PerceptGuard AI Professional Dashboard** is an enterprise-grade operational interface designed for real-time and retrospective monitoring of computer vision reliability in autonomous driving. Built on **Streamlit** and backed by a dual-mode transport layer (**FastAPI REST API** with automatic **In-Process Python Fallback**), the dashboard provides an intuitive, comprehensive, and scientifically grounded cockpit for machine learning engineers, safety certifiers, and operations personnel.

**Tagline:** *"Anticipating Detection Failures Before They Happen."*

---

## 2. Enterprise Navigation Architecture

The dashboard implements an enterprise nine-view navigation system in `dashboard/app.py`:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PerceptGuard AI Dashboard                             │
├─────────────────┬───────────────────────────────────────────────────────────────┤
│ Navigation Menu │ Active Viewport                                               │
├─────────────────┼───────────────────────────────────────────────────────────────┤
│ 1. 📊 Overview  │ Executive KPIs, subsystem health, active job summaries        │
│ 2. 🎥 Video     │ Asynchronous ingestion, progress monitoring, stage tracker    │
│ 3. 🎯 Detection │ YOLO11n + ByteTrack Kalman metrics, frame bbox inspector      │
│ 4. 🔬 Features  │ 27-dimensional multi-modal visual & kinematic distributions   │
│ 5. 🔮 Forecast  │ Multi-horizon (H1, H3, H5) PyTorch LSTM reliability outputs   │
│ 6. ⚠️ Risk      │ Derived failure risk (S = 1.0 - R), operational severity tiers│
│ 7. 🚨 Alerts    │ Early warning alert feed, suppression stats, lifecycle ops    │
│ 8. 📋 History   │ Job audit log, status filtering, one-click active context     │
│ 9. 🩺 Health    │ Subsystem probes, checkpoint integrity, host CPU/RAM metrics   │
└─────────────────┴───────────────────────────────────────────────────────────────┘
```

---

## 3. Core Scientific Data Contracts

### 3.1 Failure Risk Formulation
Reliability forecasting outputs predicted perception reliability $R_{t+h} \in [0.0, 1.0]$ across three temporal horizons:
- **H1 (t+1):** Immediate next frame ($\sim 33\text{ ms}$ forward at 30 fps)
- **H3 (t+3):** Short-horizon prediction ($\sim 100\text{ ms}$ forward)
- **H5 (t+5):** Extended-horizon prediction ($\sim 167\text{ ms}$ forward)

Failure risk $S_{t+h} \in [0.0, 1.0]$ is formally derived as:
$$S_{t+h} = 1.0 - R_{t+h}$$

### 3.2 Operational Severity Thresholds (Provisional Calibration Gate)
The dashboard strictly enforces that missing or uncalculated risk scores are rendered as `UNKNOWN` or `N/A`, and are **never** coerced to `LOW` or `0.000`:

| Risk Tier | Score Range | Operational Meaning | Action Required |
| :--- | :--- | :--- | :--- |
| **LOW** | $[0.00, 0.35)$ | Stable perception environment | Nominal operation |
| **MEDIUM** | $[0.35, 0.65)$ | Emerging visual or kinematic degradation | Elevated telemetry monitoring |
| **HIGH** | $[0.65, 0.85)$ | Impending detection dropout likely | Early warning alert triggered |
| **CRITICAL** | $[0.85, 1.00]$ | Severe perceptual breakdown imminent | Fail-safe / fallback engagement |
| **UNKNOWN** | `None` / `NaN` | Telemetry missing, corrupt, or stale | Operator alert & fallback trigger |

> [!IMPORTANT]
> **Calibration Disclosure:** Operational thresholds are designated as **`PROVISIONAL CALIBRATION`** pending extensive multi-weather (fog, heavy rain, night glare) closed-loop autonomous driving benchmarks.

---

## 4. Dual Transport Architecture

The frontend communicates with backend services via `dashboard/api_client.py`:

```
               ┌───────────────────────────────────────────────┐
               │           Streamlit Dashboard UI              │
               │         (dashboard/components/*.py)           │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │      DashboardApiClient       │
                       │   (dashboard/api_client.py)   │
                       └───────┬───────────────┬───────┘
                               │               │
                     Live REST │               │ In-Process Fallback
                      Backend  │               │ (HTTP Unreachable)
                               ▼               ▼
                 ┌───────────────────┐   ┌───────────────────────────┐
                 │  FastAPI Backend  │   │     DashboardService      │
                 │ (127.0.0.1:8000)  │   │  (In-Process Singleton)   │
                 └─────────┬─────────┘   └─────────────┬─────────────┘
                           │                           │
                           └─────────────┬─────────────┘
                                         ▼
                 ┌───────────────────────────────────────────┐
                 │       PerceptGuard Core ML Pipeline       │
                 │   - YOLO11n Object Detection              │
                 │   - ByteTrack Multi-Object Tracker        │
                 │   - 27-Feature Multi-Modal Extraction     │
                 │   - Multi-Horizon LSTM Forecaster         │
                 │   - Early-Warning Rule Engine             │
                 └───────────────────────────────────────────┘
```

1. **REST API Mode (`REST_API`):** Connects to `http://127.0.0.1:8000` via `httpx.Client`. Automatically detects network disruptions, timeout errors, or server reboots.
2. **In-Process Mode (`IN_PROCESS`):** If the backend server is unreachable or offline, the client seamlessly invokes `src.early_warning.dashboard_service.DashboardService` and `src.api.job_manager.JobManager` in-process with identical method signatures. Zero application crash or blank screen.

---

## 5. Component Breakdown

1. **Executive Overview (`dashboard/components/overview.py`):**
   - High-level KPIs: Pipeline Status, Health Score, Processed Jobs, Unique Tracks.
   - Dual-tier layout: Subsystem connectivity cards + Active Job throughput stats.
2. **Video Ingestion & Processing (`dashboard/components/video_view.py`):**
   - Three operational tabs: Upload Video, Processing Controls, Progress Telemetry.
   - Live stage tracking across 8 pipeline phases: Validating $\rightarrow$ Ingesting $\rightarrow$ Detecting $\rightarrow$ Tracking $\rightarrow$ Features $\rightarrow$ Forecasting $\rightarrow$ Risk $\rightarrow$ Alerts $\rightarrow$ Complete.
3. **Detection & Tracking Context (`dashboard/components/tracking_view.py`):**
   - YOLO11n + ByteTrack Kalman metrics: Processed Frames, Total BBoxes, Unique Tracks, Mean Confidence, Detection Latency.
   - Interactive slider for frame-by-frame bounding box image and track table inspection.
4. **Multi-Modal Feature Analytics (`dashboard/components/feature_analytics.py`):**
   - Visual Quality: Laplacian blur variance, mean brightness, RMS contrast.
   - Kinematics & Motion: Centroid displacement, frame velocities, speed.
   - Detection Properties: Count per frame, mean confidence, confidence variation.
   - Statistical matrix: Min, Mean, Max, and Standard Deviation across all features.
5. **Reliability Forecasting View (`dashboard/components/forecasting_view.py`):**
   - PyTorch LSTM architecture specifications (10-frame input, 27-feature vectors, 3 output horizons).
   - Training convergence metrics ($MSE = 0.00047$, validation loss plateau).
   - Multi-horizon prediction trajectories from official runs and active jobs.
6. **Risk Assessment Cockpit (`dashboard/components/risk_cards.py`):**
   - H1, H3, H5 risk cards with trend direction, rolling mean, volatility, and observation count.
   - Dynamic multi-horizon risk trajectory chart across video frames.
   - Operational severity calibration matrix.
7. **Early Warning Alerts Feed (`dashboard/components/alerts_feed.py`):**
   - Multi-parameter filtering: Severity, Horizon, Alert Type, Lifecycle State.
   - Suppression & deduplication statistics display.
   - Alert acknowledgment and resolution workflow controls.
8. **Processing History & Audit Log (`dashboard/components/history_view.py`):**
   - Historical job table with runtime, FPS, total detections, and status filtering.
   - One-click "Set as Active Dashboard Job" to hydrate the entire dashboard.
   - Collapsible raw `results.json` JSON explorer.
9. **System Health & Hardware Diagnostics (`dashboard/components/health_view.py`):**
   - Live backend probe latency measurement in milliseconds.
   - Checkpoint integrity: file existence and byte sizes for `models/yolo/yolo11n.pt` and `models/forecasting/lstm/model.pt`.
   - Host hardware telemetry: CPU utilization %, RAM usage (GB used / total), process memory footprint via `psutil`.

---

## 6. Hardware Compatibility & Resource Footprint

The dashboard is engineered for high performance on edge and commodity developer hardware:
- **Target Hardware:** Intel Core i3 11th Gen, 8 GB RAM, Windows 11 AMD64.
- **Compute Execution:** 100% pure CPU execution (no CUDA dependencies).
- **RAM Overhead:** Streamlit process runs at $\approx 360\text{ MB}$ baseline, peaking at $\le 530\text{ MB}$ RSS during heavy frame visualization.
- **Latencies:** UI interactions $< 100\text{ ms}$; background video processing runs at $\approx 18.5\text{ FPS}$ on 720p driving footage.
