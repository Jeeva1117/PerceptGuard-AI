# PerceptGuard AI — Operator & User Guide

## 1. Introduction

The **PerceptGuard AI Dashboard** is an enterprise operations console designed for machine learning engineers, perception QA specialists, and autonomous safety auditors. The dashboard provides a unified cockpit to monitor computer vision health, predict future bounding box failures, analyze visual/kinematic degradation, and manage early warning incidents.

---

## 2. Navigation Overview

The dashboard sidebar features a 9-view navigation menu:

```
⚙️ Controls & Navigation
├── 📊 Overview
├── 🎥 Video Processing
├── 🎯 Detection & Tracking
├── 🔬 Feature Analytics
├── 🔮 Reliability Forecasting
├── ⚠️ Risk Assessment
├── 🚨 Early Warning Alerts
├── 📋 Processing History
└── 🩺 System Health
```

The sidebar also displays the active transport mode (`REST_API` vs `IN_PROCESS`), backend URL, telemetry freshness TTL, and quick-refresh controls.

---

## 3. Detailed View Specifications

### 3.1 📊 Overview
The executive dashboard provides immediate situational awareness across three tiers:
- **Pipeline KPI Metrics:** Overall Health (`OPTIMAL`, `DEGRADED`, `CRITICAL`), Pipeline Status (`OPERATIONAL`, `STANDBY`), Health Score (0.0 to 1.0), and Telemetry Freshness.
- **Active Job Throughput:** Displays the currently selected video sequence name, total processed frames, total detections, unique Kalman tracks, and execution FPS.
- **Subsystem Status Cards:** Direct indicators of API Server, YOLO Detector, LSTM Forecaster, and Early Warning engine operational states.

### 3.2 🎥 Video Processing
The ingestion and processing cockpit allows users to submit driving clips for automated analysis:
1. **Upload Video:** Drag-and-drop or select an `.mp4`, `.avi`, or `.mov` driving video.
2. **Configure Run:** Select maximum frames to process (recommended: 30 to 100 for CPU execution).
3. **Submit Job:** Click **"Submit & Run Full Pipeline"**.
4. **Live Execution Progress:** Watch real-time execution across the 8 sequential pipeline stages:
   $$\text{Validating} \rightarrow \text{Ingesting} \rightarrow \text{Detecting} \rightarrow \text{Tracking} \rightarrow \text{Features} \rightarrow \text{Forecasting} \rightarrow \text{Risk} \rightarrow \text{Complete}$$

### 3.3 🎯 Detection & Tracking
Inspects authentic YOLO11n bounding boxes and ByteTrack Kalman identities:
- **Empirical Continuity Metrics:** Total frames, total bounding boxes, unique track IDs, max simultaneous active objects, mean detection confidence %, and detection latency (ms/frame).
- **Detected Object Class Distribution:** Bar chart showing frequency of detected objects (`car`, `pedestrian`, `traffic light`, `truck`).
- **Confidence Distribution:** Line chart showing model confidence score curves.
- **Frame-by-Frame Inspector:** Scrub through frames using the slider to view high-resolution images with overlaid bounding boxes and persistent tracking IDs.

### 3.4 🔬 Feature Analytics
Deep-dives into the 27-dimensional multi-modal feature vector extracted per frame:
- **Visual Quality Tab:** Laplacian blur variance ($\sigma^2_{\text{Laplacian}}$), mean image brightness, and RMS contrast. Sharp drops signal environmental degradation (e.g. defocussing or fog).
- **Kinematics & Motion Tab:** Centroid displacement ($\Delta x, \Delta y$) and velocity curves tracking object motion.
- **Detection Properties Tab:** Detected object count and confidence variation per frame.
- **Statistical Summary Matrix:** Comprehensive table showing Min, Mean, Max, and Standard Deviation across all 27 feature channels.

### 3.5 🔮 Reliability Forecasting
Presents future object detection reliability predicted by the PyTorch LSTM forecaster:
- **Architecture Specifications:** 10-frame input sequence length, 27 input features, 3 forecasting horizons ($H1, H3, H5$).
- **Training Convergence:** Verified training MSE ($0.00047$) and loss plateau.
- **Multi-Horizon Trajectories:** Interactive plots comparing predicted reliability across:
  - **H1 ($t+1$):** Immediate next frame ($\sim 33\text{ ms}$ forward)
  - **H3 ($t+3$):** Short-term horizon ($\sim 100\text{ ms}$ forward)
  - **H5 ($t+5$):** Extended horizon ($\sim 167\text{ ms}$ forward)

### 3.6 ⚠️ Risk Assessment
Translates reliability forecasts into actionable operational failure risk:
- **Formula:** $S_{t+h} = 1.0 - R_{t+h}$.
- **Risk Cards:** Color-coded severity cards (Green = LOW, Yellow = MEDIUM, Orange = HIGH, Red = CRITICAL) for each horizon with rolling mean, trend direction, and volatility.
- **Temporal Risk Trajectory:** Multi-line plot tracking failure probability over the video timeline with operational threshold markers at $0.35$, $0.65$, and $0.85$.
- **Severity Calibration Matrix:** Reference table mapping failure risk to mandatory automated vehicle actions.

### 3.7 🚨 Early Warning Alerts
Incident management console for alerts triggered ahead of perceptual failures:
- **Multi-Parameter Filtering:** Filter incidents by Severity (`CRITICAL`, `WARNING`, `CAUTION`, `INFO`), Horizon (`H1`, `H3`, `H5`), Alert Type, and Status (`ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`).
- **Suppression Metrics:** Displays cooldown active states, deduplicated counts, and suppression rates.
- **Lifecycle Actions:** Operators can click **"Acknowledge"** or **"Resolve"** on active incidents with mandatory audit reasons.

### 3.8 📋 Processing History
Auditable log of all video processing jobs executed on the machine:
- **Job Table:** Displays Job ID, Filename, Status, Processed Frames, FPS, and Created Timestamp.
- **One-Click Active Context:** Click **"Set as Active Dashboard Job"** next to any completed job to immediately populate the entire dashboard with its bounding boxes, features, risk scores, and alerts.
- **Raw JSON Explorer:** Expandable viewer for inspecting the raw RFC 8259-compliant `results.json` artifact.

### 3.9 🩺 System Health
Hardware and backend diagnostic dashboard:
- **API Probe Latency:** Live HTTP round-trip latency in milliseconds.
- **Model Checkpoints:** Validates disk presence and file sizes for `yolo11n.pt` and `lstm/model.pt`.
- **Storage Permissions:** Confirms read and write access to `outputs/jobs/` and `datasets/`.
- **Host Compute Telemetry:** Real-time host CPU utilization %, RAM usage (GB used / total), and process memory consumption via `psutil`.

---

## 4. Operational Best Practices

1. **Verify Checkpoints on Startup:** Always verify `🩺 System Health` upon opening the dashboard to ensure model weights are loaded and API latency is $< 10\text{ ms}$.
2. **Reviewing Alerts:** When an alert triggers in `🚨 Early Warning Alerts`, switch to `🔬 Feature Analytics` to identify the root cause (e.g. blur spike or brightness drop) and verify impending risk in `⚠️ Risk Assessment`.
3. **Testing Video Clips:** For rapid validation, use short driving clips ($15$ to $50$ frames) from `datasets/raw/` or `outputs/day14_validation/clips/`.
