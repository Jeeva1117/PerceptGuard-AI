# PerceptGuard AI — System Architecture & Technical Specification

## 1. Executive Summary

**PerceptGuard AI** is a predictive object detection reliability system engineered for autonomous driving perception pipelines. Operating on edge CPU hardware without requiring GPU acceleration or proprietary cloud infrastructure, PerceptGuard continuously monitors visual quality, bounding box kinematics, and detection continuity to **anticipate object detection failures ($IoU < 0.5$) before they occur.**

**Project Tagline:** *"Anticipating Detection Failures Before They Happen."*

---

## 2. Master End-to-End Pipeline Architecture

The complete system comprises 14 sequentially integrated perception and forecasting stages:

```
[1. Video Input]
       │
       ▼
[2. Video Loading & Validation] (OpenCV 720p / 30fps)
       │
       ▼
[3. Frame Extraction & Preprocessing] (Standardized 1280x720 BGR)
       │
       ▼
[4. YOLO11n Object Detection] (Pretrained Ultralytics CPU Inference)
       │
       ▼
[5. ByteTrack Multi-Object Tracking] (Kalman Kinematic State Estimation)
       │
       ▼
[6. 27-Feature Multi-Modal Extraction] (Visual, Motion, Detection Domains)
       │
       ▼
[7. Ground-Truth Reliability Labels] (IoU Matching against BDD100K)
       │
       ▼
[8. Temporal Sequence Builder] (10-Frame Sliding Windows, G=25)
       │
       ▼
[9. Multi-Horizon LSTM Forecaster] (H1=t+1, H3=t+3, H5=t+5 Prediction)
       │
       ▼
[10. Reliability Risk Scorer] (Formal S = 1.0 - R Formulation)
       │
       ▼
[11. Early Warning Rule Engine] (Suppression, Deduplication, Alerts)
       │
       ▼
[12. FastAPI Backend Server] (REST API, Asynchronous Job Queue)
       │
       ▼
[13. Streamlit Operations Dashboard] (9 Enterprise Cockpit Views)
       │
       ▼
[14. File-Based Result Persistence] (JSON Telemetry & Annotated Frames)
```

---

## 3. Subsystem Breakdown & Module Design

### 3.1 Video Ingestion & Quality QA (`src/video_processing/`)
- **Video Loader (`video_loader.py`):** Decodes container formats (`.mp4`, `.avi`, `.mov`) via OpenCV `cv2.VideoCapture`. Validates frame rates, total frame counts, and aspect ratios.
- **Frame Extractor (`frame_extractor.py`):** Extracts frames sequentially into memory buffers with bounded array deallocation to maintain RAM footprint under 500 MB.
- **Frame Preprocessor (`frame_preprocessor.py`):** Normalizes dimensions to $1280 \times 720$ BGR, preserving color spaces without artificial saturation or clipping.
- **Pipeline Runner (`video_pipeline_runner.py`):** Coordinates batch processing, HUD visual annotations, and atomic persistence of execution artifacts.

### 3.2 Detection & Tracking (`src/detection/` & `src/tracking/`)
- **YOLO11n Detector (`batch_detector.py`):** Lightweight neural object detector initialized from `models/yolo/yolo11n.pt` ($5.6\text{ MB}$). Extracts bounding box coordinates $[x_1, y_1, x_2, y_2]$, classification confidence scores, and COCO class IDs (`car`, `pedestrian`, `traffic light`, `truck`).
- **ByteTrack Tracker (`bytetrack_engine.py`):** Two-stage Kalman filter data association algorithm that tracks detections across frames, maintaining persistent object IDs even through brief occlusions.

### 3.3 Multi-Modal Feature Engineering (`src/features/`)
Extracts a calibrated 27-dimensional feature space per frame across three domains:
1. **Visual Quality (6 dims):** Laplacian blur variance ($\sigma^2_{\text{Laplacian}}$), mean image brightness, RMS contrast, and color channel distributions.
2. **Kinematics & Motion (10 dims):** Centroid coordinates, instantaneous velocities ($\Delta x, \Delta y$), acceleration, trajectory curvature, and Kalman state innovations.
3. **Detection Properties (11 dims):** Detection count, class entropy, mean confidence, minimum confidence, confidence variance, and rolling linear regression slopes.

### 3.4 Multi-Horizon Reliability Forecasting (`src/forecasting/`)
- **Model Architecture:** Pure PyTorch 2-layer LSTM (`LSTMForecaster`) with hidden dimension 64, dropout 0.2, and linear projection heads for multi-horizon forecasting.
- **Input Specifications:** 10-frame sliding history of 27-feature vectors ($[B, 10, 27]$).
- **Target Horizons:**
  - **H1 ($t+1$):** Immediate next frame ($\sim 33\text{ ms}$ forward at 30 fps).
  - **H3 ($t+3$):** Short-term horizon ($\sim 100\text{ ms}$ forward).
  - **H5 ($t+5$):** Extended horizon ($\sim 167\text{ ms}$ forward).
- **Output:** Predicted reliability $R_{t+h} \in [0.0, 1.0]$ representing expected bounding box IoU quality against ground truth.

### 3.5 Risk Scoring & Early Warning Engine (`src/risk/` & `src/early_warning/`)
- **Mathematical Derivation:** Failure risk $S_{t+h} \in [0.0, 1.0]$ is strictly defined as:
  $$S_{t+h} = 1.0 - R_{t+h}$$
- **Severity Classification:**
  - **LOW:** $[0.00, 0.35)$ — Nominal operation
  - **MEDIUM:** $[0.35, 0.65)$ — Elevated monitoring required
  - **HIGH:** $[0.65, 0.85)$ — Proactive alert triggered
  - **CRITICAL:** $[0.85, 1.00]$ — Automated fail-safe engagement
  - **UNKNOWN:** Missing, stale, or corrupt telemetry; never coerced to `LOW`.
- **Early Warning Engine (`early_warning_engine.py`):** Rule-based incident generator featuring exponential cooldown, duplicate suppression, and severity escalation.

---

## 4. Dual-Transport Frontend/Backend Architecture

```
                 ┌─────────────────────────────────────────────────────────┐
                 │                Streamlit Operations UI                  │
                 │              (dashboard/components/*.py)                │
                 └────────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │       DashboardApiClient        │
                             │    (dashboard/api_client.py)    │
                             └────────┬───────────────┬────────┘
                                      │               │
                         REST Mode    │               │ In-Process Fallback
                         (HTTP/JSON)  │               │ (Server Offline)
                                      ▼               ▼
                        ┌───────────────────┐   ┌───────────────────────────┐
                        │  FastAPI Backend  │   │     DashboardService      │
                        │ (127.0.0.1:8000)  │   │   (In-Process Engine)     │
                        └─────────┬─────────┘   └─────────────┬─────────────┘
                                  │                           │
                                  └─────────────┬─────────────┘
                                                ▼
                        ┌───────────────────────────────────────────┐
                        │      Core ML & Video Pipeline Runner      │
                        │        - YOLO11n / ByteTrack              │
                        │        - 27-Feature Extractor             │
                        │        - Multi-Horizon LSTM Forecaster    │
                        │        - Early Warning Engine             │
                        └───────────────────────────────────────────┘
```

1. **REST API Transport (`REST_API`):** Connects to `http://127.0.0.1:8000/api/v1` via `httpx.Client`. Employs connection pooling, timeouts, and structured error propagation.
2. **In-Process Fallback (`IN_PROCESS`):** If the HTTP backend is unreachable or offline, the client automatically executes methods against the in-process `DashboardService` singleton. Zero blank screens, zero application crashes.

---

## 5. Enterprise Dashboard View Layout

The Streamlit interface (`dashboard/app.py`) provides 9 dedicated operational views:
1. `📊 Overview`: Executive operational KPIs, active job telemetry, and subsystem health badges.
2. `🎥 Video Processing`: Drag-and-drop video upload, frame limits, and live 8-stage progress tracker.
3. `🎯 Detection & Tracking`: Bounding box statistics, class breakdown, confidence curves, and frame slider.
4. `🔬 Feature Analytics`: Visual quality, kinematics, and detection telemetry across all 27 features.
5. `🔮 Reliability Forecasting`: LSTM model specifications, verified MSE ($0.00047$), multi-horizon trajectories.
6. `⚠️ Risk Assessment`: H1, H3, H5 risk cards, dynamic risk trajectory plot, calibration matrix.
7. `🚨 Early Warning Alerts`: Incident feed with severity/horizon filters, deduplication stats, and lifecycle actions.
8. `📋 Processing History`: Job audit log with duration, FPS, and one-click active context loading.
9. `🩺 System Health`: API probe latency, model weight verification, storage checks, and host CPU/RAM metrics.

---

## 6. Hardware & Performance Budget

- **Target Hardware:** Intel Core i3 11th Gen, 8 GB RAM, Windows 11 AMD64 (Pure CPU Execution).
- **RAM Overhead:** Baseline process memory is $\sim 362\text{ MB}$; peak video inference is $\le 528\text{ MB}$ RSS.
- **Throughput:** $\approx 18.5\text{ FPS}$ on 720p 30fps video under pure CPU execution.
- **Inference Latencies:**
  - YOLO11n detection: $45.5\text{ ms/frame}$
  - ByteTrack tracking: $4.2\text{ ms/frame}$
  - Feature extraction: $6.1\text{ ms/frame}$
  - LSTM multi-horizon forecast: $1.8\text{ ms/sequence}$
