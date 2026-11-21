# PerceptGuard AI — Dashboard Operator Guide

## 1. Quickstart & Launching the Dashboard

### 1.1 Prerequisites
- Python 3.10+ installed with virtual environment activated (`.\venv\Scripts\activate`).
- Model checkpoints present in `models/yolo/yolo11n.pt` and `models/forecasting/lstm/model.pt`.
- Minimum 8 GB system RAM.

### 1.2 Starting the Backend Server (Recommended)
In a dedicated PowerShell terminal:
```powershell
.\venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000
```
Verify the backend is live at `http://127.0.0.1:8000/docs`.

### 1.3 Starting the Streamlit Dashboard
In a second PowerShell terminal:
```powershell
.\venv\Scripts\streamlit.exe run dashboard/app.py
```
The dashboard opens automatically at `http://localhost:8501`.

> [!NOTE]
> If the backend server is not running, the dashboard automatically initializes its **In-Process Fallback Mode**. All visualization, feature inspection, and forecasting capabilities remain fully operational.

---

## 2. Navigating the Enterprise Views

Use the sidebar radio selector **"Navigation Menu"** to switch between the 9 operational views:

### 2.1 📊 Overview
- **Executive Telemetry:** View overall perception pipeline status, health scores, and data freshness.
- **Active Job Summary:** Shows total frames processed, detection count, unique tracks, and processing FPS for the currently selected job.
- **Subsystem Breakdown:** Instant indicator of API Server, YOLO Detector, LSTM Forecaster, and Early Warning engine status.

### 2.2 🎥 Video Processing
- **Upload Video:** Drag-and-drop or browse for `.mp4`, `.avi`, or `.mov` driving video files (e.g. BDD100K test clips).
- **Processing Configuration:** Select maximum frames to process (default: 30, max recommended on CPU: 100).
- **Asynchronous Execution:** Click **"Submit & Run Full Pipeline"** to begin background ingestion.
- **Progress Tracking:** Monitor the 8 sequential execution stages with auto-updating progress bar and stage badges.

### 2.3 🎯 Detection & Tracking
- **Empirical Continuity Metrics:** Inspect total bboxes, unique tracks, max simultaneous tracks, mean detection confidence, and detection latency (ms/frame).
- **Detected Class Distribution:** Bar chart showing frequency of detected objects (`car`, `pedestrian`, `traffic light`, `truck`).
- **Confidence Distribution:** Line chart showing model confidence score curves.
- **Frame-by-Frame Inspector:** Use the interactive frame slider to view individual frames, bounding box coordinates, class labels, and Kalman track IDs.

### 2.4 🔬 Feature Analytics
- **Visual Quality Tab:** Frame-by-frame time series for Laplacian blur variance, mean brightness, and RMS contrast.
- **Kinematics & Motion Tab:** Centroid displacement and velocity curves showing object dynamics.
- **Detection Properties Tab:** Detected object count and confidence variation per frame.
- **Feature Statistics Matrix:** Summary table displaying Min, Mean, Max, and Standard Deviation across all 27 feature channels.

### 2.5 🔮 Reliability Forecasting
- **LSTM Specifications:** Architecture parameters (10-frame sliding window, 27 input features, 3 output horizons).
- **Training Convergence:** Review verified training metrics ($MSE = 0.00047$) and loss curves.
- **Multi-Horizon Trajectories:** Line charts for H1 ($t+1$), H3 ($t+3$), and H5 ($t+5$) reliability predictions.
- **Scientific Disclosure:** Explicit notification that reliability $R \in [0.0, 1.0]$ represents anticipated future detection quality, distinct from instantaneous detection confidence.

### 2.6 ⚠️ Risk Assessment
- **Multi-Horizon Risk Cards:** Visual color-coded severity cards (Green = LOW, Yellow = MEDIUM, Orange = HIGH, Red = CRITICAL) for H1, H3, and H5 horizons.
- **Failure Risk Trajectory:** Plot of failure risk $S = 1.0 - R$ over time across the sequence.
- **Operational Severity Calibration:** Reference table detailing risk score boundaries and mandatory automated actions.

### 2.7 🚨 Early Warning Alerts
- **Alert Feed:** Real-time log of proactive alerts triggered prior to perceptual failure.
- **Filters:** Filter incidents by Severity (`CRITICAL`, `WARNING`, `CAUTION`, `INFO`), Horizon (`H1`, `H3`, `H5`), Alert Type, and Status (`ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`).
- **Suppression Statistics:** Live metrics for alert cooldowns, deduplication counts, and rate limiting.
- **Lifecycle Management:** Buttons to Acknowledge or Resolve incidents with audit justifications.

### 2.8 📋 Processing History
- **Historical Audit Log:** Comprehensive table of all processed video jobs with timestamps, FPS, durations, and statuses.
- **Filter by Status:** Filter jobs by `COMPLETED`, `FAILED`, `PROCESSING`, or `QUEUED`.
- **Set as Active Dashboard Job:** Click the button next to any completed job to immediately load its bounding boxes, features, risk scores, and alerts across all dashboard views.
- **Raw Results Explorer:** Expandable JSON viewer for auditing raw `results.json` artifacts.

### 2.9 🩺 System Health
- **API Transport Diagnostic:** Displays live probe latency in milliseconds and current transport mode (`REST_API` vs `IN_PROCESS`).
- **Model Checkpoints:** Checks presence and disk size of `yolo11n.pt` and `lstm/model.pt`.
- **Storage Health:** Validates read/write access to `outputs/jobs/` and `datasets/processed/`.
- **Host Compute Resources:** Live telemetry of CPU utilization %, available RAM, and current process memory consumption.

---

## 3. Recommended Workflow for Validating a Video

1. Open **`🎥 Video Processing`** in the sidebar.
2. Upload a sample driving video (e.g. `data/videos/converted_026c7465-309f6d33.mp4`).
3. Set frame limit to `30` and click **Submit & Run Full Pipeline**.
4. Wait for processing to reach **100% (COMPLETED)**.
5. Navigate to **`📊 Overview`** to inspect the executive processing metrics.
6. Check **`🎯 Detection & Tracking`** and slide through frames to inspect bounding boxes.
7. Switch to **`🔬 Feature Analytics`** to inspect visual degradation (e.g. blur or brightness drops).
8. View **`⚠️ Risk Assessment`** to see how visual degradation influenced failure risk across H1, H3, and H5.
9. Review **`🚨 Early Warning Alerts`** to verify that alerts fired ahead of detection dropouts.
