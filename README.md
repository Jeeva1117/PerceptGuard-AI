# PerceptGuard AI: Predictive Object Detection Reliability System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(Pure%20CPU)-red.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"Anticipating Detection Failures Before They Happen."**

An enterprise-grade, edge-compatible autonomous driving perception reliability monitoring system that **forecasts future object detection failure ($IoU < 0.50$) across upcoming video horizons: H1 ($t+1$), H3 ($t+3$), and H5 ($t+5$).**

---

## 1. Problem Statement

In automated driving systems (ADS) and advanced driver-assistance systems (ADAS), camera-based perception pipelines rely on deep neural object detectors (e.g. YOLO, DETR). Standard systems naively assume the detector's **confidence score** ($P(\text{class} | \text{box})$) reflects real-world perception reliability:

### Confidence $\neq$ Reliability
1. **False Confidence:** Deep detectors frequently output high confidence (>85%) on severely drifted, vibrating, or partially occluded bounding boxes.
2. **Sudden Dropouts:** When encountering lens glare, rapid vehicle roll, motion blur, or environmental degradation, detections drop precipitously with **zero advance warning**.
3. **Reactive Safety Planners:** Downstream motion planners receive detections naively until track divergence occurs, leaving zero time budget for graceful fail-safe handoff or emergency braking.

```
+-----------------------------------------------------------------------------------+
| Standard Perception:     Detector Confidence [0.92] ---> Drop to [0.0] (CRASH RISK)|
| PerceptGuard-AI:         Predicts Failure Risk 100ms In Advance (SAFE HANDOFF)    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Solution & Key Features

PerceptGuard AI introduces an intelligent, predictive meta-perception monitoring layer:

- **Multi-Horizon Reliability Forecasting:** Forecasts future bounding box IoU quality over $H1$ ($t+1, \sim 33\text{ ms}$), $H3$ ($t+3, \sim 100\text{ ms}$), and $H5$ ($t+5, \sim 167\text{ ms}$).
- **27-Dimensional Multi-Modal Feature Fusion:** Real-time extraction of visual quality (Laplacian blur variance, brightness, contrast), kinematics (velocity, acceleration, curvature), and detection dynamics (confidence variance, entropy, regression slopes).
- **Recurrent Sequence Modeling:** Pure PyTorch 2-layer LSTM trained on 10-frame sliding windows ($MSE = 0.00047$).
- **Formal Failure Risk Formulation:** Mathematically grounded risk score:
  $$S_{t+h} = 1.0 - R_{t+h}$$
- **Proactive Early Warning Engine:** Automated incident triggering with exponential cooldown, duplicate suppression, and severity escalation (LOW, MEDIUM, HIGH, CRITICAL).
- **Dual-Transport Operations Dashboard:** 9-view enterprise Streamlit dashboard connected to a FastAPI backend with automated zero-downtime in-process fallback.
- **Pure CPU Edge Execution:** Fully optimized for commodity hardware (Intel Core i3, 8 GB RAM) without CUDA dependencies.

---

## 3. Master Pipeline Architecture

```
[Video Input] (720p / 30fps)
      │
      ▼
[Video Loading & Preprocessing] (OpenCV Standardized 1280x720 BGR)
      │
      ▼
[YOLO11n Object Detection] (Ultralytics Pretrained CPU Weights, 45.5ms)
      │
      ▼
[ByteTrack Multi-Object Tracking] (Kalman Kinematic State Association)
      │
      ▼
[27-Feature Multi-Modal Extraction] (Visual, Motion, Detection Domains)
      │
      ▼
[Temporal Sequence Builder] (10-Frame Sliding Windows, G=25)
      │
      ▼
[Multi-Horizon LSTM Forecaster] (H1=t+1, H3=t+3, H5=t+5 Reliability)
      │
      ▼
[Risk Assessment Engine] (S = 1.0 - R, 4 Operational Severity Tiers)
      │
      ▼
[Early Warning Engine] (Suppression, Deduplication, Alerts)
      │
      ▼
[FastAPI REST Backend] <──────> [Streamlit Operations Cockpit]
      │                                     │
      └──────────────────┬──────────────────┘
                         ▼
             [File-Based Persistence] (results.json & Annotated Frames)
```

For complete technical specifications, consult [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Language & Runtime** | Python 3.10+ / 3.11 / 3.12 / 3.13 (Verified on Python 3.13.3) |
| **Computer Vision** | OpenCV (`opencv-python`), Ultralytics YOLO11 (`ultralytics`) |
| **Object Tracking** | ByteTrack (Kalman Filter + Two-Stage Association) |
| **Deep Learning** | PyTorch (Pure CPU-optimized execution, zero CUDA required) |
| **Backend API** | FastAPI, Uvicorn, Starlette, Pydantic v2 |
| **Frontend UI** | Streamlit, Pandas, NumPy, Matplotlib |
| **Hardware Target** | Intel Core i3 11th Gen, 8 GB RAM, Windows 11 AMD64 (Pure CPU) |
| **Testing & CI/CD** | Pytest, AnyIO, GitHub Actions |

---

## 5. System Requirements & Prerequisites

- **Processor:** Intel Core i3 (11th Gen or equivalent dual-core CPU with AVX2)
- **Memory (RAM):** 8 GB RAM minimum (System consumes $\le 528\text{ MB}$ RSS during peak inference)
- **Disk Storage:** 2 GB free disk space
- **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04+, or macOS 12+
- **GPU:** None required (100% pure CPU execution).

---

## 6. Installation & Quickstart

### Step 1: Clone the Repository
```bash
git clone https://github.com/Jeeva1117/PerceptGuard-AI.git
cd PerceptGuard-AI
```

### Step 2: Create & Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS (Bash):
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Environment & Checkpoints
```bash
python main.py --info
python main.py --verify-env
```

Ensure model weights exist in `models/`:
- `models/yolo/yolo11n.pt` ($5.6\text{ MB}$)
- `models/forecasting/lstm/model.pt` ($103.5\text{ KB}$)

---

## 7. Running the Application

### 7.1 One-Command Startup (Windows 11)
Launch both the FastAPI backend and Streamlit dashboard simultaneously with automated readiness checking:
```powershell
.\scripts\start_project.ps1
```
To gracefully stop all running services and free ports 8000 & 8501:
```powershell
.\scripts\stop_project.ps1
```

### 7.2 Manual Component Startup
You can also run backend and dashboard in dedicated terminal sessions:

**Terminal 1 — FastAPI Backend:**
```powershell
# Using launcher script
.\scripts\start_backend.ps1

# Or direct invocation
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
- API Documentation: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health/live`

**Terminal 2 — Streamlit Operations Dashboard:**
```powershell
# Using launcher script
.\scripts\start_dashboard.ps1

# Or direct invocation
streamlit run dashboard/app.py --server.port 8501
```
The dashboard opens automatically at `http://localhost:8501`.

> [!NOTE]
> If the backend server is not running, the dashboard automatically activates its **In-Process Fallback Mode**. All capabilities remain fully operational within the single process.

---

## 8. Complete End-to-End Workflow

1. Open `http://localhost:8501` and navigate to **`🎥 Video Processing`**.
2. Upload a sample driving video (e.g. `datasets/raw/converted_026c7465-309f6d33.mp4` or a clip from `outputs/day14_validation/clips/`).
3. Set **Max Frames** to `30` and click **"Submit & Run Full Pipeline"**.
4. Observe the live 8-stage progress tracker until status reaches **COMPLETED**.
5. Switch to **`📊 Overview`** to review high-level throughput and detection metrics.
6. Switch to **`🎯 Detection & Tracking`** to scrub through annotated frames with persistent Kalman track IDs.
7. Switch to **`🔬 Feature Analytics`** to inspect frame-by-frame Laplacian blur, brightness, contrast, and kinematics.
8. Switch to **`⚠️ Risk Assessment`** to inspect failure risk $S = 1.0 - R$ across $H1, H3, H5$ horizons.
9. Switch to **`🚨 Early Warning Alerts`** to review and resolve any incidents triggered ahead of failure dropouts.

For detailed view instructions, consult [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

---

## 9. API Information

The FastAPI backend exposes structured endpoints:
- `GET /health` — Service health, uptime, and model availability.
- `GET /api/v1/telemetry/risk-overview` — Current H1, H3, H5 risk scores and rolling statistics.
- `GET /api/v1/telemetry/temporal-monitoring` — Historical risk trajectory time series.
- `GET /api/v1/telemetry/detection-tracking` — YOLO11n + ByteTrack tracking telemetry.
- `GET /api/v1/alerts` — Paginated early warning alert incidents.
- `POST /api/v1/video/jobs` — Asynchronous video job submission.
- `GET /api/v1/video/jobs/{job_id}/results` — Complete structured RFC 8259-compliant results.

Consult [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for full endpoint specifications.

---

## 10. Automated Testing & Verification

Execute the complete pytest suite across all subsystems:

```bash
pytest -v
```

- **Frontend Tests:** 41 / 41 passed (**100%**)
- **API Tests:** 32 / 32 passed (**100%**)
- **Full Test Suite:** 1,624 / 1,624 passed (**100%**)

---

## 11. Performance Benchmarks (Intel Core i3, 8 GB RAM)

- **Pure CPU Execution:** Zero CUDA dependencies.
- **Baseline Memory:** $362\text{ MB}$ RSS.
- **Peak Video Inference Memory:** $528\text{ MB}$ RSS ($\le 6.6\%$ of system RAM).
- **Video Processing Throughput:** $\approx 18.5\text{ FPS}$ on 720p 30fps driving video.
- **YOLO11n CPU Inference:** $45.5\text{ ms/frame}$.
- **LSTM Multi-Horizon Inference:** $1.8\text{ ms/sequence}$.
- **API Probe Latency:** $2.4\text{ ms}$ on local loopback.
- **UI Transition Latency:** $< 120\text{ ms}$.

For detailed benchmarking evidence, consult [docs/MODEL_AND_ML_VALIDATION.md](docs/MODEL_AND_ML_VALIDATION.md).

---

## 12. Known Limitations & Future Work

- **Provisional Calibration Gate:** Failure risk thresholds ($[0.35, 0.65, 0.85]$) are designated as **`PROVISIONAL CALIBRATION`** pending multi-weather road testing (fog, blizzard, heavy rain).
- **Sequential Video Queue:** Background video jobs execute sequentially via `ThreadPoolExecutor(max_workers=1)` to prevent edge CPU core thrashing.
- **Future Improvements:**
  - OpenVINO / ONNX Runtime quantization for 30+ FPS edge throughput.
  - Multi-camera 360-degree surround sensor fusion.
  - Closed-loop CAN-bus integration for automated fail-safe vehicle handoff.

Consult [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for full operational boundaries.

---

## 13. Documentation Suite

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Master system architecture and pipeline design.
- [docs/INSTALLATION.md](docs/INSTALLATION.md) — Step-by-step setup and environment configuration.
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Operator manual for all 9 dashboard views.
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — FastAPI endpoints, schemas, and data contracts.
- [docs/MODEL_AND_ML_VALIDATION.md](docs/MODEL_AND_ML_VALIDATION.md) — Machine learning validation and benchmarks.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Docker containerization and orchestration guide.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — Operational boundaries and calibration constraints.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Open source contribution guidelines.
- [LICENSE](LICENSE) — MIT License.

---

## 14. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
