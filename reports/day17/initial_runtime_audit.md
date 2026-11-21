# PerceptGuard AI — Day 17 Initial Runtime Audit Report

## 1. Executive Summary & Audit Baseline

- **Milestone:** Day 17 — Streamlit Application Integration, Repository Cleanup & Full Runtime Stabilization
- **Audit Date:** 2026-11-21
- **Lead Roles:** Principal Software Engineer, Senior Staff Engineer, Streamlit Specialist, FastAPI Architect, CV/ML Engineer, DevOps & QA Lead
- **Target Hardware:** Intel Core i3 11th Gen (2 Cores / 4 Threads), 8 GB RAM, Windows 11 AMD64 (Pure CPU Execution)
- **Previous Milestone Status (Day 16):** `PASS WITH LIMITATIONS` (Calibration Status: `PROVISIONAL CALIBRATION`)
- **Initial Working Tree:** Clean on branch `main` (`c916831`)

---

## 2. Existing Architecture

PerceptGuard AI operates a 14-stage end-to-end perception reliability forecasting pipeline:
1. **Video Ingestion:** Accepts forward-facing automotive camera videos (MP4, AVI, MOV, MKV up to 100 MB).
2. **Video Loading:** OpenCV `VideoCapture` extracts frames with bounded horizons ($5 \le \text{frames} \le 50$) to respect CPU budgets.
3. **Frame Preprocessing:** Normalization and resizing to $1280 \times 720$.
4. **YOLO Detection:** YOLO11 nano (`models/detection/yolo11n.pt`) detects vehicle and pedestrian classes on CPU.
5. **ByteTrack Tracking:** Two-stage Kalman filter data association preserves object trajectory identities across frames.
6. **27-Feature Multi-Modal Extraction:** Extracts bounding box kinematics, visual degradation (Laplacian blur, brightness, contrast), and detection dynamics per tracked object.
7. **Ground-Truth IoU Reliability:** Evaluates spatial localization accuracy against physical/synthetic ground truth.
8. **Temporal Sequence Preparation:** Assembles 10-frame sliding window sequences ($W=10, D=27$).
9. **LSTM Multi-Horizon Forecasting:** PyTorch LSTM (`models/forecasting/model.pt`) predicts future reliability across horizons $t+1$ (H1, 33ms), $t+3$ (H3, 100ms), and $t+5$ (H5, 166ms).
10. **Reliability Risk Scoring:** Maps predicted future reliability $R_{t+h}$ to failure risk $S_{t+h} = 1.0 - R_{t+h}$.
11. **Early Warning Rule Engine:** Evaluates persistence and thresholds ($[0.35, 0.65, 0.85]$) to trigger `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` alert events.
12. **FastAPI Backend Server:** `src/api/app.py` exposes REST endpoints and runs asynchronous jobs via `JobManager` (`ThreadPoolExecutor(max_workers=1)`).
13. **Streamlit Operations Console:** `dashboard/app.py` renders interactive navigation views powered by `DashboardApiClient`.
14. **Persistence Layer:** Structured file-based persistence (`results.json` and annotated frame images) and in-memory bounded telemetry queues ($W \le 100$).

---

## 3. Entry Points & Invocation Contracts

### 3.1 Streamlit Frontend Entry Point
- **Authoritative Entry Point:** [dashboard/app.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/dashboard/app.py)
- **Component Submodules:** [dashboard/components/](file:///c:/Users/acer/Desktop/PerceptGuard-AI/dashboard/components/)
- **API Transport Client:** [dashboard/api_client.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/dashboard/api_client.py)
- **Deployment Topology:** Supports dual transport mode: primary HTTP REST API (`http://127.0.0.1:8000`) and in-process `DashboardService` fallback.

### 3.2 FastAPI Backend Entry Point
- **Authoritative Entry Point:** [src/api/app.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/src/api/app.py)
- **Application Factory:** `create_app()`
- **Routers:**
  - Health: `src/api/routers/health.py` (`/health`, `/health/live`, `/health/ready`, `/health/metrics`)
  - Telemetry: `src/api/routers/telemetry.py` (`/api/v1/telemetry/*`)
  - Alerts: `src/api/routers/alerts.py` (`/api/v1/alerts/*`)
  - Video: `src/api/routers/video.py` (`/api/v1/video/*`)

---

## 4. Current Startup Commands

- **Backend (FastAPI):**
  ```powershell
  .\venv\Scripts\uvicorn.exe src.api.app:app --host 127.0.0.1 --port 8000
  ```
- **Frontend (Streamlit):**
  ```powershell
  .\venv\Scripts\streamlit.exe run dashboard/app.py --server.port 8501
  ```
- **Test Suite:**
  ```powershell
  .\venv\Scripts\pytest.exe -q
  ```
- *Note:* Dedicated Windows PowerShell startup scripts (`scripts/start_backend.ps1`, `scripts/start_dashboard.ps1`, `scripts/start_project.ps1`, `scripts/stop_project.ps1`) are required by Day 17 Phase 11.

---

## 5. Existing Integration Status

| Subsystem Component | Integration Mechanism | Status | Notes |
| :--- | :--- | :---: | :--- |
| **FastAPI Backend ↔ Streamlit** | `DashboardApiClient` HTTP REST & In-Process Fallback | **Operational** | Automatic probe on startup; seamless fallback if backend is offline. |
| **Video Job Processing** | `JobManager` + `VideoPipelineRunner` | **Operational** | Background `ThreadPoolExecutor(max_workers=1)` prevents CPU starvation. |
| **YOLO11n Detection** | `SingleImageDetector` / `BatchDetector` | **Operational** | Pretrained weights `models/detection/yolo11n.pt` verified ($5.35\text{ MB}$). |
| **ByteTrack Tracking** | `ByteTrackTracker` | **Operational** | Kalman state tracking with empirical continuity tracking. |
| **Feature Extraction** | `FeatureExtractor` (27 features) | **Operational** | Normalizers configured; handles NaN/Inf safely. |
| **LSTM Forecaster** | `LSTMForecaster` | **Operational** | Weights `models/forecasting/model.pt` verified ($62.6\text{ KB}$). |
| **Persistence** | File-Based Artifacts (`results.json`, `annotated/`) | **Operational** | Persisted under `outputs/video_jobs/{job_id}/`. |

---

## 6. Duplicate and Unused Files Analysis

1. **Root vs. `docker/` Dockerfiles:**
   - `Dockerfile` vs `docker/Dockerfile.backend` (identical contents).
   - `Dockerfile.frontend` vs `docker/Dockerfile.frontend` (identical contents).
   - *Reference Check:* `tests/deployment/test_docker_configuration.py:18,49,74,75` and `.github/workflows/ci.yml` explicitly test both root and `docker/` locations. Retained to prevent breaking CI and regression tests.
2. **Dataset Validation Reports:**
   - `datasets/processed/driving_sample/validation_report.json` vs `outputs/reports/dataset_validation_report.json`.
   - *Reference Check:* `scripts/audit_project.py:580` specifically validates `datasets/processed/driving_sample/validation_report.json`.
3. **Outputs Ephemeral Folders:**
   - Multiple historical run job folders (`outputs/jobs/job_*`) from Days 13–16 tests. These are untracked by Git (`.gitignore`).
4. **Markdown Documentation:**
   - Day 16 consolidated 11 exact duplicates. 160 Markdown files remain, 48 of which are historical audit reports that must remain intact.

---

## 7. Broken Imports & Syntax Inspection

- `src/api/routers/video.py`: The missing `List` import reported in Day 16 was resolved.
- `ultralytics` Settings Sandbox Issue: In restricted Windows sandbox without filesystem access to `%APPDATA%`, `ultralytics` raises `WinError 5: Access is denied: '...\Ultralytics\settings.json'`. When executed outside sandbox or with `YOLO_CONFIG_DIR` redirected to workspace, it executes cleanly.
- `dashboard/app.py` & `dashboard/components/`: Clean compilation verified via `py_compile`; zero broken imports detected.

---

## 8. Missing Dependencies

- All required packages (`streamlit`, `fastapi`, `uvicorn`, `httpx`, `torch`, `ultralytics`, `opencv-python-headless`, `pydantic`, `pandas`, `numpy`, `scipy`) are verified installed in the Python 3.13.3 virtual environment.

---

## 9. Known Runtime Errors & Warnings

1. **Port Conflicts:** If ports 8000 or 8501 are already in use, FastAPI or Streamlit will fail to bind. Startup scripts must detect port availability.
2. **Ultralytics Settings Access:** Access to `%APPDATA%\Roaming\Ultralytics` requires redirecting config directory to local `.yolo/` if running in constrained sandbox environments.
3. **Sequential Processing Limit:** Submitting multiple videos concurrently causes queuing because `max_workers=1`. This is intentional to prevent CPU core exhaustion on 2-core / 4-thread hardware.

---

## 10. Day 17 Blockers

- **Zero Critical Blockers:** Core ML pipeline, FastAPI backend, and Streamlit dashboard are functional.
- **Action Items for Day 17:**
  - Create dedicated PowerShell startup and shutdown scripts (`scripts/start_backend.ps1`, `scripts/start_dashboard.ps1`, `scripts/start_project.ps1`, `scripts/stop_project.ps1`).
  - Refine Streamlit interface to cleanly present all 10 requested operational views (specifically separating Object Detection and Object Tracking as distinct views while maintaining full backward compatibility).
  - Add comprehensive regression tests for Streamlit error handling, API client edge cases, and video upload validation.
  - Execute end-to-end real video processing test through the application.
  - Document runtime validation and update release documentation.
