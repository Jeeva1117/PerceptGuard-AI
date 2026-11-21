# PerceptGuard AI — Day 17 Master Runtime Validation & Application Integration Report

## 1. Executive Summary & Release Verdict

| Attribute | Specification |
| :--- | :--- |
| **Project Name** | **PerceptGuard AI** |
| **Project Tagline** | *"Anticipating Detection Failures Before They Happen."* |
| **Milestone** | Day 17 — Streamlit Application Integration, Repository Cleanup & Full Runtime Stabilization |
| **Audit Date** | 2026-11-21 |
| **Lead Roles** | Principal Software Engineer, Senior Staff Engineer, Streamlit Specialist, FastAPI Architect, CV/ML Engineer, DevOps & QA Lead |
| **Target Hardware** | Intel Core i3 11th Gen (2 Cores / 4 Threads), 8 GB RAM, Windows 11 AMD64 (Pure CPU Execution) |
| **Overall Verdict** | **`PASS WITH LIMITATIONS`** |
| **Calibration Status** | **`PROVISIONAL CALIBRATION`** |

---

## 2. Initial Audit Findings & Baseline Verification

Prior to implementing runtime stabilization, an independent baseline audit was conducted and recorded in [reports/day17/initial_runtime_audit.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/reports/day17/initial_runtime_audit.md):
- **Working Tree:** Verified clean baseline at commit `c916831`.
- **Architecture Validation:** Verified the complete 14-stage computer vision and predictive reliability forecasting pipeline without broken contracts.
- **Entry Points:**
  - Streamlit Frontend: `dashboard/app.py`
  - FastAPI Backend: `src/api/app.py`
- **Initial Test Suite:** 1,426 / 1,426 tests passed.

---

## 3. Repository Cleanup & Consolidation Results

Recorded in [reports/day17/repository_cleanup_report.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/reports/day17/repository_cleanup_report.md):
- **Duplicate Candidates Audited:**
  - `Dockerfile` vs `docker/Dockerfile.backend` and `Dockerfile.frontend` vs `docker/Dockerfile.frontend`: Retained in both locations to satisfy `tests/deployment/test_docker_configuration.py` and `.github/workflows/ci.yml`.
  - `datasets/processed/driving_sample/validation_report.json`: Preserved for `scripts/audit_project.py:580`.
  - Ephemeral outputs in `outputs/jobs/` confirmed gitignored.
- **Net Hygiene:** Zero unnecessary files added; zero required files or historical audit reports deleted.

---

## 4. Production Windows 11 Startup Workflow

Implemented 4 robust PowerShell management scripts in `scripts/`:
1. [scripts/start_project.ps1](file:///c:/Users/acer/Desktop/PerceptGuard-AI/scripts/start_project.ps1): One-command master orchestrator. Verifies virtual environment, weights, checks port 8000, starts background FastAPI backend, verifies `/health/live` response, and launches Streamlit dashboard on port 8501.
2. [scripts/start_backend.ps1](file:///c:/Users/acer/Desktop/PerceptGuard-AI/scripts/start_backend.ps1): Dedicated FastAPI launcher with port conflict detection and model weight checks.
3. [scripts/start_dashboard.ps1](file:///c:/Users/acer/Desktop/PerceptGuard-AI/scripts/start_dashboard.ps1): Dedicated Streamlit dashboard launcher with backend reachability probe.
4. [scripts/stop_project.ps1](file:///c:/Users/acer/Desktop/PerceptGuard-AI/scripts/stop_project.ps1): Graceful service shutdown utility. Identifies PIDs listening on ports 8000 and 8501, terminates processes safely, and verifies port release.

---

## 5. Backend REST API Expansion & Verification

To satisfy granular telemetry queries for external ADS clients and modular UI components, added 6 dedicated REST sub-resource endpoints to `src/api/routers/video.py`:
- `GET /api/v1/video/jobs/{job_id}/detections`: Aggregated and frame-level object detections (class, confidence, bounding box coordinates).
- `GET /api/v1/video/jobs/{job_id}/tracking`: ByteTrack Kalman trajectory tracking data and unique track IDs.
- `GET /api/v1/video/jobs/{job_id}/features`: 27-dimensional feature vectors and visual quality metrics per frame.
- `GET /api/v1/video/jobs/{job_id}/forecast`: Multi-horizon predicted reliability ($H1, H3, H5$) and horizon-level risk summaries.
- `GET /api/v1/video/jobs/{job_id}/risk`: Failure risk scores ($S = 1.0 - R$) and categorical risk assessments.
- `GET /api/v1/video/jobs/{job_id}/alerts`: Early-warning alert notifications triggered during video processing.

Updated [dashboard/api_client.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/dashboard/api_client.py) with client wrapper methods that automatically support both HTTP REST and In-Process fallback execution.

---

## 6. Streamlit Frontend Integration & Navigation

- **Authoritative Entry Point:** [dashboard/app.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/dashboard/app.py)
- **Component Views:** [dashboard/components/](file:///c:/Users/acer/Desktop/PerceptGuard-AI/dashboard/components/)
- **10 Core Operational Domains Provided:**
  1. Overview Dashboard (`render_system_overview`)
  2. Video Upload & Processing (`render_video_processing_view`)
  3. Object Detection (`render_object_detection` & `render_detection_tracking`)
  4. Object Tracking (`render_object_tracking` & `render_detection_tracking`)
  5. Feature Analytics (`render_feature_analytics`)
  6. Reliability Forecasting (`render_reliability_forecasting`)
  7. Risk Assessment (`render_risk_overview`)
  8. Early Warning Alerts (`render_early_warning_alerts`)
  9. Processing History (`render_processing_history`)
  10. System Health (`render_system_health`)
- **Dual Transport Mode:** Automatically detects live FastAPI backend (`REST_API`) and gracefully falls back to in-process `DashboardService` (`IN_PROCESS`) if the backend is unreachable.

---

## 7. Real Video End-to-End Pipeline Validation

Validated against authentic BDD100K driving video (`outputs/day14_validation/clips/scenario_a_normal_daytime.mp4`):
- **Video Specifications:** 720p H.264 driving sequence, 30 FPS.
- **Frames Processed:** 10 frames on pure CPU.
- **Detections:** Vehicle bounding boxes detected by YOLO11n ($54.1\text{ ms/frame}$ warm latency).
- **Tracking:** ByteTrack Kalman filter sustained continuous track IDs.
- **27 Multi-Modal Features:** Laplacian blur, brightness, contrast, motion kinematics, and detection metrics extracted per frame.
- **LSTM Forecasting:** Multi-horizon future reliability predicted for $t+1, t+3, t+5$ horizons.
- **Derived Risk:** Failure risk computed as $S = 1.0 - R$ and categorized against $[0.35, 0.65, 0.85]$.
- **Persistence:** Persisted to `results.json` and 10 annotated JPEG frames in `annotated/`.

---

## 8. Automated Regression Test Suite Metrics

```
====================== 1433 passed in 134.20s ======================
```

| Test Category | Target Location | Count | Pass Rate | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Day 17 Integration & Sub-Resources** | `tests/test_day17_runtime_integration.py` | 7 | 100.0% | **PASS** |
| **API Endpoints & Lifecycle** | `tests/api/` | 32 | 100.0% | **PASS** |
| **Frontend UI & Components** | `tests/frontend/` | 41 | 100.0% | **PASS** |
| **Day 16 End-to-End Validation** | `tests/test_day16_end_to_end_validation.py` | 1 | 100.0% | **PASS** |
| **Detection Subsystem** | `tests/detection/` | 78 | 100.0% | **PASS** |
| **Tracking Subsystem** | `tests/tracking/` | 113 | 100.0% | **PASS** |
| **Feature Engineering** | `tests/features/` | 282 | 100.0% | **PASS** |
| **Forecasting & LSTM** | `tests/forecasting/` | 275 | 100.0% | **PASS** |
| **Risk & Early Warning** | `tests/risk/`, `tests/early_warning/` | 114 | 100.0% | **PASS** |
| **Root Regression Wrappers** | `tests/*.py` | 490 | 100.0% | **PASS** |
| **TOTAL TEST SUITE** | **Full Repository** | **1,433** | **100.0%** | **PASS** |

---

## 9. Machine Learning Artifact Availability

| Artifact | File Path | Format | Size | Status |
| :--- | :--- | :--- | :--- | :---: |
| **YOLO11n Detector** | `models/detection/yolo11n.pt` | PyTorch TorchScript | $5.35\text{ MB}$ | **Verified** |
| **LSTM Forecaster** | `models/forecasting/model.pt` | PyTorch State Dict | $62.6\text{ KB}$ | **Verified** |
| **Driving Evaluation Video** | `outputs/day14_validation/clips/scenario_a_normal_daytime.mp4` | MP4 (H.264) | $1.53\text{ MB}$ | **Verified** |

---

## 10. Operational Limitations & Final Verdict

1. **Provisional Calibration Gate:**
   - Failure risk thresholds ($[0.35, 0.65, 0.85]$) are designated as **`PROVISIONAL CALIBRATION`** pending closed-loop on-road validation across extreme weather domains (heavy rainfall, dense fog, blizzard).
2. **Sequential Background Queue:**
   - Background video processing operates sequentially with `ThreadPoolExecutor(max_workers=1)` to avoid CPU thread contention on 2-core / 4-thread hardware.
3. **Automotive Camera Perspective:**
   - Pretrained weights and feature normalizers are specialized for forward-facing automotive camera perspectives.

$$\mathbf{FINAL \ VERDICT: \quad PASS \ WITH \ LIMITATIONS}$$
$$\text{Calibration Gate Status: \quad PROVISIONAL \ CALIBRATION}$$
