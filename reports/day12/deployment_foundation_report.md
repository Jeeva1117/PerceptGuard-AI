# Day 12 — Local Deployment Foundation & Architecture Report
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. Executive Summary

Day 12 established the local deployment foundation for PerceptGuard AI, verifying that all system components operate reproducibly on local CPU hardware (Intel Core i3, 8 GB RAM, Windows 11).

| Attribute | Deployment Verification Record |
| :--- | :--- |
| **Audit Date** | 2026-11-20 |
| **Milestone** | Day 12 — Local Deployment Foundation Engineering |
| **Overall Status** | **`PASS WITH LIMITATIONS`** |
| **Backend API** | FastAPI ASGI application (`src/api/app.py`), verified startup ($1.37\text{s}$) |
| **Frontend UI** | Streamlit console (`dashboard/app.py`), dual-topology client verified |
| **Storage Architecture** | File-based persistence (`outputs/`, `datasets/`, `models/`), atomic JSON writes |
| **YOLO Checkpoint** | `models/yolo/yolo11n.pt` ($5.6\text{ MB}$, verified present and loadable) |
| **LSTM Checkpoint** | `models/forecasting/lstm/model.pt` ($103.5\text{ KB}$, verified present and loadable) |
| **Container Status** | Multi-stage Dockerfiles and Compose verified statically. Local runtime Docker CLI **BLOCKED**. |

---

## 2. Subsystem Readiness Matrix

| Area | Status | Implementation Details |
| :--- | :---: | :--- |
| **Overall Software Readiness** | **PASS WITH LIMITATIONS** | Complete local application startup with fallback mechanisms. |
| **Backend API** | **PASS** | Lifespan initialization, standardized error envelopes, request ID tracing. |
| **Frontend Integration** | **PASS** | Dual-topology REST / In-process client, health monitoring, responsive UI. |
| **Database & Storage** | **PASS** | File-based atomic JSON persistence; zero unnecessary external database overhead. |
| **YOLO Weights** | **PASS** | `models/yolo/yolo11n.pt` verified on disk; CPU inference functional. |
| **LSTM Checkpoint** | **PASS** | `models/forecasting/lstm/model.pt` deployed and verified on CPU. |
| **Docker Configuration** | **PASS WITH LIMITATIONS** | Multi-stage Dockerfiles and Compose verified. Host CLI unavailable. |
| **Security & Hygiene** | **PASS** | Non-root execution, zero committed credentials, `.dockerignore` enforcement. |

---

## 3. Verified Local Startup Commands

### Backend Startup (FastAPI)
```powershell
.\venv\Scripts\uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- Health Probe: `http://localhost:8000/health/live`
- Readiness Probe: `http://localhost:8000/health/ready`
- OpenAPI Documentation: `http://localhost:8000/docs`

### Frontend Startup (Streamlit)
```powershell
.\venv\Scripts\streamlit run dashboard/app.py --server.port 8501
```
- Local Console: `http://localhost:8501`

---

## 4. Known Blockers & Operational Limitations

1. **Host Docker CLI Blocked:** The local Windows 11 host does not have the Docker CLI installed. Container builds are validated statically and executed on remote GitHub Actions Ubuntu runners.
2. **Provisional Calibration Gate:** Risk score thresholds remain under `PROVISIONAL CALIBRATION` pending multi-clip driving evaluations.
