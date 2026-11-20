# Deployment Foundation & Architecture Guide
## PerceptGuard AI: Predictive Object Detection Reliability System

---

## 1. System Overview

PerceptGuard AI operates a decoupled multi-service local deployment architecture (**Topology B**) optimized for resource-constrained, CPU-only edge hardware:

```
[ Camera / User Video Upload ]
            │
            ▼
┌──────────────────────────────────────────────┐
│  Streamlit Frontend Dashboard (Port 8501)    │
│  - Operations & Telemetry Console            │
│  - Video Ingestion & Annotated Frame Viewer  │
│  - Dual-Topology REST / In-Process Client    │
└──────────────────────┬───────────────────────┘
                       │ HTTP REST (127.0.0.1:8000)
                       ▼
┌──────────────────────────────────────────────┐
│  FastAPI Backend API Service (Port 8000)     │
│  - Lifespan State & Resource Management      │
│  - Background Job Manager (Thread-bounded)   │
│  - Video Ingestion & Pipeline Runner         │
│  - Health, Telemetry & SRE Metrics Probes    │
└──────────────────────┬───────────────────────┘
                       │ In-Memory & Local Storage
                       ▼
┌──────────────────────────────────────────────┐
│  Perception & Predictive Reliability Engine  │
│  - OpenCV Video Extraction & Preprocessing   │
│  - YOLO11n Object Detector (CPU Inference)   │
│  - ByteTrack Kalman Kinematics Tracker       │
│  - 27-Dimensional Multi-Modal Features       │
│  - Multi-Horizon LSTM Forecaster (H1, H3, H5)│
│  - Reliability Risk Scorer (S = 1.0 - R)     │
│  - Temporal Risk Monitor & EMA Velocity      │
│  - Early-Warning Alert Engine (P1, P2, P3)   │
└──────────────────────────────────────────────┘
```

---

## 2. Resource Constraints & CPU Optimization

- **Compute Tier:** Intel Core i3 (11th Gen), 8 GB RAM, Windows 11 AMD64.
- **Worker Allocation:** Single Uvicorn process worker (`API_WORKERS=1`) and single concurrent video processing job to avoid CPU core contention or out-of-memory states.
- **Model Checkpoint Paths:**
  - YOLO Object Detector: `models/yolo/yolo11n.pt` ($5.6\text{ MB}$)
  - LSTM Forecaster: `models/forecasting/lstm/model.pt` ($103.5\text{ KB}$)
- **Persistence Storage:**
  - Upload & Processing Workspace: `outputs/jobs/{job_id}/`
  - Ingested Datasets: `datasets/processed/`
