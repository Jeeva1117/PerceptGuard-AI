# PerceptGuard AI — Deployment Architecture & Infrastructure Strategy

## 1. Overview & Architectural Goals

The **PerceptGuard AI** deployment architecture is designed to host the perception failure anticipation pipeline, risk assessment engines, and early-warning operations console with:
- **Low latency:** Sub-millisecond risk evaluation ($0.186\text{ ms}$ overhead).
- **High reliability:** Graceful degradation when external sensors or model weights are offline.
- **Resource efficiency:** Minimal CPU and RAM utilization ($< 500\text{ MB}$ total working set).
- **Security & isolation:** Zero credential exposure and decoupled model weight storage.

---

## 2. Deployment Topology Evaluation

### Topology A: Unified Containerized Streamlit Service (Recommended for Staging & Local Edge)
```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                         │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │             Streamlit Web UI (Port 8501)            │   │
│   │                  dashboard/app.py                   │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │ In-Process Python API        │
│   ┌──────────────────────────▼──────────────────────────┐   │
│   │             DashboardService / Pipeline             │   │
│   │           src/early_warning/dashboard_service.py     │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │ Direct Ingestion             │
│   ┌──────────────────────────▼──────────────────────────┐   │
│   │   Risk Scorer -> Classifier -> Temporal Monitor     │   │
│   │                 EarlyWarningEngine                  │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │ Volume Mounts                │
│             /app/models  ◄───┴───►  /app/datasets           │
└─────────────────────────────────────────────────────────────┘
```
- **Strengths:**
  - Simplest operational footprint: 1 container, 1 command.
  - Zero inter-process serialization overhead; instant response ($< 1\text{ ms}$).
  - Fully compatible with Streamlit Community Cloud, Render, or Docker on Linux/Windows.
- **Suitability:** **Immediate deployment target for Day 11**.

---

### Topology B: Decoupled FastAPI Backend + Frontend (Recommended for Multi-Vehicle Fleets)
```
┌───────────────────────────────────────┐       ┌───────────────────────────────────────┐
│           Frontend Service            │       │             Backend Service           │
│        Streamlit or React UI          │       │           FastAPI REST API            │
│              Port 8501                │       │               Port 8000               │
│                                       │       │                                       │
│  Queries REST Endpoints:              │       │  Exposes Endpoints:                   │
│  GET /api/v1/health                   │HTTP/WS│  GET /api/v1/health                   │
│  GET /api/v1/risk/overview     ───────┼───────►  GET /api/v1/risk/overview            │
│  GET /api/v1/alerts                   │       │  POST /api/v1/alerts/{id}/ack         │
│  POST /api/v1/alerts/{id}/ack         │       │  POST /api/v1/predictions/ingest      │
└───────────────────────────────────────┘       └───────────────────────────────────────┘
```
- **Strengths:**
  - Decouples UI rendering from heavy telemetry processing.
  - Allows external vehicle sensors (ROS, MQTT, Kafka) to push live telemetry into the API.
  - Independent scaling of backend inference workers and frontend dashboard instances.
- **Suitability:** Target for enterprise fleet deployment and vehicle operations centers (TOC).

---

## 3. Container Specification (Dockerfile Reference)

```dockerfile
# Multi-stage production container for PerceptGuard AI
FROM python:3.11-slim AS runtime

# System dependencies for OpenCV and system monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-optimized Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY config/ config/
COPY src/ src/
COPY dashboard/ dashboard/
COPY experiments/ experiments/
COPY datasets/ datasets/

# Expose Streamlit default port
EXPOSE 8501

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 4. Resource Sizing & Hosting Platform Comparison

| Platform / Environment | Recommended Hardware Sizing | Deployment Model | Suitability & Notes |
| :--- | :--- | :--- | :--- |
| **Local / Edge Vehicle** | 4 CPU Cores, 4 GB RAM, CPU-only | Native Docker / Podman | **Verified:** Low power, zero network latency. |
| **Streamlit Community Cloud** | Shared vCPU, 1 GB RAM | Direct GitHub integration | **Feasible:** Instant zero-cost public preview; model weights must be pre-bundled. |
| **Render.com / Railway** | 1 vCPU, 2 GB RAM ($7/month) | Dockerized Web Service | **Recommended Cloud Host:** Automated GitHub CI/CD, custom domains, health checks. |
| **AWS ECS / GCP Cloud Run** | 1–2 vCPUs, 2–4 GB RAM | Serverless Container | **Enterprise:** High availability, automated autoscaling, IAM secret integration. |

---

## 5. Storage & Volume Mounting Strategy

1. **Model Weights (`models/forecasting/`):**
   - Checkpoint weights (`checkpoint_best.pt`, `*.pt`) are excluded from Git repository history.
   - For cloud or container deployments, weights should be mounted via Docker volume:
     `-v /host/path/models:/app/models`
     or downloaded during startup from private S3/GCS buckets via an entrypoint script.
2. **Processed Telemetry Datasets (`datasets/processed/`):**
   - Summary telemetry (`tracking_summary.json`) and predictions (`predictions.json`) are lightweight (< 500 KB) and baked into the image or volume-mounted.
   - Raw multi-gigabyte video files (`datasets/raw/*.bag`) remain strictly on local storage or external object storage.

---

## 6. Environment Variables & Production Secrets

| Variable Name | Default Value | Production Recommended Value | Purpose |
| :--- | :--- | :--- | :--- |
| `DEVICE` | `cpu` | `cpu` (or `cuda` if GPU available) | Compute device selection |
| `NUM_WORKERS` | `2` | `2`–`4` | Feature extraction worker threads |
| `CONFIDENCE_THRESHOLD` | `0.35` | `0.35` | YOLO11n object detection confidence |
| `IOU_THRESHOLD` | `0.45` | `0.45` | Non-Maximum Suppression (NMS) IoU |
| `LOG_LEVEL` | `INFO` | `INFO` or `WARNING` | Logging verbosity |
| `STREAMLIT_PORT` | `8501` | `8501` | Dashboard HTTP listening port |
| `CALIBRATION_STATUS` | `PROVISIONAL` | `PROVISIONAL` | Safety calibration gate label |
