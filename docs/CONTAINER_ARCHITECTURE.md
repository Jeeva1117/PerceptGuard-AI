# PerceptGuard AI — Container Deployment Architecture

**Module:** DevOps & MLOps Infrastructure Specification  
**Architecture:** Decoupled Microservices with High-Availability Health Orchestration  

---

## 1. System Container Architecture Diagram

```
                             [ Autonomous Vehicle Operator / Client Browser ]
                                                    │
                                                    ▼
                             ┌──────────────────────────────────────────────┐
                             │               Host Machine                   │
                             │                                              │
                             │   Port 8501                      Port 8000   │
                             └───┬──────────────────────────────┬───────────┘
                                 │                              │
                                 │ HTTP                         │ HTTP
                                 │                              │
         ┌───────────────────────▼──────────────────────────────▼───────────────────────┐
         │                    Docker Engine (perceptguard-network bridge)              │
         │                                                                             │
         │   ┌───────────────────────────────┐     ┌───────────────────────────────┐   │
         │   │   perceptguard-frontend       │     │    perceptguard-backend       │   │
         │   │   (Streamlit Operations UI)   │     │    (FastAPI Asynchronous API) │   │
         │   │                               │     │                               │   │
         │   │   Port: 8501                  │     │    Port: 8000                 │   │
         │   │   User: appuser (UID 1000)    │     │    User: appuser (UID 1000)   │   │
         │   │   Runtime: Python 3.11-slim   │     │    Runtime: Python 3.11-slim  │   │
         │   │                               │     │                               │   │
         │   │   Health Probe:               │     │    Health Probe:              │   │
         │   │   curl /_stcore/health        │     │    curl /health/live          │   │
         │   └───────────────┬───────────────┘     └───────────────▲───────────────┘   │
         │                   │                                     │                   │
         │                   │ HTTP Client Requests                │                   │
         │                   │ (http://backend:8000/api/v1/*)      │                   │
         │                   └─────────────────────────────────────┘                   │
         │                                                                             │
         └──────────────────────────────────────┬──────────────────────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │              Host Storage Volumes            │
                         │                                              │
                         │   ./models (ro)     ──────► /app/models      │
                         │   ./datasets (ro)   ──────► /app/datasets    │
                         │   backend_outputs   ──────► /app/outputs     │
                         │   backend_logs      ──────► /app/logs        │
                         └──────────────────────────────────────────────┘
```

---

## 2. Service Architectural Breakdown

### 2.1. Backend Service (`backend`)
- **Base Image:** `python:3.11-slim`
- **Application Server:** `uvicorn` (ASGI, single-worker CPU-optimized)
- **Port:** `8000`
- **Core Endpoints:**
  - `/health/live`: Container orchestrator liveness probe.
  - `/health/ready`: Dependency and model readiness check.
  - `/api/v1/telemetry/*`: Multi-horizon risk scoring ($H1, H3, H5$) and tracking context.
  - `/api/v1/alerts/*`: Early-warning alert queries and operator lifecycle state management.
  - `/api/v1/video/*`: Ingestion format validation and slice processing.
- **Resource Envelope:**
  - Reservations: `0.5 vCPU`, `512 MB RAM`
  - Limits: `2.0 vCPU`, `2048 MB RAM`

### 2.2. Frontend Service (`frontend`)
- **Base Image:** `python:3.11-slim`
- **Application Server:** `streamlit` (headless mode, disabled usage stats)
- **Port:** `8501`
- **Data Transport:** Decoupled REST queries over `http://backend:8000` via `DashboardApiClient` with automated in-process fallback for network resilience.
- **Startup Dependency:** Configured with `depends_on.backend.condition: service_healthy`. Frontend will not accept traffic until backend passes `/health/live`.
- **Resource Envelope:**
  - Reservations: `0.25 vCPU`, `256 MB RAM`
  - Limits: `1.5 vCPU`, `1024 MB RAM`

---

## 3. Storage Strategy & Data Persistence

1. **Model Weights (`/app/models`):**
   - Bound read-only (`:ro`) from host `./models`.
   - Protects YOLO11n weights and LSTM forecasting checkpoints from accidental container corruption.
   - If missing from host, backend falls back to replay mode and reports `degraded` via readiness probes without crashing.
2. **Processed Datasets (`/app/datasets/processed`):**
   - Bound read-only (`:ro`) from host `./datasets/processed`.
   - Provides verified tracking sequences and telemetry records.
3. **Application Logs & Outputs (`backend_logs`, `backend_outputs`):**
   - Stored in Docker named volumes (`perceptguard-backend-logs`, `perceptguard-backend-outputs`).
   - Ensures video validation artifacts and structured JSON logs persist across container restarts.

---

## 4. Network Isolation & Security

- **User-Defined Bridge Network (`perceptguard-net`):** Containers communicate through automatic DNS resolution (`http://backend:8000`).
- **Port Isolation:** In enterprise deployments, backend port `8000` can be removed from host mapping and kept entirely private within `perceptguard-net`, routing external traffic solely through the frontend or an ingress API gateway.
- **Non-Privileged Execution:** Zero container capabilities added (`cap_drop: ALL` compatible).
