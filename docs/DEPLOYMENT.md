# PerceptGuard AI — Deployment & Infrastructure Guide

## 1. Overview

PerceptGuard AI supports two deployment modalities:
1. **Local Direct Execution:** High-speed development and edge testing on Python 3.10+ virtual environments.
2. **Containerized Orchestration:** Production-grade reproducible deployment via **Docker** and **Docker Compose**.

---

## 2. Docker Architecture & Services

The containerized deployment consists of two coordinated services orchestrated via `docker-compose.yml`:

```
               ┌────────────────────────────────────────────────────────┐
               │                     Host Machine                       │
               │                   (Docker Engine)                      │
               └───────────┬────────────────────────────────┬───────────┘
                           │                                │
             Port 8501     │                  Port 8000     │
             (Browser)     ▼                  (REST API)    ▼
               ┌───────────────────────┐        ┌───────────────────────┐
               │      frontend         │        │        backend        │
               │ (Streamlit Dashboard) │───────>│   (FastAPI Backend)   │
               │  Dockerfile.frontend  │        │      Dockerfile       │
               └───────────┬───────────┘        └───────────┬───────────┘
                           │                                │
                           └───────────────┬────────────────┘
                                           │
                                           ▼
                           ┌────────────────────────────────┐
                           │       Shared Volumes           │
                           │   - ./outputs:/app/outputs     │
                           │   - ./models:/app/models       │
                           │   - ./datasets:/app/datasets   │
                           └────────────────────────────────┘
```

### 2.1 Backend Service (`perceptguard-backend`)
- **Base Image:** `python:3.11-slim`
- **Exposed Port:** `8000`
- **Command:** `uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1`
- **Health Check:** `curl -f http://localhost:8000/health || exit 1`
- **Resource Constraints:** Tailored for edge execution (memory limit: 2 GB, CPU limit: 2.0 cores).

### 2.2 Frontend Service (`perceptguard-frontend`)
- **Base Image:** `python:3.11-slim`
- **Exposed Port:** `8501`
- **Command:** `streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0`
- **Environment:** `API_BASE_URL=http://backend:8000`
- **Health Check:** `curl -f http://localhost:8501/_stcore/health || exit 1`

---

## 3. Docker Compose Quickstart

### Prerequisites
- Docker Engine 24.0+ and Docker Compose v2.20+
- Host system with at least 8 GB RAM

### Step 1: Clone and Configure Environment
```bash
git clone https://github.com/Jeeva1117/PerceptGuard-AI.git
cd PerceptGuard-AI
cp .env.example .env
```

### Step 2: Build and Launch Containers
```bash
docker compose up --build -d
```

### Step 3: Verify Deployment
Inspect service logs:
```bash
docker compose logs -f backend
docker compose logs -f frontend
```
Verify container status:
```bash
docker compose ps
```

### Step 4: Access the System
- **Dashboard:** Open `http://localhost:8501` in your browser.
- **API Documentation:** Open `http://localhost:8000/docs`.

### Step 5: Teardown
```bash
docker compose down
```

---

## 4. Persistent Storage Architecture

The following directories must be mounted as persistent volumes:
- `outputs/`: Retains video processing job artifacts, frame crops, and `results.json`.
- `models/`: Mounts local model weights (`models/yolo/yolo11n.pt`, `models/forecasting/lstm/model.pt`).
- `datasets/`: Mounts test video clips (`datasets/raw/`) and annotation catalogs.

---

## 5. Security & Production Hardening

1. **Non-Root Execution:** Container processes run under a dedicated unprivileged user (`perceptguard`, UID 1000).
2. **Read-Only Root Filesystem:** Temporary files are strictly constrained to `/tmp` and mounted volume shares.
3. **Defense-in-Depth Headers:** FastAPI backend enforces `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and strict CORS policies.
4. **Zero Embedded Secrets:** All configuration parameters are injected via environment variables or `.env` files; secrets are never committed into container layers.
