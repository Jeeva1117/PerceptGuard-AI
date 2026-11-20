# PerceptGuard AI — Docker Setup & Local Container Execution Guide

**Module:** Containerization Setup & Local Execution  
**Hardware Tier:** CPU-Optimized (Intel Core i3 / 8 GB RAM / Windows 11 / Linux)  
**Deployment Modality:** Docker & Docker Compose (Topology B Decoupled Architecture)  

---

## 1. Overview

PerceptGuard AI provides production-grade, multi-stage Docker containerization optimized for CPU-only inference, low memory footprints, and security-hardened non-root execution.

The setup consists of:
- **`backend`**: FastAPI REST API service (`Dockerfile` / `docker/Dockerfile.backend`) on port `8000`.
- **`frontend`**: Streamlit Operations Dashboard (`Dockerfile.frontend` / `docker/Dockerfile.frontend`) on port `8501`.
- **`docker-compose.yml`**: Multi-container orchestration, internal service networking, health checks, and volume mounts.
- **`.dockerignore`**: Strict build-context exclusions protecting secrets, raw datasets, virtual environments, and model weights.

---

## 2. Prerequisites

- **Docker Engine:** Version 20.10.0+ (or Docker Desktop on Windows/macOS)
- **Docker Compose:** Version 2.0.0+ (`docker compose`)
- **System Memory:** Minimum 4 GB RAM available for Docker daemon (8 GB host RAM recommended)
- **Disk Space:** ~2.5 GB for base layers and Python wheel caches

---

## 3. Quick Start (Docker Compose)

### Step 1: Clone and Inspect Environment
Ensure you are in the project root directory:
```bash
cd /path/to/PerceptGuard-AI
cp .env.example .env
```

### Step 2: Build the Container Images
Build both backend and frontend images using multi-stage caching:
```bash
docker compose build
```

### Step 3: Launch Services in Detached Mode
```bash
docker compose up -d
```

### Step 4: Verify Service Health & Logs
Monitor the automated container health probes:
```bash
# Check service health and running status
docker compose ps

# Follow structured access logs
docker compose logs -f backend
docker compose logs -f frontend
```

### Step 5: Access the Interfaces
- **Streamlit Operations Console:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Liveness Probe:** [http://localhost:8000/health/live](http://localhost:8000/health/live)
- **Backend Readiness Probe:** [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

### Step 6: Graceful Teardown
```bash
# Stop containers without losing persistent volume data
docker compose down

# Stop containers and remove volumes (clean reset)
docker compose down -v
```

---

## 4. Standalone Container Execution (Topology A & Single-Service)

### Standalone Backend REST API
```bash
# Build standalone backend
docker build -t perceptguard-backend:latest -f Dockerfile .

# Run standalone backend container
docker run -d \
  --name perceptguard-backend \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/datasets/processed:/app/datasets/processed:ro \
  -e APP__ENVIRONMENT=production \
  -e MODEL__DEVICE=cpu \
  perceptguard-backend:latest
```

### Standalone Frontend Operations Dashboard
```bash
# Build standalone frontend
docker build -t perceptguard-frontend:latest -f Dockerfile.frontend .

# Run standalone frontend connecting to host backend
docker run -d \
  --name perceptguard-frontend \
  -p 8501:8501 \
  -e DASHBOARD__API_BASE_URL=http://host.docker.internal:8000 \
  perceptguard-frontend:latest
```

---

## 5. Security Decisions & Non-Root Execution

1. **Non-Root User:** Both images create and switch to system user `appuser` (UID `1000`, GID `1000`). Root privileges are discarded before container launch.
2. **Minimal Surface:** Base images use Debian `python:3.11-slim`. Build compilers (`build-essential`) are isolated in the `builder` stage and omitted from the final runtime image.
3. **Secret Isolation:** `.dockerignore` blocks `.env`, API credentials, and development configs.
4. **Volume Permissions:** Persistent storage mounts (`/app/outputs`, `/app/logs`) are pre-created with `appuser` ownership.
