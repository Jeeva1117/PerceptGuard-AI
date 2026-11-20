# PerceptGuard AI — Production Cloud Deployment Guide

**Module:** Cloud Hosting & Infrastructure Deployment Guide  
**Scope:** Multi-Provider Assessment, Operational Runbooks, and Production Constraints  

---

## 1. Hosting Provider Compatibility Matrix

| Provider | Supported Tier | Topology | Feasibility | Key Operational Limitations |
| :--- | :--- | :--- | :---: | :--- |
| **Render.com** | Web Service (Docker) | Topology B (Decoupled) | **Recommended** | Free tier sleeps on idle; $7/mo paid tier provides persistent CPU & memory (2 GB RAM recommended). Ephemeral disk requires external S3/GCS or Render Disks for outputs. |
| **Railway.app** | Docker / Compose | Topology B (Decoupled) | **Recommended** | Supports multi-service compose deploys with private internal networking and attachable persistent volumes. |
| **Streamlit Community Cloud** | GitHub Native | Topology A (Unified) | **Feasible (Demo)** | Shared 1 GB vCPU container. Only supports Topology A. Model checkpoints must be committed via Git LFS or downloaded at startup. |
| **AWS ECS / Fargate** | Serverless Container | Topology B (Decoupled) | **Enterprise Target** | Full IAM security, EFS volume mounts for model checkpoints, CloudWatch logs, and ALB ingress. Recommended for multi-vehicle fleet scaling. |
| **Google Cloud Run** | Serverless Container | Backend API Only | **Feasible** | Fast cold starts for REST backend. 60-minute maximum request timeout. Requires Cloud Storage FUSE for model weights. |
| **Vercel** | Serverless Edge | Incompatible for UI | **Not Recommended** | Vercel is optimized for static/Node.js/Next.js frameworks. Streamlit's persistent WebSocket connections are incompatible with Vercel serverless execution limits. |

---

## 2. Cloud Production Deployment Runbook (Render Example)

### Architecture on Render
- **Service 1:** `perceptguard-backend` (Environment: Docker, Root Directory: `.`, Dockerfile: `Dockerfile`)
- **Service 2:** `perceptguard-frontend` (Environment: Docker, Root Directory: `.`, Dockerfile: `Dockerfile.frontend`)

### Environment Variable Checklist
```ini
# Backend Service Variables
APP__ENVIRONMENT=production
APP__DEBUG=false
API__HOST=0.0.0.0
API__PORT=8000
API__PREFIX=/api/v1
API__CORS_ORIGINS=["https://perceptguard-ui.onrender.com"]
MODEL__DEVICE=cpu
MODEL__NUM_WORKERS=2
LOG__LEVEL=INFO
SECURITY__ALLOWED_HOSTS=["perceptguard-api.onrender.com"]
SECRET_KEY=generate-a-cryptographically-secure-32-byte-hex-key

# Frontend Service Variables
APP__ENVIRONMENT=production
DASHBOARD__API_BASE_URL=https://perceptguard-api.onrender.com
DASHBOARD__API_REQUEST_TIMEOUT_SECONDS=15.0
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 3. Operational Risk Factors & Mitigation Strategies

### 1. Ephemeral Filesystem Data Loss
- **Risk:** Cloud containers reset local storage on redeployment or restart, losing user-uploaded videos or generated report artifacts.
- **Mitigation:**
  - Mount persistent network block storage (e.g. AWS EFS, Render Persistent Disk).
  - Stream large video processing outputs to S3-compatible cloud object storage.

### 2. Memory Exhaustion During PyTorch Model Initialization
- **Risk:** Running YOLO11n object detection and LSTM forecasting concurrently with video decoding can spike memory consumption above 1 GB.
- **Mitigation:**
  - Configure container memory limits to minimum 2048 MB (`2 GB`).
  - Restrict Uvicorn to 1 worker per container (`--workers 1`) to eliminate duplicated model memory overhead.

### 3. Video Upload Request Timeouts
- **Risk:** Large video files ($50\text{ MB} - 100\text{ MB}$) over slow network connections can hit reverse proxy or ingress gateway timeouts.
- **Mitigation:**
  - Enforce $100\text{ MB}$ strict upload ceiling at API level (`src/api/routers/video.py`).
  - Configure ingress timeouts to $\ge 120\text{ seconds}$.

---

## 4. Maintenance, Rolling Updates & Rollback

### Zero-Downtime Rolling Update
When updating container versions in production:
```bash
# 1. Pull or build latest container images
docker compose build --pull

# 2. Re-create containers with zero downtime (backend first)
docker compose up -d --no-deps --build backend

# 3. Wait for backend health probe to pass, then update frontend
docker compose up -d --no-deps --build frontend
```

### Emergency Rollback
```bash
# Revert to previous image tag
docker tag perceptguard-backend:previous perceptguard-backend:latest
docker tag perceptguard-frontend:previous perceptguard-frontend:latest
docker compose up -d
```
