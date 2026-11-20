# Day 11 Task 5 — Docker Containerization & Deployment Setup Report

## 1. Executive Summary

| Attribute | Value |
| :--- | :--- |
| **Audit Date** | 2026-11-20 |
| **Stage** | Day 11 — Task 5: Docker Containerization & Deployment Setup |
| **Auditor Role** | Senior Staff Software Engineer, DevOps Architect, Docker Specialist, MLOps Deployment Engineer |
| **Target Architecture** | Multi-Container Docker & Compose Orchestration (`Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`) |
| **Hardware Profile** | Intel Core i3 (11th Gen), 8 GB RAM, Windows 11, CPU execution tier |
| **Static Deployment Tests** | **6 / 6 passed (100%)** in $0.24\text{ seconds}$ |
| **Subsystem Regression Suite** | **228 / 228 passed (100%)** across config, api, frontend, early warning, risk, and deployment |
| **Host Docker Execution** | **BLOCKED** (Docker Engine/CLI not installed on host Windows 11 machine; verified via PowerShell) |
| **Vulnerability Scanning** | **NOT RUN** (Host vulnerability scanner CLI not installed) |
| **Overall Verdict** | **PASS WITH LIMITATIONS** |
| **Calibration Status Gate** | **PROVISIONAL CALIBRATION** |

The containerization infrastructure for **PerceptGuard AI** has been established and statically verified. A production-hardened, multi-stage, CPU-optimized container architecture has been engineered supporting the decoupled service model: a FastAPI backend REST service on port `8000` and a Streamlit operations console on port `8501`. Both services enforce non-root execution (`appuser:appgroup`), multi-stage dependency isolation, explicit resource limits, persistent volume mounting for logs and outputs, read-only volume protection for model checkpoints and datasets, automated container health probes, and strict build-context secret exclusion via `.dockerignore`.

---

## 2. Readiness Gate & Existing Implementation Audit

### 2.1. Host Environment & Runtime Inspection
- **Host OS:** Windows 11 AMD64
- **Host Hardware:** Intel Core i3 (11th Gen), 8 GB RAM, no dedicated CUDA GPU (CPU-only execution tier).
- **Python Runtime:** Python 3.13.3 AMD64 verified in virtual environment (`venv`).
- **Docker Tooling Audit:** Executed `docker --version`, `docker compose version`, `docker info` via PowerShell. The Docker CLI binary is **not installed or recognized on this host system**.
- **Audit Compliance:** In strict adherence to prompt instructions (*"If Docker is unavailable, do not claim builds passed. Instead, perform static configuration validation and report Docker execution as BLOCKED or NOT RUN. Never fabricate Docker build logs."*), runtime image builds and Docker Compose integration runs are accurately recorded as **BLOCKED / NOT RUN**, while comprehensive static schema parsing, multi-stage AST analysis, and security exclusion testing were executed and validated.

### 2.2. Previous Task Deliverables Inspection
All Day 11 prerequisite reports and outputs were verified:
- `reports/day11/production_readiness_report.md` (Task 1: Topology and architecture planning verified).
- `reports/day11/environment_hardening_report.md` (Task 2: Pydantic configuration hierarchy verified).
- `reports/day11/backend_hardening_report.md` (Task 3: Production FastAPI backend and 30 tests verified).
- `reports/day11/frontend_optimization_report.md` (Task 4: Modular UI components, API client, and 31 tests verified).

---

## 3. Containerization Architecture & Configuration

### 3.1. Backend Dockerfile (`Dockerfile` and `docker/Dockerfile.backend`)
- **Base Image:** `python:3.11-slim` (Debian bookworm-slim).
- **Multi-Stage Build:**
  - `builder` stage: Installs `build-essential` and compiles Python wheels in an isolated virtual environment (`/opt/venv`).
  - `runtime` stage: Copies `/opt/venv` without dragging C compiler toolchains into production.
- **Security & Non-Root Execution:** Dedicated system user `appuser` (UID 1000) and group `appgroup` (GID 1000). Root privileges dropped via `USER appuser`.
- **System Libraries:** Minimal runtime dependencies (`libgl1`, `libglib2.0-0`, `curl`).
- **Port & Health Check:** `EXPOSE 8000`. Health probe queries `curl -f http://localhost:8000/health/live`.
- **Start Command:** `CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]`.

### 3.2. Frontend Dockerfile (`Dockerfile.frontend` and `docker/Dockerfile.frontend`)
- **Base Image:** `python:3.11-slim`.
- **Multi-Stage Build:** Isolated `/opt/venv` build stage; non-root `appuser` runtime stage.
- **Environment Flags:** `STREAMLIT_SERVER_HEADLESS=true`, `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`.
- **Port & Health Check:** `EXPOSE 8501`. Health probe queries `curl -f http://localhost:8501/_stcore/health`.
- **Start Command:** `CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]`.

### 3.3. Docker Compose Orchestration (`docker-compose.yml`)
- **Services:** Decoupled `backend` (FastAPI) and `frontend` (Streamlit). Superfluous databases are intentionally omitted.
- **Inter-Service Networking:** Dedicated bridge network `perceptguard-net` with automatic DNS resolution.
- **Readiness Dependency:** `frontend` defines `depends_on: backend: condition: service_healthy`, ensuring the UI does not launch until the backend passes its `/health/live` probe.
- **Resource Constraints (Deploy Limits):**
  - `backend`: limits `cpus: "2.0"`, `memory: 2048M`; reservations `cpus: "0.5"`, `memory: 512M`.
  - `frontend`: limits `cpus: "1.5"`, `memory: 1024M`; reservations `cpus: "0.25"`, `memory: 256M`.
  - Total limits ($3.5\text{ vCPU}$, $3\text{ GB RAM}$) fit safely within the host 8 GB RAM profile.

### 3.4. Environment & Secret Isolation (`.dockerignore`)
- Strict exclusion patterns prevent credentials, private raw videos, large model weights, and virtual environments from leaking into build contexts:
  - Excludes: `.git/`, `.env*` (except `.env.example`), `venv/`, `__pycache__/`, `datasets/raw/`, `models/*.pt`, `outputs/`, `logs/`.

### 3.5. Model & Dataset Storage Strategy
- Checkpoint weights (`./models`) and processed telemetry datasets (`./datasets/processed`) are mounted as read-only volumes (`:ro`).
- Application outputs and logs are persisted across container restarts using named volumes `backend_outputs` and `backend_logs`.
- If model checkpoints are missing, the backend runs in replay mode and flags `degraded` via readiness probes without crashing.

---

## 4. Static Validation & Automated Test Results

An automated test suite was engineered under `tests/deployment/` to validate all configuration and security requirements:

| Test Identifier | Scope | Status |
| :--- | :--- | :---: |
| `test_backend_dockerfile_structure_and_security` | Multi-stage builder, non-root user, Uvicorn command, CPU device, health probe | **PASS** |
| `test_frontend_dockerfile_structure_and_security` | Multi-stage builder, non-root user, Streamlit headless command, health probe | **PASS** |
| `test_docker_dir_mirrors_exist` | Verification of `docker/Dockerfile.backend` and `docker/Dockerfile.frontend` | **PASS** |
| `test_dockerignore_rules` | Security exclusion of `.env`, `.git/`, `venv/`, `datasets/raw/`, `models/*.pt` | **PASS** |
| `test_docker_compose_schema_and_services` | PyYAML parsing, service topology, health conditions, networks, resource limits | **PASS** |
| `test_referenced_source_directories_exist` | Validates host paths referenced in COPY/volumes (`src`, `dashboard`, `models`) | **PASS** |
| **Total Deployment Tests** | **Static Validation Suite** | **6 / 6 passed in 0.24s** |

### Complete Repository Regression Verification
The complete repository regression suite was executed:
- `tests/config/`: 18 tests passed
- `tests/api/`: 30 tests passed
- `tests/frontend/`: 31 tests passed
- `tests/deployment/`: 6 tests passed
- `tests/early_warning/`: 37 tests passed
- `tests/risk/`: 106 tests passed
- **Total: 228 / 228 passed (100%) in 13.25s.**

---

## 5. Deployment Provider Compatibility Assessment

1. **Render.com (Recommended for Cloud):**
   - Excellent fit for Topology B using Docker Web Services.
   - Requires paid tier ($7/month) for persistent RAM (> 512 MB).
2. **Railway.app (Recommended for Multi-Container):**
   - Native support for `docker-compose.yml` deployments with automatic private networking.
3. **Streamlit Community Cloud (Feasible for Demo):**
   - Supports Topology A (Unified container / direct Git repository launch).
4. **AWS ECS / Fargate (Enterprise Fleet Target):**
   - Serverless container orchestration with EFS persistent storage mounts and ALB ingress.
5. **Vercel (Not Recommended):**
   - Serverless edge architecture is incompatible with Streamlit persistent WebSocket requirements.

---

## 6. Limitations & Operational Directives

1. **Host Docker Execution Limitation:** The Docker daemon is not installed on this local Windows 11 development machine. Actual image compilation and container launch must be conducted on a CI/CD build runner (e.g. GitHub Actions `ubuntu-latest`) or host with Docker Engine installed.
2. **Vulnerability Scanning Status:** Marked as **NOT RUN** due to absence of local container security scanning CLI tools (e.g., Trivy, Snyk).
3. **Provisional Calibration Gate:** Multi-horizon risk scoring remains under `PROVISIONAL CALIBRATION` until multi-clip real-world driving sequences are evaluated.
4. **Task Completion Boundary:** Day 11 Task 5 is complete. No Day 11 Task 6 work has been initiated.
