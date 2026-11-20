# PerceptGuard AI — GitHub Actions Configuration & Setup Guide

**Module:** GitHub Actions Workflows & CI Runner Setup  
**Files:** `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`  

---

## 1. Workflow Catalog

The repository maintains two decoupled GitHub Actions workflow files:

| Workflow File | Purpose | Trigger | Execution Environment |
| :--- | :--- | :--- | :--- |
| **`.github/workflows/ci.yml`** | Continuous Integration, test execution, Docker validation, security scanning. | Pushes & PRs to `main` | `ubuntu-latest` |
| **`.github/workflows/deploy.yml`** | Production release gate with manual human authorization. | Manual `workflow_dispatch` | `ubuntu-latest` (protected environment) |

---

## 2. Job-by-Job Specification (`ci.yml`)

### Job 1: `code-quality`
- **Runner:** `ubuntu-latest`, Python 3.11
- **Tasks:**
  - Compiles all Python source files via `python -m py_compile` to catch syntax regressions.
  - Scans tracked files to guarantee no `.env` or sensitive credential files are committed.
  - Verifies YAML syntax across all Compose and configuration manifests.

### Job 2: `backend-ci`
- **Runner:** `ubuntu-latest`
- **Matrix Strategy:** Python `3.11` and `3.12`
- **Tasks:**
  - Installs CPU-only PyTorch distribution (`--extra-index-url https://download.pytorch.org/whl/cpu`).
  - Caches pip packages using `actions/setup-python@v5`.
  - Runs full backend test suites (`tests/config/`, `tests/api/`, `tests/risk/`, `tests/early_warning/`, `tests/deployment/`).
  - Uploads test cache artifacts with a 7-day retention period.

### Job 3: `frontend-ci`
- **Runner:** `ubuntu-latest`, Python 3.11
- **Tasks:**
  - Installs dependencies including Streamlit, HTTPX, and Pandas.
  - Executes frontend test suite (`tests/frontend/`).
  - Validates headless Streamlit compilation and public component exports.

### Job 4: `docker-validation`
- **Runner:** `ubuntu-latest` (Docker daemon available on runner)
- **Tasks:**
  - Validates `docker compose config`.
  - Builds `perceptguard-backend:ci` and `perceptguard-frontend:ci`.
  - Asserts that containers run with unprivileged UID `1000` (`appuser`).

### Job 5: `security-scan`
- **Runner:** `ubuntu-latest`, Python 3.11
- **Tasks:**
  - Runs `pip-audit` to detect known vulnerabilities in pinned dependencies against the PyPA database.

### Job 6: `readiness-gate`
- **Tasks:**
  - Aggregates status across all 5 verification jobs.
  - Emits final deployment readiness decision.

---

## 3. Local CI Workflow Emulation

To emulate GitHub Actions checks locally before pushing to remote:
```bash
# 1. Syntax & bytecode check
python -m py_compile $(git ls-files '*.py')

# 2. Run backend and deployment tests
pytest tests/config/ tests/api/ tests/risk/ tests/early_warning/ tests/deployment/ -v

# 3. Run frontend tests
pytest tests/frontend/ -v

# 4. Verify Docker Compose YAML
python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"
```
