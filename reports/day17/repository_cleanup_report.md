# PerceptGuard AI — Day 17 Repository Cleanup & Consolidation Report

## 1. Executive Summary

- **Milestone:** Day 17 — Repository Cleanup & Full Runtime Stabilization
- **Audit Date:** 2026-11-21
- **Focus:** Safe repository cleanup, elimination of obsolete artifacts, consolidation of duplicate implementations, and verification against active test suites and entry points.

---

## 2. File Count Metrics

| Category | Initial Count (Day 17 Start) | Final Count | Net Change |
| :--- | :---: | :---: | :---: |
| **Git Tracked Files** | 493 | 493 (+ new Day 17 deliverables) | Clean |
| **Root Markdown Files** | 2 (`README.md`, `CONTRIBUTING.md`) | 2 | 0 |
| **Docs Markdown Files** | 51 (7 core suite + 44 audit spec refs) | 51 | 0 |
| **Historical Report Files** | 49 | 52 (+ Day 17 reports) | +3 |
| **Scratch Files** | 1 (`scratch/.gitkeep`) | 1 (`scratch/.gitkeep`) | 0 |

---

## 3. Duplicate Candidates & Reference Analysis

### Candidate Group 1: Dockerfiles
- **Files:** `Dockerfile` vs. `docker/Dockerfile.backend` (and `Dockerfile.frontend` vs. `docker/Dockerfile.frontend`).
- **Inspection Findings:** Files have identical byte contents.
- **Dependency & Reference Check:**
  - `tests/deployment/test_docker_configuration.py` lines 18 and 49 assert the existence of root `Dockerfile` and `Dockerfile.frontend`.
  - `tests/deployment/test_docker_configuration.py` lines 74 and 75 assert the existence of `docker/Dockerfile.backend` and `docker/Dockerfile.frontend`.
  - `.github/workflows/ci.yml` builds using root `-f Dockerfile` and `-f Dockerfile.frontend`.
- **Verdict:** **RETAIN BOTH LOCATIONS**. Removing either location breaks regression tests and CI workflows.

### Candidate Group 2: Dataset Validation Reports
- **Files:** `datasets/processed/driving_sample/validation_report.json` vs. `outputs/reports/dataset_validation_report.json`.
- **Inspection Findings:** Both files record dataset metadata.
- **Dependency & Reference Check:**
  - `scripts/audit_project.py:580` strictly checks for `datasets/processed/driving_sample/validation_report.json`.
- **Verdict:** **RETAIN AUTHORITATIVE DATASET COPY** in `datasets/processed/driving_sample/`. `outputs/reports/` is gitignored.

### Candidate Group 3: Frontend Structure
- **Candidate:** Potential duplicate frontend implementations between `dashboard/` and a potential `src/frontend/`.
- **Inspection Findings:** Checked `src/frontend/` — it does not exist. `dashboard/app.py` is the single, authoritative frontend entry point across all Docker configurations, CI/CD, documentation, and `tests/frontend/`.
- **Verdict:** **NO CONFLICT**. `dashboard/` is the authoritative frontend package.

---

## 4. Retained Critical Files & Rationale

1. **Model Weights & Checkpoints:**
   - `models/detection/yolo11n.pt` ($5.35\text{ MB}$): Pretrained YOLO11 nano model for CPU detection.
   - `models/forecasting/model.pt` ($62.6\text{ KB}$): Trained multi-horizon PyTorch LSTM forecaster checkpoint.
2. **Evaluation Video Clip:**
   - `outputs/day14_validation/clips/scenario_a_normal_daytime.mp4` ($1.53\text{ MB}$): Authentic BDD100K 720p 30fps driving clip required for real-world pipeline validation.
3. **Historical Audit Reports:**
   - `reports/day5_audit/` through `reports/day16/`: Retained to maintain full compliance history and audit continuity.
4. **Docs Technical Reference Suite:**
   - 44 Day 1–6 technical specifications in `docs/` retained to ensure `scripts/audit_project.py` and `tests/test_audit_project.py` pass 100%.

---

## 5. Summary of Cleanup Actions

- Duplicate candidates systematically audited with reference tracking prior to deletion.
- Zero required modules deleted.
- Git working tree maintained at clean status with zero untracked artifacts or leaks.
