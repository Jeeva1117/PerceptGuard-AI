# PerceptGuard AI — CI/CD Pipeline Architecture

**Module:** Continuous Integration & Delivery Architecture Specification  
**Orchestration Engine:** GitHub Actions  
**Target Environments:** GitHub Runners (CI) / Edge Workstation / Cloud Microservices  

---

## 1. Architectural Overview

The PerceptGuard AI CI/CD pipeline establishes automated, deterministic verification across all pull requests and pushes to the `main` branch. The architecture enforces zero-trust validation before any code reaches production, separating automated continuous integration from manual, authorized deployment.

```
                     [ Developer Commit / Pull Request to main ]
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │    GitHub Actions Workflow Trigger     │
                     │       (.github/workflows/ci.yml)       │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │   Job 1: Code Quality & Secret Audit   │
                     │  - Bytecode py_compile (Syntax check)  │
                     │  - Secret leakage guard (.env check)   │
                     │  - Docker Compose & YAML validation    │
                     └────────────────────┬───────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ Job 2: Backend   │            │ Job 3: Frontend  │            │ Job 4: Docker    │
│ CI (Py 3.11/12)  │            │ CI (Streamlit)   │            │ Validation       │
│ - PyTorch CPU    │            │ - Client tests   │            │ - Compose config │
│ - API & Risk     │            │ - UI components  │            │ - Backend build  │
│ - Early Warning  │            │ - State views    │            │ - Frontend build │
│ - 45+ API tests  │            │ - 31 UI tests    │            │ - Non-root check │
└─────────┬────────┘            └─────────┬────────┘            └─────────┬────────┘
          │                               │                               │
          └───────────────────────────────┼───────────────────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │     Job 5: Dependency Security Audit   │
                     │  - pip-audit vulnerability analysis    │
                     │  - PyPA advisory DB matching           │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │     Job 6: Deployment Readiness Gate   │
                     │  - Validates all prerequisite jobs     │
                     │  - Generates deployment certificate    │
                     └────────────────────────────────────────┘
```

---

## 2. Pipeline Trigger Policies & Path Filtering

- **Trigger Events:**
  - `push` to `main`: Runs full regression and container validation.
  - `pull_request` targeting `main`: Validates pull requests before merge.
  - `workflow_dispatch`: Enables manual on-demand execution.
- **Path-Based Filtering:**
  - Changes to documentation files (`docs/**`, `*.md`, `LICENSE`, `.gitignore`) are explicitly excluded from running expensive compute jobs, saving CI runner minutes.
- **Concurrency Control:**
  - `concurrency: group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true` automatically cancels obsolete pending runs when new commits are pushed.

---

## 3. Security Decisions & Boundary Safeguards

1. **Principle of Least Privilege:**
   - Default workflow permissions are explicitly restricted to `contents: read`.
2. **Zero Secret Exposure:**
   - Production API credentials, database keys, or certificates are NEVER passed as build arguments or environment variables in CI jobs.
   - The `code-quality` job scans the repository tree to prevent accidental commits of `.env` files.
3. **No Automatic Image Publication:**
   - CI builds Docker containers solely to verify compilation, dependency resolution, and non-root user configuration (`appuser`). Containers are **never** pushed to public registries during CI.
4. **No Automatic Production Deployments:**
   - Deployments are isolated in a separate, manually triggered workflow (`deploy.yml`) requiring explicit approval tokens.
