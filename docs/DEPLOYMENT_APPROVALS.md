# PerceptGuard AI — Production Deployment Approval Boundaries & Safeguards

**Module:** Release Engineering & Deployment Governance  
**Target:** Production Release Gates, Human-in-the-Loop Safeguards, and Rollback Protocols  

---

## 1. Separation of CI Validation from CD Deployment

PerceptGuard AI enforces strict architectural separation between continuous integration testing and production continuous deployment:

```
┌──────────────────────────────────────────────┐
│           Continuous Integration (CI)        │
│          Trigger: Push / PR to main          │
│                                              │
│  - Static Code Quality & Secret Checks       │
│  - Automated Backend Test Suites             │
│  - Automated Frontend UI Tests               │
│  - Docker Container Build Validation         │
│  - Dependency Security Scanning (pip-audit)  │
│                                              │
│  Output: Validated Code & Build Artifacts    │
│  (ZERO AUTOMATIC DEPLOYMENT)                 │
└──────────────────────┬───────────────────────┘
                       │
                       ▼  [ Passes All CI Gates ]
┌──────────────────────────────────────────────┐
│        Authorized Deployment Gate (CD)       │
│      Trigger: Manual workflow_dispatch ONLY  │
│                                              │
│  - Requires protected GitHub Environment     │
│  - Requires designated Approver Review       │
│  - Requires confirmation token 'DEPLOY'      │
│  - Restricts execution strictly to 'main'    │
└──────────────────────────────────────────────┘
```

---

## 2. Mandatory Production Safeguards

1. **Manual Workflow Dispatch Only:**
   - Production deployment cannot be triggered by a Git commit, pull request merge, or webhook.
   - The `.github/workflows/deploy.yml` workflow requires manual initiation via GitHub Actions UI or authorized GitHub CLI.
2. **Explicit Confirmation Token:**
   - The deployment workflow enforces an input parameter `confirmation`. The release job will immediately abort unless the operator enters the exact token string: `DEPLOY`.
3. **Branch Enforcement:**
   - Releases are restricted to commits on the official `main` branch (`if: github.ref == 'refs/heads/main'`). Feature branches or pull requests cannot trigger production deployments.
4. **Zero Embedded Secrets:**
   - Production secrets (API keys, TLS certificates, database credentials) are stored exclusively in GitHub Actions Encrypted Secrets (`Settings > Secrets and variables > Actions`) and injected at runtime.

---

## 3. Post-Deployment Verification & Health Probe Gate

Immediately following any container rollout, the deployment orchestrator executes a multi-step verification sequence:

```bash
# 1. Verify container process is alive (Liveness Probe)
curl -f -m 5 https://api.perceptguard.ai/health/live || exit 1

# 2. Verify all models and subsystems are initialized (Readiness Probe)
curl -f -m 10 https://api.perceptguard.ai/health/ready || exit 1

# 3. Verify process memory and CPU are within bounds
curl -f -m 5 https://api.perceptguard.ai/health/metrics || exit 1
```

If any probe fails within a 90-second timeout window, the automated rollback protocol is initiated.

---

## 4. Emergency Rollback Protocol

If a production regression or degradation is detected post-deployment:
1. **Instant Container Rollback:**
   ```bash
   # Re-tag and launch previous stable container release
   docker compose down
   docker tag perceptguard-backend:previous perceptguard-backend:latest
   docker tag perceptguard-frontend:previous perceptguard-frontend:latest
   docker compose up -d
   ```
2. **Git Revert:**
   ```bash
   git revert -m 1 <merge_commit_sha>
   git push origin main
   ```
