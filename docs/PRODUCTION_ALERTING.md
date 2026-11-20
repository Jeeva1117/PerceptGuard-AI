# PerceptGuard AI — Production Operational Alerting Specification

**Module:** SRE Incident Management & Operational Alerting Rules  
**Scope:** Infrastructure, Process Reliability, and API Health Alerts  

> [!IMPORTANT]
> **Operational Alerts vs. ML Early-Warning Alerts:**  
> This specification defines **SRE Operational Alerts** (infrastructure health, container crashes, API timeouts, memory leaks).  
> These are entirely distinct from PerceptGuard AI's **ML Early-Warning Alerts** (which detect computer vision perception degradation, occlusion, and future IoU failure).

---

## 1. Operational Alert Catalog

| Alert Identifier | Severity | Component | Trigger Condition | Expected Response / Runbook | Recovery Condition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`BACKEND_LIVENESS_FAILED`** | **P1 — Critical** | Backend API | `/health/live` fails or times out for 3 consecutive probes ($90\text{s}$). | Container orchestrator initiates automatic restart. SRE inspects crash logs. | `/health/live` returns HTTP 200 `alive`. |
| **`BACKEND_READINESS_UNREADY`** | **P1 — Critical** | Backend API | `/health/ready` returns HTTP 503 `unready` for $> 60\text{s}$. | Halt ingress traffic. Verify `dashboard_service` initialization and file permissions. | Subsystem returns `ready` or `degraded`. |
| **`MEMORY_PRESSURE_BREACH`** | **P2 — Warning** | Backend / Host | `memory_rss_mb` $> 1800\text{ MB}$ (90% of 2 GB limit). *(PROVISIONAL)* | Inspect video processing workers for unreleased `cv2.VideoCapture` descriptors. | Memory usage drops below $1400\text{ MB}$. |
| **`API_HIGH_ERROR_RATE`** | **P2 — Warning** | API Gateway | 5xx HTTP response rate $> 5\%$ over a 5-minute rolling window. | Inspect server logs filtered by `status=500` and correlated `X-Request-ID`. | Error rate drops below $1\%$ for 5 minutes. |
| **`CHECKPOINT_MOUNT_MISSING`** | **P3 — Info / Notice** | Model Storage | Subsystem status reports `lstm_checkpoint: UNAVAILABLE`. | Verify host Docker volume mount `./models:/app/models:ro`. System runs in replay mode. | Model checkpoint mounted and recognized. |
| **`HIGH_REQUEST_LATENCY`** | **P3 — Warning** | API Endpoints | 95th percentile latency $> 500\text{ ms}$ over 5 minutes. *(PROVISIONAL)* | Check CPU throttling or large concurrent video upload processing. | Latency p95 drops below $200\text{ ms}$. |

---

## 2. Severity Classification Framework

- **P1 — Critical:** Outage or total service disruption. Requires immediate automated container restart or on-call page.
- **P2 — Warning:** Service operational but degraded or approaching resource limits. Automatic ticket logged; engineer investigates within business hours.
- **P3 — Info / Notice:** Informational anomaly or fallback mode active. Logged to observability dashboard without paging engineers.

---

## 3. Incident Escalation Runbook

1. **Triage:** Extract `X-Request-ID` from alerting payload.
2. **Log Correlation:** Query structured JSON access logs:
   ```bash
   grep '"request_id":"<ID>"' /app/logs/*.log
   ```
3. **Health Probe Audit:**
   ```bash
   curl -i http://localhost:8000/health/ready
   curl -i http://localhost:8000/health/metrics
   ```
4. **Resolution:** If memory or CPU is locked, trigger rolling container restart:
   ```bash
   docker compose restart backend
   ```
