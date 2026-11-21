# PerceptGuard AI — Dashboard API Integration & Architecture

## 1. Overview

The PerceptGuard AI dashboard communicates with backend services through a dedicated adapter class: `dashboard.api_client.DashboardApiClient`. This client is engineered to handle production edge-cases including backend downtime, high network latency, transient dropped frames, and unscheduled server restarts.

---

## 2. Dual-Transport Operation

```
+-------------------------------------------------------------------+
|                        DashboardApiClient                         |
+-------------------------------------------------------------------+
                  |                                   |
                  | (Primary: HTTP / JSON)            | (Fallback: Direct Python)
                  v                                   v
+-----------------------------------+   +---------------------------+
|          FastAPI Server           |   |     DashboardService      |
|    http://127.0.0.1:8000/api/v1   |   |   (In-Process Engine)     |
+-----------------------------------+   +---------------------------+
```

### 2.1 Primary Mode: REST API (`REST_API`)
- Uses `httpx.Client` with configurable timeout (`settings.dashboard.api_request_timeout_seconds`, default: 10.0s).
- Emits standard JSON requests to `/api/v1` routes.
- Includes client identification headers (`User-Agent: PerceptGuard-UI/{version}`).

### 2.2 Fallback Mode: In-Process Service (`IN_PROCESS`)
- If the HTTP backend raises `httpx.ConnectError`, `httpx.NetworkError`, or `httpx.TimeoutException`, the client automatically transitions to `IN_PROCESS` mode.
- In-process operations invoke `src.early_warning.dashboard_service.DashboardService` and `src.api.job_manager.JobManager` directly within the Streamlit process memory space.
- Zero UI freezing, zero white screens, zero unhandled Python exceptions.
- The sidebar indicator visually signals the transport state with color badges:
  - Green (`#2a9d8f`): `REST_API` (Backend online)
  - Orange (`#e76f51`): `IN_PROCESS` (In-process fallback active)

---

## 3. API Endpoint Mapping

| Functional Area | HTTP Method | Endpoint Route | In-Process Fallback Method |
| :--- | :--- | :--- | :--- |
| **System Health** | `GET` | `/health` | `DashboardService.get_system_health()` |
| **Pipeline Status**| `GET` | `/telemetry/pipeline-status` | `DashboardService.get_pipeline_status()` |
| **Risk Overview** | `GET` | `/telemetry/risk-overview` | `DashboardService.get_reliability_risk_overview()` |
| **Temporal Data** | `GET` | `/telemetry/temporal-monitoring` | `DashboardService.get_temporal_monitoring_data()` |
| **Alerts Feed** | `GET` | `/alerts` | `DashboardService.get_alerts()` |
| **Alert Detail** | `GET` | `/alerts/{alert_id}` | `DashboardService.get_alert_detail()` |
| **Acknowledge** | `POST` | `/alerts/{alert_id}/acknowledge` | `DashboardService.acknowledge_alert()` |
| **Resolve Alert** | `POST` | `/alerts/{alert_id}/resolve` | `DashboardService.resolve_alert()` |
| **Tracking Info** | `GET` | `/telemetry/detection-tracking` | `DashboardService.get_detection_tracking_context()` |
| **Validate Video**| `POST` | `/video/validate` | `Direct format & header inspection` |
| **Submit Job** | `POST` | `/video/jobs` | `JobManager.submit_job()` |
| **Job Status** | `GET` | `/video/jobs/{job_id}` | `JobManager.get_job()` |
| **Job Results** | `GET` | `/video/jobs/{job_id}/results` | `JobManager.get_job_results()` |
| **List Jobs** | `GET` | `/video/jobs` | `JobManager.list_jobs()` |

---

## 4. Resilience, Staleness, and Error Boundaries

### 4.1 Stale Data Handling
- Every response timestamp is compared against the system clock.
- If data age exceeds `settings.dashboard.telemetry_freshness_ttl_seconds` (default: 30s), a visual warning banner is rendered via `render_stale_warning()`:
  > ⚠️ **Telemetry Staleness Notice:** Telemetry was last synchronized 42s ago (TTL: 30s). Displayed metrics may not reflect current frame state.

### 4.2 Error Boundaries
- All component calls are wrapped in `try/except FrontendAPIError` blocks.
- If an unrecoverable communication error occurs, `render_error_boundary()` displays:
  1. A clear user-friendly error title.
  2. The sanitized technical cause (omitting sensitive server stack traces).
  3. An interactive **"Retry Connection"** button connected to `st.rerun`.

### 4.3 Null-Safety Contracts
- Missing or `None` risk scores are rendered as `"N/A"` with tier `"UNKNOWN"`.
- Under no circumstance are missing scores coerced to `0.0` or `LOW` risk.
- Division by zero protection is strictly enforced across FPS, latency, and duration metrics using `max(1, count)`.
