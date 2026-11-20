# Frontend API Integration Specification

**PerceptGuard AI — Perception Failure Early Warning System**  
**Module:** Frontend REST Client & Transport Orchestration  
**Component:** `dashboard/api_client.py`  

---

## 1. Overview & Dual Topology Support

`DashboardApiClient` provides a unified data access layer for the PerceptGuard AI frontend, abstracting network communications and transparently supporting two distinct deployment topologies:

| Topology | Description | Data Transport | Use Case |
| :--- | :--- | :--- | :--- |
| **Topology A (Unified)** | Single standalone container or local workstation. | In-Process direct invocation via `DashboardService`. | Local execution, air-gapped environments, debugging. |
| **Topology B (Decoupled)** | Frontend container talking to backend container. | HTTP REST via `httpx.Client` (`/api/v1/*`). | Kubernetes clusters, distributed production cloud deployments. |

---

## 2. Dynamic Backend Probing & Seamless Fallback

Upon startup and manual refresh, `DashboardApiClient` evaluates backend reachability via `client.probe_backend()`:
- **Primary Transport:** If `/health/live` responds with HTTP 200 within 1.5s, the client enters `REST_API` mode.
- **Fail-Safe Fallback:** If the HTTP REST endpoint is offline, times out, or raises a network exception, the client automatically switches to `IN_PROCESS` mode, invoking the verified `DashboardService` singleton without dropping UI functionality or crashing.

```
       ┌────────────────────────┐
       │   Streamlit Dashboard  │
       │    (dashboard/app.py)  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   DashboardApiClient   │
       └─────┬────────────┬─────┘
             │            │
  HTTP OK    │            │ Unreachable / Offline
 (REST_API)  ▼            ▼ (IN_PROCESS)
┌─────────────────┐  ┌──────────────────────┐
│  FastAPI Backend │  │   DashboardService   │
│  (src/api/*)    │  │ (src/early_warning/*)│
└─────────────────┘  └──────────────────────┘
```

---

## 3. Supported API Endpoints & Methods

All client methods normalize outputs to ensure identical schemas regardless of transport mode:

| Client Method | HTTP Route | In-Process Method | Description |
| :--- | :--- | :--- | :--- |
| `get_system_health()` | `GET /api/v1/health` | `service.get_system_health()` | Operational pipeline status, overall health, model availability. |
| `get_pipeline_status()` | `GET /api/v1/telemetry/pipeline-status` | `service.get_pipeline_status()` | Detailed status across all 8 perception pipeline stages. |
| `get_risk_overview(...)` | `GET /api/v1/telemetry/risk-overview` | `service.get_reliability_risk_overview(...)` | Multi-horizon risk scores (H1, H3, H5), trends, rolling means. |
| `get_temporal_monitoring(...)` | `GET /api/v1/telemetry/temporal-monitoring` | `service.get_temporal_monitoring_data(...)` | Historical risk trajectories and state transitions table. |
| `get_alerts(...)` | `GET /api/v1/alerts` | `service.get_alerts(...)` | Paginated, filtered early-warning alerts. |
| `get_alert_detail(id)` | `GET /api/v1/alerts/{id}` | `service.get_alert_detail(id)` | Full incident metadata, rule rationale, and audit history. |
| `acknowledge_alert(id)` | `POST /api/v1/alerts/{id}/acknowledge` | `service.acknowledge_alert(id)` | Transitions alert from ACTIVE to ACKNOWLEDGED. |
| `resolve_alert(id, reason)` | `POST /api/v1/alerts/{id}/resolve` | `service.resolve_alert(id, reason)` | Transitions alert to RESOLVED with required operational reason. |
| `get_detection_tracking(...)`| `GET /api/v1/telemetry/detection-tracking` | `service.get_detection_tracking_context(...)` | Authentic YOLO11n + ByteTrack metrics. |

---

## 4. Connection Lifecycle & Teardown

- **Connection Reuse:** The client maintains an internal `httpx.Client` connection pool.
- **Graceful Cleanup:** `client.close()` safely disposes of underlying connection pools while avoiding double-closure of externally injected test clients.
