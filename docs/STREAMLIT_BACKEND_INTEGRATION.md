# Streamlit & FastAPI Backend Integration Guide
## PerceptGuard AI: Dual-Topology Architecture

---

## 1. Integration Topologies

The dashboard supports two deployment topologies managed seamlessly by `DashboardApiClient`:

```
Mode 1: REST API Transport (Decoupled Services)
[ Streamlit UI : 8501 ] ─── HTTP / REST ───> [ FastAPI Backend : 8000 ]

Mode 2: In-Process Fallback (Single Standalone Process)
[ Streamlit UI : 8501 ] ─── In-Memory ────> [ PipelineIntegrator & JobManager ]
```

When Streamlit starts, `DashboardApiClient.probe_backend()` probes `GET /health/live`. If reachable, it uses HTTP REST transport. If unreachable or during standalone local execution, it falls back to in-process invocation without crashing.

---

## 2. API Endpoints Used

| Endpoint | HTTP Method | Purpose |
| :--- | :---: | :--- |
| `/health/live` | GET | Liveness probe confirming service responsiveness. |
| `/health/ready` | GET | Readiness probe inspecting checkpoints and dependencies. |
| `/health/metrics` | GET | Process telemetry (CPU %, RAM RSS, uptime). |
| `/api/v1/telemetry/risk-overview` | GET | Instantaneous multi-horizon risk scores ($H1, H3, H5$). |
| `/api/v1/telemetry/temporal-monitoring` | GET | Historical risk trajectory and trend velocity. |
| `/api/v1/telemetry/detection-tracking` | GET | ByteTrack identities and detection summaries. |
| `/api/v1/alerts/active` | GET | Active early-warning incident notifications. |
| `/api/v1/alerts/{alert_id}/acknowledge` | POST | Operator alert acknowledgment. |
| `/api/v1/alerts/{alert_id}/resolve` | POST | Operator incident resolution. |
| `/api/v1/video/validate` | POST | Header validation and format verification. |
| `/api/v1/video/jobs` | POST | Dispatches video for background processing. |
| `/api/v1/video/jobs/{job_id}` | GET | Live stage and percentage progress polling. |
| `/api/v1/video/jobs/{job_id}/results` | GET | Full pipeline results and frame telemetry. |
| `/api/v1/video/jobs/{job_id}/frames/{frame}` | GET | Serves annotated frame images for visualization. |
