# PerceptGuard AI — Operational Monitoring & SRE Telemetry

**Module:** Site Reliability Engineering (SRE) & Operational Monitoring  
**Target:** Application Health, Process Diagnostics, and Incident Detection  

---

## 1. System Health Monitoring Architecture

PerceptGuard AI exposes a multi-tiered diagnostic monitoring interface engineered for cloud orchestrators (Kubernetes, Docker Swarm), load balancers, and SRE observability platforms:

| Endpoint Route | Monitoring Role | Expected Response | Failure Meaning |
| :--- | :--- | :--- | :--- |
| **`GET /health/live`** | Liveness Probe | `200 OK {"status": "alive"}` | Deadlock, event loop freeze, or process termination. Orchestrator must restart container. |
| **`GET /health/ready`** | Readiness Probe | `200 OK {"status": "ready"}` or `{"status": "degraded"}` | Subsystem failure (service uninitialized). Ingress must halt traffic. |
| **`GET /health/metrics`** | Process Telemetry | `200 OK {memory_rss_mb, cpu_percent, uptime_seconds}` | Process metrics unavailable. SRE scraping endpoint. |
| **`GET /health`** | Domain Health | `200 OK {health_score, pipeline_status}` | Perception pipeline failure or severe perception degradation. |

---

## 2. Process Telemetry Endpoint (`/health/metrics`)

The `/health/metrics` endpoint provides lightweight, non-blocking process telemetry without invoking machine learning inference models:

```json
{
  "service": "perceptguard-backend",
  "uptime_seconds": 1824.52,
  "memory_rss_mb": 142.35,
  "memory_vms_mb": 418.12,
  "cpu_percent": 1.2,
  "num_threads": 8,
  "status": "HEALTHY",
  "timestamp": "2026-11-20T19:15:27.060991+00:00"
}
```

### Metrics Collected
- **`uptime_seconds`**: Continuous duration since service start. Detects crash-looping containers.
- **`memory_rss_mb`**: Physical RAM allocated to the Python process. Guards against memory leaks in OpenCV decoding.
- **`cpu_percent`**: Real-time CPU utilization across cores.
- **`num_threads`**: Active worker threads. Detects thread leaks in asynchronous task queues.

---

## 3. Operational Logging & Request Tracing

Structured JSON logging is handled by `RequestLoggingMiddleware`:
- **`X-Request-ID`**: Propagated across request headers, response headers, and log lines for distributed tracing.
- **Duration Tracking**: Request execution time measured in milliseconds using high-precision `time.perf_counter()`.
- **Sensitive Header Scrubbing**: Authorization tokens, cookies, and passwords are never recorded in application logs.
- **Sanitized Tracebacks**: Internal 500 server errors write full tracebacks to server logs tagged with `request_id`, while returning generic sanitized error envelopes to client callers.
