# Health Monitoring & Diagnostic Probes Specification

## 1. Overview & Operational Goals

PerceptGuard AI exposes dedicated health and diagnostic endpoints designed for container orchestrators (Kubernetes, Docker Swarm, Nomad) and edge monitoring agents. These endpoints provide unambiguous signals regarding process liveness, dependency readiness, and perception pipeline health without executing expensive ML inferences or blocking the event loop.

---

## 2. Probe Endpoints Specification

### 2.1. Liveness Probe (`GET /health/live` and `GET /api/v1/health/live`)
- **Primary Purpose:** Verifies that the Python process and asyncio event loop are responsive.
- **HTTP Status:** Always **200 OK** if the process can serve HTTP traffic.
- **Payload:**
  ```json
  {
    "status": "alive",
    "timestamp": "2026-11-20T17:45:00.123456+00:00"
  }
  ```
- **Failure Condition:** Process hang, deadlock, or event-loop starvation results in HTTP timeout, prompting orchestrator container restart.

---

### 2.2. Readiness Probe (`GET /health/ready` and `GET /api/v1/health/ready`)
- **Primary Purpose:** Verifies that operational dependencies (dashboard service, model checkpoints, prediction artifacts) are available before routing traffic.
- **HTTP Status Rules:**
  - **200 OK (`ready`):** Core service initialized, prediction artifact mounted, checkpoints verified.
  - **200 OK (`degraded`):** Core service initialized in zero-state replay mode; optional model checkpoints pending volume mount.
  - **503 Service Unavailable (`unready`):** Core `DashboardService` or `PipelineIntegrator` failed initialization.
- **Payload Example:**
  ```json
  {
    "status": "ready",
    "subsystems": {
      "dashboard_service": "READY",
      "yolo_checkpoint": "READY",
      "lstm_checkpoint": "READY",
      "predictions_artifact": "READY"
    },
    "details": "PerceptGuard AI backend ready for operational inference and telemetry queries.",
    "timestamp": "2026-11-20T17:45:00.123456+00:00"
  }
  ```

---

### 2.3. Comprehensive Operational Health (`GET /health` and `GET /api/v1/health`)
- **Primary Purpose:** Delivers system telemetry, overall perception pipeline health score, current sequence context, and model availability.
- **HTTP Status:** **200 OK**.
- **Payload Example:**
  ```json
  {
    "project_name": "PerceptGuard AI",
    "pipeline_status": "READY",
    "overall_health": "NOMINAL",
    "health_score": 98.5,
    "current_sequence_id": "seq_driving_sample_L5_000001",
    "current_video_id": "driving_sample",
    "last_prediction_timestamp_seconds": 1.7946,
    "data_freshness": "CURRENT",
    "model_availability": {
      "lstm_forecaster": "AVAILABLE",
      "yolo_detector": "AVAILABLE",
      "bytetrack_tracker": "AVAILABLE"
    },
    "calibration_gate_status": "PROVISIONAL",
    "timestamp": "2026-11-20T17:45:00.123456+00:00"
  }
  ```

---

## 3. Kubernetes Manifest Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 2

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
  timeoutSeconds: 3
```
