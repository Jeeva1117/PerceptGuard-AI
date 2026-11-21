# PerceptGuard AI — Backend API Reference & Data Contracts

## 1. Overview & Base URL

The PerceptGuard AI backend is built on **FastAPI** and provides high-performance asynchronous REST endpoints for perception reliability telemetry, video job management, and early-warning alert handling.

- **Base URL:** `http://127.0.0.1:8000`
- **API Prefix:** `/api/v1`
- **Interactive OpenAPI Documentation:** `http://127.0.0.1:8000/docs` (Swagger UI)
- **Alternative Documentation:** `http://127.0.0.1:8000/redoc` (ReDoc)

---

## 2. Authentication & Headers

- In local edge operations, the API is configured without complex OAuth tokens for maximum execution speed and zero-friction deployment.
- **Client Header:** The frontend client transmits: `User-Agent: PerceptGuard-UI/{version}`
- **Content-Type:** `application/json` for standard requests; `multipart/form-data` for video uploads.

---

## 3. Endpoints Catalog

### 3.1 System Health & Service Info

#### `GET /`
Returns service identification, operational environment, and calibration gate status.
```json
{
  "name": "PerceptGuard-AI",
  "version": "1.0.0",
  "environment": "development",
  "api_prefix": "/api/v1",
  "status": "ONLINE",
  "calibration_gate": "PROVISIONAL CALIBRATION"
}
```

#### `GET /health` or `GET /api/v1/health`
Returns comprehensive subsystem operational health, uptime, and model availability.
```json
{
  "status": "HEALTHY",
  "timestamp": "2026-11-21T08:00:00Z",
  "overall_health": "OPTIMAL",
  "pipeline_status": "OPERATIONAL",
  "health_score": 0.98,
  "data_freshness": "CURRENT",
  "current_sequence_id": "driving_sample",
  "last_prediction_timestamp_seconds": 12.5,
  "model_availability": {
    "yolo_detector": true,
    "lstm_forecaster": true
  }
}
```

---

### 3.2 Telemetry Endpoints

#### `GET /api/v1/telemetry/pipeline-status`
Returns execution status across all 8 perception pipeline stages.

#### `GET /api/v1/telemetry/risk-overview`
Returns current multi-horizon failure risk ($S = 1.0 - R$) for $H1, H3, H5$:
```json
{
  "sequence_id": "driving_sample",
  "horizons": {
    "h1": {
      "horizon": 1,
      "risk_score": 0.1500,
      "risk_level": "LOW",
      "trend_direction": "DECREASING",
      "rolling_mean": 0.1600,
      "volatility": 0.0200,
      "observation_count": 30
    },
    "h3": {
      "horizon": 3,
      "risk_score": 0.4500,
      "risk_level": "MEDIUM",
      "trend_direction": "STABLE",
      "rolling_mean": 0.4400,
      "volatility": 0.0400,
      "observation_count": 30
    },
    "h5": {
      "horizon": 5,
      "risk_score": 0.7200,
      "risk_level": "HIGH",
      "trend_direction": "INCREASING",
      "rolling_mean": 0.6900,
      "volatility": 0.0700,
      "observation_count": 30
    }
  }
}
```

#### `GET /api/v1/telemetry/temporal-monitoring`
Fetches historical multi-horizon risk time series. Query params:
- `limit` (int, default: 50): Number of temporal steps.
- `sequence_id` (str, optional): Target sequence identifier.

#### `GET /api/v1/telemetry/detection-tracking`
Returns YOLO11n bounding box statistics and ByteTrack Kalman tracking continuity data:
```json
{
  "sequence_id": "driving_sample",
  "status": "AVAILABLE",
  "model_name": "YOLO11n",
  "tracker_type": "ByteTrack",
  "total_frames_processed": 150,
  "total_detections": 820,
  "total_unique_tracks": 42,
  "max_simultaneous_active_tracks": 14,
  "mean_confidence_overall": 0.885,
  "mean_processing_latency_ms": 18.4
}
```

---

### 3.3 Early Warning Alerts Endpoints

#### `GET /api/v1/alerts`
Fetches filtered and paginated incident notifications. Query params:
- `severity` (`ALL`, `CRITICAL`, `WARNING`, `CAUTION`, `INFO`)
- `horizon` (int: 1, 3, or 5)
- `alert_type` (str, optional)
- `status` (`ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`, `EXPIRED`)
- `page` (int, default: 1)
- `page_size` (int, default: 20)

#### `POST /api/v1/alerts/{alert_id}/acknowledge`
Acknowledges an active alert. Payload: `{"operator_id": "string"}`.

#### `POST /api/v1/alerts/{alert_id}/resolve`
Resolves an active alert with justification. Payload:
```json
{
  "operator_id": "operator_name",
  "reason": "Visual obstruction cleared; sensor nominal."
}
```

---

### 3.4 Video Processing Endpoints

#### `POST /api/v1/video/validate`
Uploads a video file to validate format, dimensions, frame count, and codec headers. Form-data param: `file`.

#### `POST /api/v1/video/jobs`
Submits a video for asynchronous pipeline execution. Form-data params:
- `file`: Video binary stream.
- `max_frames` (int, default: 30): Frame limit for processing.
Response:
```json
{
  "job_id": "3a8f10bc-9d04-4c81-9b12-ec98dfa12345",
  "filename": "driving_video.mp4",
  "status": "QUEUED",
  "stage": "QUEUED",
  "message": "Job submitted and queued for execution.",
  "max_frames": 30,
  "created_at": "2026-11-21T08:00:00Z"
}
```

#### `GET /api/v1/video/jobs/{job_id}`
Returns execution state, stage, progress %, and summary metrics for an active or completed job.

#### `GET /api/v1/video/jobs/{job_id}/results`
Retrieves full structured RFC 8259-compliant `results.json` artifact containing per-frame bounding boxes, 27-feature vectors, risk scores, and alert triggers.

#### `GET /api/v1/video/jobs`
Lists recently submitted video processing jobs with pagination (`limit=20`).

#### `GET /api/v1/video/jobs/{job_id}/frames/{frame_name}`
Streams an annotated JPEG frame image with overlaid bounding boxes and tracking IDs.

---

## 4. Error Handling & Status Codes

| HTTP Status | Exception Class | Description |
| :--- | :--- | :--- |
| **`400 Bad Request`** | `InvalidInputError` | Malformed parameters or invalid payload schema |
| **`404 Not Found`** | `ResourceNotFoundError` | Job ID or Alert ID does not exist |
| **`415 Unsupported Media`**| `InvalidVideoError` | Unrecognized video codec or unreadable file |
| **`422 Unprocessable`** | `ValidationError` | Pydantic model validation failure |
| **`500 Internal Error`** | `InternalPipelineError` | Unhandled runtime exception (stack trace sanitized) |
| **`503 Unavailable`** | `ServiceUnavailableError` | Required ML model weights missing |
