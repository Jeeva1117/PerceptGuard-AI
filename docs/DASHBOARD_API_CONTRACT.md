# Dashboard Backend API Specification & Contract

## 1. Overview

The `DashboardService` (`src/early_warning/dashboard_service.py`) fulfills the `IEarlyWarningDashboard` interface, providing a unified, bounded, in-process and REST-compatible service layer for querying perception reliability metrics, multi-horizon risk assessments, temporal trajectories, incident alerts, and tracking telemetry.

---

## 2. API Endpoints & Contracts

### 2.1 Get System Operational Health
- **Method / Function:** `service.get_system_health() -> Dict[str, Any]`
- **Description:** Returns high-level operational status, health scores, model availability, and calibration status.
- **Request Parameters:** None.
- **Response Schema:**
```json
{
  "project_name": "PerceptGuard AI",
  "pipeline_status": "READY | RUNNING | DEGRADED | BLOCKED | UNAVAILABLE",
  "overall_health": "NOMINAL | DEGRADED | CRITICAL",
  "health_score": 98.5,
  "current_sequence_id": "seq_driving_sample_L5_000001",
  "current_video_id": "driving_sample",
  "last_prediction_timestamp_seconds": 1.7946,
  "data_freshness": "CURRENT | STALE",
  "model_availability": {
    "lstm_forecaster": "AVAILABLE | UNAVAILABLE",
    "yolo_detector": "AVAILABLE | UNAVAILABLE",
    "bytetrack_tracker": "AVAILABLE | UNAVAILABLE"
  },
  "calibration_gate_status": "PROVISIONAL",
  "timestamp": "2026-11-20T15:00:00Z"
}
```

---

### 2.2 Get Pipeline Component Status
- **Method / Function:** `service.get_pipeline_status() -> Dict[str, Any]`
- **Description:** Reports the operational status and diagnostic details of all 8 perception pipeline stages.
- **Request Parameters:** None.
- **Response Schema:**
```json
{
  "components": {
    "stage1_video_processing": { "status": "COMPLETE", "details": "720p 30fps frames verified" },
    "stage2_yolo_detection": { "status": "COMPLETE", "details": "YOLO11n CPU inference verified" },
    "stage3_bytetrack_tracking": { "status": "COMPLETE", "details": "Kalman filter kinematics active" },
    "stage4_feature_engineering": { "status": "COMPLETE", "details": "27 multi-modal features" },
    "stage5_temporal_sequences": { "status": "COMPLETE", "details": "Window L=5, stride G=25" },
    "stage6_lstm_forecasting": { "status": "COMPLETE", "details": "Checkpoints verified (H1, H3, H5)" },
    "stage7_risk_scoring": { "status": "COMPLETE", "details": "Linear inverse S = 1 - R" },
    "stage8_early_warning": { "status": "COMPLETE", "details": "Rule engine & lifecycle management" }
  },
  "overall_readiness": "READY"
}
```

---

### 2.3 Get Multi-Horizon Reliability Risk Overview
- **Method / Function:** `service.get_reliability_risk_overview(sequence_id: Optional[str] = None) -> Dict[str, Any]`
- **Description:** Returns independent risk metrics for horizons $H1$, $H3$, and $H5$.
- **Request Parameters:**
  - `sequence_id` (optional, string): Unique sequence identifier. Defaults to latest tracked sequence.
- **Response Schema:**
```json
{
  "sequence_id": "seq_driving_sample_L5_000001",
  "calibration_status": "PROVISIONAL",
  "calibration_notice": "Thresholds are designated as PROVISIONAL due to single-clip zero-inflation.",
  "horizons": {
    "h1": {
      "horizon": 1,
      "risk_score": 0.0825,
      "risk_level": "LOW",
      "trend_direction": "STABLE",
      "rolling_mean": 0.0850,
      "rolling_max": 0.0910,
      "rolling_min": 0.0800,
      "volatility": 0.0035,
      "observation_count": 30,
      "timestamp_seconds": 1.0,
      "is_stale": false,
      "calibration_status": "PROVISIONAL"
    },
    "h3": { "horizon": 3, "risk_score": 0.1240, "...": "..." },
    "h5": { "horizon": 5, "risk_score": 0.1890, "...": "..." }
  },
  "timestamp": "2026-11-20T15:00:00Z"
}
```

---

### 2.4 Get Temporal Monitoring Data
- **Method / Function:** `service.get_temporal_monitoring_data(sequence_id: Optional[str], horizon: Optional[int], limit: int = 50) -> Dict[str, Any]`
- **Description:** Returns historical time series records and discrete operational state transitions for plotting.
- **Request Parameters:**
  - `sequence_id` (optional, string): Filter by sequence.
  - `horizon` (optional, integer: 1, 3, or 5): Filter by horizon.
  - `limit` (integer, default 50): Max records per horizon.
- **Response Schema:**
```json
{
  "sequence_id": "seq_driving_sample_L5_000001",
  "limit": 50,
  "horizons": {
    "h1": [
      {
        "timestamp_seconds": 0.033,
        "risk_score": 0.0512,
        "risk_level": "LOW",
        "is_valid": true,
        "is_stale": false,
        "source_frame_id": "frame_000001.jpg"
      }
    ],
    "h3": [],
    "h5": []
  },
  "state_transitions": [
    {
      "sequence_id": "seq_driving_sample_L5_000001",
      "horizon": 1,
      "previous_level": "LOW",
      "current_level": "MEDIUM",
      "timestamp_seconds": 1.25,
      "risk_score": 0.354,
      "is_genuine_transition": true,
      "notes": "Transition from LOW to MEDIUM"
    }
  ],
  "timestamp": "2026-11-20T15:00:00Z"
}
```

---

### 2.5 Get Paginated & Filtered Alerts
- **Method / Function:** `service.get_alerts(severity, horizon, alert_type, status, page, page_size) -> Dict[str, Any]`
- **Description:** Returns paginated, filtered early-warning incidents.
- **Request Parameters:**
  - `severity` (optional string): `CRITICAL`, `WARNING`, `CAUTION`, `INFO`, `DATA_QUALITY_WARNING`
  - `horizon` (optional int): 1, 3, 5
  - `alert_type` (optional string): `RISK_LEVEL_THRESHOLD`, `RAPID_RISK_INCREASE`, etc.
  - `status` (optional string): `ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`, `EXPIRED`
  - `page` (int, default 1): Page index (1-based).
  - `page_size` (int, default 20): Bounded page capacity.
- **Response Schema:**
```json
{
  "total_count": 42,
  "page": 1,
  "page_size": 20,
  "total_pages": 3,
  "alerts": [
    {
      "alert_id": "alert_seq_01_h1_RULE_CRITICAL_BREACH_1250",
      "sequence_id": "seq_01",
      "severity": "CRITICAL",
      "alert_type": "THRESHOLD_BREACH",
      "trigger_horizon": 1,
      "risk_score": 0.925,
      "lifecycle_state": "ACTIVE",
      "warning_message": "Immediate perception failure predicted (H1). S = 0.9250.",
      "timestamp_seconds": 1.25
    }
  ],
  "active_alerts_count": 8
}
```

---

### 2.6 Alert Detail & Lifecycle Actions
- **Get Detail:** `service.get_alert_detail(alert_id: str) -> Optional[Dict[str, Any]]`
  - Returns complete alert metadata, rationale, frame references, and audit event list.
- **Acknowledge:** `service.acknowledge_alert(alert_id: str, operator_id: str = "operator") -> Dict[str, Any]`
  - Transitions state from `ACTIVE` to `ACKNOWLEDGED`.
- **Resolve:** `service.resolve_alert(alert_id: str, reason: str, operator_id: str) -> Dict[str, Any]`
  - Transitions state from `ACTIVE` or `ACKNOWLEDGED` to `RESOLVED`.

---

### 2.7 Get Detection & Tracking Context
- **Method / Function:** `service.get_detection_tracking_context(sequence_id: Optional[str] = None) -> Dict[str, Any]`
- **Description:** Returns authentic YOLO11n detection and ByteTrack tracking telemetry.
- **Response Schema:**
```json
{
  "status": "AVAILABLE | UNAVAILABLE",
  "model_name": "yolo11n",
  "tracker_type": "bytetrack",
  "total_frames_processed": 30,
  "total_detections": 27,
  "total_unique_tracks": 2,
  "max_simultaneous_active_tracks": 2,
  "mean_confidence_overall": 0.4003,
  "mean_processing_latency_ms": 280.43,
  "per_class_track_counts": { "car": 2 },
  "is_synthetic": false
}
```
