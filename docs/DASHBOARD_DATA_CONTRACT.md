# Dashboard & API Data Contract Specification
## PerceptGuard AI: Schemas and Contract Invariants

---

## 1. Video Job Schemas

### Job Submission Request (`POST /api/v1/video/jobs`)
```json
{
  "file": "<multipart-form-data>",
  "max_frames": 20
}
```

### Job Submission Response (`HTTP 202 Accepted`)
```json
{
  "job_id": "job_c361ecc0c0ab",
  "filename": "annotated_video.mp4",
  "status": "QUEUED",
  "stage": "QUEUED",
  "message": "Queued for processing",
  "max_frames": 20,
  "created_at": "2026-11-20T21:54:10.123456+00:00"
}
```

### Job Status Query (`GET /api/v1/video/jobs/{job_id}`)
```json
{
  "job_id": "job_c361ecc0c0ab",
  "filename": "annotated_video.mp4",
  "status": "PROCESSING",
  "stage": "PROCESSING",
  "progress_percentage": 65.0,
  "message": "Processing frame 13/20 (YOLO + ByteTrack + LSTM)...",
  "created_at": "2026-11-20T21:54:10.123456+00:00",
  "started_at": "2026-11-20T21:54:10.234567+00:00",
  "completed_at": null,
  "max_frames_requested": 20,
  "frames_processed": 13,
  "total_detections": 8,
  "unique_tracks": 1,
  "alerts_count": 6,
  "error_message": null,
  "result_summary": null
}
```

---

## 2. Completed Job Results Artifact (`results.json`)

```json
{
  "job_id": "job_c361ecc0c0ab",
  "video_filename": "annotated_video.mp4",
  "video_metadata": {
    "filename": "annotated_video.mp4",
    "width": 640,
    "height": 360,
    "fps": 5.0,
    "total_container_frames": 20,
    "duration_seconds": 4.0
  },
  "frames_processed": 15,
  "total_detections": 8,
  "unique_tracks": 1,
  "total_alerts": 9,
  "active_alerts": 3,
  "calibration_status": "PROVISIONAL CALIBRATION",
  "per_horizon_risk": {
    "h1": {
      "horizon": 1,
      "latest_risk": 0.6034,
      "mean_risk": 0.6034,
      "max_risk": 0.6034,
      "category": "HIGH"
    },
    "h3": {
      "horizon": 3,
      "latest_risk": 0.7092,
      "mean_risk": 0.7092,
      "max_risk": 0.7092,
      "category": "HIGH"
    },
    "h5": {
      "horizon": 5,
      "latest_risk": 0.7997,
      "mean_risk": 0.7997,
      "max_risk": 0.7997,
      "category": "HIGH"
    }
  },
  "stage_durations_ms": {
    "validation_ms": 24.77,
    "detection_total_ms": 6744.62,
    "tracking_total_ms": 10.42,
    "feature_extraction_ms": 101.21,
    "forecasting_ms": 18.43,
    "risk_scoring_ms": 6.74,
    "visualization_ms": 84.4
  },
  "total_duration_seconds": 7.244
}
```
