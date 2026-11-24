# PerceptGuard-AI: Video Pipeline Validation & Architecture Report

**Document Version:** 2.0 (Release Readiness)  
**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  

---

## 1. Executive Summary

This document validates the production video ingestion, staging, streaming, execution, and cancellation pipeline for PerceptGuard-AI. The system is engineered to run on standard Intel Core i3 CPU hardware (8 GB RAM, Windows 11) with deterministic performance, memory bounds, and zero UI freezing.

---

## 2. Decoupled Ingestion & Execution Architecture

```
User File Selection (.mp4, .webm, .avi, .mov)
         │
         ▼
[POST /api/v1/video/upload-file]
 ├── Magic Byte Signature Check (ftyp, moov, mdat, WebM, AVI)
 ├── Write to outputs/jobs/uploads/
 └── Returns { video_id, filename, container_meta, stream_url }
         │
         ├───► [GET /api/v1/video/stream/{video_id}]
         │      └── Immediate HTTP 206 Video Preview (No waiting!)
         │
         ▼
[POST /api/v1/video/start-processing]
 ├── Params: video_id, max_frames, rotation, analysis_mode, sample_fps
 ├── Dispatches VideoPipelineRunner in background thread
 └── Returns { job_id, status: "QUEUED", stage: "INITIALIZING" }
         │
         ▼
[VideoPipelineRunner Execution Loop]
 ├── Fast-forward via cap.grab()
 ├── YOLO11n road user detection (conf=0.10, classes=[0,1,2,3,5,7])
 ├── ByteTrack 5-state tracking & Kalman filtering
 ├── 27 multi-modal reliability features
 ├── Multi-horizon LSTM reliability forecasting (H1, H3, H5)
 ├── Weather & visibility intelligence
 ├── Trajectory prediction & kinematics
 ├── Spatio-temporal collision risk scoring
 ├── Early warning & vehicle stop recommendations
 └── Continuous annotated video generation (annotated_video.mp4)
         │
         ▼
[Completed Artifacts]
 ├── results.json (Structured telemetry)
 ├── raw_frames/ & annotated/ (JPEGs)
 └── annotated_video.mp4 (Continuous streamable video)
```

---

## 3. Container Validation & Magic Byte Signatures

To prevent malicious uploads or corrupt containers from crashing OpenCV, the pipeline enforces strict binary header inspection before writing chunks:

| Container Format | Magic Byte Offset / Pattern | Verification Logic |
| :--- | :--- | :--- |
| **MP4 / QuickTime (MOV)** | Bytes 4..12: `ftyp`, `moov`, `mdat` | Signature search in first 64 bytes |
| **WebM / MKV (Matroska)** | First 4 bytes: `\x1a\x45\xdf\xa3` | Exact match |
| **AVI (RIFF Container)** | Bytes 0..4: `RIFF`, Bytes 8..12: `AVI ` | Exact offset check |

Any unrecognized or empty file raises `InvalidVideoError` (HTTP 400 Bad Request) immediately.

---

## 4. HTTP 206 Partial Content Streaming

Source and annotated videos are streamed via custom byte-range endpoints supporting modern HTML5 video controls:
- **Endpoint:** `GET /api/v1/video/stream/{video_id}`
- **Endpoint:** `GET /api/v1/video/jobs/{job_id}/source-video`
- **Endpoint:** `GET /api/v1/video/jobs/{job_id}/annotated-video`

Features:
- Handles `Range: bytes=start-end` request headers.
- Emits `HTTP/1.1 206 Partial Content` with `Content-Range: bytes start-end/total`.
- Enables instant timeline seeking without downloading the full container.

---

## 5. Thread-Safe Cancellation

Operators can abort long-running video inference jobs at any frame:
- **Endpoint:** `POST /api/v1/video/jobs/{job_id}/cancel`
- **Mechanism:** `is_cancelled_callback()` is evaluated before every frame in `VideoPipelineRunner`.
- When cancellation is triggered:
  1. The frame loop terminates immediately.
  2. OpenCV VideoCapture and VideoWriter descriptors are released cleanly.
  3. Job status updates to `CANCELLED`.
  4. Disk artifacts created up to the cancellation point remain intact for inspection.

---

## 6. Frame Serving Resilience

The frame serving endpoint `/api/v1/video/jobs/{job_id}/frames/{frame_name}` resolves requests intelligently:
1. Exact match in `annotated/` or `raw_frames/`.
2. Numeric index strings (e.g. `"0"` -> `annotated_000000.jpg`, `frame_000000.jpg`).
3. Filenames without extensions (e.g. `"frame_000000"` -> `frame_000000.jpg`).
4. Clamps out-of-range numeric indices to the closest available frame.

---

## 7. Verification Test Results

| Test Target | Suite | Cases | Status |
| :--- | :--- | :--- | :--- |
| Staging & Container Validation | `tests/api/test_video_endpoints.py` | 9 | **PASS** |
| Decoupled Processing Pipeline | `tests/integration/test_upload_and_processing_pipeline.py` | 7 | **PASS** |
| Pipeline Cancellation | `tests/integration/test_upload_and_processing_pipeline.py` | 1 | **PASS** |
| Frame Serving API | `tests/api/test_video_endpoints.py` | 1 | **PASS** |
| HTTP 206 Video Streaming | `tests/api/test_video_endpoints.py` | 1 | **PASS** |
| Regression Scenarios | `tests/integration/test_regression_scenarios.py` | 12 | **PASS** |
