# PerceptGuard-AI: Root-Cause Engineering Audit & Resolution Report

**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  
**Engineering Disclaimer:** PerceptGuard-AI is an independent, visibility-aware predictive road safety research prototype. It is not affiliated with Tesla, does not use Tesla software or hardware, does not control vehicle actuators, and is not certified as an automotive safety system.

---

## Executive Summary

During release-readiness system testing, four critical failure modes were identified that prevented reliable autonomous driving perception analysis:
1. **Video Upload & "Start Processing" Workflow Breakdown**: Tight coupling between staging and pipeline execution caused UI deadlocks and missing video streams.
2. **Extracted Frame Serving Breakdown**: Numeric index requests and missing extension lookups triggered 404 errors during frame scrubbing.
3. **Object Detection & Motorcycle Tracking Failure**: High-confidence filtering (`conf=0.25`) starved ByteTrack of low-confidence detections needed for Stage 2 recovery, leading to frequent track loss and class flipping for motorcycles.
4. **Five-Module Telemetry Disconnection**: Sub-pages did not share a single source of truth for temporal frame progression and active job state.

This document details the root causes, affected files, exact repairs implemented, and automated test validations.

---

## 1. Problem 1: Video Upload & Start Processing Workflow Breakdown

### Root Cause
In earlier iterations, file upload immediately triggered synchronous or monolithic processing without a distinct staging phase. The UI attempted to start processing using unverified video paths, while the backend lacked container validation for non-MP4 formats (such as `.webm`, `.avi`, and `.mov`). Furthermore, when an operator clicked "Start Processing", the client lacked job state machine transitions, causing the button to remain disabled or to freeze during background inference.

### Affected Files
- `src/api/routers/video.py`
- `src/video_processing/job_manager.py`
- `src/video_processing/video_pipeline_runner.py`
- `frontend/src/components/VideoUploadSection.tsx`
- `frontend/src/context/JobContext.tsx`

### Why It Happened
1. The upload endpoint did not validate binary container headers (magic bytes) before writing to disk, permitting corrupt uploads.
2. Job submission did not decouple file staging (`POST /api/v1/video/upload-file`) from pipeline execution (`POST /api/v1/video/start-processing`).
3. The frontend state machine lacked explicit intermediate states (`IDLE`, `SELECTING`, `UPLOADING`, `STAGED`, `PROCESSING`, `COMPLETED`, `FAILED`), leading to race conditions between React state and server status.

### Exact Repair
1. **Decoupled API Endpoints**:
   - `POST /api/v1/video/upload-file`: Accepts binary multipart uploads, validates container signatures (`ftyp`, `moov`, `mdat` for MP4/MOV; `\x1a\x45\xdf\xa3` for WebM/MKV; `RIFF...AVI ` for AVI), stages file under `outputs/jobs/uploads/`, and returns `video_id`, container metadata, and stream URL.
   - `POST /api/v1/video/start-processing`: Accepts `{video_id, max_frames, rotation, analysis_mode, sample_fps}` and submits an asynchronous background job with thread-safe cancellation callbacks.
   - `GET /api/v1/video/stream/{video_id}`: Provides HTTP 206 Partial Content byte-range streaming for seamless preview playback.
2. **6-State Frontend Upload State Machine**:
   - `frontend/src/components/VideoUploadSection.tsx` implements strict deterministic state transitions: `IDLE` -> `SELECTING` -> `UPLOADING` -> `STAGED` -> `PROCESSING` -> `COMPLETED`.
   - "Start Processing" is activated immediately upon reaching the `STAGED` state.
3. **Cancellation Hook**:
   - Added `POST /api/v1/video/jobs/{job_id}/cancel` allowing operators to safely abort long-running inferences.

### Verification Test
- `tests/api/test_video_endpoints.py::TestVideoEndpoints::test_01_upload_file_staging_and_validation` -> **PASS**
- `tests/integration/test_upload_and_processing_pipeline.py::test_full_upload_to_processing_flow` -> **PASS**

---

## 2. Problem 2: Extracted Frame Serving Breakdown

### Root Cause
When the frontend scrubbed through video frames or displayed fallback images via `/api/v1/video/jobs/{job_id}/frames/{frame_name}`, integer index requests (e.g., `/frames/0`) and missing `.jpg` extensions resulted in HTTP 404 Not Found errors.

### Affected Files
- `src/api/routers/video.py`
- `frontend/src/components/VideoCanvas.tsx`
- `frontend/src/services/api.ts`

### Why It Happened
`get_annotated_frame` performed a strict single-file lookup: `Path(job_rec.results_dir) / "annotated" / safe_name`. If `safe_name` was `"0"`, the backend searched for a file named `0` rather than `annotated_000000.jpg` or `frame_000000.jpg`, returning 404.

### Exact Repair
`get_annotated_frame` in `src/api/routers/video.py` was rewritten with multi-candidate resolution:
1. Checks exact match in `annotated/` and `raw_frames/`.
2. Resolves numeric indices (e.g., `0` -> `annotated_000000.jpg` and `frame_000000.jpg`).
3. Handles missing extensions by appending `.jpg` or `.png`.
4. Clamps out-of-bounds numeric indices to available frame files as a safety fallback.

### Verification Test
- `tests/api/test_video_endpoints.py::TestVideoEndpoints::test_05_get_frame_images_both_annotated_and_raw` -> **PASS**

---

## 3. Problem 3: Object Detection & Motorcycle Tracking Failure

### Root Cause
1. **Low-Confidence Starvation in ByteTrack**: YOLO inference was invoked with `conf=0.25`. This discarded all detections in the range `0.10 <= conf < 0.25`. However, ByteTrack's fundamental architecture relies on two-stage data association: Stage 1 associates high-confidence detections ($\ge 0.25$), while Stage 2 associates low-confidence detections ($0.10 \le conf < 0.25$) to maintain continuity for occluded, blurred, or turning road users. Motorcycles (which frequently experience temporary visual degradation) were immediately dropped.
2. **Class Identity Volatility**: When detections were ingested from raw arrays without explicit class strings, `cls_name` defaulted to `str(cls_id)`. Furthermore, between frames, fluctuating classification probabilities caused track labels to flip between `car` and generic labels.

### Affected Files
- `src/tracking/bytetrack_engine.py`
- `src/video_processing/video_pipeline_runner.py`

### Why It Happened
- `video_pipeline_runner.py` did not restrict YOLO inference to road user classes (`[0, 1, 2, 3, 5, 7]`) and used an aggressive initial cutoff of `conf=0.25`.
- `ByteTrackEngine` did not preserve persistent class assignments per track ID across the track lifecycle.

### Exact Repair
1. **Two-Stage Ingestion in Pipeline Runner**:
   - YOLO inference is now executed with `conf=0.10`, `iou=0.45`, and `classes=[0, 1, 2, 3, 5, 7]` (road users: person, bicycle, car, motorcycle, bus, truck).
   - All candidate detections (including `0.10 <= conf < 0.20`) are forwarded to `tracker.update(raw_detections)`.
   - UI display detections are cleanly filtered to `confidence >= 0.20`.
2. **Persistent Track Class Memory**:
   - `ByteTrackEngine` defines `DEFAULT_ROAD_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", ...}`.
   - `ByteTrackEngine` maintains `self._track_classes: Dict[int, Tuple[int, str]]`. Once a track is classified as a motorcycle, its semantic label remains persistent across subsequent frames, even if confidence dips or occlusion occurs.

### Verification Test
- `tests/tracking/test_motorcycle_and_road_users.py::TestMotorcycleAndRoadUsers::test_03_motorcycle_track_continuity_and_confidence_dip` -> **PASS**
- `tests/tracking/test_motorcycle_and_road_users.py::TestMotorcycleAndRoadUsers::test_06_motorcycle_reidentification_after_occlusion` -> **PASS**
- `tests/tracking/test_bytetrack_engine.py` (18 tests) -> **PASS**

---

## 4. Problem 4: Five-Module Disconnection & Data Synchronization

### Root Cause
The five primary modules (Camera & Weather, Object Detection, Trajectory Prediction, Collision Risk, and Safety Alerts) operated as siloed components. Scrubbing the timeline on one page did not synchronize playback timestamp, selected track focus, or frame index across the other four pages.

### Affected Files
- `frontend/src/context/JobContext.tsx`
- `frontend/src/pages/CameraWeatherPage.tsx`
- `frontend/src/pages/ObjectDetectionPage.tsx`
- `frontend/src/pages/TrajectoryPredictionPage.tsx`
- `frontend/src/pages/CollisionRiskPage.tsx`
- `frontend/src/pages/SafetyAlertsPage.tsx`

### Why It Happened
Each page maintained local playback or selection state rather than consuming synchronized telemetry from `JobContext`.

### Exact Repair
1. **Unified Context Synchronization**:
   - `JobContext.tsx` provides centralized state for `activeJobId`, `currentFrameIndex`, `currentTime`, `isPlaying`, `selectedTrackId`, and `activeFilters`.
   - When an operator navigates between any of the 5 modules, the current video playback position and focused track are preserved identically.
2. **Integrated Canvas Overlay**:
   - `VideoCanvas.tsx` renders unified computer-vision overlays (weather HUD banner, road corridors, bounding boxes, motion trails, and future trajectory arrows) consistently across all modules.

### Verification Test
- `npm --prefix frontend run build` -> **PASS** (Zero TypeScript errors)
- `tests/integration/test_regression_scenarios.py` (12 scenarios) -> **PASS**

---

## Conclusion

All four primary failure modes have been thoroughly analyzed, resolved with production-grade engineering fixes, and verified via automated regression test suites.
