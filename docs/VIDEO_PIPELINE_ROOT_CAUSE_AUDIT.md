# PERCEPTGUARD-AI — VIDEO PIPELINE ROOT-CAUSE AUDIT REPORT

**Document ID:** PG-AUDIT-006  
**Date:** 2026-11-24  
**Author:** Coordinated Software, Computer Vision, AI & Frontend Architecture Engineering Team  
**Status:** COMPLETED & ACTIONABLE  

---

## 1. Executive Summary

This comprehensive root-cause audit investigates the failures identified across the video upload, execution initiation, video preview, playback synchronization, ByteTrack tracking, and five-module integration in PerceptGuard-AI.

The audit examined every link in the execution chain:
- **Frontend:** Upload component, file validation, state machines, API requests, React Context (`JobContext`), canvas overlay synchronizers, and module sub-routes.
- **Backend:** FastAPI routes in `src/api/routers/video.py`, `JobManager` in `src/api/job_manager.py`, `VideoPipelineRunner` in `src/video_processing/video_pipeline_runner.py`, OpenCV decoding, ByteTrack multi-object tracking, and JSON persistence.

---

## 2. Detailed Root-Cause Analysis Matrix

### Issue 1: Video Upload Fails or Does Not Store Video Reliably

* **Symptom:** Uploading raw traffic videos fails silently, times out, or accepts unsupported/corrupt files without clear feedback.
* **Affected Files:**
  - `src/api/routers/video.py`
  - `frontend/src/components/VideoUploadSection.tsx`
  - `frontend/src/services/api.ts`
* **Root Causes:**
  1. *Extension-Only Validation:* The backend checked only string extensions (`.mp4, .avi, .mov, .mkv`) without validating file magic bytes (signatures). Files renamed with false extensions passed initial checks and failed during OpenCV decoding.
  2. *Missing `.webm` Format:* While OpenCV natively decodes `.webm` containers, `ALLOWED_VIDEO_EXTENSIONS` omitted `.webm`.
  3. *Coupled Upload-and-Execute Flow:* The backend had no standalone upload-and-persist endpoint (`POST /video/upload-file`). The only route, `POST /video/jobs`, required streaming the entire video and immediately launched pipeline execution.
  4. *Frontend Local-Only Illusion:* The frontend set `processingState = 'FILE_SELECTED'` and created a local browser `URL.createObjectURL(file)`, but never uploaded the video to the server. If the user refreshed or changed tabs, the file was lost.
* **Required Fix:**
  - Implement binary magic-byte inspection on backend for MP4/MOV (`ftyp`, `moov`, `mdat`), MKV/WebM (`\x1A\x45\xDF\xA3`), and AVI (`RIFF...AVI`).
  - Add `.webm` to supported extensions.
  - Create dedicated endpoint `POST /api/v1/video/upload-file` that inspects, validates, permanently stores the video, and returns a unique `video_id`, metadata, and stream URL.
  - Build a 6-state upload state machine in the frontend: `IDLE` $\to$ `FILE_SELECTED` $\to$ `VALIDATING` $\to$ `UPLOADING` $\to$ `UPLOAD_COMPLETED` $\to$ `UPLOAD_FAILED`.
* **Validation Method:** Upload valid MP4, MKV, MOV, WebM, empty 0-byte file, and renamed text file; assert proper acceptance or rejection with descriptive errors.

---

### Issue 2: "Start Processing" Does Not Respond or Initiate Analysis

* **Symptom:** Clicking "Start Processing" appears disabled, freezes with an ambiguous spinner, or fails without launching a background job.
* **Affected Files:**
  - `frontend/src/components/VideoUploadSection.tsx`
  - `frontend/src/context/JobContext.tsx`
  - `src/api/routers/video.py`
  - `src/api/job_manager.py`
* **Root Causes:**
  1. *Double-Action Blocking:* Because upload and start processing were coupled, clicking "Start Processing" had to upload the entire 100MB file over HTTP in the same click event. Slow uploads caused users to click repeatedly or assume the UI had frozen.
  2. *Missing Video ID Dispatch:* The frontend did not pass an existing `video_id` to start processing; it required re-streaming the file binary.
  3. *Premature Results Polling:* Upon submitting a job, the frontend immediately called `getJobResults(job_id)`. Because the job was in `QUEUED` state, `results.json` did not exist on disk, causing a 404 or null result that corrupted frontend state.
* **Required Fix:**
  - Separate upload from processing: User uploads video $\to$ receives server-confirmed `video_id` $\to$ "Start Processing" enables $\to$ User clicks "Start Processing" $\to$ sends `{ video_id, analysis_mode, rotation, sample_fps }` to `POST /api/v1/video/start-processing`.
  - Prevent duplicate requests with idempotency protection and button disabling.
  - Implement clean polling on `/jobs/{job_id}` for progress until `status === 'COMPLETED'`, and only then fetch `/jobs/{job_id}/results`.
* **Validation Method:** Submit job for uploaded video, observe `QUEUED` $\to$ `PROCESSING` (with live progress percentage) $\to$ `COMPLETED`, verify no unhandled promise rejections.

---

### Issue 3: Uploaded Video Does Not Appear in Video Player

* **Symptom:** After selecting or uploading a video, the main video player displays "No Active Continuous Stream" instead of rendering the video.
* **Affected Files:**
  - `frontend/src/components/VideoCanvas.tsx`
  - `frontend/src/components/PersistentVideoWorkspace.tsx`
  - `frontend/src/context/JobContext.tsx`
* **Root Causes:**
  1. *Premature Guard Clause in `VideoCanvas.tsx`:*
     ```tsx
     if (!activeJobId || !results || !results.frames || results.frames.length === 0) {
       return (<div>No Active Continuous Stream</div>);
     }
     ```
     This guard prevented the `<video>` element from mounting if `results.frames` was empty, which is always true for newly uploaded videos or active in-flight jobs!
  2. *Drawer Occlusion in `PersistentVideoWorkspace.tsx`:* When `!hasVideoLoaded`, the workspace replaced the video canvas with the upload drawer instead of keeping the video player visible.
* **Required Fix:**
  - Update `VideoCanvas.tsx` to mount and play the `<video>` element whenever an active video source exists:
    - Server stream URL (`/api/v1/video/stream/{video_id}`)
    - Historical job stream URL (`/api/v1/video/jobs/{job_id}/source-video`)
    - Staged preview URL (`URL.createObjectURL(file)`)
  - Render HUD overlays only when telemetry frames are available, but keep video playback functional at all times.
* **Validation Method:** Select/upload a video; verify that the HTML5 video preview appears immediately in the main center workspace, is playable, and persists across navigation.

---

### Issue 4: Video Playback and Processing Are Not Synchronized

* **Symptom:** Bounding boxes, trajectories, and risk badges lag behind or lead video playback, especially after scrubbing.
* **Affected Files:**
  - `frontend/src/components/VideoCanvas.tsx`
  - `frontend/src/context/JobContext.tsx`
  - `frontend/src/components/InteractiveTimeline.tsx`
* **Root Causes:**
  1. *Coarse Sampling Resolution:* The pipeline samples frames at 2 FPS or 5 FPS, whereas the video plays at 30 FPS. Simple index matching failed because frame indices did not match time seconds.
  2. *Fast Binary Search with Tolerance:* Time synchronization did not account for variable frame rate or duration mismatches between container header and decoded frames.
* **Required Fix:**
  - Use exact monotonic timestamp matching via binary search: `findClosestFrameIndex(frames, video.currentTime)`.
  - Cache the matched frame and re-render canvas overlays synchronously in `requestAnimationFrame` or `onTimeUpdate`.
  - Bind `<InteractiveTimeline />` seeking directly to `video.currentTime = targetTime`.
* **Validation Method:** Scrub through video at 0.5x, 1.0x, and 2.0x speed; verify bounding boxes align with the physical vehicles across all timestamps.

---

### Issue 5: ByteTrack Tracking Is Incomplete or Unreliable

* **Symptom:** Vehicle IDs flicker, change ID numbers between frames, or fail to track across brief occlusions.
* **Affected Files:**
  - `src/tracking/bytetrack_engine.py`
  - `src/video_processing/video_pipeline_runner.py`
* **Root Causes:**
  1. *Missing Lost-Track Re-identification:* When an object was temporarily lost for 1-2 frames due to a dip in YOLO detection score ($< 0.25$), ByteTrack was dropping the track rather than moving it to `LOST` state for Hungarian re-matching.
  2. *Lifecycle State Propagation:* The tracker did not expose explicit lifecycle states (`TENTATIVE`, `CONFIRMED`, `LOST`, `REIDENTIFIED`, `REMOVED`) to the pipeline runner results or frontend telemetry.
* **Required Fix:**
  - Fully integrate the 5-state ByteTrack state machine with 30-frame lost buffer.
  - Re-identify returning tracks with high IoU match without allocating new IDs.
  - Export `track_lifecycle` in `tracks_info` and display on Module 2 UI.
* **Validation Method:** Run `test_18_track_lifecycle_progression` and verify persistent IDs remain constant throughout video playback.

---

### Issue 6: The Five Modules Do Not Consistently Display Their Own Analysis

* **Symptom:** Modules 1 through 5 show generic or duplicate cards, stale values, or empty errors.
* **Affected Files:**
  - `frontend/src/pages/CameraWeatherPage.tsx`
  - `frontend/src/pages/ObjectDetectionPage.tsx`
  - `frontend/src/pages/TrajectoryPredictionPage.tsx`
  - `frontend/src/pages/CollisionRiskPage.tsx`
  - `frontend/src/pages/SafetyAlertsPage.tsx`
* **Root Causes:**
  - Modules were querying separate endpoints or expecting raw frames instead of reading from the unified `JobContext` results.
  - When embedded in the persistent workspace, pages rendered duplicate video players and conflicting upload widgets.
* **Required Fix:**
  - Pass `isEmbedded={true}` to each page when rendered in the right-hand analytical column.
  - Ensure each module binds directly to the synchronized `currentFrame` and displays domain-specific metrics.
* **Validation Method:** Navigate through all five modules during playback; verify each renders its distinct data synchronized with the active frame.

---

### Issue 7: Stale Analysis Results Appear After Uploading a Different Video

* **Symptom:** Uploading a new video displays telemetry, alerts, and bounding boxes from the previous video.
* **Affected Files:**
  - `frontend/src/context/JobContext.tsx`
  - `frontend/src/components/VideoUploadSection.tsx`
  - `src/api/job_manager.py`
* **Root Causes:**
  - `JobContext` did not wipe `results`, `currentFrameIndex`, `selectedTrackId`, and `activeJob` when a new file was staged or uploaded.
  - The backend `clean_stale_jobs` was not invoked on new upload sessions, allowing old job caches to persist.
* **Required Fix:**
  - Implement atomic session isolation in `JobContext`: `clearActiveJob()` and `startNewUploadSession()` reset all telemetry, alerts, and overlays.
  - Every upload receives a unique `video_id = vid_{uuid}` and every analysis receives a unique `job_id = job_{uuid}`.
* **Validation Method:** Process Video 1, upload Video 2; assert that Video 2 starts with zero alerts, clean telemetry, and independent track IDs.

---

## 3. Implementation Roadmap

| Phase | Milestone | Priority | Target Components |
| :--- | :--- | :--- | :--- |
| **Phase 2** | Reliable Video Upload UI & State Machine | CRITICAL | `src/api/routers/video.py`, `frontend/src/components/VideoUploadSection.tsx` |
| **Phase 3** | Two-Step Start Processing Architecture | CRITICAL | `src/api/routers/video.py`, `src/api/job_manager.py`, `frontend/src/services/api.ts` |
| **Phase 4** | Continuous Video Playback & Full-Duration Processing | HIGH | `frontend/src/components/VideoCanvas.tsx`, `PersistentVideoWorkspace.tsx` |
| **Phase 5** | ByteTrack Tracking & Lifecycle Visualization | HIGH | `src/tracking/bytetrack_engine.py`, `ObjectDetectionPage.tsx` |
| **Phase 6** | Five-Module Domain Synchronization | HIGH | All 5 module pages |
| **Phase 7** | Video-First Professional HUD UI | MEDIUM | `Header.tsx`, `InteractiveTimeline.tsx` |
| **Phase 8** | Session Isolation & Cleanup | HIGH | `JobContext.tsx`, `job_manager.py` |
| **Phase 9** | Production Error Handling & Cancellation | HIGH | `video_pipeline_runner.py`, `JobManager.cancel_job` |
| **Phase 10** | CPU & Memory Bounded Execution | HIGH | OpenCV sequential decoding, 2 FPS sampling |
| **Phase 11** | Integration & Acceptance Testing | CRITICAL | Automated test suites & end-to-end verification |

---

## 4. Conclusion

This audit establishes the concrete root causes and engineering remediations required to make the PerceptGuard-AI video upload, processing, and visualization pipeline reliable, production-grade, and performant under CPU constraints.
