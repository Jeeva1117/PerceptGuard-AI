# PERCEPTGUARD-AI — VIDEO INPUT & PROCESSING PIPELINE AUDIT REPORT
**System:** PerceptGuard-AI Autonomous Driving Predictive Safety System  
**Audit Scope:** Video Ingestion, Preview Player, Job Lifecycle, Result Synchronization & Isolation  
**Date:** November 23, 2026  
**Auditor:** Principal Full-Stack ADAS Engineer & Computer Vision Architect  
**Classification:** Engineering Architecture & Reliability Audit  

---

## 1. EXECUTIVE SUMMARY & PROBLEM FORMULATION

### 1.1 Observed Symptoms & User Discrepancies
During testing and evaluation of the PerceptGuard-AI Camera & Weather Intelligence module, the user observed a series of severe state inconsistencies and usability failures:
1. **Target Stream Dropdown Congestion:** The dropdown list contained multiple historical jobs, sample videos (`scenario_a_normal_daytime.mp4`, `job_sample.mp4`), and previously uploaded sessions (`WhatsApp Video 2026-11-23 at 12.04.44 PM.mp4`).
2. **Auto-Loaded Historical Job & Phantom Metrics:** Upon launching or refreshing the frontend at `http://localhost:3000/camera-weather`, the application immediately presented a completed status (`COMPLETED`, `PROVISIONAL CALIBRATION`) with specific telemetry: **60 frames, 12 tracks, 9 alerts**, and **94% EXCELLENT** visibility. The operator had not uploaded a video or selected any job.
3. **Source Video Preview Failure ("No Video Loaded for Preview"):** Despite the stream status showing `COMPLETED` and metrics indicating 60 analyzed frames, the "Source Video Preview" component rendered an empty placeholder: *"No Video Loaded for Preview — Select a video file on the left to verify footage before dispatching to the inference engine."*
4. **Disabled "Start Processing" Button:** The primary action button was greyed out and unclickable despite a video stream appearing active in the top navigation header.
5. **Session Contamination & Lack of Isolation:** Uploading a new video or inspecting previous sessions produced confusing overlays where previous test execution metrics bled into the active view.

---

## 2. DEEP ARCHITECTURAL AUDIT & ROOT CAUSE ANALYSIS

A comprehensive audit of the frontend React application, FastAPI backend routers, background worker thread pools, and persistent storage directories revealed **five distinct structural defects**:

```
+----------------------------------------------------------------------------------------------------+
|                                    DEFECT INTERACTION MAP                                          |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ Backend Disk Storage ]                                                                          |
|       |                                                                                            |
|       +--> Unfiltered Rehydration: outputs/jobs/job_* -> In-Memory Job Dictionary                 |
|                 |                                                                                  |
|  [ GET /api/v1/video/jobs ]                                                                        |
|       |                                                                                            |
|       +--> Returns all historical jobs (sample clips, WhatsApp video, aborted tests)               |
|                 |                                                                                  |
|  [ Frontend JobContext.tsx ]                                                                       |
|       |                                                                                            |
|       +--> Defect 1: Auto-selects 1st completed job (e.g., WhatsApp video, 60 frames, 9 alerts)   |
|                 |                                                                                  |
|                 +---> Header & Metrics: Shows 60 frames, 12 tracks, 9 alerts [STALE GHOST DATA]    |
|                 |                                                                                  |
|                 +---> VideoUploadSection.tsx: Isolated local state (selectedFile = null)           |
|                            |                                                                       |
|                            +--> Defect 2: previewUrl is NULL -> "No Video Loaded for Preview"     |
|                            +--> Defect 3: Start Processing is DISABLED (!selectedFile)            |
|                            +--> Defect 4: No GET endpoint to stream source MP4 from backend        |
|                            +--> Defect 5: No deletion/pruning API for stale test runs             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Root Cause 1: Auto-Selection on Boot in `JobContext.tsx`
* **File:** `frontend/src/context/JobContext.tsx`, lines 73–81:
  ```typescript
  if (!activeJobId && jobList.length > 0) {
    const completedWithFrames = jobList.find(
      (j) => j.status === 'COMPLETED' && ((j.frames_processed ?? 0) > 0 || ((j as any).result_summary?.frames_processed ?? 0) > 0)
    );
    const completed = jobList.find((j) => j.status === 'COMPLETED');
    const defaultJob = completedWithFrames || completed || jobList[0];
    setActiveJobId(defaultJob.job_id);
  }
  ```
* **Flaw:** Whenever a user navigated to the application or refreshed the browser, `JobContext` automatically queried `listJobs()`. If any historical job was present on disk from prior test suites or prior uploads, it automatically selected the first completed job (`job_e7a4dc71abdf` with 60 frames, 12 tracks, 9 alerts).
* **Impact:** The application presented stale, phantom metrics before the user ever interacted with the interface. The user expected a clean slate ready for new input.

### Root Cause 2: Decoupled Upload State & Preview Disconnect in `VideoUploadSection.tsx`
* **File:** `frontend/src/components/VideoUploadSection.tsx`, lines 34–35, 105–107, 445–467:
  ```typescript
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  ...
  const objectUrl = URL.createObjectURL(file);
  setSelectedFile(file);
  setPreviewUrl(objectUrl);
  ```
* **Flaw:** `VideoUploadSection` had its own local state (`selectedFile`, `previewUrl`) completely detached from `JobContext.activeJobId` and `JobContext.activeJob`.
* **Impact:** When `JobContext` auto-selected `job_e7a4dc71abdf`, `selectedFile` remained `null` and `previewUrl` remained `null`. Therefore:
  - The right column checked `if (previewUrl)` -> evaluated to `false` -> rendered *"No Video Loaded for Preview"*.
  - The "Start Processing" button checked `disabled={!selectedFile || ...}` -> evaluated to `true` -> button disabled!
  - The operator was presented with an impossible UI state: a job appeared active and finished in the top header, but the workspace claimed no video existed and refused to process.

### Root Cause 3: Backend Missing Source Video Streaming Endpoints
* **File:** `src/api/routers/video.py`
* **Flaw:** The backend provided `GET /api/v1/video/jobs/{job_id}/frames/{frame_name}` to serve extracted JPEG frames, but had **zero endpoints** to serve or stream the original uploaded video (`.mp4`, `.mov`, `.avi`).
* **Storage Inconsistency:** When videos were uploaded via `POST /api/v1/video/jobs`, the backend saved the raw file to `outputs/jobs/uploads/up_{uuid}_{filename}`. However, `video_path` was not persisted in `JobRecord` nor in `results.json`. On server reboot or rehydration, the job record had no reference to the underlying video file.
* **Impact:** Even if the frontend wanted to render the video for an existing job, there was no API endpoint to request the video file with HTTP range headers (`Accept-Ranges: bytes`) for HTML5 `<video>` playback and seeking.

### Root Cause 4: Unfiltered Job Rehydration & Lack of Job Pruning
* **File:** `src/api/job_manager.py`, lines 94–134:
  ```python
  def _rehydrate_existing_jobs(self) -> None:
      for child in self.jobs_dir.iterdir():
          if child.is_dir() and child.name.startswith("job_"):
              res_file = child / "results.json"
              if res_file.exists():
                  # Rehydrated unconditionally
  ```
* **Flaw:** Every directory in `outputs/jobs/` was blindly rehydrated into memory on every backend startup. Over 24 historical jobs from automated unit test runs (`job_sample.mp4`, `scenario_a_normal_daytime.mp4`) and ad-hoc testing accumulated with no expiration or cleanup API.
* **Impact:** The Target Stream selector was cluttered with irrelevant test jobs. Furthermore, there was no `DELETE /jobs/{job_id}` endpoint to purge old runs or reset the workspace to a pristine state.

### Root Cause 5: Ambiguous Action Semantics ("Start Processing")
* **File:** `frontend/src/components/VideoUploadSection.tsx`
* **Flaw:** The "Start Processing" button only accepted a new local file upload. When a historical job was selected, the button remained permanently disabled without explaining why, and gave no mechanism to re-run or clear the selection.

---

## 3. THREE-TIER SEPARATION OF CONCERNS SPECIFICATION

To resolve all inconsistencies, the system architecture is restructured into three cleanly separated lifecycle domains:

```
+-----------------------------------------------------------------------------------------------+
|                        THREE-TIER SEPARATION OF CONCERNS ARCHITECTURE                         |
+-----------------------------------------------------------------------------------------------+

  TIER 1: INPUT ASSETS (Uploaded Videos)
  =====================================
  - Stored in persistent directory: outputs/uploads/
  - Unique asset identification: upload_id (e.g., up_a8f9c1b2_filename.mp4)
  - Integrity validation: Format (.mp4, .avi, .mov, .mkv), <= 100 MB, OpenCV decode verification
  - Extracted telemetry: Width, Height, FPS, Duration, Total Frames, Bitrate
  - Dedicated Streaming Endpoint: GET /api/v1/video/uploads/{upload_id}/stream
    (Supports HTTP 206 Partial Content, Accept-Ranges: bytes, Content-Range)

  TIER 2: EXECUTION INSTANCES (Processing Jobs)
  =============================================
  - Unique execution identifier: job_id (e.g., job_e7a4dc71abdf)
  - Explicit binding to an input asset (upload_id or video_path)
  - Lifecycle state machine: IDLE -> QUEUED -> VALIDATING -> PROCESSING -> COMPLETED | FAILED
  - Bounded CPU execution: max_workers=1, max_frames=15|30|60, rotation=0|90|180|270
  - Dedicated lifecycle management:
    - POST /api/v1/video/jobs (Create & launch)
    - GET /api/v1/video/jobs/{job_id} (Status & progress)
    - DELETE /api/v1/video/jobs/{job_id} (Cancel active job OR delete finished job from memory & disk)
    - POST /api/v1/video/jobs/clean (Purge stale test jobs)

  TIER 3: TELEMETRY & RESULTS (Analysis Artifacts)
  ================================================
  - Strictly bound to job_id; isolated from new upload sessions
  - Stored in outputs/jobs/{job_id}/results.json, /annotated/, /raw_frames/
  - Dual Preview Synchronization:
    - Local File Mode: Blob URL (URL.createObjectURL) with local HTML5 player controls
    - Stream Selection Mode: Backend Stream (GET /api/v1/video/jobs/{job_id}/source-video)
    - Clean State Mode: No job selected; metrics display "--" / UNKNOWN; zero phantom data
+-----------------------------------------------------------------------------------------------+
```

---

## 4. AUDIT FINDINGS TABLE & COMPLIANCE MATRIX

| Audit Dimension | Prior Implementation | Root Cause | Architectural Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Boot State** | Auto-selected first completed job; displayed 60 frames, 12 tracks, 9 alerts | Hardcoded auto-select in `JobContext.tsx` | Clean slate on boot; `activeJobId = null`; empty metrics indicators | **AUDITED & ROOT CAUSED** |
| **Video Preview** | Showed "No Video Loaded for Preview" when historical job was chosen | `selectedFile` decoupled from `activeJob` | Dual-source preview (local Blob URL OR backend stream URL) | **AUDITED & ROOT CAUSED** |
| **Video Streaming API**| None; only extracted JPEGs (`/frames/{frame_name}`) were served | Backend lacked video streaming route | Dedicated `GET /jobs/{job_id}/source-video` with HTTP 206 Byte Ranges | **AUDITED & ROOT CAUSED** |
| **Stream Selector** | Cluttered with 24+ test jobs (`job_sample.mp4`, etc.) | Unfiltered disk rehydration; no delete API | Clean selector with "Upload New Video / Clean State" + Delete Job API | **AUDITED & ROOT CAUSED** |
| **Button States** | "Start Processing" disabled with no feedback when job chosen | Boolean check `disabled={!selectedFile}` | Distinct UI actions: "Start Processing" (new file) or "Clear / New Upload" | **AUDITED & ROOT CAUSED** |
| **Job Deletion** | Only marked in-flight jobs as `CANCELLED`; cannot remove from disk/memory | Missing deletion logic in `JobManager` | Full job deletion endpoint removing records and artifacts | **AUDITED & ROOT CAUSED** |

---

## 5. REMEDIATION PLAN & IMPLEMENTATION BLUEPRINT

### Backend Implementation Steps:
1. **Source Video Retain & Track:** In `src/api/job_manager.py`, add `video_path` to `JobRecord` and include `has_source_video: bool` and `video_filename: str` in `to_dict()`. Ensure `_rehydrate_existing_jobs` checks whether the video file actually exists in `outputs/jobs/uploads/` or `outputs/uploads/`.
2. **Video Streaming Route (`GET /api/v1/video/jobs/{job_id}/source-video`):** In `src/api/routers/video.py`, implement an HTTP Byte-Range supporting file streamer for MP4, MOV, AVI, MKV.
3. **Job Deletion Route (`DELETE /api/v1/video/jobs/{job_id}`):** Enhance to remove the job record from memory and delete the directory from disk if completed/failed.
4. **Stale Job Pruning Route (`POST /api/v1/video/jobs/clean-stale`):** Allow removing legacy test jobs that lack authentic user videos.

### Frontend Implementation Steps:
1. **Clean Slate Initialization in `JobContext.tsx`:** Remove auto-selection of historical jobs on boot. Default to `activeJobId = null`. Add `clearActiveJob()` and `deleteJob()` methods.
2. **Dual-Source Video Preview in `VideoUploadSection.tsx`:**
   - If `selectedFile` is present: render local Blob URL.
   - Else if `activeJob` is selected and `has_source_video`: render `getSourceVideoUrl(activeJob.job_id)`.
   - Else: render clean dropzone prompting for video upload.
3. **Target Stream Selector in `Header.tsx`:** Include `+ New Video Upload / Clean State` as default option. Allow switching between clean upload mode and inspecting completed runs. Add delete button for historical jobs.
4. **Zero Ghost Data Guarantee:** When `activeJobId === null`, all metric cards across Camera & Weather display `--` and "Awaiting video stream telemetry ingestion."

---
*Report certified by Principal ADAS Systems Architect & Computer Vision Lead.*
