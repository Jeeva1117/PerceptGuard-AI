# PERCEPTGUARD-AI — VIDEO UPLOAD PIPELINE REPAIR & VALIDATION REPORT

**Document ID:** PG-REP-006  
**Date:** 2026-11-24  
**Author:** Coordinated AI/Full-Stack & Computer Vision Engineering Team  
**Status:** FULLY REPAIRED & PRODUCTION-VERIFIED  

---

## 1. Executive Summary

The video upload mechanism was re-architected from a brittle, client-only staged object URL into an automotive-grade, server-backed, asynchronous upload and validation pipeline.

Prior to this engineering effort:
- The frontend created a temporary browser-local object URL (`URL.createObjectURL(file)`) without uploading the file to the backend until the user clicked "Start Processing".
- File validation checked only superficial file extension strings (`.mp4, .avi, .mov, .mkv`), allowing invalid, corrupted, or non-video files to enter the pipeline.
- Video container format `.webm` was omitted despite OpenCV decoding support.
- The UI provided no true upload state machine, resulting in UI lockups and ambiguous button behavior.

All of these root causes have been resolved.

---

## 2. Root Cause & Architectural Rectification

### 2.1 Two-Step Upload & Execution Decoupling
The backend and frontend were decoupled into two clear, distinct architectural phases:

```
[User Selects Video]
        │
        ▼
[Frontend File & Size Validation] ──(Valid?)──► [State: VALIDATING]
        │
        ▼
[POST /api/v1/video/upload-file] ──► [Binary Magic Bytes Check]
        │                             │ (Valid Container Header?)
        │                             ▼
        │                     [OpenCV Header Inspection]
        │                             │ (Extract FPS, Dim, Duration)
        │                             ▼
        │                     [Permanent Disk Storage]
        │                             │ (outputs/jobs/uploads/vid_...)
        ▼                             ▼
[Receive video_id & Metadata] ◄─── [HTTP 201 Created]
        │
        ▼
[State: UPLOAD_COMPLETED]
[Mount HTML5 Video Preview]
[Enable 'Start Processing']
        │
[User Clicks 'Start Processing']
        │
        ▼
[POST /api/v1/video/start-processing] ──► [Job Created: job_...]
        │                                  │ (Background ThreadPool)
        ▼                                  ▼
[State: PROCESSING] ◄────────────── [HTTP 202 Accepted]
        │
        ▼
[Poll /jobs/{job_id}] ──► [Progress 0% -> 100%] ──► [Populate 5 Modules]
```

### 2.2 Deep Binary Magic-Byte Inspection
Rather than trusting client-reported extensions or MIME types, the backend in `src/api/routers/video.py` validates the first 128 bytes against standardized container signatures:
- **MP4 / MOV:** Checks for `ftyp` (including `isom`, `mp42`, `qt  `, `M4V `) or `moov` / `mdat` atoms.
- **WebM / MKV:** Checks for the EBML element header ID `\x1a\x45\xdf\xa3`.
- **AVI:** Checks for `RIFF` header with `AVI ` format identifier.

Non-video files, corrupt files, or renamed text files are rejected with HTTP 400 `INVALID_VIDEO_FILE`.

### 2.3 Strict 6-State Upload State Machine
The frontend component `frontend/src/components/VideoUploadSection.tsx` implements:
1. `IDLE`: No file selected; drag-and-drop zone active.
2. `FILE_SELECTED`: Local file staged and basic constraints checked.
3. `VALIDATING`: Client and initial container integrity check running.
4. `UPLOADING`: Streaming file to `POST /api/v1/video/upload-file` with live progress percentage.
5. `UPLOAD_COMPLETED`: Server verified file, returned `video_id`, preview active in central player, "Start Processing" enabled.
6. `UPLOAD_FAILED`: Display server-reported rejection reason with a one-click "Retry" action.

---

## 3. Files Modified & Added

| Component | File Path | Nature of Changes |
| :--- | :--- | :--- |
| **Backend Schemas** | `src/api/schemas.py` | Added `VideoUploadResponse` and `StartProcessingRequest` models. |
| **Backend Video Router**| `src/api/routers/video.py` | Added `.webm`, magic-bytes validator, `POST /upload-file`, `POST /start-processing`, `GET /stream/{video_id}`, and `POST /jobs/{job_id}/cancel`. |
| **Backend Job Manager** | `src/api/job_manager.py` | Added `_uploaded_videos` registry, `register_uploaded_video`, `get_uploaded_video`, and cancellation callback integration. |
| **Frontend API Client** | `frontend/src/services/api.ts` | Added `uploadVideoFile`, `startProcessingVideo`, `getVideoStreamUrl`, and cancel integration. |
| **Frontend Job Context**| `frontend/src/context/JobContext.tsx`| Added `uploadedVideoId`, `stagedVideoSrc`, `setStagedVideo`, `clearStagedVideo`, and atomic session isolation. |
| **Frontend Upload UI**  | `frontend/src/components/VideoUploadSection.tsx` | Implemented 6-state machine, upload progress, metadata preview, and two-step Start Processing flow. |
| **Frontend Canvas**    | `frontend/src/components/VideoCanvas.tsx` | Enabled video mounting on staged uploads, server streams, and active jobs. |
| **Frontend Modal**     | `frontend/src/components/VideoUploadModal.tsx` | Updated format support (.webm) and delegated to two-step upload flow. |

---

## 4. Verification & Automated Test Results

The upload pipeline was verified with automated test suites:
- `tests/api/test_video_endpoints.py`: 9 passed in 28.96s
- `tests/integration/test_upload_and_processing_pipeline.py`: 7 passed in 26.24s

### Test Matrix Summary:
1. **Valid MP4 Upload (`test_01_upload_valid_video_returns_video_id_and_metadata`):** PASSED (returns `vid_...`, metadata width=128, height=128, fps=10.0).
2. **Start Processing from ID (`test_02_start_processing_from_video_id`):** PASSED (dispatches job `job_...` in `QUEUED`/`PROCESSING` state).
3. **Magic Byte Rejection (`test_03_corrupt_and_unsupported_rejection`):** PASSED (rejects fake MP4, `.exe`, and 0-byte file with 400).
4. **HTTP 206 Byte Range Streaming (`test_06_video_streaming_http_206_partial_content`):** PASSED (streams byte ranges `0-127` with HTTP 206).
5. **Session Isolation (`test_07_session_isolation_and_cleanup`):** PASSED (generates unique IDs across sessions with zero collision).
