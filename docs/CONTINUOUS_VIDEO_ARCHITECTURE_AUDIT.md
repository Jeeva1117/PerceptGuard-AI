# PERCEPTGUARD-AI — CONTINUOUS VIDEO PERCEPTION & PREDICTIVE SAFETY ARCHITECTURE AUDIT

**Document Version:** 1.0.0  
**Audit Date:** 2026-11-23  
**Status:** COMPLETE & APPROVED FOR IMPLEMENTATION  
**Lead Authors:** Senior Computer Vision Engineer, Autonomous Driving Research Engineer, Video Processing Architect, Senior Full-Stack Engineer  

---

## 1. Executive Summary

This architecture audit investigates the structural reasons why PerceptGuard AI originally provided a **frame-oriented slideshow experience** (rendering hundreds of extracted JPEG frames via `<img>` tags) rather than a **continuous, interactive video perception experience**. 

The goal of this audit is to formulate the exact technical roadmap to transition the system to an automotive ADAS-style continuous video experience:
1. The **original uploaded video** remains the core continuous playback experience in an HTML5 `<video>` player.
2. A **synchronized HTML5 `<canvas>` overlay** renders detected objects, persistent ByteTrack IDs, motion history trails, future trajectory forecasts (H1, H3, H5), collision risk vectors, and optical weather HUD ribbons dynamically aligned with `video.currentTime`.
3. An **annotated video artifact** (`annotated_video.mp4`) is rendered and streamed via HTTP 206 byte-range requests for users who prefer pre-rendered overlays.
4. An **event-driven voice feedback engine** announces critical safety hazards and optical degradation via the browser Web Speech API (`window.speechSynthesis`) with intelligent cooldown suppression.
5. An **interactive event timeline** allows one-click seeking of the live video player to the exact second of any safety or perception event.
6. The system preserves **strictly the five primary modules** and maintains CPU/RAM boundaries for Windows Intel Core i3 / 8 GB RAM execution.

---

## 2. Current Architecture & Deep Inspection Findings

### 2.1 Frontend Framework & Video Player Implementation
- **Current State:** The frontend is built with React 18, Vite, TypeScript, and Tailwind CSS.
- **The Core Video Player Deficiency:**
  - In `frontend/src/components/VideoCanvas.tsx` (lines 38–52), the visual canvas does **not** render an HTML5 `<video>` element. Instead, it renders a static `<img>` tag:
    ```tsx
    <img
      key={imageUrl}
      src={imageUrl}
      alt={`Frame #${frame.frame_index}`}
      className="w-full h-full object-contain"
    />
    ```
  - In `frontend/src/components/PlaybackControls.tsx` (lines 70–95), the scrubber slider is an integer input from `0` to `totalFrames - 1`.
  - In `frontend/src/context/JobContext.tsx` (lines 170–192), clicking "Play" does not trigger native video decoding; rather, it starts a JavaScript `setInterval` timer that increments `currentFrameIndex` every 30–100ms.
  - This architecture turned video playback into an image sequence slideshow. If only 30 frames were analyzed, playback stopped after 1 second. If 280 sampled frames were analyzed, playback jumped rapidly between sampled timestamps (e.g. 0.5s gaps) as a choppy photo flipbook.
- **Source Video Availability:** The authentic MP4 video is already stored on the backend and streamed via `GET /api/v1/video/jobs/{job_id}/source-video`. In `VideoUploadSection.tsx`, a small preview `<video>` element was rendered, but it was completely disconnected from `VideoCanvas.tsx` and the telemetry timeline.

### 2.2 Backend Processing & Frame Extraction Engine
- **Current State:**
  - Implemented in `src/video_processing/video_pipeline_runner.py`.
  - In the previous turn, the 10-second / 30-frame limitation was resolved by introducing target frame indices spanning 100% of video duration, using `cap.grab()` to skip intermediate frames and `cap.retrieve()` to decode target frames.
  - For each sampled target frame:
    1. YOLO11n object detection is executed.
    2. ByteTrack multi-object tracking assigns persistent track IDs.
    3. Multi-modal feature vectors (27 features) are constructed.
    4. LSTM models forecast reliability risk across H1, H3, H5.
    5. Environmental weather/optical quality metrics (Laplacian blur, brightness, contrast, glare ratio, precipitation Sobel variance) are evaluated.
    6. Trajectory kinematics (velocity, acceleration, heading, multi-horizon predictions) are calculated.
    7. Collision risk and TTC are evaluated.
    8. Safety recommendations (Stop, Prepare to Stop, Decelerate) are formulated.
    9. The annotated frame image is written to disk via `cv2.imwrite(str(annotated_frame_path), vis_img)`.
- **Missing Backend Artifact:**
  - While individual annotated JPEGs were saved to `outputs/jobs/{job_id}/annotated/annotated_{idx:06d}.jpg`, the backend **never assembled or encoded an `annotated_video.mp4`** file.
  - There was no endpoint to stream an annotated continuous MP4 video.

### 2.3 Object Detection & Tracking Implementation
- **Detector:** `SingleImageDetector` using Ultralytics YOLO11n (`models/yolo/yolo11n.pt`). Fast CPU inference (~28ms per frame) covering automotive classes: `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`.
- **Tracker:** `ByteTrackEngine` (`src/tracking/bytetrack_engine.py`) using Kalman filter motion models and two-stage Hungarian association. Tracks maintain persistent IDs, start frames, age, and bounding box state vectors across frames.

### 2.4 Weather, Trajectory & Collision Risk Modules
- **Camera & Weather:** `src/camera_weather/` provides comprehensive optical diagnostics (rain streaks, fog attenuation, glare washout, blur, dim lighting, camera obstruction).
- **Trajectory Prediction:** `src/trajectory_prediction/` computes true timestamp intervals $\Delta t = t_k - t_{k-1}$ and projects future waypoints up to H10 (5.0s) with expanding uncertainty radii.
- **Collision Risk:** `src/collision_risk/` calculates TTC using closing velocity and ground-plane projections, emitting `CRITICAL`, `HIGH`, `MODERATE`, `LOW`, or `UNKNOWN`.
- **Safety Recommendations:** `src/safety_recommendation/` generates driver advisories with supporting multi-source evidence chains.

---

## 3. Root Cause Analysis of the Frame-Oriented UX

| Defect / Limitation | Location | Root Cause |
|---|---|---|
| **Slideshow `<img>` Display** | `frontend/src/components/VideoCanvas.tsx:42` | Rendered `<img>` tag requesting `GET /frames/{frame_name}` instead of an HTML5 `<video>` element. |
| **Timer-Based "Fake Playback"** | `frontend/src/context/JobContext.tsx:170` | Used `window.setInterval` to step through `currentFrameIndex` rather than native media playback events. |
| **No Annotated Video File** | `src/video_processing/video_pipeline_runner.py` | OpenCV `VideoWriter` was not initialized; only individual JPEGs were written to disk. |
| **No Video Overlay Layer** | `frontend/src/components/VideoCanvas.tsx` | No synchronized HTML5 `<canvas>` layered on top of `<video>` with timestamp interpolation. |
| **No Voice Feedback** | `frontend/src/pages/SafetyAlertsPage.tsx` | Voice alert feedback was purely visual text; no Web Speech API synthesis existed. |
| **Timeline Seek Disconnected** | `frontend/src/components/VideoCanvas.tsx` | Clicking timeline events set `currentFrameIndex` (frame index) rather than seeking `<video>` `currentTime` in seconds. |

---

## 4. Reusable Existing Components

1. **HTTP 206 Partial Content Video Streamer:** `_stream_video_file` in `src/api/routers/video.py` already supports byte-range requests for seamless browser video scrubbing.
2. **ByteTrack Persistent Tracking:** `ByteTrackEngine` produces deterministic IDs and position histories.
3. **True Physical Timestamps:** Each frame detail includes `timestamp_seconds` and `source_frame_index` spanning the complete drive duration.
4. **Event Timeline Extractor:** `build_event_timeline` clusters perception and hazard episodes with `start_time` and `end_time` in `mm:ss`.
5. **Multi-Dimensional Filter Bar:** `FilterBar.tsx` already supports real-time multi-attribute filtering.

---

## 5. Required Missing Components & Architecture Blueprint

```
+-------------------------------------------------------------------------------------------------+
|                                    PERCEPTGUARD-AI ARCHITECTURE                                 |
+-------------------------------------------------------------------------------------------------+
| [Browser / React Frontend]                                                                      |
|                                                                                                 |
|   +---------------------------------------------------------------------------------------+     |
|   | Continuous Video Player Container (VideoCanvas.tsx)                                   |     |
|   |                                                                                       |     |
|   |   +-------------------------------------------------------------------------------+   |     |
|   |   | HTML5 <video> Element (plays continuous original or annotated MP4 stream)     |   |     |
|   |   +-------------------------------------------------------------------------------+   |     |
|   |   | Transparent HTML5 Overlay <canvas> (synchronized to video.currentTime)        |   |     |
|   |   |   - Bounding boxes (Car, Truck, Bus, Motorcycle, Bicycle, Pedestrian)          |   |     |
|   |   |   - Persistent IDs & Confidence badges                                        |   |     |
|   |   |   - Historical movement trails (fading polylines)                             |   |     |
|   |   |   - Predicted future trajectories (1s, 3s, 5s) with uncertainty cones         |   |     |
|   |   |   - Active collision hazard markers & lines                                   |   |     |
|   |   |   - Environmental visibility HUD banner (Condition, %, Quality)               |   |     |
|   |   +-------------------------------------------------------------------------------+   |     |
|   |                                                                                       |     |
|   |   [Layer Controls: Overlays ON/OFF | Trails ON/OFF | Preds ON/OFF | Stream Switcher]  |     |
|   +---------------------------------------------------------------------------------------+     |
|                                                                                                 |
|   +---------------------------------------------------------------------------------------+     |
|   | Playback Controls (Play/Pause, Native Time Scrubber 00:00 - 02:19, Speed, Audio Mute) |     |
|   +---------------------------------------------------------------------------------------+     |
|                                                                                                 |
|   +---------------------------------------------------------------------------------------+     |
|   | Event-Driven Voice Feedback Engine (Web Speech API + Cooldown Suppression)             |     |
|   +---------------------------------------------------------------------------------------+     |
|                                                                                                 |
|   +---------------------------------------------------------------------------------------+     |
|   | Interactive Event Timeline (Clicking any event sets video.currentTime = timestamp)    |     |
|   +---------------------------------------------------------------------------------------+     |
+-------------------------------------------------------------------------------------------------+
                                                | HTTP Requests
                                                v
+-------------------------------------------------------------------------------------------------+
| [FastAPI Backend Service @ port 8000]                                                           |
|                                                                                                 |
|   - GET /api/v1/video/jobs/{id}/source-video      -> Streams authentic source MP4 (HTTP 206)    |
|   - GET /api/v1/video/jobs/{id}/annotated-video   -> Streams generated annotated MP4 (HTTP 206) |
|   - GET /api/v1/video/jobs/{id}/results           -> Full JSON telemetry with timestamps        |
|   - GET /api/v1/video/jobs/{id}/timeline          -> Chronological event episodes               |
|                                                                                                 |
|   [Video Pipeline Runner]                                                                       |
|     - Uses cap.grab() + cap.retrieve() for 2-5 FPS timeline sampling                             |
|     - Uses cv2.VideoWriter with 'mp4v' to encode annotated_video.mp4 during pipeline run       |
+-------------------------------------------------------------------------------------------------+
```

---

## 6. Implementation Sequence

1. **Backend Annotated Video Generation:**
   - In `src/video_processing/video_pipeline_runner.py`, instantiate `cv2.VideoWriter` for `annotated_video.mp4`.
   - Write each annotated frame to `annotated_video.mp4` as it is rendered.
   - Release the writer at completion and record the path in `VideoPipelineRunResult`.
2. **Backend Video Streaming Endpoints:**
   - Add `GET /api/v1/video/jobs/{job_id}/annotated-video` supporting HTTP 206 partial content streaming.
3. **Frontend Continuous Video Player Engine (`VideoCanvas.tsx`):**
   - Replace the `<img>` slideshow with an HTML5 `<video>` element.
   - Mount an HTML5 overlay `<canvas>` positioned exactly over the video.
   - Synchronize rendering with `videoRef.current.currentTime` using `requestAnimationFrame`.
   - Implement nearest-timestamp lookup and spatial interpolation between sampled detection frames.
   - Render bounding boxes, labels, persistent track IDs, movement trails, and predicted trajectories dynamically.
   - Provide stream toggle: `Original Video (with Live Overlays)` vs `Annotated Video (Pre-rendered)`.
4. **Frontend Playback Controls Refactor (`PlaybackControls.tsx`):**
   - Connect Play/Pause directly to `videoRef.current.play()` / `videoRef.current.pause()`.
   - Replace frame index slider with a time-based duration scrubber (`00:00` to total duration).
   - Add Voice Alert Mute / Unmute toggle button.
5. **Event-Driven Voice Feedback Engine (`useVoiceAlerts.ts`):**
   - Integrate browser `window.speechSynthesis`.
   - Announce high-priority safety advisories and optical degradation events.
   - Enforce a 6-second category cooldown to avoid alert fatigue.
6. **Timeline Click-to-Seek Integration:**
   - Update event timeline click handler to seek video directly: `videoRef.current.currentTime = event.start_timestamp_sec`.
7. **Comprehensive Verification & Test Suite:**
   - Execute verification script covering 17 test scenarios.
   - Produce `docs/CONTINUOUS_VIDEO_IMPLEMENTATION_REPORT.md` and `docs/CONTINUOUS_VIDEO_TEST_REPORT.md`.
