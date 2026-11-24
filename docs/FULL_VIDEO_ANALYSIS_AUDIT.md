# PERCEPTGUARD-AI — FULL VIDEO INTELLIGENCE & VISIBILITY FAULT DETECTION AUDIT

**System:** PerceptGuard-AI Autonomous Driving Predictive Safety System  
**Audit Target:** Video Duration Handling, Frame Sampling, Visibility Fault Detection, Event Timeline & Alert Architecture  
**Date:** November 23, 2026  
**Auditor:** Senior Computer Vision Engineer, Senior ML Engineer & Backend Architect  
**Classification:** Engineering Architecture & Defect Audit  

---

## 1. EXECUTIVE SUMMARY & THE 10-SECOND RESTRICTION PROBLEM

During real-world evaluation with long dashcam and traffic recordings (e.g. WhatsApp video recordings of 2.32 minutes / 139.07s or ~2.9 minutes / 174s), operators observed that **only the first ~1 to 10 seconds of video were ever processed**, regardless of the overall video length.

Specifically, for a 139.07-second video containing 4,173 frames at 30.01 FPS:
- The system extracted and processed only **30 frames** (0.97 seconds) or **60 frames** (1.97 seconds).
- The remaining **137+ seconds (98.6% of the video)** were completely ignored.
- No events, weather changes, fog, glare, rain, pedestrian crossings, or collision risks occurring after the first 2 seconds were ever observed or detected.
- The user interface labeled the 60-frame run as "Comprehensive (~16s)" and displayed `COMPLETED`, misleading operators into believing the full video had been evaluated.

---

## 2. ROOT CAUSE ANALYSIS & PROVEN CODEBASE EVIDENCE

An exhaustive audit of the frontend, backend routers, job execution, and computer vision pipelines identified **six specific root causes** responsible for the timeline truncation:

```
+---------------------------------------------------------------------------------------------------------+
|                                    PROVEN ROOT CAUSE INTERACTION CHAIN                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ Frontend UI: VideoUploadSection.tsx & VideoUploadModal.tsx ]                                         |
|       |                                                                                                 |
|       +--> Hardcoded Frame Budget selector: only [15, 30, 60] frames offered                            |
|       +--> Slider hardcoded: min="10", max="60"                                                          |
|       +--> Dispatches max_frames=30 or 60 as a contiguous slice count                                   |
|                 |                                                                                       |
|  [ FastAPI Endpoints: src/api/routers/video.py ]                                                         |
|       |                                                                                                 |
|       +--> Query Param Constraint: max_frames: int = Query(30, ge=5, le=100)                             |
|       +--> FastAPI Pydantic validator strictly rejects any request > 100 frames!                        |
|                 |                                                                                       |
|  [ Job Manager: src/api/job_manager.py ]                                                                |
|       |                                                                                                 |
|       +--> JobRecord defaults: max_frames_requested: int = 30                                           |
|       +--> Passes max_frames integer to VideoPipelineRunner.run()                                       |
|                 |                                                                                       |
|  [ Pipeline Runner: src/video_processing/video_pipeline_runner.py ]                                     |
|       |                                                                                                 |
|       +--> Line 624: frames_to_process = min(max_frames, total_container_frames)                         |
|       +--> Lines 649-653: while frame_idx < frames_to_process: ret, frame = cap.read()                  |
|       +--> Reads 30 or 60 CONSECUTIVE frames from position 0!                                          |
|       +--> Line 671: timestamp_sec = float(frame_idx) / fps -> stops at t=0.97s or 1.97s!              |
|       +--> Unconditionally breaks out after frame 30 or 60.                                             |
|       +--> Zero stride, zero sampling across duration, 98% of container frames never decoded!           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 2.1 Proven Evidence in `src/video_processing/video_pipeline_runner.py`
* **File Reference:** `src/video_processing/video_pipeline_runner.py`
* **Lines 532, 624, 649–653, 671:**
  ```python
  # Line 532:
  def run(self, video_path, job_id, output_dir, max_frames: int = 30, rotation: int = 0, ...):
      ...
      # Line 624:
      frames_to_process = min(max_frames, total_container_frames if total_container_frames > 0 else max_frames)
      ...
      # Lines 649-653:
      frame_idx = 0
      while frame_idx < frames_to_process:
          ret, frame_bgr = cap.read()
          if not ret or frame_bgr is None:
              break
          ...
          # Line 671:
          timestamp_sec = float(frame_idx) / fps
  ```
* **Analysis:**
  - `cap.read()` decodes consecutive frames starting at frame 0.
  - When `max_frames = 30`, it reads frames 0 through 29.
  - At 30 FPS, frame 29 corresponds to $29 / 30 = 0.967\text{ seconds}$.
  - At 60 frames, it reaches $59 / 30 = 1.967\text{ seconds}$.
  - The loop immediately terminates upon reaching `frame_idx == frames_to_process`.
  - There was **zero frame stride or temporal downsampling** implemented.
  - The rest of the video (from 00:02 to 02:54) was never reached or analyzed.

### 2.2 Proven Evidence in `src/api/routers/video.py`
* **File Reference:** `src/api/routers/video.py`
* **Lines 180, 244, 296:**
  ```python
  # Line 180:
  async def process_video(file: UploadFile = File(...), max_frames: int = Query(30, ge=1, le=100, ...)):
  ...
  # Line 244:
  async def submit_video_job(file: UploadFile = File(...), max_frames: int = Query(30, ge=5, le=100, ...)):
  ...
  # Line 296:
  async def upload_video(file: UploadFile = File(...), max_frames: int = Query(30, ge=5, le=100, ...)):
  ```
* **Analysis:**
  - The FastAPI query parameter was strictly constrained with `le=100`.
  - Any client attempting to request more than 100 frames received an HTTP 422 Unprocessable Entity validation failure.
  - No parameter existed to specify an **Analysis Mode** (`FULL_BALANCED`, `FULL_DETAILED`, `QUICK_PREVIEW`) or a **Target Sampling Rate** (e.g. 2 FPS, 5 FPS).

### 2.3 Proven Evidence in Frontend Controls
* **File References:**
  - `frontend/src/components/VideoUploadSection.tsx`, lines 38, 407–418:
    ```tsx
    const [maxFrames, setMaxFrames] = useState<number>(30);
    ...
    <select value={maxFrames} onChange={(e) => setMaxFrames(Number(e.target.value))}>
      <option value="15">15 Frames (Ultra-Fast ~3s)</option>
      <option value="30">30 Frames (Recommended CPU ~8s)</option>
      <option value="60">60 Frames (Comprehensive ~16s)</option>
    </select>
    ```
  - `frontend/src/components/VideoUploadModal.tsx`, lines 14, 126–130:
    ```tsx
    <input type="range" min="10" max="60" step="5" value={maxFrames} ... />
    ```
* **Analysis:**
  - The UI offered only fixed frame counts (15, 30, 60) without clarifying that they represented a contiguous initial slice rather than full-video sampling.
  - The label "Comprehensive" was misleading because 60 consecutive frames only covers ~2 seconds of 30 FPS video.

### 2.4 Contrast with Dormant Implementation in `src/video_processing/frame_extractor.py`
* **File Reference:** `src/video_processing/frame_extractor.py`, lines 324–379:
  ```python
  def _compute_target_source_indices(self, source_fps: float, total_source_frames: int) -> List[int]:
      step = source_fps / self._extraction_fps
      ...
      while True:
          idx = int(round(k * step))
          if idx >= total_source_frames:
              break
          ...
  ```
* **Analysis:**
  - A sampling algorithm calculating `step = source_fps / target_fps` and skipping frames using `cap.grab()` existed in `frame_extractor.py`.
  - However, `VideoPipelineRunner` did **not** use `frame_extractor.py` or this sampling logic. It executed its own naive contiguous loop.

---

## 3. AUDIT OF WEATHER, OBJECT DETECTION, TRAJECTORIES, RISK & ALERTS ACROSS FULL TIMELINES

### 3.1 Feature 1: Camera & Weather Intelligence
* **Current State:**
  - `VisibilityQualityEngine` computes sharpness, contrast, luminance, exposure, blur, fog, rain, glare, and obstruction per frame.
  - `TemporalWeatherTracker` uses an exponential moving average (`history_window=30`).
* **Timeline Deficiency:**
  - When only 30 contiguous frames (1 second) are processed, the temporal tracker never sees weather transitions (e.g. entering rain at 00:35, sudden glare at 01:15, or lens fogging at 02:00).
  - Condition summary only reports the condition of the first second.
* **Required Upgrade:**
  - Maintain temporal smoothing across chronologically sampled frames spanning the entire video.
  - Cluster degraded frames into localized continuous fault segments: `[start_time, end_time, condition, severity, evidence, confidence]`.
  - Clearly distinguish weather (rain, fog) from optical sensor faults (defocus blur, sensor overexposure, obstruction).

### 3.2 Feature 2: Object Detection & Tracking (YOLO + ByteTrack)
* **Current State:**
  - YOLO11n executes per frame, detects 80 classes, filters COCO traffic classes (car, truck, bus, motorcycle, bicycle, person).
  - ByteTrack matches detections using Kalman filter and IoU association.
* **Timeline Deficiency:**
  - When sampling at 2 FPS or 5 FPS, the time delta between sampled frames is $\Delta t = 0.5\text{s}$ (at 2 FPS) or $\Delta t = 0.2\text{s}$ (at 5 FPS).
  - ByteTrack's Kalman filter assumes $\Delta t = 1 / \text{frame\_rate}$. If initialized with `frame_rate=30` while frames arrive at 2 FPS, the Kalman velocity updates will be distorted!
* **Required Upgrade:**
  - ByteTrack's `frame_rate` must be parameterized to match the **effective sampling frame rate** (`effective_fps = 2` or `5`), ensuring smooth track continuity and accurate velocity estimates across seconds.

### 3.3 Feature 3: Object & Trajectory Prediction
* **Current State:**
  - `MultimodalTrajectoryForecaster` computes velocity, acceleration, and predicts positions at $H=1\text{s}, 3\text{s}, 5\text{s}$.
* **Timeline Deficiency:**
  - Assumed consecutive frame intervals ($\Delta t \approx 0.033\text{s}$).
* **Required Upgrade:**
  - Compute velocities and kinematics using actual physical timestamps: $v_x = \frac{\Delta x}{\Delta t_{\text{seconds}}}$.
  - Clearly communicate prediction uncertainty and explicitly label pixel distances as image-space projections rather than uncalibrated metric distances.

### 3.4 Feature 4: Collision & Road Risk Assessment
* **Current State:**
  - `CollisionRiskEngine` calculates Time-To-Collision (TTC) and fuses optical degradation with trajectory conflicts.
* **Timeline Deficiency:**
  - TTC was only computed for the opening 1–2 seconds. No hazards occurring later in the drive were detected.
* **Required Upgrade:**
  - Continuous chronological risk assessment across the full timeline.
  - Missing or unreliable telemetry must produce `UNKNOWN` risk rather than defaulting to false `LOW`.

### 3.5 Feature 5: Safety Recommendations & Event Timeline
* **Current State:**
  - Frame-by-frame recommendations (`NORMAL`, `CAUTION`, `REDUCE_SPEED`, `PREPARE_TO_STOP`, `STOP_RECOMMENDED`, `UNKNOWN`).
  - No aggregate event timeline allowing an operator to browse the entire video duration.
* **Timeline Deficiency:**
  - The UI had no clickable event list showing where in the 2.9-minute video anomalies occurred.
* **Required Upgrade:**
  - Construct an end-to-end **Complete Video Event Timeline**:
    - Segment time ranges (e.g. `00:00–00:24 Normal`, `00:25–00:48 Rain-like conditions`, `01:10–01:30 Optical Blur`).
    - Clicking any event jumps to that timestamp in the video preview, displays the evidence frame, and highlights involved tracks.

---

## 4. AUDIT OF FRONTEND FILTERS

* **Current State:**
  - `ObjectDetectionPage.tsx` had a single slider for `confFilter`.
  - No unified filter bar existed for:
    1. Event Type (Weather, Glare, Blur, Collision Risk, Stop Advisory).
    2. Severity Level (Critical, High, Moderate, Low, Info).
    3. Time Range (Start Seconds to End Seconds).
    4. Object Class (Car, Truck, Bus, Motorcycle, Bicycle, Pedestrian).
    5. Track ID.
    6. Confidence Threshold.
    7. Visibility Condition (Clear, Rain, Fog, Blur, Low Light, Glare).
  - No "Reset Filters" action existed.
* **Required Upgrade:**
  - Build a centralized, reusable Filter Bar and filter state manager.
  - Ensure filters operate on displayed results in memory without requiring a full re-inference pipeline run.

---

## 5. AUDIT SUMMARY TABLE

| Area | Prior Implementation | Root Cause / Defect | Required Architecture |
| :--- | :--- | :--- | :--- |
| **Duration Coverage** | 10–60 frames (0.3s to 2.0s) | Contiguous `cap.read()` loop breaking at `max_frames` | Chronological sampling across 100% of video timeline |
| **FastAPI Bound** | `Query(30, le=100)` | Artificial schema limit capped at 100 frames | Configurable modes: `FULL_BALANCED` (2 FPS), `FULL_DETAILED` (5 FPS), `QUICK_PREVIEW` |
| **UI Selection** | Dropdown with [15, 30, 60] frames | Missing Analysis Mode selector | Analysis Mode selector with explicit sampling FPS and duration estimates |
| **Video Sampling** | Direct contiguous read | Missing stride / grab logic | `cap.grab()` fast-forward to target indices across the entire video |
| **Timeline Navigation**| None; single-frame view | No event aggregation or temporal clustering | Interactive event timeline with one-click timestamp seeking |
| **Filters** | Only 1 slider on 1 page | Missing multi-dimensional filters | Comprehensive filter bar: Event, Severity, Time, Class, Track, Reset |
| **CPU Safety** | Thread bounded, but naive frame loop | Memory buffering risk if all frames decoded | Grab-and-retrieve streaming without frame buffering |

---
*Audit certified by Senior Computer Vision & Backend Architect.*
