# PerceptGuard-AI: Five-Module Acceptance & Release Report

**Document Version:** 2.0 (Release Readiness)  
**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  
**Core Scope Rule:** Strictly Five Primary Modules. No additional primary sidebar navigation tabs.  

---

## 1. System Architecture Overview

PerceptGuard-AI functions as a unified perception, tracking, prediction, and risk-assessment platform for road safety research. The system operates on standard CPU hardware with end-to-end synchronization across all five primary modules:

```
[Uploaded Dashcam Video Stream]
              │
              ├──► Module 1: Camera & Weather Intelligence
              │
              ├──► Module 2: Object Detection & Tracking
              │
              ├──► Module 3: Object & Trajectory Prediction
              │
              ├──► Module 4: Collision & Road Risk Assessment
              │
              └──► Module 5: Early Warning & Vehicle Stop Recommendation
```

---

## 2. Module-by-Module Acceptance Criteria & Verification

### Module 1: Camera & Weather Intelligence
- **Implementation**: `src/camera_weather/visibility_quality_engine.py`, `src/camera_weather/temporal_weather_tracker.py`, `frontend/src/pages/CameraWeatherPage.tsx`.
- **Capabilities**:
  - Computes 6 optical quality metrics: Laplacian blur variance, brightness mean/std, Michelson/standard deviation contrast, sun-glare saturation ratio, Sobel vertical rain variance, and edge density.
  - Classifies driving conditions into discrete categories: `CLEAR`, `RAIN`, `FOG`, `BLUR`, `SUN_GLARE`, `LOW_LIGHT`, `OBSTRUCTION`.
  - Calculates 0-100% visibility score with 15-frame exponential smoothing to eliminate high-frequency noise.
  - Forecasts visibility trends over future frames.
- **Acceptance Status**: **ACCEPTED (PASS)**
- **Automated Verification**: `test_regression_scenarios.py` (adverse rain, fog, and sun glare scenarios).

---

### Module 2: Object Detection & Tracking
- **Implementation**: `src/detection/yolo_model_loader.py`, `src/tracking/bytetrack_engine.py`, `frontend/src/pages/ObjectDetectionPage.tsx`.
- **Capabilities**:
  - YOLO11n detection filtered strictly to 6 road user classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
  - ByteTrack 5-state tracking lifecycle: `TENTATIVE`, `CONFIRMED`, `LOST`, `REIDENTIFIED`, `REMOVED`.
  - Persistent track class assignment (`_track_classes`) prevents semantic flipping across frames.
  - Low-confidence two-stage association recovers occluded and motion-blurred road users.
  - Interactive class filters, confidence slider (10% to 85%), and focus track inspection.
- **Acceptance Status**: **ACCEPTED (PASS)**
- **Automated Verification**: `test_motorcycle_and_road_users.py` (7 tests) and `test_bytetrack_engine.py` (18 tests).

---

### Module 3: Object & Trajectory Prediction
- **Implementation**: `src/trajectory_prediction/multimodal_forecaster.py`, `src/forecasting/lstm_model.py`, `frontend/src/pages/TrajectoryPredictionPage.tsx`.
- **Capabilities**:
  - Centroid trajectory history buffer (up to 40 chronological frames).
  - First-order and second-order kinematic estimation: $v_x, v_y, a_x, a_y$ and speed in km/h.
  - Multi-horizon forward predictions: H1 (+0.03s), H3 (+0.10s), H5 (+0.17s), H10 (+0.33s).
  - Physics-based quadratic extrapolation with damping to prevent diverging curves.
  - Dynamic uncertainty radii calibrated against speed and prediction horizon.
- **Acceptance Status**: **ACCEPTED (PASS)**
- **Automated Verification**: `test_regression_scenarios.py` and `test_bytetrack_engine.py`.

---

### Module 4: Collision & Road Risk Assessment
- **Implementation**: `src/collision_risk/collision_risk_engine.py`, `src/collision_risk/schemas.py`, `src/collision_risk/trajectory_conflict_analyzer.py`, `frontend/src/pages/CollisionRiskPage.tsx`.
- **Capabilities**:
  - Spatio-temporal trajectory conflict analysis: verifies both spatial intersection and arrival time synchronicity ($\Delta t \le 0.5\text{s}$).
  - Categorization into valid `ConflictType` enums: `REAR_END`, `CROSSING`, `HEAD_ON`, `MERGING`, `PEDESTRIAN_CROSSING`, `CYCLIST_CONFLICT`.
  - Positive closing velocity gate ($v_{\text{closing}} > 5.0\text{ px/s}$) eliminates false alarms on diverging vehicles.
  - Defensible Time-to-Collision (TTC) calculation and discrete 5-state product decision taxonomy: `NO COLLISION RISK`, `NEAR PASS`, `POTENTIAL COLLISION RISK`, `COLLISION OBSERVED`, `UNKNOWN`.
  - Road corridor boundaries and lane tracking.
  - Fail-safe optical uncertainty withholding when visibility falls below $40\%$.
- **Acceptance Status**: **ACCEPTED (PASS)**
- **Automated Verification**: `tests/collision_risk/test_conflict_type_regression.py` (4 tests), `tests/collision_risk/test_collision_risk.py` (18 tests), `tests/integration/test_end_to_end_scenarios_1_to_10.py` (10 scenarios).

---

### Module 5: Early Warning & Vehicle Stop Recommendation
- **Implementation**: `src/safety_recommendation/safety_recommendation_engine.py`, `src/early_warning/early_warning_engine.py`, `frontend/src/pages/SafetyAlertsPage.tsx`.
- **Capabilities**:
  - Discrete actionable driving recommendations:
    - `NORMAL_DRIVING` (Clear road, no conflicts)
    - `MAINTAIN_AWARENESS` (High traffic density or mild weather)
    - `REDUCE_SPEED` (Degraded visibility, rain, fog, or moderate TTC)
    - `PREPARE_TO_STOP` (Crossing pedestrian or high conflict probability)
    - `STOP_RECOMMENDED` (Imminent collision risk or pedestrian in path)
  - Full evidence chain linking sensor telemetry to advisory actions.
  - Timeline scrubbing with clickable events that jump the video player to exact conflict frames.
  - Clear system identity disclaimer.
- **Acceptance Status**: **ACCEPTED (PASS)**
- **Automated Verification**: `test_regression_scenarios.py` and `test_video_endpoints.py`.

---

## 3. Five-Module Acceptance Summary

| Module Number | Module Name | Primary Role | Status |
| :---: | :--- | :--- | :---: |
| **1** | Camera & Weather Intelligence | Sensor health, optical faults, weather conditions | **ACCEPTED** |
| **2** | Object Detection & Tracking | Road user perception, ByteTrack identity continuity | **ACCEPTED** |
| **3** | Trajectory Prediction | Multi-horizon future motion forecasting & kinematics | **ACCEPTED** |
| **4** | Collision & Road Risk | Spatio-temporal conflict detection, TTC gating | **ACCEPTED** |
| **5** | Safety Alerts & Stop Advisory | Actionable vehicle recommendations & evidence chains | **ACCEPTED** |

**Zero 6th sidebar modules or superfluous navigation items exist in the codebase.**

---

## 4. UI State Machine & Pipeline Stabilization Addendum

### 4.1 Root Cause & Resolution of "Start Processing" Button State
- **Root Cause**: In `PersistentVideoWorkspace.tsx`, `<VideoUploadSection />` (which contained the only Start Processing action) was conditionally unmounted and concealed inside a collapsed drawer whenever `hasVideoLoaded` evaluated to `true`. When a video was staged or uploaded, the entire action bar disappeared, leaving the user with only the video canvas and no way to trigger processing.
- **Permanent Solution**: Replaced the collapsed drawer with a permanent, un-collapsible, unified **Video Control & Processing Action Bar** in `PersistentVideoWorkspace.tsx` that remains mounted and visible above the `VideoCanvas` across all five modules.
- **State Machine Integration**: Centralized `VideoProcessingState` (`NO_VIDEO`, `UPLOADING`, `VIDEO_READY`, `PROCESSING`, `COMPLETED`, `UPLOAD_FAILED`, `PROCESSING_FAILED`, `CANCELLED`) within `JobContext.tsx`.
- **Status Display Fix**: In `CameraWeatherPage.tsx`, updated the status badge to accurately show `Video Ready for Processing` when a video is loaded instead of misleading `Ready for Upload`.

### 4.2 Backend `ConflictType` Resolution & Regression Verification
- **Issue**: `NameError: name 'ConflictType' is not defined` occurred in `src/collision_risk/collision_risk_engine.py` during job execution.
- **Fix**: Explicitly imported `ConflictType` into `src/collision_risk/collision_risk_engine.py` with safe enum and string introspection.
- **Automated Verification**: `tests/collision_risk/test_conflict_type_regression.py` (4/4 passed) and `tests/integration/test_end_to_end_scenarios_1_to_10.py` (10/10 passed).

