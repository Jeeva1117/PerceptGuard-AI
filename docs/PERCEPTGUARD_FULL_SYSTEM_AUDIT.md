# PERCEPTGUARD-AI — FULL SYSTEM AUDIT REPORT
**Product Identity:** PerceptGuard-AI  
**Category:** Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  
**Date of Audit:** November 2026  
**Auditors:** Coordinated Engineering Team (Staff Software, ML/DL, Computer Vision, MOT, ADAS Safety, Frontend, Backend & QA)

---

## 1. EXECUTIVE SUMMARY

PerceptGuard-AI is an autonomous-driving-inspired road perception and safety assistance prototype engineered for CPU-bounded offline and edge deployment. The system ingests forward-facing road video streams and performs multi-stage perception, object tracking, trajectory lookahead forecasting, collision risk assessment, and actionable safety decision support across **strictly Five Primary Modules**:

1. **Camera & Weather Intelligence:** Continuous optical visibility scoring (0–100%), atmospheric condition classification (Rain, Fog, Glare, Blur, Low Light), and lookahead horizon forecasting.
2. **Object Detection & Tracking:** YOLO11n object detection paired with CPU-optimized ByteTrack multi-object tracking, assigning deterministic, persistent tracking IDs and formal lifecycle states (`TENTATIVE`, `CONFIRMED`, `LOST`, `REIDENTIFIED`, `REMOVED`).
3. **Object & Trajectory Prediction:** Kinematic state estimation ($v_x, v_y, a_x, a_y$, heading, speed in km/h), observed trajectory history (solid line), and multi-horizon future waypoint forecasting ($H_1 = 1\text{s}, H_3 = 3\text{s}, H_5 = 5\text{s}$, dashed lines) with uncertainty covariance radii.
4. **Collision & Road Risk Assessment:** Evidence-driven Time-To-Collision ($\text{TTC}$) evaluation, closing velocity validation, spatio-temporal trajectory conflict detection, pedestrian intent analysis, and explicit distinction between lookahead risk prediction and physical collision events.
5. **Early Warning & Vehicle Stop Recommendation:** Transparent ADAS decision support (`NORMAL`, `CAUTION`, `HAZARD`, `STOP RECOMMENDED`), deduplicated event alerting, voice advisories, and explainable evidence chains.

---

## 2. AUDIT FINDINGS & ROOT CAUSES

### 2.1 Frontend Architecture & Video Playback Interruption (CRITICAL)
- **Previous Defect:** Each module page (`CameraWeatherPage`, `ObjectDetectionPage`, `TrajectoryPredictionPage`, `CollisionRiskPage`, `SafetyAlertsPage`) mounted its own isolated `<VideoCanvas />` and `<PlaybackControls />`. When a user clicked between primary sidebar features, React destroyed the `<video>` element, causing video reload, stream buffering, playhead reset to frame 0, and screen flash.
- **Root Cause:** Page-level encapsulation of video playback elements rather than workspace-level persistence.
- **Resolution:** Redesigned into a unified **Video-First Architecture** (`PersistentVideoWorkspace`). The video canvas, playback controls, and interactive event timeline are mounted once at the workspace level and persist across module switches. Route changes update only the right-side dynamic telemetry panel.

### 2.2 Proximity-Only False Collision Alarms (CRITICAL)
- **Previous Defect:** Normal driving scenes (e.g. following a lead vehicle at 45 km/h, parallel traffic in adjacent lanes, or parked roadside vehicles) frequently triggered `CRITICAL RISK` and `COLLISION DETECTED` alarms.
- **Root Cause:** Proximity distance thresholds ($d < 30\text{m}$) were evaluated without requiring closing kinematics ($v_y > 0$) or verified trajectory overlap. Furthermore, `recommendation_rules.py` contained an artificial bias `if len(tracks) > 0: state = CAUTION`, preventing `NORMAL` state from ever rendering whenever any car was present.
- **Resolution:**
  - Removed artificial track count checks.
  - Required closing velocity ($v_y > 5\text{ px/s}$) and $\text{TTC} \le 3.5\text{s}$ before triggering collision risk.
  - Enforced lateral lane boundaries: vehicles in adjacent lanes with constant lateral offset maintain `CollisionRiskState.LOW`.

### 2.3 Pedestrian Intent & Roadway Context Blindness (CRITICAL)
- **Previous Defect:** Pedestrians walking safely on sidewalks or stationary at bus stops triggered emergency collision alarms and voice commands.
- **Root Cause:** The perception pipeline lacked spatial road zone awareness (sidewalk vs road edge vs roadway) and pedestrian velocity vector analysis.
- **Resolution:**
  - Implemented `RoadContextAnalyzer` to detect lane perspective, vanishing points, and road zones (`SIDEWALK`, `ROAD_EDGE`, `ROADWAY`, `CROSSWALK`).
  - Implemented `PedestrianAwarenessEngine` with 5 strict decision states:
    1. `PERSON DETECTED`: Sidewalk presence, zero roadway incursion $\rightarrow$ `LOW` risk, silent.
    2. `PEDESTRIAN APPROACHING ROAD`: Moving toward roadway edge $\rightarrow$ `CAUTION`, monitor trajectory.
    3. `PEDESTRIAN CROSSING`: Active crossing in crosswalk or roadway with clearance buffer $\rightarrow$ `CAUTION`, yield advisory.
    4. `POTENTIAL PEDESTRIAN COLLISION RISK`: Trajectory incursion into vehicle corridor with $\text{TTC} \le 3.5\text{s}$ $\rightarrow$ `HIGH`/`CRITICAL`, prepare to stop.
    5. `PEDESTRIAN TRACKING LOST`: Tracking occluded $\rightarrow$ maintain caution.

### 2.4 Tracker State Machine Incompleteness (MODERATE)
- **Previous Defect:** ByteTrack records only emitted `track_status="active"`, lacking standard multi-object tracking lifecycle stages.
- **Resolution:** Extended `TrackRecord` and `ByteTrackEngine` to maintain lifecycle states:
  - `TENTATIVE`: Initial detection observed for $< 3$ consecutive frames.
  - `CONFIRMED`: Established track observed across $\ge 3$ frames.
  - `LOST`: Track unassociated in current frame but retained in buffer.
  - `REIDENTIFIED`: Track re-associated after being lost.
  - `REMOVED`: Stale track purged after buffer timeout.

### 2.5 Long Video Timeline Truncation (MODERATE)
- **Previous Defect:** 3-minute video uploads were truncated after ~10 seconds (~300 frames) due to rigid processing budgets.
- **Resolution:** Implemented `VideoTimelineTelemetry` with adaptive uniform striding. The pipeline samples frames evenly across the entire duration ($100\%$ timeline coverage) while strictly adhering to CPU time and memory budgets.

---

## 3. SUBSYSTEM ARCHITECTURAL VERIFICATION

| Subsystem | Audit Status | Key Guarantee |
| :--- | :---: | :--- |
| **Video Decoding** | VERIFIED | Sequential OpenCV `VideoCapture` with uniform striding; auto-detects duration and FPS. |
| **YOLO11n Detector** | VERIFIED | One-pass inference, pure CPU execution, bounding box normalization $[x_1, y_1, x_2, y_2]$. |
| **ByteTrack MOT** | VERIFIED | Pure CPU Kalman filter + SciPy Hungarian assignment fallback; persistent IDs starting at 1. |
| **Trajectory Predictor** | VERIFIED | Constant-acceleration + Kalman projection; distinct $H_1, H_3, H_5$ horizons with uncertainty radii. |
| **Collision Classifier** | VERIFIED | Closing kinematics enforced; zero alarms during parallel cruising or safe headway. |
| **Pedestrian Engine** | VERIFIED | 5-state road context decision machine; voice alert deduplication with cooldown. |
| **Safety Engine** | VERIFIED | Deterministic rules; transparent evidence chain; `NORMAL` state on safe roads. |
| **Frontend Workspace** | VERIFIED | Video-first continuous playback; zero unmounting across module switches; click-to-seek timeline. |

---

## 4. AUDIT CONCLUSION

All identified deficiencies—including false collision screaming, video player reload on module navigation, sidewalk pedestrian false alarms, tracker lifecycle omission, and timeline truncation—have been systematically resolved and verified by automated regression test suites. The PerceptGuard-AI system operates strictly within its 5-module specification and provides defensible, evidence-driven autonomous road safety intelligence.
