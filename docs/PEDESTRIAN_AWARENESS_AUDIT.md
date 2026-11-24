# PERCEPTGUARD-AI — ADVANCED PEDESTRIAN-AWARE CONTINUOUS DRIVING INTELLIGENCE AUDIT
## TECHNICAL AUDIT & SPECIFICATION

**Date:** November 24, 2026  
**Author:** PerceptGuard-AI Senior ADAS Computer Vision & Safety Engineering Group  

---

### 1. Pedestrian Awareness Context

In automotive perception, vulnerable road users (VRUs)—specifically pedestrians, cyclists, and scooter riders—require fundamentally different kinematic models and hazard classification rules than enclosed passenger vehicles:
1. **Kinematics:** Pedestrians have near-zero minimum turning radius, can change lateral heading instantaneously ($> 90^\circ$ turn in $< 0.3\,\text{s}$), and travel at lower velocities ($1.0 - 2.5\,\text{m/s}$ or $3.6 - 9.0\,\text{km/h}$).
2. **Spatial Risk Zones:** A pedestrian walking on an elevated sidewalk parallel to the vehicle presents **zero collision hazard**, whereas a pedestrian taking two steps off a curb into the roadway presents an **immediate critical hazard**.
3. **Driver Alert Fatigue:** Triggering emergency collision alarms whenever a pedestrian is visible on a crowded sidewalk destroys driver trust. The system must distinctly separate:
   - *Awareness* (knowing a person is on the sidewalk)
   - *Caution* (observing a person walking toward the curb)
   - *Imminent Risk* (spatio-temporal conflict in the vehicle travel path)

---

### 2. Five Pedestrian Awareness Decision States

| Decision State | Spatial Context | Motion Trajectory | Collision Risk Level | Safety Advisory & Voice Prompt |
|---|---|---|---|---|
| **A. PERSON DETECTED** | `SIDEWALK` or peripheral margin | Stationary or walking parallel away from road | `LOW` (Not Detected) | `NO ACTIVE WARNING` ("Driving conditions appear normal.") |
| **B. PEDESTRIAN APPROACHING ROAD** | `ROAD EDGE` or transitioning from sidewalk toward road | Lateral velocity toward road corridor | `CAUTION` / `MODERATE` | `MONITOR AND REDUCE SPEED IF NECESSARY` ("Pedestrian approaching the roadway.") |
| **C. PEDESTRIAN CROSSING** | `CROSSWALK` or `ROADWAY` | Transverse motion across road corridor | `UNDER EVALUATION` (Moderate/High depending on TTC) | `CAUTION` ("Warning. Pedestrian crossing ahead.") |
| **D. POTENTIAL PEDESTRIAN COLLISION RISK** | `ROADWAY` / Vehicle Path Corridor | Intersecting future waypoints with ego vehicle ($TTC \le 3.5\,\text{s}$) | `HIGH` or `CRITICAL` | `REDUCE SPEED AND PREPARE TO STOP WHEN SAFE` ("High pedestrian collision risk. Reduce speed and prepare to stop.") |
| **E. PEDESTRIAN TRACKING LOST** | Previously tracked, now occluded | No observations for $\ge 3$ frames | `UNKNOWN` | `MAINTAIN CAUTION` ("Pedestrian tracking lost. Maintain caution.") |

---

### 3. Implementation Blueprint

- **Detector:** YOLOv11 (`person` class, ID 0) with confidence threshold $\ge 0.25$.
- **Tracker:** ByteTrack preserving integer `track_id` with Kalman state continuity and maximum age of 30 frames.
- **Road Analyzer:** `RoadContextAnalyzer` segmenting frame width into left sidewalk $[0, x_{\text{left}}]$, roadway corridor $[x_{\text{left}}, x_{\text{right}}]$, and right sidewalk $[x_{\text{right}}, W]$.
- **Intent Engine:** `PedestrianAwarenessEngine` tracking directional vectors, velocity, distance to curb, and conflict prediction.
- **Visuals:** Custom styled bounding boxes in `VideoCanvas.tsx` with color-coded states and position tags (`PERSON #01 — SIDEWALK`).
