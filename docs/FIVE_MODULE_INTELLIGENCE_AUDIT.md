# PERCEPTGUARD-AI — EVIDENCE-DRIVEN FIVE-MODULE INTELLIGENCE & PEDESTRIAN AWARENESS AUDIT
## COMPREHENSIVE ARCHITECTURAL AUDIT REPORT

**Date:** November 24, 2026  
**Author:** PerceptGuard-AI Senior Autonomous Driving Research & Vision Architecture Team  
**Scope:** Strict preservation of Exactly 5 Primary ADAS Modules with Evidence-Driven Context Awareness  

---

### Executive Summary

PerceptGuard-AI was subjected to a thorough architectural audit across all five primary ADAS modules:
1. Camera & Weather Intelligence
2. Object Detection & Tracking
3. Object & Trajectory Prediction
4. Collision & Road Risk Assessment
5. Early Warning & Vehicle Stop Recommendation

The objective of this audit is to identify why the system previously tended to over-escalate normal driving scenes into caution/risk, lacked pedestrian-specific road position context (sidewalk vs road edge vs crossing), and how to evolve the platform into an **evidence-driven decision engine** that accurately distinguishes between:
- **Normal Driving:** Clean roads, vehicles following lanes, parked vehicles on roadsides, pedestrians on sidewalks -> `NORMAL` / `NO COLLISION RISK DETECTED` / `LOW` risk / `NO ACTIVE ALERTS`.
- **Actual Collision Risks:** Pedestrians crossing into travel corridors, motorcycles cutting across future vehicle paths, sudden decelerating obstacles -> `COLLISION RISK DETECTED` / `HIGH` or `CRITICAL` risk with validated TTC and time-to-conflict.
- **Uncertainty / Insufficient Evidence:** Degraded perception, missing tracking history, or uncalibrated distance -> `UNKNOWN` (never fabricated).

---

### 1. Root-Cause Analysis of Previous Limitations

#### 1.1 The "False Caution on Normal Road" Logic Defect
In `src/safety_recommendation/recommendation_rules.py` (lines 321–337):
```python
if (
    vis_score < 75.0
    or vis_cond != "CLEAR"
    or uncertainty_score >= 0.25
    or detection_reliability < 0.75
    or len(tracks_list) > 0  # <--- CRITICAL ROOT CAUSE
):
    reason = "Peripheral road users detected outside critical path; safe separation maintained."
    action = "Pay increased attention and maintain active scan of adjacent corridors."
    return SafetyRecommendation(..., recommendation_state=RecommendationState.CAUTION, ...)
```
**Impact:** Because almost every real-world road video has at least one tracked object (`len(tracks_list) > 0`), the rules engine **never returned `RecommendationState.NORMAL`**! A car driving smoothly in its own lane 50 meters away or parked cars on the roadside immediately forced the entire system into `CAUTION` instead of `NORMAL`.

#### 1.2 Proximity Over-Sensitivity in Collision Risk Engine
In `src/collision_risk/risk_classifier.py` (lines 121–124):
```python
elif (min_ttc is not None and min_ttc <= self.moderate_ttc) or len(conflicts) > 0 or (
    closest_dist_m is not None and closest_dist_m < 8.0
) or (closest_dist_px is not None and closest_dist_px < 90.0):
    state = CollisionRiskState.MODERATE
```
**Impact:** Any object whose bounding box centroid came within 90 pixels of the virtual ego origin (bottom center) immediately escalated risk to `MODERATE`, regardless of whether the object was moving away, stationary, or laterally separated.

#### 1.3 Absence of Dedicated Road & Lane Context
Previously, the system lacked a road scene contextualizer. As a result:
- There was no distinction between a vehicle in its own lane vs encroaching on the ego lane.
- Pedestrians could not be classified as being on the `SIDEWALK`, `ROAD EDGE`, `CROSSWALK`, or `ROADWAY`.
- Sidewalk pedestrians walking safely parallel to the vehicle were evaluated with the same collision geometry as objects in the travel corridor.

#### 1.4 Absence of Pedestrian-Aware Decision States
The prompt specifically mandates five distinct pedestrian awareness states:
1. `PERSON DETECTED`: Person visible on sidewalk with no travel path conflict -> `COLLISION RISK: NOT DETECTED`, `NO ACTIVE WARNING`.
2. `PEDESTRIAN APPROACHING ROAD`: Person moving towards roadway edge -> `RISK: CAUTION`, `MONITOR AND REDUCE SPEED IF NECESSARY`.
3. `PEDESTRIAN CROSSING`: Person actively in crossing maneuver -> `COLLISION RISK: UNDER EVALUATION`.
4. `POTENTIAL PEDESTRIAN COLLISION RISK`: Credible spatio-temporal intersection with ego travel path -> `RISK LEVEL: HIGH/CRITICAL`, `REDUCE SPEED AND PREPARE TO STOP WHEN SAFE`.
5. `PEDESTRIAN TRACKING LOST`: Previously tracked pedestrian occluded -> `UNKNOWN`, `MAINTAIN CAUTION`.

Previously, all objects (cars, trucks, persons) were evaluated through a generic vehicle bounding-box distance check without pedestrian-specific spatial intent.

---

### 2. Required Architectural Upgrades

```
[Uploaded Dashcam Video]
         │
         ▼
[VideoPipelineRunner (13-Stage Pipeline)]
  │
  ├── 1. Optical Visibility & Weather (Clear / Rain / Fog / Blur / Glare)
  ├── 2. YOLOv11n Detection (Car, Truck, Bus, Motorcycle, Bicycle, Person)
  ├── 3. ByteTrack Multi-Object Tracking (Persistent Track IDs)
  │
  ├── 4. [NEW] Road & Lane Context Analyzer
  │     ├── Lane boundary & marking detection (LANE DETECTED / NOT VISIBLE / UNKNOWN)
  │     ├── Ego travel corridor geometry (lateral boundary bounds)
  │     └── Roadway spatial zones: SIDEWALK, ROAD EDGE, CROSSWALK, ROADWAY
  │
  ├── 5. [NEW] Pedestrian Awareness & Intent Engine
  │     ├── Tracked person spatial classification (Sidewalk vs Roadway)
  │     ├── Motion intent (Stationary, Walking Along, Approaching Road, Crossing)
  │     └── 5 Pedestrian Decision States (Detected, Approaching, Crossing, Conflict, Lost)
  │
  ├── 6. Multi-Horizon Kinematic Trajectory Prediction (H1: 1s, H3: 3s, H5: 5s)
  │
  ├── 7. Evidence-Driven Collision Risk Assessment
  │     ├── Mathematical TTC check (valid only when closing speed > 0)
  │     ├── Spatio-temporal trajectory conflict checking (synchronized in time)
  │     └── Normal Driving gate: NO COLLISION RISK DETECTED if paths don't conflict
  │
  └── 8. Early Warning & Driver Advisory Recommendations
        ├── Normal Driving decision: NORMAL / NO ACTIVE ALERTS
        ├── Pedestrian advisory messages
        └── Actionable stop recommendations (advisory only, non-actuating)
         │
         ▼
[Frontend Unified Telemetry & Video Canvas]
  ├── Standardized 5-Module States (CLEAR / DETECTED / NORMAL PATH / NO RISK / NO ALERTS)
  ├── Distinct Pedestrian Visual Styles (Cyan=Sidewalk, Amber=Approaching, Orange=Crossing, Red=Conflict)
  └── Web Speech API Voice Feedback Engine with 6s Cooldown & Normal Driving Silence
```

---

### 3. Verification Criteria & Test Plan

1. **Normal Driving Cases:**
   - Empty road -> `NORMAL`, `NO COLLISION RISK DETECTED`, `NO ACTIVE ALERTS`.
   - Cars moving in lane -> `NORMAL`, `NO COLLISION RISK DETECTED`.
   - Parked cars -> stationary, `NO COLLISION RISK DETECTED`.
   - Pedestrians on sidewalk -> `PERSON DETECTED`, `LOW` risk, `NO ACTIVE WARNING`.
2. **Pedestrian Risk Cases:**
   - Pedestrian approaching road -> `PEDESTRIAN APPROACHING ROAD`, `CAUTION`.
   - Pedestrian crossing in front of vehicle -> `COLLISION RISK DETECTED`, `HIGH`/`CRITICAL`.
3. **Uncertainty Cases:**
   - Missing tracking / severe glare -> `UNKNOWN`.
4. **Scope Preservation:**
   - Exactly the 5 primary modules. No new sidebar entries.
