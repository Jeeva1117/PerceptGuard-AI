# PerceptGuard-AI — Five-Module Evidence-Driven Intelligence Implementation Report

**Document ID:** `PG-RPT-008-INTELLIGENCE-V2`  
**System Version:** 2.0.0 (Evidence-Driven Automotive Advisory ADAS)  
**Date:** 2026-11-24  
**Classification:** Engineering Specification & Technical Report  

---

## 1. Executive Summary

This report documents the architectural upgrade of **PerceptGuard-AI** from an unconstrained frame-level heuristic system into an **evidence-driven, context-aware road perception and predictive safety platform**.

Previously, the system suffered from systemic "False Caution" and false alarm escalation because:
1. Object detection counts alone (`len(tracks) > 0`) triggered `CAUTION` recommendations even when vehicles maintained safe lane-following headway.
2. Collision proximity thresholds (`distance_px < 90.0` or `distance_m < 8.0`) fired without requiring verified closing kinematics (positive closing velocity or trajectory intersection).
3. Pedestrians on sidewalks or roadside curbs were treated as immediate road hazards without spatial context classification.

The updated engine enforces **strict evidence gates**:
- **Normal Driving:** Vehicles moving along lanes with safe headway or roadside parked vehicles output `NORMAL` driving state, `NO COLLISION RISK DETECTED`, `RISK LEVEL: LOW`, and `NO ACTIVE ALERTS`.
- **Verified Collision Risks:** Require either confirmed trajectory intersection within the prediction horizon ($H1/H3/H5$) or mathematically defensible Time-To-Collision ($\text{TTC} \le 3.5\text{s}$) with converging lateral or longitudinal motion.
- **Degraded Perception:** Optical blackouts ($< 20\%$) or excessive uncertainty ($> 70\%$) output `UNKNOWN` with actionable guidance ("Maintain manual vigilance").

---

## 2. Five Primary Module Architecture

The system retains exactly the five primary automotive ADAS perception modules:

```
[ Input Stream (Video / Dashcam) ]
               │
               ▼
┌───────────────────────────────────────────────┐
│ Module 1: Camera & Weather Intelligence       │
│ - Optical Visibility Index (0 - 100%)         │
│ - Weather: CLEAR / MODERATE / LOW / UNKNOWN   │
│ - Atmospheric Degradation & Quality Metrics   │
└──────────────────────┬────────────────────────┘
                       ▼
┌───────────────────────────────────────────────┐
│ Module 2: Object Detection & Tracking         │
│ - YOLO11n + ByteTrack Multi-Object Tracking   │
│ - Persistent IDs & Motion Continuity          │
│ - Status: DETECTED / NOT DETECTED / UNKNOWN   │
└──────────────────────┬────────────────────────┘
                       ▼
┌───────────────────────────────────────────────┐
│ Module 3: Object & Trajectory Prediction      │
│ - Multi-Horizon Kinematic Projection (H1/3/5) │
│ - Road Context & Lane Perspective Modeling    │
│ - Status: NORMAL PATH / CONFLICT / UNKNOWN    │
└──────────────────────┬────────────────────────┘
                       ▼
┌───────────────────────────────────────────────┐
│ Module 4: Collision & Road Risk Assessment    │
│ - Spatio-Temporal Trajectory Convergence      │
│ - Time-To-Collision (TTC) & Dynamic Headway   │
│ - Status: NO COLLISION RISK / DETECTED / UNK  │
└──────────────────────┬────────────────────────┘
                       ▼
┌───────────────────────────────────────────────┐
│ Module 5: Early Warning & Safety Decision     │
│ - Context-Aware Pedestrian Intent Engine      │
│ - Explainable Advisory Stop Guidance          │
│ - Status: NO ACTIVE ALERTS / ACTIVE / UNKNOWN │
└───────────────────────────────────────────────┘
```

---

## 3. Engineering Implementation Details

### Module 1: Camera & Weather Intelligence
- **Implementation:** `src/camera_weather/visibility_quality_engine.py` & `schemas.py`
- **Output standardization:**
  - `CLEAR`: Visibility $\ge 75\%$, nominal ambient lighting and contrast.
  - `MODERATE`: Visibility $50\% - 74\%$, mild fog/rain attenuation or blur.
  - `LOW`: Visibility $< 50\%$, dense fog, severe blur, or blinding glare.
  - `UNKNOWN`: Ingestion corrupted or camera completely blacked out.

### Module 2: Object Detection & Tracking
- **Implementation:** `src/detection/` & `src/tracking/bytetrack_engine.py`
- **Standardized states:**
  - `DETECTED`: Active tracked road agents (cars, trucks, buses, motorcycles, bicycles, persons).
  - `NOT DETECTED`: Zero road agents visible in forward field of view.
  - `UNKNOWN`: Degraded perception preventing reliable inference.

### Module 3: Object & Trajectory Prediction
- **Implementation:** `src/trajectory_prediction/` & `src/collision_risk/road_context_analyzer.py`
- **Road & Lane Context Analyzer:**
  - Hough transform lane detection in lower road ROI.
  - Perspective vanishing point interpolation mapping left/right boundary coordinates across all scanlines.
  - Classifies road zones into: `Left Sidewalk`, `Left Road Edge`, `Ego Travel Corridor`, `Right Road Edge`, `Right Sidewalk`.
- **Standardized states:**
  - `NORMAL PATH`: Forward agents following parallel or diverging trajectories.
  - `POTENTIAL CONFLICT`: Projected convergence or lateral path cut-in.
  - `INSUFFICIENT HISTORY`: Track initialized with $< 2$ frames of motion telemetry.

### Module 4: Collision & Road Risk Assessment
- **Implementation:** `src/collision_risk/risk_classifier.py` & `collision_risk_engine.py`
- **Evidence Gating:**
  - Proximity alone is forbidden from escalating risk without closing velocity.
  - A lead vehicle maintaining headway yields `LOW` risk.
  - Parked cars on roadside with zero relative velocity yield `LOW` risk.
  - Cross-cutting trajectory with $\text{TTC} \le 1.8\text{s}$ triggers `CRITICAL`.
- **Standardized states:**
  - `NO COLLISION RISK DETECTED`: All agents maintain clearance.
  - `COLLISION RISK DETECTED`: Validated closing trajectory conflict.
  - `UNKNOWN`: Excessive uncertainty ($U \ge 0.70$) or blackout.

### Module 5: Early Warning & Safety Recommendations
- **Implementation:** `src/safety_recommendation/pedestrian_awareness_engine.py`, `recommendation_rules.py`, `safety_engine.py`
- **Decision Hierarchy:**
  - `STOP RECOMMENDED`: Critical imminent collision hazard ($\text{TTC} \le 1.8\text{s}$ or pedestrian in path).
  - `PREPARE TO STOP`: Developing conflict ($\text{TTC} \le 3.5\text{s}$).
  - `REDUCE SPEED`: Pedestrian crossing roadway or moderate corridor conflict.
  - `CAUTION`: Pedestrian approaching road edge or optical degradation.
  - `NORMAL`: Nominal road condition, safe following distance, zero path conflicts.
  - `UNKNOWN`: Insufficient sensor evidence.
- **Audio Feedback Policy:**
  - Completely silent during `NORMAL` driving.
  - Web Speech API triggered with 6-second cooldown on hazards only.
  - Mute toggle persisted to `localStorage`.

---

## 4. Verification & Validation Summary

| Test Suite | Tests Executed | Passed | Status |
|:---|:---:|:---:|:---:|
| `test_pedestrian_awareness.py` | 6 | 6 | **PASS (100%)** |
| `test_normal_vs_collision_validation.py` | 4 | 4 | **PASS (100%)** |
| `test_five_features_end_to_end.py` | 1 | 1 | **PASS (100%)** |
| Frontend TypeScript & Vite Build | 2,550 modules | 2,550 | **PASS (100%)** |

---

## 5. Deployment Guidelines & Prototype Notice

PerceptGuard-AI is an experimental autonomous-driving-inspired decision-support prototype. It operates strictly in advisory capacity (e.g. recommending "Stop when safe to do so" and "Reduce speed") and does NOT issue CAN-bus actuation commands to steering or braking systems.
