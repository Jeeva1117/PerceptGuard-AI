# PerceptGuard-AI — Pedestrian Risk & Spatial Kinematics Validation Report

**Document ID:** `PG-RPT-011-PEDESTRIAN-VALIDATION`  
**System Version:** 2.0.0 (Vulnerable Road User Safety Assurance)  
**Date:** 2026-11-24  
**Classification:** Empirical Verification & Safety Assurance Report  

---

## 1. Overview & Evaluation Goals

The purpose of this validation report is to empirically verify that the **Pedestrian-Aware Driving Intelligence Engine** correctly identifies, segments, tracks, and evaluates vulnerable road users (pedestrians/persons) across diverse traffic scenarios without false alarms or delayed hazard reactions.

---

## 2. Decision State Test Matrix

| State Identifier | State Name | Test Objective | Test Condition | Asserted Output | Pass/Fail |
|:---|:---|:---|:---|:---|:---:|
| **State A** | `PERSON DETECTED` | Verify safe sidewalk presence does not trigger alarm | Pedestrian at $x=150\text{px}$ (sidewalk, lane $350-930\text{px}$), $v = 4.5\text{km/h}$ parallel | Risk: `LOW`, Action: Nominal speed, Voice: `None` | **PASS** |
| **State B** | `PEDESTRIAN APPROACHING ROAD` | Verify early warning when pedestrian moves toward road edge | Pedestrian at $x=280\text{px}$ with lateral trajectory toward lane boundary | Risk: `CAUTION`, Action: Monitor trajectory, Voice: Active | **PASS** |
| **State C** | `PEDESTRIAN CROSSING` | Verify recognition of road occupancy or crosswalk use | Pedestrian at $x=450\text{px}$ traversing roadway | Risk: `MODERATE`, Action: Monitor corridor, Voice: Active | **PASS** |
| **State D** | `POTENTIAL PEDESTRIAN COLLISION RISK` | Verify imminent conflict triggers immediate vehicle stop recommendation | Pedestrian in travel path ($x=640\text{px}$) with closing $\text{TTC} \le 1.8\text{s}$ | Risk: `CRITICAL`, Recommendation: `STOP RECOMMENDED`, Priority: `CRITICAL` | **PASS** |
| **State E** | `PEDESTRIAN TRACKING LOST` | Verify occlusion triggers graceful transition to caution | Pedestrian disappears behind obstacle | State: `PEDESTRIAN TRACKING LOST`, Risk: `UNKNOWN`, Action: Maintain caution | **PASS** |

---

## 3. Spatial Zone & Intent Geometry

### Roadway Corridor Bounds:
- `left_boundary_x`: $350.0\text{px}$
- `right_boundary_x`: $930.0\text{px}$
- `lane_center_px`: $640.0\text{px}$
- `edge_margin`: $\pm 69.6\text{px}$ ($12\%$ of lane width $580\text{px}$)

### Zone Mapping Verification:
1. Coordinates $x < 280.4\text{px}$ or $x > 999.6\text{px}$ classify strictly as `SIDEWALK`.
2. Coordinates $280.4\text{px} \le x \le 419.6\text{px}$ or $860.4\text{px} \le x \le 999.6\text{px}$ classify strictly as `ROAD EDGE`.
3. Coordinates $419.6\text{px} < x < 860.4\text{px}$ classify strictly as `ROADWAY`.

---

## 4. Audio Alert Behavior

The audio policy was validated against three essential requirements:
1. **Zero Spoken Alerts for Sidewalk Pedestrians:** In State A, `voice_prompt` evaluates to `None`. No speech utterance is queued or synthesized.
2. **Immediate Escalation for Crossing or Collision:** In State B, C, and D, targeted warnings ("Pedestrian approaching roadway", "Warning. Pedestrian crossing ahead", "High pedestrian collision risk. Reduce speed and prepare to stop") are triggered.
3. **Throttling & Preemption:** A 6-second cooldown prevents repeated notifications for the same track, while high-priority State D pre-empts and cancels lower-priority speech immediately.

---

## 5. Summary Generation

`pedestrian_engine.generate_summary()` produces structured stream telemetry:
- `total_unique_pedestrians`: count of unique ByteTrack IDs.
- `frames_with_pedestrians`: number of frames containing active pedestrians.
- `primary_state_distribution`: dictionary of state occurrences across video run.
- `potential_collision_events`: total count of critical frames.
- `crossings_observed`: total count of road-crossing frames.
- `approaches_observed`: total count of curb-approach frames.
- `lost_tracking_events`: count of occlusion-loss incidents.

---

## 6. Conclusion

The Pedestrian Awareness subsystem meets all ADAS safety criteria. It provides dependable detection, spatial classification, and predictive intent forecasting to safeguard vulnerable road users while maintaining high operator confidence during normal safe driving.
