# PerceptGuard-AI — Pedestrian-Aware Driving Intelligence Implementation Report

**Document ID:** `PG-RPT-010-PEDESTRIAN-TRACKING`  
**System Version:** 2.0.0 (Vulnerable Road User Spatial Kinematics)  
**Date:** 2026-11-24  
**Classification:** ADAS Perception & VRU Safety Specification  

---

## 1. Executive Summary

Pedestrians are Vulnerable Road Users (VRUs) whose interactions with an ego vehicle require continuous spatial and kinematic awareness. Standard bounding box detectors cannot distinguish between a pedestrian walking harmlessly on a sidewalk and one stepping into the vehicle's path.

This document details the engineering implementation of the **Pedestrian-Aware Driving Intelligence Engine** in PerceptGuard-AI. The engine fuses:
1. Object detection & persistent ByteTrack tracking (`person` class).
2. Hough transform lane perspective boundaries.
3. Multi-horizon trajectory projections.
4. Spatio-temporal corridor intrusion analysis.

---

## 2. Spatial Road Zone & Movement Intent Classification

### Spatial Road Zones
Using perspective lane line equations $x_{\text{left}}(y)$ and $x_{\text{right}}(y)$:

```
           Vanishing Point / Horizon
                      ▲
                     / \
                    /   \
  Left Sidewalk    / Ego \     Right Sidewalk
     [Zone 1]     / Travel\       [Zone 5]
                 / Corridor\
  Left Road Edge/  [Zone 3] \ Right Road Edge
     [Zone 2]  /             \    [Zone 4]
              /               \
```

- **SIDEWALK:** Center point $(c_x, c_y)$ is outside the road edges ($c_x < x_{\text{left}}(c_y) - \delta$ or $c_x > x_{\text{right}}(c_y) + \delta$).
- **ROAD EDGE:** Within distance margin $\delta$ of the lane boundary ($x_{\text{boundary}} \pm \delta$).
- **ROADWAY:** Strictly within the ego travel corridor ($x_{\text{left}} + \delta < c_x < x_{\text{right}} - \delta$).
- **CROSSWALK:** Within marked pedestrian crosswalk patterns across the roadway.

### Movement Intent
Calculated from tracking history $\Delta x, \Delta y$ over time $\Delta t$:
- **STATIONARY:** Speed $< 8\text{px/s}$ ($< 1.5\text{km/h}$).
- **ALONG_SIDEWALK:** Moving parallel to the road corridor ($|\Delta y| \gg |\Delta x|$).
- **APPROACHING_ROAD:** Moving laterally towards the lane center from the sidewalk or curb ($|\Delta x| > 10\text{px/s}$).
- **CROSSING:** Moving laterally across the roadway corridor.
- **MOVING_AWAY:** Moving laterally away from the road corridor.

---

## 3. The 5 Pedestrian Awareness Decision States

| Decision State | Spatial & Kinematic Condition | Risk Level | System Action & Advisory | Visual Overlay Styling | Voice Alert |
|:---|:---|:---:|:---|:---:|:---|
| **A. PERSON DETECTED** | Pedestrian on sidewalk or safe periphery; no trajectory convergence. | `LOW` | Nominal speed maintained; zero collision risk. | Cyan box (`#00f0ff`), label: `PED #ID - SIDEWALK` | **Silent** (No alarm) |
| **B. PEDESTRIAN APPROACHING ROAD** | Pedestrian on road edge moving towards ego lane ($|\Delta x| > 10\text{px/s}$). | `CAUTION` | Monitor pedestrian trajectory; reduce speed if necessary. | Amber box (`#f59e0b`), label: `[APPROACHING] PED #ID (SIDE)` | "Pedestrian approaching the roadway" |
| **C. PEDESTRIAN CROSSING** | Pedestrian occupying roadway or crosswalk zone. | `MODERATE` | Monitor corridor; prepare to yield. | Orange box (`#f97316`), label: `[CROSSING] PED #ID` | "Warning. Pedestrian crossing ahead" |
| **D. POTENTIAL PEDESTRIAN COLLISION RISK** | Pedestrian in ego corridor or projected to enter with $\text{TTC} \le 2.5\text{s}$. | `HIGH` / `CRITICAL` | Reduce speed and prepare to stop when safe. | Pulsing Red (`#ef4444`, 3.5px), label: `[COLLISION RISK] PED #ID TTC:Xs` | "High pedestrian collision risk. Reduce speed and prepare to stop" |
| **E. PEDESTRIAN TRACKING LOST** | Pedestrian previously tracked disappears due to occlusion. | `UNKNOWN` | Pedestrian tracking lost due to occlusion. Maintain caution. | Muted grey border, label: `[TRACK LOST]` | "Pedestrian tracking lost. Maintain caution" |

---

## 4. Voice Alert Deduplication Policy

To eliminate operator alert fatigue:
1. **Silence on Safe Conditions:** When all pedestrians are in State A (`PERSON DETECTED`), the audio engine remains completely silent.
2. **6-Second Cooldown:** Repeated spoken phrases are throttled by a 6,000ms cooldown window per alert category.
3. **High-Priority Interruption:** A critical State D alert immediately cancels any lower-priority utterances currently being synthesized by the browser.
4. **Mute Control:** Operator voice toggle persisted to `localStorage`.

---

## 5. Verification Matrix

The test suite in `tests/integration/test_pedestrian_awareness.py` validates all five states with 100% test passage.
