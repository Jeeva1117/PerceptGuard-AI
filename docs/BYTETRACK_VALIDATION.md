# PerceptGuard-AI: ByteTrack Multi-Object Tracking Validation Report

**Document Version:** 2.0 (Release Readiness)  
**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  

---

## 1. Executive Summary

This report documents the validation of Feature 2's tracking component: the production ByteTrack tracking engine in PerceptGuard-AI. The engine associates detections across frames, maintains consistent track identities, recovers occluded road users, and ensures semantic class stability. It operates entirely on CPU using 2D discrete Kalman state estimation and linear sum assignment with zero external server dependencies.

---

## 2. ByteTrack Two-Stage Association Architecture

ByteTrack differs fundamentally from classical trackers by associating both high-confidence and low-confidence detections in a structured two-stage pipeline:

```
Frame t Detections (conf >= 0.10)
               │
               ├── Stage 1: Detections (conf >= 0.25)
               │     │
               │     ▼
               │   Match against Confirmed / Active Tracks (IoU distance)
               │     ├── Matched Tracks ──► Update Kalman State (CONFIRMED)
               │     └── Unmatched Tracks ─┐
               │                           │
               └── Stage 2: Detections (0.10 <= conf < 0.25)
                     │                     │
                     ▼                     ▼
                   Match low-score detections against leftover unassigned tracks
                     ├── Matched ──► Recovered track! (Prevents track fragmentation)
                     └── Unmatched ─► Marked LOST (buffered for 30 frames)
```

---

## 3. Five-State Lifecycle State Machine

Each tracked object transitions through a formal finite state machine:

```
                 [First Observation]
                          │
                          ▼
                    ┌───────────┐
                    │ TENTATIVE │ (Observations < 3)
                    └─────┬─────┘
                          │ 3 consecutive matches
                          ▼
                    ┌───────────┐
      ┌────────────►│ CONFIRMED │
      │             └─────┬─────┘
      │                   │ Missed in frame
      │                   ▼
      │             ┌───────────┐
Re-identified       │   LOST    │ (Buffered up to track_buffer=30 frames)
      │             └─────┬─────┘
      │                   │ Exceeds buffer threshold
      └───────────────────┼────────────────┐
                          ▼                ▼
                  ┌──────────────┐   ┌─────────┐
                  │ REIDENTIFIED │   │ REMOVED │
                  └──────────────┘   └─────────┘
```

1. **`TENTATIVE`**: A brand new track candidate. Suppresses false-positive flickers.
2. **`CONFIRMED`**: A robust, persistent object. Active in kinematic and collision calculations.
3. **`LOST`**: An object temporarily occluded or moving outside the camera's high-visibility zone.
4. **`REIDENTIFIED`**: A previously lost object that reappears and matches its predicted Kalman position.
5. **`REMOVED`**: Purged from memory to bound CPU resource usage.

---

## 4. Motorcycle Tracking & Class Persistence Fix

### The Problem
In earlier iterations:
- Detections were passed to ByteTrack only if `conf >= 0.25`. As a result, when a motorcycle turned, leaned, or passed behind a vehicle, its score dropped to ~0.18 and was discarded. ByteTrack lost the track immediately.
- Furthermore, when the motorcycle reappeared, its label sometimes defaulted to `"3"` (numeric COCO ID) or was misclassified as `"car"`.

### The Repair
1. **Pass All Candidates**: The pipeline passes all detections with `conf >= 0.10` to `tracker.update()`, enabling ByteTrack's Stage 2 association to recover the motorcycle.
2. **Persistent Track Class Memory (`_track_classes`)**:
   `ByteTrackEngine` maintains a dictionary mapping each assigned `track_id` to its established `(class_id, class_name)`. Once identified as a motorcycle:
   - Subsequent minor classification ambiguities do not overwrite the motorcycle label.
   - The label remains `"motorcycle"` across the entire video duration, occlusions, and re-identifications.

---

## 5. Linear Assignment Solver Environment Shim

Ultralytics `BYTETracker` imports the C-extension `lap` by default. In offline or standard Windows environments where compilation tools are absent, `src/tracking/bytetrack_engine.py` provides a transparent shim:
- Automatically binds `scipy.optimize.linear_sum_assignment`.
- Exposes the exact `lapjv` signature.
- Guarantees 100% operational portability across all CPU execution environments without installation failures.

---

## 6. Verification Test Matrix

| Test Suite | Case Name | Description | Status |
| :--- | :--- | :--- | :--- |
| `test_motorcycle_and_road_users.py` | `test_01_default_road_classes_mapping` | Verifies COCO road class ID definitions | **PASS** |
| `test_motorcycle_and_road_users.py` | `test_02_motorcycle_detection_ingestion` | Ingestion and initial lifecycle state check | **PASS** |
| `test_motorcycle_and_road_users.py` | `test_03_motorcycle_track_continuity_and_confidence_dip` | Verifies Stage 2 recovery when conf dips to 0.18 | **PASS** |
| `test_motorcycle_and_road_users.py` | `test_04_motorcycle_class_stability` | Verifies no class flipping to generic/car | **PASS** |
| `test_motorcycle_and_road_users.py` | `test_05_multi_road_user_concurrent_tracking` | Simultaneous tracking of motorcycle, car, truck, ped | **PASS** |
| `test_motorcycle_and_road_users.py` | `test_06_motorcycle_reidentification_after_occlusion` | Re-identifies motorcycle with `REIDENTIFIED` state | **PASS** |
| `test_motorcycle_and_road_users.py` | `test_07_engine_reset_session_isolation` | Verifies clean reset between video sessions | **PASS** |
| `test_bytetrack_engine.py` | 18 automated unit tests | Comprehensive tracker lifecycle and corner cases | **PASS** |
| `test_regression_scenarios.py` | 12 automated scenarios | Integration across realistic driving sequences | **PASS** |
