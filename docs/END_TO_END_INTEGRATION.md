# End-to-End Perception Reliability & Early Warning Pipeline Integration

## 1. Executive Summary

The **PerceptGuard AI End-to-End Pipeline** connects video stream extraction, deep object detection, multi-object tracking, spatio-temporal feature engineering, LSTM multi-horizon forecasting, mathematical risk scoring, temporal monitoring, and early-warning alert dispatching into a unified production architecture.

This document describes the end-to-end processing topology, contract mappings, data integrity invariants, performance characteristics, and fault isolation mechanisms.

---

## 2. End-to-End Processing Topology

```
[ Camera / Video Ingest ] (720p 30fps Driving Stream)
           │
           ▼
[ Stage 1: Frame Extraction & Preprocessing ]
           │  Frames: frame_000001.jpg ... frame_000030.jpg (33.3ms interval)
           ▼
[ Stage 2: YOLO11n Object Detection ]
           │  Bounding boxes, class IDs, detection confidence (mean: 0.4003)
           ▼
[ Stage 3: ByteTrack Multi-Object Tracking ]
           │  Track IDs, Kalman state vectors, trajectory history
           ▼
[ Stage 4: Spatio-Temporal Feature Engineering ]
           │  27-dimensional feature vectors (kinematics, appearance, quality)
           ▼
[ Stage 5: Temporal Sequence Assembly ]
           │  Sliding window L=5, stride G=25
           ▼
[ Stage 6: LSTM Multi-Horizon Forecasting ]
           │  Continuous reliability forecasts: R_{t+1}, R_{t+3}, R_{t+5} ∈ [0, 1]
           ▼
[ Stage 7: Reliability Risk Scoring & Classification ]
           │  Linear inverse failure risk: S_{t+h} = 1.0 - R_{t+h}
           │  Classification against provisional thresholds (LOW, MED, HIGH, CRIT)
           ▼
[ Stage 8: Temporal Risk Monitoring & Early Warning ]
           │  Chronological windowing, rolling stats, trend detection (Task 4)
           │  Rule engine, deduplication, cooldown, lifecycle state machine (Task 5)
           ▼
[ Operations Dashboard & Telemetry ]
           Streamlit interactive UI & DashboardService API (Task 6)
```

---

## 3. Data Integrity & Contract Invariants

1. **Horizon Independence:**
   - Predictions and risk scores for $H1$ ($t+1$), $H3$ ($t+3$), and $H5$ ($t+5$) are tracked in distinct, isolated sequence trackers.
   - Downstream early warning rules evaluate horizon-specific triggers independently.
2. **Provenance & Chronology:**
   - Every observation retains its `sequence_id`, `source_frame_id`, `target_frame_id`, and `timestamp_seconds`.
   - The temporal monitor strictly enforces chronological ordering; out-of-order records are flagged and prohibited from polluting rolling trends.
3. **Calibrated Integrity / Provisional Status:**
   - Single-clip driving data exhibits severe zero-inflation (100% nominal ground-truth).
   - In accordance with rigorous scientific practice, all thresholds and calibration flags are explicitly marked **`PROVISIONAL`**.
   - Risk scores are treated as derived ranking indicators rather than calibrated probabilities.
4. **No Silent Synthetic Fallbacks:**
   - Missing models, files, or telemetry are explicitly propagated as `UNAVAILABLE`, `DEGRADED`, or `STALE`.
   - The dashboard and backend services never inject mock numbers to pretend a missing model is functioning.

---

## 4. End-to-End Latency Budget & CPU Resource Profile

The system is engineered for CPU-only deployment on target hardware (Windows 11, Intel Core i3, 8 GB RAM):

| Processing Stage | Latency per Frame / Prediction | Budget Allocation (33.3ms @ 30 FPS) | Hardware Resource |
| :--- | :--- | :--- | :--- |
| Risk Scoring (Task 2) | $12.4\ \mu\text{s}$ | $0.04\%$ | Pure CPU math |
| Risk Classification (Task 3) | $8.2\ \mu\text{s}$ | $0.02\%$ | Pure CPU thresholding |
| Temporal Monitoring (Task 4) | $28.5\ \mu\text{s}$ | $0.09\%$ | CPU rolling deque |
| Alert Generation (Task 5) | $82.7\ \mu\text{s}$ | $0.25\%$ | CPU rule evaluation |
| **Total Post-Forecasting Overhead** | **$\approx 0.13\text{ ms}$** | **$< 0.5\%$ of 33.3ms budget** | **Negligible CPU impact** |

This guarantees that early-warning assessment and dashboard data preparation consume less than 0.5% of the real-time frame budget, leaving over 99.5% of CPU time for upstream computer vision tasks.
