# Day 14 — Real-World Validation & Engineering Stabilization Report
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. Executive Summary

Day 14 accomplished a comprehensive, evidence-based real-world validation and engineering stabilization of PerceptGuard AI. The complete 12-stage computer vision and failure anticipation pipeline was tested across six realistic driving scenarios derived from authentic **BDD100K driving video** on host CPU hardware (**Intel Core i3 11th Gen, 8 GB RAM, Windows 11 AMD64**).

| Audit Attribute | Verification Record |
| :--- | :--- |
| **Audit Date** | 2026-11-20 |
| **Milestone** | Day 14 — Real-World Testing & Reliability Verification |
| **Overall Verdict** | **`PASS WITH LIMITATIONS`** |
| **Calibration Status Gate** | **`PROVISIONAL CALIBRATION`** |
| **Active Test Pass Rate** | **1,614 / 1,614 Passed (100.0%)** |
| **Tested Input Videos** | Authentic BDD100K daytime, low-light, motion-heavy, degraded, and edge-case clips |
| **Peak Host Process RAM** | $527.9\text{ MB}$ RSS ($< 1.5\text{ GB}$ ceiling) |
| **Pipeline Latency** | $125.3\text{ to } 488.0\text{ ms/frame}$ on CPU ($2.05\text{ to } 7.98\text{ fps}$) |

---

## 2. Day 13 Readiness & Evidence Gate

All Day 13 integrations were independently re-tested:
- **FastAPI Backend:** Verified clean startup ($1.37\text{s}$) with health and telemetry routes active.
- **Streamlit Operations Console:** Verified dual-topology operation with REST API connectivity and automatic fallback.
- **Model Checkpoints:** Verified presence and execution of `models/yolo/yolo11n.pt` ($5.6\text{ MB}$) and `models/forecasting/lstm/model.pt` ($103.5\text{ KB}$).
- **Video Processing Pipeline:** Successfully ran 12 stages without mock data or fabricated outputs.

---

## 3. Real-World Video Scenario Validation

| Scenario | Input Identity | Frames | Detections | Tracks | Risk Summary | Alerts | Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Scenario A: Normal Daytime** | `scenario_a_normal_daytime.mp4` | 15 | 3 | 1 | $H1: 0.598\text{ (MED)}, H3: 0.590\text{ (MED)}, H5: 0.446\text{ (MED)}$ | 14 | **PASS** |
| **Scenario B: Low Lighting** | `scenario_b_low_lighting.mp4` | 15 | 0 | 0 | $H1: 0.746\text{ (HIGH)}, H3: 0.502\text{ (MED)}, H5: 0.687\text{ (HIGH)}$ | 6 | **PASS** |
| **Scenario C: Motion-Heavy** | `scenario_c_motion_heavy.mp4` | 15 | 1 | 0 | $H1: 0.510\text{ (MED)}, H3: 0.664\text{ (HIGH)}, H5: 0.390\text{ (MED)}$ | 16 | **PASS** |
| **Scenario D: Visual Degradation** | `scenario_d_blur_and_contrast.mp4` | 15 | 5 | 1 | $H1: 0.706\text{ (HIGH)}, H3: 0.469\text{ (MED)}, H5: 0.602\text{ (HIGH)}$ | 11 | **PASS** |
| **Scenario E: 1-Frame Edge Case**| `scenario_e_single_frame.mp4` | 1 | 0 | 0 | $H1: 0.598\text{ (MED)}, H3: 0.756\text{ (HIGH)}, H5: 0.571\text{ (MED)}$ | 1 | **PASS** |
| **Scenario F: Corrupted Stream** | `scenario_f_corrupted.mp4` | 0 | - | - | Rejection with `ValueError`: stream cannot be decoded | 0 | **PASS** |

---

## 4. Defects Identified & Rectified

1. **Defect 1 — Cross-Job Alert State Leakage:**
   - *Root Cause:* `EarlyWarningEngine` was cached on `VideoPipelineRunner` instance, causing active alerts and cooldown timestamps to persist across distinct video jobs.
   - *Fix:* Scoped `EarlyWarningEngine` locally per `run()` invocation.
2. **Defect 2 — Missing Server Restart Persistence:**
   - *Root Cause:* `JobManager` stored jobs solely in memory. Server reboots caused historical jobs to disappear from `/jobs` endpoints.
   - *Fix:* Implemented `_rehydrate_existing_jobs()` scanning `outputs/jobs/` for `results.json` artifacts on startup.
3. **Defect 3 — Video Frame-Rate Edge Cases:**
   - *Root Cause:* `int(cap.get(cv2.CAP_PROP_FPS))` failed when videos reported $\le 0$, NaN, or fractional FPS $< 1$.
   - *Fix:* Added `frame_rate = max(1, int(round(fps)))` with default fallback to 30.0.
4. **Defect 4 — Arbitrary Feature Indices in Pipeline Runner:**
   - *Root Cause:* Features were partially filled in arbitrary positions and padded with zeros.
   - *Fix:* Aligned feature extraction strictly to the 27-feature `feature_manifest.json` contract (kinematic displacements, speeds, velocities, Laplacian blur, contrast, and rolling regression trend slopes) with `np.nan_to_num` sanitization.
5. **Defect 5 — Uncleaned Upload Staging Files:**
   - *Root Cause:* Staging files in `outputs/jobs/uploads/` remained on disk after processing.
   - *Fix:* Added automatic staging file unlinking in `JobManager._execute_job_worker` finally block.

---

## 5. Automated Regression Test Evidence

- **Regression Test Module:** `tests/early_warning/test_day14_reliability_and_edge_cases.py` (4 tests).
- **Video Pipeline Tests:** `tests/early_warning/test_video_pipeline_integration.py` (3 tests).
- **API Video Endpoints:** `tests/api/test_video_endpoints.py` (6 tests).
- **Full Active Test Suite:** **1,614 tests passed (100.0%)** on CPU.
