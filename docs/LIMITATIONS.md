# PerceptGuard AI — Known Limitations & Operational Boundaries

## 1. Executive Summary

This document transparently defines the known technical, machine learning, and hardware boundaries of **PerceptGuard AI**. In safety-critical applications such as autonomous driving and intelligent transportation systems, clear recognition of operational constraints prevents improper reliance on predictive systems.

---

## 2. Machine Learning & Calibration Constraints

### 2.1 Provisional Calibration Status
- **Current Designation:** `PROVISIONAL CALIBRATION`
- **Root Cause:** The risk formulation $S_{t+h} = 1.0 - R_{t+h}$ and operational classification thresholds ($[0.35, 0.65, 0.85]$) were calibrated using verified BDD100K driving datasets under daylight and typical weather conditions.
- **Operational Constraint:** Until extensive road testing across severe adverse weather conditions (heavy blizzard, zero-illumination rural roads, torrential rain, optical flare) is completed, risk thresholds must be treated as **early advisory warnings**, not hard safety interlocks.

### 2.2 Difference Between Detection Confidence and Future Reliability
- **Detection Confidence ($P(\text{class}|\text{box})$):** Emitted instantaneously by YOLO11n. Represents the detector's classification certainty on current frame pixels.
- **Predicted Reliability ($R_{t+h}$):** Forecasted by the LSTM model. Represents anticipated bounding box IoU quality against future ground truth annotations.
- **Empirical Boundary:** A detector can output $0.95$ confidence on a severely blurred or vibrating target that drops out in the very next frame. Operators must monitor `⚠️ Risk Assessment` rather than raw YOLO confidence alone.

### 2.3 Domain & Perspective Specificity
- Models and feature extractors are optimized for forward-facing automotive camera perspectives on passenger vehicles.
- Alternative robotics domains (e.g. aerial drone surveillance, warehouse AMR ground-level cameras, maritime vessels) require feature scaler recalibration and domain transfer fine-tuning.

---

## 3. Hardware & Compute Constraints

### 3.1 Edge CPU Execution Budget
- **Target Hardware:** Intel Core i3 11th Gen, 8 GB RAM, Windows 11 AMD64.
- **Processing Throughput:**
  - Real-world processing throughput is benchmarked at $\approx 18.5\text{ FPS}$ for 720p 30fps video under pure CPU execution.
  - Full-resolution 4K or 60+ FPS zero-latency edge streaming requires dedicated hardware acceleration (NVIDIA TensorRT or OpenVINO NPU).
- **RAM Constraints:**
  - Baseline process memory: $\sim 362\text{ MB}$.
  - Peak inference memory: $\sim 528\text{ MB}$ RSS.
  - Processing long video sequences ($> 500$ frames) on an 8 GB machine requires frame-rate downsampling or batch chunking to avoid host memory pressure.

### 3.2 Single Concurrent Pipeline Worker
- The background `JobManager` enforces a single concurrent worker thread (`ThreadPoolExecutor(max_workers=1)`) to avoid CPU core thrashing on dual-core / quad-thread hardware.
- Multiple submitted jobs are queued and processed sequentially.

---

## 4. UI & Session State Boundaries

### 4.1 Browser Session State Scoping
- In the Streamlit dashboard, the active job selection (`st.session_state["active_job_id"]`) is maintained in per-browser tab memory.
- Opening the dashboard in a new tab or browser window defaults to the most recently completed job on disk (`outputs/jobs/`).
- Historical results and bounding boxes are permanently persisted in `outputs/jobs/{job_id}/results.json` and are never lost on browser refresh.

### 4.2 Annotated Image Storage
- Frame-by-frame visual bounding box inspection relies on locally saved JPEG images in `outputs/jobs/{job_id}/frames/annotated/`.
- In containerized deployments without persistent volume mounts for `outputs/`, frame images will not persist across container recreations (though tabular telemetry in `results.json` persists if mounted).

---

## 5. Summary Verdict Matrix

| Dimension | Specification | Operational Status | Production Verdict |
| :--- | :--- | :--- | :--- |
| **Dashboard Navigation** | 9 Views, 15 Reusable Components | 100% Operational | `PASS` |
| **API Transport & Fallback**| Dual-Mode REST + In-Process Fallback | 100% Operational | `PASS` |
| **Data Contracts** | $S = 1.0 - R$, Null-Safe, Zero Coercion | 100% Operational | `PASS` |
| **Hardware Footprint** | $\approx 18.5\text{ FPS}$, $< 530\text{ MB}$ RSS on Intel i3 | 100% Operational | `PASS` |
| **Calibration Status** | Provisional Thresholds Pending Weather Gate | Provisional | `PASS WITH LIMITATIONS` |
| **OVERALL VERDICT** | **Complete Release Readiness Milestone** | **Validated** | **`PASS WITH LIMITATIONS`** |
