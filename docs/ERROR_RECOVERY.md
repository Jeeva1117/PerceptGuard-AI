# Error Recovery, Fault Tolerance & Resilience Guide
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. Resilience Philosophy & Safeguards

PerceptGuard AI implements defensive programming across all application boundaries to protect edge compute hardware, prevent data corruption, and ensure clear operational feedback:

1. **Crash Isolation:** Unhandled exceptions in the video processing worker thread transition the job record to `FAILED` with explicit error causes, preventing server crashes.
2. **Restart Persistence:** `JobManager` scans `outputs/jobs/` upon startup and rehydrates historical completed jobs from atomic `results.json` files.
3. **Finite Mathematical Defense:** All multi-modal feature inputs are sanitized via `np.nan_to_num` to block NaN/Inf values from reaching PyTorch tensors.
4. **Isolated Alert Lifecycles:** Each video job run instantiates an isolated `EarlyWarningEngine`, preventing cooldown suppression or state leakage across independent video clips.

---

## 2. Failure Modes & Automatic Recovery Matrix

| Failure Mode | Trigger Scenario | System Behavior & Mitigation | Recovery Action |
| :--- | :--- | :--- | :--- |
| **Corrupted Video Stream** | Broken file header, non-video bytes | OpenCV returns `isOpened() == False` or 0 frames | Raises descriptive `ValueError`; job marked `FAILED`; HTTP 400 error. |
| **Missing Model Checkpoint** | Missing `model.pt` weights | `VideoPipelineRunner` searches primary then fallback checkpoint path | Falls back gracefully to reference architecture with logged warning. |
| **Zero Detected Objects** | Pitch-black night / heavy occlusions | Bounding box arrays empty ($0\text{ detections}$) | Feature extractor logs 0 detections; LSTM predicts high failure risk. |
| **Fractional or Non-Positive FPS** | Truncated video headers, exotic codecs | OpenCV reports $\text{FPS} \le 0$ or $\text{NaN}$ | Automatically validated and clamped to `frame_rate = max(1, int(round(fps)))`. |
| **Backend Process Termination** | Unscheduled server restart | In-memory jobs dict wiped | On reboot, `_rehydrate_existing_jobs` restores all completed jobs from disk. |
| **Disk Space Accumulation** | High-volume video uploads | Staging videos in `uploads/` | Worker automatically unlinks upload staging files in a `finally` block upon completion. |
