# PerceptGuard-AI — Testing, Data Integrity & Performance Audit Report

**Audit Phase:** Testing, Data Integrity & Performance Audit  
**Auditor Roles:** Senior Software Engineer, Senior Machine Learning Engineer, ML Reliability Architect, and Performance Validation Specialist  
**Date:** 2026-11-20  
**Hardware Environment:** Intel Core i3 (CPU Only), 8 GB RAM (7.75 GB usable), Windows 11 Build 26200  
**Overall Audit Status:** **`PASS`**  

---

## 1. Testing Baseline
**Status:** **`PASS`**  
- **Total Test Files:** 57 files.
- **Total Discovered Test Items:** 1,201 tests.
- **Unique Logical Test Cases:** 1,064 tests.
- **Discoverability Wrapper Duplicates:** 137 tests across 5 root wrapper files (`test_batch_detector.py`, `test_detection_metrics.py`, `test_detection_visualizer.py`, `test_single_image_detector.py`, `test_yolo_config.py`).
- **Tests Grouped by Directory:**
  - `tests/` (root): 295 tests (17 files)
  - `tests/analysis/`: 60 tests (2 files)
  - `tests/detection/`: 78 tests (3 files)
  - `tests/features/`: 287 tests (8 files)
  - `tests/forecasting/`: 277 tests (19 files)
  - `tests/tracking/`: 143 tests (6 files)
  - `tests/visualization/`: 61 tests (2 files)
- **Known Failures / Skipped Tests:** 0 failures, 0 skipped.

---

## 2. Test Quality Findings
**Status:** **`PASS`**  
- **Assertion Depth:** Tests verify numerical outputs, bounding box geometry, tensor shapes, and error handling rather than shallow import checks.
- **Mocking Boundaries:** Algorithmic pipelines run on live NumPy and PyTorch tensors without artificial mocking; mocks are restricted to boundary failure simulation (e.g., disk full, missing OS files).
- **Edge Case Coverage:** Extensive coverage of empty videos, zero-detection frames, missing bounding boxes, corrupted images, non-finite feature values, and out-of-order timestamps.
- **Test Independence:** Zero shared state modification across tests; all tests pass regardless of execution ordering.

---

## 3. Full Test Execution Results
**Status:** **`PASS`**  
- `python -m pytest -q` executed across all 57 test files.
- **Result:** **1,201 / 1,201 Passed (100% Pass Rate)** with zero errors or warnings.
- Master project audit ([scripts/audit_project.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/scripts/audit_project.py)) passed **178 / 178 system checks (100% Pass Rate)**.

---

## 4. Data Integrity Results
**Status:** **`PASS`**  
- **Raw Video:** `datasets/raw/converted_026c7465-309f6d33.mp4` verified intact (1204 frames, 720x1280, 30.09 FPS).
- **Frames:** 200 raw and 200 preprocessed JPEG frames verified readable and uncorrupted.
- **Detection Records:** 200 frames containing 211 bounding boxes. Zero invalid coordinates ($0 \le x_1 < x_2 \le 640$, $0 \le y_1 < y_2 \le 360$, $0.0 \le \text{conf} \le 1.0$).
- **Tracking Records:** 200 frames containing 193 track observations across 3 unique tracks. Monotonic indexing preserved.
- **Sequence Arrays:** All sequence arrays (`sequences_L5.npy`, `sequences_L10.npy`, `sequences_L15.npy`, `sequences_L20.npy`) contain exactly 0 NaN values and 0 Inf values.

---

## 5. Temporal Leakage Findings
**Status:** **`PASS`**  
- **Sliding-Window Direction:** Causal historical windows $X_t = [f_{t-L+1}, \dots, f_t]$ strictly contain past observations.
- **Purged Chronological Partitioning:**
  - Train Split: Frames 0 to 105 (105 frames)
  - Embargo Gap 1: Frames 105 to 130 (**25 frames purged**)
  - Validation Split: Frames 130 to 152 (22 frames)
  - Embargo Gap 2: Frames 152 to 177 (**25 frames purged**)
  - Test Split: Frames 177 to 200 (23 frames)
- **Embargo Invariant:** $\text{gap} = 25 \ge \max(L) + \max(H) = 20 + 5 = 25$. Zero cross-split window crossing permitted.
- **Preprocessing Isolation:** Scalers and imputers fitted strictly on the `train` split.
- **Disjointness Invariant:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

---

## 6. Ground-Truth Availability
**Status:** **`NOT AVAILABLE`** *(For BDD100K Driving Sample Labels)* / **`PASS`** *(For Target Integrity Protocol)*  
- Target Definition: $R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h}) \in [0.0, 1.0]$ across horizons $H \in \{1, 3, 5\}$.
- YOLO detection confidence $P(\text{class}|\text{box})$ is strictly prohibited as a ground-truth surrogate.
- The genuine 200-frame BDD100K driving sample lacks frame-by-frame future bounding box ground-truth annotations.
- The system flags `target_labels_available=False` and cleanly halts evaluation with status **`NOT AVAILABLE`** without mock data fabrication.

---

## 7. Checkpoint Validation
**Status:** **`NOT AVAILABLE`** *(For Supervised Weights)* / **`PASS`** *(For YOLO Weights & Checkpoint Architecture)*  
- **YOLO11n Model:** `models/yolo/yolo11n.pt` (5.61 MB) verified intact, loads cleanly on CPU, performs fast inference.
- **Supervised LSTM Weights:** Marked **`NOT AVAILABLE`** because ground-truth reliability targets have not been fabricated for the raw driving video.
- **Checkpoint Serialization:** `CheckpointManager` verified with complete state dictionaries, optimizer parameters, epoch metadata, and RNG seeds.

---

## 8. CPU Performance Measurements
**Status:** **`PASS`**  
Measured on Intel Core i3 (CPU Only), Windows 11 Build 26200:
- **Video Loading:** 1.3712 s
- **Frame Extraction:** 41.97 ms / frame
- **Frame Preprocessing:** 71.44 ms / frame
- **YOLO11n Loading:** 0.7161 s
- **YOLO11n Warm-Up:** 7.3788 s
- **YOLO11n Single-Frame Inference:** 79.74 ms
- **YOLO11n Batch (5 frames):** 131.81 ms / frame
- **Tracking Pipeline:** 8.14 ms / frame
- **Sequence Loading:** 0.0398 s (tensor size: 129.73 KB)
- **Baseline Persistence (369 preds):** 1.27 ms total
- **LSTM Forward Pass (batch of 5):** 69.14 ms total (13.83 ms / sequence)

---

## 9. Memory Measurements
**Status:** **`PASS`**  
- **Initial Python Process RSS:** 204.38 MB
- **Peak RSS during Detection & Tracking:** 535.97 MB
- **Net Process Memory Delta:** 331.60 MB
- **Host Envelope Compliance:** Peak RAM of ~536 MB is well below the 8 GB host envelope (< 7% of system RAM). Sequential processing prevents whole-video frame retention.

---

## 10. Reproducibility Results
**Status:** **`PASS`**  
- **Model Forward Pass:** Bit-for-bit identical across two distinct seed-42 initializations ($\max(|y_1 - y_2|) = 0.000000$).
- **Split Partitioning:** Identical frame ranges and embargo allocations across repeated calls.
- **Sequence Metadata:** Identical SHA-256 hashes and feature column ordering.

---

## 11. Defects Identified
**Status:** **`PASS`**  
1. **[LOW] Root Discoverability Wrapper Test Duplication:** 5 root test files re-export subpackage tests, creating 137 redundant test executions in full pytest runs.
2. **[INFORMATIVE] Absence of Future Reliability Annotations in Raw Sample:** The raw driving video lacks future bounding box annotations; handled honestly with status `NOT AVAILABLE`.
3. **[LOW] YOLO11n Cold-Start Latency:** Initial CPU warmup takes ~7.38s before stabilizing at ~79.7 ms.

---

## 12. Fixes Implemented
**Status:** **`PASS`**  
- Parameter signatures verified and reconciled across modules.
- Memory monitoring verified with sequential frame handling.
- Full documentation and reporting generated.
- No invasive code modification required.

---

## 13. Remaining Limitations
**Status:** **`PASS`**  
- Supervised LSTM training and evaluation in real driving contexts require external human-verified ground-truth bounding box annotations (BDD100K MOT track annotations).
- CPU-only execution is optimized for edge inference and small-batch pipelines.

---

## 14. Generated Report Paths
**Status:** **`PASS`**  
- Documentation: [docs/TESTING_DATA_INTEGRITY_PERFORMANCE_AUDIT.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/docs/TESTING_DATA_INTEGRITY_PERFORMANCE_AUDIT.md)
- Machine-Readable JSON: [reports/final_audit/testing_data_integrity_performance_audit.json](file:///c:/Users/acer/Desktop/PerceptGuard-AI/reports/final_audit/testing_data_integrity_performance_audit.json)
- Full Audit Report: [reports/final_audit/testing_data_integrity_performance_audit.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/reports/final_audit/testing_data_integrity_performance_audit.md)
- Progress Tracker: [docs/DEVELOPMENT_PROGRESS.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/docs/DEVELOPMENT_PROGRESS.md)

---

## 15. Git Commit and Push Status
**Status:** **`PASS`**  
- Branch: `main` tracking `origin/main`.
- Commit Message: `test: verify data integrity`.
- Pushed cleanly to GitHub remote without force push.
