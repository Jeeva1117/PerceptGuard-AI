# PerceptGuard AI – Final System Audit & Pipeline Validation Report

---

## A. Executive Summary

This document presents the comprehensive, evidence-based final system audit and quality validation report for **PerceptGuard AI** (`main` branch) following the completion of Day 1, Day 2, and Day 3 (Tasks 1–6).

### Overall Project Status
PerceptGuard AI is an autonomous-driving computer vision system designed to anticipate perception reliability across upcoming video frames. The project has successfully implemented and verified the complete perception foundation, spanning video loading, letterboxed frame extraction, metadata generation, dataset validation, YOLO11n inference loading, model configuration, single-image detection, batch sequence inference, visual annotation, and statistical metrics analysis.

### Implemented Pipeline Capabilities
- **Video Processing Pipeline:** Multi-format OpenCV video capture, frame extraction at 5.0 FPS, and aspect-ratio-preserving letterboxing ($640 \times 360$).
- **Data Engineering & Governance:** Standardized dataset layout, manifest generation, quality scoring (Laplacian variance blur, luminance brightness, RMS contrast), and automated 4-tier dataset validation.
- **YOLO Perception Infrastructure:** Lazy-loaded CPU-compatible YOLO11n model loader, strongly typed configuration hierarchy with runtime presets (`balanced`, `high_precision`, `high_recall`, `cpu_lightweight`).
- **Detection Engines:** Production-grade `SingleImageDetector` and memory-bounded `BatchDetector` with chronological natural sorting and Day 2 metadata traceability.
- **Visualization Engine:** Decoupled `DetectionVisualizer` generating high-contrast, deterministic class-colored bounding boxes and labels without re-running inference.
- **Metrics & Analysis Engine:** Decoupled `DetectionMetrics` evaluating statistical detection volume, parametric confidence profiling, 9-class frequency breakdowns, sequence temporal consistency ($S_{\text{temporal}} = 0.8166$), and heuristic frame reliability ($R_{\text{heuristic}} = 0.4619$).

### Major Technical Limitations
1. **Unannotated Video Sequence:** The evaluated BDD100K driving video (`converted_026c7465-309f6d33.mp4`) does not include ground-truth 2D bounding box annotations. Consequently, accuracy metrics (IoU-matched Precision, Recall, F1, and mAP) are marked **`NOT AVAILABLE`** for this sequence, although the matching algorithms are fully implemented and verified via unit tests.
2. **Heuristic vs. Predictive Reliability:** The current reliability indicator is an exploratory frame-level heuristic combining confidence and sequence volatility. Recurrent temporal forecasting (LSTM/GRU) will be trained in Day 6.
3. **Monocular Camera Modality:** The pipeline currently processes monocular 2D camera streams; multi-sensor LiDAR/Radar fusion is out of scope.

---

## B. Component Audit

Every implemented subsystem was audited against working code, verified artifacts, and automated test suites:

| Subsystem Component | Implementation Module | Test Suite File | Status | Audit Evidence & Verification Summary |
| :--- | :--- | :--- | :---: | :--- |
| **YOLO Setup & Loader** | `src/detection/yolo_model_loader.py` | `tests/test_yolo_model_loader.py` | **PASS** | Successfully deserialized local `yolo11n.pt` (5.35 MB, SHA-256 verified) on CPU in 191.96 ms. 80 COCO classes initialized. FP32 single precision enforced. Zero unexpected downloads. |
| **YOLO Configuration** | `src/detection/yolo_config.py` | `tests/test_yolo_config.py` | **PASS** | Validated dataclass hierarchy (`ModelConfig`, `RuntimeConfig`, `InferenceConfig`, `PerformanceConfig`, `ValidationConfig`). Enforces stride-32 conformity, parameter boundaries, and presets (`balanced`, `high_precision`, `cpu_lightweight`). |
| **Single-Image Detector** | `src/detection/single_image_detector.py` | `tests/test_single_image_detector.py` | **PASS** | Validated on real driving frame `frame_000001.jpg` (2 detections: `traffic light` at conf 0.4381 and `truck` at conf 0.2961) and `frame_000013.jpg` (0 detections, status `success`). Steady-state CPU latency: 47.05 ms (~21.3 FPS). Atomic JSON persistence complying with Schema v1.0. |
| **Batch Detection Engine** | `src/detection/batch_detector.py` | `tests/test_batch_detector.py` | **PASS** | Traversed 200 preprocessed driving frames in strictly natural order. Single model load invariant verified. 211 detections extracted across 200 frames. 0 failed frames. Full run assigned `PASS`, partial smoke run assigned `WARNING`. Emitted 4 atomic JSON artifacts under `batch_runs/<run_id>/`. |
| **Detection Visualizer** | `src/visualization/detection_visualizer.py` | `tests/test_detection_visualizer.py` | **PASS** | Visualized 200 frames in 2.88 seconds (14.40 ms/frame, ~69.4 FPS). Bounding boxes clamped to $[0, W] \times [0, H]$. Labels repositioned inside margins to prevent top clipping. 34 zero-detection frames preserved as clean copies. Decoupled from inference (zero model calls). |
| **Metrics & Analysis Engine**| `src/analysis/detection_metrics.py` | `tests/test_detection_metrics.py` | **PASS** | Evaluated 200 frames (211 detections, 9 classes). Calculated confidence distributions ($\mu=0.4215, \sigma=0.1045$), sequence temporal consistency ($S=0.8166$), and heuristic reliability ($R=0.4619$). Ground-truth evaluation marked `unavailable` due to unannotated dataset. Emitted 5 atomic JSON artifacts under `analysis/<run_id>/`. |
| **Video Loader & Extractor** | `src/video_processing/` | `tests/test_video_loader.py`, `tests/test_frame_extractor.py` | **PASS** | Captured `converted_026c7465-309f6d33.mp4` (720x1280, 30.09 FPS, 1,204 frames) and extracted 200 frames at 5.0 FPS with monotonic timestamping. |
| **Frame Preprocessor** | `src/video_processing/frame_preprocessor.py` | `tests/test_frame_preprocessor.py` | **PASS** | Resized 200 frames to letterboxed $640 \times 360$ px with aspect ratio preservation. Computed blur, brightness, and contrast scores. |
| **Dataset Validator** | `src/validation/dataset_validator.py` | `tests/test_dataset_validator.py` | **PASS** | Evaluated dataset structure, integrity, image readability, and manifest consistency. Passed all 4 quality tiers. |
| **Ground-Truth Matching** | `src/analysis/detection_metrics.py` | `tests/analysis/test_detection_metrics.py` | **PASS** (Algorithm) / **NOT AVAILABLE** (Dataset) | Bipartite matching algorithm, IoU calculation, TP/FP/FN accounting, precision, recall, and F1 formulas fully verified on synthetic test data. Transparently marked `NOT AVAILABLE` on `driving_sample` dataset. |

---

## C. Dataset Audit

- **Raw Source Video:** `datasets/raw/converted_026c7465-309f6d33.mp4` (47.47 MB, FMP4 codec, 720 × 1280 resolution, 30.09 FPS, 1,204 total frames, 40.01 seconds duration). Verified intact and readable via OpenCV.
- **Extracted Frames Directory:** `datasets/processed/driving_sample/frames/` (200 JPEG images, `frame_000001.jpg` to `frame_000200.jpg`, 45.41 MB).
- **Preprocessed Frames Directory:** `datasets/processed/driving_sample/preprocessed_frames/` (200 JPEG images, letterboxed to $640 \times 360$ px, 7.02 MB).
- **Image Readability Check:** 200 / 200 frames (100.0% readable, 0 corrupt files, 0 zero-byte files).
- **Metadata Synchronization:** `datasets/processed/driving_sample/metadata.csv` contains exactly 200 records ($t = 0.0000\text{s} \to 39.6800\text{s}$) fully matching extracted frame IDs ($1 \dots 200$) and source indices ($0 \dots 1,194$).
- **Missing or Duplicate Frames:** 0 missing frames, 0 duplicate frame identifiers.

---

## D. Detection Audit

Evaluated against actual Task 4 batch inference output (`run_20261117_115348_e1789b`):

| Detection Property | Verified Metric |
| :--- | :--- |
| **Total Frames Ingested** | 200 |
| **Successfully Processed Frames** | 200 (100.0%) |
| **Failed Frames** | 0 (0.0%) |
| **Frames With Detections** | 166 (83.0%) |
| **Frames Without Detections** | 34 (17.0%) |
| **Total Objects Detected** | 211 |
| **Average Objects / Frame** | 1.05 (Min: 0, Max: 4) |
| **Mean Softmax Confidence** | 0.4215 |
| **Median Softmax Confidence** | 0.4195 |
| **Confidence Standard Deviation**| 0.1045 |
| **Confidence Dynamic Range** | [0.2522, 0.7449] |
| **Low-Confidence Detections (< 0.30)** | 33 detections (15.64%) |

### Verified Semantic Class Distribution
1. `traffic light`: 159 detections (75.36%, mean conf: 0.4489, range: [0.2596, 0.7449])
2. `truck`: 31 detections (14.69%, mean conf: 0.3289, range: [0.2537, 0.5763])
3. `car`: 7 detections (3.32%, mean conf: 0.3359, range: [0.2522, 0.5132])
4. `potted plant`: 6 detections (2.84%, mean conf: 0.3541)
5. `bus`: 3 detections (1.42%, mean conf: 0.3364)
6. `surfboard`: 2 detections (0.95%, mean conf: 0.4361)
7. `person`: 1 detection (0.47%, mean conf: 0.3705)
8. `train`: 1 detection (0.47%, mean conf: 0.3787)
9. `vase`: 1 detection (0.47%, mean conf: 0.3392)

---

## E. Metrics Audit

- **Supported Metrics:** Detection statistics, parametric confidence distributions, 5-bin histogram, class-wise distributions, frame-level telemetry, and temporal volatility metrics.
- **Temporal Consistency Analysis:**
  - Mean detection count variation: 0.3769 objects/frame.
  - Maximum count variation: 3 objects.
  - Mean confidence variation across consecutive frames: 0.0896.
  - Max consecutive zero-detection streak: 15 frames.
  - Temporal stability score: **0.8166**.
- **Heuristic Reliability Indicator:** Mean score of **0.4619** across 200 frames. Neutral baseline (0.85) applied to zero-detection frames; low-confidence and delta penalties applied to detection frames.
- **Ground-Truth Evaluation:** Marked **`NOT AVAILABLE`** due to lack of ground-truth annotations in `driving_sample`. Precision, recall, and F1 are set to `null` with explicit disclaimer. No metrics were fabricated.

---

## F. Test Suite Audit

Executed full repository automated regression suite using `pytest`:
- **Total Test Cases Executed:** **386 tests**.
- **Passing Tests:** **386 / 386 (100.0% Pass Rate)**.
- **Failures / Errors:** **0**.
- **Execution Duration:** 28.38 seconds on Intel Core i3 CPU.

### Test Coverage Breakdown by Subsystem:
- Video Loader & Extractor: 24 tests
- Frame Preprocessor: 18 tests
- Metadata Generator: 15 tests
- Dataset Structure & Validator: 44 tests
- Environment Verification: 1 test
- YOLO Model Loader & Setup: 32 tests
- YOLO Model Configuration: 23 tests
- Single-Image Detector: 26 tests
- Batch Object Detector: 29 tests
- Object Detection Visualizer: 29 tests
- Detection Metrics & Analysis: 30 tests
- Discoverability Root Wrappers: 115 tests

---

## G. Performance and Resource Audit

Benchmarked on host hardware (Intel Core i3 with 8 logical threads, 8 GB RAM, CPU-only, Windows 11):

| Pipeline Stage | Evaluated Scope | Latency / Throughput | Resource Utilization |
| :--- | :--- | :--- | :--- |
| **Model Deserialization** | YOLO11n (`yolo11n.pt`) | 191.96 ms | Disk I/O: 5.35 MB read once |
| **Cold Single-Frame Inference** | Frame 1 (JIT compile) | 4,448.47 ms | CPU peak ~85%, RAM < 300 MB |
| **Warm Single-Frame Inference** | Frame 13 (Steady-state) | **47.05 ms** (~**21.3 FPS**) | CPU ~40%, RAM < 250 MB |
| **Batch Inference (Sequential)** | 200 frames ($640 \times 360$) | **23.27 s** (Median: 67.89 ms, ~14.7 FPS)| Stable RSS < 300 MB |
| **Detection Visualization** | 200 frames (211 boxes) | **2.88 s** (Mean: 14.40 ms, ~69.4 FPS) | Stable RSS < 250 MB |
| **Metrics & Analysis Engine** | 200 detection records | **0.12 s** | Instantaneous, RAM < 150 MB |
| **Complete End-to-End Test** | 3-frame controlled slice | **0.48 s** | Seamless inter-stage data flow |

---

## H. GitHub Audit

- **Repository URL:** `https://github.com/Jeeva1117/PerceptGuard-AI.git`
- **Active Branch:** `main`
- **Synchronization:** Local `main` is perfectly synchronized with `origin/main` (0 ahead, 0 behind).
- **Working Tree:** Clean (`git status` reports working tree clean).
- **Data Governance & Exclusions:** Verified that `.gitignore` strictly excludes:
  - Video binaries (`*.mp4`, `datasets/raw/*`)
  - Processed image datasets (`datasets/processed/*`)
  - Model checkpoints (`*.pt`, `models/*`)
  - Environment variables and keys (`.env`, `*.pem`, `*.key`)
  - Caches and virtual environments (`venv/`, `__pycache__/`, `.pytest_cache/`)
- **Commit History Standard:** Recent commits adhere to conventional format with short, focused subjects without Day or Task numbers.

---

## I. Remaining Issues

1. **Ground-Truth Annotations Missing for Raw Sequence:** The raw driving video lacks accompanying 2D bounding box annotations (e.g., BDD100K labels). Precision, recall, and mAP cannot be evaluated until annotated driving sequences are integrated.
2. **First-Frame JIT Warm-Up Overhead:** PyTorch CPU execution incurs an initial ~4.4 second JIT kernel compilation overhead on the first forward pass. Steady-state latency immediately drops to ~47 ms.
3. **Absence of Temporal Object Tracking:** Current temporal metrics evaluate sequence frequency volatility rather than individual vehicle tracklets. Multi-object tracking (ByteTrack) is scheduled for Day 3 Task 5+ / Day 4.

---

## J. Final Verdict

### **PASS WITH LIMITATIONS**

**Justification:**
All implemented modules—from video loading and frame preprocessing through single-image detection, batch processing, visualization, and metrics analysis—are fully implemented, structurally decoupled, memory-bounded on CPU, and verified with **386 passing automated tests (100% pass rate)**. The limitation reflects the honest, transparent absence of ground-truth bounding box labels for the raw BDD100K driving video, which appropriately marks ground-truth accuracy metrics as `NOT AVAILABLE` rather than fabricating synthetic evaluation claims.

The PerceptGuard AI computer vision foundation is verified, sound, reproducible, and ready for multi-object tracking and recurrent reliability forecasting.
