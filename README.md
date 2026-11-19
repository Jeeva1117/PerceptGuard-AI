# PerceptGuard-AI: Predictive Object Detection Reliability System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CPU%20%2F%20CUDA)-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous-driving perception research system that **forecasts future object detection reliability degradation** over upcoming video frames ($t+1, t+3, t+5$).

---

## 1. Problem Statement

In automated driving systems (ADS) and advanced driver-assistance systems (ADAS), camera-based perception pipelines rely heavily on deep object detectors (e.g., YOLO, SSD, DETR). Standard systems rely directly on the detector's **confidence score** ($P(\text{class} | \text{box})$) as a proxy for perception quality.

### Confidence $\neq$ Reliability

Detection confidence is fundamentally miscalibrated for real-time safety guarantees:
1. **False Confidence:** Deep detectors frequently assign high confidence (>85%) to bounding boxes that are poorly localized, partially occluded, or visually drifted.
2. **Sudden Dropouts:** When encountering sudden environmental perturbations (lens glare, rapid ego-vehicle pitch/roll, motion blur, roadside occlusion), confidence drops precipitously with **zero advance warning**.
3. **Reactive Safety Planners:** Downstream autonomous trajectory planners and emergency braking modules receive bounding boxes naively until track divergence or missed detection occurs, leaving no time budget for graceful degradation or sensor handoff.

```
+-----------------------------------------------------------------------------------+
| Standard Perception:     Detector Confidence [0.92] ---> Drop to [0.0] (CRASH RISK)|
| PerceptGuard-AI:         Predicts Reliability Drop 100ms In Advance (SAFE HANDOFF) |
+-----------------------------------------------------------------------------------+
```

---

## 2. Project Vision & Objectives

PerceptGuard-AI introduces a predictive meta-perception layer:
- **Forecast Future Detection Reliability:** Predict how much object localization quality (measured by future Intersection over Union, $\text{IoU}$) will decrease over the next **1 frame (~33ms), 3 frames (~100ms), and 5 frames (~166ms)**.
- **Multi-Modal Feature Fusion:** Extract temporal confidence dynamics, kinematic motion vectors, bounding box scale/aspect ratios, optical quality (Laplacian blur, contrast), and spatial boundary proximity.
- **Recurrent Forecasting:** Train a recurrent model (LSTM / GRU) on sliding historical sequences to anticipate perception failure modes before they manifest.
- **Human-in-the-Loop & Fleet Telemetry:** Provide a real-time Streamlit dashboard visualizing predicted reliability trajectories, early warning alerts, and track health telemetry.

---

## 3. System Architecture

The pipeline processes video streams sequentially through modular, decoupled stages:

```
Video Stream (Dashcam / BDD100K / KITTI)
                 │
                 ▼
[ Module 1 ] Video Frame Extraction & Ingestion (OpenCV)
                 │
                 ▼
[ Module 2 ] Real-Time Object Detection (YOLOv8)
                 │
                 ▼
[ Module 3 ] Multi-Object Tracking (ByteTrack)
                 │
                 ▼
[ Module 4 ] Feature Extraction & Temporal Sequence Buffer (W=10 frames)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[ Module 5 ]          [ Module 6 ]
Offline GT IoU         Recurrent Forecaster (LSTM / GRU)
Supervision Target     Multi-Horizon Prediction (t+1, t+3, t+5)
      └──────────┬──────────┘
                 ▼
[ Module 7 ] Evaluation & Safety Metrics (MSE, MAE, Early-Warning F1)
                 │
                 ▼
[ Module 8 ] Streamlit Interactive Perception Dashboard
```

For complete mathematical definitions and module interface contracts, consult [docs/architecture.md](docs/architecture.md).

---

## 4. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Runtime & Language** | Python 3.10+ / 3.11 / 3.13, Windows PowerShell & POSIX Bash compatible |
| **Deep Learning** | PyTorch (CPU-optimized baseline, seamless CUDA acceleration) |
| **Computer Vision** | OpenCV (`opencv-python`), Ultralytics YOLOv8 (`ultralytics`) |
| **Object Tracking** | ByteTrack (Kalman Filter + Two-Stage High/Low Confidence Association) |
| **Sequence Modeling** | PyTorch LSTM / GRU recurrent architectures |
| **Data & Scientific** | NumPy, Pandas, Scikit-learn, SciPy |
| **Visualization & UI** | Streamlit, Matplotlib |
| **Tooling & Standards** | PEP 8, Dataclass contracts, Git, Python virtual environment (`venv`) |

---

## 5. Project Directory Structure

```
PerceptGuard-AI/
├── config/                     # Centralized application & hyperparameter configuration
│   ├── __init__.py
│   └── settings.py             # Pathlib paths, thresholds, device auto-detection
├── datasets/                   # Dataset storage (excluded from Git tracking)
│   ├── raw/                    # Raw driving sequences & annotations (.gitkeep)
│   └── processed/              # Engineered feature sequences & sliding windows (.gitkeep)
├── docs/                       # Engineering specifications & research documentation
│   └── architecture.md         # Full system architecture & mathematical formulation
├── dashboard/                  # Streamlit perception monitoring dashboard
│   └── __init__.py
├── models/                     # Checkpoints & serialized weights (.gitkeep)
├── notebooks/                  # Exploratory data analysis & feature research (.gitkeep)
├── outputs/                    # Visualizations, telemetry logs, evaluation plots (.gitkeep)
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── detection/              # Frame extraction & YOLO object detection module
│   │   └── __init__.py
│   ├── tracking/               # ByteTrack multi-object tracking module
│   │   └── __init__.py
│   ├── features/               # Feature extraction & temporal windowing
│   │   └── __init__.py
│   ├── forecasting/            # Perception reliability forecasting module
│   │   ├── __init__.py
│   │   ├── forecasting_config.py
│   │   ├── forecasting_environment.py
│   │   ├── forecasting_data_contract.py
│   │   ├── baseline_forecaster.py
│   │   ├── baseline_data_loader.py
│   │   ├── baseline_experiment.py
│   │   ├── baseline_metrics.py
│   │   ├── lstm_forecaster.py
│   │   ├── model_factory.py
│   │   ├── training_data.py
│   │   ├── early_stopping.py
│   │   ├── checkpoint_manager.py
│   │   ├── training_logger.py
│   │   ├── training_engine.py
│   │   ├── train_lstm.py
│   │   └── multi_horizon_forecaster.py
│   ├── prediction/             # Legacy prediction stubs
│   └── evaluation/             # Ground-truth IoU labeling & validation metrics
│       └── __init__.py
├── tests/                      # Unit & integration test suites
│   ├── __init__.py
│   └── verify_environment.py   # System & dependency verification script
├── .env.example                # Template environment variables
├── .gitignore                  # Git exclusion rules (datasets, checkpoints, venv)
├── main.py                     # Primary system CLI entrypoint
├── README.md                   # Project technical documentation
└── requirements.txt            # Production dependencies
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.13 installed.
- Git installed.
- Works on Windows, macOS, and Linux. Compatible with lightweight CPUs (e.g., Intel Core i3 with 8 GB RAM) or dedicated NVIDIA GPUs.

### 1. Clone the Repository
```bash
git clone https://github.com/Jeeva1117/PerceptGuard-AI.git
cd PerceptGuard-AI
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install dependencies using the CPU-compatible PyTorch distribution:
```bash
pip install -r requirements.txt
```

### 4. Verify Environment Setup
Run the automated verification suite:
```bash
python tests/verify_environment.py
```

### 5. Inspect Runtime Configuration
```bash
python main.py --info
```

### 6. Verify YOLO Perception Environment & Presets
Run the CPU-compatible YOLO detection verification tool with default settings or operational presets:
```bash
# Default balanced configuration
python -m src.detection.yolo_model_loader --verify

# High-precision or lightweight CPU presets
python -m src.detection.yolo_model_loader --verify --preset high_precision
python -m src.detection.yolo_model_loader --verify --preset cpu_lightweight
```

### 7. Run Single Image Object Detection Engine
Execute object detection on a single camera frame and persist machine-readable JSON:
```bash
python -m src.detection.single_image_detector --image datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg --save-json --overwrite
```
For architecture specifications, schema documentation, and benchmarks, consult [docs/SINGLE_IMAGE_DETECTION.md](docs/SINGLE_IMAGE_DETECTION.md) and [docs/YOLO_MODEL_CONFIGURATION.md](docs/YOLO_MODEL_CONFIGURATION.md).

### 8. Run Advanced Batch Object Detection Engine
Execute sequential, memory-bounded batch inference across entire video frame datasets with Day 2 metadata correlation:
```bash
# Full dataset batch run on standard driving sample
python -m src.detection.batch_detector --dataset driving_sample

# Rapid smoke test (first 10 frames) with verbose output
python -m src.detection.batch_detector --dataset driving_sample --max-frames 10 --overwrite --verbose

# Sliced frame range processing
python -m src.detection.batch_detector --dataset driving_sample --start-index 50 --end-index 150
```
For architecture specifications, schema v1.0 contracts, and benchmarks, consult [docs/BATCH_OBJECT_DETECTION.md](docs/BATCH_OBJECT_DETECTION.md).

### 9. Run Object Detection Visualization Engine
Render deterministic, high-contrast bounding boxes, class labels, and confidence scores directly from Task 4 detection JSON without re-running inference:
```bash
# Full dataset visualization on actual Task 4 detection output
python -m src.visualization.detection_visualizer --detection-file datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --overwrite

# Rapid 10-frame smoke visualization test
python -m src.visualization.detection_visualizer --detection-file datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --max-frames 10 --overwrite --verbose
```
For architecture specifications, schema v1.0 visual contracts, and benchmarks, consult [docs/DETECTION_VISUALIZATION.md](docs/DETECTION_VISUALIZATION.md).

### 10. Run Detection Metrics & Perception Analysis Engine
Evaluate detection statistical distributions, parametric confidence profiles, sequence temporal stability, and heuristic reliability indicators:
```bash
# Full dataset detection metrics evaluation on Task 4 output
python -m src.analysis.detection_metrics --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --overwrite

# Custom confidence and IoU matching thresholds
python -m src.analysis.detection_metrics --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --confidence-threshold 0.35 --iou-threshold 0.60
```
For architecture specifications, mathematical formulations, and benchmarks, consult [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md).

### 11. Run Tracking Environment & Configuration Verification
Verify ByteTrack tracking infrastructure, parameter boundaries, and solver availability:
```bash
# Validate default ByteTrack configuration
python -m src.tracking.tracker_config --validate

# Run tracking runtime environment and hardware audit
python -m src.tracking.tracking_environment

# Emit machine-readable telemetry JSON for CI/CD pipelines
python -m src.tracking.tracking_environment --json
```
For architecture specifications, parameter definitions, and compatibility analysis, consult [docs/TRACKING_ENVIRONMENT_SETUP.md](docs/TRACKING_ENVIRONMENT_SETUP.md).

### 12. Run ByteTrack Multi-Object Tracking Pipeline
Associate sequential frame detections into persistent tracklets ($T_{id}$) across video frames:
```bash
# Ingest pre-computed batch detections (default latest run in driving_sample)
python -m src.tracking.tracking_pipeline

# Specify explicit detection results file
python -m src.tracking.tracking_pipeline --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json

# Rapid 10-frame smoke test with verbose logging
python -m src.tracking.tracking_pipeline --max-frames 10 --verbose --overwrite
```
For architecture specifications, mathematical formulations, and benchmarks, consult [docs/BYTETRACK_INTEGRATION.md](docs/BYTETRACK_INTEGRATION.md).

### 13. Run Single-Object Tracking Engine
Isolate a designated target object and follow its persistent tracking identity, state transitions, and kinematic motion:
```bash
# Follow target by explicit Track ID
python -m src.tracking.single_object_tracker \
  --input datasets/processed/driving_sample/tracking/track_20261117_142221_09ee12/tracking_results.json \
  --target-id 1

# Follow target by Semantic Class with disambiguation
python -m src.tracking.single_object_tracker \
  --input datasets/processed/driving_sample/tracking/track_20261117_142221_09ee12/tracking_results.json \
  --target-class truck \
  --selection-rule highest_confidence \
  --max-lost-frames 30

# Rapid smoke test on first 50 frames
python -m src.tracking.single_object_tracker --target-id 1 --max-frames 50 --overwrite
```
For architecture specifications, state machine definitions, and benchmarks, consult [docs/SINGLE_OBJECT_TRACKING.md](docs/SINGLE_OBJECT_TRACKING.md).

### 14. Run Production Multi-Object Tracking (MOT) Engine
Detect and track multiple objects simultaneously with persistent identities, lifecycle states, and per-object histories:
```bash
# Ingest pre-computed batch detections (fastest, zero duplicate YOLO passes)
python -m src.tracking.multi_object_tracker \
  --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json

# Run end-to-end detection and tracking on preprocessed frames
python -m src.tracking.multi_object_tracker \
  --input datasets/processed/driving_sample/preprocessed_frames \
  --max-frames 30 \
  --overwrite

# Run on raw driving video file (.mp4)
python -m src.tracking.multi_object_tracker \
  --input datasets/raw/converted_026c7465-309f6d33.mp4 \
  --max-frames 50
```
For architecture specifications, lifecycle machine definitions, and benchmarks, consult [docs/MULTI_OBJECT_TRACKING.md](docs/MULTI_OBJECT_TRACKING.md).

### 15. Run Production Tracking Visualization Engine
Render sequential tracking results into annotated frames and compiled MP4 video with persistent colors, centroid motion trajectories, and telemetry HUD overlays:
```bash
# Full dataset visualization on multi-object tracking output with video export
python -m src.visualization.tracking_visualizer \
  --tracking-file datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
  --save-video \
  --overwrite

# Rapid 30-frame smoke test with video export
python -m src.visualization.tracking_visualizer \
  --tracking-file datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
  --max-frames 30 \
  --save-video

# Visualize single-object tracking sequence
python -m src.visualization.tracking_visualizer \
  --tracking-file datasets/processed/driving_sample/tracking/single_object/single_target_1_20261117_154123_fa3f36/single_object_tracking.json \
  --save-video
```
For architecture details, deterministic color generation, and benchmarks, consult [docs/TRACKING_VISUALIZATION.md](docs/TRACKING_VISUALIZATION.md).

### 16. Run Production Tracking Metrics & Perception Analysis Engine
Evaluate tracklet stability, kinematic displacement, observation continuity, and frame-level telemetry from sequential tracking results:
```bash
# Full sequence evaluation on multi-object tracking results
python -m src.analysis.tracking_metrics \
  --tracking-file datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
  --overwrite

# Rapid 30-frame smoke test evaluation
python -m src.analysis.tracking_metrics \
  --tracking-file datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
  --max-frames 30 \
  --overwrite

# Evaluate single-object tracking sequence
python -m src.analysis.tracking_metrics \
  --tracking-file datasets/processed/driving_sample/tracking/single_object/single_target_1_20261117_154123_fa3f36/single_object_tracking.json \
  --overwrite
```
For metric formulations, ground-truth rule definitions, and benchmarks, consult [docs/TRACKING_METRICS_ANALYSIS.md](docs/TRACKING_METRICS_ANALYSIS.md).

### 17. Run Feature Engineering Environment Verification
Audit system hardware, CPU compute allocations, scientific dependencies, configuration adherence, and upstream artifact integrity prior to feature extraction:
```bash
# Human-readable console audit report
python -m src.features.feature_environment

# Machine-readable JSON telemetry
python -m src.features.feature_environment --json

# Validate configuration and inspect operational presets
python -m src.features.feature_config --validate
python -m src.features.feature_config --preset fast_cpu
```
For architectural specifications, Schema v1.0 field definitions, and presets, consult [docs/FEATURE_ENGINEERING_SETUP.md](docs/FEATURE_ENGINEERING_SETUP.md).

### 18. Run Detection Feature Extractor
Extract detection-level kinematic geometries and frame-level aggregated perception features from existing YOLO inference results:
```bash
# Ingest pre-computed batch detections (default driving_sample run)
python -m src.features.detection_feature_extractor \
  --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json \
  --overwrite

# Run with custom destination and frame limit
python -m src.features.detection_feature_extractor \
  --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json \
  --output datasets/processed/driving_sample/features/detection/latest \
  --max-frames 30 \
  --overwrite
```
For architectural specifications, feature formulas, and benchmark results, consult [docs/DETECTION_FEATURE_EXTRACTION.md](docs/DETECTION_FEATURE_EXTRACTION.md).

### 19. Run Motion Feature Extractor
Extract frame-to-frame displacements, velocities, 8-cardinal directional sectors, and scale changes from sequential tracking results:
```bash
# Ingest Multi-Object Tracking results (default driving_sample run)
python -m src.features.motion_feature_extractor \
  --input datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
  --overwrite

# Run with custom destination, stationary threshold, and gap tolerance
python -m src.features.motion_feature_extractor \
  --input datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
  --output datasets/processed/driving_sample/features/motion/latest \
  --stationary-threshold 1.0 \
  --max-gap 5 \
  --overwrite
```
For architectural specifications, kinematic formulas, and benchmark results, consult [docs/MOTION_FEATURE_EXTRACTION.md](docs/MOTION_FEATURE_EXTRACTION.md).

### 20. Run Visual Quality Feature Extractor
Extract optical blur (Laplacian variance), brightness statistics (mean, median, std, min, max), contrast metrics (RMS contrast, dynamic range, P95-P5 percentile range, normalized contrast), and image geometry directly from raw frame pixels without re-running YOLO or ByteTrack:
```bash
# Ingest preprocessed video frames with Day 2 metadata.csv
python -m src.features.visual_quality_extractor \
  --input datasets/processed/driving_sample/preprocessed_frames \
  --metadata datasets/processed/driving_sample/metadata.csv \
  --output datasets/processed/driving_sample/features/visual_quality/run_official \
  --dataset-id driving_sample \
  --overwrite

# Run partial batch with custom frame limits
python -m src.features.visual_quality_extractor \
  --input datasets/processed/driving_sample/preprocessed_frames \
  --max-frames 50 \
  --overwrite
```
For architectural specifications, mathematical formulations, and benchmark results, consult [docs/VISUAL_QUALITY_FEATURE_EXTRACTION.md](docs/VISUAL_QUALITY_FEATURE_EXTRACTION.md).

### 21. Run Temporal Feature Aggregator
Unify multi-modal perception streams (detection, motion, and visual quality) and compute trailing past-looking window aggregate statistics and Ordinary Least Squares (OLS) linear trend slopes with zero future-frame leakage:
```bash
# Ingest detection, motion, and visual quality feature artifacts
python -m src.features.temporal_feature_aggregator \
  --detection datasets/processed/driving_sample/features/detection/det_features_driving_sample_20261118_141511_9e022a/frame_detection_features.json \
  --motion datasets/processed/driving_sample/features/motion/motion_features_driving_sample_20261118_143109_bb4a56/frame_motion_features.json \
  --visual datasets/processed/driving_sample/features/visual_quality/run_official/visual_quality_features.json \
  --output datasets/processed/driving_sample/features/temporal/run_official \
  --window-sizes 3,5,10 \
  --overwrite

# Run partial batch with custom frame slice
python -m src.features.temporal_feature_aggregator \
  --detection datasets/processed/driving_sample/features/detection/det_features_driving_sample_20261118_141511_9e022a/frame_detection_features.json \
  --motion datasets/processed/driving_sample/features/motion/motion_features_driving_sample_20261118_143109_bb4a56/frame_motion_features.json \
  --visual datasets/processed/driving_sample/features/visual_quality/run_official/visual_quality_features.json \
  --max-frames 50 \
  --overwrite
```
For architectural specifications, mathematical formulations, and benchmark results, consult [docs/TEMPORAL_FEATURE_AGGREGATION.md](docs/TEMPORAL_FEATURE_AGGREGATION.md).

### 22. Run Temporal Sequence Builder
Construct model-ready chronological sliding-window sequence tensors for multi-horizon predictive reliability forecasting ($L \in \{5, 10, 15, 20\}$ and horizons $H \in \{1, 3, 5\}$):
```bash
# Generate model sequences with standardization and training-derived imputation
python -m src.features.temporal_sequence_builder \
  --input datasets/processed/driving_sample/features/temporal/run_official/temporal_features.json \
  --output datasets/processed/driving_sample/features/sequences/run_official \
  --sequence-lengths 5,10,15,20 \
  --horizons 1,3,5 \
  --scaling standardization \
  --missing-policy impute_mean \
  --overwrite \
  --verbose

# Dry-run validation of inputs and sequence feasibility
python -m src.features.temporal_sequence_builder \
  --input datasets/processed/driving_sample/features/temporal/run_official/temporal_features.json \
  --validate-only
```
For architectural specifications, mathematical formulations, and benchmark results, consult [docs/TEMPORAL_SEQUENCE_CONSTRUCTION.md](docs/TEMPORAL_SEQUENCE_CONSTRUCTION.md).

---

## 7. Development Roadmap

- [x] **Day 1: Technical Foundation & Architecture Finalization**
  - Project architecture, data contracts, and mathematical formulation finalized.
  - Isolated virtual environment initialized.
  - Dependencies installed and verified (PyTorch CPU, OpenCV, Ultralytics, NumPy, Pandas, Scikit-learn, Streamlit).
  - Modular project structure, configuration separation, and validation scripts created.
  - GitHub repository readiness achieved.
- [x] **Day 2: Dataset Collection & Video Processing Pipeline**
  - Acquired real-world BDD100K driving sequence (`converted_026c7465-309f6d33.mp4`).
  - Implemented `VideoLoader`, `FrameExtractor`, and letterboxed `FramePreprocessor` ($640 \times 360$).
  - Implemented `MetadataGenerator`, `DatasetOrganizer`, and multi-tier `DatasetValidator`.
  - Audited 200 extracted and preprocessed frames with 100% quality gate approval.
- [x] **Day 3: Object Detection & Perception Pipeline**
  - [x] **Task 1: Advanced YOLO Environment & Pretrained Model Setup**
    - Configured CPU-optimized YOLO11n (`yolo11n.pt`) with lazy-loaded inference engine.
    - Verified sub-50ms CPU inference (~20.6 FPS) on real driving frames.
    - Implemented `YOLOModelLoader`, configuration schema, and 18-test test suite.
  - [x] **Task 2: Advanced YOLO Model Configuration**
    - Strongly typed, validated dataclass configuration hierarchy (`src/detection/yolo_config.py`).
    - Stride conformity checks (multiples of 32), boundary validations, and COCO class filtering.
    - Documented perception presets (`balanced`, `high_precision`, `high_recall`, `cpu_lightweight`).
    - Automated test suite (23 configuration tests, 158 repository tests passing).
  - [x] **Task 3: Single Image Object Detection Engine**
    - Production-grade `SingleImageDetector` with image validation and channel standardization.
    - Sub-50ms warm inference on CPU; calibrated bounding box coordinates and COCO class mapping.
    - Versioned Schema v1.0 JSON output and atomic persistence.
    - Automated test suite (26 detection tests, 210 repository tests passing).
  - [x] **Task 4: Advanced Batch Object Detection Engine**
    - Deterministic dataset discovery, natural chronological ordering, and Day 2 metadata synchronization.
    - Single-pass model load invariant; memory-bounded sequential CPU inference loop (< 300 MB RSS).
    - Per-frame fault containment; 4 versioned (v1.0) output artifacts (`results`, `summary`, `errors`, `metadata`).
    - Automated test suite (29 batch detector tests, 268 repository tests passing).
  - [x] **Task 5: Advanced Object Detection Visualization Engine**
    - High-contrast bounding box and label rendering decoupled from inference (zero YOLO reload).
    - Mathematically deterministic class color system; boundary clamping and non-clipping label positioning.
    - Pristine zero-detection frame handling; atomic multi-file persistence (`annotated_frames/`, summary, errors, metadata).
    - Automated test suite (29 visualization tests, 297+ repository tests passing).
  - [x] **Task 6: Detection Metrics & Reliability Analysis**
    - Statistical detection volume, confidence distributions, and class-wise breakdowns.
    - Sequence temporal consistency analysis ($S_{\text{temporal}} = 0.8166$) and frame-level heuristic reliability indicator.
    - Ground-truth bipartite matching architecture; honest `"unavailable"` accounting when labels are absent.
    - Automated test suite (30 analysis tests, 356+ repository tests passing).
  - [x] **Task 7: Final Perception Audit & Verification**
    - Comprehensive system audit and pipeline validation documented in `docs/FINAL_VALIDATION_REPORT.md`.
    - 386 regression tests passing (100% pass rate).
- [ ] **Multi-Object Tracking Architecture**
  - [x] **Advanced Tracking Environment & Configuration Setup**
    - Evaluated ByteTrack two-stage association against CPU constraints.
    - Decoupled `config/tracker_config.yaml` with 4 operational presets.
    - Strongly typed, immutable `TrackerConfig` loader with schema validation.
    - Standalone `TrackingEnvironmentVerifier` telemetry and CLI tool.
    - 51 automated unit and integration tests (437 repository tests passing).
  - [x] **ByteTrack Multi-Object Tracking Engine**
    - Association of sequential frame detections into persistent tracklets ($T_{id}$).
    - Kinematic state estimation using 2D discrete Kalman filter.
    - Multi-file Schema v1.0 persistence (`results`, `summary`, `metadata`, `errors`).
    - Validated on 200 real BDD100K driving frames at 0.62 ms / frame (~1,600 FPS on CPU).
    - 32 automated unit and integration tests (469 repository tests passing).
  - [x] **Advanced Single-Object Tracking Architecture**
    - Isolate target identity across sequential frames with formal 6-state machine (`UNINITIALIZED`, `INITIALIZED`, `TRACKING`, `TEMPORARILY_LOST`, `REACQUIRED`, `TERMINATED`).
    - Deterministic target selection via explicit Track ID or semantic class targeting with disambiguation rules.
    - Kinematic centroid trajectory $(c_x, c_y)$, frame displacement $(\Delta d)$, and pixel velocity $(\Delta d / \Delta t)$.
    - Multi-file Schema v1.0 persistence (`single_object_tracking.json`, `target_history.json`, `tracking_summary.json`, `tracking_metadata.json`).
    - Validated on 200 real frames at **0.007 ms / frame** (140,000+ FPS on CPU).
    - 30 automated unit and integration tests (499 repository tests passing, 100% pass rate).
  - [x] **Production Multi-Object Tracking (MOT) Engine**
    - Simultaneous multi-object detection and tracking across sequential frames using YOLO and ByteTrack.
    - Formal 5-state lifecycle management (`NEW`, `TRACKING`, `TEMPORARILY_LOST`, `REACQUIRED`, `TERMINATED`).
    - Robust per-object identity management without arbitrary ID generation.
    - Per-track historical trajectory with displacement, velocity, and duration metrics.
    - Multi-file Schema v1.0 persistence (`multi_object_tracking.json`, `track_histories.json`, `tracking_summary.json`, `metadata.json`, `errors.json`).
    - Validated on real driving dataset (30 frames with YOLO inference at 280ms/frame, 200 frames batch ingestion in 0.20s).
    - 30 automated unit and integration tests (529 repository tests passing, 100% pass rate).
  - [x] **Advanced Tracking Visualization Engine**
    - Ingests multi-object tracking, ByteTrack, or single-object tracking JSON results with zero YOLO or ByteTrack re-execution.
    - Deterministic, persistent identity color synthesis via golden-ratio HSV algorithm ($H = (T_{id} \times 0.61803) \pmod{1.0}$).
    - Motion trajectory rendering with occlusion gap protection ($|idx_2 - idx_1| \le 2$).
    - Diagnostic telemetry HUD banner (frame ID, timestamp, active track count, run ID).
    - Optional sequential OpenCV MP4 video generation with post-write validation probe.
    - Multi-file Schema v1.0 persistence (`annotated_frames/`, `annotated_video.mp4`, `visualization_summary.json`, `visualization_metadata.json`, `errors.json`).
    - Validated on 200 real driving frames at **12.33 ms / frame** (81+ FPS on CPU).
    - 32 automated unit and integration tests (561 repository tests passing, 100% pass rate).
  - [x] **Tracking Metrics & Quality Analysis**
    - Track-level kinematics (duration, displacement, velocity, bbox area).
    - Continuity indicators (observation ratio, gap analysis, reacquisition, fragmentation).
    - Ground-truth evaluation compliance (MOTA/IDF1/HOTA = NOT AVAILABLE when GT is absent).
    - Multi-file Schema v1.0 persistence (`tracking_analysis_summary.json`, `track_metrics.json`, `frame_metrics.json`, `class_metrics.json`, `metadata.json`, `errors.json`).
    - Validated on 200 real driving frames at 0.0078s execution duration (~128,000 FPS throughput on CPU).
    - 30 automated unit and integration tests (591 repository tests passing, 100% pass rate).
- [x] **Day 5: Feature Engineering & Temporal Sequence Pipeline**
  - [x] **Task 1: Feature Engineering Environment Setup**
    - Strongly typed, immutable `FeatureConfig` and `config/feature_config.yaml` with 4 operational presets.
    - Production Feature Schema v1.0 defining 25+ fields spanning detection, motion, visual quality, tracking, and metadata.
    - Multi-artifact pre-flight validation engine (`validate_input_artifacts`) auditing detection, tracking, metadata, and frames.
    - Automated `FeatureEnvironmentVerifier` runtime diagnostics and CLI tool.
    - 66 automated unit and integration tests (657 repository tests passing, 100% pass rate).
  - [x] **Task 2: Detection Feature Extraction**
    - Extracted 22 detection-level features (identity, confidence tier, absolute and normalized geometry) and 16 frame-level aggregated metrics.
    - Zero YOLO re-execution; operations run directly on pre-computed detection artifacts at ~14,500 FPS on CPU.
    - Full temporal frame continuity: zero-detection frames preserved with explicit `null` policy.
    - Atomic multi-file Schema v1.0 persistence (`detection_features.json`, `frame_detection_features.json`, summary, metadata, errors).
    - 39 automated unit and integration tests (105 feature tests, 696 repository tests passing, 100% pass rate).
  - [x] **Task 3: Motion & Kinematic Feature Extraction**
    - Extracted 35 observation-level kinematic metrics (centroid, displacement, velocity, speed, 8-sector direction, bbox changes) and 16 track-level summaries.
    - Zero YOLO or ByteTrack re-execution; sub-millisecond CPU execution across 200 frames (193 observations, 184 valid motion steps).
    - Multi-object, single-object, and ByteTrack JSON schema ingestion with spatial normalization.
    - Robust temporal continuity: gap-based motion tracking, first observation handling, and zero interpolation invariant.
    - Atomic multi-file Schema v1.0 persistence (`motion_features.json`, `track_motion_features.json`, `frame_motion_features.json`, summary, metadata, errors).
    - 43 automated unit and integration tests (148 feature tests, 739 repository tests passing, 100% pass rate).
  - [x] **Task 4: Visual Quality Feature Extraction**
    - Extracted optical blur (Laplacian variance `var(Laplacian(gray, CV_64F))`), brightness statistics (mean, median, std, min, max), and contrast metrics (RMS contrast, dynamic range, P95-P5 percentile range, normalized contrast).
    - Configurable heuristic classification tags: blur (`sharp`, `moderate`, `blurry`), brightness (`dark`, `normal`, `bright`), contrast (`low`, `normal`, `high`).
    - Zero YOLO or ByteTrack re-execution; direct pixel extraction via OpenCV and NumPy.
    - Memory-efficient CPU pipeline with explicit buffer releases (`del img`), achieving ~16 ms per frame (~62 FPS) on CPU.
    - Full alignment with Day 2 `metadata.csv` (preserving `extracted_frame_id`, `source_frame_index`, and `timestamp_seconds`).
    - Atomic multi-file Schema v1.0 persistence (`visual_quality_features.json`, `visual_quality_summary.json`, `visual_quality_metadata.json`, `visual_quality_errors.json`).
    - 42 automated unit and integration tests (190 feature tests, 781 repository tests passing, 100% pass rate).
  - [x] **Task 5: Temporal Feature Aggregation**
    - Unified multi-source feature alignment across detection (Task 2), motion (Task 3), and visual quality (Task 4) using `dataset_id` and canonical `frame_id`.
    - Fused 27 base numerical frame metrics into synchronized temporal vectors with granular source and numerical coverage tracking.
    - Configurable past-looking trailing windows (`w3`, `w5`, `w10`) with mathematical guarantee of **zero future-frame leakage**.
    - Numerical rolling statistics: `mean`, `min`, `max`, `std` (population), `median`, `first`, `last`, `diff` with explicit null observation policy (no silent zero infill).
    - Ordinary Least Squares (OLS) linear trend slopes for confidence, blur, brightness, contrast, displacement, and speed metrics.
    - Atomic Schema v1.0 persistence: `temporal_features.json`, `aggregated_frame_features.json`, `temporal_feature_summary.json`, `feature_metadata.json`, and `feature_errors.json`.
    - 45 automated unit and integration tests (235 feature tests, 826 repository tests passing, 100% pass rate).
  - [x] **Task 6: Temporal Sequence Construction**
    - Built production `TemporalSequenceBuilder` transforming multi-modal chronological frames into model-ready sliding-window sequences ($L \in \{5, 10, 15, 20\}$).
    - Multi-horizon forecasting alignment ($H \in \{1, 3, 5\}$ frames ahead) with strict target identification and frame gap calculation.
    - Zero future-frame leakage invariant: input sequences strictly evaluate historical frames without future contamination.
    - Purged chronological dataset partitioning: enforced temporal embargo gaps ($\text{gap} \ge \max(L) + \max(H) = 25$ frames) between Train, Val, and Test to eliminate overlapping window leakage from the same video.
    - Training-only imputation (mean, median) and scaling (standardization, min-max): parameters strictly fitted on training split data only.
    - Honest target label policy: preserves `target_label_available: false` until ground-truth annotations are ingested in Day 6 (no confidence score fabrication).
    - Atomic persistence: emitted `.npy` sequence tensors (`sequences_L*.npy`), `sequence_metadata.json`, `feature_manifest.json`, `dataset_splits.json`, `scaler_parameters.json`, `imputation_parameters.json`, `sequence_summary.json`, and `sequence_errors.json`.
    - 52 automated unit and integration tests (287 feature tests, 878 repository tests passing, 100% pass rate).
  - [x] **Task 7: Feature Pipeline Validation & Final Audit**
    - Comprehensive 7-task engineering audit documented in `docs/FINAL_VALIDATION_REPORT.md`.
    - End-to-end pipeline execution and verification across all 6 feature engineering stages.
    - Verified independent pixel calculations (blur, brightness, contrast) and motion kinematics against raw artifacts.
    - Certified data leakage prevention: zero future-frame leakage, purged temporal embargo gaps ($G=25$), disjoint split frame sets, and training-only scaler/imputer fitting.
    - 100% test pass rate: **878 / 878 tests passed** across entire repository (287 feature tests).
    - Status: **PASS (100% Verified & Production-Ready)**.
- [ ] **Day 6: Predictive Perception Reliability Forecasting**
  - [x] **Task 1: Advanced Forecasting Environment & Model Setup**
    - Established strongly typed, CPU-first `ForecastingConfig` and `config/forecasting_config.yaml` supporting multi-horizon configuration ($H \in \{1, 3, 5\}$).
    - Runtime verification engine (`ForecastingEnvironmentVerifier`) enforcing CPU-only execution, bounded CPU threads (`torch.set_num_threads(2)`), deterministic seed initialization (42), and checkpoint directory readiness.
    - Deep learning data contract validation engine (`ForecastingDataContract`) auditing Day 5 sequence outputs across 14 strict validation checks (3D tensor dimensionality, float32 dtypes, finite numbers, 27-feature manifest alignment, split disjointness, and chronological monotonicity).
    - Model input spec defined as `(batch_size, sequence_length, feature_count)` float32; output spec defined as `(batch_size, number_of_horizons)` float32 with continuous reliability score in $[0.0, 1.0]$.
    - Honest target label policy: reported `ground_truth_labels_available: false` and `supervised_readiness: "NOT AVAILABLE"` because YOLO confidence alone is not ground-truth reliability.
    - 47 automated unit and integration tests across forecasting modules (100% pass rate).
  - [x] **Task 2: Advanced Baseline Forecasting Model Development**
    - Implemented reusable `BaselineForecaster` supporting 3 reference strategies: Persistence ($\hat{y}_{t+h} = y_t$), Historical Mean ($\hat{y}_{t+h} = \bar{y}_{\text{train}}$ fitted on training split only), and Linear Trend Extrapolation with OLS slope and $[0.0, 1.0]$ target clipping.
    - Implemented `BaselineDataLoader` ingesting Day 5 sequence metadata and tensors across split partitions (train: 101, val: 18, test: 19) without video reprocessing or future-frame leakage.
    - Implemented multi-horizon evaluation metrics (`calculate_mae`, `calculate_rmse`, `evaluate_predictions_by_horizon`) per horizon ($H \in \{1, 3, 5\}$) and per strategy.
    - Built reproducible `BaselineExperiment` orchestrator emitting standardized experiment run artifacts (`experiment_config.yaml`, `predictions.json`, `baseline_metrics.json`, `experiment_summary.json`, `metadata.json`, `errors.json`).
    - Enforced scientific target readiness gate: documented `targets_available: false` and `supervised_readiness: "NOT AVAILABLE"` on real Day 5 driving sequences (no YOLO confidence fabrication).
    - 41 automated unit and integration tests across baseline forecasting modules (88 total forecasting tests, 100% pass rate).
  - [x] **Task 3: Advanced LSTM Architecture Development**
    - Implemented modular, CPU-optimized `LSTMForecaster` recurrent neural network processing historical sequences `(batch_size, sequence_length, feature_count)` to predict multi-horizon reliability scores `(batch_size, 3)`.
    - Enforced strict unidirectional temporal encoding (zero future lookahead), Xavier/orthogonal parameter initialization, and final representation forecasting head with bounded $[0.0, 1.0]$ sigmoid activation.
    - Preserved explicit column-to-horizon mappings: Column 0 $\rightarrow h=1$, Column 1 $\rightarrow h=3$, Column 2 $\rightarrow h=5$.
    - Implemented reusable `build_forecasting_model` factory in `model_factory.py` with dynamic input dimension resolution from `feature_manifest.json` ($F=27$).
    - Decoupled forecasting loss factory (`create_forecasting_loss`) supporting MSE and MAE with input validation.
    - Benchmarked CPU execution: **0.587 ms** forward latency (~13,620 samples/sec throughput, 7,907 parameters, 30.89 KB parameter memory) on Intel Core i3 CPU.
    - Verified compatibility with actual Day 5 sequence tensors ($L \in \{5, 10, 15, 20\}$).
    - 40 automated unit and integration tests across LSTM modules (128 total forecasting tests, 100% pass rate).
  - [x] **Task 4: Advanced LSTM Model Training Pipeline Development**
    - Implemented `ForecastingDataset` and `ForecastingDataModule` enforcing strict dataset split integrity (train/val/test disjointness, no sequence ID overlap).
    - Enforced Supervised Target Readiness Gate: documented `supervised_readiness: "NOT AVAILABLE"` on real Day 5 driving sequences because YOLO confidence scores are detection probabilities, not ground truth future reliability labels. Halts training loop before Epoch 1 on unlabelled data.
    - Implemented `EarlyStopping` with configurable patience, minimum delta, validation loss tracking, and automatic best weight restoration.
    - Implemented `CheckpointManager` providing atomic persistence (`checkpoint_last.pt`, `checkpoint_best.pt`), strict architectural dimension validation, and resume capabilities.
    - Implemented structured `TrainingLogger` persisting `training_history.json`, `training_summary.json`, `experiment_config.yaml`, and `metadata.json`.
    - Built `LSTMTrainingEngine` coordinating AdamW optimization, gradient clipping, ReduceLROnPlateau scheduling, and safe non-finite loss detection.
    - Created `train_lstm.py` CLI supporting `--config`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--resume`, and `--verbose`.
    - 53 automated unit and integration tests across training modules (181 total forecasting tests, 100% pass rate).
  - [x] **Task 5: Multi-Horizon Forecasting Engine**
    - Implemented `MultiHorizonForecaster` recurrent neural network inference engine generating multi-horizon perception reliability predictions across future frames ($h \in \{1, 3, 5\}$).
    - Enforced exact temporal frame alignment with 3-tier fallback resolution (`metadata_exact`, `frame_index_offset`, `sequential_extrapolation`), preserving source frame indices and timestamps.
    - Designed versioned 16-field prediction schema and multi-file artifact persistence (`predictions.json`, `horizon_1_predictions.json`, `horizon_3_predictions.json`, `horizon_5_predictions.json`, `prediction_summary.json`, `model_metadata.json`, `feature_manifest.json`, `inference_metadata.json`).
    - CPU inference optimized via `torch.inference_mode()` with latency of **0.054 ms** per sample (> 18,400 predictions/sec throughput) on Intel Core i3 CPU.
    - Implemented CLI runner `python -m src.forecasting.multi_horizon_forecaster` supporting `--input`, `--checkpoint`, `--config`, `--horizons`, `--batch-size`, and `--device`.
    - Maintained scientific integrity: reported `trained_checkpoint: "NOT AVAILABLE"` on real Day 5 driving sequences until ground-truth annotations are provided.
    - 45 automated unit and integration tests (226 total forecasting tests, 100% pass rate).
  - [x] **Task 6: Model Evaluation & Analysis**
    - Implemented production-grade `ForecastingModelEvaluator` evaluating multi-horizon perception reliability predictions across horizons $h \in \{1, 3, 5\}$.
    - Target Integrity Safeguard: strictly rejects YOLO confidence or heuristic indicators as ground-truth reliability; reports `NOT AVAILABLE` cleanly on unannotated Day 5 sequences without fabricating metrics.
    - Strict temporal alignment on composite key `(dataset_id, video_id, target_frame_index, horizon)` with test split isolation and structured exclusion tracking (`excluded_records.json`).
    - Standardized regression metrics computed per horizon: MAE, RMSE, Bias, Median AE, Error Std, MAPE, and directional counts (underprediction vs. overprediction vs. exact).
    - Comparative baseline benchmarking against Persistence, Historical Mean, and Linear Trend with relative percentage improvement calculations (`baseline_comparison.json`).
    - Statistical error analysis: full error distribution percentiles (p5, p25, p50, p75, p95), horizon breakdowns, and top worst error cases (`error_analysis.json`).
    - Diagnostic visualization suite (`ForecastingEvaluationVisualizer`) generating 6 publication-grade charts via headless Matplotlib `Agg` backend (`actual_vs_predicted_reliability.png`, `mae_by_horizon.png`, `rmse_by_horizon.png`, `prediction_error_distribution.png`, `baseline_vs_lstm_comparison.png`, `residual_analysis.png`).
    - CLI runner `python -m src.forecasting.model_evaluator` supporting `--predictions`, `--ground-truth`, `--baseline-results`, `--split`, `--horizons`, `--output`, and `--generate-plots`.
    - 51 automated unit and integration tests (277 total forecasting tests, 1,169 repository-wide tests passing, 100% pass rate).
  - [ ] **Task 7: Final Forecasting Audit & Pipeline Certification**
- [ ] **Day 7: Live Inference & Perception Telemetry Pipeline**
- [ ] **Day 8: Evaluation & Streamlit Dashboard**
  - Model evaluation against test sequences.
  - Interactive dashboard with live visual alerts and reliability telemetry.

---

## 8. Research Limitations & Future Work

1. **Camera-Only Modality:** Current implementation targets monocular camera streams. Future extensions will incorporate LiDAR point clouds and Radar range-rate features for multi-modal perception reliability.
2. **Offline Ground-Truth Supervision:** Constructing true future reliability targets during training requires ground-truth 2D/3D annotations (e.g., from BDD100K or Waymo Open Dataset).
3. **Severe Sensor Occlusion:** If an object is completely occluded for extended durations (> 30 frames), tracking identity switches can degrade historical feature continuity.
4. **Edge Compute Budgets:** Real-time inference at 30 FPS on embedded edge platforms (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi 5) requires lightweight recurrent models (GRU) with int8 quantization.

---

## 9. License

This project is licensed under the MIT License - see the LICENSE file for details.

