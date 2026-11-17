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
│   ├── prediction/             # LSTM / GRU reliability forecasting models
│   │   └── __init__.py
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
- [ ] **Day 3: Object Detection & Perception Pipeline**
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
  - [ ] **Task 6: Detection Metrics & Reliability Analysis**
  - [ ] **Task 7: Final Perception Audit & Verification**
- [ ] **Day 4: Feature Extraction & Sequence Construction**
  - Implement 16-dimensional per-track feature vector extraction.
  - Build sliding historical temporal sequence windows ($W=10$).
- [ ] **Day 5: Ground-Truth Reliability Labeling**
  - Hungarian bipartite matching with annotated ground-truth boxes.
  - Construct multi-horizon supervision labels ($t+1, t+3, t+5$).
- [ ] **Day 6: Recurrent Forecasting Model (LSTM/GRU)**
  - Model architecture definition, PyTorch dataset loaders, and training pipeline.
- [ ] **Day 7: Evaluation & Streamlit Dashboard**
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

