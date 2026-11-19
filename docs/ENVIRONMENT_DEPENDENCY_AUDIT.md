# PerceptGuard-AI: Environment & Dependency Audit

**Audit Date:** 2026-11-19  
**Auditor Roles:** Senior Software Engineer, Senior Machine Learning Engineer, ML Infrastructure Architect, and Python Environment Specialist  
**Execution Environment:** Intel Core i3 (CPU Only), 8 GB RAM, Windows 11  
**Overall Environment Status:** **`PASS`**  

---

## 1. Executive Summary

This document presents the comprehensive environment, dependency, and hardware compatibility audit for **PerceptGuard-AI**. The audit evaluates the reproducibility, stability, and runtime constraints of the computer vision and predictive reliability forecasting pipeline on an Intel Core i3 CPU-only development environment running Windows 11 with 8 GB of physical RAM.

All required frameworks (PyTorch CPU, Ultralytics YOLO11n, OpenCV, NumPy, SciPy, Pandas, Scikit-learn, Matplotlib) were empirically tested for inter-library interoperability, memory limits, and CPU computational stability. Two missing direct dependency declarations (`scipy` and `psutil`) were identified and formally added to both [`requirements.txt`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/requirements.txt) and [`pyproject.toml`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/pyproject.toml). An automated 10-check environment smoke test suite was added to [`tests/test_environment_smoke.py`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/test_environment_smoke.py), achieving a 100% pass rate.

---

## 2. Hardware Specifications & Operating System

| Parameter | System Value | Audit Assessment |
| :--- | :--- | :--- |
| **CPU Architecture** | Intel64 Family 6 Model 190 Stepping 0 (Intel Core i3 / AMD64) | 8 Logical Processors / Cores |
| **Total Physical RAM** | 7.75 GB (8,322,465,792 bytes) | Verified via `GlobalMemoryStatusEx` |
| **Available RAM** | ~0.88 GB – 1.20 GB (Memory load ~88%) | Constrained; requires strict batch sizing |
| **GPU / Hardware Acceleration** | None (CUDA Available: `False`) | CPU-only execution strictly enforced |
| **Operating System** | Microsoft Windows 11 Build 26200 | 64-bit WindowsPE |
| **Threading Policy** | `torch.set_num_threads(2)`, `num_workers=0` | Prevents CPU starvation and Windows IPC deadlocks |

---

## 3. Python Runtime & Virtual Environment

| Component | Target / Required | Active Detected Value | Status |
| :--- | :--- | :--- | :---: |
| **Python Version** | `>=3.10` | `3.13.3 (64-bit AMD64)` | **PASS** |
| **Python Executable** | Local Virtualenv | `C:\Users\acer\Desktop\PerceptGuard-AI\venv\Scripts\python.exe` | **PASS** |
| **Package Installer (pip)** | `>=23.0` | `25.0.1` | **PASS** |
| **Installed Packages** | Production & Dev | `71 distributions` in virtual environment | **PASS** |
| **Interpreter Isolation** | Isolated venv | System Python (broken PyTorch DLL) successfully isolated | **PASS** |

---

## 4. Dependency Classification & Inventory

Dependencies declared in [`requirements.txt`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/requirements.txt) and [`pyproject.toml`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/pyproject.toml) are classified across 8 functional categories:

| Category | Package | Installed Version | Declared Specifier | Purpose / Role in Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **1. Core Runtime** | `python` | 3.13.3 | `>=3.10` | Core language runtime |
| | `python-dotenv` | 1.2.4 | `>=1.0.0` | 12-factor environment configuration |
| | `pyyaml` | 6.0.3 | `>=6.0.0` | Configuration parser for YAML schemas |
| | `psutil` | 7.2.2 | `>=5.9.0` | Hardware and memory resource telemetry |
| **2. Computer Vision** | `opencv-python` | 5.0.0.93 | `>=4.8.0` | Video frame decoding, image I/O, resizing |
| | `ultralytics` | 8.4.215 | `>=8.1.0` | YOLOv8/11 object detection and tracking |
| | `torchvision` | 0.29.2+cpu | `>=0.17.0` | Vision tensor transforms and NMS ops |
| **3. Machine Learning** | `scikit-learn` | 1.9.1 | `>=1.3.0` | Baseline regressors, scaling, metric calculations |
| | `scipy` | 1.18.1 | `>=1.10.0` | Hungarian assignment algorithm (`linear_sum_assignment`) |
| **4. Deep Learning** | `torch` | 2.15.0+cpu | `>=2.2.0` | CPU tensor engine, LSTM neural network |
| **5. Data Processing** | `numpy` | 2.5.4 | `>=1.24.0` | High-performance N-dimensional array processing |
| | `pandas` | 3.1.1 | `>=2.0.0` | Structured DataFrame feature aggregations |
| **6. Visualization** | `matplotlib` | 3.11.2 | `>=3.7.0` | Headless metric plots, calibration diagrams |
| | `streamlit` | 1.67.0 | `>=1.30.0` | Interactive monitoring dashboard (Day 7) |
| **7. Testing** | `pytest` | 9.2.0 | `>=8.0.0` | Automated regression test framework |
| **8. Development Tools**| `setuptools` | 78.1.0 | `>=61.0` | Package metadata and editable builds |

---

## 5. ML Framework Interoperability & Compatibility Findings

### PyTorch CPU Validation
- CPU tensor allocation, mathematical matrix multiplication, and linear layer forward passes operate without errors.
- Confirmed `torch.cuda.is_available() == False`.
- Thread concurrency bounded via `torch.set_num_threads(2)` and `torch.set_num_interop_threads(1)` to prevent CPU thrashing.

### Ultralytics YOLO Validation
- Weights file verified at [`models/yolo/yolo11n.pt`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/models/yolo/yolo11n.pt) (6.4 MB).
- Successfully loads onto CPU using [`src/detection/single_image_detector.py`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/src/detection/single_image_detector.py).
- CPU forward inference on a 640x640 frame completes cleanly; zero detections on blank inputs are correctly handled with `detection_count: 0`.

### OpenCV Validation
- Image I/O read/write (`cv2.imread`, `cv2.imwrite`), color conversions (`COLOR_BGR2GRAY`), and Gaussian filtering execute without issues.
- Direct pointer-free memory bridge between OpenCV NumPy arrays (`uint8`) and PyTorch tensors verified.

### NumPy & Zero-Copy Bridge
- Bidirectional zero-copy conversions (`torch.from_numpy()` and `tensor.numpy()`) operate cleanly on contiguous float32 and int64 buffers.

### LSTM Forecasting Architecture
- Model instantiated via [`src/forecasting/model_factory.py`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/src/forecasting/model_factory.py).
- Input dimension resolved to $F=27$ features; sequence length $L=10$; output horizons $H \in \{1, 3, 5\}$.
- Forward pass tensor shapes verified: $(B, 10, 27) \to (B, 3)$ on CPU device.

---

## 6. Model Artifacts & Scientific Honesty Assessment

| Artifact | Location | Availability | Scientific Status |
| :--- | :--- | :---: | :--- |
| **YOLO11 Nano Weights** | `models/yolo/yolo11n.pt` | **PASS** | Valid pre-trained MS COCO weights for bounding-box detection. |
| **Supervised LSTM Checkpoints** | `models/forecasting/` | **NOT AVAILABLE** | Supervised model training is **`NOT AVAILABLE`** because the 200-frame BDD100K driving sample lacks ground-truth reliability labels. YOLO confidence is **never** treated as ground-truth reliability. |
| **Baseline Heuristic Weights** | In-memory / Code | **PASS** | Simple exponential moving average & persistence baselines operate deterministically without checkpoint files. |

---

## 7. Safe Environment Repairs Implemented

1. **Declared Direct Dependency `scipy`:**
   - Problem: `src/tracking/bytetrack_engine.py` and `src/features/feature_environment.py` import `scipy.optimize.linear_sum_assignment` directly, but `scipy` was omitted from `requirements.txt` and `pyproject.toml`.
   - Resolution: Added `scipy>=1.10.0` to [`requirements.txt`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/requirements.txt) and [`pyproject.toml`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/pyproject.toml).
2. **Declared Direct Dependency `psutil`:**
   - Problem: Hardware environment checkers in `src/features/feature_environment.py` and `src/tracking/tracking_environment.py` import `psutil` for CPU/RAM telemetry, but `psutil` was unlisted.
   - Resolution: Added `psutil>=5.9.0` to [`requirements.txt`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/requirements.txt) and [`pyproject.toml`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/pyproject.toml).
3. **Enhanced Environment Verification Script:**
   - Updated [`tests/verify_environment.py`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/verify_environment.py) to check 11 core dependencies (including SciPy, psutil, and PyYAML).
4. **Created Automated Smoke Test Suite:**
   - Created [`tests/test_environment_smoke.py`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/test_environment_smoke.py) covering all 10 required environment smoke checks.

---

## 8. Remaining Limitations & Edge Constraints

1. **CPU Inference Latency:** YOLO11 Nano inference on CPU requires ~4.8 seconds for initial cold-load and ~100–300 ms per frame on downscaled inputs. Real-time 30 FPS inference is not feasible without hardware neural accelerators (NPU/GPU) or lower input resolutions (e.g. 320x320).
2. **Constrained RAM Headroom:** Available physical memory is ~0.88 GB out of 7.75 GB total. Background processes must remain minimal, and DataLoader `batch_size` must not exceed 16–32 items.
3. **DataLoader Worker Limits:** Multiprocessing DataLoaders (`num_workers > 0`) must remain disabled on Windows to prevent IPC overhead and deadlocks.

---

## 9. Final Verdict

The environment and dependency audit demonstrates that the PerceptGuard-AI runtime is **reproducible, compatible, stable, and strictly bounded for CPU-only execution**.

**Final Verdict:** **`PASS`**
