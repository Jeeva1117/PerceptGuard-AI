# PerceptGuard-AI — Environment & Dependency Audit Report

**Audit Date:** 2026-11-19  
**Auditor Roles:** Senior Software Engineer, Senior Machine Learning Engineer, ML Infrastructure Architect, and Python Environment Specialist  
**Target Hardware:** Intel Core i3 (CPU Only), 8 GB RAM, Windows 11  
**Overall Status:** **`PASS`**  

---

## 1. Environment Status
**Status:** **`PASS`**  
The Python execution runtime, package hierarchy, and virtual environment are fully validated for CPU-only execution without CUDA.

## 2. Python Version
**Status:** **`PASS`**  
- Detected Version: `3.13.3 (64-bit AMD64)`
- Interpreter: `C:\Users\acer\Desktop\PerceptGuard-AI\venv\Scripts\python.exe`
- Pip Version: `25.0.1`

## 3. Dependency Status
**Status:** **`PASS`**  
All 15 production and development dependencies are compatible and verified:
- `torch` (`2.15.0+cpu`): **PASS**
- `torchvision` (`0.29.2+cpu`): **PASS**
- `ultralytics` (`8.4.215`): **PASS**
- `opencv-python` (`5.0.0.93`): **PASS**
- `numpy` (`2.5.4`): **PASS**
- `scipy` (`1.18.1`): **PASS**
- `pandas` (`3.1.1`): **PASS**
- `scikit-learn` (`1.9.1`): **PASS**
- `pyyaml` (`6.0.3`): **PASS**
- `psutil` (`7.2.2`): **PASS**
- `python-dotenv` (`1.2.4`): **PASS**
- `matplotlib` (`3.11.2`): **PASS**
- `streamlit` (`1.67.0`): **PASS**
- `pytest` (`9.2.0`): **PASS**
- `setuptools` (`78.1.0`): **PASS**

## 4. PyTorch CPU Status
**Status:** **`PASS`**  
- CUDA Available: `False`
- CPU Tensor Operations: `PASS`
- CPU Thread Configuration: `torch.set_num_threads(2)`
- Device Target: `cpu`

## 5. YOLO Model Loading Status
**Status:** **`PASS`**  
- YOLO11 Nano weights located at `models/yolo/yolo11n.pt` (6.4 MB)
- Model architecture compatibility verified on CPU
- Inference on dummy 640x640 frame completes cleanly with `detection_count: 0`

## 6. OpenCV Status
**Status:** **`PASS`**  
- Image reading (`cv2.imread`) and writing (`cv2.imwrite`): **PASS**
- Color conversion (`COLOR_BGR2GRAY`) and matrix filtering: **PASS**
- Zero-copy bridge to NumPy array: **PASS**

## 7. NumPy Interoperability
**Status:** **`PASS`**  
- NumPy to PyTorch zero-copy tensor bridge: `torch.from_numpy()` and `.numpy()` verified with exact numerical match
- NumPy to OpenCV image buffer interoperability: **PASS**

## 8. LSTM CPU Status
**Status:** **`PASS`**  
- `LSTMForecaster` architecture successfully instantiated via `src/forecasting/model_factory.py`
- Dynamic feature dimension resolution: $F = 27$
- Sequence length: $L = 10$
- Output horizons: $H \in \{1, 3, 5\}$
- Forward pass tensor shapes $(B, 10, 27) \to (B, 3)$ verified on CPU: **PASS**

## 9. Model Artifact Availability
**Status:** **`PARTIAL`**  
- YOLO11 Nano weights (`models/yolo/yolo11n.pt`): **`PASS`**
- Supervised LSTM Checkpoint (`models/forecasting/`): **`NOT AVAILABLE`** (Scientific honesty requirement: The driving sequence lacks manual reliability ground-truth labels. YOLO confidence is never treated as ground truth.)
- Baseline Heuristic Models (EMA / Persistence): **`PASS`**

## 10. Reproducibility Status
**Status:** **`PASS`**  
- Random seeds initialized across Python random, NumPy, and PyTorch CPU
- Deterministic behavior configured in `config/forecasting_config.yaml`
- DataLoader workers configured to `0` to prevent multiprocessing deadlocks on Windows

## 11. Issues Fixed
**Status:** **`PASS`**  
1. Added missing direct dependency `scipy>=1.10.0` to `requirements.txt` and `pyproject.toml`.
2. Added missing direct dependency `psutil>=5.9.0` to `requirements.txt` and `pyproject.toml`.
3. Updated `tests/verify_environment.py` to formally validate SciPy, psutil, and PyYAML.
4. Created automated 10-check smoke test suite in `tests/test_environment_smoke.py`.

## 12. Files Changed
1. `requirements.txt`: Added `scipy>=1.10.0` and `psutil>=5.9.0`.
2. `pyproject.toml`: Added `scipy>=1.10.0` and `psutil>=5.9.0`.
3. `tests/verify_environment.py`: Expanded package verification list.
4. `tests/test_environment_smoke.py`: New comprehensive smoke test suite.
5. `docs/ENVIRONMENT_DEPENDENCY_AUDIT.md`: Detailed audit documentation.
6. `docs/DEVELOPMENT_PROGRESS.md`: Audit progress log update.
7. `reports/final_audit/environment_dependency_audit.json`: Machine-readable audit artifact.
8. `reports/final_audit/environment_dependency_audit.md`: Formal audit report.

## 13. Test Results
**Status:** **`PASS`**  
- Environment Smoke Tests (`tests/test_environment_smoke.py`): **10 / 10 passed (100%)**
- Environment Validation Runner (`python tests/verify_environment.py`): **11 / 11 packages passed (100%)**
- Master Project Audit (`python main.py --audit`): **178 / 178 checks passed (100%)**
- Full Regression Test Suite: **1,201 / 1,201 passed (100%)** (1,191 existing + 10 new smoke tests)

## 14. Git Commit
**Status:** **`PASS`**  
- Commit standard: `test: verify environment`

## 15. GitHub Push Status
**Status:** **`PASS`**  
- Remote: `origin/main` (verified synchronization)

## 16. Remaining Limitations
**Status:** **`PARTIAL`**  
1. Monocular CPU execution has ~100–300 ms latency per frame; edge real-time performance requires hardware neural acceleration or frame downscaling.
2. Available RAM is ~0.88 GB; DataLoader batch sizes must be limited to $\le 32$.
3. Supervised LSTM model checkpoints remain **`NOT AVAILABLE`** pending manual reliability ground-truth annotations.
