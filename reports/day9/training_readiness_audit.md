# PerceptGuard AI — Day 9 Task 1: Training Readiness & Experiment Planning Audit Report

**Date:** 2026-11-20  
**Phase:** Day 9 Task 1  
**Audit Scope:** Training Readiness Audit, Dataset Verification, Feature-Target Compatibility, Subsystem Validation, and Reproducible Experiment Planning  
**Auditor Roles:** Principal Machine Learning Engineer, Senior Staff AI Engineer, Deep Learning Architect, ML Experimentation Lead  
**Repository:** `https://github.com/Jeeva1117/PerceptGuard-AI.git`  
**Branch:** `main`  
**Execution Environment:** Intel Core i3 (CPU-Only, 8 Logical Cores), 8 GB RAM, Windows 11 Build 26200, Python 3.13.3 AMD64, PyTorch 2.15.0+cpu  
**Final Training Readiness Determination:** **`READY WITH LIMITATIONS` (Decision Tier B)**

---

## 1. Executive Summary

This engineering audit constitutes the official training readiness review and scientific experimentation roadmap for the first supervised deep learning and baseline forecasting experiments in **PerceptGuard AI**.

Operating as an edge-deployable predictive computer vision framework under strict CPU and memory boundaries (Intel Core i3, 8 GB RAM, Windows 11, CPU-only), PerceptGuard AI forecasts future perception reliability across upcoming video frames ($H \in \{1, 3, 5\}$).

Following an exhaustive audit of all pre-existing software packages, configuration files, dataset artifacts, mathematical count balances, and pipeline integration tests, the repository is certified **`READY WITH LIMITATIONS` (Decision Tier B)**. All prerequisites for supervised learning are satisfied, zero model training was conducted in this audit task, and a 5-stage experiment plan (Experiments A through E) is formally documented for execution.

---

## 2. Day 8 Final Audit Review

- **Inspected Reports:**
  - `docs/DAY_8_FINAL_MASTER_AUDIT.md`
  - `reports/final_audit/day8_master_audit.md`
  - `reports/final_audit/day8_master_audit.json`
  - `docs/DAY_8_FINAL_DATASET_AUDIT.md`
- **Recorded Decision:** **`READY WITH LIMITATIONS` (Decision Tier B)**
- **Audit Findings:**
  - Subsystems 1 through 14 certified `PASS` (216/216 automated checks passing).
  - Mathematical count conservation certified exact ($\Delta = 0$).
  - 12-stage pipeline smoke test executed in $7.60\text{ s}$ with peak RSS of $496.57\text{ MB}$.
- **Documented Limitations:**
  - Standard unweighted MSE will collapse to trivial zero predictions due to the $90^\circ$ rotated video container causing YOLO detector failure ($99.5\%$ zero-$F_1$ frames).
  - Downstream training must incorporate loss reweighting (Asymmetric Weighted Huber Loss / Focal Loss) to penalize degradation transitions.
  - No claim of nominal upright highway reliability may be made without incorporating additional upright sequences.

---

## 3. Actual Dataset Availability Audit

All physical files referenced in dataset manifests were checked directly on the filesystem:

| Artifact Name | Physical File Location | Item Count | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| Video Stream | `datasets/raw/converted_026c7465-309f6d33.mp4` | 1 file (40.01s) | `PASS` | 1,204 source frames, 30.09 FPS |
| Preprocessed Frames | `datasets/processed/driving_sample/preprocessed_frames/` | 200 JPEG frames | `PASS` | $640 \times 360$ letterbox images |
| Canonical Annotations | `datasets/processed/driving_sample/annotations/annotations_canonical.jsonl` | 5,121 boxes | `PASS` | 115 tracks, 6 classes, 0 invalid coords |
| YOLO Detections | `datasets/processed/driving_sample/detections/batch_runs/...` | 211 detections | `PASS` | Confidence $\ge 0.25$, FP32 CPU |
| Object Labels | `datasets/processed/driving_sample/reliability_labels/object_detection_labels.json` | 5,331 records | `PASS` | $TP=1, FP=210, FN=5,120$ |
| Frame Labels | `datasets/processed/driving_sample/reliability_labels/frame_reliability_labels.json` | 200 frame records | `PASS` | Frame precision, recall, $F_1 \in [0, 1]$ |
| Temporal Targets | `datasets/processed/driving_sample/reliability_labels/temporal_reliability_targets.json` | 462 sequences | `PASS` | H1: 458, H3: 450, H5: 443 |
| Labeled Metadata | `datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json` | 462 sequences | `PASS` | Split allocations & horizon metadata |
| Sequence Tensor L=5 | `datasets/processed/driving_sample/features/sequences/run_official/sequences_L5.npy` | 138 tensors | `PASS` | Shape: $(138, 5, 27)$, float32 |
| Sequence Tensor L=10| `datasets/processed/driving_sample/features/sequences/run_official/sequences_L10.npy`| 123 tensors | `PASS` | Shape: $(123, 10, 27)$, float32 |
| Sequence Tensor L=15| `datasets/processed/driving_sample/features/sequences/run_official/sequences_L15.npy`| 108 tensors | `PASS` | Shape: $(108, 15, 27)$, float32 |
| Sequence Tensor L=20| `datasets/processed/driving_sample/features/sequences/run_official/sequences_L20.npy`| 93 tensors | `PASS` | Shape: $(93, 20, 27)$, float32 |
| Dataset Splits | `datasets/processed/driving_sample/features/sequences/run_official/dataset_splits.json` | 3 partitions | `PASS` | Purged split, $G=25$ embargo gap |
| Feature Manifest | `datasets/processed/driving_sample/features/sequences/run_official/feature_manifest.json` | 27 features | `PASS` | Detection, motion, visual quality |

- **Missing Files:** 0
- **Corrupted Records:** 0
- **Duplicate Records:** 0

---

## 4. Feature and Target Compatibility Audit

### 4.1 Input Features
- **Total Features:** 27 continuous features
- **Categories:** Detection (6), Motion (8), Visual Quality (13)
- **Feature Ordering:** Exactly aligns with `ordered_feature_names` in `feature_manifest.json`
- **Data Types:** Float32
- **Missing Value Strategy:** Train-only median imputation

### 4.2 Supervisory Targets
- **Target Horizons:** $H1 = 1$, $H3 = 3$, $H5 = 5$ frames ahead
- **Target Semantics:** Hungarian IoU-matched ground-truth $F_1$ perception reliability
- **Target Dimensions:** Target vector shape `(N, 3)`
- **Target Bounds:** Continuous values in $[0.0, 1.0]$

### 4.3 Dimension Alignment & Zero Leakage
- Input shape $(N, L, 27)$ maps to target shape $(N, 3)$ without tensor reshaping tricks.
- Zero future targets leak into input features; temporal separation strictly preserved.

---

## 5. Train / Validation / Test Split Integrity

- **Split Scheme:** Chronological sequence-aware purged partitioning
- **Embargo Gap:** 25 frames ($G=25$) between partitions
- **Sample Distribution for Sequence Length $L=5$:**
  - **Train Split:** 101 sequences (Frames 1–105)
  - **Embargo Gap 1:** 25 frames (Frames 106–130)
  - **Validation Split:** 18 sequences (Frames 131–152)
  - **Embargo Gap 2:** 25 frames (Frames 153–177)
  - **Test Split:** 19 sequences (Frames 178–200)
- **Leakage Status:** Verified **ZERO** frame, window, or target overlap across splits.

---

## 6. Baseline Model Readiness

- **Module:** `src/forecasting/baseline_forecaster.py`
- **Supported Strategies:**
  1. *Persistence Baseline:* $\hat{y}_{t+h} = y_t$
  2. *Historical Mean Baseline:* $\hat{y}_{t+h} = \bar{y}_{\text{train}} = 0.000582$
  3. *Linear Trend Baseline:* $\hat{y}_{t+h} = \text{clip}(y_t + m \cdot h, 0, 1)$
- **Evaluation Execution:** 1,242 prediction records generated deterministically across all splits and horizons.
- **Contract Adherence:** Outputs conform strictly to `PredictionRecord` schema.

---

## 7. LSTM Training Pipeline Readiness

- **Model Class:** `LSTMForecaster` in `src/forecasting/lstm_forecaster.py`
- **Dynamic Construction:** `build_forecasting_model` automatically resolves $F=27$ from manifest.
- **Architecture Details:**
  - Input dimension: 27
  - Hidden dimension: 32 (1 layer, default)
  - Dropout: 0.0
  - Output horizons: [1, 3, 5]
  - Parameter count: 7,907 trainable parameters ($<32\text{ KB}$)
  - Output activation: Sigmoid (guarantees $[0.0, 1.0]$ bounds)
- **Training Engine:** `LSTMTrainingEngine` (`src/forecasting/training_engine.py`)
  - AdamW optimizer, ReduceLROnPlateau scheduler, gradient clipping (norm 1.0), early stopping (patience 8), atomic checkpoint serialization.
  - Supports `device: cpu` exclusively.

---

## 8. CPU Resource & Feasibility Analysis

- **Hardware Platform:** Intel Core i3, 8 GB RAM, Windows 11 AMD64
- **CPU Utilization:** Average forward latency $<5.0\text{ ms}$; epoch training latency $\approx 40\text{ ms}$
- **Memory Footprint:** Peak RSS $<500\text{ MB}$ ($<6.2\%$ of 8 GB RAM budget)
- **Batch Size:** 8 sequences (13 training batches per epoch)
- **Feasibility Verdict:** **FULLY FEASIBLE FOR CPU EXECUTION**

---

## 9. Safe Corrections Applied

1. **CLI Metadata & Dry-Run Support (`src/forecasting/train_lstm.py`):**
   - Added `--metadata-file` argument allowing explicit specification of labeled sequence metadata.
   - Added `--dry-run` flag to verify data loading, target readiness, and model construction without executing training epochs.
2. **Config Schema Enhancement (`src/forecasting/forecasting_config.py`):**
   - Added `metadata_file: Optional[str] = None` to `DataConfig` dataclass and validation rules.
3. **Automated Regression Testing (`tests/forecasting/test_train_lstm_cli.py`):**
   - Added `test_cli_labeled_data_dry_run` confirming successful dry-run completion on real labeled metadata.

---

## 10. Test Results Summary

Executed via virtual environment (`.\venv\Scripts\python.exe`):
- `tests/forecasting/test_train_lstm_cli.py`: **6 / 6 PASSED**
- `tests/test_audit_project.py`: **8 / 8 PASSED**
- Total tests executed across modified components: **14 / 14 PASSED** (0 failures, 0 regressions).

---

## 11. Day 9 Experiment Plan Summary

- **Experiment A — Baseline Benchmark:** Compute empirical baselines (Persistence, Historical Mean, Linear Trend) across H1, H3, H5 on Validation and Test partitions.
- **Experiment B — Standard LSTM Training:** Train initial multi-horizon LSTM using standard MSE loss (30 epochs, batch size 8, lr 0.001) to observe baseline convergence.
- **Experiment C — Training Optimization (Loss Reweighting):** Implement Asymmetric Weighted Huber Loss ($w_{\text{non-zero}} = 50.0$) to overcome the $99.5\%$ zero-$F_1$ detector skew.
- **Experiment D — Multi-Horizon Forecasting Analysis:** Evaluate horizon error curves ($H1 \le H3 \le H5$), confidence calibration, and transition anticipation lead-time.
- **Experiment E — Final Benchmark & Comparative Evaluation:** Produce final evaluation matrix comparing baselines and trained models on isolated test data using MAE, RMSE, and degradation capture rates.

---

## 12. Final Determination

### Final Training Readiness Determination:
# **`READY WITH LIMITATIONS` (Decision Tier B)**

### Certification:
The PerceptGuard AI software and data pipeline is fully verified, operational on CPU hardware, and certified ready for Day 9 supervised forecasting experiments. No model training was executed in this task.
