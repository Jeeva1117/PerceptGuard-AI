# PerceptGuard AI — Training Readiness & Experiment Planning Specification

**Milestone:** Day 9 Task 1: Advanced Training Readiness & Experiment Planning Audit  
**Date:** 2026-11-20  
**Roles:** Principal Machine Learning Engineer, Senior Staff AI Engineer, Deep Learning Architect, ML Experimentation Lead  
**Repository:** `https://github.com/Jeeva1117/PerceptGuard-AI.git`  
**Branch:** `main`  
**Execution Environment:** Intel Core i3 (CPU-Only, 8 Logical Cores), 8 GB RAM, Windows 11 Build 26200, Python 3.13.3 AMD64, PyTorch 2.15.0+cpu  
**Final Training Readiness Determination:** **`READY WITH LIMITATIONS` (Decision Tier B)**

---

## 1. Executive Summary

This specification establishes the definitive training readiness verification and reproducible experiment roadmap for **Day 9 Supervised Forecasting Experiments** in PerceptGuard AI.

PerceptGuard AI operates under strict edge-deployment constraints (Intel Core i3, 8 GB RAM, Windows 11, CPU-only). Its operational mission is **Anticipating Detection Failures Before They Happen** by predicting future object detection reliability ($H \in \{1, 3, 5\}$ frames into the future) using temporal sliding windows of feature vectors extracted from YOLO detections, ByteTrack tracking trajectories, and visual scene degradation metrics.

Following an independent audit of the complete Day 1–Day 8 codebase, dataset artifacts, mathematical invariants, split boundaries, and model implementations, the system has been certified ready for supervised forecasting experiments under the documented limitations established in Day 8.

### Core Audit Findings:
1. **Day 8 Final Master Audit Review:** Confirmed final readiness decision **`READY WITH LIMITATIONS` (Decision Tier B)** in `docs/DAY_8_FINAL_MASTER_AUDIT.md` and `reports/final_audit/day8_master_audit.json`.
2. **Actual Dataset Availability:** 200 preprocessed driving frames ($640 \times 360$), 5,121 canonical ground-truth bounding boxes, 211 YOLO detections, 5,331 object reliability records, 200 frame reliability records, and 462 temporal sequences are physically verified on disk with zero file fabrication.
3. **Feature & Target Compatibility:** 27 multimodal input features dynamically match model expectations; 3 continuous target horizons ($H \in \{1, 3, 5\}$) reside strictly within $[0.0, 1.0]$.
4. **Leakage-Safe Partitioning:** Chronological split enforces a 25-frame embargo gap ($G=25$), guaranteeing zero temporal feature or sequence leakage across Train (101 seqs at L=5), Val (18 seqs), and Test (19 seqs).
5. **CPU Feasibility:** `LSTMForecaster` architecture consists of 7,907 trainable parameters ($<32\text{ KB}$ weights footprint), executing with a batch size of 8 in $<5\text{ ms}$ forward latency and $<500\text{ MB}$ peak memory RSS.
6. **Safe Corrections Applied:** Added `--metadata-file` and `--dry-run` options to `src/forecasting/train_lstm.py`, and added `metadata_file` support to `DataConfig` in `src/forecasting/forecasting_config.py`.
7. **No Models Trained:** In strict accordance with engineering governance, zero models were trained during this readiness audit.

---

## 2. Decision Gate & Day 8 Audit Review

The Day 8 final audit reports and machine-readable records were audited:
- [docs/DAY_8_FINAL_MASTER_AUDIT.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/docs/DAY_8_FINAL_MASTER_AUDIT.md)
- [reports/final_audit/day8_master_audit.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/reports/final_audit/day8_master_audit.md)
- [reports/final_audit/day8_master_audit.json](file:///c:/Users/acer/Desktop/PerceptGuard-AI/reports/final_audit/day8_master_audit.json)
- [docs/DAY_8_FINAL_DATASET_AUDIT.md](file:///c:/Users/acer/Desktop/PerceptGuard-AI/docs/DAY_8_FINAL_DATASET_AUDIT.md)

### Verification of Decision:
- **Recorded Decision:** **`READY WITH LIMITATIONS` (Decision Tier B)**
- **Subsystem Breakdown:**
  - Engineering Pipeline: `TRAINING READY`
  - Ground-Truth Annotations: `TRAINING READY`
  - Reliability Labels: `TRAINING READY`
  - Temporal Targets: `TRAINING READY`
  - Train/Validation/Test Splits: `TRAINING READY`
  - Supervised Model Training: `READY WITH LIMITATIONS`
- **Documented Limitations & Protocol:**
  - *Permitted:* Supervised multi-horizon forecasting experiments on the certified labeled sequences.
  - *Mandatory Downstream Mitigation:* Standard unweighted MSE loss will collapse to predicting zero due to the $99.5\%$ zero-$F_1$ detector skew caused by the $90^\circ$ rotated video container in `driving_sample.mp4`. Downstream training must utilize asymmetric weighted loss formulations (Weighted Huber / Focal Loss) or dual-head classification/regression.
  - *Prohibited:* Claiming nominal upright highway reliability without incorporating additional upright driving sequences.

---

## 3. Actual Dataset Availability Audit

An exhaustive physical inspection of actual dataset artifacts on disk was performed:

| Dataset Artifact | Physical Path | Actual Count | Disk Footprint | Integrity Status |
| :--- | :--- | :---: | :---: | :---: |
| **Video Stream** | `datasets/raw/converted_026c7465-309f6d33.mp4` | 1 file (40.01s) | $18.4\text{ MB}$ | `PASS` (1,204 frames) |
| **Preprocessed Frames** | `datasets/processed/driving_sample/preprocessed_frames/` | 200 JPEG images | $\approx 22\text{ MB}$ | `PASS` ($640 \times 360$) |
| **Canonical Annotations** | `datasets/processed/driving_sample/annotations/annotations_canonical.jsonl` | 5,121 boxes | $1.2\text{ MB}$ | `PASS` (115 tracks, 6 classes) |
| **YOLO Detections** | `datasets/processed/driving_sample/detections/batch_runs/...` | 211 detections | $145\text{ KB}$ | `PASS` (FP32 CPU detections) |
| **Object Reliability Labels** | `datasets/processed/driving_sample/reliability_labels/object_detection_labels.json` | 5,331 records | $2.8\text{ MB}$ | `PASS` ($TP=1, FP=210, FN=5120$) |
| **Frame Reliability Labels** | `datasets/processed/driving_sample/reliability_labels/frame_reliability_labels.json` | 200 frame records | $98\text{ KB}$ | `PASS` (All $F_1 \in [0.0, 1.0]$) |
| **Temporal Sequence Targets** | `datasets/processed/driving_sample/reliability_labels/temporal_reliability_targets.json` | 462 sequences | $310\text{ KB}$ | `PASS` (H1: 458, H3: 450, H5: 443) |
| **Labeled Sequence Metadata** | `datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json` | 462 sequences | $485\text{ KB}$ | `PASS` (Train: 374, Val: 42, Test: 46) |
| **Sequence Tensors (L=5)** | `datasets/processed/driving_sample/features/sequences/run_official/sequences_L5.npy` | 138 tensors | $74.5\text{ KB}$ | `PASS` (Shape: $138 \times 5 \times 27$) |
| **Sequence Tensors (L=10)**| `datasets/processed/driving_sample/features/sequences/run_official/sequences_L10.npy`| 123 tensors | $132.8\text{ KB}$ | `PASS` (Shape: $123 \times 10 \times 27$) |
| **Sequence Tensors (L=15)**| `datasets/processed/driving_sample/features/sequences/run_official/sequences_L15.npy`| 108 tensors | $175.0\text{ KB}$ | `PASS` (Shape: $108 \times 15 \times 27$) |
| **Sequence Tensors (L=20)**| `datasets/processed/driving_sample/features/sequences/run_official/sequences_L20.npy`| 93 tensors | $200.9\text{ KB}$ | `PASS` (Shape: $93 \times 20 \times 27$) |
| **Dataset Splits Manifest** | `datasets/processed/driving_sample/features/sequences/run_official/dataset_splits.json` | 3 partitions | $2.1\text{ KB}$ | `PASS` ($G=25$ embargo gap) |
| **Feature Manifest** | `datasets/processed/driving_sample/features/sequences/run_official/feature_manifest.json` | 27 features | $4.5\text{ KB}$ | `PASS` (Detection, Motion, Quality) |

### Provenance & Governance:
- **Source:** UC Berkeley BDD100K Driving Video (`026c7465-309f6d33.mp4`).
- **License:** BSD 3-Clause / BDD100K terms for academic and non-commercial research.
- **Data Integrity:** Zero fabricated records, zero synthetic labels injected, zero corrupted coordinates.

---

## 4. Feature and Target Compatibility Audit

### 4.1 Input Feature Schema (27 Features)
1. **Detection Features (6):** `detection_count`, `mean_confidence`, `max_confidence`, `min_confidence`, `confidence_variance`, `foreground_coverage_ratio`.
2. **Motion Features (8):** `active_track_count`, `new_track_count`, `lost_track_count`, `mean_track_age`, `mean_displacement`, `max_displacement`, `mean_velocity_x`, `mean_velocity_y`.
3. **Visual Quality Features (13):** `mean_intensity`, `intensity_variance`, `contrast_michelson`, `blur_laplacian_variance`, `blur_sobel_variance`, `dark_channel_prior_mean`, `color_saturation_mean`, `temporal_flicker`, `optical_flow_magnitude_mean`, `optical_flow_magnitude_max`, `optical_flow_divergence`, `optical_flow_curl`, `scene_brightness_drift`.

- **Feature Ordering:** Strictly locked to `feature_manifest.json` `ordered_feature_names`.
- **Normalization & Scaling:** Train-only robust median/IQR scaling (`scaler_parameters.json`).
- **Imputation:** Train-only median imputation (`imputation_parameters.json`).

### 4.2 Target Contract & Boundedness
- **Target Horizons:** $H \in \{1, 3, 5\}$ frames into the future ($0.2\text{s}, 0.6\text{s}, 1.0\text{s}$).
- **Target Metric:** Ground-truth frame detection reliability ($F_1$ score calculated via Hungarian IoU matching at $\tau = 0.50$).
- **Target Boundaries:** Continuous values strictly bounded in $[0.0, 1.0]$.
- **Target Dimensions:** Target vector shape `(N, 3)` matches sequence batch tensors `(N, L, 27)`.

---

## 5. Train / Validation / Test Split Integrity

The dataset uses a **chronological purged split** with an explicit **25-frame temporal embargo gap** ($G=25$):

```
Video Timeline (Frames 1 – 200, 40.01s):
[============= TRAIN =============] [== GAP ==] [==== VAL ====] [== GAP ==] [==== TEST ====]
Frames 1 – 105                      Frames 106-130  Frames 131-152  Frames 153-177  Frames 178-200
(101 seqs at L=5)                   (Embargo G=25) (18 seqs at L=5) (Embargo G=25) (19 seqs at L=5)
```

### Leakage-Prevention Verification:
- **No Frame Overlap:** Train, Val, and Test frame indices are mutually exclusive.
- **No Sequence Overlap:** Sliding windows never cross split boundaries.
- **No Target Leakage:** Sequences near split ends do not look ahead into subsequent splits across the 25-frame embargo gaps ($H_{\max} = 5 \ll G = 25$).
- **Strict Fitting Discipline:** Feature scalers, imputers, and baseline means are fitted strictly on the Train partition.

---

## 6. Baseline Model Readiness

The baseline forecasting framework (`src/forecasting/baseline_forecaster.py`) was audited:

| Baseline Strategy | Mathematical Formulation | Fit Dependency | Horizon Support | Output Status |
| :--- | :--- | :--- | :---: | :---: |
| **Strategy A: Persistence** | $\hat{y}_{t+h} = y_t$ (last observed target) | None (zero params) | $H \in \{1, 3, 5\}$ | Ready (`PredictionRecord`) |
| **Strategy B: Historical Mean** | $\hat{y}_{t+h} = \bar{y}_{\text{train}} = 0.000582$ | Train targets only ($N=303$) | $H \in \{1, 3, 5\}$ | Ready (`PredictionRecord`) |
| **Strategy C: Linear Trend** | $\hat{y}_{t+h} = \text{clip}(y_t + m \cdot h, 0, 1)$ | Rolling historical series | $H \in \{1, 3, 5\}$ | Ready (`PredictionRecord`) |

- **Verification Execution:** Evaluated 1,242 prediction records across 138 sequences ($L=5$).
- **Train-Only Mean Verified:** `training_mean = 0.000582178` derived strictly from 303 training targets.

---

## 7. LSTM Training Pipeline Readiness

The deep learning pipeline components in `src/forecasting/` were inspected and verified:

- **Model Architecture (`src/forecasting/lstm_forecaster.py`):**
  - Architecture: `LSTMForecaster`
  - Input dimension: $F = 27$
  - Hidden dimension: $32$ units (1 layer, default)
  - Output projection: Linear layer projecting hidden state to 3 horizon predictions
  - Output activation: `Sigmoid` (enforces strict $[0.0, 1.0]$ bounds)
  - Trainable parameters: **7,907** (weights footprint $<32\text{ KB}$)
- **Model Factory (`src/forecasting/model_factory.py`):**
  - `build_forecasting_model` automatically resolves feature dimensions from `feature_manifest.json`.
- **Training Engine (`src/forecasting/training_engine.py`):**
  - Optimizer: `AdamW` (learning rate $0.001$, weight decay $0.0001$)
  - Scheduler: `ReduceLROnPlateau` (factor $0.5$, patience $3$, min lr $1\times 10^{-6}$)
  - Gradient clipping: Max norm $1.0$ (prevents gradient explosion)
  - Early stopping: Patience $8$ epochs, minimum delta $1\times 10^{-4}$
  - Checkpoint manager: Atomic `.pt` serialization preserving model state, optimizer state, epoch, and telemetry.
- **Hardware Enforcement:** CPU execution strictly mandated (`device: cpu`, 2 OpenMP threads).

---

## 8. CPU Resource & Feasibility Analysis

All training parameters were evaluated against the target hardware (Intel Core i3, 8 GB RAM, Windows 11):

| Resource / Parameter | Budget Ceiling | Allocated in Config | Safety Margin | Feasibility Status |
| :--- | :---: | :---: | :---: | :---: |
| **Compute Device** | Intel Core i3 (CPU) | `device: cpu` | Pure CPU execution | **`PASS`** |
| **Thread Allocation** | 8 logical cores | 2 threads | 6 cores free for OS | **`PASS`** |
| **Process Memory RSS** | 8.00 GB RAM | $<500\text{ MB}$ RSS | $>7.5\text{ GB}$ ($>93\%$) headroom | **`PASS`** |
| **Model Parameters** | $<500,000$ | 7,907 parameters | $>98\%$ below ceiling | **`PASS`** |
| **Batch Size** | $\le 32$ | 8 sequences | Fast CPU mini-batching | **`PASS`** |
| **Batches per Epoch** | $<100$ | 13 training batches | Negligible CPU load | **`PASS`** |
| **Epoch Latency** | $<5.0\text{ s}$ / epoch | $\approx 40\text{ ms}$ / epoch | $>100\times$ faster than ceiling | **`PASS`** |
| **Full Training Run (30 ep)** | $<10\text{ min}$ | $\approx 2\text{ s}$ total compute | Instantaneous execution | **`PASS`** |

---

## 9. Safe Corrections Applied

During the readiness inspection, the following low-risk improvements were implemented and verified with regression tests:

1. **CLI Parameter Support (`src/forecasting/train_lstm.py`):**
   - *Issue:* `train_lstm.py` lacked a `--metadata-file` argument, preventing explicit loading of Day 8 labeled metadata without editing code.
   - *Fix:* Added `--metadata-file` CLI argument and `--dry-run` flag.
2. **Config Schema Enhancement (`src/forecasting/forecasting_config.py`):**
   - *Issue:* `DataConfig` lacked a `metadata_file` field.
   - *Fix:* Added `metadata_file: Optional[str] = None` to `DataConfig` dataclass and validation logic.
3. **Automated Regression Test (`tests/forecasting/test_train_lstm_cli.py`):**
   - *Addition:* Added `test_cli_labeled_data_dry_run` verifying clean dry-run execution on real Day 8 labeled metadata.
   - *Result:* 6 of 6 tests passed in $2.1\text{ s}$.

---

## 10. Automated Testing Verification

Executed test suite across forecasting, evaluation, and validation packages using project virtual environment:

- `tests/forecasting/test_train_lstm_cli.py`: **6 / 6 PASSED**
- `tests/test_audit_project.py`: **8 / 8 PASSED**
- Full test pass rate across modified components: **100% PASS** (0 failed, 0 regressions).

---

## 11. Day 9 Experiment Plan

The following 5 structured experiments are scheduled for execution across subsequent Day 9 tasks:

```mermaid
flowchart TD
    A["Exp A: Baseline Benchmark\n(Persistence, Mean, Trend)"] --> B["Exp B: LSTM Baseline Training\n(Standard MSE Loss, 30 Epochs)"]
    B --> C["Exp C: Training Optimization\n(Weighted Huber Loss / Focal Loss)"]
    C --> D["Exp D: Multi-Horizon Forecasting\n(Horizons H1, H3, H5 Analysis)"]
    D --> E["Exp E: Final Evaluation & Benchmark\n(MAE, RMSE, Baseline Comparison)"]
```

---

### Experiment A — Baseline Benchmark

- **Objective:** Establish formal empirical performance floors using non-parametric baseline forecasters across all three forecasting horizons.
- **Models:**
  1. Persistence Forecaster ($\hat{y}_{t+h} = y_t$).
  2. Historical Mean Forecaster ($\hat{y}_{t+h} = \bar{y}_{\text{train}}$).
  3. Linear Trend Forecaster ($\hat{y}_{t+h} = \text{clip}(y_t + m \cdot h, 0, 1)$).
- **Target Horizons:** $H1$ ($t+1$), $H3$ ($t+3$), $H5$ ($t+5$).
- **Partitioning:** Models fit on Train ($N=101$), evaluated on Validation ($N=18$) and Test ($N=19$).
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- **Validation Criteria:** Every baseline prediction must be deterministic and reside strictly within $[0.0, 1.0]$.
- **Execution Command:**
  ```bash
  python -m src.forecasting.baseline_forecaster --input datasets/processed/driving_sample/features/sequences/run_official --strategy all --horizons 1 3 5
  ```

---

### Experiment B — LSTM Standard Training

- **Objective:** Train the initial multi-horizon LSTM model using standard unweighted Mean Squared Error (MSE) loss to establish the deep learning baseline.
- **Architecture:** `LSTMForecaster` (Input: 27, Hidden: 32, Layers: 1, Output: 3 horizons, Sigmoid activation).
- **Training Hyperparameters:**
  - Optimizer: `AdamW` ($\text{lr} = 0.001$, $\text{weight\_decay} = 0.0001$)
  - Scheduler: `ReduceLROnPlateau` ($\text{patience} = 3$, $\text{factor} = 0.5$)
  - Batch Size: 8
  - Max Epochs: 30
  - Early Stopping: Patience 8 epochs
  - Loss Function: Standard Mean Squared Error (`nn.MSELoss()`)
- **Validation Criteria:**
  - Training and validation loss must decrease or stabilize without NaN/Inf values.
  - Best model checkpoint must be saved based strictly on validation split loss.
  - Zero test data used during model optimization.
- **Expected Outcome / Limitation:** Due to the $99.5\%$ zero-$F_1$ detector skew, model predictions are expected to collapse near zero ($\hat{y} \approx 0.0$), matching the historical mean.
- **Execution Command:**
  ```bash
  python -m src.forecasting.train_lstm --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json --sequence-length 5 --batch-size 8 --epochs 30 --device cpu
  ```

---

### Experiment C — Training Optimization (Loss Reweighting)

- **Objective:** Overcome the extreme detector zero-skew by training the LSTM with an **Asymmetric Weighted Huber Loss** that heavily penalizes errors on non-zero reliability frames ($F_1 > 0$).
- **Loss Formulation:**
  $$\mathcal{L}(y, \hat{y}) = w_i \cdot \ell_{\text{Huber}}(y_i, \hat{y}_i), \quad w_i = \begin{cases} 50.0 & \text{if } y_i > 0 \\ 1.0 & \text{if } y_i = 0 \end{cases}$$
- **Comparative Variants:**
  1. *Run C1:* Standard MSE Loss (baseline from Exp B).
  2. *Run C2:* Weighted Huber Loss ($w_{\text{non-zero}} = 25.0$).
  3. *Run C3:* High-Penalty Weighted Huber Loss ($w_{\text{non-zero}} = 50.0$).
- **Monitoring & Telemetry:** Log gradient norms, learning rate decays, and validation loss trajectories.
- **Validation Criteria:** Gradients remain stable ($\|\mathbf{g}\| \le 1.0$ via clipping); model predicts non-zero reliability transitions in frames 85–95.

---

### Experiment D — Multi-Horizon Forecasting Analysis

- **Objective:** Evaluate forecasting fidelity across the three distinct temporal horizons ($H1 = 1$, $H3 = 3$, $H5 = 5$ frames).
- **Analyses to Execute:**
  1. *Horizon-Wise Error Curves:* Verify the fundamental forecasting theorem:
     $$\text{Error}(H1) \le \text{Error}(H3) \le \text{Error}(H5)$$
  2. *Prediction Calibration:* Check if forecast confidence diminishes appropriately over longer horizons.
  3. *Transition Lead-Time:* Measure how many frames in advance the model anticipates the Frame 90 detection failure transition.
- **Outputs:** Tabular horizon metrics, error curves, and prediction alignment logs.

---

### Experiment E — Final Benchmark & Comparative Evaluation

- **Objective:** Perform a rigorous, multi-model benchmark comparing all baseline strategies against the trained LSTM variants on the isolated test partition.
- **Evaluation Matrix:**

| Model / Strategy | Loss Function | H1 MAE | H3 MAE | H5 MAE | Overall RMSE | Degradation Capture Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistence Baseline | None | — | — | — | — | — |
| Historical Mean | None | — | — | — | — | — |
| Linear Trend | None | — | — | — | — | — |
| LSTM (Unweighted) | MSE | — | — | — | — | — |
| LSTM (Weighted Huber) | Asymmetric | — | — | — | — | — |

- **Evaluation Protocol:**
  - Evaluated strictly on the isolated Test partition (19 sequences).
  - Out-of-bounds boundary frames properly masked out per horizon.
  - Full error distribution analysis (mean, median, standard deviation, max error).

---

## 12. Final Training Readiness Determination

### Determination:
# **`READY WITH LIMITATIONS` (Decision Tier B)**

### Formal Certification:
All software interfaces, dataset tensors, ground-truth supervisory labels, training loops, and test suites are 100% verified and operational on CPU hardware. The system is certified ready to begin Experiment A (Baseline Benchmark) and Experiment B (LSTM Training) in Day 9 Task 2.
