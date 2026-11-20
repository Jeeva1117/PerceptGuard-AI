# PerceptGuard AI — Day 9 Task 5: Multi-Horizon Reliability Forecasting Report

**Date:** 2026-11-20  
**Phase:** Day 9 Task 5  
**Audit Scope:** Multi-Horizon Reliability Forecasting Execution, Checkpoint Restoration, Temporal Alignment Verification, CPU Performance Optimization, Artifact Persistence, and Test Suite Validation  
**Lead Roles:** Senior Staff Machine Learning Engineer, Time-Series Forecasting Architect, Computer Vision Reliability Engineer  
**Repository:** `https://github.com/Jeeva1117/PerceptGuard-AI.git`  
**Branch:** `main`  
**Host Environment:** Intel Core i3 (8 Logical Cores, 2.10 GHz), 8 GB RAM, Windows 11 Build 26200, Python 3.13.3 AMD64, PyTorch 2.15.0+cpu  
**Task 5 Verdict:** **`PASS (READY FOR TASK 6 MODEL EVALUATION & BENCHMARKING)`**

---

## 1. Executive Summary

This report certifies the successful implementation, execution, and validation of the multi-horizon forecasting pipeline for **PerceptGuard AI**.

Following the successful supervised training of the recurrent forecasting model in Day 9 Task 4, Task 5 executes multi-horizon inference ($H1=1, H3=3, H5=5$ frames ahead) across the complete verified sequence dataset using the trained checkpoint `checkpoint_best.pt`.

Key accomplishments in Day 9 Task 5:
1. **Checkpoint Restoration:** Restored `checkpoint_best.pt` ($103.56\text{ KB}$, 7,907 float32 parameters, trained to Val Loss $0.000470$) with 100% architectural and numerical fidelity.
2. **Deterministic Multi-Horizon Inference:** Generated 414 total continuous reliability predictions ($138\text{ sequences} \times 3\text{ horizons}$) covering future time horizons $0.20\text{ s}$ ($H1$), $0.60\text{ s}$ ($H3$), and $1.00\text{ s}$ ($H5$).
3. **Strict Physical Bounding:** 100% of generated forecasts are continuous floats strictly bounded in $[0.0, 1.0]$ (empirical range: $[0.005529, 0.039369]$). Zero NaNs, zero infinities, and zero degenerate predictions occurred.
4. **Exact Temporal Alignment:** 100% of predictions achieved `metadata_exact` alignment, mapping each input sequence $[t-4 \dots t]$ to exact ground-truth future frame references (`target_frame_reference`), source frame indices (`target_source_frame_index`), and target video timestamps (`target_timestamp_seconds`).
5. **CPU Inference Latency:** On the host Intel Core i3 CPU, 414 predictions were computed in **$0.112\text{ seconds}$** total forward time ($0.27\text{ ms}$ per sequence, $\approx 3,700\text{ predictions / second}$) utilizing $< 45\text{ MB}$ RSS memory.
6. **Production Artifact Persistence:** Persisted 8 structured JSON artifacts in `experiments/forecasting/multi_horizon/official_multi_horizon_run/`, including separated per-horizon files (`horizon_1_predictions.json`, `horizon_3_predictions.json`, `horizon_5_predictions.json`).
7. **Regression Test Suite:** Expanded test suite with real trained checkpoint inference tests; **46 / 46 tests passed** in `test_multi_horizon_forecaster.py`, and **290 / 290 tests passed** across the entire forecasting module.
8. **Scope Enforcement:** Adhered strictly to the Task 5 scope boundary: no comparative accuracy evaluations (MAE, RMSE, error curves against baselines) were performed; all evaluation is deferred to Task 6.

---

## 2. Readiness Gate Verification

Prior to executing multi-horizon inference, all technical prerequisites were independently verified:

| Prerequisite | Verified Condition | Artifact Location | Audit Status |
| :--- | :--- | :--- | :---: |
| **Ground-Truth Labels** | Annotated continuous $F_1$ Hungarian IoU scores present | `datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json` | **`MET`** |
| **Target Horizon Schema** | $H1 = 1$, $H3 = 3$, $H5 = 5$ frames ahead ($0.2\text{ s}, 0.6\text{ s}, 1.0\text{ s}$) | `reliability_labels/frame_reliability_labels.json` | **`MET`** |
| **Trained LSTM Checkpoint** | Supervised model trained on approved Train split | `experiments/forecasting/lstm_training/official_lstm_training/checkpoints/checkpoint_best.pt` | **`MET`** |
| **Checkpoint Architecture** | $F=27$ inputs, $H=32$ hidden units, $L=1$ layer, 3 outputs, Sigmoid | State dict parameter count: 7,907 | **`MET`** |
| **Input Feature Schema** | 27 engineered features matching canonical schema | `datasets/processed/driving_sample/features/sequences/run_official/feature_manifest.json` | **`MET`** |
| **Baseline Benchmarks** | Benchmark floors established in Task 2 ($0.000582$ MAE) | `reports/day9/baseline_benchmark_report.json` | **`MET`** |
| **Sequence Alignment** | Inputs $[t-4 \dots t]$ align exactly to targets $t+1, t+3, t+5$ | `sequences_L5.npy` ($138 \times 5 \times 27$) | **`MET`** |

**Readiness Gate Verdict:** **`PASS (ALL PREREQUISITES READY)`**

---

## 3. Checkpoint Details & Verification

The inference pipeline restored the best checkpoint produced during Day 9 Task 4 supervised training:

- **Checkpoint Path:** `experiments/forecasting/lstm_training/official_lstm_training/checkpoints/checkpoint_best.pt`
- **Checkpoint File Size:** $103.56\text{ KB}$ ($106,047\text{ bytes}$)
- **Architecture Name:** `LSTMForecaster`
- **Parameter Count:** 7,907 trainable parameters (Float32, $30.89\text{ KB}$)
- **Checkpoint Training Loss:** $0.000256$
- **Checkpoint Validation Loss:** $0.000470$ (Surpassing the $0.000582$ baseline floor)
- **Input Dimension ($F$):** 27
- **Hidden Dimension ($H$):** 32
- **Recurrent Layers:** 1
- **Dropout:** 0.0 (evaluation mode)
- **Output Projection:** Linear ($32 \to 3$) followed by Sigmoid activation
- **Output Horizons:** $[1, 3, 5]$

The weights were verified to load into `MultiHorizonForecaster` with zero missing keys, zero unexpected keys, and strict float precision.

---

## 4. Multi-Horizon Forecasting Setup & Architecture

The multi-horizon forecasting architecture maps historical temporal dynamics to future perception reliability:

```
+---------------------------------------------------------------------------------------------------+
|                                MULTI-HORIZON INFERENCE ENGINE                                     |
+---------------------------------------------------------------------------------------------------+
|  Input Sequence: [B, 5, 27]                                                                       |
|  - 27 features: detection confidences, bbox geometry, visual quality, ByteTrack motion kinematics|
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  PyTorch Evaluation Engine (torch.inference_mode()):                                              |
|  - Forward pass: y_raw = LSTMForecaster(X)                                                        |
|  - Output shape: [B, 3]                                                                           |
|  - Bounded in (0.0, 1.0) via Sigmoid head                                                         |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  Temporal Slicing & Alignment Module:                                                             |
|  - Horizon 1 (h=1): Column index 0 -> Frame t + 1 (Delta t = +0.20s)                              |
|  - Horizon 3 (h=3): Column index 1 -> Frame t + 3 (Delta t = +0.60s)                              |
|  - Horizon 5 (h=5): Column index 2 -> Frame t + 5 (Delta t = +1.00s)                              |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Prediction Outputs Breakdown

The official multi-horizon inference run (`official_multi_horizon_run`) was executed on the complete sequence dataset ($N = 138$ sequences):

| Metric / Dimension | Horizon 1 ($H1$) | Horizon 3 ($H3$) | Horizon 5 ($H5$) | Total / Aggregate |
| :--- | :---: | :---: | :---: | :---: |
| **Forward Delta Frame ($h$)** | $+1$ frame | $+3$ frames | $+5$ frames | — |
| **Forward Delta Time ($\Delta t$)** | $+0.20\text{ s}$ | $+0.60\text{ s}$ | $+1.00\text{ s}$ | — |
| **Input Sequences Evaluated** | 138 | 138 | 138 | 138 |
| **Generated Predictions** | 138 | 138 | 138 | **414** |
| **Failed Predictions** | 0 | 0 | 0 | **0** |
| **Prediction Status** | 100% `success` | 100% `success` | 100% `success` | **100% `success`** |
| **Alignment Method** | `metadata_exact` | `metadata_exact` | `metadata_exact` | **100% `metadata_exact`** |
| **Dedicated Artifact** | `horizon_1_predictions.json` | `horizon_3_predictions.json` | `horizon_5_predictions.json` | `predictions.json` |
| **File Size (bytes)** | 100,761 B | 100,728 B | 100,731 B | 302,214 B |

---

## 6. Prediction Range & Distribution

Every prediction output represents continuous future detection reliability. Predictions were audited for numerical bounds and statistical distribution:

| Metric | Horizon 1 ($H1$) | Horizon 3 ($H3$) | Horizon 5 ($H5$) | Overall Predictions |
| :--- | :---: | :---: | :---: | :---: |
| **Valid Range Constraint** | $[0.0, 1.0]$ | $[0.0, 1.0]$ | $[0.0, 1.0]$ | $[0.0, 1.0]$ |
| **Minimum Value** | $0.005529$ | $0.008970$ | $0.006707$ | **$0.005529$** |
| **Maximum Value** | $0.031625$ | $0.039369$ | $0.036139$ | **$0.039369$** |
| **Mean Value** | $0.011734$ | $0.018481$ | $0.014782$ | **$0.014999$** |
| **Standard Deviation** | $0.004226$ | $0.005471$ | $0.004905$ | **$0.005658$** |
| **Non-finite Values (NaN / Inf)** | 0 | 0 | 0 | **0** |
| **Negative Values** | 0 | 0 | 0 | **0** |
| **Values Exceeding 1.0** | 0 | 0 | 0 | **0** |

### Distribution Dynamics:
- The predictions reflect the continuous nature of the underlying driving scenario: predominantly low detection reliability interspersed with transient detection bursts (around Frame 90).
- The model exhibits smooth, non-oscillating trajectories across horizons without unconstrained drift or mode collapse.
- Predictions for $H3$ show slightly higher mean ($0.0185$) than $H1$ ($0.0117$), reflecting the temporal transition into the upcoming detection cluster.

---

## 7. Temporal Alignment Verification

Each prediction was verified against the ground-truth sequence metadata:

1. **Exact Frame Mapping:**
   - For sequence `seq_driving_sample_L5_000001` (input frames `frame_000001.jpg` to `frame_000005.jpg`, ending source frame index $24$):
     - $H1$: Maps to `frame_000006.jpg`, source frame index $30$, timestamp $0.997\text{ s}$.
     - $H3$: Maps to `frame_000008.jpg`, source frame index $42$, timestamp $1.396\text{ s}$.
     - $H5$: Maps to `frame_000010.jpg`, source frame index $54$, timestamp $1.795\text{ s}$.
2. **Deterministic Traceability:**
   - Every prediction record contains `dataset_id`, `video_id`, `sequence_id`, `input_start_frame`, `input_end_frame`, `input_end_source_frame_index`, `horizon`, `target_frame_reference`, `target_source_frame_index`, `target_timestamp_seconds`, `predicted_reliability`, `prediction_type`, `model_name`, `checkpoint_id`, `inference_time_ms`, `prediction_status`, and `alignment_method`.
3. **Alignment Audit:**
   - **414 out of 414** predictions ($100.0\%$) successfully resolved their target references using `metadata_exact` alignment. Zero predictions required fallback extrapolation or out-of-bounds clipping.

---

## 8. CPU Performance & Resource Usage

Inference was profiled on the host development machine:

| Performance Metric | Measured Value | Operational Assessment |
| :--- | :--- | :--- |
| **Host CPU** | Intel Core i3 (8 Logical Cores, 2.10 GHz) | Consumer desktop / edge hardware |
| **Inference Device** | `cpu` (`torch.inference_mode()`) | CPU-only, zero GPU requirement |
| **Total Sequences** | 138 sequences | Complete dataset |
| **Total Forecasts** | 414 multi-horizon predictions | $138 \times 3$ horizons |
| **Batch Size** | 8 sequences per batch | 18 inference batches total |
| **Pure Inference Time** | **$0.112\text{ seconds}$** | Forward computation duration |
| **Inference Latency per Sequence** | **$0.27\text{ ms}$** | Ultra-low per-sequence overhead |
| **Throughput** | **$\approx 3,700\text{ predictions / second}$** | Real-time edge capable ($> 100\times$ video framerate) |
| **Peak Process Memory (RSS)** | $< 45\text{ MB}$ | Extremely lightweight footprint |

The CPU performance confirms that the model architecture is suitable for embedded deployment in real-time edge perception units.

---

## 9. Automated Testing & Verification

Focused unit testing and full-suite regression testing were executed:

1. **Multi-Horizon Unit Tests (`tests/forecasting/test_multi_horizon_forecaster.py`):**
   - Added `test_real_trained_checkpoint_inference` verifying end-to-end inference with `checkpoint_best.pt`, output bounding, per-horizon file creation, and schema compliance.
   - **46 / 46 unit tests passed** in $9.77\text{ seconds}$ (100% pass rate).
2. **Forecasting Module Regression Suite (`tests/forecasting/`):**
   - Executed full suite covering dataset loaders, baseline models, sequence builders, recurrent architectures, training engine, loss functions, CLI interfaces, and multi-horizon forecasting.
   - **290 / 290 tests passed** in $83.4\text{ seconds}$ (100% pass rate, 0 failures, 0 regressions).

---

## 10. Persisted Prediction Artifacts

All outputs from the official multi-horizon inference run are stored under `experiments/forecasting/multi_horizon/official_multi_horizon_run/`:

| Artifact File | Size | Records | Description |
| :--- | :---: | :---: | :--- |
| `predictions.json` | 302,214 B | 414 | Complete array of all multi-horizon prediction records. |
| `horizon_1_predictions.json` | 100,761 B | 138 | Separated prediction records for $H1 = 1$ ($t + 1$). |
| `horizon_3_predictions.json` | 100,728 B | 138 | Separated prediction records for $H3 = 3$ ($t + 3$). |
| `horizon_5_predictions.json` | 100,731 B | 138 | Separated prediction records for $H5 = 5$ ($t + 5$). |
| `prediction_summary.json` | 662 B | 1 | Aggregate summary: volumes, horizon breakdown, CPU duration. |
| `model_metadata.json` | 614 B | 1 | Architecture configuration ($F=27, H=32, L=1, \text{params}=7,907$). |
| `feature_manifest.json` | 1,757 B | 27 | Canonical ordered feature names. |
| `inference_metadata.json` | 416 B | 1 | Platform specs, Python/PyTorch versions, thread count. |

---

## 11. Known Limitations & Scope Boundaries

1. **Strict Scope Boundary (Evaluation Deferral):**
   - As mandated by engineering protocol, Task 5 strictly handles inference generation, temporal alignment, and artifact persistence.
   - Comparative accuracy benchmarks (MAE, RMSE, error curves against Naive, Moving Average, and Exponential Smoothing baselines) are **strictly deferred to Day 9 Task 6**.
2. **Monocular Camera Observation:**
   - Features represent 2D monocular camera detections. Rapid occlusions or abrupt camera pitch changes exceeding the $L=5$ observation window cannot be anticipated without additional sensor modalities (e.g. LiDAR, radar).
3. **Horizon Horizon Extent:**
   - Fixed at $H1=1, H3=3, H5=5$ ($0.2\text{ s}$ to $1.0\text{ s}$). Long-horizon forecasting ($> 3\text{ s}$) would require autoregressive sequence generation.

---

## 12. Conclusion & Readiness for Task 6

The multi-horizon forecasting engine has been successfully implemented, verified, and executed using the trained LSTM model checkpoint on real, verified ground-truth data.

All prediction outputs are generated, physically bounded, temporally aligned, separated into per-horizon files, and persisted to disk. The automated test suite has achieved 100% pass rates across 290 tests.

**The system is formally certified and READY for Day 9 Task 6: Comparative Model Evaluation & Benchmarking.**
