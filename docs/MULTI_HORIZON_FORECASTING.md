# PerceptGuard AI — Multi-Horizon Forecasting Engine
## Technical Specification, Horizon Alignment & CPU Inference Architecture

---

## 1. Executive Summary

This document specifies the architecture, horizon alignment mechanics, prediction schema, and CPU-optimized inference workflow of the **Multi-Horizon Forecasting Engine** (`MultiHorizonForecaster`) updated and verified in **Day 9 Task 5**.

The engine operates on historical perception sequences ($L = 5$ with $F = 27$ engineered features) to forecast future object-detection reliability across multiple forward time horizons:
- **Horizon 1 ($h = 1$):** Forecast reliability for frame $t + 1$ ($\Delta t = 0.20\text{ s}$)
- **Horizon 3 ($h = 3$):** Forecast reliability for frame $t + 3$ ($\Delta t = 0.60\text{ s}$)
- **Horizon 5 ($h = 5$):** Forecast reliability for frame $t + 5$ ($\Delta t = 1.00\text{ s}$)

The system is optimized for CPU execution (Intel Core i3, 8 GB RAM, Windows 11) using `torch.inference_mode()`, chunked batch processing ($B = 8$), and deterministic ground-truth temporal alignment (`metadata_exact`).

---

## 2. Multi-Horizon Forecasting Concept & Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                MULTI-HORIZON INFERENCE PIPELINE                                   |
+---------------------------------------------------------------------------------------------------+
|  Input Sequence: [B, L=5, F=27]                                                                   |
|  (Historical temporal perception features ending at frame t)                                      |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  Trained LSTMForecaster Checkpoint (checkpoint_best.pt from Day 9 Task 4)                         |
|  - Validates feature count (F=27) and horizons ([1, 3, 5])                                        |
|  - Evaluation mode: model.eval() with torch.inference_mode()                                      |
|  - Total Trainable Parameters: 7,907 (Float32, 30.89 KB parameter footprint)                     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  Forward Pass: y_raw = model(X) ∈ (0.0, 1.0)^(B × 3)                                              |
|  Continuous reliability bounded in [0.0, 1.0] via Sigmoid activation                              |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  Temporal Horizon Slicing & Frame Alignment Engine:                                               |
|  - Column 0 -> Horizon 1 -> Aligned Target Frame: t + 1 (Source Index & Timestamp)                |
|  - Column 1 -> Horizon 3 -> Aligned Target Frame: t + 3 (Source Index & Timestamp)                |
|  - Column 2 -> Horizon 5 -> Aligned Target Frame: t + 5 (Source Index & Timestamp)                |
|  - Strategy: 'metadata_exact' (derived from Day 8 verified sequence metadata)                     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  Multi-File Prediction Artifact Persistence (experiments/forecasting/multi_horizon/<run_id>/):    |
|  - predictions.json             (Full array of all 414 prediction records)                        |
|  - horizon_1_predictions.json   (138 records for h=1)                                             |
|  - horizon_3_predictions.json   (138 records for h=3)                                             |
|  - horizon_5_predictions.json   (138 records for h=5)                                             |
|  - prediction_summary.json      (Record counts, timing, and distribution summary)                 |
|  - model_metadata.json          (7,907 params, F=27, H=32, L=1, sigmoid)                           |
|  - feature_manifest.json        (Ordered list of 27 engineered feature names)                     |
|  - inference_metadata.json      (Host environment, PyTorch CPU runtime, batch size)               |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Checkpoint Requirements & Compatibility Validation

`MultiHorizonForecaster` requires a valid checkpoint dictionary generated by `CheckpointManager`:

| Key | Expected Type | Validation Criteria | Verified in Day 9 Task 5 |
| :--- | :--- | :--- | :---: |
| `model_state_dict` | `OrderedDict` | Must contain matching weights for `lstm` and `fc` projection head. | **`MATCHED`** |
| `input_size` | `int` | Must strictly equal 27 (matches feature engineering schema). | **`27`** |
| `hidden_size` | `int` | Must match model hidden dimension (default 32, $\ge 4$). | **`32`** |
| `output_horizons` | `List[int]` | Must strictly equal `[1, 3, 5]`. | **`[1, 3, 5]`** |
| `experiment_id` | `str` | Identifier tracked as `checkpoint_id` in prediction records. | **`official_lstm_training`** |

If a checkpoint file is missing, corrupted, or exhibits feature/horizon dimension mismatches, the engine halts with `MultiHorizonForecastingError`. It never silently modifies input tensors or weights.

---

## 4. Horizon Alignment & Frame Index Handling

Temporal alignment must never assume consecutive frame numbers. The engine employs a 3-tier resolution strategy:

1. **`metadata_exact` (Primary Strategy — Used in Official Day 9 Run)**:
   Extracts pre-computed horizon metadata from the Day 8 `sequence_metadata_labeled.json` (e.g. `target_frame_id`, `target_source_frame_index`, `target_timestamp_seconds`). This guarantees exact alignment with actual video frames.
2. **`frame_index_offset` (Secondary Strategy)**:
   When a chronological list of available frame records is provided, the current frame index is located, and the target is resolved at offset `current_index + horizon`. Handles non-continuous frame drops gracefully.
3. **`sequential_extrapolation` (Fallback Strategy)**:
   Extracts frame numbers via regex (`frame_(\d+).ext`), projects target frame number `frame_num + horizon`, and calculates expected timestamp offsets ($\Delta t = h \times 0.1994\text{s}$).

Each prediction record explicitly logs its `alignment_method` (`metadata_exact`, `frame_index_offset`, `sequential_extrapolation`, or `out_of_bounds`). In the official run, 100% of the 414 predictions achieved `metadata_exact` alignment.

---

## 5. Prediction Record Schema (v1.0)

Every forecast generated by the engine conforms to the versioned schema:

```json
{
  "dataset_id": "driving_sample",
  "video_id": "driving_sample",
  "sequence_id": "seq_driving_sample_L5_000001",
  "input_start_frame": "frame_000001.jpg",
  "input_end_frame": "frame_000005.jpg",
  "input_end_source_frame_index": 24,
  "horizon": 1,
  "target_frame_reference": "frame_000006.jpg",
  "target_source_frame_index": 30,
  "target_timestamp_seconds": 0.997,
  "predicted_reliability": 0.014573777094483376,
  "prediction_type": "continuous_reliability",
  "model_name": "LSTMForecaster",
  "checkpoint_id": "official_lstm_training",
  "inference_time_ms": 8.978,
  "prediction_status": "success",
  "alignment_method": "metadata_exact"
}
```

---

## 6. Official Multi-Horizon Prediction Artifacts

Outputs from the official inference run are persisted under `experiments/forecasting/multi_horizon/official_multi_horizon_run/`:

| Artifact File | Size | Description |
| :--- | :---: | :--- |
| `predictions.json` | 302,214 B | Complete array of all 414 prediction records across all 138 sequences and 3 horizons. |
| `horizon_1_predictions.json` | 100,761 B | 138 records for $H1 = 1$ ($t + 1$, $\Delta t = 0.20\text{ s}$). |
| `horizon_3_predictions.json` | 100,728 B | 138 records for $H3 = 3$ ($t + 3$, $\Delta t = 0.60\text{ s}$). |
| `horizon_5_predictions.json` | 100,731 B | 138 records for $H5 = 5$ ($t + 5$, $\Delta t = 1.00\text{ s}$). |
| `prediction_summary.json` | 662 B | Aggregated volume counts (414 success, 0 failure) and latency metrics. |
| `model_metadata.json` | 614 B | Model architecture specification ($F=27, H=32, L=1, \text{params}=7,907$). |
| `feature_manifest.json` | 1,757 B | Canonical ordered list of 27 input features. |
| `inference_metadata.json` | 416 B | Host platform, runtime environment, PyTorch CPU version, CPU thread count. |

*Note: Evaluation metrics (accuracy, MAE, RMSE against ground-truth) are strictly reserved for Day 9 Task 6.*

---

## 7. Command-Line Interface (CLI)

The inference CLI is invoked via `python -m src.forecasting.multi_horizon_forecaster`:

```powershell
.\venv\Scripts\python.exe -m src.forecasting.multi_horizon_forecaster `
  --checkpoint experiments/forecasting/lstm_training/official_lstm_training/checkpoints/checkpoint_best.pt `
  --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json `
  --sequence-length 5 `
  --horizons 1 3 5 `
  --run-id official_multi_horizon_run
```

### CLI Arguments
- `--input <path>`: Path to Day 5 sequence directory (default: `datasets/processed/driving_sample/features/sequences/run_official`).
- `--metadata-file <path>`: Path to Day 8 labeled sequence metadata (`sequence_metadata_labeled.json`).
- `--checkpoint <path>`: Path to trained LSTM checkpoint file (`.pt`).
- `--config <path>`: Path to YAML configuration (default: `config/forecasting_config.yaml`).
- `--horizons <int...>`: Forecast horizons to predict (default: `1 3 5`).
- `--batch-size <int>`: Batch size for inference (default: 8).
- `--device {cpu,cuda}`: Inference device (default: `cpu`).
- `--sequence-length <int>`: Sequence length to evaluate ($L \in \{5, 10, 15, 20\}$).
- `--output <dir>`: Destination folder for output artifacts.
- `--run-id <str>`: Unique run identifier for subdirectory creation.
- `--max-sequences <int>`: Optional limit on sequences processed.
- `--verbose`: Enable detailed logging.

---

## 8. CPU Optimization & Empirical Performance

- **Inference Mode**: Forward passes run inside `with torch.inference_mode():`, which eliminates autograd graph tracking and version counters.
- **Batch Processing**: Chunked batches ($B=8$) maintain minimal CPU cache invalidation and ensure steady memory residency.
- **Execution Latency**: 414 predictions across 138 sequences completed in **$0.112\text{ s}$** CPU inference time ($0.27\text{ ms}$ per sequence, $\approx 3,700\text{ predictions / sec}$) on an Intel Core i3 host.
- **Memory Overhead**: Peak inference memory remained strictly $< 45\text{ MB}$ RSS.
- **Numerical Stability**: 100% of predictions produced valid finite floats strictly in the range $[0.005529, 0.039369] \subset [0.0, 1.0]$. Zero NaNs, zero infinities.

---

## 9. Ground-Truth Requirements & Known Limitations

1. **Prediction Scope Boundary**: Task 5 strictly handles multi-horizon inference generation, temporal alignment verification, and structured artifact persistence. Model evaluation metrics (MAE, RMSE, R2, degradation analysis vs baselines) are strictly reserved for Day 9 Task 6.
2. **Monocular Input Representation**: Forecasts are conditioned solely on camera-derived detection, motion, and visual quality features. Sudden external occlusions exceeding the historical observation window ($L=5$) require adaptive uncertainty estimation.
3. **Temporal Horizon Extent**: Horizons are pinned to $H1=1, H3=3, H5=5$ ($0.2\text{ s}$ to $1.0\text{ s}$ ahead). Long-horizon extrapolation ($> 2.0\text{ s}$) would require autoregressive feedback or multi-stage sequence models.
