# PerceptGuard AI — Forecasting Environment & Model Setup

## 1. Overview and Forecasting Architecture

PerceptGuard AI forecasts future computer vision perception reliability across upcoming video frames.
In the previous phases (Days 1–5), the pipeline established:
```
Video Loading 
  → Frame Extraction 
  → Preprocessing 
  → YOLO Detection 
  → ByteTrack Object Tracking 
  → Detection & Motion Features 
  → Visual Quality Features 
  → Temporal Feature Aggregation 
  → Temporal Sequence Construction
```

Day 6 initiates the predictive forecasting phase. Day 6 Task 1 establishes a reliable, CPU-compatible, reproducible deep learning forecasting environment, configuration loader, and strict Day 5 sequence data contract validation engine.

### Architectural Pipeline
```
[Day 5 Historical Sequence Tensors (N, L, 27)]
                  │
                  ▼
   [Forecasting Data Contract Validator]
                  │
                  ▼
    [Forecasting Configuration Engine]
                  │
                  ▼
   [Forecasting Environment Verifier]
                  │
                  ▼
  [Future Deep Learning Forecaster (LSTM/GRU)]
                  │
                  ▼
[Multi-Horizon Forecasts: t+1, t+3, t+5 Reliability Scores]
```

---

## 2. Environment Requirements

The forecasting environment is tailored for reproducible execution under resource-constrained desktop and edge CPU environments:

- **Operating System:** Windows 11 (AMD64)
- **Processor:** Intel Core i3 (8 logical CPU threads)
- **System Memory:** 8.00 GB RAM
- **GPU Availability:** None (Pure CPU-first execution)
- **Virtual Environment:** Python venv (`.\venv\Scripts\python.exe`)

---

## 3. CPU Compatibility & Resource Management

Due to the absence of CUDA GPU hardware, the environment enforces strict CPU-first execution:
- **PyTorch CPU Execution:** Zero CUDA requirements or invocations. `torch.cuda.is_available()` is monitored; if CUDA is requested on non-CUDA machines, validation fails cleanly.
- **Bounded CPU Threading:** Automatically sets `torch.set_num_threads(2)` by default (configurable via `runtime.num_threads`). This prevents CPU thrashing and thread oversubscription on 4-to-8 core processors.
- **Memory Optimization:** Default batch size is set to `8` to maintain minimal RAM usage (< 500 MB footprint during forward/backward passes).
- **Worker Configuration:** Default `data.num_workers` is set to `0` for stability and zero-IPC overhead on Windows.

---

## 4. Installed Dependency Versions

The forecasting environment relies on compatible, verified versions:
- **Python:** `3.13.3` (AMD64)
- **PyTorch:** `2.15.0+cpu`
- **NumPy:** `2.5.4`
- **PyYAML:** `6.0.3`
- **pytest:** `9.2.0`

---

## 5. Configuration Parameters

The configuration file `config/forecasting_config.yaml` provides strongly-typed parameters managed by `ForecastingConfig`:

```yaml
project:
  name: PerceptGuard AI
  stage: forecasting
  random_seed: 42

runtime:
  device: cpu
  num_threads: 2
  deterministic: true

data:
  sequence_directory: datasets/processed
  sequence_lengths: [5, 10, 15, 20]
  forecast_horizons: [1, 3, 5]
  batch_size: 8
  num_workers: 0

model:
  architecture: lstm
  input_size: auto
  hidden_size: 32
  num_layers: 1
  dropout: 0.0
  output_horizons: [1, 3, 5]

training:
  learning_rate: 0.001
  weight_decay: 0.0001
  max_epochs: 50
  early_stopping_patience: 8

evaluation:
  metrics:
    - mae
    - rmse

artifacts:
  checkpoint_directory: models/forecasting
  experiment_directory: experiments/forecasting
```

### Parameter Schema
- `project.random_seed`: Seed applied across `random`, `np.random`, and `torch.manual_seed` for reproducibility.
- `runtime.device`: Mandated as `"cpu"`.
- `runtime.num_threads`: Thread limit for PyTorch CPU operations (default: `2`).
- `runtime.deterministic`: Enforces deterministic algorithmic execution where supported.
- `data.sequence_lengths`: Supported historical input lengths $L \in \{5, 10, 15, 20\}$.
- `data.forecast_horizons`: Target forecasting horizons $H \in \{1, 3, 5\}$.
- `model.architecture`: Deep sequence model (`"lstm"` or `"gru"`).
- `model.input_size`: Automatically derived from feature manifest (`"auto"` or integer).
- `model.hidden_size`: Hidden dimension (default: `32`).
- `model.num_layers`: Recurrent layer depth (default: `1`).
- `model.output_horizons`: Subset or exact match to `data.forecast_horizons`.
- `training.learning_rate`: Optimization step size (default: `0.001`).
- `training.weight_decay`: L2 regularization penalty (default: `0.0001`).
- `training.max_epochs`: Maximum training epochs (default: `50`).
- `training.early_stopping_patience`: Validation patience epochs (default: `8`).

---

## 6. Sequence Input Contract

The input data contract ingests sliding-window feature sequences constructed in Day 5 Task 6.

### Input Specification (`ModelInputSpec`)
- **Tensor Shape:** `(batch_size, sequence_length, feature_count)`
- **Dimensionality:** Strictly 3D (`ndim == 3`)
- **Data Type:** `torch.float32` (or `np.float32`)
- **Supported Sequence Lengths:** $L \in \{5, 10, 15, 20\}$
- **Feature Count:** Dynamically verified against `feature_manifest.json` (currently 27 features).
- **Numerical Integrity:** Zero NaNs, zero infinities (`np.isfinite(arr).all() == True`).

### Feature Ordering (27 features)
1. Detection (7): `detection_count`, `unique_class_count`, `mean_confidence`, `confidence_std`, `mean_bbox_area`, `mean_bbox_width`, `mean_bbox_height`
2. Motion (9): `observed_track_count`, `moving_track_count`, `stationary_track_count`, `mean_displacement`, `maximum_displacement`, `mean_speed`, `mean_velocity_x`, `mean_velocity_y`, `motion_feature_coverage`
3. Visual Quality (6): `blur_score`, `mean_brightness`, `brightness_std`, `contrast_std`, `contrast_range`, `contrast_percentile_range`
4. Temporal Aggregation (5): `numeric_feature_coverage`, `w3_confidence_trend_slope`, `w3_blur_trend_slope`, `w5_confidence_trend_slope`, `w5_blur_trend_slope`

---

## 7. Forecast Output Contract

### Output Specification (`ModelOutputSpec`)
- **Tensor Shape:** `(batch_size, number_of_horizons)`
- **Dimensionality:** Strictly 2D (`ndim == 2`)
- **Data Type:** `torch.float32`
- **Forecast Horizons:** $H = [1, 3, 5]$ frames ahead.
- **Target Formulation:** Continuous perception reliability score in the range $[0.0, 1.0]$.
  - `0.0`: Perception failure (e.g., zero bounding box recall, complete occlusion, severe blur dropout).
  - `1.0`: Optimal perception reliability (e.g., high ground-truth bounding box IoU / detection recall).

---

## 8. Ground-Truth Reliability Label Policy (CRITICAL)

PerceptGuard AI enforces scientific integrity regarding ground truth:
1. **YOLO Detection Confidence is NOT Ground Truth:**
   - Model confidence reflects internal detector activation magnitude, not ground-truth detection recall or bounding box accuracy under real-world degradation.
   - Using YOLO confidence as self-supervised pseudo ground-truth creates circular feedback loops and false accuracy metrics.
2. **Current Label Status:**
   - Day 5 sequence metadata records `target_label_available: false` and `target_label: null`.
   - The environment and data contract honestly report `ground_truth_labels_available: false` and `supervised_readiness: "NOT AVAILABLE"`.
   - Unrelated environment verification checks still receive `PASS`.

---

## 9. Day 5 Artifact Verification & Dataset Readiness

Actual Day 5 artifacts verified on disk in `datasets/processed/driving_sample/features/sequences/run_official/`:
- `sequences_L5.npy`: `(138, 5, 27)` float32 — PASS
- `sequences_L10.npy`: `(123, 10, 27)` float32 — PASS
- `sequences_L15.npy`: `(108, 15, 27)` float32 — PASS
- `sequences_L20.npy`: `(93, 20, 27)` float32 — PASS
- `feature_manifest.json`: 27 ordered features across 4 categories — PASS
- `dataset_splits.json`: Disjoint chronological purged splits (train: 70%, val: 15%, test: 15%, temporal gap: 25 frames) — PASS
- `sequence_metadata.json`: 462 total sequence metadata records with frame alignment and horizon targets — PASS
- `scaler_parameters.json` & `imputation_parameters.json`: Standardization and mean imputation parameters — PASS

---

## 10. CLI Usage

### Validate Forecasting Configuration
```bash
python -m src.forecasting.forecasting_config --validate
```

### Verify Forecasting Environment & System Constraints
```bash
python -m src.forecasting.forecasting_environment
```

### Validate Day 5 Sequence Data Contract
```bash
python -m src.forecasting.forecasting_data_contract --validate
```

---

## 11. Test Execution

Execute the full forecasting test suite:
```bash
python -m pytest tests/forecasting/ -v
```

All 47 unit and integration tests validate:
- Configuration schemas, YAML parsing, and boundary checks.
- CPU thread bounds, seed reproducibility, and missing dependency resilience.
- 3D tensor dimensionality, float dtypes, NaN/inf rejection, manifest feature alignment, and split integrity.

---

## 12. Current Limitations & Scope Boundaries

- **Task 1 Scope Bound:** Day 6 Task 1 implements environment setup, configuration validation, and data contracts only. Baseline forecasting models, LSTM/GRU neural network architectures, training loops, evaluation metrics, and Streamlit dashboards are reserved for subsequent tasks.
- **Supervised Training Readiness:** Marked as `NOT AVAILABLE` due to the lack of ground-truth perception reliability annotations. Sequence inputs and horizon indexing are fully verified and ready for downstream modeling.
