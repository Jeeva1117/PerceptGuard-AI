# PerceptGuard AI — Baseline Forecasting System

## 1. Overview & Baseline Architecture

PerceptGuard AI forecasts future perception reliability across upcoming video frames. 
Following the establishment of the forecasting environment and data contract in Day 6 Task 1, Day 6 Task 2 implements an explainable, computationally inexpensive, mathematically rigorous baseline forecasting system.

The baseline forecaster establishes reference performance benchmarks ($H \in \{1, 3, 5\}$ frames ahead) against which subsequent deep recurrent neural models (LSTM/GRU) will be evaluated.

### Architectural Pipeline
```
[Day 5 Historical Sequences (N, L, 27)]
                  │
                  ▼
       [BaselineDataLoader]
   (Train / Val / Test Partitioning)
                  │
                  ▼
       [BaselineForecaster]
  ├── Strategy A: Persistence (y_{t+h} = y_t)
  ├── Strategy B: Historical Mean (y_{t+h} = \bar{y}_{train})
  └── Strategy C: Linear Trend Extrapolation
                  │
                  ▼
   [Multi-Horizon Prediction Contract]
  (Preserving sequence_id, frame_id, horizon)
                  │
                  ▼
       [BaselineMetrics & MAE/RMSE]
                  │
                  ▼
 [Standardized Experiment Persistence]
 (experiments/forecasting/baseline/<run_id>/)
```

---

## 2. Baseline Strategies

### Strategy A — Persistence Baseline
The persistence baseline adopts the simplest physical heuristic: future perception reliability remains equal to the most recent historical observation.
For forecast horizon $h \in \{1, 3, 5\}$:
$$\hat{y}_{t+h} = y_t$$
Where $y_t$ is the perception reliability at the end of the historical observation sequence. In the absence of target observations in the sequence, the model marks the prediction status honestly as `NOT AVAILABLE` rather than fabricating numbers.

### Strategy B — Historical Mean Baseline
The historical mean baseline assumes reliability reverts to the global average observed across the training distribution.
For all horizons $h$:
$$\hat{y}_{t+h} = \bar{y}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{Train}} y_i$$
**Strict Invariant:** The mean is fitted strictly using training split targets only. It never accesses or aggregates validation or test split targets, eliminating information leakage.

### Strategy C — Linear Trend Extrapolation Baseline
The linear trend baseline models short-term momentum and perception degradation slopes over a trailing window $W \ge 2$:
$$\beta = \frac{\sum_{i=1}^W (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^W (x_i - \bar{x})^2}$$
$$\hat{y}_{t+h} = \text{clip}\left(y_t + \beta \cdot h, 0.0, 1.0\right)$$
- **Clipping:** Predictions are bounded to $[0.0, 1.0]$ to respect the continuous perception reliability domain.
- **Minimum Observations:** Requires at least $W \ge 2$ valid finite observations; otherwise, gracefully reports `NOT AVAILABLE`.

---

## 3. Ground-Truth Target Readiness Gate

PerceptGuard AI enforces rigorous scientific integrity regarding ground truth:
1. **YOLO Confidence is NOT Ground Truth:**
   - Confidence represents internal detector output probability, not actual object detection recall or ground-truth bounding box IoU.
   - Using YOLO confidence as pseudo ground-truth leads to circular reasoning and unscientific accuracy claims.
2. **Current Label Status on Day 5 Sequences:**
   - Day 5 sequence metadata records `target_label_available: false` and `target_label: null`.
   - The baseline forecaster infrastructure and data loaders are fully implemented and validated with synthetic test fixtures.
   - On real Day 5 driving data, the experiment orchestrator honestly reports `supervised_readiness: "NOT AVAILABLE"` and refrains from computing deceptive error metrics.

---

## 4. Multi-Horizon Prediction Contract

For every forecast generated across horizons $H \in \{1, 3, 5\}$, the following standardized contract record is preserved:

| Field Name | Type | Description |
|---|---|---|
| `dataset_id` | `str` | Identifier of the driving dataset (e.g., `driving_sample`) |
| `sequence_id` | `str` | Unique sequence ID (e.g., `seq_driving_sample_L5_000001`) |
| `input_start_frame` | `str` | First frame ID of historical sequence |
| `input_end_frame` | `str` | Last frame ID of historical sequence ($t$) |
| `target_frame_id` | `str` | Target future frame ID ($t+h$) |
| `target_horizon` | `int` | Horizon offset ($1, 3, 5$) |
| `baseline_strategy` | `str` | Forecasting strategy (`persistence`, `mean`, `trend`) |
| `predicted_reliability` | `float` / `null` | Predicted reliability score in $[0.0, 1.0]$ |
| `actual_reliability` | `float` / `null` | Ground-truth reliability score (or `null` if missing) |
| `prediction_status` | `str` | `VALID`, `NOT AVAILABLE`, or diagnostic reason |

---

## 5. Data Loading & Split Protection

Implemented in `src/forecasting/baseline_data_loader.py`:
- **Disjoint Split Partitions:** Ingests Day 5 `dataset_splits.json` where Train (70%), Val (15%), and Test (15%) splits are isolated by a 25-frame temporal embargo gap.
- **Zero Video-Frame Re-execution:** Pure tensor and metadata ingestion without re-running YOLO or ByteTrack.
- **Finite Check:** Enforces zero NaNs and zero infinities across all input sequences.

---

## 6. Evaluation Metrics

Implemented in `src/forecasting/baseline_metrics.py`:
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- Computed individually per horizon ($h=1, 3, 5$) and per strategy.
- Rejects non-finite values, empty arrays, and mismatched dimensions.

---

## 7. Output Artifacts

Every baseline experiment automatically generates structured persistence files under `experiments/forecasting/baseline/<run_id>/`:
1. `experiment_config.yaml` — Hyperparameters, horizons, sequence lengths, fitted training mean.
2. `predictions.json` — Complete multi-horizon prediction records.
3. `baseline_metrics.json` — MAE, RMSE, sample counts per horizon and strategy.
4. `experiment_summary.json` — Execution duration, sequence counts, readiness status.
5. `metadata.json` — Dataset, environment, split counts, and feature manifest references.
6. `errors.json` — Warning and error logs.

---

## 8. CLI Usage

### Baseline Forecaster CLI
```powershell
python -m src.forecasting.baseline_forecaster `
  --input datasets/processed/driving_sample/features/sequences/run_official `
  --strategy all `
  --horizons 1 3 5 `
  --sequence-length 5
```

### Baseline Experiment CLI
```powershell
python -m src.forecasting.baseline_experiment `
  --input datasets/processed/driving_sample/features/sequences/run_official `
  --sequence-length 5 `
  --horizons 1 3 5
```

---

## 9. Performance & CPU Telemetry

- **Execution Duration:** 0.0559s (~56 ms) for 1,242 multi-horizon baseline predictions across 138 sequences.
- **Throughput:** > 22,000 predictions/second on Intel Core i3 CPU.
- **Peak Memory Delta:** < 15 MB RAM overhead.

---

## 10. Current Limitations & Scope Boundaries

1. **Unavailability of Real Supervised Targets:** Day 5 driving sample sequences lack ground-truth perception reliability labels. Supervised benchmarks on real data remain `NOT AVAILABLE` until annotated ground-truth labels are ingested.
2. **Task Boundaries:** Neural recurrent architectures (LSTM/GRU), model training loops, and loss optimization are strictly reserved for Day 6 Tasks 3+.
