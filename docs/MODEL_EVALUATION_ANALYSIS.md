# PerceptGuard AI — Model Evaluation & Analysis

## 1. Executive Summary

The **Model Evaluation & Analysis** module (`src/forecasting/model_evaluator.py` and `src/forecasting/evaluation_visualizer.py`) provides production-grade, mathematically rigorous, and scientifically honest evaluation of multi-horizon perception reliability forecasting models in PerceptGuard AI.

Operating across forward forecasting horizons $h \in \{1, 3, 5\}$ frames, the module establishes strict target integrity validation, temporal pair alignment, regression error computation, baseline model benchmarking, statistical residual distribution breakdown, and publication-ready diagnostic visualizations.

```
                              EVALUATION PIPELINE
                                
  Multi-Horizon Predictions                   Ground-Truth Observations
  [predictions.json]                          [ground_truth.json]
          │                                           │
          ▼                                           ▼
┌──────────────────┐                         ┌──────────────────┐
│ Schema & Horizon │                         │ Target Integrity │
│ Parsing (h=1,3,5)│                         │ & Bound Checking │
└─────────┬────────┘                         └────────┬─────────┘
          │                                           │
          └───────────────────┬───────────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Temporal Pair Alignment │
                 │ (dataset, video, frame, │
                 │  horizon, test split)   │
                 └────────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ Metric Computation      │       │ Baseline Benchmarking   │
│ - MAE, RMSE, Bias       │       │ - Persistence Baseline  │
│ - Median AE, Error Std  │       │ - Historical Mean       │
│ - Directional Counts    │       │ - Linear Trend          │
└────────────┬────────────┘       └────────────┬────────────┘
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │  Error Analysis & Stats │
                 │  - Percentiles (p5-p95) │
                 │  - Worst Cases          │
                 │  - Horizon Breakdown    │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │  Diagnostic Plots &     │
                 │  Structured Artifacts   │
                 └─────────────────────────┘
```

---

## 2. Target Integrity & Ground-Truth Safeguards

### 2.1 Critical Scientific Principle: YOLO Confidence is NOT Reliability
A core requirement of PerceptGuard AI is uncompromising scientific honesty:
- **YOLO Confidence** ($\hat{c} \in [0, 1]$) represents the detector's internal softmax output for a bounding box. It reflects what the network *thinks*, which is frequently overconfident under visual degradation, out-of-distribution blur, rain, occlusions, and sensor noise.
- **True Reliability** ($y \in [0, 1]$) is an objective ground-truth measure of detection success (e.g., true positive intersection over union $\text{IoU} \ge 0.5$ against human-verified bounding box annotations, tracking persistence, and absence of false alarms).

### 2.2 Target Integrity Enforcement
The evaluator actively inspects ground-truth metadata and observation payloads:
```python
target_def = str(meta.get("target_definition", "")).lower()
if "yolo" in target_def or "confidence" in target_def or "heuristic" in target_def:
    raise ModelEvaluationError(
        f"CRITICAL TARGET INTEGRITY VIOLATION: Target definition '{target_def}' uses "
        "YOLO confidence or heuristic proxies. Ground truth must represent true object "
        "detection reliability."
    )
```

If real video driving artifacts lack validated ground-truth reliability targets (`target_labels_available: false`), the evaluator marks the evaluation status as **`NOT AVAILABLE`** and exits cleanly without generating fabricated or misleading metrics.

---

## 3. Mathematical Metric Definitions

For a set of $N$ aligned evaluation pairs $\{(\hat{y}_i, y_i)\}_{i=1}^N$ at forecast horizon $h \in \{1, 3, 5\}$:

### 3.1 Error and Absolute Error
- **Signed Error:**
  $$e_i = \hat{y}_i - y_i$$
- **Absolute Error:**
  $$|e_i| = |\hat{y}_i - y_i|$$

### 3.2 Mean Absolute Error (MAE)
Quantifies average magnitude of errors regardless of direction:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$

### 3.3 Root Mean Squared Error (RMSE)
Penalizes large outlier forecast errors quadratically:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$

### 3.4 Forecast Bias (Mean Error)
Measures systematic tendency toward overprediction ($\text{Bias} > 0$) or underprediction ($\text{Bias} < 0$):
$$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

### 3.5 Median Absolute Error (MedAE)
Robust non-parametric measure of error dispersion resilient to extreme anomalies:
$$\text{MedAE} = \text{median}(|e_1|, |e_2|, \ldots, |e_N|)$$

### 3.6 Error Standard Deviation
Quantifies the spread and variability of forecast deviations around the mean bias:
$$\sigma_e = \sqrt{\frac{1}{N} \sum_{i=1}^N (e_i - \bar{e})^2}$$

### 3.7 Directionality Counts
Tracks operational failure dynamics:
- **Underpredictions:** Count where $e_i < -10^{-6}$ (model overestimated risk or underestimated reliability).
- **Overpredictions:** Count where $e_i > +10^{-6}$ (dangerous false sense of safety; model overestimated detector reliability).
- **Exact Predictions:** Count where $|e_i| \le 10^{-6}$.

---

## 4. Multi-Horizon Temporal Alignment

To prevent temporal leakage and mismatch errors, predictions and ground-truth observations are aligned strictly across composite keys:
- **Primary Match Key:** `(dataset_id, video_id, target_frame_index, horizon)`
- **Secondary Match Key:** `(dataset_id, sequence_id, horizon)`

### 4.1 Split Isolation
Evaluation defaults to `split="test"` to ensure no training or validation observations corrupt benchmark metrics. Passing `--split all` allows full-dataset diagnostic reviews.

### 4.2 Excluded Records Accounting
All anomalies are quarantined into `excluded_records.json` with descriptive reasons:
- `UNMATCHED_PREDICTION`: Prediction produced without matching ground-truth target.
- `UNMATCHED_GROUND_TRUTH`: Target observation without matching model prediction.
- `DUPLICATE_PREDICTION`: Multiple predictions targeting the identical frame and horizon.
- `DUPLICATE_GROUND_TRUTH`: Conflicting duplicate annotations for the same timestamp.
- `NON_FINITE_VALUE`: Encountered `NaN`, `Inf`, or unphysical value.

---

## 5. Baseline Comparison Protocol

The evaluator benchmarks the trained `LSTMForecaster` directly against Day 6 Task 2 baselines:
1. **Persistence Forecaster:** Assumes reliability at $t+h$ equals observed reliability at $t$.
2. **Historical Mean Forecaster:** Assumes reliability at $t+h$ equals historical sequence mean.
3. **Linear Trend Forecaster:** Extrapolates reliability via ordinary least-squares linear trend.

### Relative Percentage Improvement
For each horizon $h \in \{1, 3, 5\}$ and baseline strategy $B$:
$$\Delta_{\text{MAE}} (\%) = \left(\frac{\text{MAE}_B - \text{MAE}_{\text{LSTM}}}{\text{MAE}_B}\right) \times 100\%$$

---

## 6. Diagnostic Visualization Suite

The visualizer (`ForecastingEvaluationVisualizer`) runs in headless CPU environments using `matplotlib.use("Agg")` and generates 6 publication-grade figures:

| File Name | Chart Type | Key Diagnostic Insight |
| :--- | :--- | :--- |
| `actual_vs_predicted_reliability.png` | Scatter Plot + $y=x$ Line | Overall calibration, correlation ($r$), dispersion |
| `mae_by_horizon.png` | Bar Chart | Error escalation across forward frames (1f, 3f, 5f) |
| `rmse_by_horizon.png` | Bar Chart | Outlier error penalty scaling over forecast horizon |
| `prediction_error_distribution.png` | Histogram + KDE | Bias, normality, skewness, percentile indicators |
| `baseline_vs_lstm_comparison.png` | Grouped Bar Chart | Performance advantage of LSTM over persistence/mean/trend |
| `residual_analysis.png` | Residual Scatter ($e$ vs $\hat{y}$) | Homoscedasticity, non-linear error bias |

---

## 7. Artifact Directory Structure

Structured evaluation results are written to `artifacts/evaluation/`:
```
artifacts/evaluation/
├── evaluation_summary.json         # Executive summary, status, metadata
├── evaluation_results.json         # List of aligned evaluation pairs
├── horizon_1_metrics.json          # Horizon 1 detailed metrics
├── horizon_3_metrics.json          # Horizon 3 detailed metrics
├── horizon_5_metrics.json          # Horizon 5 detailed metrics
├── baseline_comparison.json        # Comparative benchmarking records
├── error_analysis.json             # Error percentiles, worst cases, breakdowns
├── excluded_records.json           # Quarantined unmatched/duplicate records
└── plots/                          # Publication-grade diagnostic charts
    ├── actual_vs_predicted_reliability.png
    ├── mae_by_horizon.png
    ├── rmse_by_horizon.png
    ├── prediction_error_distribution.png
    ├── baseline_vs_lstm_comparison.png
    └── residual_analysis.png
```

---

## 8. CLI Usage Guide

### 8.1 Standard Evaluation on Test Split
```bash
python -m src.forecasting.model_evaluator \
    --predictions artifacts/predictions/multi_horizon_predictions.json \
    --ground-truth artifacts/ground_truth/ground_truth.json \
    --baseline-results artifacts/baseline_forecasting/baseline_experiment_results.json \
    --split test \
    --generate-plots
```

### 8.2 Real Data Gate Check
When executed against real Day 5 driving sequences that lack ground truth:
```bash
python -m src.forecasting.model_evaluator \
    --ground-truth artifacts/day5_features/temporal_sequences.json
```
**Output:**
```
[GATE CHECK] Ground truth validation status: NOT_AVAILABLE
[REASON] Ground-truth target labels are marked as not available in metadata.
[STATUS] Ground truth is marked NOT AVAILABLE. Clean exit.
```

---

## 9. Verification & Test Coverage

The evaluation and visualization suite is tested by 51 dedicated automated unit and integration tests across:
- `tests/forecasting/test_model_evaluator.py` (35 tests)
- `tests/forecasting/test_evaluation_visualizer.py` (16 tests)

Combined with all prior tasks:
- **Total Forecasting Tests:** 277 passing tests
- **Full Repository Tests:** 1,169 passing tests (100% pass rate)
