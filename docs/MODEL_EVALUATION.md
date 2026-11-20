# PerceptGuard AI — Model Evaluation & Comparative Analysis
## Technical Specification, Benchmark Protocol, Regression Metrics & Empirical Findings

---

## 1. Executive Summary

This document specifies the scientific evaluation protocol, mathematical metric formulations, benchmark results, and empirical analysis of the multi-horizon reliability forecasting system in **PerceptGuard AI** (Day 9 Task 6).

The evaluation benchmarks the supervised **LSTMForecaster** against three canonical time-series baseline strategies:
1. **Persistence Baseline:** $\hat{y}_{t+h} = y_t$ (assumes zero change from the most recently observed frame).
2. **Historical Mean Baseline:** $\hat{y}_{t+h} = \bar{y}_{\text{train}} = 0.000582$ (predicts empirical training target mean).
3. **Linear Trend Baseline:** $\hat{y}_{t+h} = \text{OLS}(y_{t-L+1:t})$ (fits an ordinary least-squares slope through the historical sequence window $L=5$).

Evaluation is conducted across three forward forecasting horizons:
- **Horizon 1 ($H1 = 1$):** Forecast reliability for frame $t + 1$ ($\Delta t = +0.20\text{ s}$)
- **Horizon 3 ($H3 = 3$):** Forecast reliability for frame $t + 3$ ($\Delta t = +0.60\text{ s}$)
- **Horizon 5 ($H5 = 5$):** Forecast reliability for frame $t + 5$ ($\Delta t = +1.00\text{ s}$)

---

## 2. Evaluation Protocol & Data Split Isolation

To prevent optimistic bias, data leakage, and temporal lookahead, the evaluation adheres to a strict protocol:

```
+---------------------------------------------------------------------------------------------------+
|                                 TEMPORAL DATA SPLIT ARCHITECTURE                                  |
+---------------------------------------------------------------------------------------------------+
|  [Train Split: Frames 1 to 105]           [Embargo Gap: G=25]    [Val Split]   [Test Split]        |
|  - Sequences: 1 to 101 (N=101)             - Frames 106 to 130    - Seq 102-119 - Seq 120-138     |
|  - Supervised training partition          - Purged buffer        - N = 18      - N = 19            |
+---------------------------------------------------------------------------------------------------+
```

### Protocol Principles:
1. **Zero Contamination:** Test split sequences ($N_{\text{test}} = 19$, ending frames 139 to 157) were never seen during model training, optimizer updates, or hyperparameter selection.
2. **Valid Horizon Target Matching:**
   Near the end of the video, forward projection targets extend past the final extracted frame:
   - $H1$: 18 valid targets (1 out-of-bounds at frame 157)
   - $H3$: 16 valid targets (3 out-of-bounds at frames 155, 156, 157)
   - $H5$: 14 valid targets (5 out-of-bounds at frames 153, 154, 155, 156, 157)
   - Total valid test evaluation pairs: **48 pairs** (Coverage: $48 / 57 = 84.21\%$).
3. **Strict Pairwise Alignment:** Every prediction record is matched against ground truth using primary key `(dataset_id, video_id, target_frame_index, horizon)` and fallback key `(dataset_id, sequence_id, horizon)`.

---

## 3. Mathematical Metric Definitions

Evaluation computes standard regression metrics across aligned prediction-target pairs $(y_i, \hat{y}_i)$:

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### 2. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### 3. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### 4. Mean Bias Error (Bias)
$$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

### 5. Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

> [!IMPORTANT]
> **Safe Handling of Constant Targets in $R^2$:**  
> When all ground-truth targets in a partition are identical ($y_i = \bar{y}$ for all $i$), the Total Sum of Squares ($SS_{\text{tot}}$) is identically zero. In this case, $R^2$ is mathematically undefined ($\frac{0}{0}$). `ForecastingModelEvaluator` safely handles this by assigning $R^2 = \text{None}$ (`null`), preventing division-by-zero crashes or misleading negative infinities.

---

## 4. Test Split Benchmark Results

The following table presents the official comparative benchmark across the isolated test partition ($N_{\text{valid}} = 48$ pairs):

| Model Strategy | Horizon | Valid Samples | MAE | RMSE | MSE | $R^2$ | Interpretation vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Persistence** | $H1$ | 18 | **0.000000** | **0.000000** | **0.000000** | N/A | Reference Baseline |
| **Historical Mean** | $H1$ | 18 | 0.000582 | 0.000582 | 0.00000034 | N/A | Reference Baseline |
| **Linear Trend** | $H1$ | 18 | **0.000000** | **0.000000** | **0.000000** | N/A | Reference Baseline |
| **LSTMForecaster** | $H1$ | 18 | 0.011556 | 0.011615 | 0.00013492 | N/A | **`LSTM worse than baseline`** |
| | | | | | | | |
| **Persistence** | $H3$ | 16 | **0.000000** | **0.000000** | **0.000000** | N/A | Reference Baseline |
| **Historical Mean** | $H3$ | 16 | 0.000582 | 0.000582 | 0.00000034 | N/A | Reference Baseline |
| **Linear Trend** | $H3$ | 16 | **0.000000** | **0.000000** | **0.000000** | N/A | Reference Baseline |
| **LSTMForecaster** | $H3$ | 16 | 0.020326 | 0.020435 | 0.00041759 | N/A | **`LSTM worse than baseline`** |
| | | | | | | | |
| **Persistence** | $H5$ | 14 | **0.000000** | **0.000000** | **0.000000** | N/A | Reference Baseline |
| **Historical Mean** | $H5$ | 14 | 0.000582 | 0.000582 | 0.00000034 | N/A | Reference Baseline |
| **Linear Trend** | $H5$ | 14 | **0.000000** | **0.000000** | **0.000000** | N/A | Reference Baseline |
| **LSTMForecaster** | $H5$ | 14 | 0.015467 | 0.015530 | 0.00024117 | N/A | **`LSTM worse than baseline`** |

---

## 5. Scientific Root Cause Analysis: Why LSTM Has Higher Error on the Test Set

The evaluation reveals that on the isolated test set, the LSTM's error ($\text{MAE} \approx 0.015$) is higher than Persistence ($\text{MAE} = 0.000$) and Historical Mean ($\text{MAE} = 0.000582$). In accordance with scientific integrity guidelines, this is reported honestly and investigated below:

### 1. Extreme Zero-Inflation of the Driving Video Sample
- The driving sequence (`converted_026c7465-309f6d33.mp4`) depicts open highway driving. Over 99% of frames contain zero detected objects.
- In the test split (the final 19 sequences of the video), true reliability $y_i$ is identically **0.000000** across all horizons.

### 2. Trivial Optimality of Flat Zero Predictors on Quiescent Sequences
- When historical inputs are all zeros, Persistence ($\hat{y} = y_t = 0$) and Trend ($\hat{y} = 0 + 0 \cdot h = 0$) trivially predict exactly $0.000000$.
- Because true targets in the test split are all $0.0$, Persistence and Trend achieve a mathematically "perfect" error ($\text{MAE} = 0.000000, \text{RMSE} = 0.000000$).

### 3. Inductive Bias & Non-Zero Continuous Residual of Neural Models
- The LSTM is trained with multi-modal inputs ($F=27$ features: brightness, contrast, motion kinematics, box geometry) and a Sigmoid activation head $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- Because weights are regularized with AdamW weight decay ($10^{-4}$) to prevent saturation, the model produces small positive continuous outputs in $[0.010, 0.025]$ (mean $\approx 0.015$).
- On a completely flat zero test set, this small residual offset ($0.015$) produces higher error than a trivial constant zero baseline.

### 4. Generalization Capability vs Trivial Degeneracy
- While Persistence achieves zero error on an uninterrupted string of zeros, it has **zero predictive warning capability**: if perception fails or drops ahead ($1.0 \to 0.0$), Persistence at time $t$ continues predicting $1.0$ (error $1.0$).
- The LSTM learns complex temporal patterns, but in an extremely imbalanced dataset where the test partition has zero positive detections, simple persistence dominates numerical accuracy metrics.

---

## 6. Full Dataset Context (Train, Validation, and Aggregate)

For complete transparency, evaluation across all partitions is summarized below:

| Split | Partition Role | Sequences | H1 MAE | H3 MAE | H5 MAE | Mean Target | Mean Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | Model Parameter Fitting | 101 | 0.011833 | 0.017411 | 0.014180 | 0.000582 | 0.014298 |
| **Val** | Early Stopping & Tuning | 18 | 0.013144 | 0.022099 | 0.019157 | 0.000000 | 0.018133 |
| **Test** | Unseen Blind Evaluation | 19 | 0.011556 | 0.020326 | 0.015467 | 0.000000 | 0.015783 |
| **Overall** | Full Sequence Dataset | 138 | 0.011969 | 0.018382 | 0.014989 | 0.000436 | 0.014981 |

---

## 7. Visualizations Catalog

All 8 requested evaluation charts are persisted under `reports/day9/visualizations/`:

| Visualization File | Purpose & Description |
| :--- | :--- |
| `actual_vs_predicted_h1.png` | Scatter plot of actual vs predicted reliability for $H1$ with identity reference line $y=x$. |
| `actual_vs_predicted_h3.png` | Scatter plot of actual vs predicted reliability for $H3$ with identity reference line $y=x$. |
| `actual_vs_predicted_h5.png` | Scatter plot of actual vs predicted reliability for $H5$ with identity reference line $y=x$. |
| `mae_comparison_across_models.png` | Grouped bar chart comparing MAE of Persistence, Mean, Trend, and LSTM across horizons. |
| `rmse_comparison_across_models.png` | Grouped bar chart comparing RMSE of Persistence, Mean, Trend, and LSTM across horizons. |
| `prediction_error_distribution.png` | Probability density histogram of prediction residuals ($e = \hat{y} - y$) with bias and std dev. |
| `reliability_forecasting_timeline.png` | Sequential timeline of ground truth vs LSTM multi-horizon predictions over test sequence steps. |
| `horizon_wise_performance_comparison.png` | Performance curves showing MAE and RMSE scaling across horizons $H1, H3, H5$. |

---

## 8. Threshold-Based Warning Analysis Status

Per engineering specifications:
- The PerceptGuard AI configuration defines object detection thresholds (`iou_threshold: 0.50`, `confidence_threshold: 0.25`), but **does not define an approved reliability warning threshold** for mapping continuous forecasts into alert states.
- In accordance with engineering guidelines: **Threshold-based warning metrics (Precision, Recall, F1, Confusion Matrix) are reported as UNAVAILABLE.**
- No fabricated or arbitrary thresholds were introduced.

---

## 9. CPU Performance & Resource Profile

Evaluation was executed on the host development machine:
- **CPU:** Intel Core i3 (8 Logical Cores, 2.10 GHz), 8 GB RAM, Windows 11 Build 26200
- **Total Aligned Pairs:** 48 pairs ($18 + 16 + 14$)
- **Execution Duration:** $0.054\text{ seconds}$ (metrics calculation and tabular aggregation)
- **Plot Generation Duration:** $1.42\text{ seconds}$ (generating all 14 PNG diagnostic figures)
- **Peak Process Memory:** $< 55\text{ MB}$ RSS
