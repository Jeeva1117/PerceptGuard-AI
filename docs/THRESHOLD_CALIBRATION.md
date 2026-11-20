# Perception Failure Risk Threshold Calibration

## 1. Executive Summary & Calibration Gate Status

This document defines the mathematical methodology, empirical validation procedure, and scientific integrity constraints for calibrating operational risk thresholds in PerceptGuard AI.

> [!WARNING]
> **CALIBRATION STATUS: PROVISIONAL**
> In accordance with strict scientific validation standards, the threshold boundaries for mapping continuous perception failure risk scores into discrete operational tiers are designated as **`PROVISIONAL`**.
>
> **Reason:** Inspection of the ground-truth reliability targets in the validation split (`datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json` with `split == "val"`, $N=18$ sequences, 54 multi-horizon target pairs) confirmed **100% zero-inflation** (all $y_i = 0.0$ across $H1, H3, H5$). Because the ground-truth distribution is degenerate (single-class: 100% failure outcomes under $y < 0.50$), true negative counts and false positive rates cannot be computed. Under these conditions, statistical multi-class cutoff optimization (such as ROC-AUC curve analysis or Youden's $J$ index) is mathematically degenerate. Claiming a fully calibrated production threshold would constitute false reassurance and violate engineering ethics.

---

## 2. Scientific Integrity & Anti-Leakage Constraints

The threshold calibration engine (`src/risk/threshold_calibrator.py`) enforces strict architectural firewalls:

1. **Strict Split Isolation:**
   - Threshold calibration **MUST ONLY** be performed using the **validation split** (`split == "val"`).
   - Attempting to run calibration against the test split (`split == "test"`) immediately raises `CalibrationError("Violation of scientific methodology: Test split cannot be used for threshold calibration!")`.
   - Attempting to run calibration against training data (`split == "train"`) immediately raises `CalibrationError("Violation of scientific methodology: Training split cannot be used for threshold calibration!")`.
   - Test data remains strictly isolated for unbiased post-deployment evaluation.
2. **Deterministic Reproducibility:**
   - Candidate threshold evaluation grids are evaluated systematically across candidate cutoffs ($0.10$ to $0.90$ with step size $0.05$).
   - Confusion matrices ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$), Precision, Recall, Accuracy, and $F_1$-score are recorded for every candidate cutoff.
3. **No Numerical Guesswork:**
   - If a dataset is single-class or uninformative, the system transparently reports status as `PROVISIONAL` and issues explicit diagnostic warnings.

---

## 3. Empirical Validation Split Analysis

Calibration was executed against the official PerceptGuard AI validation partition:
- **Dataset Partition:** `driving_sample` (Validation Split, $L=5$)
- **Sequence Count:** 18 sequences
- **Evaluation Points:** 54 horizon predictions ($18 \times 3$ for $H1, H3, H5$)
- **Ground-Truth Failure Rate:** $100\%$ ($54 / 54$ targets with $R_t \le 0.00$)
- **Model Inferences:** Official LSTM multi-horizon forecasts (`experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json`)

### Risk Score Distribution on Validation Split

| Metric | Empirical Value | Description |
| :--- | :--- | :--- |
| **Count** | 54 | Total horizon evaluation pairs ($18 \text{ sequences} \times 3 \text{ horizons}$) |
| **Min Risk Score** | $0.960631$ | Lowest predicted risk score in validation split |
| **P25 (25th percentile)** | $0.978168$ | First quartile |
| **Median (P50)** | $0.983944$ | Median predicted failure risk score |
| **Mean Risk Score** | $0.981867$ | Mean predicted failure risk score |
| **P75 (75th percentile)** | $0.986851$ | Third quartile |
| **P90 (90th percentile)** | $0.988167$ | Ninetieth percentile |
| **Max Risk Score** | $0.989719$ | Peak predicted failure risk score |
| **Standard Deviation** | $0.006935$ | Tightly clustered high-risk predictions |

The LSTM forecaster appropriately recognized the low detection reliability across the validation sample, forecasting high failure risks ($0.96 - 0.99$).

---

## 4. Threshold Profile Specification

Under the provisional calibration profile, operational cutoffs are defined as:

| Boundary Parameter | Provisional Value | Operational Tier |
| :--- | :--- | :--- |
| `low_upper` ($\theta_{\text{low}}$) | $0.30$ | $[0.00, 0.30] \implies \text{LOW}$ |
| `medium_upper` ($\theta_{\text{med}}$) | $0.60$ | $(0.30, 0.60] \implies \text{MEDIUM}$ |
| `high_upper` ($\theta_{\text{high}}$) | $0.85$ | $(0.60, 0.85] \implies \text{HIGH}$ |
| Peak Tier | $> 0.85$ | $(0.85, 1.00] \implies \text{CRITICAL}$ |

### Candidate Cutoff Grid Evaluation (Validation Split)

Because 100% of samples are ground-truth failures, any threshold $\theta \le 0.96$ captures 100% of failures ($\text{Recall} = 1.0$, $\text{TP} = 54, \text{FN} = 0$).

| Candidate Threshold ($\theta$) | TP | FP | TN | FN | Precision | Recall | $F_1$ Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.10$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.30$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.50$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.60$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.70$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.85$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.90$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| $0.95$ | 54 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

*Diagnostic Note: Because true negatives cannot occur in a single-class positive dataset ($\text{TN}=0, \text{FP}=0$), specificity cannot be calculated and precision is trivially 1.0. Full calibration will be executed once multi-video BDD100K sequences containing diverse nominal and failure conditions are ingested.*

---

## 5. Roadmap to Full Production Calibration

1. **Multi-Video BDD100K Ingestion:** Ingest validation sequences spanning day, night, adverse weather, highway, and urban scenes to achieve balanced distribution of $R_t \in [0.0, 1.0]$.
2. **False Alarm vs Miss Trade-off Analysis:** Plot ROC-AUC and Precision-Recall curves to tune $\theta_{\text{med}}$ and $\theta_{\text{high}}$ for maximum driver safety while maintaining false alarm rates $< 5\%$.
3. **Horizon-Specific Retuning:** Tune $H1, H3, H5$ independently to match vehicle braking dynamics ($H1$) vs advisory advance notices ($H5$).
