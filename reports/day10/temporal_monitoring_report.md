# PerceptGuard AI — Day 10 Task 4 Engineering Report
## Temporal Risk Monitoring Engine

**Date:** 2026-11-20  
**Phase:** Day 10 — Early Warning & Risk Assessment  
**Task:** Task 4 — Temporal Risk Monitoring Engine  
**Status:** PASS WITH LIMITATIONS  
**Calibration Gate Status:** PROVISIONAL  

---

## 1. Executive Summary & Readiness Gate

Task 4 implements the **Temporal Risk Monitoring Engine** for PerceptGuard AI. Building directly on the foundational risk scoring engine from Task 2 (`src/risk/risk_scorer.py`) and the risk tier categorization engine from Task 3 (`src/risk/risk_classifier.py`), this engine provides chronological risk tracking, trend direction estimation, rolling statistics, operational state-transition logging, and multi-horizon isolation across sequential video frames.

### Readiness Gate Verification:
1. **Task 2 Risk Score Availability:** Verified that `HorizonRiskRecord` and `AggregatedRiskRecord` provide mathematically bounded failure risk scores $S \in [0.0, 1.0]$.
2. **Task 3 Classification & Calibration Status:** Verified that `RiskClassifier` maps continuous scores into operational levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, `UNKNOWN`). Calibration status is verified and preserved as **`PROVISIONAL`** due to single-video zero-inflation.
3. **Timestamp & Frame Index Consistency:** Verified clear separation between event time (`timestamp_seconds`) and inference generation time (`forecast_generation_timestamp`).
4. **Identity & Traceability:** Sequence ID, prediction ID, and model checkpoint ID are strictly propagated.
5. **Fail-Safe Missing Telemetry Invariant:** Missing or stale data is strictly classified as `RiskLevel.UNKNOWN` and **never defaults to `LOW` or zero risk**.

---

## 2. Technical Architecture & Component Implementation

### 2.1 Module Layout

```
                      Multi-Horizon Risk Predictions
                 (HorizonRiskRecord / AggregatedRiskRecord)
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │        RiskClassifier       │
                      │  (LOW, MED, HIGH, CRITICAL) │
                      └──────────────┬──────────────┘
                                     │ Classified Observations
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         TemporalRiskMonitor                            │
│                                                                        │
│  Isolated Trackers: (sequence_id, horizon)                             │
│  ┌───────────────────────┐         ┌────────────────────────┐          │
│  │  Sequence 1 Tracker   │         │   Sequence 2 Tracker   │   ...    │
│  │  - H1 Deque (W=30)    │         │   - H1 Deque (W=30)    │          │
│  │  - H3 Deque (W=30)    │         │   - H3 Deque (W=30)    │          │
│  │  - H5 Deque (W=30)    │         │   - H5 Deque (W=30)    │          │
│  └───────────────────────┘         └────────────────────────┘          │
│                                                                        │
│  Incremental Calculations (O(1)):                                      │
│  - Instantaneous Delta: ΔS = S_t - S_{t-1}                             │
│  - Directional Trend: INCREASING, DECREASING, STABLE, UNKNOWN          │
│  - Rolling Statistics: Mean, Min, Max, Volatility (Std Dev)            │
│  - Consecutive Increasing Count                                        │
│  - State Transitions (Level_{t-1} -> Level_t)                          │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
                            TemporalRiskSummary
                    (Streamed to Downstream Alerting)
```

### 2.2 Implemented Components

| Component | File | Role |
| :--- | :--- | :--- |
| `TemporalMonitoringConfig` | `src/risk/risk_config.py` | Strongly typed immutable configuration for window size, trend stability threshold, gap thresholds, and deduplication policy. |
| `RiskTrendDirection` | `src/risk/temporal_risk_monitor.py` | Directional trend enum: `INCREASING`, `DECREASING`, `STABLE`, `UNKNOWN`. |
| `MonitoredRiskRecord` | `src/risk/temporal_risk_monitor.py` | Individual tracked timepoint preserving event time, generation time, scores, levels, and audit flags. |
| `StateTransitionEvent` | `src/risk/temporal_risk_monitor.py` | Structured event logged upon genuine operational tier changes ($L_t \neq L_{t-1}$). |
| `TemporalRiskSummary` | `src/risk/temporal_risk_monitor.py` | Instantaneous summary snapshot of rolling statistics, volatility, trend, and consecutive counters. |
| `SequenceHorizonTracker` | `src/risk/temporal_risk_monitor.py` | Isolated internal state engine maintaining sliding deques and incremental accumulators per $(S_{\text{id}}, H)$ pair. |
| `TemporalRiskMonitor` | `src/risk/temporal_risk_monitor.py` | Master monitoring engine implementing `ITemporalRiskMonitor` with multi-horizon routing and polymorphism. |

---

## 3. Mathematical & Algorithmic Formulations

### 3.1 Instantaneous Change & Trend Direction
For incoming score $S_t$ and prior valid score $S_{t-1}$:
$$\Delta S = S_t - S_{t-1}$$
$$\text{Relative Change} = \frac{S_t - S_{t-1}}{S_{t-1}} \quad (\text{for } S_{t-1} > 10^{-4})$$

Under sensitivity cutoff $\epsilon_{\text{trend}} = 0.01$:
$$\text{Trend}(S_t) = \begin{cases}
\text{INCREASING} & \Delta S > \epsilon_{\text{trend}} \\
\text{DECREASING} & \Delta S < -\epsilon_{\text{trend}} \\
\text{STABLE} & |\Delta S| \le \epsilon_{\text{trend}} \\
\text{UNKNOWN} & \text{First frame, missing data, or large gap}
\end{cases}$$

### 3.2 Consecutive Increasing Counter
$$\text{Count}_{\text{inc}}(t) = \begin{cases}
\text{Count}_{\text{inc}}(t-1) + 1 & \text{if Trend}(S_t) = \text{INCREASING} \\
0 & \text{otherwise (DECREASING, STABLE, or gap)}
\end{cases}$$

### 3.3 Rolling Window Statistics
For all valid scores $\{S_1, \dots, S_K\}$ in the active window ($K \le W$):
$$\bar{S} = \frac{1}{K}\sum_{i=1}^K S_i, \quad S_{\max} = \max_i S_i, \quad S_{\min} = \min_i S_i$$
$$\sigma_S = \sqrt{\frac{1}{K-1}\sum_{i=1}^K (S_i - \bar{S})^2} \quad (K \ge 2)$$

---

## 4. Chronological Integrity & Fail-Safe Invariants

1. **Chronological Ordering & Out-of-Order Rejection:**
   Observations must arrive with strictly increasing event timestamps ($t_k > t_{k-1}$). Out-of-order timestamps raise `TemporalMonitoringError` by default (or set `is_out_of_order=True` if configured).
2. **Duplicate Detection:**
   Identical timestamps or frame indices for the same sequence/horizon raise `TemporalMonitoringError` under the default `reject` policy.
3. **Cross-Sequence Isolation:**
   Each sequence tracker is completely sandboxed. History from `seq_1` never contaminates `seq_2`, even if timestamps overlap.
4. **Multi-Horizon Isolation:**
   Forecasts for $H1$, $H3$, and $H5$ are tracked independently. A jump in $H5$ risk score does not contaminate the rolling statistics or trend of $H1$.
5. **Zero Lookahead Bias:**
   Calculations at timestamp $T$ utilize strictly observations at or before $T$.
6. **Large Timestamp Gap Handling:**
   If $\Delta t > 1.5\text{s}$ (configurable), `has_timestamp_gap=True` is raised, consecutive counters are reset, and the delta is not treated as a continuous instantaneous trend.
7. **Model Checkpoint Drift:**
   Detects changes in `checkpoint_id` across frames, setting `checkpoint_change_detected=True`.
8. **Missing Telemetry Safety:**
   Missing or corrupted forecasts map to `RiskLevel.UNKNOWN` with `NaN` score; **never defaults to zero risk**.

---

## 5. Performance & Resource Benchmarks

Benchmarked on host hardware (**Intel Core i3, 8 GB RAM, Windows 11 AMD64, CPU-only, zero CUDA**):

| Benchmark Scenario | Input Size | Total Time | Latency per Update | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **Synthetic Micro-Benchmark** | 10,000 updates | $0.6553\text{ s}$ | **$65.53\ \mu\text{s}$** | $15,260\text{ ops/sec}$ |
| **Authentic Dataset Streaming** | 414 predictions | $0.0117\text{ s}$ | **$28.16\ \mu\text{s}$** | $35,510\text{ ops/sec}$ |

### Resource Consumption:
- **Frame Budget Overhead (30 FPS = 33.3 ms):** $0.028\text{ ms} / 33.3\text{ ms} \approx \mathbf{0.08\%}$ of available frame budget.
- **Memory Footprint:** $< 2\text{ MB}$ additional working set; lightweight deques prevent memory leaks.

---

## 6. Automated Testing Suite

Comprehensive unit and integration test suite authored in [tests/risk/test_temporal_risk_monitor.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/risk/test_temporal_risk_monitor.py):

| Test Category | Tests | Verification Scope | Status |
| :--- | :---: | :--- | :---: |
| **Chronological & Integrity** | 4 | Insertion ordering, out-of-order rejection, duplicate rejection & ignore policy. | PASS |
| **Isolation Guarantees** | 2 | Cross-sequence independence, multi-horizon independence. | PASS |
| **Rolling Stats & Trends** | 3 | Rolling mean, min, max, volatility, trend direction, consecutive increasing, eviction. | PASS |
| **State Transitions** | 2 | Tier transition logging, suppression of repeated identical transitions. | PASS |
| **Edge Cases & Fail-Safes** | 8 | Missing telemetry, stale flags, timestamp gaps, NaN rejection, version change, empty history, zero lookahead. | PASS |
| **Task 2 & 3 Integration** | 4 | Ingestion of `HorizonRiskRecord`, `AggregatedRiskRecord`, `ClassificationResult`, `record_and_check`. | PASS |
| **Authentic Dataset Streaming** | 1 | End-to-end streaming of 414 authentic predictions from `predictions.json`. | PASS |
| **Total Task 4 Tests** | **24 / 24** | **100% Pass Rate in 0.92 seconds** | **PASS** |

### Regression Suite Status:
- `tests/risk/`: 76 tests passing
- `tests/early_warning/`: 18 tests passing
- **Total Combined Day 10 Passing Tests:** **94 of 94 passing (100%)**

---

## 7. Audit Verdict & Project Boundary

$$\mathbf{AUDIT\ VERDICT:\ PASS\ WITH\ LIMITATIONS}$$
$$\mathbf{CALIBRATION\ STATUS:\ PROVISIONAL}$$

### Limitations & Known Context:
1. **Zero-Inflation Ground Truth:** The single driving clip dataset (`driving_sample`) contains 100% failure outcomes ($y=0.0$), causing predicted risk scores to cluster predominantly in the HIGH/CRITICAL tiers.
2. **Provisional Thresholds:** Operational boundaries remain marked as `PROVISIONAL` pending multi-video BDD100K evaluation.

### Scope Boundary Notice:
- **Task 4 is COMPLETE.**
- Task 5 (Early Warning Alert Generation & Threshold Suppression) has **NOT** been started.
- All development activity stops here awaiting review and instructions.
