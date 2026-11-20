# Temporal Risk Monitoring Engine

## 1. Executive Summary & Purpose

The **Temporal Risk Monitoring Engine** provides real-time, chronological tracking of multi-horizon perception reliability risks in PerceptGuard AI. Operating directly downstream of the Risk Scoring Engine (Task 2) and Risk Classification Engine (Task 3), the temporal monitor tracks risk evolution over sliding chronological windows, evaluates directional trends, detects operational state transitions, maintains history per video sequence and forecast horizon, and enforces zero-leakage temporal isolation.

The engine acts as the historical state aggregator and debouncing foundation for downstream Early Warning Alert Generation (Task 5).

---

## 2. Core Architecture & Architectural Boundaries

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

### Architectural Principles:
1. **Isolated Sequence Boundaries:**
   Each video sequence and forecast horizon is tracked in an independent `SequenceHorizonTracker`. History, rolling statistics, and consecutive counts never cross sequence or video boundaries.
2. **Event Time vs. Prediction Generation Time:**
   - **Event Time (`timestamp_seconds`):** Video or sensor acquisition timestamp representing the physical moment being evaluated.
   - **Prediction Generation Time (`forecast_generation_timestamp`):** System wall-clock timestamp of when the neural network inference executed.
3. **Zero Lookahead Bias:**
   All summary metrics, rolling statistics, and state transitions at time $T$ are computed strictly using observations at or prior to $T$.
4. **Safety Rule on Risk Changes:**
   A temporary increase in risk score indicates a deteriorating trend, but is **never interpreted as a confirmed perception failure**. Downstream early-warning components require temporal persistence before issuing driver-facing alerts.

---

## 3. Chronological Tracking & Data Integrity

The monitor guarantees chronological integrity through strict validation rules:

| Condition | Default Policy | System Action |
| :--- | :--- | :--- |
| **Normal Sequential Ingestion** | $t_k > t_{k-1}$ | Ingested, rolling window updated, delta computed. |
| **Duplicate Timestamp / Frame** | `reject` | Raises `TemporalMonitoringError`. (`ignore` and `replace` policies also available). |
| **Out-of-Order Timestamp** | `reject` | Raises `TemporalMonitoringError` if `allow_out_of_order=False`. If allowed, flagged with `is_out_of_order=True`. |
| **Large Timestamp Gap** | $\Delta t > 1.5\text{s}$ | Flagged with `has_timestamp_gap=True`; resets consecutive increasing count; delta treated as non-continuous. |
| **Non-Finite Risk Score** | Strictly invalid | Rejects `NaN`, `Inf`, $S < 0.0$, or $S > 1.0$ with `RiskValidationError`. |
| **Missing Telemetry** | Explicit Sentinel | Evaluates as `RiskLevel.UNKNOWN` with `NaN` score; **never defaults to zero risk**. |
| **Stale Prediction** | Flagged Sentinel | Evaluates as `is_stale=True`; excluded from genuine operational state transitions. |
| **Model Checkpoint Change** | Version Sentinel | Flags `checkpoint_change_detected=True` when model ID differs from previous frame. |

---

## 4. Trend Analysis & Statistical Metrics

For each valid observation $S_t$ ingested into a sequence tracker of window size $W$:

### 4.1 Instantaneous Change ($\Delta S$)
$$\Delta S = S_t - S_{t-1}$$
$$\text{Relative Change} = \frac{S_t - S_{t-1}}{S_{t-1}} \quad (\text{for } S_{t-1} > 10^{-4})$$

### 4.2 Directional Trend Classification
Under a configurable sensitivity threshold $\epsilon_{\text{trend}} = 0.01$:
$$\text{Trend}(S_t) = \begin{cases}
\text{INCREASING} & \Delta S > \epsilon_{\text{trend}} \\
\text{DECREASING} & \Delta S < -\epsilon_{\text{trend}} \\
\text{STABLE} & |\Delta S| \le \epsilon_{\text{trend}} \\
\text{UNKNOWN} & \text{First observation, missing telemetry, or large timestamp gap}
\end{cases}$$

### 4.3 Consecutive Increasing Count
$$\text{Count}_{\text{inc}}(t) = \begin{cases}
\text{Count}_{\text{inc}}(t-1) + 1 & \text{if Trend}(S_t) = \text{INCREASING} \\
0 & \text{otherwise (DECREASING, STABLE, or gap)}
\end{cases}$$

### 4.4 Rolling Window Statistics
For all valid scores $\{S_1, \dots, S_K\}$ in the active window ($K \le W$):
$$\bar{S} = \frac{1}{K}\sum_{i=1}^K S_i, \quad S_{\max} = \max_i S_i, \quad S_{\min} = \min_i S_i$$
$$\sigma_S = \sqrt{\frac{1}{K-1}\sum_{i=1}^K (S_i - \bar{S})^2} \quad (\text{for } K \ge 2)$$

---

## 5. State Transition Mechanism

State transitions between operational tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, `UNKNOWN`) are logged as structured `StateTransitionEvent` objects.

### Transition Rules:
1. **Change Required:** A transition event is logged **only when $L_t \neq L_{t-1}$**. If the risk tier remains identical across frames, no redundant transition event is emitted.
2. **Genuineness Check:** A transition is marked `is_genuine_transition = True` only if neither tier is `UNKNOWN` and the incoming frame is not stale. Transitions involving missing or stale telemetry are recorded with `is_genuine_transition = False` for auditability without triggering operational false alarms.
3. **Provisional Traceability:** All state transitions maintain the active `calibration_status` (`PROVISIONAL`).
