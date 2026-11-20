# Temporal Risk Monitoring Data Contract

## 1. Scope & System Role

This document specifies the interface contracts, data schemas, validation invariants, and error policies for the **Temporal Risk Monitoring Engine** (`src/risk/temporal_risk_monitor.py`).

---

## 2. Input Contracts

`TemporalRiskMonitor.update(...)` accepts polymorphic representations of risk evaluations:

### 2.1 From Task 2: `HorizonRiskRecord`
```python
@dataclass
class HorizonRiskRecord:
    prediction_id: str
    sequence_id: str
    source_frame_id: str
    target_frame_id: str
    horizon: int
    predicted_reliability: float
    risk_score: float  # In [0.0, 1.0]
    source_frame_index: Optional[int] = None
    target_frame_index: Optional[int] = None
    target_timestamp_seconds: Optional[float] = None
    track_id: Optional[int] = None
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "unknown"
```

### 2.2 From Task 2: `AggregatedRiskRecord`
```python
@dataclass
class AggregatedRiskRecord:
    sequence_id: str
    source_frame_id: str
    aggregated_risk_score: float  # In [0.0, 1.0]
    aggregation_strategy: str
    primary_horizon: int
    horizon_risk_scores: Dict[int, float]
    horizon_reliabilities: Dict[int, float]
    weights_applied: Dict[int, float]
    target_timestamp_seconds: Optional[float] = None
```

### 2.3 From Task 3: `ClassificationResult`
```python
@dataclass
class ClassificationResult:
    sequence_id: str
    risk_score: float
    risk_level: RiskLevel
    horizon: Optional[int] = None
    prediction_id: Optional[str] = None
    source_frame_id: Optional[str] = None
    target_frame_id: Optional[str] = None
    target_timestamp_seconds: Optional[float] = None
    calibration_status: str = "PROVISIONAL"
    boundary_policy: str = "closed_right"
```

---

## 3. Output Contracts

### 3.1 `MonitoredRiskRecord`
Represents an individual point in chronological history:

| Field | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | `str` | Video sequence identifier |
| `risk_score` | `float` | Continuous risk score in $[0.0, 1.0]$ or `NaN` |
| `risk_level` | `RiskLevel` | Enum: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, `UNKNOWN` |
| `timestamp_seconds` | `float` | Event time in video sequence |
| `horizon` | `Optional[int]` | Forecast horizon (1, 3, 5) or `None` for aggregated |
| `source_frame_id` | `str` | Input frame ID reference |
| `forecast_generation_timestamp` | `Optional[str]` | Wall-clock prediction generation timestamp |
| `calibration_status` | `str` | Status: `PROVISIONAL` or `CALIBRATED` |
| `checkpoint_id` | `str` | Model checkpoint identifier |
| `is_valid` | `bool` | True if prediction is valid; False if corrupted |
| `is_stale` | `bool` | True if prediction age exceeded freshness TTL |
| `is_out_of_order` | `bool` | True if record arrived after newer timestamp |
| `is_duplicate` | `bool` | True if record duplicated an existing timestamp |

### 3.2 `StateTransitionEvent`
Represents a discrete change in operational risk tier:

| Field | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | `str` | Video sequence identifier |
| `horizon` | `Optional[int]` | Forecast horizon (1, 3, 5, or None) |
| `previous_level` | `RiskLevel` | Prior operational tier |
| `current_level` | `RiskLevel` | New operational tier |
| `timestamp_seconds` | `float` | Event timestamp where transition occurred |
| `risk_score` | `float` | Current continuous risk score |
| `previous_risk_score` | `Optional[float]` | Prior continuous risk score |
| `is_genuine_transition` | `bool` | False if triggered by missing/stale telemetry |
| `calibration_status` | `str` | Calibration status (`PROVISIONAL`) |

### 3.3 `TemporalRiskSummary`
Comprehensive instantaneous snapshot:

| Field | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | `str` | Video sequence identifier |
| `horizon` | `Optional[int]` | Forecast horizon |
| `current_score` | `float` | Current risk score |
| `previous_score` | `Optional[float]` | Score from preceding frame |
| `absolute_change` | `Optional[float]` | $S_t - S_{t-1}$ |
| `relative_change` | `Optional[float]` | $(S_t - S_{t-1}) / S_{t-1}$ |
| `rolling_mean` | `float` | Mean of valid scores in window |
| `rolling_max` | `float` | Maximum score in window |
| `rolling_min` | `float` | Minimum score in window |
| `volatility` | `Optional[float]` | Sample standard deviation in window |
| `trend_direction` | `RiskTrendDirection` | `INCREASING`, `DECREASING`, `STABLE`, `UNKNOWN` |
| `consecutive_increasing_count` | `int` | Number of consecutive increasing frames |
| `current_risk_level` | `RiskLevel` | Current discrete level |
| `level_history` | `List[RiskLevel]` | History of discrete levels in window |
| `observation_count` | `int` | Count of records currently in window |
| `window_size` | `int` | Maximum capacity of sliding window |
| `is_stale` | `bool` | Staleness flag |
| `has_timestamp_gap` | `bool` | Continuity gap flag |

---

## 4. Configuration Schema (`TemporalMonitoringConfig`)

```python
@dataclass(frozen=True)
class TemporalMonitoringConfig:
    history_window_size: int = 30           # Sliding window capacity
    min_observations_for_trend: int = 2     # Minimum observations required for slope
    max_staleness_seconds: float = 1.0      # TTL before prediction is marked stale
    max_timestamp_gap_seconds: float = 1.5  # Gap threshold resetting consecutive counts
    trend_stability_threshold: float = 0.01 # Minimum |ΔS| for directional classification
    allow_out_of_order: bool = False        # Reject vs flag out-of-order records
    deduplication_policy: str = "reject"    # "reject", "ignore", "replace"
    track_state_transitions: bool = True    # Enable state transition logging
    alert_persistence_threshold: int = 3    # Frames of HIGH/CRITICAL for persistence
```
