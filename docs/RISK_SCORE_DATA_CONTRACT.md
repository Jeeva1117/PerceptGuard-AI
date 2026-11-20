# PerceptGuard AI — Risk Score Data Contract Specification

**Document Version:** 1.0.0  
**Phase:** Day 10 Task 2 — Risk Score Output Schemas  
**Status:** Approved  

---

## 1. Overview & Purpose

The **Risk Score Data Contract** defines the formal schemas and invariant constraints governing all risk scores produced by the `src/risk/` engine.

It establishes explicit data representations for:
1. **Single-Horizon Risk Records** (`HorizonRiskRecord`)
2. **Multi-Horizon Aggregated Risk Records** (`AggregatedRiskRecord`)
3. **Retrospective Temporal Risk Features** (`TemporalRiskFeatures`)

---

## 2. Single-Horizon Risk Record (`HorizonRiskRecord`)

| Field Name | Type | Constraint | Description |
| :--- | :--- | :--- | :--- |
| `prediction_id` | `str` | Non-empty string | Unique prediction identifier (`{sequence_id}_H{horizon}`) |
| `sequence_id` | `str` | Non-empty string | Source temporal sequence identifier |
| `source_frame_id` | `str` | Valid filename | Filename of the final observed history frame |
| `target_frame_id` | `str` | Valid filename | Filename of the forecasted future frame |
| `horizon` | `int` | $\in \{1, 3, 5\}$ | Forward forecast horizon |
| `predicted_reliability` | `float` | $\in [0.0, 1.0]$ | Input continuous predicted reliability score |
| `risk_score` | `float` | $\in [0.0, 1.0]$ | Derived perception failure risk ($1.0 - \hat{R}$) |
| `source_frame_index` | `Optional[int]` | $\ge 0$ | Zero-indexed source video frame position |
| `target_frame_index` | `Optional[int]` | $> \text{source\_frame\_index}$ | Zero-indexed target video frame position |
| `target_timestamp_seconds`| `Optional[float]` | $\ge 0.0$ | Physical video presentation timestamp |
| `track_id` | `Optional[int]` | $\ge 0$ | Object tracking identifier (if active) |
| `model_name` | `str` | String (`LSTMForecaster`) | Upstream forecasting model name |
| `checkpoint_id` | `str` | String | Model checkpoint identifier |
| `experiment_id` | `Optional[str]` | String | Experiment run identifier |
| `transformation_method` | `str` | `"linear_inverse"` | Method used to transform reliability to risk |
| `created_at` | `str` | ISO 8601 UTC | Creation timestamp |

---

## 3. Aggregated Risk Record (`AggregatedRiskRecord`)

| Field Name | Type | Constraint | Description |
| :--- | :--- | :--- | :--- |
| `sequence_id` | `str` | Non-empty string | Unique temporal sequence identifier |
| `source_frame_id` | `str` | Valid filename | Source frame reference |
| `aggregated_risk_score` | `float` | $\in [0.0, 1.0]$ | Synthesized scalar failure risk score |
| `aggregation_strategy` | `str` | Allowed strategy name | Strategy used (`weighted_linear`, `worst_case`, `primary_horizon`, `none`) |
| `primary_horizon` | `int` | $\in \{1, 3, 5\}$ | Primary horizon focus (default: 3) |
| `horizon_risk_scores` | `Dict[int, float]` | Values $\in [0.0, 1.0]$ | Dictionary of individual risk scores per horizon |
| `horizon_reliabilities` | `Dict[int, float]` | Values $\in [0.0, 1.0]$ | Dictionary of input reliability scores per horizon |
| `weights_applied` | `Dict[int, float]` | Sum $= 1.0 \pm 10^{-5}$ | Normalized weights applied during aggregation |
| `target_timestamp_seconds`| `Optional[float]` | $\ge 0.0$ | Video presentation timestamp |
| `source_frame_index` | `Optional[int]` | $\ge 0$ | Source frame position |
| `track_id` | `Optional[int]` | $\ge 0$ | Tracking identifier |
| `temporal_features` | `Optional[TemporalRiskFeatures]` | Valid features | Extracted retrospective temporal indicators |
| `is_calibrated_probability`| `bool` | Must be `False` | Explicit flag confirming score is not a calibrated probability |
| `model_name` | `str` | String | Forecasting model provenance |
| `checkpoint_id` | `str` | String | Model checkpoint provenance |
| `created_at` | `str` | ISO 8601 UTC | Creation timestamp |

---

## 4. Retrospective Temporal Risk Features (`TemporalRiskFeatures`)

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | `str` | Sequence identifier |
| `current_predicted_reliability` | `float` | Immediate reliability score ($\hat{R}_{H1}$) |
| `horizon_reliabilities` | `Dict[int, float]` | Reliabilities across $\{1, 3, 5\}$ |
| `cross_horizon_spread` | `Optional[float]` | $\hat{R}_{H1} - \hat{R}_{H5}$ |
| `is_monotonic_degradation` | `bool` | `True` if $\hat{R}_{H1} \ge \hat{R}_{H3} \ge \hat{R}_{H5}$ |
| `historical_trend_slope` | `Optional[float]` | OLS slope across retrospective window $W=5$ |
| `degradation_rate` | `Optional[float]` | Cross-horizon drop per frame: $(\hat{R}_{H1} - \hat{R}_{H5})/4$ |
| `recent_volatility` | `Optional[float]` | Standard deviation of recent reliability scores |
| `available_history_points` | `int` | Number of valid historical observations used |

---

## 5. Serialization Example (JSON)

```json
{
  "sequence_id": "seq_driving_sample_L5_000001",
  "source_frame_id": "frame_000005.jpg",
  "aggregated_risk_score": 0.981881,
  "aggregation_strategy": "weighted_linear",
  "primary_horizon": 3,
  "horizon_risk_scores": {
    "1": 0.985426,
    "3": 0.976163,
    "5": 0.981434
  },
  "horizon_reliabilities": {
    "1": 0.014574,
    "3": 0.023837,
    "5": 0.018566
  },
  "weights_applied": {
    "1": 0.5,
    "3": 0.35,
    "5": 0.15
  },
  "target_timestamp_seconds": 0.997,
  "source_frame_index": 24,
  "track_id": null,
  "is_calibrated_probability": false,
  "model_name": "LSTMForecaster",
  "checkpoint_id": "official_lstm_training",
  "temporal_features": {
    "sequence_id": "seq_driving_sample_L5_000001",
    "current_predicted_reliability": 0.014574,
    "horizon_reliabilities": {
      "1": 0.014574,
      "3": 0.023837,
      "5": 0.018566
    },
    "cross_horizon_spread": -0.003992,
    "is_monotonic_degradation": false,
    "historical_trend_slope": null,
    "degradation_rate": -0.000998,
    "recent_volatility": null,
    "available_history_points": 0
  }
}
```
