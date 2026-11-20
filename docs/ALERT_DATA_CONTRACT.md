# Early Warning Alert Data Contract

## 1. Overview & Data Schema

The `EarlyWarningAlert` data structure represents a standardized, serializable early warning notification produced by the `EarlyWarningEngine`.

```python
@dataclass
class EarlyWarningAlert:
    alert_id: str                          # Deterministic unique ID: alert_{seq}_h{horizon}_{rule}_{ts_ms}
    sequence_id: str                       # Video sequence identifier
    risk_level: RiskLevel                  # Discrete RiskLevel enum (LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN)
    risk_score: float                      # Continuous risk score in [0.0, 1.0]
    trigger_horizon: int                   # Forecast horizon (1, 3, 5)
    source_frame_id: str                   # Frame ID of source observation
    target_frame_id: str                   # Frame ID of forecasted target
    target_timestamp_seconds: Optional[float] # Forecasted event timestamp
    warning_message: str                   # Operator / system warning message
    recommended_action: str                # Prescribed mitigation action
    is_suppressed: bool = False            # True if suppressed by cooldown
    severity: AlertSeverity = AlertSeverity.WARNING
    alert_type: AlertType = AlertType.THRESHOLD_BREACH
    lifecycle_state: AlertLifecycleState = AlertLifecycleState.NEW
    video_id: str = "driving_sample"
    frame_id: str = ""                     # Synonymous with source_frame_id
    timestamp: float = 0.0                 # Event timestamp
    generated_at: Optional[str] = None     # Wall-clock timestamp
    forecast_horizon: Optional[int] = None # Synonymous with trigger_horizon
    risk_category: Optional[RiskLevel] = None # Synonymous with risk_level
    trigger_reason: str = ""               # Traceable explanation
    risk_trend: str = "UNKNOWN"            # Directional trend string
    previous_category: Optional[RiskLevel] = None
    current_category: Optional[RiskLevel] = None
    calibration_status: str = "PROVISIONAL"
    model_version: str = "LSTMForecaster"
    rule_id: str = "RULE_DEFAULT"
    acknowledgement_status: str = "UNACKNOWLEDGED"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=...)
```

---

## 2. Enumerated Domains

### 2.1 `AlertSeverity`
| Severity Value | Operational Priority | Typical Triggers |
| :--- | :---: | :--- |
| `INFO` | Low | Mild reliability dip (`MEDIUM` tier), advisory logging. |
| `DATA_QUALITY_WARNING` | Guardrail | Stale predictions, dropped telemetry, temporal gaps. |
| `TREND_WARNING` | Medium | Rapid monotonic risk degradation across consecutive frames. |
| `WARNING` | High | `HIGH` risk threshold breach, state escalation. |
| `CRITICAL` | Severe | `CRITICAL` risk threshold breach, imminent detection failure. |

### 2.2 `AlertType`
- `THRESHOLD_BREACH`: Risk score exceeded configured upper cutoff.
- `RAPID_DEGRADATION`: Consecutive frames exhibiting increasing risk slope.
- `STATE_TRANSITION`: Operational risk level escalated between frames.
- `DATA_QUALITY`: Stale telemetry, missing forecast, or sensor gap.

### 2.3 `AlertLifecycleState`
- `NEW`: Newly generated alert.
- `ACTIVE`: Alert currently active and unacknowledged/unresolved.
- `ACKNOWLEDGED`: Operator or downstream controller confirmed the alert.
- `RESOLVED`: Condition resolved (reliability returned to nominal `LOW`).
- `EXPIRED`: Alert exceeded time-to-live (`alert_ttl_seconds`).
