# Predictive Early Warning & Alert Generation Engine

## 1. Overview & Operational Purpose

In automated driving systems and robotic perception pipelines, continuous reliability forecasts and discrete risk classifications must translate into actionable, timely, and traceable operational early warnings. Downstream vehicle control systems, Automated Driving System (ADS) managers, and human-machine interfaces (HMI) require well-structured alerts before detection degradation results in critical safety events.

The **Predictive Early Warning & Alert Generation Engine** (`src/early_warning/early_warning_engine.py`) provides rule-based alert triggering, horizon-aware alert separation ($H1, H3, H5$), temporal debouncing and cooldown, evidence-based traceability, and full state-machine lifecycle management.

---

## 2. System Architecture

```
                       TemporalRiskMonitor (Task 4)
                        - Rolling Stats (Mean, Min, Max)
                        - Directional Trend (INCREASING)
                        - Consecutive Increasing Count
                        - State Transitions (L_{t-1} -> L_t)
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          EarlyWarningEngine                            │
│                                                                        │
│  Configurable Rule Engine:                                             │
│  ├── RULE_CRITICAL_BREACH (S > 0.85)     ──► AlertSeverity.CRITICAL    │
│  ├── RULE_HIGH_BREACH     (S > 0.60)     ──► AlertSeverity.WARNING     │
│  ├── RULE_MEDIUM_ADVISORY (S > 0.30)     ──► AlertSeverity.INFO        │
│  ├── RULE_RAPID_DEGRADATION (Trend Inc)  ──► AlertSeverity.TREND_WARN  │
│  ├── RULE_STATE_ESCALATION (Tier Jump)   ──► Escalated Alert           │
│  └── RULE_DATA_QUALITY   (Stale/Missing) ──► DATA_QUALITY_WARNING      │
│                                                                        │
│  Cooldown & Deduplication Controller:                                  │
│  ├── Suppress identical alerts within cooldown period (default: 1.0s)  │
│  └── Critical escalation bypasses cooldown                             │
│                                                                        │
│  Lifecycle State Machine:                                              │
│  ├── NEW -> ACTIVE -> ACKNOWLEDGED -> RESOLVED / EXPIRED               │
│  └── Auto-resolve on nominal LOW risk                                  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
                            EarlyWarningAlert
                     (Audit Log & Actionable Output)
```

---

## 3. Configurable Alert Rules & Severity Mapping

| Rule ID | Trigger Condition | Severity Enum | Alert Type | Operational Meaning & Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| `RULE_CRITICAL_BREACH` | $S_t > 0.85$ (`CRITICAL`) | `CRITICAL` | `THRESHOLD_BREACH` | Imminent perception failure predicted ($R < 0.15$). Immediate acoustic chime; initiate takeover request or fail-safe stop. |
| `RULE_HIGH_BREACH` | $0.60 < S_t \le 0.85$ (`HIGH`) | `WARNING` | `THRESHOLD_BREACH` | Substantial detection failure risk. Display Stage-1 visual warning; dampen throttle and verify sensor field of view. |
| `RULE_MEDIUM_ADVISORY` | $0.30 < S_t \le 0.60$ (`MEDIUM`) | `INFO` | `THRESHOLD_BREACH` | Mild degradation observed. Informational driver display flag; secondary sensor validation enabled. |
| `RULE_RAPID_DEGRADATION` | `Trend == INCREASING` for $\ge 3$ consecutive frames | `TREND_WARNING` | `RAPID_DEGRADATION` | Rapid temporal deterioration in reliability. Pre-charge braking system and prepare for potential tracking drop. |
| `RULE_STATE_ESCALATION` | Operational tier escalated (e.g. `LOW -> HIGH`) | `WARNING` or `CRITICAL` | `STATE_TRANSITION` | Sudden jump in failure risk. Escalate driver warning tier immediately. |
| `RULE_DATA_QUALITY` | Telemetry is stale, missing, or has large timestamp gap | `DATA_QUALITY_WARNING` | `DATA_QUALITY` | Telemetry dropped or stale. **NEVER defaults to LOW risk.** Fail-safe perception mode engaged. |

---

## 4. Multi-Horizon Separation ($H1, H3, H5$)

Alerts are evaluated independently across forward forecasting horizons:
- **$H1$ (~0.133s):** Immediate tactical risk. Direct influence on automated emergency braking (AEB) and collision avoidance.
- **$H3$ (~0.233s):** Short-term operational risk. Informs lane centering and adaptive cruise control dampening.
- **$H5$ (~0.333s):** Advance strategic notice. Advance warning to driver to resume steering control before critical threshold is reached.

**Isolation Invariant:** Forecasts from different horizons or different video sequences are strictly isolated. No aggregate alert is emitted unless mathematically and empirically justified.

---

## 5. Deduplication & Cooldown Policy

1. **Cooldown Period:** Under default configuration (`cooldown_seconds = 1.0`), an active alert for a specific `(sequence_id, horizon, rule_id)` suppresses repeated alert emissions for the same condition.
2. **Critical Escalation Bypass:** If an incoming observation escalates severity (e.g. `WARNING` $\to$ `CRITICAL`), the cooldown suppression is bypassed and the active alert is updated with the higher severity.
3. **Auto-Resolution:** When reliability recovers to nominal `LOW` ($S_t \le 0.30$), active threshold and trend alerts are automatically transitioned to `RESOLVED`.
4. **Safety Invariant:** If input telemetry disappears or is flagged `UNKNOWN`, active alerts **remain ACTIVE** and are **never resolved**.
