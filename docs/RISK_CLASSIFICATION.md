# Perception Reliability Risk Classification

## 1. Overview and Operational Motivation

In safety-critical automated driving and autonomous robotic perception, continuous reliability forecasts $R \in [0.0, 1.0]$ and derived failure risk scores $S \in [0.0, 1.0]$ must be mapped into discrete, deterministic operational action tiers. Perception failure risks cannot be left as unquantified numbers; downstream actuation controllers, driver monitoring systems, and degraded-mode managers require explicit, predictable state categories to trigger appropriate warnings, handover requests, or perception redundancy checks.

PerceptGuard AI's **Risk Classification Engine** provides a zero-leakage, deterministic categorization system that evaluates multi-horizon risk scores ($H1, H3, H5$) and synthesized sequence risk metrics against rigorously defined threshold boundaries.

---

## 2. Risk Tier Definitions

The classification engine maps any valid risk score $S \in [0.0, 1.0]$ into four operational severity tiers, along with explicit sentinel states for anomalous or missing telemetry:

| Risk Level | Tier Name | Default Score Range ($S$) | Operational Semantics | System Response |
| :--- | :--- | :--- | :--- | :--- |
| `LOW` | Nominal Operation | $[0.00, 0.30]$ | Detection reliability is nominal; tracking and visual quality stable. | Standard perception pipeline active; no driver or control intervention needed. |
| `MEDIUM` | Advisory / Caution | $(0.30, 0.60]$ | Emerging reliability degradation; mild occlusion or feature drift detected. | Advisory logging; secondary verification enabled; preparatory flag set. |
| `HIGH` | Warning / Degraded | $(0.60, 0.85]$ | Substantial risk of imminent detection drop or track loss within forward horizons. | Stage-1 early warning issued; visual/acoustic alert to driver; throttle dampening. |
| `CRITICAL` | Imminent Failure | $(0.85, 1.00]$ | Severe detection failure imminent; near-zero reliability predicted. | Immediate Stage-2 warning; maximum visual/acoustic alert; handover or fail-safe stop. |
| `UNKNOWN` | Invalid / Unavailable | N/A (Missing / Stale / Error) | Forecast unavailable, telemetry dropped, or TTL expired. | **NEVER defaults to LOW.** Fail-safe state; sensor diagnostic flag raised. |

---

## 3. Boundary Policies and Mathematical Conventions

To eliminate ambiguity at exact boundary transitions, the classifier supports two explicit boundary policies:

### 3.1 Closed-Right Policy (`closed_right`, Default)
In `closed_right` mode, tier intervals are closed on the right:
- $\text{LOW}: 0.00 \le S \le \theta_{\text{low}}$
- $\text{MEDIUM}: \theta_{\text{low}} < S \le \theta_{\text{med}}$
- $\text{HIGH}: \theta_{\text{med}} < S \le \theta_{\text{high}}$
- $\text{CRITICAL}: \theta_{\text{high}} < S \le 1.00$

Under default provisional thresholds ($\theta_{\text{low}}=0.30, \theta_{\text{med}}=0.60, \theta_{\text{high}}=0.85$):
- $S = 0.3000 \implies \text{LOW}$
- $S = 0.3001 \implies \text{MEDIUM}$
- $S = 0.6000 \implies \text{MEDIUM}$
- $S = 0.6001 \implies \text{HIGH}$
- $S = 0.8500 \implies \text{HIGH}$
- $S = 0.8501 \implies \text{CRITICAL}$

### 3.2 Closed-Left Policy (`closed_left`)
In `closed_left` mode, tier intervals are closed on the left:
- $\text{LOW}: 0.00 \le S < \theta_{\text{low}}$
- $\text{MEDIUM}: \theta_{\text{low}} \le S < \theta_{\text{med}}$
- $\text{HIGH}: \theta_{\text{med}} \le S < \theta_{\text{high}}$
- $\text{CRITICAL}: \theta_{\text{high}} \le S \le 1.00$

---

## 4. Multi-Horizon and Horizon-Specific Profiles

Each forward horizon ($H1 \approx 0.133\text{s}, H3 \approx 0.233\text{s}, H5 \approx 0.333\text{s}$) can be configured with an independent `ThresholdProfile`. This allows for tighter tolerances on immediate horizons where reaction times are shorter, and wider advisory bounds on more distant horizons:

```python
from src.risk.risk_config import ClassificationConfig, ThresholdProfile
from src.risk.risk_classifier import RiskClassifier

config = ClassificationConfig(
    default_thresholds=ThresholdProfile(low_upper=0.30, medium_upper=0.60, high_upper=0.85),
    horizon_thresholds={
        1: ThresholdProfile(low_upper=0.20, medium_upper=0.50, high_upper=0.75, horizon=1),
        3: ThresholdProfile(low_upper=0.30, medium_upper=0.60, high_upper=0.85, horizon=3),
        5: ThresholdProfile(low_upper=0.40, medium_upper=0.70, high_upper=0.90, horizon=5),
    }
)
classifier = RiskClassifier(config=config)
```

---

## 5. Safety and Fail-Safe Invariants

1. **No False Reassurance on Missing Telemetry:**
   If a prediction record is missing, telemetry dropped, or timestamp expired, `classify_score` or `classify_record` **strictly returns `RiskLevel.UNKNOWN`**. Under no circumstances does missing data default to `LOW`.
2. **Strict Score Validation:**
   Input risk scores must be finite floating-point numbers in $[0.0, 1.0]$. Passing `NaN`, `Inf`, $S < 0.0$, or $S > 1.0$ immediately raises `RiskValidationError`.
3. **Traceability and Provenance:**
   Every `ClassificationResult` carries the exact threshold bounds applied, the calibration status (`PROVISIONAL` or `CALIBRATED`), boundary policy, and timing metadata.
