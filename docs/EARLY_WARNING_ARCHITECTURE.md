# PerceptGuard AI — Predictive Early Warning & Perception Risk Assessment Architecture

**Document Version:** 1.0.0  
**Phase:** Day 10 Architecture & Environment Specification  
**Status:** Approved  
**Author:** Principal AI Engineer, Senior Machine Learning Systems Architect, Computer Vision Reliability Engineer  

---

## 1. Executive Summary & Purpose

The **Predictive Early Warning and Risk Assessment System** represents the actionable decision-making layer of **PerceptGuard AI**. 

In Days 1 through 8, the framework established multi-modal computer vision feature extraction, ByteTrack tracking, temporal feature aggregation, and ground-truth reliability labeling. In Day 9, the system achieved supervised multi-horizon forecasting ($H \in \{1, 3, 5\}$ frames ahead) using a lightweight PyTorch `LSTMForecaster` network.

The objective of Day 10 is to transform those raw, continuous multi-horizon reliability forecasts into an actionable, real-time safety alerting system that anticipates object detection degradation **before catastrophic localization or tracking failures occur**.

```
+-------------+     +-------------------+     +------------------+     +--------------------+
| Video Input | --> | YOLO Detection v8 | --> | ByteTrack Multi- | --> | Multi-Modal Feature|
| (Stream/MP4)|     | (Boxes & Classes) |     | Object Tracker   |     | Extractors (F=27)  |
+-------------+     +-------------------+     +------------------+     +--------------------+
                                                                                 |
                                                                                 v
+-------------------+     +-------------------+     +-------------------+     +--------------------+
| Early Warning     | <-- | Temporal Risk     | <-- | Risk              | <-- | Recurrent LSTM     |
| Generation & Feed |     | Monitor & Debounce|     | Classification    |     | Forecaster (H1/3/5)|
+-------------------+     +-------------------+     +-------------------+     +--------------------+
         |
         v
+-------------------+
| Dashboard & Video |
| Overlay Stream    |
+-------------------+
```

---

## 2. Day 9 Readiness Gate & Foundation

Before architecting the Early Warning layer, the Day 9 Master Final Audit was inspected:
- **Audit Verdict:** `PASS WITH LIMITATIONS`.
- **Trained Model Checkpoint:** `checkpoint_best.pt` ($103.56\text{ KB}$, 7,907 float32 parameters) converged to validation loss $0.000470$ in 30 epochs.
- **Output Properties:** 414 continuous reliability forecasts bounded strictly in $[0.0064, 0.0241]$ with 0 `NaN` or `Inf` instances.
- **Target Horizons:** $H1 = 1$ frame, $H3 = 3$ frames, $H5 = 5$ frames into the future.
- **Hardware Profile:** Single forward pass executes in **$0.27\text{ ms}$** on Intel Core i3 CPU with $<55\text{ MB}$ memory footprint.
- **Known Limitations:** Ground-truth targets on the single 10-second driving clip are zero-inflated (>99% zero). The early warning system is designed to consume probabilistic predictions without assuming unrealistic failure rates, leaving thresholds fully configurable for multi-video BDD100K calibration.

---

## 3. Modular System Architecture & Pipeline Stages

The system is decomposed into decoupled, independently testable functional stages:

```
Video Stream ──► [Stage 1: Ingestion & Object Detection] (YOLOv8)
                       │
                       ▼
                 [Stage 2: Temporal Object Tracking] (ByteTrack)
                       │
                       ▼
                 [Stage 3: Feature Extraction & Aggregation] (T=5, F=27)
                       │
                       ▼
                 [Stage 4: Multi-Horizon Forecasting] (LSTM H1, H3, H5)
                       │
                       ▼
                 [Stage 5: Data Contract Ingestion & Validation]
                       │
                       ▼
                 [Stage 6: Multi-Horizon Risk Assessment Engine]
                       │
                       ▼
                 [Stage 7: Risk Classification & Level Mapping]
                       │
                       ▼
                 [Stage 8: Temporal Risk Monitoring & Debouncing]
                       │
                       ▼
                 [Stage 9: Early Warning Generation & Cooldown]
                       │
                       ▼
                 [Stage 10: Telemetry & Dashboard Presentation]
```

### Module Responsibilities:

| Module | Core Responsibility | Input Contract | Output Contract |
| :--- | :--- | :--- | :--- |
| `EarlyWarningConfig` | Centralized typed configuration for horizons, thresholds, debouncing, and runtime | YAML file / environment | Validated `EarlyWarningConfig` |
| `EarlyWarningDataContract` | Validates, sanitizes, and groups multi-horizon prediction records | Day 9 predictions JSON / dicts | Validated `EarlyWarningInputRecord` batch |
| `IRiskAssessmentEngine` | Synthesizes multi-horizon reliability, trends, and degradation into a scalar risk score | `EarlyWarningInputRecord` sequence | Quantitative `RiskScore` ($[0.0, 1.0]$) |
| `IRiskClassifier` | Maps continuous risk score into discrete categorical levels | Float risk score | `RiskLevel` Enum |
| `ITemporalRiskMonitor` | Tracks sliding window persistence and filters single-frame transient noise | Sequence of `RiskScore` | Boolean alert trigger condition |
| `IEarlyWarningGenerator` | Formats actionable alert records with message, severity, and cooldown | `RiskScore`, `EarlyWarningInputRecord` | `EarlyWarningAlert` or `None` |
| `IEarlyWarningDashboard` | Visualizes perception reliability trajectories and alert feeds | `RiskScore`, active alerts | Diagnostic telemetry & overlays |

---

## 4. Risk Level Taxonomy & Operational Semantics

Perception failure risk is classified into four discrete operational severity tiers plus an explicit `UNKNOWN` safety fallback:

| Risk Level | Operational Meaning | System Implication | Recommended Automated / Driver Action |
| :---: | :--- | :--- | :--- |
| **`LOW`** | Nominal perception reliability ($\hat{R} \ge \tau_{\text{nominal}}$). High confidence in object detections. | Standard autonomous / ADAS operation. | None; maintain nominal cruise & tracking. |
| **`MEDIUM`** | Minor degradation forecasted ($H1/H3/H5$ indicate slight drop or unfavorable visual trend). | Heightened perception monitoring. | Log telemetry; tighten tracking association gates; prepare sensor fallback. |
| **`HIGH`** | Significant perception failure anticipated within upcoming 1–5 frames. | High probability of missed detections or tracking loss. | Issue visual/audible early warning; reduce speed; prompt operator readiness. |
| **`CRITICAL`** | Severe perception collapse predicted (e.g. sensor blinding, extreme occlusion, zero IoU). | Perception cannot be relied upon for safety-critical collision avoidance. | Immediate emergency handover / initiate fail-safe controlled deceleration. |
| **`UNKNOWN`** | Missing sensor frames, corrupted input features, or model inference failure. | System cannot verify perception integrity. | Enter fail-safe degraded mode; alert system diagnostic monitor. |

---

## 5. Risk Assessment Design Specification

### 5.1 Directional Inversion
Reliability ($\hat{R} \in [0.0, 1.0]$) measures perception success: $1.0$ is perfect detection overlap, while $0.0$ represents total detection failure.  
Risk ($S \in [0.0, 1.0]$) measures failure probability: $0.0$ is zero risk (safe), while $1.0$ is maximum failure danger.

$$\text{Baseline Risk}(t+H) = 1.0 - \hat{R}_{t+H}$$

### 5.2 Multi-Horizon Synthesis
Rather than assessing only the next frame ($H=1$), the risk engine synthesizes near ($H=1$), medium ($H=3$), and far ($H=5$) predictions using configurable weights:
$$S_{\text{multi}}(t) = w_1 (1 - \hat{R}_{t+1}) + w_3 (1 - \hat{R}_{t+3}) + w_5 (1 - \hat{R}_{t+5}), \quad \sum_i w_i = 1.0$$
*Default configuration weights:* $w_1 = 0.50$, $w_3 = 0.35$, $w_5 = 0.15$.

### 5.3 Temporal Degradation Slope
In addition to static reliability values, a rapid negative slope ($\frac{d\hat{R}}{dt} < 0$) indicates sudden perception collapse (e.g. entering a tunnel, sudden headlight glare). The risk engine incorporates a degradation penalty when:
$$\Delta \hat{R} = \hat{R}_{t} - \hat{R}_{t-k} < \tau_{\text{degradation}}$$

---

## 6. Temporal Risk Monitoring, Debouncing & Cooldown

To prevent **alert fatigue** caused by single-frame transient noise or momentary occlusions:
1. **Sliding Window Persistence:** An alert is only triggered if elevated risk persists for at least $M$ consecutive frames (default: $M=2$ frames within a window of $W=5$).
2. **Debounce Logic:** Consecutive warnings for the same object or sequence are grouped.
3. **Alert Cooldown:** After an alert is dispatched, a cooldown period of $C=5$ frames is enforced before repeating an alert of the same severity tier, unless severity escalates to `CRITICAL`.

---

## 7. Hardware & Resource Budget

The entire Early Warning subsystem is engineered for edge CPU execution:
- **Processor Target:** Intel Core i3 (dual-core / 4 threads)
- **RAM Ceiling:** 8.00 GB DDR4
- **Operating System:** Windows 11 Build 26200
- **Execution Target:** CPU-only, zero CUDA requirement.
- **Latency Budget:** Risk evaluation + classification + alert generation must complete in **$< 0.10\text{ ms}$ per sequence**, leaving ample room for video decoding and YOLO detection.
- **Memory RSS Budget:** Subsystem memory overhead is strictly bounded under **$< 20\text{ MB}$**.

---

## 8. Verification Strategy & Regression Protection

1. **Schema Validation:** Ensure 100% of prediction records ingested from Day 9 meet the data contract.
2. **Horizon Recognition:** Enforce strict checking of horizons $\{1, 3, 5\}$.
3. **Value Range Enforcement:** Guarantee continuous outputs within $[0.0, 1.0]$.
4. **Clean Decoupling:** Keep interfaces abstract so downstream tasks (Risk Scoring Engine in Task 2, Threshold Calibration in Task 3, Alert Generation in Task 4) can be developed and verified independently.
