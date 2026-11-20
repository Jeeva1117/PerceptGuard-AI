# PerceptGuard AI — Predictive Early Warning & Perception Reliability Dashboard

## 1. Overview & System Purpose

The **PerceptGuard AI Early Warning Dashboard** (`dashboard/app.py`) is an interactive Streamlit-based operations and telemetry console designed for Autonomous Driving Systems (ADS), safety operators, and perception engineers.

It unifies the outputs of the entire 8-stage perception pipeline into a centralized, responsive interface:
- **Instantaneous perception operational health** (Section A)
- **Multi-horizon reliability risk breakdown** ($H1$, $H3$, $H5$) with independent metrics (Section B)
- **Temporal risk monitoring and trajectory charts** (Section C)
- **Early-warning incident feeds, filtering, and lifecycle management** (Section D)
- **Authentic YOLO11n detection and ByteTrack tracking telemetry** (Section E)
- **Granular 8-stage perception pipeline status**

The dashboard strictly adheres to scientific integrity principles:
1. **Zero fabricated metrics:** All metrics reflect verified predictions, genuine tracking telemetry, and verified pipeline artifacts.
2. **Horizon separation:** $H1$ ($t+1$), $H3$ ($t+3$), and $H5$ ($t+5$) are never conflated or averaged into misleading single numbers.
3. **Provisional calibration notice:** Risk scores are explicitly presented as derived failure risk ($S = 1.0 - R$) under **PROVISIONAL** calibration status, avoiding unjustified claims of calibrated probability.

---

## 2. Dashboard Architecture

```
                                  Streamlit Web Interface (dashboard/app.py)
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ [ Header & Branding ]  PerceptGuard AI  |  CPU-Optimized  |  Calibration Gate: PROVISIONAL             │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [ Section A: System Overview ]                                                                         │
 │  ├── Pipeline Status: READY / DEGRADED / BLOCKED / UNAVAILABLE                                         │
 │  ├── Overall Health: NOMINAL / DEGRADED / CRITICAL                                                     │
 │  ├── Freshness: CURRENT / STALE                                                                        │
 │  └── Model Availability: LSTMForecaster, YOLO11n, ByteTrack                                            │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [ Section B: Multi-Horizon Risk Overview ]                                                             │
 │  ├── Horizon 1 (t+1): Score, Category, Trend, Rolling Mean, Volatility (σ), Observation Count          │
 │  ├── Horizon 3 (t+3): Score, Category, Trend, Rolling Mean, Volatility (σ), Observation Count          │
 │  └── Horizon 5 (t+5): Score, Category, Trend, Rolling Mean, Volatility (σ), Observation Count          │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [ Section C: Temporal Risk Monitoring ]                                                                │
 │  ├── Multi-horizon risk trajectory chart (Streamlit line chart over time)                             │
 │  └── Operational state transitions table (Level changes, genuine vs jitter)                            │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [ Section D: Early Warning Alerts & Management ]                                                       │
 │  ├── Filter Controls: Severity, Horizon, Alert Type, Lifecycle State                                  │
 │  ├── Incident Feed: Filterable table with ID, Severity, Type, Score, State, Message                    │
 │  └── Operator Actions: Detail inspection drawer, Acknowledge & Resolve buttons                         │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [ Section E: Detection & Tracking Telemetry ]                                                          │
 │  └── Real YOLO11n & ByteTrack statistics: frames, bboxes, unique tracks, mean confidence, latency     │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    ▲
                                                    │ In-Process Service Layer
 ┌──────────────────────────────────────────────────┴─────────────────────────────────────────────────────┐
 │                                   DashboardService (src/early_warning/)                                │
 │                                  implements IEarlyWarningDashboard                                     │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │  ├── get_system_health()                  ├── get_alerts()                                             │
 │  ├── get_pipeline_status()                ├── get_alert_detail()                                       │
 │  ├── get_reliability_risk_overview()      ├── acknowledge_alert()                                      │
 │  ├── get_temporal_monitoring_data()       ├── resolve_alert()                                          │
 │  └── get_detection_tracking_context()                                                                  │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Dashboard Sections

### Section A — System Operational Overview
Displays high-level health and deployment availability:
- **Pipeline Status:** Explicitly reports `READY`, `RUNNING`, `DEGRADED`, `BLOCKED`, or `UNAVAILABLE`. Never falsely displays `READY` when models or inputs are absent.
- **Overall Health:** Qualitative health tier (`NOMINAL`, `DEGRADED`, `CRITICAL`) with numerical health percentage.
- **Data Freshness:** `CURRENT` if telemetry arrives within configured TTL, `STALE` if expired.
- **Model Availability:** Verification status for `LSTMForecaster`, `YOLO11n`, and `ByteTrack`.
- **Calibration Banner:** Visual indicator affirming `CALIBRATION: PROVISIONAL`.

### Section B — Multi-Horizon Reliability Risk Overview
Provides side-by-side inspection cards for the three certified horizons:
- **Horizon 1 ($t+1$):** Short-term immediate collision / track drop risk ($0.033\text{ s} - 0.20\text{ s}$).
- **Horizon 3 ($t+3$):** Intermediate tactical trajectory risk ($0.10\text{ s} - 0.60\text{ s}$).
- **Horizon 5 ($t+5$):** Extended strategic planning horizon risk ($0.20\text{ s} - 1.00\text{ s}$).
Each card displays:
- Instantaneous derived risk score $S_t \in [0.0, 1.0]$
- Operational Risk Category (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- Directional Trend symbol and label (`↗ INCREASING`, `↘ DECREASING`, `→ STABLE`, `↕ VOLATILE`)
- Rolling window mean, volatility ($\sigma$), and valid frame observation counts.

### Section C — Temporal Risk Monitoring
- **Trajectory Visualization:** Multi-trace interactive line chart mapping risk scores over sensor time (seconds). Handles missing or null observations cleanly without plotting misleading false zeroes.
- **State Transition Audit Log:** Expandable table recording all discrete risk level transitions (e.g. `LOW -> MEDIUM`, `HIGH -> CRITICAL`), timestamp, trigger score, and whether the change was genuine or transient jitter.

### Section D — Early Warning Alerts Feed & Lifecycle Management
- **Multi-Dimensional Filtering:**
  - Severity: `ALL`, `CRITICAL`, `WARNING`, `CAUTION`, `INFO`, `DATA_QUALITY_WARNING`
  - Horizon: `ALL`, `H1 (t+1)`, `H3 (t+3)`, `H5 (t+5)`
  - Alert Type: `ALL`, `RISK_LEVEL_THRESHOLD`, `RAPID_RISK_INCREASE`, `VOLATILITY_SPIKE`, `CONSECUTIVE_DEGRADATION`, `HORIZON_DIVERGENCE`, `DATA_QUALITY`
  - Lifecycle State: `ALL`, `ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`, `EXPIRED`
- **Interactive Feed:** Displays active and historical alerts with ID, horizon, score, severity, and explanation.
- **Operator Lifecycle Actions:** Operators can inspect any alert's full trigger rationale, source/target frames, model provenance, and click **Acknowledge** or **Resolve** with timestamped audit logging.

### Section E — Detection & Tracking Context
Surfaces authentic telemetry from Day 2 YOLO11n detection and Day 3 ByteTrack tracking runs (`datasets/processed/driving_sample/tracking/multi_object/.../tracking_summary.json`):
- Model: `YOLO11n` (CPU inference)
- Tracker: `ByteTrack` (two-stage Kalman association)
- Total frames processed: 30
- Total detections: 27 bounding boxes
- Total unique tracks: 2 vehicles
- Max simultaneous active tracks: 2
- Mean detection confidence: 40.0% (0.4003)
- Mean processing latency: 280.4 ms/frame

---

## 4. Execution & Launch Instructions

### Local Launch via Streamlit
To launch the interactive dashboard locally:

```bash
# Using the project virtual environment
.\venv\Scripts\streamlit.exe run dashboard/app.py
```

The application starts by default on `http://localhost:8501`.

### Background Automation & Testing
The dashboard service layer (`DashboardService`) is decoupled from Streamlit rendering, enabling comprehensive automated unit and integration testing via pytest:

```bash
.\venv\Scripts\pytest.exe tests/early_warning/test_dashboard_integration.py tests/early_warning/test_dashboard_frontend.py -v
```
