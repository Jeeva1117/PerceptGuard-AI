# Streamlit Dashboard User & Operator Guide
## PerceptGuard AI: Predictive Object Detection Reliability Dashboard

---

## 1. Dashboard Overview

The **PerceptGuard AI Dashboard** is a production-grade Streamlit operations console that allows perception engineers and fleet operators to monitor object-detection reliability, upload driving camera videos, inspect real-time background processing stages, and evaluate anticipated perception failures across multiple forward horizons ($H1, H3, H5$).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PerceptGuard AI Console                         │
├────────────────────────────────────────────────────────────────────────┤
│ Navigation:                                                            │
│ 1. 📊 Operations Overview       - High-level KPIs & system status      │
│ 2. 🎥 Video Pipeline & Ingestion - Upload, process & inspect videos    │
│ 3. 📈 Temporal Risk & Tracking  - Multi-horizon risk & ByteTrack IDs   │
│ 4. 🚨 Early-Warning Alerts      - Incident feed & operational actions  │
│ 5. 🔍 Diagnostics & Health      - Checkpoint & process metrics         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Navigation Views

### A. Operations Overview
Displays high-level executive health scorecards, active alert tallies, and multi-horizon failure risk gauges ($H1 = t+1$, $H3 = t+3$, $H5 = t+5$).

### B. Video Pipeline & Ingestion
Allows operators to:
1. Upload real autonomous-driving video files (`.mp4`, `.avi`, `.mov`, `.mkv` up to $100\text{ MB}$).
2. Specify the analysis horizon ($5$ to $50$ frames, default $20$ frames for CPU efficiency).
3. Monitor the live background execution stage:
   `VALIDATING` $\to$ `EXTRACTION` $\to$ `DETECTION` $\to$ `TRACKING` $\to$ `FEATURES` $\to$ `FORECASTING` $\to$ `RISK_SCORING` $\to$ `ALERTS` $\to$ `COMPLETED`.
4. Inspect annotated video frames using an interactive frame slider.
5. Review detected object classes, bounding boxes, confidences, and tracking identities.

### C. Temporal Risk & Tracking
Visualizes temporal risk trajectories across time using Plotly, highlighting risk transitions and EMA trend velocity.

### D. Early-Warning Alerts
Provides a filtered incident feed with severity badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), trigger reasons, affected horizons, and operator acknowledgment/resolution capabilities.

### E. Diagnostics & Health
Shows Kubernetes liveness/readiness probe states, model checkpoint availability (`yolo11n.pt` and `model.pt`), and SRE process metrics (CPU %, RAM RSS, uptime).

---

## 3. Running the Dashboard

```powershell
.\venv\Scripts\streamlit run dashboard/app.py --server.port 8501
```
Navigate to: `http://localhost:8501`
