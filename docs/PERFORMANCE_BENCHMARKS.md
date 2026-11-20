# Performance Profiling & Resource Benchmarks
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. Benchmark Environment Specifications

- **Processor:** Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (4 logical CPUs).
- **Physical RAM:** 8.00 GB DDR4.
- **Operating System:** Windows 11 Home 64-bit (Build 26100).
- **Python Virtualenv:** Python 3.13.3 AMD64 (`venv`).
- **Acceleration:** None (Pure CPU inference for YOLO and PyTorch LSTM).

---

## 2. Stage-by-Stage Latency Breakdown

The table below reports genuine execution benchmarks measured during real BDD100K driving video inference (15 frames, 720x1280 resolution):

| Pipeline Stage | Module / Component | Latency per Frame | Share of Budget | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Stream Validation** | `OpenCV VideoCapture` | $1.2\text{ ms}$ | $0.2\%$ | Header decoding, dimension check |
| **Frame Read & Write**| Disk I/O | $18.4\text{ ms}$ | $3.8\%$ | Raw JPEG disk write |
| **YOLO Object Detection** | `SingleImageDetector` (YOLO11n) | $315.6\text{ ms}$ | $64.7\%$ | Primary CPU compute bottleneck |
| **ByteTrack Tracking** | `ByteTrackEngine` | $14.2\text{ ms}$ | $2.9\%$ | Kalman state prediction & association |
| **27 Feature Extraction** | `FeatureExtractors` (Multi-modal) | $26.8\text{ ms}$ | $5.5\%$ | Laplacian, contrast, displacement, trend slopes |
| **LSTM Forecasting** | `LSTMForecaster` (H1, H3, H5) | $6.4\text{ ms}$ | $1.3\%$ | PyTorch CPU tensor forward pass |
| **Risk Scoring & Monitor**| `ReliabilityRiskScorer` | $3.1\text{ ms}$ | $0.6\%$ | $S = 1.0 - R$ & EMA tracking |
| **Early Warning Rules** | `EarlyWarningEngine` | $1.9\text{ ms}$ | $0.4\%$ | Rule matching, deduplication & cooldown |
| **HUD Visualization** | `cv2.putText` / `cv2.rectangle` | $100.4\text{ ms}$ | $20.6\%$ | Annotated JPEG rendering & disk write |
| **Total Pipeline Cycle** | Full 12-Stage Execution Loop | **$488.0\text{ ms}$** | **$100.0\%$** | **$\approx 2.05\text{ frames per second}$** |

---

## 3. Memory & Resource Footprint

- **Baseline FastAPI Backend:** $65.2\text{ MB}$ RSS.
- **Post Model Initialization (YOLO11n + LSTM):** $361.1\text{ MB}$ RSS.
- **Active 15-Frame Video Pipeline Peak:** $527.9\text{ MB}$ RSS.
- **RAM Delta During Active Inference:** $+166.8\text{ MB}$ RSS.
- **Leakage Check:** Repeated execution across 6 video scenarios resulted in $+0.1\text{ MB}$ to $+0.3\text{ MB}$ steady-state variation with zero uncollected garbage tensors.
