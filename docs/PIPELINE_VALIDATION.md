# Complete Pipeline Validation & Data Contract Verification
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. 12-Stage Pipeline Architectural Topology

PerceptGuard AI executes a strictly ordered 12-stage sequential perception and predictive reliability pipeline:

```
[ Video Ingestion & Stream Header Validation ]
                     │
                     ▼
[ OpenCV Video Capture & Bounded Frame Extraction ]
                     │
                     ▼
[ YOLO11n Object Detection (CPU PyTorch Inference) ]
                     │
                     ▼
[ ByteTrack Multi-Object Kalman Kinematics Tracking ]
                     │
                     ▼
[ 27-Dimensional Multi-Modal Feature Extraction ]
  ├── Detection Features (Count, Classes, Confidence Mean/Std, BBox Geometry)
  ├── Motion Features (Track Counts, Displacements, Speeds, Velocities, Coverage)
  ├── Visual Quality (Laplacian Blur, Brightness Mean/Std, Contrast Percentile)
  └── Temporal Trend Slopes (W3/W5 Rolling Linear Slopes)
                     │
                     ▼
[ Temporal Sequence Padding & Window Formation (T=10) ]
                     │
                     ▼
[ PyTorch LSTM Multi-Horizon Forecaster (H1, H3, H5) ]
                     │
                     ▼
[ Reliability Risk Scoring (S = 1.0 - R) & Range Bounding ]
                     │
                     ▼
[ Temporal Risk Monitoring & State Transition Tracking ]
                     │
                     ▼
[ Early-Warning Alert Engine (P1, P2, P3 Rules & Lifecycle) ]
                     │
                     ▼
[ Telemetry HUD Overlay & Annotated Frame Persistence ]
                     │
                     ▼
[ Atomic JSON Packaging (outputs/jobs/{job_id}/results.json) ]
```

---

## 2. Feature Vector Contract Verification

The 27-dimensional feature vector aligns directly with the official trained LSTM model contract (`models/forecasting/lstm/model.pt`) and `datasets/processed/.../feature_manifest.json`:

| Index | Feature Key | Category | Range / Type | Mathematical Meaning |
| :---: | :--- | :--- | :---: | :--- |
| `0` | `detection_count` | Detection | $[0, \infty)$ float | Total bounding boxes detected by YOLO in current frame |
| `1` | `unique_class_count` | Detection | $[0, \infty)$ float | Number of distinct COCO/driving classes present |
| `2` | `mean_confidence` | Detection | $[0.0, 1.0]$ | Mean detection score $\frac{1}{N}\sum c_i$ |
| `3` | `confidence_std` | Detection | $[0.0, 1.0]$ | Standard deviation of confidence scores |
| `4` | `mean_bbox_area` | Detection | $[0.0, \text{pixels}^2]$ | Average bounding box pixel area $(x_2-x_1)(y_2-y_1)$ |
| `5` | `mean_bbox_width` | Detection | $[0.0, W]$ | Average bounding box horizontal dimension |
| `6` | `mean_bbox_height` | Detection | $[0.0, H]$ | Average bounding box vertical dimension |
| `7` | `observed_track_count` | Motion | $[0, \infty)$ float | Active ByteTrack tracks confirmed in current frame |
| `8` | `moving_track_count` | Motion | $[0, \infty)$ float | Tracks with centroid displacement $> 2.0\text{ pixels}$ |
| `9` | `stationary_track_count`| Motion | $[0, \infty)$ float | Tracks with displacement $\le 2.0\text{ pixels}$ |
| `10` | `mean_displacement` | Motion | $[0.0, \infty)$ pixels | Average Euclidean distance traveled by tracked centroids |
| `11` | `maximum_displacement` | Motion | $[0.0, \infty)$ pixels | Maximum single-track displacement across frame step |
| `12` | `mean_speed` | Motion | $[0.0, \infty)\text{ px/s}$ | Mean instantaneous velocity magnitude $d \cdot \text{FPS}$ |
| `13` | `mean_velocity_x` | Motion | $(-\infty, \infty)\text{ px/s}$| Mean horizontal velocity vector |
| `14` | `mean_velocity_y` | Motion | $(-\infty, \infty)\text{ px/s}$| Mean vertical velocity vector |
| `15` | `motion_feature_coverage`| Motion | $\{0.0, 1.0\}$ | Flag indicating whether motion tracking was active |
| `16` | `blur_score` | Visual Quality | $[0.0, \infty)$ | Variance of Laplacian filter response $\text{Var}(\nabla^2 I)$ |
| `17` | `mean_brightness` | Visual Quality | $[0.0, 255.0]$ | Grayscale channel mean intensity |
| `18` | `brightness_std` | Visual Quality | $[0.0, 255.0]$ | Grayscale channel standard deviation |
| `19` | `contrast_std` | Visual Quality | $[0.0, 255.0]$ | Root-mean-square contrast metric |
| `20` | `contrast_range` | Visual Quality | $[0.0, 255.0]$ | Dynamic range: $\max(I) - \min(I)$ |
| `21` | `contrast_percentile_range`| Visual Quality| $[0.0, 255.0]$ | $P_{95}(I) - P_{5}(I)$ robust contrast span |
| `22` | `numeric_feature_coverage`| Visual Quality| $1.0$ | Fixed contract indicator |
| `23` | `w3_confidence_trend_slope`| Temporal Trend| $(-\infty, \infty)$ | Linear regression slope over 3-frame confidence window |
| `24` | `w3_blur_trend_slope` | Temporal Trend| $(-\infty, \infty)$ | Linear regression slope over 3-frame blur score window |
| `25` | `w5_confidence_trend_slope`| Temporal Trend| $(-\infty, \infty)$ | Linear regression slope over 5-frame confidence window |
| `26` | `w5_blur_trend_slope` | Temporal Trend| $(-\infty, \infty)$ | Linear regression slope over 5-frame blur score window |

---

## 3. Data Sanitization & Invariants

- **Numerical Finiteness:** All 27 values pass through `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)` prior to tensor construction, ensuring strict adherence to PyTorch's `torch.isfinite(x).all()` invariant.
- **Horizon Independence:** $H1, H3, H5$ forecasts and risk records maintain isolated temporal tracking state and discrete alert lifecycle evaluation.
