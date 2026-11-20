# Video Processing Pipeline Workflow
## PerceptGuard AI: Autonomous Driving Perception Reliability

---

## 1. End-to-End Execution Sequence

```
1. Video Ingestion
   User uploads video (MP4/AVI/MOV up to 100 MB). Checked by cv2.VideoCapture.
          │
          ▼
2. Frame Extraction
   Sequentially decodes frames up to max_frames (e.g. 20 frames for CPU safety).
   Extracts physical timestamps: t = source_frame_index / fps.
          │
          ▼
3. Frame Preprocessing
   Normalizes and manages color channels (BGR -> RGB) with immediate memory heap release.
          │
          ▼
4. YOLO Object Detection
   Evaluates each frame via YOLO11n on CPU (conf=0.25, iou=0.45).
   Extracts bounding boxes [x1, y1, x2, y2], classes, and confidence scores.
          │
          ▼
5. ByteTrack Multi-Object Tracking
   Associates detections chronologically across frames using 2D Kalman filters.
   Assigns stable, persistent track IDs.
          │
          ▼
6. Multi-Modal Feature Extraction (27 Features)
   - Detection: count, mean confidence, variance, bounding box areas.
   - Motion: tracklet displacement, velocity magnitude.
   - Visual Quality: Laplacian blur variance, mean brightness, RMS contrast.
   - Temporal: rolling statistics over the temporal window.
          │
          ▼
7. Temporal Sequence Construction
   Constructs sequence tensor of shape (1, 10, 27) with historical padding.
          │
          ▼
8. Multi-Horizon LSTM Forecasting
   Passes tensor into LSTMForecaster (loaded from trained PyTorch checkpoint).
   Predicts future detection reliability R_{t+1}, R_{t+3}, R_{t+5} in [0.0, 1.0].
          │
          ▼
9. Reliability Risk Scoring
   Translates reliability into risk: S_{t+h} = 1.0 - R_{t+h} in [0.0, 1.0].
          │
          ▼
10. Risk Classification & Temporal Monitoring
    Classifies risk levels: Low (< 0.25), Medium (0.25-0.50), High (0.50-0.75), Critical (>= 0.75).
    Tracks exponential moving averages (EMA) and risk velocity trends.
          │
          ▼
11. Early-Warning Alert Generation
    EarlyWarningEngine triggers P1, P2, P3 alerts with trigger reasons and horizons.
          │
          ▼
12. Frame Annotation & Result Persistence
    Overlays bounding boxes, track IDs, detection labels, and HUD risk badges.
    Persists results.json and annotated frames to outputs/jobs/{job_id}/.
```

---

## 2. Resource Boundedness & CPU Guarantees

- **Single Worker Thread:** Video processing runs in `ThreadPoolExecutor(max_workers=1)` to prevent CPU saturation on Intel Core i3 systems.
- **Immediate Heap Deallocation:** Frames are processed sequentially and saved to disk; no entire video arrays are buffered in RAM.
- **Mean Processing Throughput:** $\approx 70 - 150\text{ ms}$ per frame on CPU across all 12 stages.
