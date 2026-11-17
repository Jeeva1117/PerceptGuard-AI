# PerceptGuard-AI: System Architecture Specification
## Predictive Object Detection Reliability for Autonomous Driving Perception

**Author:** Staff Machine Learning Engineer, Autonomous Driving & Perception Systems  
**Document Version:** 1.0.0  
**Status:** Approved for Implementation (Day 1 Baseline)  
**Target Platform:** Python 3.10+, PyTorch (CPU / CUDA compatible)

---

## 1. Executive Summary & Project Objective

Autonomous driving systems (AD) operate in safety-critical, non-stationary environments where camera perception must remain fail-safe. Standard computer vision perception pipelines rely on deep convolutional or transformer-based object detectors (such as YOLO, DETR, or SSD) coupled with multi-object tracking (such as ByteTrack or SORT). 

A critical vulnerability of state-of-the-art vision pipelines is that **detection confidence is not detection reliability**:
- Object detectors routinely output high confidence scores (>85%) on severely mislocalized bounding boxes, partially occluded objects, or visual artifacts.
- Conversely, detector confidence often plummets precipitously without prior warning during sudden illumination changes, camera glare, motion blur, or partial occlusion behind roadside infrastructure.
- Downstream autonomous planners and safety supervisors (e.g., Emergency Braking, Trajectory Optimization) receive detection boxes blindly until track loss occurs, resulting in reactive rather than proactive safety maneuvers.

**Project Objective:**  
Develop an edge-deployable, research-grade predictive AI system that forecasts detection reliability degradation over upcoming temporal horizons: **$t+1$ frame (~33ms), $t+3$ frames (~100ms), and $t+5$ frames (~166ms)** ahead. By extracting historical kinematic, visual quality, tracking stability, and confidence dynamics, the system trains a recurrent neural network (LSTM/GRU) to anticipate perception failures before they trigger track divergence or missed detections.

---

## 2. Problem Statement & Mathematical Formulation

### 2.1 The Critical Distinction: Confidence vs. Reliability

In deep object detectors, the confidence score for a candidate box $b$ is given by:
$$C(b) = P(\text{Object}) \times P(\text{Class}_k | \text{Object})$$
This score reflects only the network's internal activation calibration on static training distributions. It does **not** quantify:
1. Spatial Localization Accuracy: Whether the box boundary tightly aligns with physical ground truth (Intersection over Union, $\text{IoU}$).
2. Temporal Coherence: Whether the object will remain detectable under impending motion blur, visual scale shrinkage, or occlusion.
3. System Reliability: The probability that downstream perception will correctly localize the target across subsequent frames.

### 2.2 Mathematical Formulation of Future Reliability

Let an object track be denoted by $\tau_i$. At frame index $t$, the system observes a temporal sequence of feature vectors extracted from the previous $W$ frames:
$$\mathbf{X}_{t}^{(i)} = [\mathbf{x}_{t - W + 1}^{(i)}, \mathbf{x}_{t - W + 2}^{(i)}, \dots, \mathbf{x}_{t}^{(i)}] \in \mathbb{R}^{W \times D}$$
where $W$ is the historical sliding window length (default: 10 frames) and $D$ is the feature dimensionality.

Let $B_{t+h}^{(i)}$ denote the detected bounding box of track $i$ at future frame $t+h$, and let $G_{t+h}^{(i)}$ denote the true physical ground-truth bounding box.

The ground-truth future reliability target $R_{t+h}^{(i)}$ at horizon $h \in \{1, 3, 5\}$ is defined as:
$$R_{t+h}^{(i)} = \begin{cases} 
\text{IoU}(B_{t+h}^{(i)}, G_{t+h}^{(i)}) & \text{if object } i \text{ is detected at } t+h \\
0.0 & \text{if object } i \text{ is missed (False Negative) or track lost at } t+h
\end{cases}$$

Additionally, we define a binary safety-critical degradation label:
$$Y_{t+h}^{(i)} = \mathbb{I}(R_{t+h}^{(i)} < \theta_{\text{safe}})$$
where $\theta_{\text{safe}} = 0.50$ (standard PASCAL VOC criterion) or $0.70$ (strict autonomous driving localization criterion).

The objective of the predictive model $f_{\mathbf{\Theta}}$ parameterized by weights $\mathbf{\Theta}$ is to map historical temporal features to multi-horizon future reliability estimates:
$$\hat{\mathbf{R}}_{t}^{(i)} = [\hat{R}_{t+1}^{(i)}, \hat{R}_{t+3}^{(i)}, \hat{R}_{t+5}^{(i)}] = f_{\mathbf{\Theta}}(\mathbf{X}_{t}^{(i)})$$

---

## 3. End-to-End System Pipeline

```
+-----------------------------------------------------------------------------------------+
|                                    Video Input Stream                                   |
|                (Automotive Dashcam / BDD100K / KITTI / Waymo Camera Sequences)          |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| Module 1: Video Frame Ingestion & Preprocessing (OpenCV)                                 |
| - Frame decoding, timestamp alignment, color normalization, resizing                     |
| - Output: FramePacket(frame_id, timestamp, image_bgr, global_quality_metrics)           |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| Module 2: Real-Time Object Detection (YOLOv8)                                           |
| - Edge inference (vehicles, pedestrians, cyclists, road signs)                           |
| - Non-Maximum Suppression (NMS)                                                         |
| - Output: DetectionSet(frame_id, [BoundingBox(x1, y1, x2, y2), Confidence, ClassID])    |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| Module 3: Multi-Object Tracking (ByteTrack)                                              |
| - Two-stage association using high-confidence and low-confidence detection matching     |
| - Kalman filtering for state prediction (centroid velocity, aspect ratio)                |
| - Output: TrackedObjectSet(frame_id, [Track(id, bbox, conf, class_id, velocity, age)])  |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| Module 4: Feature Engineering & Temporal Sequence Construction                          |
| - Per-object feature extraction:                                                        |
|   1. Confidence Dynamics (d_conf/dt, rolling variance, exponential moving average)      |
|   2. Kinematics & Scale (velocity, acceleration, area expansion rate, aspect ratio)      |
|   3. Visual Quality & Image Degradation (crop Laplacian variance/blur, contrast, SNR)   |
|   4. Spatial & Contextual Geometry (distance to image border, optical flow, occlusion)  |
| - Sliding window buffer: Shape [N_tracks, Window_Size=10, Feature_Dim=16]               |
+-----------------------------------------------------------------------------------------+
                                             |
                     +-----------------------+-----------------------+
                     | (Training / Supervision)                      | (Online Inference)
                     v                                               v
+-----------------------------------------+     +-----------------------------------------+
| Module 5: Ground-Truth Reliability Gen  |     | Module 6: Recurrent Forecasting Model   |
| - Hungarian matching with GT annotations|     | (LSTM / GRU Multi-Horizon Architecture) |
| - Computes true IoU at t+1, t+3, t+5    |     | - Bidirectional / Unidirectional GRU    |
| - Generates supervision targets R_{t+h} |     | - Multi-head output: R_{t+1}, t+3, t+5  |
+-----------------------------------------+     +-----------------------------------------+
                     \                                               /
                      \                                             /
                       v                                           v
+-----------------------------------------------------------------------------------------+
| Module 7: Evaluation & Reliability Metrics                                              |
| - Regression: Mean Squared Error (MSE), Mean Absolute Error (MAE)                       |
| - Warning Classification: Precision, Recall, F1-Score, ROC-AUC for reliability collapse |
| - Time-to-Failure (TTF) lead warning latency                                            |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| Module 8: Interactive Monitoring Dashboard (Streamlit)                                  |
| - Live bounding box overlays with reliability health color coding (Green/Amber/Red)     |
| - Multi-horizon forecast radar (t+1, t+3, t+5 degradation warning)                      |
| - Object-level temporal reliability trajectories & system telemetry                     |
+-----------------------------------------------------------------------------------------+
```

---

## 4. Module-by-Module Communication & Data Contracts

Each module interacts with adjacent modules via strict dataclass contracts to ensure modularity, decoupled testing, and production maintainability.

### Module 1: Video Frame Ingestion (`src/detection/frame_extractor.py`)
- **Input:** Video stream path or camera device index.
- **Processing:** Decodes frames sequentially using OpenCV; extracts frame timestamps and global visual metrics (mean brightness, global sharpness index).
- **Output Data Contract:** `FramePacket`
  ```python
  @dataclass
  class FramePacket:
      frame_id: int
      timestamp_ms: float
      image: np.ndarray  # HxWxC uint8
      global_sharpness: float
      global_brightness: float
  ```

### Module 2: Object Detection (`src/detection/yolo_detector.py`)
- **Input:** `FramePacket`
- **Processing:** Forward pass through YOLO (YOLOv8 nano/small for edge CPU efficiency). Filters raw logits through bounding box decoding and NMS.
- **Output Data Contract:** `DetectionSet`
  ```python
  @dataclass
  class Detection:
      bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] normalized or absolute
      confidence: float
      class_id: int
      class_name: str

  @dataclass
  class DetectionSet:
      frame_id: int
      timestamp_ms: float
      detections: List[Detection]
  ```

### Module 3: Object Tracking (`src/tracking/byte_tracker.py`)
- **Input:** `DetectionSet` + current `FramePacket`
- **Processing:** ByteTrack associates both high-score and low-score detections using IoU distance and Kalman filter predictions. Maintains continuous `track_id` assignments across occlusions.
- **Output Data Contract:** `TrackedObjectSet`
  ```python
  @dataclass
  class TrackedObject:
      track_id: int
      bbox: Tuple[float, float, float, float]
      confidence: float
      class_id: int
      velocity: Tuple[float, float]  # (vx, vy) pixels/frame
      age: int                       # Total active frames
      time_since_update: int         # Consecutive missed frames
  ```

### Module 4: Feature Extraction & Temporal Sequencing (`src/features/feature_extractor.py`)
- **Input:** `TrackedObjectSet` + `FramePacket`
- **Processing:** For each active track ID, extracts a 16-dimensional feature vector:
  1. `confidence`: Instantaneous detector confidence score.
  2. `delta_confidence`: First temporal derivative of confidence ($C_t - C_{t-1}$).
  3. `rolling_conf_std`: Standard deviation of confidence over past 5 frames (flicker metric).
  4. `bbox_area_norm`: Bounding box area normalized by frame area (scale indicator).
  5. `delta_area`: Rate of change in bounding box area (approach velocity).
  6. `aspect_ratio`: Width-to-height ratio ($w/h$).
  7. `delta_aspect_ratio`: Distortion of aspect ratio (occlusion indicator).
  8. `velocity_x`, `velocity_y`: Kinematic motion vectors.
  9. `acceleration_norm`: Second temporal derivative of centroid motion.
  10. `border_proximity`: Normalized minimum distance to frame edge (truncation risk).
  11. `crop_laplacian_var`: Variance of Laplacian on object crop (motion/optical blur).
  12. `crop_contrast`: RMS contrast within object bounding box.
  13. `crop_entropy`: Shannon entropy of crop pixel intensities.
  14. `occlusion_overlap_ratio`: Max IoU overlap with adjacent detected bounding boxes.
  15. `track_age`: Longevity of current track.
  16. `missed_frame_ratio`: Ratio of missed to total expected detections.
- **Output Data Contract:** `TemporalFeatureSequence`
  ```python
  @dataclass
  class TemporalFeatureSequence:
      track_id: int
      frame_id: int
      feature_tensor: np.ndarray  # Shape [W=10, D=16]
  ```

### Module 5: Ground-Truth Reliability Generator (`src/evaluation/reliability_labeler.py`)
- **Input:** `TrackedObjectSet` at $t$, Ground-Truth Annotations at $t+1, t+3, t+5$.
- **Processing:** Performs bipartite Hungarian matching based on spatial IoU between tracked predicted boxes and verified ground-truth labels at future horizons.
- **Output Data Contract:** `ReliabilityTarget`
  ```python
  @dataclass
  class ReliabilityTarget:
      track_id: int
      frame_id: int
      target_iou_h1: float  # Ground truth IoU at t+1
      target_iou_h3: float  # Ground truth IoU at t+3
      target_iou_h5: float  # Ground truth IoU at t+5
      is_degraded_h1: bool  # IoU < 0.50
      is_degraded_h3: bool
      is_degraded_h5: bool
  ```

### Module 6: Recurrent Forecasting Model (`src/prediction/reliability_model.py`)
- **Input:** Batch of temporal feature tensors of shape $[B, W, D]$ (e.g., $[B, 10, 16]$).
- **Processing:** 
  - Backbone: 2-layer Gated Recurrent Unit (GRU) or Long Short-Term Memory (LSTM) with hidden dimension 64, dropout 0.2.
  - Forecasting Head: Multi-output linear projection with Sigmoid activation for continuous IoU prediction $\hat{R}_{t+h} \in [0, 1]$ and optional classification heads for failure risk alerts.
- **Output:** Multi-horizon forecast matrix $\hat{\mathbf{R}} \in [B, 3]$ for $h \in \{1, 3, 5\}$.

---

## 5. Evaluation Metrics for Production & Research

To evaluate predictive reliability rigorously, we employ dual evaluation criteria:

### 5.1 Continuous Regression Metrics
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** across each horizon:
  $$\text{RMSE}_h = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (R_{t+h}^{(i)} - \hat{R}_{t+h}^{(i)})^2}$$
- **Mean Absolute Error (MAE)**:
  $$\text{MAE}_h = \frac{1}{N} \sum_{i=1}^{N} |R_{t+h}^{(i)} - \hat{R}_{t+h}^{(i)}|$$

### 5.2 Failure Anticipation Classification Metrics
A drop in reliability below $\theta = 0.50$ represents a safety hazard. We evaluate the model's ability to trigger early warnings:
- **Precision:** Of all predicted reliability drops, how many were genuine?
- **Recall (Detection Failure Capture Rate):** Of all actual perception drops, what percentage was predicted in advance?
- **F1-Score:** Harmonic balance of precision and recall.
- **Area Under ROC Curve (ROC-AUC):** Discriminative ability across varying alert thresholds.
- **Lead Time Warning (Latency):** Verification that warnings are delivered at least 100ms ($h \ge 3$) before track loss.

---

## 6. Streamlit Interactive Dashboard Specification

The dashboard (`dashboard/app.py`) provides human-in-the-loop inspection and real-time visualization:
1. **Perception Feed Panel:** Displays live video playback with bounding boxes color-coded by predicted future reliability:
   - **Green:** High Future Reliability ($\hat{R} \ge 0.75$)
   - **Amber:** Moderate Reliability Warning ($0.50 \le \hat{R} < 0.75$)
   - **Red:** Critical Reliability Drop Alert ($\hat{R} < 0.50$)
2. **Object Reliability Telemetry:** Graphs historical detection confidence side-by-side with predicted reliability over time for selected track IDs.
3. **Forecasting Horizon Radar:** Shows predicted reliability trajectory across $t+1$, $t+3$, and $t+5$.
4. **Perception Health Metrics:** System-level false warning rate, mean localization stability, and frame processing FPS.

---

## 7. Hardware & Deployment Constraints

The design accounts for edge constraints (such as Intel Core i3 development workstations and edge compute modules):
- **CPU Optimization:** Small-footprint models (YOLOv8n + PyTorch CPU GRU) consuming < 2GB RAM.
- **No Hard GPU Lock:** All PyTorch tensors and modules default gracefully to CPU if CUDA is not present.
- **Decoupled Training & Inference:** Heavy offline sequence generation is separated from online streaming inference.

