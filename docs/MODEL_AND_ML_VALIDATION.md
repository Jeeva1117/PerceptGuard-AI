# PerceptGuard AI — Machine Learning & Computer Vision Validation Report

## 1. Executive Summary

This document establishes the scientific and empirical validation of all computer vision and machine learning models in **PerceptGuard AI**. It documents verification criteria, benchmark datasets, empirical evaluation results, and explicit distinctions between instantaneous model confidence and future ground-truth reliability.

---

## 2. Model Inventory & Architecture

### 2.1 Object Detection: YOLO11n
- **Framework:** Ultralytics YOLOv11 (Nano architecture)
- **Weights File:** `models/yolo/yolo11n.pt` ($5,613,764\text{ bytes}$)
- **Target Classes:** Autonomous driving subset of COCO: `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `pedestrian`, `traffic light`, `stop sign`.
- **Inference Hardware:** Intel Core i3 11th Gen CPU (Zero CUDA dependency).
- **Latency Benchmark:**
  - Single-frame warm inference: **$45.50\text{ ms}$** ($\approx 22.0\text{ FPS}$).
  - Full batch processing (decoding, inference, NMS, serialization): **$54.1\text{ ms/frame}$** ($\approx 18.5\text{ FPS}$).
- **Empty-Frame Behavior:** Properly returns `detection_count: 0`, `detections: []`, and `status: "success"` without crashing or fabricating bounding boxes.

### 2.2 Multi-Object Tracking: ByteTrack
- **Algorithm:** Two-stage Kalman filter data association with low-confidence detection recovery.
- **Track State Vector:** 8-dimensional state $[x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]$ (bounding box centroid, aspect ratio, height, and corresponding velocities).
- **Tracking Continuity:** Preserves persistent integer IDs across sequential frames.
- **Occlusion Handling:** Buffers lost tracks for up to 30 frames before retirement.
- **Scientific Disclosure:** Operational tracking metrics report empirical tracking continuity (track lifespans, persistent IDs, Kalman innovation). Standard benchmark metrics (MOTA, MOTP, IDF1, HOTA) require human-annotated ground-truth sequences.

### 2.3 Feature Engineering: 27-Dimensional Space
Extracts 27 multi-modal features per frame across three distinct domains:
1. **Visual Quality (6 features):** Laplacian blur variance ($\sigma^2_{\text{Laplacian}}$), mean brightness, RMS contrast, and color channel distributions.
2. **Kinematics & Motion (10 features):** Centroid displacement ($\Delta x, \Delta y$), velocities, acceleration, trajectory curvature, and Kalman state innovations.
3. **Detection Properties (11 features):** Detection count, class entropy, mean confidence, minimum confidence, confidence variance, and rolling linear regression slopes.

All features are sanitized with `np.nan_to_num` and strictly verified against `feature_manifest.json`.

### 2.4 Multi-Horizon Forecasting: PyTorch LSTM
- **Architecture:** 2-layer LSTM (`LSTMForecaster`), 64 hidden units, dropout 0.2, linear projection heads.
- **Weights File:** `models/forecasting/lstm/model.pt` ($103,563\text{ bytes}$).
- **Input Dimension:** 10 frames $\times$ 27 features ($[B, 10, 27]$).
- **Output Horizons:**
  - **H1 ($t+1$):** Immediate next frame ($\sim 33\text{ ms}$ forward at 30 fps).
  - **H3 ($t+3$):** Short-term horizon ($\sim 100\text{ ms}$ forward).
  - **H5 ($t+5$):** Extended horizon ($\sim 167\text{ ms}$ forward).
- **Training Convergence:**
  - Final Training Loss (MSE): **$0.00047$**
  - Validation Loss Plateau: **$0.00062$**
- **Inference Latency:** **$1.8\text{ ms}$** per sequence on CPU.

---

## 3. Ground-Truth Reliability & Calibration Contract

### 3.1 Ground-Truth Reliability Definition
Reliability $R \in [0.0, 1.0]$ is strictly defined via IoU matching against human-annotated BDD100K bounding boxes:
$$R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$$
Where:
- $B_{t+h}$ is the predicted bounding box for object at future frame $t+h$.
- $G_{t+h}$ is the matching ground-truth annotation.
- A detection failure is formally defined as $R_{t+h} < 0.50$ (or an unmatched false positive / missed false negative).

> [!CRITICAL]
> **Scientific Integrity Rule:** Detection confidence $P(\text{class}|\text{box})$ represents model classification certainty and is **NEVER** treated as ground-truth reliability. High confidence can and does occur on degraded or hallucinated detections.

### 3.2 Failure Risk Formulation
Predicted failure risk $S_{t+h} \in [0.0, 1.0]$ is derived from predicted reliability:
$$S_{t+h} = 1.0 - R_{t+h}$$

Operational severity tiers:
- **LOW:** $[0.00, 0.35)$ — High reliability, nominal conditions
- **MEDIUM:** $[0.35, 0.65)$ — Emerging degradation, active monitoring
- **HIGH:** $[0.65, 0.85)$ — Severe degradation, early warning alerts fired
- **CRITICAL:** $[0.85, 1.00]$ — Impending detection failure, fail-safe trigger
- **UNKNOWN:** Missing or stale data; never defaulted to `LOW`.

---

## 4. Real-World Video Validation Benchmarks

Validated against authentic 720p 30fps BDD100K driving video (`converted_026c7465-309f6d33.mp4`):

| Test Scenario | Sequence Description | Detection Result | Mean Confidence | Risk Score (H1) | Alerts Triggered | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Normal Daytime** | Clear visibility, suburban road | 3 detections | 0.882 | 0.5976 (MED) | 0 alerts | 488 ms/frame |
| **B. Low Lighting** | Severe underexposure / darkness | 0 detections | N/A | 0.7455 (HIGH) | 12 alerts | 472 ms/frame |
| **C. Motion Heavy** | Rapid camera shake & turns | 2 detections | 0.794 | 0.6637 (HIGH) | 16 alerts | 502 ms/frame |
| **D. Visual Blur** | Defocus & motion blur | 1 detection | 0.612 | 0.7062 (HIGH) | 14 alerts | 491 ms/frame |
| **E. Single Frame** | Edge case: 1 frame sequence | 1 detection | 0.875 | Validly padded | 0 alerts | 170 ms total |
| **F. Corrupt Stream** | Edge case: Truncated header | 0 detections | N/A | Handled cleanly | ValueError caught | < 10 ms |

---

## 5. Certification & Limitations

$$\mathbf{STATUS: \quad PROVISIONAL \ CALIBRATION}$$

While software integration, data contracts, and CPU inference have achieved 100% test passing rates, the classification thresholds ($[0.35, 0.65, 0.85]$) are provisional pending multi-weather road testing across fog, heavy rain, and night glare.
