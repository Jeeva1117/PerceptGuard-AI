# PerceptGuard-AI: Annotation Schema & Reliability Label Design Specification

**Document Version:** 1.0.0  
**Status:** Approved Architectural Standard  
**Authors:** Principal Computer Vision Scientist, Staff ML Engineer, Senior Data Architect, ML Reliability Research Engineer  
**Date:** Day 8 – Task 3 Implementation  
**Target Repository:** `PerceptGuard-AI` (Branch: `main`)  
**Target Hardware:** Intel Core i3 (CPU-Only), 8 GB RAM, Windows 11  

---

## 1. Executive Summary

This document establishes the canonical data schemas, mathematical formulations, spatial matching specifications, and temporal target definitions required to transition PerceptGuard-AI from heuristic tracking proxies to scientifically defensible, ground-truth-validated object detection reliability forecasting.

In Day 1 through Day 6, the system demonstrated functional multi-horizon forecasting ($H \in \{1, 3, 5\}$ frames, corresponding to $+0.2\text{s}, +0.6\text{s}, +1.0\text{s}$ at 5 FPS) utilizing an LSTM architecture. However, as audited in Day 8 Task 1 and Task 2:
1. Target labels in `data/processed/temporal_sequences/sequence_metadata.json` were unpopulated placeholders (`target_label: null`).
2. Downstream validation lacked ground-truth alignment with human-annotated bounding boxes.
3. Tracking metrics (ByteTrack persistence, Kalman filter state) measured association stability rather than true perceptual accuracy.

This specification formalizes the foundational contract between model predictions ($B_t$), human ground truth ($G_t$), spatial bipartite matching ($M_t$), and temporal forecasting targets ($Y_t$). It guarantees mathematical rigor, data leakage prevention, strict schema compliance, and reproducible offline validation on resource-constrained CPU hardware.

---

## 2. Mathematical Foundation: Why Confidence Is Not Ground-Truth Reliability

A critical failure mode in deployed autonomous perception systems is equating **detection confidence** ($s_{\text{det}} \in [0, 1]$) or **tracking continuity** with **ground-truth reliability** ($R \in [0, 1]$).

### 2.1 The Overconfidence Problem in Deep Neural Networks
Modern convolutional and transformer-based object detectors (including Ultralytics YOLOv8) employ softmax activation functions over non-calibrated classification logits and apply non-maximum suppression (NMS) thresholds. These models are notorious for overconfidence:
$$\hat{p} = \max_{k} \sigma(z_k) \gg P(\hat{y} = y^* \mid X)$$

Under adverse driving conditions—such as heavy rain, lens flare, optical distortion, motion blur, and out-of-distribution occlusions:
- Detectors often predict phantom bounding boxes with confidences exceeding $0.85$ (hallucination).
- Detectors often drop small, low-contrast, or partially occluded pedestrians and vehicles entirely, yielding confidence $0.0$ despite critical safety hazards (missed detection / false negative).
- Bounding box coordinates drift significantly under heavy motion, while confidence remains artificially elevated.

### 2.2 The Association Problem in Multi-Object Tracking
ByteTrack and Kalman filter pipelines maintain trajectory continuity by associating detections across frames using bounding box velocity and spatial proximity. However:
- A tracker will happily track a false-positive hallucination indefinitely as long as subsequent frames produce spatially coherent noise.
- A tracker maintaining track continuity for 100 consecutive frames does not imply the detection covers the true object; it merely implies the prediction is temporally smooth.
- Tracking metrics like MOTA or IDF1 require ground truth to evaluate; tracker internal states cannot serve as their own ground truth.

### 2.3 The Ground-Truth Principle
True perceptual reliability is strictly defined by spatial and semantic agreement between predicted hypothesis bounding boxes $B_t = \{b_1, \dots, b_N\}$ and authoritative, human-verified ground-truth annotations $G_t = \{g_1, \dots, g_M\}$:
$$R_t = f(B_t, G_t) \in [0.0, 1.0]$$

Reliability cannot be observed from predictions alone; it requires verification against independent empirical ground truth.

---

## 3. Canonical Ground-Truth Annotation Schema

The ground-truth annotation schema standardizes human annotations ingested from the BDD100K driving dataset (or equivalent production sources) onto the canonical PerceptGuard coordinate system.

### 3.1 Coordinate System & Canvas Alignment
- **Format:** Pixel coordinate 4-tuple $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
- **Origin:** Top-left corner of image frame $(0, 0)$.
- **Resolution:** Native inference canvas of $640 \times 360$ pixels (16:9 aspect ratio).
- **Geometric Invariants:**
  $$0.0 \le x_{\min} < x_{\max} \le W = 640.0$$
  $$0.0 \le y_{\min} < y_{\max} \le H = 360.0$$
  $$\text{Area} = (x_{\max} - x_{\min}) \cdot (y_{\max} - y_{\min}) > 0.0$$

### 3.2 Canonical Ground-Truth Field Specification

| Field Name | Type | Unit / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `annotation_id` | `str` | UUID or sequence hash | Unique deterministic identifier for the ground-truth box |
| `extracted_frame_id`| `str` | Format: `frame_XXXXXX.jpg` | Foreign key matching the preprocessed video frame filename |
| `frame_index` | `int` | $\ge 0$ | Zero-indexed continuous frame sequence counter |
| `timestamp_seconds` | `float` | $\ge 0.0$, step: $0.2\text{s}$ at 5 FPS | Video presentation timestamp |
| `bbox_xyxy` | `List[float]` | Length 4, floats | Bounding box coordinates in $640 \times 360$ canvas space |
| `category` | `str` | Normalized string | Mapped COCO/PerceptGuard class (e.g., `car`, `truck`, `bus`, `pedestrian`) |
| `class_id` | `int` | $[0, 79]$ | Numerical class identifier matching YOLO ontology |
| `track_id` | `Optional[int]`| $\ge 0$ or `null` | Ground-truth object track identity across consecutive frames |
| `attributes` | `Dict[str, Any]`| Key-value dictionary | Metadata (occluded, truncated, crowd, weather, timeofday) |

---

## 4. Detection Prediction Schema

The detection schema captures raw inferences output by the YOLOv8 object detector prior to temporal aggregation.

### 4.1 Canonical Detection Field Specification

| Field Name | Type | Unit / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `detection_id` | `str` | Unique string | Unique identifier for detection instance |
| `extracted_frame_id`| `str` | Format: `frame_XXXXXX.jpg` | Foreign key matching preprocessed video frame |
| `frame_index` | `int` | $\ge 0$ | Continuous frame counter |
| `timestamp_seconds` | `float` | $\ge 0.0$ | Video presentation timestamp |
| `bbox_xyxy` | `List[float]` | Length 4, floats | Detected bounding box coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ |
| `class_name` | `str` | String | Predicted class label |
| `class_id` | `int` | $[0, 79]$ | Numerical class identifier |
| `confidence` | `float` | $[0.0, 1.0]$ | Model detector confidence score |
| `image_width` | `int` | 640 | Canvas width |
| `image_height` | `int` | 360 | Canvas height |

---

## 5. Object-Level Spatial Matching Specification

To evaluate whether a prediction correctly detects a real object, predictions and ground truth must be matched spatially and semantically.

### 5.1 Intersection over Union (IoU) Formulation
Given predicted bounding box $b = [x_{1}^b, y_{1}^b, x_{2}^b, y_{2}^b]$ and ground-truth box $g = [x_{1}^g, y_{1}^g, x_{2}^g, y_{2}^g]$:
$$\text{Area}(b) = (x_{2}^b - x_{1}^b) \cdot (y_{2}^b - y_{1}^b)$$
$$\text{Area}(g) = (x_{2}^g - x_{1}^g) \cdot (y_{2}^g - y_{1}^g)$$
$$x_{1}^i = \max(x_{1}^b, x_{1}^g), \quad y_{1}^i = \max(y_{1}^b, y_{1}^g)$$
$$x_{2}^i = \min(x_{2}^b, x_{2}^g), \quad y_{2}^i = \min(y_{2}^b, y_{2}^g)$$
$$\text{Intersection}(b, g) = \max(0.0, x_{2}^i - x_{1}^i) \cdot \max(0.0, y_{2}^i - y_{1}^i)$$
$$\text{Union}(b, g) = \text{Area}(b) + \text{Area}(g) - \text{Intersection}(b, g)$$
$$\text{IoU}(b, g) = \frac{\text{Intersection}(b, g)}{\text{Union}(b, g)}$$

If $\text{Union}(b, g) \le 0.0$, $\text{IoU}(b, g) \triangleq 0.0$.

### 5.2 Matching Strategies: Hungarian Bipartite vs Greedy Matching

#### Greedy Matching
- Iteratively pairs the prediction and ground-truth box with highest pairwise IoU.
- **Limitation:** Can lead to suboptimal global assignments in dense traffic clusters where a slightly higher local IoU prevents an overall optimal pairing.

#### Hungarian (Munkres) Bipartite Matching (Recommended Standard)
- Formulates object assignment as a global linear sum assignment problem over cost matrix $C$:
  $$C_{i, j} = \begin{cases} 1.0 - \text{IoU}(b_i, g_j), & \text{if } \text{class}(b_i) = \text{class}(g_j) \text{ and } \text{IoU}(b_i, g_j) \ge \tau_{\text{IoU}} \\ \infty, & \text{otherwise} \end{cases}$$
- Solved in polynomial time via `scipy.optimize.linear_sum_assignment`:
  $$\min_{\pi} \sum_{i} C_{i, \pi(i)}$$
- Guarantees one-to-one matching: exactly one prediction matches at most one ground-truth box.
- Enforces duplicate detection penalties: if multiple predictions cover the same ground-truth object, the highest-IoU prediction becomes True Positive ($TP$), and all redundant predictions become False Positives ($FP$).

### 5.3 IoU Threshold Rationale ($\tau_{\text{IoU}} = 0.50$)
1. **PASCAL VOC / COCO Standard:** $\tau_{\text{IoU}} = 0.50$ is the universally recognized threshold for object localization success.
2. **Autonomous Driving Suitability:** At $\text{IoU} \ge 0.50$, over 70% of object dimensions are aligned, sufficient for trajectory planning and collision time-to-contact estimation.
3. **Resolution Robustness:** On downscaled $640 \times 360$ frames, small distant objects ($< 32\times 32$ pixels) experience pixel discretization noise; a stricter threshold (e.g., 0.75) introduces high variance due to 1-pixel boundary shifts.

---

## 6. Object-Level Reliability Formulation

For each individual predicted bounding box $b_i$:
$$R(b_i) = \begin{cases} \text{IoU}(b_i, g^*), & \text{if matched to ground-truth box } g^* \text{ with } \text{IoU} \ge \tau_{\text{IoU}} \\ 0.0, & \text{if unmatched (False Positive)} \end{cases}$$

For each ground-truth object $g_j$:
$$R(g_j) = \begin{cases} \text{IoU}(b^*, g_j), & \text{if matched by prediction } b^* \text{ with } \text{IoU} \ge \tau_{\text{IoU}} \\ 0.0, & \text{if unmatched (False Negative / Missed Detection)} \end{cases}$$

Every object is categorized into exactly one outcome state:
- **True Positive ($TP$):** Matched pair meeting class agreement and $\text{IoU} \ge \tau_{\text{IoU}}$.
- **False Positive ($FP$):** Predicted box unmatched, duplicate, or wrong class.
- **False Negative ($FN$):** Ground-truth box missed by detector.

---

## 7. Frame-Level Reliability Metric Comparison & Recommendation

To forecast system reliability at future frames $t+h$, object-level outcomes must be aggregated into an authoritative scalar reliability score $R_t \in [0.0, 1.0]$.

### 7.1 Evaluated Metric Candidates

#### Candidate 1: Frame-Level mAP@0.5
- **Definition:** Area under the precision-recall curve computed across confidence thresholds within a single frame.
- **Strengths:** Standard benchmark in batch object detection evaluation.
- **Fatal Weaknesses:** Single frames typically contain very few objects (e.g., 2 to 10). Computing 11-point interpolated AP on 3 objects produces discrete, non-smooth steps ($0.0, 0.33, 0.5, 0.66, 1.0$) that degrade neural network regression gradients. Undefined when frame contains 0 ground-truth objects.

#### Candidate 2: Mean IoU (mIoU)
- **Definition:** $\frac{1}{|TP|} \sum_{i \in TP} \text{IoU}(b_i, g_i)$.
- **Strengths:** Pure spatial localization metric.
- **Fatal Weaknesses:** Ignores False Positives and False Negatives entirely. A detector predicting 100 hallucinations and 1 correct box with $\text{IoU}=0.9$ would achieve $\text{mIoU} = 0.90$. Undefined when $TP = 0$.

#### Candidate 3: Frame F1-Score (Recommended Primary Metric)
- **Definition:** Harmonic mean of frame Precision and Recall:
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
  $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
- **Strengths:**
  - Simultaneously penalizes perceptual hallucinations ($FP$) and missed obstacles ($FN$).
  - Bounded in $[0.0, 1.0]$ with smooth operational behavior.
  - Aligns with safety-critical driving requirements: both false emergency braking (from $FP$) and collisions (from $FN$) are penalized.

#### Candidate 4: Composite Reliability Metric
- **Definition:** Weighted combination of detection count accuracy, localization overlap, and hallucination penalty:
  $$R_{\text{comp}} = w_1 \cdot F_1 + w_2 \cdot \overline{\text{IoU}}_{TP} + w_3 \cdot \max\left(0, 1.0 - \frac{FP}{N_{\text{pred}}}\right)$$
- **Weights:** $w_1 = 0.60, w_2 = 0.30, w_3 = 0.10$.
- **Strengths:** Rewards tight bounding box localization while maintaining $F_1$ safety bounds.

### 7.2 Metric Comparison Summary

| Metric | Bounded | Penalizes FP | Penalizes FN | Continuous Gradients | Handles Empty Frames | Recommendation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Frame F1-Score** | Yes $[0, 1]$ | Yes | Yes | High | Yes (Deterministic Rule) | **PRIMARY RECOMMENDED** |
| **Composite Metric**| Yes $[0, 1]$ | Yes | Yes | High | Yes (Deterministic Rule) | **SECONDARY / AUDIT** |
| **Mean IoU** | Yes $[0, 1]$ | No | No | Low | No (Division by zero) | Rejected as standalone |
| **Frame mAP@0.5** | Yes $[0, 1]$ | Yes | Yes | Step-like | No | Rejected for regression |

---

## 8. Empty Frame & Edge Case Handling Policy

Autonomous driving videos encounter edge cases that must be handled deterministically without runtime exceptions or numerical instabilities.

### 8.1 Edge Case Decision Matrix

| Ground Truth ($|G|$) | Predictions ($|B|$) | Matches ($TP$) | Frame $F_1$ Score | Semantic Meaning & Justification |
| :---: | :---: | :---: | :---: | :--- |
| **$0$** | **$0$** | **$0$** | **$1.0$** | **Perfect Rejection:** Empty scene correctly identified; zero false alarms. |
| **$0$** | **$> 0$** | **$0$** | **$0.0$** | **Pure Hallucination:** Detector predicts objects where none exist. |
| **$> 0$** | **$0$** | **$0$** | **$0.0$** | **Complete Blackout:** Detector missed all real objects in scene. |
| **$> 0$** | **$> 0$** | **$0$** | **$0.0$** | **Total Localization Failure:** Boxes present but zero spatial overlap. |
| **$> 0$** | **$> 0$** | **$> 0$** | $\frac{2 TP}{2 TP + FP + FN}$ | **Standard Detection:** Proportional balance of precision and recall. |

### 8.2 Geometric Validation & Sanitization Policies
1. **Coordinate Clipping:** Any raw coordinate $x < 0$ or $x > W$ is clipped to canvas bounds $[0, W]$; $y < 0$ or $y > H$ is clipped to $[0, H]$.
2. **Degenerate Bounding Boxes:** If $x_{\max} \le x_{\min}$ or $y_{\max} \le y_{\min}$, box area is non-positive. Degenerate boxes are rejected during schema validation with an explicit `ValueError`.
3. **NaN / Inf Guard:** Any floating-point NaN or Inf encountered during computation triggers an immediate validation failure.

---

## 9. Temporal Target Definition for Multi-Horizon Forecasting

The PerceptGuard-AI forecasting model operates on an input sequence of length $L = 5$ frames ($1.0\text{s}$ at 5 FPS) to predict future reliability across multiple horizons $H \in \{1, 3, 5\}$ frames.

```
Input History Window (t - 4 to t)                     Prediction Horizons
[ t-4 ] -> [ t-3 ] -> [ t-2 ] -> [ t-1 ] -> [  t  ] ===> [ t+1 ]  (h=1, +0.2s)
                                                    ===> [ t+3 ]  (h=3, +0.6s)
                                                    ===> [ t+5 ]  (h=5, +1.0s)
```

### 9.1 Multi-Horizon Target Options

#### Option A: Exact Future Frame Reliability (Recommended Primary Target)
$$Y_t = [R_{t+1}, R_{t+3}, R_{t+5}]^T \in [0.0, 1.0]^3$$
- Direct absolute estimation of detection quality at specific future timestamps.
- Preserves absolute magnitude: easily translated to safety thresholds (e.g., $R < 0.5 \implies \text{trigger fallback}$).
- Direct alignment with Day 4–6 multi-output LSTM architecture (`output_dim = 3`).

#### Option B: Reliability Degradation Delta
$$\Delta Y_t = [R_{t+1} - R_t, R_{t+3} - R_t, R_{t+5} - R_t]^T \in [-1.0, 1.0]^3$$
- Focuses on the rate of degradation (e.g., rapid drop upon entering a tunnel).
- Weakness: Predicting a delta of $-0.10$ does not distinguish between $1.0 \to 0.9$ (safe) and $0.2 \to 0.1$ (critical failure).

#### Option C: Binary Failure Indicator
$$I_t = [\mathbb{I}(R_{t+1} < \tau), \mathbb{I}(R_{t+3} < \tau), \mathbb{I}(R_{t+5} < \tau)]^T \in \{0, 1\}^3$$
- Discards continuous spatial information, reducing regression to binary classification.
- Highly sensitive to threshold $\tau$.

### 9.2 Recommendation
**Option A (Exact Future Reliability Vector)** is selected as the canonical target. It is continuous, directly interpretable, bounded, and preserves full geometric resolution.

---

## 10. Data Leakage Prevention Guarantees

In sequential perception forecasting, temporal data leakage invalidates experimental findings. The PerceptGuard-AI pipeline enforces four architectural leakage barriers:

```
+------------------------------------+       +-----------------------+
| Input Window: [t - L + 1, ..., t]  |       | Target: R_{t+h}       |
| Uses ONLY frames <= t              |       | Evaluated at t+h ONLY |
+------------------------------------+       +-----------------------+
                 |                                       ^
                 +----------------- NO LEAKAGE ----------+
```

1. **Feature Construction Boundary:** Input feature vectors $X_t$ are constructed strictly from sensory data available at or before frame $t$. No ground-truth annotations from frame $t$ or any future frame $t+k$ are included in $X_t$.
2. **Target Isolation Boundary:** Target label $R_{t+h}$ is computed strictly between YOLO predictions at frame $t+h$ and ground-truth annotations at frame $t+h$. It has no retroactive visibility into intermediate frames.
3. **Temporal Partitioning with Embargo Gap:**
   - Datasets are partitioned strictly by continuous temporal segments (e.g., Train: frames 0–140, Validation: 141–170, Test: 171–200).
   - A temporal embargo buffer of $G = 25$ frames ($5.0\text{s}$) is enforced between splits to eliminate autocorrelation leakage caused by persistent stationary objects.
4. **Scaler Isolation:** Feature scalers (`StandardScaler`, `MinMaxScaler`) are fit strictly on the training partition and transformed across validation/test partitions.

---

## 11. Data Contract & Validation Specification

All data structures are implemented as validated Pydantic models in `src/evaluation/reliability_schema.py`:

```
CanonicalBoundingBox
  |-- xmin, ymin, xmax, ymax (Canvas invariants validated)
  |-- confidence [0, 1]
  +-- class_id >= 0

ObjectReliabilityRecord
  |-- frame_id, object_id, class_name
  |-- outcome (TP, FP, FN)
  |-- iou [0, 1]
  +-- reliability_score [0, 1]

FrameReliabilityRecord
  |-- frame_id, timestamp_seconds
  |-- ground_truth_count, prediction_count
  |-- tp_count, fp_count, fn_count
  |-- precision, recall, f1_score [0, 1]
  |-- mean_iou [0, 1]
  +-- composite_reliability [0, 1]

TemporalTargetRecord
  |-- sequence_id, source_frame_id, timestamp_seconds
  |-- horizons {h1, h3, h5} -> HorizonTargetSpec
  +-- target_vector [r_h1, r_h3, r_h5]
```

### 11.1 Validation Checks Enforced by Contract
- `ReliabilityDataContract.validate_bounding_box(bbox, width, height)`: Enforces valid non-degenerate coordinates within canvas boundaries.
- `ReliabilityDataContract.validate_frame_record(record)`: Verifies mathematical integrity: $TP + FP = N_{\text{pred}}$, $TP + FN = N_{\text{gt}}$, and $F_1 \in [0, 1]$.
- `ReliabilityDataContract.validate_temporal_target(record)`: Enforces horizon completeness, non-null targets, and bounded output vectors.

---

## 12. Configuration Specification (`config/reliability_label_config.yaml`)

The labeling pipeline is fully parameter-driven via YAML configuration:

```yaml
version: "1.0.0"
dataset:
  name: "BDD100K Driving Sample"
  canvas_width: 640
  canvas_height: 360
  frame_rate_fps: 5.0

matching:
  strategy: "hungarian"        # hungarian or greedy
  iou_threshold: 0.50          # PASCAL VOC / COCO standard
  class_aware: true            # Must match class labels
  cost_metric: "iou"           # 1.0 - iou cost

metrics:
  primary: "f1_score"          # Primary training target
  secondary: "composite"       # Secondary audit metric
  composite_weights:
    w_f1: 0.60
    w_iou: 0.30
    w_fp_penalty: 0.10

temporal_horizons:
  horizons: [1, 3, 5]          # Target frames ahead
  time_offsets_sec: [0.2, 0.6, 1.0]
  input_sequence_length: 5     # L = 5 frames history
```

---

## 13. Mathematical Notation Reference Table

| Symbol | Mathematical Definition | Range | Context / Description |
| :---: | :--- | :---: | :--- |
| $B_t$ | $\{b_1, b_2, \dots, b_N\}$ | Set | Set of $N$ predicted bounding boxes at frame $t$ |
| $G_t$ | $\{g_1, g_2, \dots, g_M\}$ | Set | Set of $M$ human ground-truth boxes at frame $t$ |
| $b_i$ | $[x_{\min}^i, y_{\min}^i, x_{\max}^i, y_{\max}^i]$ | $\mathbb{R}^4$ | Bounding box coordinates in canvas pixel space |
| $\text{IoU}(b, g)$| $\frac{\text{Area}(b \cap g)}{\text{Area}(b \cup g)}$ | $[0.0, 1.0]$ | Intersection over Union spatial overlap |
| $\tau_{\text{IoU}}$| $0.50$ | Float | Minimum IoU threshold for valid localization match |
| $TP_t$ | $\sum_i \mathbb{I}(\text{matched}(b_i, g_j) \land \text{IoU} \ge \tau)$ | $\mathbb{Z}_{\ge 0}$ | Count of True Positive detections |
| $FP_t$ | $|B_t| - TP_t$ | $\mathbb{Z}_{\ge 0}$ | Count of False Positive detections |
| $FN_t$ | $|G_t| - TP_t$ | $\mathbb{Z}_{\ge 0}$ | Count of False Negative missed objects |
| $P_t$ | $\frac{TP_t}{TP_t + FP_t}$ | $[0.0, 1.0]$ | Frame detection precision |
| $R_{\text{rec}, t}$| $\frac{TP_t}{TP_t + FN_t}$ | $[0.0, 1.0]$ | Frame detection recall |
| $F_{1, t}$ | $\frac{2 \cdot P_t \cdot R_{\text{rec}, t}}{P_t + R_{\text{rec}, t}} = \frac{2 TP_t}{2 TP_t + FP_t + FN_t}$| $[0.0, 1.0]$ | Frame F1-score (Primary Reliability Metric) |
| $R_t$ | $f(B_t, G_t)$ | $[0.0, 1.0]$ | General frame-level detection reliability score |
| $H$ | $\{1, 3, 5\}$ | Set | Forecasting horizon frame steps |
| $Y_t$ | $[R_{t+1}, R_{t+3}, R_{t+5}]^T$ | $[0.0, 1.0]^3$ | Canonical multi-horizon target reliability vector |
| $L$ | $5$ | Integer | Input history sequence length (1.0 second at 5 FPS) |
| $G$ | $25$ | Integer | Temporal embargo buffer between train/val/test splits |

---

## 14. Impact on Downstream Task 4 (Label Generation) and Retraining

This design specification directly unblocks and governs Day 8 Task 4:
1. **Target Population:** In Task 4, the label generator will ingest `detection_results.json` ($B_t$) and `ground_truth_annotations.json` ($G_t$), run Hungarian matching via `reliability_schema.py`, and populate the missing `target_label` fields in `sequence_metadata.json`.
2. **Dataset Grounding:** Up to 186 temporal sequences will receive valid three-dimensional ground-truth target vectors $Y_t = [F_{1, t+1}, F_{1, t+3}, F_{1, t+5}]^T$.
3. **Model Retraining Readiness:** Day 4–6 baseline and LSTM forecasting models will transition from placeholder targets to real ground-truth targets, enabling rigorous training with MSE loss and honest offline validation against human labels.
