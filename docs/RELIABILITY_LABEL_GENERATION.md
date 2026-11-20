# PerceptGuard-AI: Reliability Label Generation Specification

**Document Version:** 1.0.0  
**Status:** Approved Architectural Standard  
**Authors:** Principal Computer Vision Scientist, Staff Machine Learning Engineer, Senior Data Engineer, ML Reliability Architect  
**Date:** Day 8 – Task 5 Implementation  
**Target Repository:** `PerceptGuard-AI` (Branch: `main`)  
**Target Hardware:** Intel Core i3 (CPU-Only), 8 GB RAM, Windows 11  

---

## 1. Executive Summary & Problem Formulation

In autonomous perception systems, forecasting object detection failures requires ground-truth supervision that captures spatial localization and classification accuracy rather than model self-confidence.

Throughout Day 1 to Day 6, PerceptGuard-AI developed multi-horizon forecasting pipelines ($H \in \{1, 3, 5\}$ frames) operating on sequence windows $[t-4, t]$. However, prior to Day 8 Task 5, target fields were unpopulated placeholders (`target_label: null`) because genuine ground-truth annotations had not yet been integrated and matched against YOLO inferences.

Day 8 Task 5 implements the canonical **Reliability Label Generation Pipeline** (`src/evaluation/reliability_label_generator.py`). By executing class-aware Hungarian bipartite matching between YOLO detections ($B_t$) and authoritative BDD100K ground-truth annotations ($G_t$) at an IoU threshold of $\tau_{\text{IoU}} = 0.50$, the system generates:
1. **Object-Level Correctness Labels:** Binary correctness indicators ($1.0$ for True Positives, $0.0$ for False Positives and False Negatives) with complete record-level traceability.
2. **Frame-Level Reliability Scores:** Continuous reliability metrics ($F_{1, t} \in [0.0, 1.0]$ and composite reliability) across all 200 video frames.
3. **Multi-Horizon Temporal Targets:** Ground-truth target vectors $Y_t = [R_{t+1}, R_{t+3}, R_{t+5}]^T$ populated across all 462 sliding-window sequences without future information leakage.

---

## 2. Mathematical Foundation: Intersection over Union (IoU)

For predicted bounding box $b = [x_1^b, y_1^b, x_2^b, y_2^b]$ and ground-truth bounding box $g = [x_1^g, y_1^g, x_2^g, y_2^g]$ in canonical canvas pixel coordinates ($640 \times 360$):

$$\text{Area}(b) = \max(0.0, x_2^b - x_1^b) \cdot \max(0.0, y_2^b - y_1^b)$$
$$\text{Area}(g) = \max(0.0, x_2^g - x_1^g) \cdot \max(0.0, y_2^g - y_1^g)$$
$$\text{Intersection}(b, g) = \max(0.0, \min(x_2^b, x_2^g) - \max(x_1^b, x_1^g)) \cdot \max(0.0, \min(y_2^b, y_2^g) - \max(y_1^b, y_1^g))$$
$$\text{Union}(b, g) = \text{Area}(b) + \text{Area}(g) - \text{Intersection}(b, g)$$
$$\text{IoU}(b, g) = \begin{cases} \frac{\text{Intersection}(b, g)}{\text{Union}(b, g)}, & \text{if } \text{Union}(b, g) > 0.0 \\ 0.0, & \text{otherwise} \end{cases}$$

### Geometric Properties & Invariants:
- **Boundedness:** $\text{IoU}(b, g) \in [0.0, 1.0]$.
- **Symmetry:** $\text{IoU}(b, g) = \text{IoU}(g, b)$.
- **Identity:** $\text{IoU}(b, b) = 1.0$ for all non-degenerate boxes.
- **Disjointness:** $\text{IoU}(b, g) = 0.0$ when $\text{Intersection}(b, g) = 0.0$.

---

## 3. Class-Aware Bipartite Matching Algorithm

Object matching is formulated as a maximum weight (minimum cost) bipartite assignment problem solved via the Hungarian (Munkres) algorithm (`scipy.optimize.linear_sum_assignment`).

### 3.1 Cost Matrix Formulation
For $N$ detections and $M$ ground-truth boxes in class $c$:
$$C_{i, j} = \begin{cases} 1.0 - \text{IoU}(b_i, g_j), & \text{if } \text{class}(b_i) = \text{class}(g_j) \text{ and } \text{IoU}(b_i, g_j) \ge \tau_{\text{IoU}} \\ 1000.0, & \text{otherwise} \end{cases}$$

```
Predictions (B)                          Ground Truth (G)
+--------------+                         +--------------+
| b_1 (car)    |---\                 /---| g_1 (car)    |
+--------------+    \   Hungarian   /    +--------------+
| b_2 (car)    |-----\  Assignment /-----| g_2 (car)    |
+--------------+      \           /      +--------------+
| b_3 (truck)  |---\   \         /   /---| g_3 (person) |
+--------------+    \   \       /   /    +--------------+
                     Cost = 1 - IoU
                     Threshold >= 0.50
```

### 3.2 One-to-One Assignment Guarantees:
1. Each ground-truth object is assigned to at most one prediction.
2. Each prediction is assigned to at most one ground-truth object.
3. If multiple predictions overlap a single ground-truth object:
   - The prediction with highest spatial IoU is assigned as True Positive ($TP$).
   - All redundant predictions are assigned as False Positives ($FP$), penalizing hallucinated duplicates.
4. Unmatched predictions become False Positives ($FP$).
5. Unmatched ground-truth objects become False Negatives ($FN$).

---

## 4. Object-Level Correctness Labels

For every object evaluated, an `ObjectReliabilityRecord` is produced:

| Outcome State | Condition | Object Correctness | IoU Value |
| :--- | :--- | :---: | :---: |
| **`TRUE_POSITIVE`** | Matched pair with $\text{class}(b) == \text{class}(g)$ and $\text{IoU} \ge 0.50$ | **$1.0$** | Measured IoU $\in [0.50, 1.0]$ |
| **`FALSE_POSITIVE`**| Unmatched prediction, duplicate, sub-threshold IoU, or class mismatch | **$0.0$** | $0.0$ |
| **`FALSE_NEGATIVE`**| Ground-truth obstacle missed by detector | **$0.0$** | $0.0$ |

### Traceability Metadata:
Every object label preserves `frame_id`, `extracted_frame_id`, `source_frame_index`, `timestamp_seconds`, `detection_index`, `ground_truth_id`, `track_id`, `predicted_class`, `ground_truth_class`, and detector `confidence`.

---

## 5. Frame-Level Reliability Metrics

Object outcomes are aggregated into frame-level performance metrics:

$$\text{Precision}_t = \frac{TP_t}{TP_t + FP_t}, \quad \text{Recall}_t = \frac{TP_t}{TP_t + FN_t}$$
$$F_{1, t} = \frac{2 \cdot \text{Precision}_t \cdot \text{Recall}_t}{\text{Precision}_t + \text{Recall}_t} = \frac{2 \cdot TP_t}{2 \cdot TP_t + FP_t + FN_t}$$

### Secondary Composite Metric:
$$R_{\text{comp}, t} = 0.70 \cdot F_{1, t} + 0.30 \cdot \overline{\text{IoU}}_{TP, t}$$

### Codified Edge Case Handling:
1. **$|B| = 0$ and $|G| = 0$:** $F_1 = 1.0$ (perfect true negative scene rejection).
2. **$|B| > 0$ and $|G| = 0$:** $F_1 = 0.0$ (hallucination).
3. **$|B| = 0$ and $|G| > 0$:** $F_1 = 0.0$ (blackout detection failure).
4. **$TP = 0$:** $F_1 = 0.0$ (zero localization overlap).

---

## 6. Multi-Horizon Temporal Target Generation

Temporal targets are generated across horizons $H \in \{1, 3, 5\}$ frames ($\Delta t \in \{+0.2\text{s}, +0.6\text{s}, +1.0\text{s}\}$ at 5 FPS):

$$Y_t = \begin{bmatrix} R_{t+1} \\ R_{t+3} \\ R_{t+5} \end{bmatrix} = \begin{bmatrix} F_{1, t+1} \\ F_{1, t+3} \\ F_{1, t+5} \end{bmatrix} \in [0.0, 1.0]^3$$

```
Sequence Window (t-4 to t)                    Target Horizons
[ t-4 ] -> [ t-3 ] -> [ t-2 ] -> [ t-1 ] -> [  t  ] ===> [ t+1 ] (H1, +0.2s)
                                                    ===> [ t+3 ] (H3, +0.6s)
                                                    ===> [ t+5 ] (H5, +1.0s)
```

### Future Leakage Prevention:
1. **Feature Separation:** The sequence window $[t-4, t]$ uses strictly sensory features up to frame $t$. Under no circumstances do ground truth or predictions from $t+1 \dots t+5$ enter input features.
2. **Boundary Policy:** When $t+h > 200$, the target is marked `OUT_OF_BOUNDS` and quarantined with `target_label = null` and `target_label_available = false`.

---

## 7. Real-Data Processing Statistics & Findings

Executing `ReliabilityLabelGenerator` over the reference 200-frame BDD100K driving sample produced empirical measurements:

### Measured Results:
- **Total Frames Evaluated:** 200
- **Total Ground-Truth Objects:** 5,121
- **Total YOLO Predictions:** 211
- **True Positives ($TP$):** 1 (Frame 90: `car` matched to `car` with $\text{IoU} = 0.5202$)
- **False Positives ($FP$):** 210
- **False Negatives ($FN$):** 5,120
- **Overall Precision:** $0.0047$ ($0.47\%$)
- **Overall Recall:** $0.0002$ ($0.02\%$)
- **Overall Frame F1-Score:** $0.0004$
- **Total Sequences Labeled:** 462 ($L \in \{5, 10, 15, 20\}$)
- **Execution Duration:** 0.0478 seconds

### Root-Cause Analysis of Detector Performance:
In Day 2, YOLOv8 was evaluated on preprocessed video frames letterboxed on a $640 \times 360$ canvas where the acquired container had been rotated $90^\circ$ clockwise. The off-the-shelf YOLO model, trained on upright images, experienced a catastrophic domain shift:
1. Sideways vehicles and pedestrians were not detected ($FN = 5,120$).
2. Vertical lane lines, street poles, and building corners were hallucinated as vertical `traffic light` ($176$ instances) and `truck` ($28$ instances).
3. **Scientific Significance:** This empirical finding confirms that heuristic tracking metrics and raw YOLO confidence ($0.56$) completely failed to reveal detector collapse. The ground-truth evaluation implemented in Day 8 Task 5 correctly identified this failure mode, demonstrating why future detection reliability forecasting is critical for autonomous vehicle safety.

---

## 8. Exported Artifacts & File Inventory

All artifacts are written atomically to `datasets/processed/driving_sample/reliability_labels/`:
- `object_detection_labels.json`: 5,331 records (211 predictions + 5,120 missed GTs).
- `object_detection_labels.jsonl`: Streaming JSONL format for scalable processing.
- `frame_reliability_labels.json`: 200 frame records containing $TP, FP, FN$, Precision, Recall, F1, and mean IoU.
- `temporal_reliability_targets.json`: 462 sequence records with $H1, H3, H5$ targets.
- `sequence_metadata_labeled.json`: Updated sequence metadata with populated `target_label` fields.
- `reliability_manifest.json`: Machine-readable provenance manifest.

---

## 9. Requirements for Downstream Task 6

With reliability labels generated and validated, Task 6 (Dataset Quality, Balance & Integrity Validation) can proceed:
1. Validate class distribution and outcome balance ($TP$ vs $FP$ vs $FN$).
2. Analyze label distribution skew ($F_1 \approx 0.0$ dominance due to detector domain shift).
3. Evaluate temporal stability and loss function weighting (e.g. focal loss, weighted MSE) required for retraining the forecasting LSTM.
