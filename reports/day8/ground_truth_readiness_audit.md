# PerceptGuard-AI: Ground-Truth Dataset Readiness Audit
**Document ID:** `reports/day8/ground_truth_readiness_audit.md`  
**Execution Timestamp:** 2026-11-20T06:20:00+05:30  
**Phase:** Day 8 – Task 1: Ground-Truth Dataset Readiness Audit  
**Auditor Roles:** Principal Computer Vision Engineer, Staff Machine Learning Engineer, Senior Data Engineer, ML Dataset Quality Architect  
**Hardware Profile:** Intel Core i3 (CPU Only), 8 GB RAM (7.75 GB usable), Windows 11 Build 26200  
**Repository:** [https://github.com/Jeeva1117/PerceptGuard-AI.git](https://github.com/Jeeva1117/PerceptGuard-AI.git)  
**Target Branch:** `main`

---

## 1. Executive Summary

An independent, rigorous Ground-Truth Dataset Readiness Audit was conducted across the **PerceptGuard-AI** repository to evaluate whether the project possesses the required data, ground-truth annotations, and coordinate alignment to construct a scientifically valid object detection reliability forecasting dataset.

The core scientific objective of PerceptGuard-AI is predicting future object detection reliability $R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$ across upcoming video frames ($H \in \{1, 3, 5\}$). A fundamental tenet of machine learning engineering is that model confidence $P(\text{class}|\text{box})$, tracking continuity, or optical heuristics must never be substituted for authentic ground truth.

### Key Audit Conclusions:
1. **Engineering Pipeline & Data Integrity: `PASS`**  
   The video extraction, letterbox preprocessing, YOLO11n object detection, ByteTrack tracking, multi-domain feature extraction, and temporal sequence assembly pipelines are fully operational, numerically stable, schema-compliant, and verified with zero corrupted frames, zero NaNs, zero Infs, and zero invalid bounding boxes.
2. **Ground-Truth Annotation Availability: `NOT AVAILABLE`**  
   A comprehensive scan of the repository discovered **zero** ground-truth annotation files across all industry-standard formats (YOLO TXT, COCO JSON, Pascal VOC XML, MOT CSV, BDD100K JSON). Only visualization debug artifacts with burned-in YOLO detection boxes exist.
3. **Labeling Readiness (Frame, Object, Temporal Levels): `NOT AVAILABLE`**  
   Because ground-truth bounding boxes $G_{t+h}$ do not exist in the repository, computing authentic Intersection over Union (IoU) labels at frame level, object level, or future forecasting horizons ($H \in \{1, 3, 5\}$) is physically impossible.
4. **Supervised Training Readiness: `BLOCKED`**  
   While PyTorch Dataset containers and the LSTM forecasting architecture are fully tested using synthetic pipeline verification tensors, live supervised model training on authentic driving targets is strictly gated until genuine ground-truth annotations are integrated.

---

## 2. Repository Inventory

The entire repository was independently audited. Directory structure and file presence were verified:

| Directory Path | Purpose / Subsystem | Status | Item Count / Notes |
|:---|:---|:---:|:---|
| `datasets/raw/` | Raw video source storage | **PASS** | 1 MP4 video file (`converted_026c7465-309f6d33.mp4`, 47.47 MB) |
| `datasets/processed/` | Processed driving datasets | **PASS** | 4 subdirectories: `driving_sample`, `driving_test`, `synthetic_test_seq`, `test` |
| `datasets/processed/driving_sample/` | Official 200-frame driving dataset | **PASS** | Frames, detections, tracking, features, sequences, manifests |
| `src/` | Core source codebase | **PASS** | 32 Python modules across 8 subpackages (`video_processing`, `detection`, `tracking`, `features`, `forecasting`, `analysis`, `validation`, `visualization`) |
| `config/` | System configuration YAML & Python | **PASS** | 6 configuration files (`yolo_config.yaml`, `tracker_config.yaml`, `video_config.yaml`, `feature_config.yaml`, `forecasting_config.yaml`, `settings.py`) |
| `tests/` | Comprehensive test suite | **PASS** | 57 test files covering all modules and regression suites |
| `reports/` | Engineering & release audit reports | **PASS** | 18 reports across `final_audit/`, `day6_audit/`, `day5_audit/`, and `github/` |
| `outputs/` | Test logs & validation reports | **PASS** | `outputs/reports/dataset_validation_report.json` |
| `models/` | Model weights & checkpoints | **PASS** | `models/yolo/yolo11n.pt` (local, Git-ignored) |
| `data/` | Legacy or external data path | **NOT AVAILABLE** | Does not exist; all data centralized in `datasets/` |
| `artifacts/` | Build or scratch directory | **NOT AVAILABLE** | Does not exist; outputs stored in designated subdirectories |

---

## 3. Available Datasets

The repository contains one primary genuine autonomous driving sample dataset and multiple lightweight synthetic validation test directories:

### 3.1 Primary Driving Sample (`driving_sample`)
- **Source Video:** `datasets/raw/converted_026c7465-309f6d33.mp4`
  - **Origin:** Berkeley DeepDrive (BDD100K) training split, sequence `026c7465-309f6d33`.
  - **File Size:** 49,774,590 bytes (47.47 MB).
  - **SHA-256 Checksum:** `12c6965c1ace511721e9ac2fbbb36d2723c5888d1dba698d5b8458adba08ad4b`.
  - **Video Properties:** 1,204 total frames, 30.09 FPS, duration ~40.01 seconds, resolution $720 \times 1280$ (portrait orientation, aspect ratio 9:16).
- **Extracted Frames (`frames/`):**
  - **Total Extracted:** 200 frames (`frame_000001.jpg` to `frame_000200.jpg`).
  - **Extraction FPS:** 5.0 FPS (temporal step $\Delta t \approx 6$ video frames, spanning ~40 seconds).
  - **Resolution:** $1280 \times 720$ ($H=1280, W=720$, 3 channels BGR).
  - **Storage Size:** 13.78 MB.
- **Preprocessed Frames (`preprocessed_frames/`):**
  - **Total Preprocessed:** 200 frames (`frame_000001.jpg` to `frame_000200.jpg`).
  - **Target Resolution:** $360 \times 640$ ($H=360, W=640$, 3 channels BGR).
  - **Transformation:** Center letterboxing with black padding (`[0, 0, 0]`). Scale factor $s = 0.28125$. Active content area: $W \approx 203$ pixels, horizontal offset $dx = 218.75$ pixels ($x \in [218.75, 421.25]$).
  - **Storage Size:** 3.51 MB.
  - **Visual Quality Baseline:** Average blur (Laplacian variance) = 436.75; average brightness = 102.05; average contrast = 62.04.
- **Object Detection Outputs (`detections/`):**
  - **Artifact:** `detections/batch_runs/run_20261117_115348_e1789b/detection_results.json`
  - **Inference Model:** YOLO11n (`yolo11n.pt`, confidence threshold $\tau = 0.25$, IoU threshold $= 0.45$).
  - **Frames Processed:** 200 frames.
  - **Total Detections:** 211 detections (mean 1.055 detections/frame).
  - **Class Distribution:** `traffic light`: 159, `truck`: 31, `car`: 7, `potted plant`: 6, `bus`: 3, `surfboard`: 2, `person`: 1, `train`: 1, `vase`: 1.
  - **Confidence Statistics:** Min = 0.2522, Max = 0.7449, Mean = 0.4215.
  - **Coordinate Space:** Canvas pixels $[218.7, 420.8] \times [0.8, 359.7]$ (strictly inside active letterbox region).
- **Tracking Outputs (`tracking/`):**
  - **Artifact:** `tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json`
  - **Tracker:** ByteTrack algorithm.
  - **Frames:** 200 frames (166 active frames, 34 empty frames).
  - **Track Observations:** 193 observations.
  - **Unique Tracks:** 3 tracks (`track_id` 1: truck, `track_id` 2: bus, `track_id` 8: traffic light).
- **Temporal Sequence Arrays (`features/sequences/run_official/`):**
  - **Features per Timestep:** $F = 27$ standard numerical features.
  - **Sequence Tensors:**
    - `sequences_L5.npy`: Shape $(138, 5, 27)$, Train: 101, Val: 18, Test: 19.
    - `sequences_L10.npy`: Shape $(123, 10, 27)$, Train: 96, Val: 13, Test: 14.
    - `sequences_L15.npy`: Shape $(108, 15, 27)$, Train: 91, Val: 8, Test: 9.
    - `sequences_L20.npy`: Shape $(93, 20, 27)$, Train: 86, Val: 3, Test: 4.
  - **Temporal Leakage Safeguard:** Chronological purged split with temporal buffer gap $G = 25$ frames.
  - **Target Availability Flag:** `target_labels_available = false`.

---

## 4. Annotation Inventory

A systematic recursive scan of the repository was executed to identify any ground-truth annotation files:

| Annotation Standard | Target Extension / Filenames | Discovered Files | Readiness Status |
|:---|:---|:---:|:---:|
| **YOLO TXT** | `*.txt` (`<class> <x_center> <y_center> <w> <h>`) | 0 | **NOT AVAILABLE** |
| **COCO JSON** | `*coco*.json`, `*annotations*.json` | 0 | **NOT AVAILABLE** |
| **Pascal VOC XML** | `*.xml` (`<annotation><bndbox>...`) | 0 | **NOT AVAILABLE** |
| **MOT CSV / Challenge** | `*.csv` (`frame, id, bb_left, bb_top, bb_w, bb_h, ...`) | 0 | **NOT AVAILABLE** |
| **BDD100K JSON** | `bdd100k_mot_labels_*.json`, `026c7465-309f6d33.json` | 0 | **NOT AVAILABLE** |
| **Custom Annotation JSON** | `*gt*.json`, `*ground_truth*.json` | 0 | **NOT AVAILABLE** |

### Clarification on Visualization Artifacts:
Files matching `*annotated*` were identified in:
`datasets/processed/driving_sample/visualizations/vis_20261117_173857_5b836e/annotated_frames/frame_000001_annotated.jpg` ... `frame_000200_annotated.jpg` and `annotated_video.mp4`.
**Audit Finding:** These files are rendered JPEG/MP4 visualizations where YOLO-predicted bounding boxes and tracker IDs are burned into the image pixels for debugging. They are **not** annotation files, contain no extractable ground truth, and cannot be used for supervision.

---

## 5. Annotation Formats

Because no ground-truth annotation files currently exist in the repository, the audit evaluated the compatibility of potential external annotation formats with PerceptGuard-AI's ingestion pipeline:

| Format Name | Native Coordinates | Ingestion Complexity | Required Transformation | Compatibility |
|:---|:---|:---:|:---|:---:|
| **BDD100K MOT JSON** | Absolute pixels in raw resolution ($x_1, y_1, x_2, y_2$) + track IDs | Low | Match sequence ID `026c7465-309f6d33`, sample at 5 FPS, apply letterbox affine transform | **HIGH (Native Match)** |
| **COCO Detection JSON** | Absolute pixels ($x, y, w, h$) | Medium | Map image names `frame_XXXXXX.jpg`, apply letterbox affine transform | **HIGH** |
| **YOLO Standard TXT** | Normalized coordinates $[0, 1]$ relative to image | Medium | Map to canvas pixels, account for letterbox black borders | **MEDIUM** |
| **MOT Challenge CSV** | Frame-indexed pixel bounding boxes | Medium | Resample 30 FPS to 5 FPS, apply letterbox affine transform | **MEDIUM** |
| **Pascal VOC XML** | XML bounding boxes $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ | High | XML parsing overhead, coordinate rescale | **LOW** |

---

## 6. Frame and Annotation Alignment Results

### 6.1 Internal Pipeline Alignment: `PASS`
Internal synchronization between extracted frames, preprocessed frames, YOLO detections, tracking outputs, and temporal features was validated with 100% exact alignment:
- All 200 frames maintain identical indexing: `frame_000001.jpg` to `frame_000200.jpg`.
- Sequence builders and temporal feature extractors correctly identify source frame indices $0, 6, 12, \dots, 1194$ and exact timestamp offsets $0.0\text{s}$ to $39.7\text{s}$.

### 6.2 External Ground-Truth Alignment: `NOT AVAILABLE` / `PARTIAL`
Because external ground-truth annotation files are not yet present in the repository, external alignment cannot be tested. However, geometric analysis identified a critical requirement:
- **Raw Video Dimensions:** $720 \times 1280$ ($W=720, H=1280$, 9:16 vertical ratio).
- **Preprocessed Frame Dimensions:** $640 \times 360$ ($W=640, H=360$, 16:9 canvas).
- **Affine Transformation:** The preprocessor scales the raw image by $s = 0.28125$ and centers it horizontally with offset $dx = 218.75$ pixels:
  $$x_{\text{canvas}} = x_{\text{raw}} \cdot 0.28125 + 218.75$$
  $$y_{\text{canvas}} = y_{\text{raw}} \cdot 0.28125 + 0.0$$
- **Alignment Warning:** Any imported raw BDD100K ground-truth bounding box $(x_1, y_1, x_2, y_2)$ **must** undergo this exact affine transformation before computing IoU against YOLO detections. Evaluating raw coordinates against preprocessed detections without this transformation will result in zero overlap ($\text{IoU} = 0.0$).

---

## 7. Data Quality Findings

A data quality audit was conducted across all existing artifacts:

| Audit Dimension | Target Criterion | Measured Result | Status |
|:---|:---|:---:|:---:|
| **File Readability** | 200 preprocessed frames decoded via OpenCV | 200 readable, 0 corrupted | **PASS** |
| **Image Dimensions** | Consistent $(360, 640, 3)$ shape | 200 / 200 match exact shape | **PASS** |
| **Bounding Box Validity** | $0 \le x_1 < x_2 \le 640$, $0 \le y_1 < y_2 \le 360$ | 211 / 211 valid, 0 inverted or OOB | **PASS** |
| **Numerical Stability** | Feature arrays contain zero NaNs and zero Infs | 0 NaNs, 0 Infs across all $L \in \{5, 10, 15, 20\}$ | **PASS** |
| **Temporal Monotonicity** | Timestamps strictly monotonically increasing | $t_{k+1} > t_k$ across all 200 frames | **PASS** |
| **Split Purging** | Zero overlap between train, val, test with 25-frame gap | Embargo gap verified, zero leakage | **PASS** |
| **Storage Constraints** | Fits within 8 GB RAM and local drive limits | Peak RAM 536 MB (< 7%), Disk: ~452 MB | **PASS** |

---

## 8. Ground-Truth Availability

### Official Status: `NOT AVAILABLE`

The repository contains:
- **Genuine Raw Driving Video:** Available (`converted_026c7465-309f6d33.mp4`).
- **Extracted & Preprocessed Video Frames:** Available (200 frames).
- **YOLO Detections & Tracking Outputs:** Available.
- **Multimodal Feature Sequences:** Available.
- **Human Ground-Truth Annotations:** **`NOT AVAILABLE`**.

### Strict Scientific Demarcation:
In accordance with strict project rules:
1. **YOLO Detection Confidence is NOT Ground Truth:** Confidence $P(\text{class}|\text{box})$ represents model certainty, not physical correctness. High-confidence hallucinations and low-confidence true positives violate any assumption of ground-truth equivalence.
2. **ByteTrack Continuity is NOT Ground Truth:** Kalman filter persistence measures temporal association stability, not spatial detection accuracy.
3. **Optical Heuristics are NOT Ground Truth:** Laplacian blur variance, illumination, and contrast quantify camera sensor conditions, not object presence or boundary precision.
4. **Synthetic Data is NOT Ground Truth:** Synthetic sequences used in unit tests verify software execution contracts, not autonomous driving perception reliability.

---

## 9. Frame-Level Labeling Readiness

### Official Status: `NOT AVAILABLE`

Frame-level reliability reflects the overall detection accuracy of a frame:
$$R_t = \frac{1}{|B_t|} \sum_{i \in B_t} \max_{j \in G_t} \text{IoU}(B_{t,i}, G_{t,j})$$
or standard Mean Average Precision (mAP@0.5).

**Readiness Assessment:**
- Without ground-truth bounding box set $G_t$, neither frame IoU nor mAP can be calculated.
- Empty frame handling rules are formulated ($R_t = 1.0$ if both $G_t = \emptyset$ and $B_t = \emptyset$; $R_t = 0.0$ if $G_t = \emptyset$ and $B_t \ne \emptyset$), but cannot be evaluated without ground truth.
- Frame-level labeling readiness is currently **`NOT AVAILABLE`**.

---

## 10. Object-Level Labeling Readiness

### Official Status: `NOT AVAILABLE`

Object-level reliability evaluates the spatial precision of each detected object:
$$R_{t, i} = \text{IoU}(B_{t, i}, G_{t, \pi(i)})$$
where $\pi(i)$ is the bipartite optimal matching assignment.

**Readiness Assessment:**
- Optimal matching requires solving the assignment problem between predicted boxes $B_{t}$ and ground-truth boxes $G_{t}$ under the class constraint $\text{class}(B_i) = \text{class}(G_{\pi(i)})$ and threshold $\tau \ge 0.50$.
- Because ground-truth object sets $G_t$ do not exist in the repository, matching cannot be executed.
- Object-level labeling readiness is currently **`NOT AVAILABLE`**.

---

## 11. Temporal Labeling Readiness

### Official Status: `NOT AVAILABLE`

Temporal reliability forecasting predicts future reliability vectors:
$$Y_t = [R_{t+1}, R_{t+3}, R_{t+5}]^T \in [0.0, 1.0]^3$$
at horizons $H \in \{1, 3, 5\}$ frames ahead.

**Readiness Assessment:**
- `temporal_sequence_builder.py` and `sequence_metadata.json` already have the data structures and horizon references indexed:
  - For sequence $k$, target frame IDs for $h=1, 3, 5$ are explicitly identified (e.g. `frame_000006.jpg`, `frame_000008.jpg`, `frame_000010.jpg`).
  - Target status is marked `valid` for horizon indexing, but `target_label` is `None` and `target_label_available` is `False`.
- Because future ground-truth bounding boxes $G_{t+h}$ do not exist, future IoU values $R_{t+h}$ cannot be computed.
- Temporal labeling readiness is currently **`NOT AVAILABLE`**.

---

## 12. Missing Requirements

To achieve complete dataset readiness for supervised reliability forecasting, the following items must be fulfilled:

1. **Ground-Truth Annotation Source:**
   - The official BDD100K MOT annotation JSON for sequence `026c7465-309f6d33` (or equivalent verified human annotations covering frames 1 to 200).
2. **Coordinate Affine Transformation Engine:**
   - A verified mapping utility converting raw video bounding box coordinates $(x_1, y_1, x_2, y_2)$ into preprocessed canvas letterbox space ($640 \times 360$).
3. **Ground-Truth Label Ingestion & Bipartite Matching Module:**
   - An automated IoU calculation and Hungarian bipartite matching module pairing YOLO detections with ground-truth boxes to compute $R_{t} \in [0.0, 1.0]$.
4. **Multi-Horizon Target Join Utility:**
   - A target assignment tool populating `target_labels` in `temporal_sequence_builder.py` to yield aligned target matrices $(N, 3)$ for horizons $H \in \{1, 3, 5\}$.

---

## 13. Risks and Limitations

1. **Risk of False Evaluation (Surrogate Leakage):**
   - Attempting to train the LSTM using YOLO confidence or tracker duration as surrogate targets would produce an unscientific model predicting model self-certainty rather than true object reliability.
2. **Coordinate Misalignment Risk:**
   - Neglecting the letterbox affine transform ($s=0.28125, dx=218.75$) when importing external annotations would result in zero IoU overlap between predictions and ground truth.
3. **Class Taxonomy Mismatch:**
   - COCO classes detected by YOLO11n (e.g., `traffic light`, `truck`, `car`, `bus`, `person`) must align with BDD100K annotation categories. Any semantic mismatch (e.g. `vehicle` vs `car`/`truck`) must be harmonized.
4. **Hardware & Memory Limitations:**
   - Host environment has an Intel Core i3 CPU and 8 GB RAM. Any future annotation parsing must remain lightweight and avoid loading multi-gigabyte annotation archives into memory.

---

## 14. Recommended Next Action

### Recommendation: Proceed to Day 8 Task 2 (Annotation Integration & Target Generation)
1. In Task 2, acquire or ingest official BDD100K MOT ground-truth annotations corresponding to sequence `026c7465-309f6d33`.
2. Implement the letterbox affine coordinate transform to align ground-truth bounding boxes with preprocessed YOLO detections.
3. Execute Hungarian bipartite matching to generate authentic object-level and frame-level IoU scores.
4. Join frame IoU values into temporal sequences to construct ground-truth target tensors $Y \in [0.0, 1.0]^{N \times 3}$ for horizons $H \in \{1, 3, 5\}$.
5. Update `target_labels_available = true` and unblock supervised LSTM training.

---
**Audit Verification:**
- Auditor: Independent Software & ML Dataset Quality Architect
- Audit Decision: **Audit PASS | Supervised Training BLOCKED on Annotations**
