# PerceptGuard-AI: Ground-Truth Annotation Dataset Selection & Integration
**Document Version:** 1.0.0  
**Last Updated:** 2026-11-20  
**Phase:** Day 8 – Task 2: Annotation Dataset Selection & Integration  
**Lead Roles:** Principal Computer Vision Engineer, Staff Machine Learning Engineer, Senior Dataset Engineer, ML Data Integration Architect  
**Status:** **Approved & Integrated**

---

## 1. Review of Task 1 Readiness Audit Findings

The Day 8 Task 1 Ground-Truth Dataset Readiness Audit established the following baseline facts:
1. **Pipeline & Data Assets:** The existing video extraction (200 frames), preprocessing ($360 \times 640$ letterbox), YOLO11n object detection (211 detections), ByteTrack tracking (193 track observations, 3 unique tracks), and temporal feature engineering ($F=27$ numerical features across $L \in \{5, 10, 15, 20\}$) are fully functional, numerically stable, and free from defects.
2. **Annotation Gap (Situation C):** Zero ground-truth annotation files existed in the repository across all standard formats (YOLO TXT, COCO JSON, Pascal VOC XML, MOT CSV, BDD100K JSON).
3. **Source Video Provenance:** The acquired video `datasets/raw/converted_026c7465-309f6d33.mp4` originates directly from the **Berkeley DeepDrive (BDD100K)** training split sequence `026c7465-309f6d33` (47.47 MB, 1,204 frames @ 30.09 FPS).
4. **Coordinate Shift:** Preprocessed frames are letterboxed into $640 \times 360$ from the raw container format with an active content offset ($x \in [218.75, 421.25]$), requiring an explicit affine coordinate transformation.

---

## 2. Dataset Candidates Evaluated

To support scientifically valid object detection reliability forecasting ($R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$), five prominent computer vision dataset candidates were evaluated:

| Dataset Candidate | Source & Provider | Native Annotation Format | Sequence Structure | Bounding Box Availability | Object Identity (MOT) | Autonomous Driving Relevance | Final Suitability |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BDD100K (MOT)** | UC Berkeley / BAIR | JSON (Scalabel / BDD100K MOT) | Continuous 40s video clips (30 FPS video, 5 FPS annotations) | 2D Bounding Boxes $(x_1, y_1, x_2, y_2)$ | Persistent Track ID (`id`) across frames | **HIGH** (Urban driving, dashcam, ego-motion) | **RECOMMENDED (Direct Native Match)** |
| **KITTI Tracking** | KIT / Toyota Tech Inst | Custom Text (15 attributes / line) | Short sequences (10 FPS) | 2D & 3D Bounding Boxes | Object Tracking ID | **MEDIUM** (Rural/suburban Karlsruhe driving) | **NOT RECOMMENDED** (FPS mismatch, format overhead) |
| **MOT17 / MOT20** | MOTChallenge Consortium | CSV (`frame, id, x, y, w, h...`) | Variable length sequences | 2D Bounding Boxes | Persistent Pedestrian ID | **LOW** (Pedestrians in indoor/mall scenes) | **NOT RECOMMENDED** (Zero vehicle / driving relevance) |
| **COCO Detection** | Microsoft / COCO Consortium | JSON (`images`, `annotations`) | Isolated static images | 2D Bounding Boxes | **NONE** (Zero temporal tracking) | **LOW** (Web photography, zero video sequences) | **UNSUITABLE** (Cannot support temporal forecasting) |
| **Waymo Open** | Waymo LLC | TFRecord / Protocol Buffers | 20s continuous driving sequences | 3D & 2D Bounding Boxes | Persistent Track ID | **HIGH** (Autonomous driving sensor suite) | **NOT RECOMMENDED** (Heavy TFRecord format, 8 GB RAM limit) |

---

## 3. Technical Justification for Selecting BDD100K

**BDD100K (Berkeley DeepDrive)** was selected as the sole optimal ground-truth dataset for PerceptGuard-AI based on decisive technical criteria:
1. **Direct Provenance Alignment:** The project's existing raw video sample (`converted_026c7465-309f6d33.mp4`) is *already* sequence `026c7465-309f6d33` from BDD100K. Utilizing official BDD100K annotations ensures genuine ground-truth correspondence.
2. **Temporal & Frame Rate Synchronization:** BDD100K MOT tracks are annotated at 5.0 FPS (every 6th video frame from 30 FPS raw capture). Our frame extraction pipeline extracts exactly 200 frames at 5.0 FPS over the 40-second duration, yielding a 1-to-1 temporal correspondence:
   - Frame 1 $\longleftrightarrow$ `frameIndex 0` (timestamp 0.00s)
   - Frame 2 $\longleftrightarrow$ `frameIndex 1` (timestamp 0.20s)
   - ...
   - Frame 200 $\longleftrightarrow$ `frameIndex 199` (timestamp 39.80s)
3. **Persistent Object Tracking IDs:** Unlike static detection datasets, BDD100K MOT provides continuous track identifiers (`id`) across all frames, enabling track-level continuity verification.
4. **Hardware & Resource Efficiency:** The annotation dataset for sequence `026c7465-309f6d33` is compact (~381 KB serialized, ~2.0 MB formatted JSON), avoiding multi-gigabyte downloads on the host 8 GB RAM system.

---

## 4. Licensing and Source Governance

- **Dataset Title:** BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
- **License Holder:** © The Regents of the University of California. All rights reserved.
- **Permitted Usage:** Permitted for academic, educational, and research use without fee.
- **Citation:**
  ```bibtex
  @inproceedings{yu2020bdd100k,
    title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={2636--2645},
    year={2020}
  }
  ```
- **Local Storage Policy:** Raw annotations reside in `datasets/annotations/bdd100k/` and processed annotations in `datasets/processed/driving_sample/annotations/`. Both are strictly excluded from GitHub tracking via `.gitignore` (`datasets/annotations/*` and `datasets/processed/*`).

---

## 5. Coordinate Transformation Engine

The original BDD100K video is captured in 720p HD landscape ($1280 \times 720$). The acquired video asset `converted_026c7465-309f6d33.mp4` has vertical container dimensions ($720 \times 1280$) corresponding to a 90-degree clockwise orientation. The preprocessing pipeline further centers this content inside a $640 \times 360$ letterbox canvas.

To map native BDD100K ground-truth bounding boxes $(x_1, y_1, x_2, y_2)_{\text{nat}}$ into the exact coordinate system of YOLO predictions on the $640 \times 360$ canvas, the adapter implements the composite affine transformation:

### Mathematical Derivation:
1. **Clockwise Rotation to Raw Video Space ($720 \times 1280$):**
   $$x_{\text{raw}} = 720 - y_{\text{nat}}$$
   $$y_{\text{raw}} = x_{\text{nat}}$$
2. **Letterbox Affine Scaling & Translation to Canvas Space ($640 \times 360$):**
   $$x_{\text{canvas}} = x_{\text{raw}} \cdot s + dx = (720 - y_{\text{nat}}) \cdot 0.28125 + 218.75$$
   $$y_{\text{canvas}} = y_{\text{raw}} \cdot s + dy = x_{\text{nat}} \cdot 0.28125 + 0.0$$
3. **Active Boundary Verification:**
   - For all $y_{\text{nat}} \in [0, 720]$: $x_{\text{canvas}} \in [218.75, 421.25]$ pixels.
   - For all $x_{\text{nat}} \in [0, 1280]$: $y_{\text{canvas}} \in [0.0, 360.0]$ pixels.
   - This matches the active detection region of YOLO predictions ($X \in [218.7, 420.8], Y \in [0.8, 359.7]$).

---

## 6. Class Taxonomy Mapping

BDD100K object categories are mapped to standard YOLO COCO classes as follows:

| BDD100K Class | Mapped COCO Class | COCO Class ID | Rationale |
|:---|:---|:---:|:---|
| `car` | `car` | 2 | Direct semantic match |
| `truck` | `truck` | 7 | Direct semantic match |
| `bus` | `bus` | 5 | Direct semantic match |
| `pedestrian` | `person` | 0 | Standard COCO person taxonomy |
| `rider` | `person` | 0 | Human operator on vehicle |
| `other person` | `person` | 0 | Pedestrian variant |
| `motorcycle` | `motorcycle` | 3 | Direct semantic match |
| `bicycle` | `bicycle` | 1 | Direct semantic match |
| `traffic light` | `traffic light` | 9 | Direct semantic match |
| `traffic sign` | `traffic light` | 9 | Mapped to roadway signal marker |
| `train` | `train` | 6 | Direct semantic match |
| `trailer` | `truck` | 7 | Heavy haul vehicle |
| `other vehicle` | `car` | 2 | Passenger vehicle variant |

---

## 7. Integration Architecture

The ground-truth ingestion architecture is implemented across the following components:

```
PerceptGuard-AI/
├── config/
│   └── dataset_config.yaml                  # Global dataset & annotation configuration
├── datasets/
│   ├── annotations/
│   │   ├── .gitkeep                         # Preserves directory in git
│   │   └── bdd100k/
│   │       └── bdd100k_mot_026c7465_309f6d33.json # Raw BDD100K sequence JSON
│   └── processed/driving_sample/
│       └── annotations/
│           ├── annotation_manifest.json     # Portable reproducible manifest
│           ├── ground_truth_annotations.json # Standardized canvas-aligned GT
│           └── validation_report.json       # Bounded integrity validation audit
├── src/
│   └── evaluation/
│       ├── __init__.py                      # Package exports
│       └── bdd100k_adapter.py               # Core adapter, models & transformer
└── tests/
    ├── __init__.py
    ├── test_bdd100k_adapter.py              # Root test discoverability wrapper
    └── evaluation/
        ├── __init__.py
        └── test_bdd100k_adapter.py          # 16 comprehensive unit & integration tests
```

---

## 8. Data Integrity Validation Results

The adapter performed bounded integrity auditing across the entire 200-frame sequence:

| Audit Parameter | Verification Criterion | Result | Status |
|:---|:---|:---:|:---:|
| **Sequence ID** | Matches BDD100K sequence `026c7465-309f6d33` | Confirmed | **PASS** |
| **Total Frames** | Exactly 200 frames (matching extracted driving sample) | 200 frames | **PASS** |
| **Image Existence** | Referenced frames exist in `preprocessed_frames/` | 200 / 200 present | **PASS** |
| **Total Ground-Truth Boxes** | Ingested and transformed bounding boxes | 5,121 boxes | **PASS** |
| **Invalid Bounding Boxes** | $x_2 \le x_1$ or $y_2 \le y_1$ or non-finite values | 0 invalid boxes | **PASS** |
| **Corrupted Frames** | Unreadable or malformed frame structures | 0 corrupt frames | **PASS** |
| **Duplicate Frame Records** | Frame IDs appear more than once | 0 duplicates | **PASS** |
| **Unique Tracking IDs** | Persistent multi-frame object trajectories | 115 unique tracks | **PASS** |
| **Class Distribution** | `car`: 3,113; `person`: 1,231; `bicycle`: 377; `truck`: 226; `bus`: 107; `motorcycle`: 67 | 5,121 total | **PASS** |
| **Overall Integrity Status** | Zero fatal errors, zero missing images | **PASS** | **PASS** |

---

## 9. Dataset Split Readiness

The integrated dataset strictly supports the project's **chronological purged** splitting strategy:
- **Temporal Sequence Integrity:** Video frames $0 \dots 199$ are strictly ordered in time ($t_{k+1} > t_k$).
- **Embargo Buffer Gaps ($G = 25$ frames):**
  - Train: frames 0 to 105 (105 frames, ~2,840 annotations)
  - Gap 1: frames 105 to 130 (25 frames embargo, zero leakage)
  - Val: frames 130 to 152 (22 frames, ~620 annotations)
  - Gap 2: frames 152 to 177 (25 frames embargo, zero leakage)
  - Test: frames 177 to 200 (23 frames, ~610 annotations)
- **Zero Temporal Overlap:** Because ground-truth targets $R_{t+h}$ require future frames $t+h$, the 25-frame embargo buffer prevents train-set sequences from querying validation-set frames.

---

## 10. Verification and Test Results

The test suite was executed via `pytest tests/test_bdd100k_adapter.py -v`:
- Total Tests: **16 passed in 1.48s (100% Pass Rate)**.
- Coverage includes:
  - Bounding box properties, area, validity, and clamping
  - Native to raw video rotation (90 deg clockwise)
  - Raw video to canvas letterbox transformation
  - End-to-end composite transformation
  - Category taxonomy and class mapping
  - Synthetic sequence parsing and frame indexing
  - Schema error handling (empty lists, bad JSON)
  - Manifest relative path generation
  - Data integrity validation (missing images, duplicate frames)
  - Live integration test against `datasets/annotations/bdd100k/bdd100k_mot_026c7465_309f6d33.json`

---

## 11. Remaining Limitations & Next Task

### Current Limitations:
- Dataset selection and ground-truth ingestion are complete, but object detection reliability labels ($R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$) have **not** yet been computed.
- Supervised LSTM training remains gated until Task 3 generates authentic multi-horizon reliability targets.

### Recommended Next Task:
- **Day 8 – Task 3 (Reliability Labeling & Target Construction):**
  Implement Hungarian bipartite matching between YOLO detections and ground-truth boxes to compute frame-level and object-level IoU scores across horizons $H \in \{1, 3, 5\}$, update `target_labels_available = true`, and generate aligned target tensors.
