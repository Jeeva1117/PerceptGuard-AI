# PerceptGuard-AI: Ground-Truth Dataset Selection & Integration Report
**Document ID:** `reports/day8/dataset_selection_integration.md`  
**Execution Timestamp:** 2026-11-20T07:07:00+05:30  
**Phase:** Day 8 – Task 2: Annotation Dataset Selection & Integration  
**Auditor Roles:** Principal Computer Vision Engineer, Staff Machine Learning Engineer, Senior Dataset Engineer, ML Data Integration Architect  
**Hardware Profile:** Intel Core i3 (CPU Only), 8 GB RAM (7.75 GB usable), Windows 11 Build 26200  
**Repository:** [https://github.com/Jeeva1117/PerceptGuard-AI.git](https://github.com/Jeeva1117/PerceptGuard-AI.git)  
**Target Branch:** `main`

---

## 1. Executive Summary

In Day 8 Task 2, an evidence-based dataset selection and integration pipeline was designed, validated, and executed for **PerceptGuard-AI**.

Based on the findings of the Day 8 Task 1 Readiness Audit, which established that the repository had zero ground-truth annotation files despite hosting a genuine BDD100K driving sample (`converted_026c7465-309f6d33.mp4`), five major computer vision datasets were evaluated against the project's multi-horizon forecasting architecture.

**BDD100K Multiple Object Tracking (MOT)** was selected as the sole optimal choice. The official ground-truth annotations for sequence `026c7465-309f6d33` were acquired, parsed, and ingested via a newly implemented, production-grade adapter (`src/evaluation/bdd100k_adapter.py`).

### Key Results:
- **Dataset Selection:** BDD100K MOT selected (**`PASS`**).
- **Annotation Ingestion:** 200 frames aligned; 5,121 bounding boxes ingested across 115 persistent tracking IDs (**`PASS`**).
- **Coordinate Conversion:** Affine transformation from native $1280 \times 720$ landscape through rotated raw video into preprocessed $640 \times 360$ letterbox canvas verified (**`PASS`**).
- **Data Integrity Validation:** 0 corrupted frames, 0 missing images, 0 invalid bounding boxes, 0 out-of-bounds errors (**`PASS`**).
- **Dataset Manifest Generation:** Reproducible, portable manifest created with relative paths (**`PASS`**).
- **Unit and Integration Tests:** 16 / 16 adapter tests passed; 50 / 50 total dataset tests passed (**`PASS`**).
- **Supervised Training Readiness:** **`BLOCKED`** (pending Task 3 calculation of multi-horizon reliability labels $R_{t+h}$).

---

## 2. Review of Task 1 Findings

The starting evidence from `reports/day8/ground_truth_readiness_audit.md` and `ground_truth_readiness_audit.json` was reviewed:
- **Situation C Applied:** No valid ground-truth annotations existed in the repository.
- **Video Source Authenticity:** `converted_026c7465-309f6d33.mp4` was confirmed as a legitimate 40-second clip from the BDD100K training split.
- **Transformation Constraint:** Preprocessing applies centered letterbox padding into $640 \times 360$ with active content area $x \in [218.75, 421.25]$, necessitating an explicit affine coordinate transformation for any external bounding boxes.

---

## 3. Dataset Candidates Comparison

An evidence-based comparative evaluation was conducted across five candidate datasets:

| Dataset | Provider / Source | Annotation Standard | Sequence Structure | Bounding Boxes | Object Tracking ID | Driving Context | Final Suitability |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BDD100K (MOT)** | UC Berkeley / BAIR | JSON | Continuous 40s @ 5 FPS annotations | Yes $(x_1, y_1, x_2, y_2)$ | Yes (`id` string) | Urban street driving | **RECOMMENDED (Direct Native Match)** |
| **KITTI Tracking** | KIT / Toyota Tech Inst | Text (.txt) | Short sequences @ 10 FPS | Yes $(x_1, y_1, x_2, y_2)$ | Yes (Integer ID) | Rural / suburban | **NOT RECOMMENDED** (Frame rate mismatch) |
| **MOT17 / MOT20** | MOTChallenge | CSV | Variable sequences | Yes (xywh) | Yes (Pedestrian ID) | Indoor / Surveillance | **NOT RECOMMENDED** (Lacks driving context) |
| **COCO Detection** | Microsoft / Consortium | JSON | Discontinuous static images | Yes (xywh) | **None** | Web photos | **UNSUITABLE** (Zero temporal sequences) |
| **Waymo Open** | Waymo LLC | Protocol Buffers | 20s sequences | Yes (3D / 2D) | Yes (Track ID) | Autonomous driving | **NOT RECOMMENDED** (Heavy TFRecord format) |

---

## 4. Selected Dataset & Technical Justification

**BDD100K (Berkeley DeepDrive)** was officially selected based on four technical imperatives:
1. **Direct Sequence Identity:** The acquired video in `datasets/raw/converted_026c7465-309f6d33.mp4` is sequence `026c7465-309f6d33` from BDD100K. Utilizing official BDD100K annotations ensures genuine ground-truth correspondence.
2. **Exact Temporal Alignment:** BDD100K MOT annotates every 6th frame of 30 FPS video (~5.0 FPS). Our pipeline extracts 200 frames at 5.0 FPS over 40 seconds, yielding 1-to-1 frame alignment across all 200 frames.
3. **Temporal Object Tracking IDs:** BDD100K provides persistent object IDs across frames, enabling track-level validation.
4. **Lightweight Footprint:** The sequence annotation is compact (~381 KB serialized, ~2.0 MB JSON), respecting the host 8 GB RAM and disk envelope.

---

## 5. Source, Licensing and Data Governance

- **Source:** Berkeley DeepDrive Consortium / BAIR, UC Berkeley (`https://bdd-data.berkeley.edu/`).
- **License:** © The Regents of the University of California. Academic, research, and educational use permitted without fee.
- **Git Security:** In accordance with project rules, `.gitignore` was updated to explicitly exclude `datasets/annotations/*`. Raw and processed annotations remain local and are never committed to public GitHub.

---

## 6. Coordinate Transformation & Alignment

Because the acquired video `converted_026c7465-309f6d33.mp4` has vertical container dimensions ($720 \times 1280$) corresponding to a 90-degree clockwise orientation, and is preprocessed into $640 \times 360$ via letterbox padding, the adapter implements the exact composite transformation:

$$x_{\text{canvas}} = (720 - y_{\text{nat}}) \cdot 0.28125 + 218.75$$
$$y_{\text{canvas}} = x_{\text{nat}} \cdot 0.28125 + 0.0$$

- All native $y_{\text{nat}} \in [0, 720]$ map to $x_{\text{canvas}} \in [218.75, 421.25]$ pixels.
- All native $x_{\text{nat}} \in [0, 1280]$ map to $y_{\text{canvas}} \in [0.0, 360.0]$ pixels.
- This mathematically aligns with YOLO detections ($X \in [218.7, 420.8], Y \in [0.8, 359.7]$).

---

## 7. Class Taxonomy Mapping

| BDD100K Category | Target COCO Class | COCO Class ID | Count in Driving Sample |
|:---|:---|:---:|:---:|
| `car` | `car` | 2 | 3,113 |
| `pedestrian` | `person` | 0 | 847 |
| `rider` | `person` | 0 | 310 |
| `other person` | `person` | 0 | 74 |
| `bicycle` | `bicycle` | 1 | 377 |
| `truck` | `truck` | 7 | 192 |
| `trailer` | `truck` | 7 | 34 |
| `bus` | `bus` | 5 | 107 |
| `motorcycle` | `motorcycle` | 3 | 67 |
| **Total** | — | — | **5,121** |

---

## 8. Data Integrity Validation Results

Bounded data integrity validation was executed across all 200 frames:

| Metric | Measured Value | Threshold / Target | Status |
|:---|:---:|:---:|:---:|
| **Sequence ID** | `026c7465-309f6d33` | Match raw video | **PASS** |
| **Processed Frames** | 200 | 200 frames | **PASS** |
| **Total Ground-Truth Boxes** | 5,121 | > 0 | **PASS** |
| **Corrupted Frames** | 0 | 0 | **PASS** |
| **Missing Image Files** | 0 | 0 | **PASS** |
| **Invalid Bounding Boxes** | 0 | 0 | **PASS** |
| **Duplicate Frame Records** | 0 | 0 | **PASS** |
| **Unique Tracking IDs** | 115 | > 0 | **PASS** |
| **Empty Frames** | 0 | Acceptable if 0 | **PASS** |
| **Overall Validation Status** | **`PASS`** | PASS | **PASS** |

---

## 9. Dataset Split Readiness

The integrated annotations adhere strictly to PerceptGuard-AI's **chronological purged** splitting policy:
- **Train Split:** Frames 0 to 105 (105 frames, ~2,840 annotations)
- **Embargo Gap 1:** Frames 105 to 130 (25 frames buffer, zero leakage)
- **Validation Split:** Frames 130 to 152 (22 frames, ~620 annotations)
- **Embargo Gap 2:** Frames 152 to 177 (25 frames buffer, zero leakage)
- **Test Split:** Frames 177 to 200 (23 frames, ~610 annotations)

No cross-split contamination or temporal leakage exists.

---

## 10. Unit and Integration Test Results

- Test module: `tests/test_bdd100k_adapter.py`
- Total Tests: **16 passed in 1.48 seconds (100% Pass Rate)**.
- Full regression suite including dataset structure and validators: **50 / 50 passed**.

---

## 11. Remaining Limitations & Blockers

- **Reliability Labels Not Yet Computed:** Ground truth annotations are ingested, but multi-horizon reliability targets ($R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$) have not yet been generated.
- **Supervised Training Blocked:** Gated on Task 3 target label construction.

---

## 12. Recommended Next Action

**Proceed to Day 8 Task 3 (Reliability Labeling & Multi-Horizon Target Construction):**
1. Implement Hungarian bipartite matching between YOLO detections and ground truth boxes.
2. Compute frame-level and object-level IoU metrics.
3. Generate future horizon targets for $H \in \{1, 3, 5\}$ frames.
4. Populate `target_labels` in `temporal_sequence_builder.py` and set `target_labels_available = true`.
