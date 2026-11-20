# PerceptGuard-AI: Ground-Truth Annotation Processing Pipeline Specification

**Document Version:** 1.0.0  
**Status:** Approved Architectural Standard  
**Authors:** Principal Computer Vision Engineer, Staff Machine Learning Engineer, Senior Data Engineer, Annotation Pipeline Architect  
**Date:** Day 8 – Task 4 Implementation  
**Target Repository:** `PerceptGuard-AI` (Branch: `main`)  
**Target Hardware:** Intel Core i3 (CPU-Only), 8 GB RAM, Windows 11  

---

## 1. Executive Summary & Context

This document defines the production architecture, data models, coordinate transformation engine, and ingestion pipeline for ground-truth object detection annotations in PerceptGuard-AI.

In Day 8 Tasks 1 through 3, the project established that:
1. Object detection reliability cannot be inferred from heuristic detector confidences ($s_{\text{det}}$) or tracker continuity.
2. Ground-truth reliability requires spatial evaluation ($R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h})$) against human-annotated bounding boxes.
3. The official Berkeley DeepDrive 100K (BDD100K) Multiple Object Tracking (MOT) sequence `026c7465-309f6d33` provides an exact temporal match for the project's reference driving video.
4. A canonical 17-field annotation schema, Hungarian bipartite matching specification, and multi-horizon target vectors ($Y_t = [R_{t+1}, R_{t+3}, R_{t+5}]^T$) were designed and approved.

Day 8 Task 4 implements the deterministic annotation processing pipeline that parses raw BDD100K JSON annotations, transforms and validates spatial coordinates, standardizes semantic taxonomies, aligns annotations with physical video frames, and serializes streaming canonical outputs.

---

## 2. Selected Annotation Dataset & Format

### 2.1 Dataset Profile
- **Dataset Name:** Berkeley DeepDrive 100K (BDD100K) Multiple Object Tracking (MOT)
- **Sequence Identifier:** `026c7465-309f6d33`
- **Source Video:** `datasets/raw/converted_026c7465-309f6d33.mp4` (47.47 MB, 1,204 frames @ 30.09 FPS)
- **Extracted Sample:** 200 preprocessed frames (`frame_000001.jpg` to `frame_000200.jpg`) sampled at 5.0 FPS ($0.2\text{s}$ interval, spanning $39.8\text{s}$)
- **Raw Annotation File:** `datasets/annotations/bdd100k/bdd100k_mot_026c7465_309f6d33.json` (2,055.84 KB)
- **Licensing:** UC Berkeley Regents Academic / Research License

### 2.2 Raw BDD100K Annotation Schema
The raw dataset format organizes annotations into sequences and frames:
```json
{
  "name": "026c7465-309f6d33",
  "frames": [
    {
      "frame_index": 0,
      "video_frame": 0,
      "labels": [
        {
          "id": "107627",
          "category": "car",
          "box2d": {
            "x1": 100.0,
            "y1": 200.0,
            "x2": 300.0,
            "y2": 400.0
          },
          "attributes": {
            "crowd": false,
            "occluded": false,
            "truncated": true
          }
        }
      ]
    }
  ]
}
```

---

## 3. Canonical Annotation Data Model

The canonical schema implemented in `src/evaluation/annotation_processor.py` (`CanonicalAnnotationRecord`) strictly defines 17 core fields and provenance metadata:

| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `dataset_id` | `str` | Non-empty string | Dataset identifier (`driving_sample`) |
| `sequence_id` | `str` | Non-empty string | BDD100K sequence ID (`026c7465-309f6d33`) |
| `frame_id` | `str` | Pattern: `frame_XXXXXX.jpg` | Filename of aligned video frame |
| `timestamp` | `Optional[float]` | $\ge 0.0$ (step: $0.20\text{s}$) | Video presentation timestamp in seconds |
| `image_path` | `str` | Strictly relative path | Portable relative path to preprocessed image |
| `image_width` | `int` | Exactly 640 | Canvas width in pixels |
| `image_height`| `int` | Exactly 360 | Canvas height in pixels |
| `annotation_id`| `str` | Non-empty string | Unique annotation identifier |
| `ground_truth_object_id`| `Optional[str]` | String or `null` | Persistent object/track identity across frames |
| `class_id` | `int` | Integer $[0, 79]$ | Standardized COCO/YOLO class index |
| `class_name` | `str` | Standardized string | Standardized class label (`car`, `person`, etc.) |
| `x_min` | `float` | $0.0 \le x_{\min} < x_{\max} \le 640.0$ | Left bounding box boundary |
| `y_min` | `float` | $0.0 \le y_{\min} < y_{\max} \le 360.0$ | Top bounding box boundary |
| `x_max` | `float` | $0.0 \le x_{\min} < x_{\max} \le 640.0$ | Right bounding box boundary |
| `y_max` | `float` | $0.0 \le y_{\min} < y_{\max} \le 360.0$ | Bottom bounding box boundary |
| `coordinate_format` | `str` | `"x_min, y_min, x_max, y_max"` | Coordinate convention |
| `annotation_source` | `str` | `"bdd100k_mot"` | Origin source tag |
| `validation_status` | `str` | `VALID`, `INVALID_...` | Integrity validation flag |
| `original_class_name`| `Optional[str]`| Raw string | Raw unmapped category from source file |
| `attributes` | `Dict[str, Any]`| Key-value dictionary | Occlusion, truncation, and crowd attributes |
| `schema_version` | `str` | `"1.0.0"` | Schema version string |

---

## 4. Coordinate Normalization & Geometric Transformations

### 4.1 Coordinate Space Progression
The pipeline transforms coordinates across three distinct spaces:
1. **Native BDD100K Space:** $1280 \times 720$ pixels (landscape orientation).
2. **Rotated Video Space:** $720 \times 1280$ pixels (acquired video container).
3. **Canonical Canvas Space:** $640 \times 360$ pixels (letterboxed inference canvas).

### 4.2 Exact Mathematical Transformation
1. **90-Degree Clockwise Rotation:**
   $$x_{\text{raw}} = 720.0 - y_{\text{native}}$$
   $$y_{\text{raw}} = x_{\text{native}}$$
2. **Letterbox Scaling and Center Offsetting:**
   $$\text{scale} = 0.28125, \quad dx = 218.75, \quad dy = 0.0$$
   $$x_{\text{canvas}} = x_{\text{raw}} \cdot 0.28125 + 218.75 = (720.0 - y_{\text{native}}) \cdot 0.28125 + 218.75$$
   $$y_{\text{canvas}} = y_{\text{raw}} \cdot 0.28125 + 0.0 = x_{\text{native}} \cdot 0.28125$$
3. **Geometric Invariants & Validation:**
   - Active horizontal image content occupies $x \in [218.75, 421.25]$. Left and right letterbox margins are $x \in [0.0, 218.75)$ and $x \in (421.25, 640.0]$.
   - Boxes must satisfy: $0.0 \le x_{\min} < x_{\max} \le 640.0$ and $0.0 \le y_{\min} < y_{\max} \le 360.0$.
   - Bounding box area must satisfy: $(x_{\max} - x_{\min}) \cdot (y_{\max} - y_{\min}) \ge 4.0\text{ px}^2$.

---

## 5. Class Mapping & Taxonomy Preservation

The `ClassMapper` translates BDD100K 11-class tracking ontology to canonical COCO/YOLO detection categories while preserving raw category names in provenance metadata:

| BDD100K Category | Canonical Category | Canonical Class ID | Mapping Justification |
| :--- | :--- | :---: | :--- |
| `car` | `car` | 2 | Direct semantic match |
| `truck` | `truck` | 7 | Direct semantic match |
| `bus` | `bus` | 5 | Direct semantic match |
| `pedestrian` | `person` | 0 | Standard COCO taxonomy mapping |
| `rider` | `person` | 0 | Vulnerable road user category |
| `other person` | `person` | 0 | Pedestrian variant |
| `motorcycle` | `motorcycle` | 3 | Direct semantic match |
| `bicycle` | `bicycle` | 1 | Direct semantic match |
| `traffic light` | `traffic light` | 9 | Infrastructure class |
| `traffic sign` | `traffic light` | 9 | Mapped for driving relevance |
| `train` | `train` | 6 | Rare transit obstacle |
| `trailer` | `truck` | 7 | Heavy haul vehicle |
| `other vehicle` | `car` | 2 | Light vehicular obstacle |

If an unknown category is encountered (e.g. `animal` or `debris`), it is marked `UNKNOWN_CLASS`, assigned `class_id = -1`, and reported in the processing error log.

---

## 6. Frame & Video Alignment Strategy

Annotations are aligned 1-to-1 with extracted frames:
- **Extraction Timing:** 5.0 FPS sampling from 30.09 FPS video corresponds to a frame extraction stride of 6 video frames ($\Delta t = 0.20\text{s}$).
- **Filename Convention:** `frame_{:06d}.jpg` (`frame_000001.jpg` to `frame_000200.jpg`).
- **Disk Verification:** The `FrameAligner` verifies the physical presence of each image on disk.
- **Relative Path Resolution:** Portable outputs store relative paths (`datasets/processed/driving_sample/preprocessed_frames/frame_XXXXXX.jpg`) without machine-specific Windows/Linux absolute paths.

---

## 7. Empty Frame & Missing Annotation Policy

The pipeline explicitly distinguishes six distinct frame states:

1. **`VALID_WITH_OBJECTS`:** Image exists, annotations present and spatially valid.
2. **`VALID_EMPTY`:** Image exists, scene genuinely contains zero target objects (perfect negative frame).
3. **`MISSING_IMAGE`:** Annotation exists, but corresponding frame image is missing from disk.
4. **`CORRUPTED_ANNOTATION`:** Annotation record contains unparseable syntax or non-numeric values.
5. **`PARTIAL_ANNOTATION`:** Frame contains a mix of valid and invalid annotations; valid annotations are kept, invalid ones quarantined in error log.
6. **`UNVERIFIED`:** Unverified state prior to validation.

*Crucial Rule:* Missing annotation files are never silently treated as empty scenes. They are marked `MISSING_FILE` or `UNVERIFIED`.

---

## 8. Pipeline Architecture & Reusable Components

The processing pipeline is modularized into reusable components in `src/evaluation/annotation_processor.py`:

```
+--------------------------------------------------------------------------+
|                        AnnotationProcessor                               |
+--------------------------------------------------------------------------+
     |                    |                    |                     |
     v                    v                    v                     v
+------------+   +-------------------+   +-------------+   +---------------+
| BDD100K    |   | Coordinate        |   | ClassMapper |   | FrameAligner  |
| MOTParser  |   | Normalizer        |   |             |   |               |
|            |   | (Rot + Scale +    |   | (BDD ->     |   | (Matches img  |
| (JSON I/O) |   |  Validation)      |   |  COCO ID)   |   |  on disk)     |
+------------+   +-------------------+   +-------------+   +---------------+
```

### Execution Flow:
1. **Dataset Discovery:** Validates raw input paths and frame directory.
2. **Annotation Source Loading:** Safely parses JSON via `BDD100KMOTParser`.
3. **Parsing & Sequence Iteration:** Iterates through sorted frames ($0 \le i < 200$).
4. **Class Mapping:** Normalizes raw categories through `ClassMapper`.
5. **Coordinate Conversion:** Rotates, scales, offsets, and clamps coordinates through `CoordinateNormalizer`.
6. **Integrity Validation:** Rejects collapsed, inverted, or non-finite boxes.
7. **Frame Alignment:** Aligns bounding boxes with verified preprocessed image files.
8. **Atomic Serialization:** Emits JSONL records and JSON processing summaries atomically via temporary `.tmp` files.

---

## 9. Real-Data Validation Results

The pipeline was executed against the complete 200-frame BDD100K driving dataset:

```
Command: python -m src.evaluation.annotation_processor --max-frames 200
```

### Measured Real-Data Statistics:
- **Total Source Files:** 1 (`bdd100k_mot_026c7465_309f6d33.json`)
- **Total Records Discovered:** 5,121
- **Valid Canonical Records:** 5,121 (100.0%)
- **Invalid Records:** 0
- **Duplicate Records:** 0
- **Unmapped Classes:** 0
- **Unique Track IDs Preserved:** 115
- **Frames Processed & Aligned:** 200 / 200 (100.0%)
- **Empty Frames:** 0
- **Missing Images:** 0
- **Processing Duration:** 0.1787 seconds (~28,600 annotations/sec on CPU)
- **Class Breakdown:**
  - `car`: 3,113 (60.79%)
  - `person`: 1,231 (24.04%)
  - `bicycle`: 377 (7.36%)
  - `truck`: 226 (4.41%)
  - `bus`: 107 (2.09%)
  - `motorcycle`: 67 (1.31%)

### Generated Output Artifacts:
- **Canonical Streaming JSONL:** `datasets/processed/driving_sample/annotations/annotations_canonical.jsonl` (1.38 MB, 5,121 lines)
- **Processing Summary JSON:** `datasets/processed/driving_sample/annotations/annotation_processing_summary.json`

---

## 10. Requirements for Downstream Task 5 (Reliability Label Generation)

With canonical annotations normalized and aligned, Task 5 is fully unblocked:
1. **Inputs Available:** Canonical ground truth (`annotations_canonical.jsonl`) and YOLO predictions (`detection_results.json`) are both in canvas coordinates ($640 \times 360$).
2. **Matching Ready:** Hungarian bipartite matching algorithm defined in Task 3 will pair predictions and ground-truth boxes on each frame.
3. **Metric Calculation:** Compute frame-level F1-score ($F_{1, t}$) for all 200 frames.
4. **Target Population:** Populate multi-horizon target vectors $Y_t = [F_{1, t+1}, F_{1, t+3}, F_{1, t+5}]^T$ across all 186 sliding-window sequences.
