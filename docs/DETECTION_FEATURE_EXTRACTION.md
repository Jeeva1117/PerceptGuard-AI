# PerceptGuard AI — Detection Feature Extraction Architecture & Specification

## 1. Executive Summary & Architecture

**PerceptGuard AI** predicts future object detection reliability ($R_{t+h}$) across upcoming video frames. Rather than assuming deep detector confidence ($P(\text{class} \mid \text{box})$) equals localization reliability, PerceptGuard extracts multidimensional temporal feature vectors to anticipate perception dropouts and degradation modes.

The perception pipeline progresses sequentially:

$$\text{Video Loading} \longrightarrow \text{Frame Extraction} \longrightarrow \text{Preprocessing} \longrightarrow \text{YOLO Detection} \longrightarrow \text{ByteTrack} \longrightarrow \mathbf{Detection\ Feature\ Extraction} \longrightarrow \text{Motion/Quality Features} \longrightarrow \text{LSTM/GRU Forecasting}$$

**Day 5 Task 2** introduces the **Detection Feature Extraction Engine** (`DetectionFeatureExtractor`).
- Operates entirely on pre-computed YOLO detection artifacts (Schema v1.0).
- **Strictly zero duplicate YOLO re-inference**, achieving microsecond-level CPU performance.
- Decouples perception outputs into dual-tier representations: **detection-level feature vectors** and **frame-level aggregated telemetry**.
- Enforces strict coordinate, confidence, and class validation rules, segregating rejected detections into audit error logs.
- Preserves complete temporal frame continuity by representing zero-detection frames with explicit `null` statistics.

---

## 2. Input JSON Artifact Specification

The extractor seamlessly ingests both batch detection runs (`detection_results.json`) and single-image detection outputs (`frame_000001_detections.json`).

### Batch Detection Artifact (`detection_results.json`)
```json
{
  "schema_version": "1.0",
  "run_id": "run_20261117_115348_e1789b",
  "dataset_id": "driving_sample",
  "total_frames": 200,
  "frame_results": [
    {
      "image_filename": "frame_000001.jpg",
      "extracted_frame_id": 1,
      "source_frame_index": 0,
      "timestamp_seconds": 0.0,
      "image_width": 640,
      "image_height": 360,
      "detection_count": 2,
      "detections": [
        {
          "detection_index": 0,
          "class_id": 9,
          "class_name": "traffic light",
          "confidence": 0.4381,
          "bbox_xyxy": [219.5, 238.1, 276.4, 359.1]
        }
      ]
    }
  ]
}
```

---

## 3. Extracted Detection Features & Mathematical Formulations

Features are extracted across 4 functional domains for every valid detection:

### A. Detection Identity
- `dataset_id`: Identifier of source dataset (e.g. `"driving_sample"`).
- `frame_id`: Normalized string frame identifier (`"frame_000001.jpg"`).
- `source_frame_index`: Chronological integer sequence index ($\ge 0$).
- `timestamp_seconds`: Elapsed video capture time in seconds.
- `detection_index`: Zero-indexed detection rank within the frame.
- `class_id`: COCO integer category identifier.
- `class_name`: Standardized semantic label (e.g. `"car"`, `"truck"`, `"traffic light"`).

### B. Confidence Features
- `confidence`: Raw YOLO probability $c \in [0.0, 1.0]$.
  > [!NOTE]
  > Confidence indicates model belief under training priors; it does **not** equal ground-truth reliability or bounding-box localization accuracy.
- `confidence_category`: Categorical confidence tier:
  $$\text{tier} = \begin{cases} \text{"high"} & \text{if } c \ge 0.75 \\ \text{"medium"} & \text{if } 0.40 \le c < 0.75 \\ \text{"low"} & \text{if } c < 0.40 \end{cases}$$

### C. Bounding Box Geometry Features
From input coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$:
- $\text{bbox\_width} = x_{\max} - x_{\min}$
- $\text{bbox\_height} = y_{\max} - y_{\min}$
- $\text{bbox\_area} = \text{bbox\_width} \times \text{bbox\_height}$
- $\text{bbox\_center\_x} = \frac{x_{\min} + x_{\max}}{2.0}$
- $\text{bbox\_center\_y} = \frac{y_{\min} + y_{\max}}{2.0}$

### D. Normalized Bounding Box Features
When valid image dimensions ($W > 0, H > 0$) are available:
- $\text{normalized\_center\_x} = \frac{\text{bbox\_center\_x}}{W}$
- $\text{normalized\_center\_y} = \frac{\text{bbox\_center\_y}}{H}$
- $\text{normalized\_width} = \frac{\text{bbox\_width}}{W}$
- $\text{normalized\_height} = \frac{\text{bbox\_height}}{H}$
- $\text{normalized\_bbox\_area} = \frac{\text{bbox\_area}}{W \times H}$

If image dimensions are missing or non-positive, all normalized fields are explicitly assigned `null` rather than fabricated.

---

## 4. Frame-Level Detection Features & Missing-Value Policy

Every processed video frame is represented in `frame_detection_features.json`:

| Feature Name | Type | Value when Detections Present | Value when Zero Detections |
| :--- | :--- | :--- | :--- |
| `frame_id` | `str` | Host frame identifier | Host frame identifier |
| `source_frame_index` | `int` | Sequential frame index | Sequential frame index |
| `timestamp_seconds` | `float` | Capture elapsed time | Capture elapsed time |
| `image_width` | `int` | Frame pixel width | Frame pixel width |
| `image_height` | `int` | Frame pixel height | Frame pixel height |
| `detection_count` | `int` | Count of valid detections ($N \ge 1$) | `0` |
| `unique_class_count` | `int` | Number of distinct classes detected | `0` |
| `mean_confidence` | `float` | $\frac{1}{N} \sum c_i$ | `null` |
| `minimum_confidence` | `float` | $\min(c_i)$ | `null` |
| `maximum_confidence` | `float` | $\max(c_i)$ | `null` |
| `confidence_std` | `float` | $\sqrt{\frac{1}{N} \sum (c_i - \bar{c})^2}$ ($0.0$ if $N=1$) | `null` |
| `total_bbox_area` | `float` | $\sum \text{area}_i$ | `null` |
| `mean_bbox_area` | `float` | $\frac{1}{N} \sum \text{area}_i$ | `null` |
| `maximum_bbox_area` | `float` | $\max(\text{area}_i)$ | `null` |
| `mean_bbox_width` | `float` | $\frac{1}{N} \sum \text{width}_i$ | `null` |
| `mean_bbox_height` | `float` | $\frac{1}{N} \sum \text{height}_i$ | `null` |
| `class_wise_detection_counts`| `dict` | `{"car": 2, "traffic light": 1}` | `{}` |

> [!IMPORTANT]
> Zero-detection frames are legitimate video frames where no objects exceeded the detector threshold. Representing absent metrics as `null` prevents skewing temporal models with fabricated zeros for confidence or bounding box dimensions.

---

## 5. Feature Quality Validation & Fault Containment

The extractor evaluates every detection candidate against strict validation gates:
1. **Class Integrity:** `class_id` must be a non-negative integer; `class_name` must be a non-empty string.
2. **Confidence Bounds:** Confidence must be finite and within $[0.0, 1.0]$. Negative, $>1.0$, NaN, and Inf values are rejected.
3. **Bounding Box Validity:**
   - Coordinates must be non-negative ($x_{\min} \ge 0, y_{\min} \ge 0$).
   - Coordinates must be strictly positive-span ($x_{\max} > x_{\min}$ and $y_{\max} > y_{\min}$).
   - Area must be strictly positive ($\text{area} > 0$).
   - Coordinates must be finite (NaN and Inf rejected).
4. **Identity & Uniqueness:**
   - Within a frame, duplicate `detection_index` values trigger error logs.
   - Across the video sequence, duplicate `frame_id` values trigger error logs and skip duplicate ingestion.

### Non-Silent Error Accounting
Rejected detections are never silently dropped. Every rejection is recorded in `feature_errors.json`:
```json
{
  "frame_id": "frame_000005.jpg",
  "detection_index": 2,
  "error_type": "INVERTED_X_COORDINATES",
  "message": "Inverted or zero width bbox: x_min=300.0, x_max=250.0",
  "raw_data": {"bbox_xyxy": [300.0, 100.0, 250.0, 200.0]}
}
```

---

## 6. Output Artifact Hierarchy

Outputs are written under `datasets/processed/<dataset_id>/features/detection/<run_id>/`:
- `detection_features.json`: Array of all valid detection feature records.
- `frame_detection_features.json`: Array of all frame-level feature records (including empty frames).
- `detection_feature_summary.json`: High-level run telemetry, frame/detection counts, and class distributions.
- `feature_metadata.json`: Complete execution metadata, system environment, configuration snapshot, and feature registry.
- `feature_errors.json`: Full catalog of validation errors and rejected records.

---

## 7. CLI User Manual

```powershell
# Standard execution on pre-computed batch detections
.\venv\Scripts\python.exe -m src.features.detection_feature_extractor \
  --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json

# Custom destination with overwrite enabled
.\venv\Scripts\python.exe -m src.features.detection_feature_extractor \
  --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json \
  --output datasets/processed/driving_sample/features/detection/latest \
  --overwrite

# Smoke test on partial frame sequence
.\venv\Scripts\python.exe -m src.features.detection_feature_extractor \
  --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json \
  --max-frames 30 \
  --verbose
```

### CLI Arguments
- `--input`, `-i`: Path to `detection_results.json` (required).
- `--config`, `-c`: Optional path to `feature_config.yaml`.
- `--output`, `-o`: Custom output directory.
- `--max-frames`, `-m`: Limit processing to first $N$ frames (flags run as partial).
- `--start-frame`: Starting source frame index (inclusive).
- `--end-frame`: Ending source frame index (inclusive).
- `--overwrite`: Overwrite destination directory if it exists.
- `--verbose`, `-v`: Output debug logs and exception tracebacks.

---

## 8. Real-Data Validation Results

The module was validated against genuine BDD100K driving detections (`driving_sample`):
- **Source Artifact:** `run_20261117_115348_e1789b/detection_results.json`
- **Frames Processed:** 200 / 200 (100% success)
- **Valid Detections Extracted:** 211
- **Invalid Detections:** 0
- **Zero-Detection Frames Handled:** 34
- **Extraction Latency (200 Frames):** **0.0138 seconds** (~14,500 FPS throughput on CPU)
- **Validation Status:** **`PASS`**

### Sample Bounding Box Calculation Verification
- **Frame:** `frame_000001.jpg`, Detection `0` (`traffic light`, confidence: `0.4381`)
- **Raw Bbox:** $[219.5, 238.1, 276.4, 359.1]$ in $640 \times 360$ image.
- **Computed Width:** $276.4 - 219.5 = 56.9$
- **Computed Height:** $359.1 - 238.1 = 121.0$
- **Computed Area:** $56.9 \times 121.0 = 6884.9$
- **Computed Center:** $(247.95, 298.6)$
- **Normalized Center:** $(0.3874, 0.8294)$
- **Normalized Area:** $\frac{6884.9}{230400} = 0.0299$
- **Manual Verification:** Matches output in `detection_features.json` to 4 decimal places.

---

## 9. Verification & Quality Assurance

- **Unit & Integration Suite (`tests/features/`):**
  - `test_detection_feature_extractor.py`: 39 tests passing (100% pass rate).
  - `test_feature_config.py`: 26 tests passing.
  - `test_feature_schema.py`: 25 tests passing.
  - `test_feature_environment.py`: 15 tests passing.
- **Total Feature Package Tests:** **105 / 105 Passed (100% Pass Rate)**
- **Regression Suite:** Full repository suite verified with zero regressions.
