# PerceptGuard AI – Advanced Object Detection Visualization Engine

## 1. Executive Overview & Technical Scope

This document specifies the architecture, data contracts, visual rendering protocols, CLI commands, and operational benchmarks for the **Advanced Object Detection Visualization Engine** (`src/visualization/detection_visualizer.py`) in **PerceptGuard AI**.

As the spatial verification component of Day 3, the `DetectionVisualizer` takes the serialized batch inference results from Task 4 (`detection_results.json`), correlates each record with its source camera frame, validates bounding box geometries, renders high-contrast class labels and confidence scores, and persists annotated frames alongside execution metadata.

### Strict Architectural Guarantees:
1. **Decoupled Visualization:** Operates purely on serialized detection results. Strictly **zero** model deserialization, **zero** weight loading, and **zero** forward inference passes occur during visualization.
2. **Deterministic Color System:** Every COCO class ID maps to an immutable, mathematically deterministic BGR color across all runs and frames.
3. **Canvas Dimension Preservation:** Preserves source image spatial dimensions ($640 \times 360$ pixels) without distortion or canvas stretching.
4. **Boundary Clamping & Non-Clipping Labels:** Clamps bounding boxes to physical image boundaries $[0, W] \times [0, H]$ and automatically shifts label banners inside the canvas when boxes contact upper or lateral margins.
5. **Zero-Detection Frame Handling:** Frames with 0 detections and successful processing are saved as pristine unannotated copies and recorded as successful visualizations.
6. **Per-Frame Fault Containment:** Corrupted or unreadable image files are cataloged into `visualization_errors.json` without halting batch execution.
7. **Multi-File Versioned Output Architecture (Schema v1.0):** Emits `annotated_frames/`, `visualization_summary.json`, `visualization_errors.json`, and `visualization_metadata.json`.
8. **Source Immutability:** Original preprocessed camera frames in `datasets/processed/driving_sample/preprocessed_frames/` are accessed in read-only mode and remain bit-for-bit identical (SHA-256 verified).

---

## 2. System Architecture & Component Interaction

```
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│  Task 4 detection_results.json (v1.0)  │       │  Day 2 Preprocessed Frames (Read-Only) │
└───────────────────┬────────────────────┘       └───────────────────┬────────────────────┘
                    │                                                │
                    ▼                                                │
┌────────────────────────────────────────┐                           │
│     JSON Schema & Record Validator     │                           │
│ - Validates schema_version == "1.0"    │                           │
│ - Verifies run_id, dataset_id, frames  │                           │
└───────────────────┬────────────────────┘                           │
                    │                                                │
                    ▼                                                │
┌────────────────────────────────────────┐                           │
│     Source Image Path Resolution       │ ◄─────────────────────────┘
│ - Checks exact image_path              │
│ - Resolves against preprocessed layout │
│ - In-memory OpenCV safe decoding       │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│   Box & Label Extraction / Clamping    │
│ - Coordinate finiteness assertion      │
│ - Degenerate geometry rejection        │
│ - Dynamic line thickness & font scale  │
│ - Top/lateral margin overflow clamp    │
│ - Deterministic class color assignment │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│      OpenCV Annotation Rendering       │
│ - Bounding rectangle drawing           │
│ - High-contrast text on solid banner   │
│ - Zero-detection unannotated pass      │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Atomic Multi-File Persistence                       │
│ datasets/processed/<dataset>/visualizations/<vis_run_id>/              │
│   ├── annotated_frames/                                                │
│   │     ├── frame_000001_annotated.jpg                                 │
│   │     └── ...                                                        │
│   ├── visualization_summary.json                                       │
│   ├── visualization_errors.json                                        │
│   └── visualization_metadata.json                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Input Detection Schema Validation

The visualizer ingests Task 4 `detection_results.json` complying with Schema v1.0:
```json
{
  "schema_version": "1.0",
  "run_id": "run_20261117_115348_e1789b",
  "dataset_id": "driving_sample",
  "total_frames": 200,
  "frame_results": [
    {
      "image_filename": "frame_000001.jpg",
      "image_path": "C:\\...\\frame_000001.jpg",
      "extracted_frame_id": 1,
      "source_frame_index": 0,
      "timestamp_seconds": 0.0,
      "image_width": 640,
      "image_height": 360,
      "processing_status": "success",
      "inference_time_ms": 7402.39,
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

Validation rules:
- `schema_version`: Must equal `"1.0"`, else raises `UnsupportedSchemaVersionError`.
- Structural integrity: Rejects malformed JSON, missing fields, or empty arrays.
- Detection counts: Compares `len(detections)` against `detection_count`. Discrepancies are logged in `invalid_detection_records`.

---

## 4. Source Image Resolution & Reading

The engine resolves images using a priority hierarchy:
1. Exact absolute `image_path` stored in the detection JSON record.
2. User-provided `--image-dir` parameter (e.g. `--image-dir custom_frames/`).
3. Standard dataset directory: `datasets/processed/<dataset_id>/preprocessed_frames/<filename>`.
4. Raw extraction directory: `datasets/processed/<dataset_id>/frames/<filename>`.

### Safe In-Memory Decoding
To prevent Windows C-runtime file handle collisions and properly handle unicode paths:
```python
with open(image_path, "rb") as f:
    raw_bytes = f.read()
img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
```

---

## 5. Deterministic Class-Based Color Mapping

Bounding box and label banner colors are assigned deterministically:
- Curated high-visibility BGR tuples for primary autonomous driving perception classes:
  - `person` (0): Vibrant Green `(0, 255, 0)`
  - `bicycle` (1): Cyan `(255, 255, 0)`
  - `car` (2): Vivid Blue `(255, 128, 0)`
  - `motorcycle` (3): Pink `(180, 105, 255)`
  - `bus` (5): Magenta `(255, 0, 255)`
  - `truck` (7): Bright Orange `(0, 140, 255)`
  - `traffic light` (9): Gold `(0, 215, 255)`
  - `stop sign` (11): Pure Red `(0, 0, 255)`
- All other COCO classes: Golden-angle hue distribution in HSV space converted to BGR.
- Guarantee: Class ID 9 always produces `(0, 215, 255)` regardless of detection order or frame number.

---

## 6. Bounding Box & Label Rendering Specification

1. **Box Coordinates:** $[x_1, y_1, x_2, y_2]$ in image space.
2. **Finiteness & Geometry:** Bounding boxes with non-finite values (`NaN`, `Inf`) or inverted geometries ($x_2 \le x_1$ or $y_2 \le y_1$) are rejected.
3. **Clamping:** Out-of-bounds coordinates are clipped to $[0.0, W]$ and $[0.0, H]$.
4. **Dynamic Scaling:**
   - Line thickness: $\max(1, \text{round}(\min(W, H) / 320.0)) = 2\text{ px}$ for $640 \times 360$.
   - Font scale: $\max(0.35, \min(W, H) / 800.0) = 0.45$ for $640 \times 360$.
5. **Label Legibility Banner:**
   - Format: `<class_name> <confidence:.2f>` (e.g. `traffic light 0.44`).
   - Solid background banner drawn in the class color.
   - Text color dynamically switches between black `(0, 0, 0)` and white `(255, 255, 255)` based on background luminance ($0.299R + 0.587G + 0.114B$).
6. **Margin Clamping:** If $y_1 - \text{label\_height} < 0$, the label banner is drawn inside the upper box border rather than clipped off the top canvas edge.

---

## 7. Zero-Detection Frame Handling

In autonomous driving video sequences, camera frames frequently contain zero objects (e.g., clear skies, open rural highways).
- **Behavior:** The visualizer loads the input image and saves a clean copy as `frame_XXXXXX_annotated.jpg`.
- **Classification:** Recorded as `status: "success"` with `boxes_rendered: 0`.
- **Accounting:** Increments `frames_without_detections` and `successful_visualizations`. It is **never** counted as a visualization failure.

---

## 8. Output Directory Layout & Multi-File Artifacts

All visualization runs are persisted under:
`datasets/processed/<dataset_name>/visualizations/<run_id>/`

### 8.1 `visualization_summary.json`
```json
{
  "schema_version": "1.0",
  "run_id": "vis_20261117_173857_5b836e",
  "dataset_id": "driving_sample",
  "detection_run_id": "run_20261117_115348_e1789b",
  "total_frames_in_detection_results": 200,
  "total_frames_attempted": 200,
  "successful_visualizations": 200,
  "failed_visualizations": 0,
  "frames_with_detections": 166,
  "frames_without_detections": 34,
  "total_boxes_rendered": 211,
  "invalid_detection_records": 0,
  "total_visualization_time_seconds": 2.8797,
  "average_visualization_time_ms": 14.4,
  "generated_at": "2026-11-17T12:09:00.797462+00:00",
  "final_status": "PASS"
}
```

### 8.2 `visualization_errors.json`
```json
{
  "schema_version": "1.0",
  "run_id": "vis_20261117_173857_5b836e",
  "error_count": 0,
  "errors": []
}
```

### 8.3 `visualization_metadata.json`
Records annotation style settings, color samples, software versions, and hardware environment for scientific reproducibility.

---

## 9. Command Line Interface (CLI)

```powershell
# Full batch visualization of Task 4 detection output
python -m src.visualization.detection_visualizer --detection-file datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --overwrite

# Rapid 10-frame smoke test with verbose logging
python -m src.visualization.detection_visualizer --detection-file datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --max-frames 10 --overwrite --verbose

# Sliced range visualization
python -m src.visualization.detection_visualizer --detection-file <path> --start-index 50 --end-index 100 --output-dir outputs/vis_slice
```

### CLI Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--detection-file` | `str` | *(Required)* | Path to `detection_results.json` generated by Task 4. |
| `--image-dir` | `str` | `None` | Custom source image directory (auto-resolved from dataset layout if omitted). |
| `--output-dir` | `str` | `None` | Destination root directory for annotated frames and metadata. |
| `--max-frames` | `int` | `None` | Maximum frames to visualize (for rapid smoke testing). |
| `--start-index` | `int` | `0` | 0-based frame start index for range slicing. |
| `--end-index` | `int` | `None` | 0-based frame end index (exclusive). |
| `--overwrite` | `flag` | `False` | Overwrite destination directory if it already exists. |
| `--verbose` | `flag` | `False` | Enable DEBUG diagnostic logging. |

---

## 10. Automated Test Suite (29 Tests)

Verified by [tests/visualization/test_detection_visualizer.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/visualization/test_detection_visualizer.py):

| Test ID | Test Method | Description |
| :---: | :--- | :--- |
| `01` | `test_01_valid_detection_json` | Valid detection JSON parsing. |
| `02` | `test_02_malformed_json` | Syntax-corrupted JSON rejection. |
| `03` | `test_03_missing_json_file` | Missing JSON file error handling. |
| `04` | `test_04_unsupported_schema_version` | Incompatible schema version rejection. |
| `05` | `test_05_missing_frame_results` | Missing required `frame_results` key. |
| `06` | `test_06_missing_source_image` | Missing source image error reporting. |
| `07` | `test_07_corrupt_source_image` | Corrupted image decoding error handling. |
| `08` | `test_08_valid_bounding_box` | Coordinate and geometry validation. |
| `09` | `test_09_invalid_bounding_box_geometry` | Rejection of inverted boxes ($x_2 \le x_1$). |
| `10` | `test_10_out_of_bounds_bounding_box_clamping` | Canvas boundary coordinate clamping. |
| `11` | `test_11_non_finite_coordinates` | Rejection of `NaN` and `Inf` coordinates. |
| `12` | `test_12_invalid_confidence` | Rejection of confidence outside $[0.0, 1.0]$. |
| `13` | `test_13_missing_class_name` | Fallback class name generation. |
| `14` | `test_14_zero_detection_frame` | Pristine copy preservation on zero detections. |
| `15` | `test_15_multiple_detections` | Accurate multi-detection box rendering. |
| `16` | `test_16_class_based_deterministic_colors` | Deterministic color consistency check. |
| `17` | `test_17_label_rendering` | Contrast label rendering without top clipping. |
| `18` | `test_18_image_dimension_preservation` | Verification that width/height are unchanged. |
| `19` | `test_19_output_directory_creation` | Automatic nested output directory creation. |
| `20` | `test_20_existing_output_protection` | `OutputPersistenceError` when overwrite is False. |
| `21` | `test_21_explicit_overwrite` | Overwriting destination when overwrite is True. |
| `22` | `test_22_failed_visualization_reporting` | Error catalog entry creation on failure. |
| `23` | `test_23_summary_count_accuracy` | Accurate calculation of summary metrics. |
| `24` | `test_24_partial_run_status` | Status set to `WARNING` on sliced runs. |
| `25` | `test_25_batch_visualization_ordering` | Sequential frame order preservation. |
| `26` | `test_26_no_yolo_model_initialization` | Invariant: Zero model loader calls. |
| `27` | `test_27_no_repeated_inference` | Invariant: Zero detector inference calls. |
| `28` | `test_28_source_image_unchanged` | Invariant: Source image SHA-256 hash unchanged. |
| `29` | `test_29_real_artifact_integration` | Live integration test on actual Task 4 detection JSON. |

---

## 11. Real-Data Benchmark Telemetry

Execution on actual Task 4 detection output across all 200 preprocessed driving frames:

| Metric | Measured Value |
| :--- | :--- |
| **Dataset** | `driving_sample` |
| **Detection Source Run** | `run_20261117_115348_e1789b` |
| **Hardware** | Intel Core i3 (8 logical threads), 8 GB RAM, CPU-only |
| **Frames Discovered** | 200 |
| **Frames Attempted** | 200 |
| **Successful Visualizations** | **200 (100.0%)** |
| **Failed Visualizations** | **0 (0.0%)** |
| **Frames With Detections** | 166 (83.0%) |
| **Frames Without Detections** | 34 (17.0%) |
| **Total Boxes Rendered** | **211 boxes** |
| **Average Visualization Latency**| **14.40 ms / frame** (~**69.4 FPS** on CPU) |
| **Total Visualization Time** | **2.88 s** |
| **Output Status** | **`PASS`** |

---

## 12. Decoupling & Invariant Certification

- **Zero Re-Inference:** Confirmed via test mocks that neither `YOLOModelLoader.load_model` nor `SingleImageDetector.detect` is called during visualization.
- **Source Preservation:** The original preprocessed driving frames in `datasets/processed/driving_sample/preprocessed_frames/` were verified bit-for-bit immutable.
- **Memory Footprint:** Sequential single-image streaming ensures RSS memory usage remains under 300 MB throughout the 200-frame execution.

