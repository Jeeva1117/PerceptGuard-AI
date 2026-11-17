# PerceptGuard AI – Advanced Batch Object Detection Engine

## 1. Executive Overview & Technical Scope

This document details the architecture, data contracts, failure containment mechanisms, CLI interface, and operational benchmarks of the **Advanced Batch Object Detection Engine** (`src/detection/batch_detector.py`) in **PerceptGuard AI**.

As the sequence-level inference pipeline connecting Day 2 frame extraction to Day 3 tracking and Day 4 reliability forecasting, the `BatchDetector` is engineered to process complete video frame sequences or sliced frame subsets under safety-conscious autonomous driving computer vision standards.

Key capabilities and guarantees:
1. **Deterministic Dataset Discovery:** Automatically locates standardized image sequences in `datasets/processed/<dataset_name>/preprocessed_frames/` or flat directory hierarchies.
2. **Chronological Natural Sorting:** Enforces alphanumeric natural ordering (`frame_000001.jpg` before `frame_000002.jpg`) avoiding alphabetical ordering bugs.
3. **Day 2 Metadata Traceability:** Joins frame files against `metadata.csv` to correlate bounding boxes with physical camera timestamps ($t$), extracted frame IDs, and raw source indices.
4. **Single-Pass Model Loading:** Instantiates `YOLOModelLoader` and deserializes model weights exactly once per batch run, eliminating initialization overhead.
5. **Memory-Bounded CPU Execution:** Processes frames sequentially, releasing decoded OpenCV image arrays immediately after inference to maintain RSS memory usage below 300 MB on hardware-constrained host environments (e.g. Intel Core i3, 8 GB RAM).
6. **Per-Frame Fault Containment:** Catches corrupted, malformed, or unreadable frames without halting the batch run, cataloging failures into a dedicated error register.
7. **Multi-File Versioned Output Architecture (v1.0):** Emits four structured JSON artifacts (`detection_results.json`, `detection_summary.json`, `detection_errors.json`, `run_metadata.json`) per run.
8. **Run Status Protocol:** Distinguishes complete, pristine dataset executions (`PASS`) from smoke tests or sliced ranges (`WARNING`) and catastrophic run aborts (`FAIL`).

---

## 2. System Architecture & Component Interaction

```
                     ┌──────────────────────────────────────────────┐
                     │   Dataset Directory & Day 2 Metadata.csv     │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │           BatchDetector Pipeline             │
                     │  - Natural Sorting & Extension Filtering     │
                     │  - Slice / Range Resolution                  │
                     │  - Metadata Hash Join (extracted_frame_id)   │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │         Single-Pass Model Loader             │
                     │  - Loads YOLO11n weights once (FP32 CPU)     │
                     │  - Injects into SingleImageDetector instance │
                     └──────────────────────┬───────────────────────┘
                                            │
             ┌──────────────────────────────┴──────────────────────────────┐
             │                                                             │
             ▼                                                             ▼
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│     Success Path (Frame i)   │                              │     Failure Path (Frame j)   │
│ - Validate image buffer      │                              │ - Corrupt image / decode err │
│ - Predict bounding boxes     │                              │ - Contain exception cleanly  │
│ - Calibrate coordinates      │                              │ - Append to error catalog    │
│ - Release image matrix       │                              │ - Emit failed FrameRecord    │
│ - Append FrameDetectionRecord│                              └──────────────┬───────────────┘
└────────────┬─────────────────┘                                             │
             │                                                             │
             └──────────────────────────────┬──────────────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │         Atomic Multi-File Persistence        │
                     │  datasets/processed/<dataset>/detections/    │
                     │  batch_runs/<run_id>/                        │
                     │    ├── detection_results.json                │
                     │    ├── detection_summary.json                │
                     │    ├── detection_errors.json                 │
                     │    └── run_metadata.json                     │
                     └──────────────────────────────────────────────┘
```

---

## 3. Dataset Discovery, Natural Sorting & Metadata Association

### 3.1 Directory Resolution
The detector dynamically resolves dataset paths:
- Named dataset: `driving_sample` resolves automatically to `<project_root>/datasets/processed/driving_sample/preprocessed_frames/` (falling back to `<project_root>/datasets/processed/driving_sample/` if flat).
- Custom directory: An arbitrary directory path can be supplied directly.
- Supported extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive). Non-image files (`.txt`, `.json`, `.csv`) are safely skipped.

### 3.2 Natural Sorting Algorithm
Standard lexical sorting can place `frame_10.jpg` before `frame_2.jpg`. The `BatchDetector` applies natural key tokenization:
```python
def _natural_sort_key(path: Path) -> List[Union[int, str]]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]
```
This guarantees strictly monotonic chronological processing across extracted camera frame sequences.

### 3.3 Day 2 Metadata Synchronization
If `metadata.csv` exists in the dataset root or parent directory, it is parsed and indexed by `preprocessed_image_filename` and `original_image_filename`. For each frame, the engine correlates:
- `extracted_frame_id`: Sequential integer ID ($1 \dots N$).
- `source_frame_index`: Original video source frame number ($0, 6, 12, \dots$ based on sampling step).
- `timestamp_seconds`: Physical elapsed video time in seconds ($0.0 \dots T$).

If metadata is missing, the engine gracefully processes frames without fabricating values (`extracted_frame_id=None`, `timestamp_seconds=None`), logging a clear warning.

---

## 4. Single-Pass Model Loading & Memory-Bounded CPU Loop

### 4.1 Invariant: Single Model Initialization
Repeated model loading consumes disk I/O and CPU memory. The `BatchDetector` enforces that the YOLO neural network is deserialized exactly once:
1. `YOLOModelLoader.load_model()` executes once at the start of `process_batch()`.
2. The initialized model reference is injected into a shared `SingleImageDetector` instance.
3. Subsequent frames execute `detector.detect()` without model re-allocation.

### 4.2 Sequential Memory Boundary
For long driving sequences ($N > 10,000$ frames), caching full BGR frames in memory can crash host systems. The sequential processing loop strictly operates on one frame at a time:
- In-memory decoding via `cv2.imdecode(np.frombuffer(...))`.
- Forward inference pass on CPU (FP32).
- Data extraction into lightweight typed dataclasses (`FrameDetectionRecord`).
- Explicit deallocation of the numpy array before fetching the next frame.
- Measured steady-state RSS memory usage remains under 300 MB throughout a 200-frame run.

---

## 5. Fault Containment & Error Cataloging

Frame-level corruptions (truncated files, non-image content, missing permissions) must not crash an autonomous driving perception batch.

When an error occurs during frame validation or inference:
1. The exception is caught and categorized (`ImageDecodeError`, `UnsupportedImageFormatError`, `DetectionInferenceError`, etc.).
2. A `BatchErrorRecord` is appended to the error catalog with:
   - `frame_path`: Absolute path of the failing file.
   - `frame_id`: Correlated extracted frame ID (if available).
   - `source_frame_index`: Correlated video frame index.
   - `error_category`: Exact class name of the raised exception.
   - `error_message`: Full diagnostic string.
   - `timestamp`: UTC ISO-8601 timestamp.
3. A tombstone `FrameDetectionRecord` is recorded with `processing_status: "failed"`, `detection_count: 0`, and `detections: []`.
4. The loop proceeds immediately to frame $i+1$.
5. The run concludes with all errors serialized to `detection_errors.json`.

---

## 6. Output Schema Specification (v1.0)

Every batch execution produces a timestamped run directory under:
`datasets/processed/<dataset_name>/detections/batch_runs/<run_id>/`
(or a user-specified custom `--output-dir`).

### 6.1 `detection_results.json`
Contains frame-by-frame detections, bounding boxes, confidences, and latencies:
```json
{
  "schema_version": "1.0",
  "run_id": "run_20261117_115348_e1789b",
  "dataset_id": "driving_sample",
  "generated_at": "2026-11-17T11:54:12.069944+00:00",
  "model_name": "yolo11n",
  "model_weights_reference": "C:\\Users\\acer\\Desktop\\PerceptGuard-AI\\models\\yolo\\yolo11n.pt",
  "effective_configuration": { ... },
  "total_frames": 200,
  "frame_results": [
    {
      "image_filename": "frame_000001.jpg",
      "image_path": "...\\frame_000001.jpg",
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

### 6.2 `detection_summary.json`
Aggregated operational statistics and semantic class distribution:
```json
{
  "schema_version": "1.0",
  "run_id": "run_20261117_115348_e1789b",
  "dataset_id": "driving_sample",
  "total_frames_discovered": 200,
  "total_frames_attempted": 200,
  "successful_frames": 200,
  "failed_frames": 0,
  "frames_with_detections": 166,
  "frames_without_detections": 34,
  "total_detections": 211,
  "average_detections_per_successful_frame": 1.05,
  "average_inference_time_ms": 111.0,
  "median_inference_time_ms": 67.89,
  "total_processing_time_seconds": 23.27,
  "class_distribution": {
    "traffic light": 159,
    "truck": 31,
    "car": 7,
    "potted plant": 6,
    "bus": 3,
    "surfboard": 2,
    "train": 1,
    "person": 1,
    "vase": 1
  },
  "metadata_association_count": 200,
  "missing_metadata_count": 0,
  "model_name": "yolo11n",
  "device": "cpu",
  "generated_at": "2026-11-17T11:54:12.069944+00:00",
  "is_partial_run": false,
  "final_status": "PASS"
}
```

### 6.3 `detection_errors.json`
Exhaustive catalog of failed frames (empty on clean runs):
```json
{
  "schema_version": "1.0",
  "run_id": "run_20261117_115348_e1789b",
  "error_count": 0,
  "errors": []
}
```

### 6.4 `run_metadata.json`
Hardware, environment, and configuration snapshot for scientific reproducibility:
```json
{
  "schema_version": "1.0",
  "run_id": "run_20261117_115348_e1789b",
  "dataset_id": "driving_sample",
  "model_name": "yolo11n",
  "execution_device": "cpu",
  "model_load_duration_ms": 191.96,
  "software_versions": {
    "python": "3.13.3",
    "torch": "2.15.0+cpu",
    "ultralytics": "8.4.215",
    "opencv": "5.0.0"
  },
  "environment": {
    "os": "Windows-11-10.0.26200-SP0",
    "processor": "Intel64 Family 6 Model 190 Stepping 0, GenuineIntel",
    "cuda_available": false
  },
  "effective_configuration": { ... },
  "generated_at": "2026-11-17T11:54:12.069944+00:00"
}
```

---

## 7. Run Status Protocol

To prevent test-mode artifacts from masquerading as full validation datasets in production:
- **`PASS`**: Every discovered frame in the dataset was processed, 0 frames failed, and no slicing parameters (`--max-frames`, `--start-index`, `--end-index`) were applied.
- **`WARNING`**: Execution finished without runtime errors, but a subset of frames was processed (e.g. `--max-frames 10` smoke test) or some individual frames failed during processing.
- **`FAIL`**: Batch execution encountered an unrecoverable system exception (e.g. missing dataset, empty dataset, corrupted model weights).

---

## 8. Command Line Interface (CLI)

The module provides a rich, enterprise-grade CLI interface:

```powershell
# Full dataset batch run on standard driving sample
python -m src.detection.batch_detector --dataset driving_sample

# Smoke test (first 10 frames) with verbose telemetry and overwrite enabled
python -m src.detection.batch_detector --dataset driving_sample --max-frames 10 --overwrite --verbose

# Sliced range processing (frames 50 through 150)
python -m src.detection.batch_detector --dataset driving_sample --start-index 50 --end-index 150

# Custom output destination and configuration preset
python -m src.detection.batch_detector --dataset driving_sample --output-dir outputs/detections/run_01 --preset edge_device
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `driving_sample` | Dataset name in `datasets/processed/` or absolute path to frame directory. |
| `--config` | `str` | `None` | Optional path to custom YAML model configuration. |
| `--weights` | `str` | `None` | Path to YOLO weights checkpoint (`.pt`). |
| `--preset` | `str` | `None` | Runtime configuration preset (`default`, `high_accuracy`, `edge_device`). |
| `--output-dir` | `str` | `None` | Output directory destination for the 4 batch run artifacts. |
| `--max-frames` | `int` | `None` | Maximum number of frames to process (useful for rapid smoke tests). |
| `--start-index` | `int` | `0` | 0-based frame start index for range slicing. |
| `--end-index` | `int` | `None` | 0-based frame end index (exclusive). |
| `--overwrite` | `flag` | `False` | Overwrite destination directory if it already exists. |
| `--verbose` | `flag` | `False` | Enable DEBUG level diagnostic logging. |

---

## 9. Automated Verification Suite (29 Tests)

The engine is verified by a 29-test comprehensive test suite (`tests/detection/test_batch_detector.py` and `tests/test_batch_detector.py`):

| Test Case | Method | Description |
| :--- | :--- | :--- |
| `01` | `test_01_dataset_discovery` | Verifies resolution of nested `preprocessed_frames/` directory. |
| `02` | `test_02_deterministic_image_ordering` | Enforces natural numeric sort order across frame paths. |
| `03` | `test_03_supported_extensions` | Confirms `.jpg`, `.jpeg`, and `.png` image discovery. |
| `04` | `test_04_unsupported_extensions_ignored_or_handled` | Rejects non-image files (`.txt`, `.json`, `.csv`). |
| `05` | `test_05_missing_dataset_directory` | Raises `DatasetNotFoundError` on non-existent paths. |
| `06` | `test_06_empty_dataset_directory` | Raises `DatasetEmptyError` on directory without valid images. |
| `07` | `test_07_metadata_association` | Verifies `metadata.csv` joining and field propagation. |
| `08` | `test_08_missing_metadata` | Handles absent `metadata.csv` without fabricating timestamps. |
| `09` | `test_09_duplicate_metadata_entries` | Safely handles duplicate CSV rows without breaking. |
| `10` | `test_10_corrupt_image_recovery` | Confirms corrupted image recovery and logging to errors catalog. |
| `11` | `test_11_model_initialized_only_once` | Proves single model initialization invariant. |
| `12` | `test_12_sequential_processing` | Confirms sequential execution in strictly natural order. |
| `13` | `test_13_successful_inference` | Verifies detection result aggregation across multiple frames. |
| `14` | `test_14_zero_detection_frames` | Validates frames with 0 detections marked as successful. |
| `15` | `test_15_multiple_detections` | Extracts and records multiple detections within a single frame. |
| `16` | `test_16_bounding_box_validation` | Asserts clamping and bounding box boundary integrity. |
| `17` | `test_17_confidence_validation` | Verifies detection confidence scores adhere to $[0.0, 1.0]$. |
| `18` | `test_18_class_mapping` | Maps numerical YOLO class IDs to standard COCO semantic names. |
| `19` | `test_19_per_frame_inference_recovery` | Recovers cleanly from individual frame inference exceptions. |
| `20` | `test_20_model_init_failure_halts_run` | Halts batch run when model initialization fails. |
| `21` | `test_21_output_json_schemas` | Validates schema integrity of all 4 output JSON artifacts. |
| `22` | `test_22_summary_statistics` | Validates calculations for averages, medians, and class counts. |
| `23` | `test_23_error_report_generation` | Confirms errors are serialized to `detection_errors.json`. |
| `24` | `test_24_unique_run_id_behavior` | Ensures distinct timestamped run directory creation. |
| `25` | `test_25_existing_output_protection` | Raises `FileExistsError` without `--overwrite`. |
| `26` | `test_26_partial_run_status_warning` | Confirms smoke/partial runs set status to `WARNING`. |
| `27` | `test_27_full_run_status_pass` | Confirms full dataset runs set status to `PASS`. |
| `28` | `test_28_range_processing_start_end_index` | Tests slicing with `--start-index` and `--end-index`. |
| `29` | `test_29_real_model_smoke_batch` | Live integration test on real driving frames using YOLO11n. |

---

## 10. Driving Sample Benchmark & Execution Metrics

Execution on standard autonomous driving benchmark dataset (`driving_sample`, 200 preprocessed frames at $640 \times 360$):

| Metric | Measured Value |
| :--- | :--- |
| **Hardware** | Intel Core i3 (8 logical threads), 8 GB RAM |
| **Execution Device** | CPU (`torch.float32`) |
| **Model** | YOLO11 Nano (`yolo11n.pt`, 2.6M params) |
| **Discovered Frames** | 200 |
| **Attempted Frames** | 200 |
| **Successful Frames** | 200 (100.0%) |
| **Failed Frames** | 0 (0.0%) |
| **Frames With Detections** | 166 (83.0%) |
| **Frames Without Detections** | 34 (17.0%) |
| **Total Objects Detected** | 211 |
| **Average Objects / Frame** | 1.05 |
| **Model Load Duration** | 191.96 ms |
| **First Frame Latency (JIT)** | 7,402.39 ms |
| **Steady-State Median Latency**| 67.89 ms (~14.7 FPS) |
| **Total Batch Processing Time**| 23.27 s |
| **Memory Footprint (RSS)** | < 300 MB constant |
| **Final Run Status** | **PASS** |

### Semantic Class Distribution Across Sequence
- `traffic light`: 159 detections (75.4%)
- `truck`: 31 detections (14.7%)
- `car`: 7 detections (3.3%)
- `potted plant`: 6 detections (2.8%)
- `bus`: 3 detections (1.4%)
- `surfboard`: 2 detections (0.9%)
- `person`: 1 detection (0.5%)
- `train`: 1 detection (0.5%)
- `vase`: 1 detection (0.5%)

---

## 11. Downstream Module Integration

The `BatchDetector` artifacts directly feed upcoming project milestones:
1. **Day 3 Task 5 (Multi-Object Tracking):** The sorted sequence in `detection_results.json` provides bounding boxes, confidences, and timestamps directly to the ByteTrack / SORT tracking engine to link frame-to-frame bounding boxes into persistent spatial-temporal tracklets.
2. **Day 4 (Feature Engineering):** Sequential bounding box dynamics ($\Delta x, \Delta y, \Delta \text{area}$), detection frequencies, and confidence fluctuations are extracted across rolling temporal windows.
3. **Day 4 Task 4+ (LSTM Reliability Forecasting):** Tracklet histories and confidence trajectories serve as input sequences to predict future detection reliability ($R_{t+h}$).

