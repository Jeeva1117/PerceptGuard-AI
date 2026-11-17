# Metadata Generation Engine Specification

## 1. Overview & Architectural Role

The **Metadata Generation Engine** (`src/video_processing/metadata_generator.py`) serves as Stage 4 of the **PerceptGuard AI** perception pipeline. It acts as the immutable data contract and audit record connecting raw driving video streams (Stage 1), extracted sequential frames (Stage 2), and standardized preprocessed tensor images (Stage 3).

Its primary objective is generating schema-validated, strictly structured tabular (`metadata.csv`) and aggregate telemetry (`summary.json`) files that preserve end-to-end provenance for future YOLOv8 detection, ByteTrack tracking, temporal feature construction, and LSTM reliability forecasting.

```
datasets/
├── raw/
│   └── converted_026c7465-309f6d33.mp4   # Raw Video Stream (30.09 FPS, 1204 frames)
└── processed/
    └── driving_sample/
        ├── frames/                        # Raw Extracted Frames (Stage 2)
        ├── preprocessed_frames/           # Standardized 640x360 Letterbox Frames (Stage 3)
        ├── metadata.csv                   # Per-Frame Provenance Contract (Stage 4)
        └── summary.json                   # Dataset Aggregate Telemetry (Stage 4)
```

---

## 2. Metadata Schema Definition

The schema version is **`1.0.0`**. Every row in `metadata.csv` corresponds to exactly one extracted discrete source frame.

### Complete Column Specification

| Column Name | Type | Nullable | Description & Domain Interpretation |
| :--- | :--- | :--- | :--- |
| `dataset_id` | `string` | No | Logical dataset collection identifier (`driving_sample`). |
| `source_video` | `string` | No | Basename of original source video file (`converted_026c7465-309f6d33.mp4`). |
| `extracted_frame_id` | `integer` | No | 1-indexed sequential extracted frame identifier ($1, 2, 3 \dots 200$). |
| `source_frame_index` | `integer` | No | 0-indexed discrete frame position in source container ($0, 6, 12 \dots 1,198$). |
| `timestamp_seconds` | `float` | No | Physical capture time from video onset ($t = \text{source\_frame\_index} / \text{source\_fps}$). |
| `original_image_filename` | `string` | No | Basename of raw extracted image (`frame_000001.jpg`). |
| `original_image_path` | `string` | No | Filesystem path to raw extracted image. |
| `preprocessed_image_filename` | `string` | Yes | Basename of preprocessed image (null if preprocessing failed). |
| `preprocessed_image_path` | `string` | Yes | Filesystem path to preprocessed image (null if preprocessing failed). |
| `file_size_bytes` | `integer` | Yes | Storage footprint in bytes of the preprocessed image file. |
| `original_width` | `integer` | Yes | Native pixel width of raw extracted frame ($720$ px). |
| `original_height` | `integer` | Yes | Native pixel height of raw extracted frame ($1,280$ px). |
| `processed_width` | `integer` | Yes | Standardized pixel width of preprocessed frame ($640$ px). |
| `processed_height` | `integer` | Yes | Standardized pixel height of preprocessed frame ($360$ px). |
| `channel_count` | `integer` | Yes | Number of color channels ($3$ for BGR/RGB). |
| `blur_score` | `float` | Yes | Laplacian variance sharpness indicator $\text{Var}(\nabla^2 I_{\text{gray}})$. |
| `brightness_score` | `float` | Yes | Mean grayscale intensity $\mu \in [0.0, 255.0]$. |
| `contrast_score` | `float` | Yes | RMS contrast standard deviation $\sigma$. |
| `preprocessing_status` | `string` | No | Status indicator (`SUCCESS`, `FAILED`, `SKIPPED`). |
| `validation_status` | `string` | No | Validation outcome (`VALID`, `INVALID`). |
| `failure_reason` | `string` | Yes | Diagnostic failure description (null if successful). |

---

## 3. Data Integrity & Validation Invariants

Before serialization to disk, the engine enforces strict invariant validation rules:

1. **Uniqueness:**
   - Every `extracted_frame_id` must be unique and strictly sequential ($1 \dots N$).
   - Every `source_frame_index` must be unique and strictly monotonic ($i_{k+1} > i_k$).
2. **Chronological Monotonicity:**
   - Timestamps must be non-negative ($t \ge 0.0$) and strictly increasing ($t_{k+1} > t_k$).
3. **Spatial Feasibility:**
   - Spatial dimensions ($W_0, H_0, W_{\text{proc}}, H_{\text{proc}}$) must be strictly positive integers.
   - Channel count must equal 1, 3, or 4.
4. **Numeric Score Bounds:**
   - $\text{blur\_score} \ge 0.0$
   - $0.0 \le \text{brightness\_score} \le 255.0$
   - $\text{contrast\_score} \ge 0.0$
5. **Fault-Tolerant Preservation:**
   - If an image fails decoding or validation, it is NOT dropped from `metadata.csv`. Its record is retained with `preprocessing_status="FAILED"`, `validation_status="INVALID"`, and diagnostic `failure_reason`.

---

## 4. RFC 8259 Strict JSON Serialization Policy

Standard Python `json.dumps()` may serialize `float('nan')` to non-standard JavaScript `NaN`. To prevent breaking downstream production consumers, `MetadataGenerator`:

- Maps all `None`, `math.isnan()`, or non-finite float metrics directly to strict JSON `null`.
- Records generation timestamps in standard ISO 8601 format with explicit UTC timezone (`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`).

### Sample `summary.json` Structure
```json
{
  "dataset_id": "driving_sample",
  "source_video": "converted_026c7465-309f6d33.mp4",
  "source_video_fps": 30.09,
  "source_video_resolution": [720, 1280],
  "total_source_frames": 1204,
  "total_extracted_frames": 200,
  "total_preprocessed_frames": 200,
  "total_failed_frames": 0,
  "average_blur_score": 436.75,
  "average_brightness_score": 102.05,
  "average_contrast_score": 62.04,
  "processed_image_dimensions": [640, 360],
  "metadata_generation_timestamp": "2026-11-17T08:54:22.483474+00:00",
  "metadata_schema_version": "1.0.0"
}
```

---

## 5. Usage Example

```python
from pathlib import Path
from src.video_processing import MetadataGenerator

# Initialize generator
generator = MetadataGenerator(
    dataset_id="driving_sample",
    source_video_path="datasets/raw/converted_026c7465-309f6d33.mp4",
    raw_frames_dir="datasets/processed/driving_sample/frames",
    preprocessed_frames_dir="datasets/processed/driving_sample/preprocessed_frames",
)

# Compile metadata records and persist files
records = generator.generate_metadata()
csv_file = generator.save_metadata_csv()
json_file = generator.save_summary_json()

print(f"Generated {len(records)} records.")
print(f"Metadata CSV: {csv_file}")
print(f"Summary JSON: {json_file}")
```

---

## 6. Verification & Automated Test Suite

Automated test suite (`tests/test_metadata_generator.py`) verifies 16 test cases covering schema invariants, missing values, duplicate detection, invalid timestamps, failed preprocessing, and real dataset integration:

```powershell
python tests/test_metadata_generator.py
```

```text
test_01_valid_record_creation ... ok
test_02_csv_generation ... ok
test_03_csv_read_back_validation ... ok
test_04_json_summary_generation ... ok
test_05_missing_image_handling ... ok
test_06_missing_optional_metrics ... ok
test_07_duplicate_frame_id_detection ... ok
test_08_duplicate_source_frame_index_detection ... ok
test_09_invalid_timestamps_detection ... ok
test_10_incorrect_image_dimensions ... ok
test_11_original_to_preprocessed_mapping ... ok
test_12_failed_preprocessing_records_retention ... ok
test_13_empty_dataset_handling ... ok
test_14_numeric_quality_bounds ... ok
test_15_rfc8259_json_serialization_nulls ... ok
test_16_real_dataset_integration ... ok

----------------------------------------------------------------------
Ran 16 tests in 6.424s
OK
```

---

## 7. Real-Dataset Production Telemetry

Generated from actual BDD100K driving sample assets:

- **Dataset Identifier:** `driving_sample`
- **Schema Version:** `1.0.0`
- **Total Frame Records:** `200`
- **Successfully Preprocessed Records:** `200` (100.0%)
- **Failed Records:** `0` (0.0%)
- **First Frame:** `frame_000001.jpg` (Source index: 0, Time: `0.0000 s`)
- **Last Frame:** `frame_000200.jpg` (Source index: 1,198, Time: `39.8139 s`)
- **Quality Indicator Summary:**
  - Mean Blur: `436.75` (Range: `[207.58, 645.51]`)
  - Mean Brightness: `102.05` (Range: `[93.20, 111.08]`)
  - Mean Contrast: `62.04` (Range: `[54.37, 65.19]`)
- **Output Files:**
  - `datasets/processed/driving_sample/metadata.csv` (`48.2 KB`)
  - `datasets/processed/driving_sample/summary.json` (`548 bytes`)
- **Data Governance:** `datasets/processed/*` strictly excluded from Git via `.gitignore`.
