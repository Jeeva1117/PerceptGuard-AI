# Dataset Quality Assurance & Validation Pipeline Specification

## 1. Executive Summary & Quality Assurance Role

The **Dataset Quality Assurance and Validation Pipeline** (`src/validation/dataset_validator.py`) represents the formal gatekeeper for Day 2 of the **PerceptGuard AI** autonomous driving perception research project. 

Its architectural mandate is to perform an exhaustive, multi-tier, and non-destructive audit of the complete dataset generated across Tasks 1 through 6 (`driving_sample`), definitively establishing whether the dataset is mathematically, structurally, and statistically qualified for Day 3 deep learning perception tasks (YOLOv8 object detection, ByteTrack multi-object tracking, and reliability modeling).

```
+---------------------------------------------------------------------------------------------------+
|                                  DAY 2 PIPELINE GATEWAY AUDIT                                     |
+---------------------------------------------------------------------------------------------------+
|  [Raw Video]        [Extracted Frames]    [Preprocessed Frames]    [Metadata & Manifest]          |
|  converted_*.mp4    200 JPEGs (720x1280)   200 JPEGs (640x360)     metadata.csv & manifests       |
+---------------------------------------------------------------------------------------------------+
                                            │
                                            ▼
+---------------------------------------------------------------------------------------------------+
|                         DatasetValidator (Non-Destructive Read-Only Engine)                       |
+---------------------------------------------------------------------------------------------------+
|  1. Directory Structure Integrity        5. Tabular Metadata Contract & Bounds                    |
|  2. Source Video Stream Health           6. Manifest & Summary JSON Cross-Conformity              |
|  3. Extracted Frames Decodability        7. End-to-End Bidirectional File Traceability            |
|  4. Preprocessed Frame Tensors & Channels                                                         |
+---------------------------------------------------------------------------------------------------+
                                            │
                                            ▼
+---------------------------------------------------------------------------------------------------+
|                            GATEWAY DECISION & PERSISTENCE ENGINE                                  |
+---------------------------------------------------------------------------------------------------+
|  Status: PASS (0 Errors, 0 Warnings)                                                              |
|  is_ready_for_day_3: TRUE                                                                         |
|  Artifacts: validation_report.json & outputs/reports/dataset_validation_report.json               |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Guarantees

1. **Non-Destructive & Read-Only:**
   Under no circumstances does the validator alter, overwrite, move, or delete any source video, extracted JPEG, preprocessed JPEG, CSV record, or manifest file. All checks operate strictly in read-only access mode.
2. **Deterministic & Idempotent:**
   Executing the validation suite multiple times produces identical verdicts, error lists, warning lists, and aggregate statistics without state leakage.
3. **Multi-Tier Categorical Auditing:**
   Validation checks are partitioned into 7 isolated categories. Each category yields a discrete `CategoryValidationResult` containing status (`PASS`, `WARNING`, or `FAIL`), passing details, warnings, and error reasons.
4. **Hierarchical Decision Logic:**
   - A single `FAIL` in any check triggers overall dataset status `FAIL` and blocks Day 3 execution (`is_ready_for_day_3 = False`).
   - A `WARNING` allows the dataset to proceed conditionally (`is_ready_for_day_3 = True`), flagging advisory notices (e.g., non-fatal visual artifacts or channel deviations).
   - Only a flawless execution across all categories earns a status `PASS`.
5. **Strict Cross-Modal Bidirectional Traceability:**
   Every single frame index in the raw video must map to a physical file in `frames/`, a valid standardized file in `preprocessed_frames/`, and a matching row in `metadata.csv` with identical timestamps.
6. **RFC 8259 Compliant JSON Reporting:**
   Audit results are formatted as strict JSON adhering to `validation_schema_version: 1.0.0` with UTC ISO 8601 timestamps and serialized to auditable filesystem locations.

---

## 3. Seven Categorical Verification Dimensions

| Category ID | Verification Scope | Critical Criteria (Failure Triggers) | Warning Triggers |
| :--- | :--- | :--- | :--- |
| `directory_structure` | Filesystem layout | Missing `raw/`, `processed/<id>/frames/`, `preprocessed_frames/`, or empty required files (`metadata.csv`, `summary.json`, `dataset_manifest.json`). | Unexpected or extraneous temporary directories. |
| `source_video` | Video container & codec | Missing video, 0-byte file, unreadable stream, resolution < 64x64, FPS <= 0 or > 120, non-positive total frames. | Non-standard container format or variable frame rate container. |
| `extracted_frames` | Raw frame extraction | Missing directory, 0 images found, duplicate filenames, non-decodable JPEGs, empty image matrices (`size == 0`). | Non-consecutive frame numbering gaps (if intentional extraction rate). |
| `preprocessed_frames` | Model input tensors | Frame count mismatch with extracted frames, image decode failure, dimension deviation from target ($640 \times 360$). | Single-channel grayscale image (when 3-channel BGR is expected for YOLO). |
| `metadata` | Per-frame tabular ledger | Missing `metadata.csv`, 0 rows, missing required columns (21 columns), duplicate `extracted_frame_id` or `source_frame_index`, negative or non-monotonic timestamps, out-of-bounds metrics (e.g. brightness outside $[0, 255]$). | Extreme outliers in blur, brightness, or contrast distributions. |
| `manifest` | Top-level contracts | Missing `summary.json` or `dataset_manifest.json`, unparseable JSON syntax, `dataset_id` mismatch, dimension mismatch, missing schema version keys. | Extra unrecognized metadata fields. |
| `traceability` | End-to-end provenance | Missing physical file referenced by `original_image_path` or `preprocessed_image_path`, record count mismatch against physical files in directories. | Broken relative link fallbacks. |

---

## 4. Quantitative Quality Metrics & Statistical Bounds

During validation, aggregate quality metrics from `metadata.csv` are audited across all successful frames. In autonomous driving, these values establish baseline sensor telemetry:

| Metric | Measurement Technique | Valid Range | `driving_sample` Actual Value | Engineering Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Blur Metric** | Variance of Laplacian ($\sigma^2(\nabla^2 I)$) | $\ge 0.0$ (Higher = sharper) | **Mean: 436.75** (Std: 126.00, Min: 207.58, Max: 645.51) | Extremely sharp imagery across all 200 frames. Well above minimal threshold ($> 100$), confirming zero motion-blur corruption. |
| **Brightness** | Mean pixel intensity ($\mu_Y$) | $[0.0, 255.0]$ | **Mean: 102.05** (Std: 3.38, Min: 93.20, Max: 111.08) | Balanced daytime urban lighting. Highly stable across the sequence ($\sigma = 3.38$), indicating absence of sudden exposure blowout or darkness. |
| **Contrast** | Standard deviation of grayscale pixels ($\sigma_Y$) | $[0.0, 128.0]$ | **Mean: 62.04** (Std: 2.86, Min: 54.37, Max: 65.19) | Strong dynamic range with clear separation between road surface, vehicles, and sky. |
| **Tensor Size** | Compressed preprocessed JPEG size | $> 0$ bytes | **Mean: 36.79 KB** (Min: 31.58 KB, Max: 39.73 KB) | Consistent storage footprint without anomalous 0-byte corruptions. |
| **Completeness**| Preprocessed successful count / Extracted count | $100.0\%$ | **100.0% (200 / 200)** | Zero dropped or failed frames throughout the entire video sequence. |

---

## 5. Decision Matrix & Gatekeeper Protocol

The validation engine computes overall dataset certification using strict boolean precedence:

```python
has_fail = any(res.status == "FAIL" for res in categories)
has_warning = any(res.status == "WARNING" for res in categories)

if has_fail:
    overall_status = "FAIL"
    is_ready_for_day_3 = False
elif has_warning:
    overall_status = "WARNING"
    is_ready_for_day_3 = True  # Conditional clearance with advisory warnings
else:
    overall_status = "PASS"
    is_ready_for_day_3 = True  # Full production certification
```

---

## 6. Production Audit Results (`driving_sample`)

Executing `DatasetValidator(dataset_id="driving_sample")` against the production dataset produced the following categorical results:

```
============================================================
Dataset Validation Summary: driving_sample
Overall Status: PASS
Ready for Day 3 Perception: True
Total Issues: 0
  - directory_structure: PASS
  - source_video: PASS
  - extracted_frames: PASS
  - preprocessed_frames: PASS
  - metadata: PASS
  - manifest: PASS
  - traceability: PASS
============================================================
```

### Detailed Categorical Audit Log

1. **`directory_structure` -> PASS:**
   - Verified existence of `datasets/raw/`
   - Verified existence of `datasets/processed/driving_sample/`
   - Verified existence of `datasets/processed/driving_sample/frames/`
   - Verified existence of `datasets/processed/driving_sample/preprocessed_frames/`
   - Verified non-empty `metadata.csv` (72,701 bytes)
   - Verified non-empty `summary.json` (591 bytes)
   - Verified non-empty `dataset_manifest.json` (893 bytes)
2. **`source_video` -> PASS:**
   - Verified presence of `datasets/raw/converted_026c7465-309f6d33.mp4`
   - Verified container decodability via OpenCV `VideoCapture`
   - Native dimensions: $720 \times 1280$
   - Native frame rate: $30.09\text{ FPS}$
   - Total source frames: $1,204$
   - Duration: $40.01\text{ seconds}$
   - FourCC Codec: `'FMP4'`
3. **`extracted_frames` -> PASS:**
   - Verified 200 raw extracted JPEG image files
   - Spot-checked image matrix decoding at indices $[0, 100, 199]$:
     - `frame_000001.jpg`: $720 \times 1280$ px
     - `frame_000101.jpg`: $720 \times 1280$ px
     - `frame_000200.jpg`: $720 \times 1280$ px
   - Verified 0 duplicate filenames and strict alphanumeric sequence formatting
4. **`preprocessed_frames` -> PASS:**
   - Verified 200 standardized preprocessed JPEG image files
   - Verified exact 1:1 match with extracted frame count ($200 = 200$)
   - Spot-checked image matrix dimensions at indices $[0, 100, 199]$:
     - All images decode to exact target dimensions: $640 \times 360$ px
     - All images maintain 3 color channels (BGR color space)
5. **`metadata` -> PASS:**
   - Verified `metadata.csv` contains exactly 200 data records + 1 header row
   - All 21 required columns present and correctly typed
   - Verified uniqueness: 200 distinct `extracted_frame_id` values $[1, 200]$
   - Verified uniqueness: 200 distinct `source_frame_index` values $[0, 1194]$
   - Verified strict monotonicity: timestamps advance from $0.0000\text{s}$ to $39.6766\text{s}$
   - Verified metric bounds: blur $\in [207.58, 645.51]$, brightness $\in [93.20, 111.08]$, contrast $\in [54.37, 65.19]$
   - Preprocessing status: 100% `SUCCESS`, validation status: 100% `VALID`
6. **`manifest` -> PASS:**
   - Validated JSON syntax for `summary.json` and `dataset_manifest.json`
   - Verified dataset ID consistency: `"driving_sample"`
   - Verified target dimension alignment ($640 \times 360$)
   - Verified manifest schema version `"1.0.0"`
7. **`traceability` -> PASS:**
   - Audited physical file existence for all 200 `original_image_path` entries
   - Audited physical file existence for all 200 `preprocessed_image_path` entries
   - 100% unbroken 1:1 mapping: Source Video $\to$ Frame Index $\to$ Extracted Frame $\to$ Preprocessed Tensor $\to$ Tabular Record

---

## 7. Artifact Persistence & Schema

Validation reports are serialized into two distinct locations for auditability:
- Primary dataset directory: `datasets/processed/driving_sample/validation_report.json`
- Top-level reporting directory: `outputs/reports/dataset_validation_report.json`

### Report Schema Reference (`validation_report.json`)

```json
{
  "validation_schema_version": "1.0.0",
  "dataset_id": "driving_sample",
  "validation_timestamp": "2026-11-17T09:23:48.363570+00:00",
  "overall_status": "PASS",
  "is_ready_for_day_3": true,
  "checks": {
    "directory_structure": { "status": "PASS", "details": [...], "warnings": [], "errors": [] },
    "source_video":        { "status": "PASS", "details": [...], "warnings": [], "errors": [] },
    "extracted_frames":    { "status": "PASS", "details": [...], "warnings": [], "errors": [] },
    "preprocessed_frames": { "status": "PASS", "details": [...], "warnings": [], "errors": [] },
    "metadata":            { "status": "PASS", "details": [...], "warnings": [], "errors": [] },
    "manifest":            { "status": "PASS", "details": [...], "warnings": [], "errors": [] },
    "traceability":        { "status": "PASS", "details": [...], "warnings": [], "errors": [] }
  },
  "statistics": {
    "dataset_id": "driving_sample",
    "total_extracted_frames": 200,
    "total_preprocessed_frames": 200,
    "total_failed_frames": 0,
    "completeness_percentage": 100.0,
    "dimension_consistency_percentage": 100.0,
    "blur_score": { "mean": 436.75, "std": 126.0, "min": 207.58, "max": 645.51 },
    "brightness_score": { "mean": 102.05, "std": 3.38, "min": 93.2, "max": 111.08 },
    "contrast_score": { "mean": 62.04, "std": 2.86, "min": 54.37, "max": 65.19 },
    "file_size_bytes": { "mean": 36793.18, "std": 2276.1, "min": 31579.0, "max": 39731.0 }
  },
  "issues": [],
  "recommendations": [
    "Dataset validation succeeded. Dataset is certified READY for Day 3 YOLO detection."
  ]
}
```

---

## 8. Verification & Test Suite

The test suite in `tests/test_dataset_validator.py` executes 21 dedicated unit and integration tests covering synthetic failure modes and real-dataset validation:

| Test ID | Test Case | Target Dimension | Expected Outcome |
| :--- | :--- | :--- | :--- |
| `test_01` | `test_01_valid_dataset` | Complete clean dataset | Overall status `PASS`, 0 issues |
| `test_02` | `test_02_missing_raw_video` | Source video file deleted | `source_video` status `FAIL` |
| `test_03` | `test_03_missing_frames_directory` | `frames/` deleted | `directory_structure` status `FAIL` |
| `test_04` | `test_04_missing_preprocessed_frames` | `preprocessed_frames/` deleted | `preprocessed_frames` status `FAIL` |
| `test_05` | `test_05_corrupted_image` | Header-corrupted image file | `preprocessed_frames` status `FAIL` |
| `test_06` | `test_06_missing_metadata_csv` | `metadata.csv` unlinked | `metadata` status `FAIL` |
| `test_07` | `test_07_missing_metadata_columns` | CSV missing columns | Schema check detects `FAIL` |
| `test_08` | `test_08_duplicate_frame_ids` | Duplicate frame IDs | Uniqueness check detects `FAIL` |
| `test_09` | `test_09_invalid_timestamps` | Negative / non-monotonic timestamps | Monotonicity check detects `FAIL` |
| `test_10` | `test_10_broken_frame_paths` | Metadata references non-existent files | `traceability` status `FAIL` |
| `test_11` | `test_11_invalid_summary_json` | Corrupted JSON syntax in summary | `manifest` status `FAIL` |
| `test_12` | `test_12_invalid_manifest` | Missing keys in manifest | `manifest` status `FAIL` |
| `test_13` | `test_13_frame_counts_mismatch` | Extracted count != preprocessed count | `preprocessed_frames` status `FAIL` |
| `test_14` | `test_14_broken_source_to_frame_mapping`| Deleted preprocessed frame on valid row | `traceability` status `FAIL` |
| `test_15` | `test_15_empty_dataset` | Zero extracted frame images | `extracted_frames` status `FAIL` |
| `test_16` | `test_16_warning_status_behavior` | Single-channel grayscale image | Check yields `WARNING`, no errors |
| `test_17` | `test_17_critical_failure_behavior` | Critical check failure | Overall status `FAIL`, Day 3 blocked |
| `test_18` | `test_18_overall_status_calculation` | FAIL > WARNING > PASS hierarchy | Decision precedence validated |
| `test_19` | `test_19_validation_report_generation` | RFC 8259 JSON serialization | Valid JSON structure persisted |
| `test_20` | `test_20_repeated_validation_non_destructive`| Idempotency check | File hashes identical before/after |
| `test_21` | `test_21_real_dataset_integration` | Live `driving_sample` audit | Live dataset passes all checks |

### Running Dataset Validation

```powershell
# Execute CLI validation on primary dataset
.\venv\Scripts\python.exe src/validation/dataset_validator.py --dataset-id driving_sample --output-report outputs/reports/dataset_validation_report.json

# Execute unit and integration tests
.\venv\Scripts\python.exe tests/test_dataset_validator.py

# Execute full repository pytest suite
.\venv\Scripts\pytest.exe
```
