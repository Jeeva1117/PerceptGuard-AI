# Final Dataset Structure & Organization Specification

## 1. Overview & Architectural Role

The **Dataset Organization and Management Engine** (`src/video_processing/dataset_organizer.py`) completes Stage 5 of the **PerceptGuard AI** perception pipeline. It establishes an auditable, standardized, and reproducible directory hierarchy that cleanly isolates raw ingest streams from processed computer vision tensors.

The organization decouples intermediate preprocessing artifacts while maintaining bidirectional traceability across source video containers, discrete extracted frames, standardized preprocessed images, tabular metadata (`metadata.csv`), aggregate telemetry (`summary.json`), and the formal dataset contract (`dataset_manifest.json`).

---

## 2. Standardized Directory Layout

```
datasets/
│
├── raw/                                           # Immutable Raw Ingest Containers
│   └── converted_026c7465-309f6d33.mp4            # BDD100K 720x1280 30.09 FPS Driving Sequence
│
└── processed/                                     # Derived ML-Ready Datasets
    └── driving_sample/                            # Primary Dataset Namespace (dataset_id)
        │
        ├── frames/                                # Stage 2: Raw Extracted JPEGs (720x1280)
        │   ├── frame_000001.jpg                   # Extracted ID 1, Source Index 0,  t = 0.0000s
        │   ├── frame_000002.jpg                   # Extracted ID 2, Source Index 6,  t = 0.1994s
        │   └── ... (200 frames total)
        │
        ├── preprocessed_frames/                   # Stage 3: Standardized Preprocessed Tensors (640x360)
        │   ├── frame_000001.jpg                   # Aspect-Ratio Preserved Letterbox BGR JPEG
        │   ├── frame_000002.jpg                   # Centered 202x360 content, 219px horizontal pad
        │   └── ... (200 frames total)
        │
        ├── metadata.csv                           # Stage 4: Per-Frame Tabular Provenance Contract
        ├── summary.json                           # Stage 4: Aggregate Telemetry & Statistics
        └── dataset_manifest.json                  # Stage 5: Formal Top-Level Dataset Manifest Contract
```

---

## 3. Directory Purposes & Storage Conventions

| Directory / File | Status | Retention Policy | Primary Consumer |
| :--- | :--- | :--- | :--- |
| `datasets/raw/` | Read-Only | Permanent (Uncommitted) | `VideoLoader`, `FrameExtractor` |
| `datasets/processed/<id>/frames/` | Read-Only | Intermediate Cache | `FramePreprocessor`, Quality Diagnostics |
| `datasets/processed/<id>/preprocessed_frames/` | Standardized | ML-Ready Target | `YOLOv8Detector`, `ByteTrackTracker` |
| `datasets/processed/<id>/metadata.csv` | Immutable | Per-Frame Ledger | Temporal Feature Extractor, LSTM Forecaster |
| `datasets/processed/<id>/summary.json` | Immutable | Dataset Summary | Model Evaluation & Telemetry Dashboard |
| `datasets/processed/<id>/dataset_manifest.json` | Immutable | Primary Contract | Dataset Ingestion Pipeline & Model Registry |

---

## 4. Dataset Manifest Contract (`dataset_manifest.json`)

The manifest provides a portable, self-describing declaration of the dataset contents, schema versions, and relative filesystem paths.

### Manifest Schema Reference (v1.0.0)

| Field Name | Type | Value / Example | Definition |
| :--- | :--- | :--- | :--- |
| `manifest_schema_version` | `str` | `"1.0.0"` | Semantic version of the dataset manifest schema. |
| `dataset_id` | `str` | `"driving_sample"` | Unique logical collection namespace. |
| `dataset_name` | `str` | `"PerceptGuard Autonomous Driving Sample..."` | Descriptive title for documentation. |
| `source_video` | `str` | `"converted_026c7465-309f6d33.mp4"` | Basename of original source video container. |
| `dataset_created_at` | `str` | `"2026-11-17T09:02:49.246363+00:00"` | ISO 8601 generation timestamp with UTC timezone. |
| `dataset_root` | `str` | `"datasets"` | Relative path to dataset root directory. |
| `raw_video_directory` | `str` | `"datasets/raw"` | Relative path to raw videos directory. |
| `extracted_frames_directory` | `str` | `"datasets/processed/driving_sample/frames"` | Relative path to extracted frames. |
| `preprocessed_frames_directory`| `str` | `"datasets/processed/driving_sample/preprocessed_frames"`| Relative path to preprocessed frames. |
| `metadata_csv_path` | `str` | `"datasets/processed/driving_sample/metadata.csv"` | Relative path to tabular metadata. |
| `summary_json_path` | `str` | `"datasets/processed/driving_sample/summary.json"` | Relative path to aggregate summary. |
| `total_extracted_frames` | `int` | `200` | Verified count of extracted source frames. |
| `total_preprocessed_frames`| `int` | `200` | Verified count of standardized preprocessed frames. |
| `total_failed_frames` | `int` | `0` | Count of unrecoverable or corrupted frames. |
| `image_format` | `str` | `"jpg"` | Image encoding format extension. |
| `processed_image_width` | `int` | `640` | Standardized pixel width. |
| `processed_image_height`| `int` | `360` | Standardized pixel height. |
| `metadata_schema_version`| `str` | `"1.0.0"` | Schema version of associated metadata.csv. |

---

## 5. End-to-End Traceability Architecture

Every standardized preprocessed frame maintains an explicit, verifiable link back to the exact video frame:

```
Source Container: converted_026c7465-309f6d33.mp4 (FPS: 30.09)
       │
       ▼  Sampling Step S = 30.09 / 5.0 = 6.018
Source Frame Index: i_k = round(k * 6.018) -> i_0 = 0, i_1 = 6, ..., i_199 = 1198
       │
       ▼  Timestamp t = i_k / 30.09
Physical Timestamp: 0.0000s, 0.1994s, ..., 39.8139s
       │
       ▼  Extraction ID k + 1
Extracted Raw Image: frames/frame_000001.jpg (720x1280 px)
       │
       ▼  Aspect-Preserved Resizing (s = 0.28125, Letterbox Pad: 219px)
Preprocessed Image: preprocessed_frames/frame_000001.jpg (640x360 px)
       │
       ▼  Ledger Accounting
metadata.csv (Row 1):
  extracted_frame_id=1, source_frame_index=0, timestamp=0.0000s,
  blur=400.19, brightness=104.09, contrast=58.34, status=SUCCESS
```

Traceability guarantees:
1. No synthetic or fabricated frame indices.
2. 1:1 filename correspondence between `frames/frame_XXXXXX.jpg` and `preprocessed_frames/frame_XXXXXX.jpg`.
3. Complete correspondence between `metadata.csv` rows and physical files on disk.

---

## 6. Centralized ML Dataset Access Interface

To prevent hardcoding file paths across future perception modules (Day 3 YOLO detection and ByteTrack tracking), `DatasetOrganizer` provides a clean accessor API:

```python
from src.video_processing import DatasetOrganizer

# Initialize interface
organizer = DatasetOrganizer(dataset_id="driving_sample")

# 1. Retrieve preprocessed frames for batch YOLO inference
frame_paths = organizer.get_preprocessed_frame_paths()
print(f"Discovered {len(frame_paths)} ML-ready frames.")
# First frame: Path(".../datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg")

# 2. Retrieve provenance artifacts
metadata_csv = organizer.get_metadata_csv_path()
summary_json = organizer.get_summary_json_path()
manifest_json = organizer.get_manifest_path()

# 3. Verify dataset integrity before training/inference
report = organizer.verify_dataset_structure()
assert report.is_valid, f"Dataset corrupt: {report.integrity_issues}"
```

---

## 7. Filesystem Safety & Idempotency

- **Non-Destructive Guarantee:** File operations are strictly non-destructive. `organize_dataset()` and `generate_dataset_manifest()` will never delete, move, or overwrite raw video files or extracted frames.
- **Dry-Run Capability:** `organizer.organize_dataset(dry_run=True)` evaluates and reports planned filesystem operations without creating directories or mutating disk state.
- **Strict Idempotency:** Executing `organize_dataset()` or `generate_dataset_manifest()` multiple times produces identical state without duplicate files or resource churn.

---

## 8. Verification & Test Suite

The automated test suite (`tests/test_dataset_structure.py`) covers 13 test cases including directory layout, missing folder traps, existing directory preservation, manifest syntax, metadata existence, frame references, duplicate filename detection, dry-run safety, and idempotency:

```powershell
python tests/test_dataset_structure.py
```

```text
test_01_correct_directory_structure ... ok
test_02_missing_directory_handling ... ok
test_03_existing_directory_handling ... ok
test_04_manifest_generation ... ok
test_05_manifest_json_validity ... ok
test_06_metadata_file_existence ... ok
test_07_frame_file_references_access ... ok
test_08_duplicate_filenames_detection ... ok
test_09_missing_preprocessed_frames_mismatch ... ok
test_10_dataset_path_consistency ... ok
test_11_dry_run_behavior ... ok
test_12_idempotent_execution ... ok
test_13_real_dataset_integration ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.626s
OK
```

---

## 9. Actual Production Dataset Status

- **Dataset Identifier:** `driving_sample`
- **Source Video:** `datasets/raw/converted_026c7465-309f6d33.mp4` (47.47 MB)
- **Extracted Frames Count:** `200` (`frame_000001.jpg` to `frame_000200.jpg`, 45.41 MB)
- **Preprocessed Frames Count:** `200` (`640 × 360` px, 7.02 MB)
- **Failed Frames:** `0`
- **Metadata CSV:** `datasets/processed/driving_sample/metadata.csv` (200 records, 48.2 KB)
- **Summary JSON:** `datasets/processed/driving_sample/summary.json` (548 bytes)
- **Dataset Manifest:** `datasets/processed/driving_sample/dataset_manifest.json` (986 bytes)
- **Git Version Control Status:** Strictly excluded from Git via `.gitignore` (`datasets/raw/*`, `datasets/processed/*`).

