# PerceptGuard AI — Feature Engineering Environment & Architecture Specification

## 1. Executive Summary & Pipeline Context

**PerceptGuard AI** is a predictive reliability forecasting system for edge-deployed computer vision. Instead of treating object detection bounding boxes and confidence scores ($P(\text{class} \mid \text{box})$) as ground-truth reliability indicators, PerceptGuard forecasts **future detection reliability** ($R_{t+h}$) across upcoming video frames.

The end-to-end perception pipeline evolves across modular stages:

$$\text{Video Loading} \longrightarrow \text{Frame Extraction} \longrightarrow \text{Preprocessing} \longrightarrow \text{YOLO Detection} \longrightarrow \text{ByteTrack Association} \longrightarrow \text{Multi-Object Tracking} \longrightarrow \text{Tracking Metrics} \longrightarrow \mathbf{Feature\ Engineering} \longrightarrow \text{LSTM/GRU Forecasting}$$

**Day 5 Task 1** establishes the **Feature Engineering Environment Foundation**:
- Strongly typed, immutable configuration architecture (`FeatureConfig`) with YAML serialization and operational presets.
- Comprehensive mathematical and structural feature contract (**Schema v1.0**) defining 25+ fields spanning detection, motion, visual quality, tracking, and temporal metadata.
- Pre-flight input artifact validator (`validate_input_artifacts`) ensuring synchronization across upstream detection runs, tracking results, dataset metadata, and preprocessed frames.
- Automated runtime environment verification engine (`FeatureEnvironmentVerifier`) profiling CPU allocations, RAM budgets, and scientific library integrity under strict edge hardware constraints.

---

## 2. Hardware Constraints & Operational Profile

All feature engineering components are architected and validated under realistic edge hardware profiles:
- **Processor:** Intel Core i3 (8 logical CPU threads)
- **System Memory:** 8.0 GB RAM
- **Compute Acceleration:** CPU-only execution (strictly zero CUDA GPU dependencies)
- **Operating System:** Windows 11 AMD64
- **Runtime Environment:** Python 3.13+ virtual environment with CPU-specific PyTorch (`torch-2.15.0+cpu`)

Resource budgets enforced in `config/feature_config.yaml`:
- **Worker Allocation:** Sequential processing (`max_workers: 1`) to eliminate inter-thread lock contention.
- **Batch Size:** 100 records per chunk for memory efficiency.
- **Memory Ceiling:** Maximum 500 MB RAM allocation during feature ingestion and record validation.

---

## 3. Directory & Module Topology

```
PerceptGuard-AI/
├── config/
│   ├── feature_config.yaml         # Centralized Feature Engineering YAML configuration
│   └── tracker_config.yaml         # Upstream ByteTrack configuration
├── src/
│   └── features/
│       ├── __init__.py             # Package exports for configuration, schema, and environment
│       ├── feature_config.py       # Strongly typed FeatureConfig dataclass, presets, and CLI
│       ├── feature_schema.py       # Schema v1.0 definitions, record validator, artifact validator
│       └── feature_environment.py  # Runtime environment diagnostic engine and CLI
├── tests/
│   └── features/
│       ├── __init__.py
│       ├── test_feature_config.py  # 26 automated unit and integration tests
│       ├── test_feature_schema.py  # 25 automated unit and integration tests
│       └── test_feature_environment.py # 15 automated unit and integration tests
└── docs/
    └── FEATURE_ENGINEERING_SETUP.md # This specification document
```

---

## 4. Configuration Specification (`feature_config.yaml`)

Configuration is managed via `config/feature_config.yaml` and loaded into the immutable `FeatureConfig` dataclass:

```yaml
schema_version: "1.0"

input_artifacts:
  detection_results_path: "datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json"
  tracking_results_path: "datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json"
  dataset_metadata_path: "datasets/processed/driving_sample/metadata.csv"
  preprocessed_frames_dir: "datasets/processed/driving_sample/preprocessed_frames"

output:
  features_output_dir: "datasets/processed/driving_sample/features"
  save_json: true
  save_parquet: false
  save_csv: true
  float_precision: "float32"
  decimal_places: 4

feature_groups:
  detection_features: true
  motion_features: true
  visual_quality_features: true
  tracking_features: true

temporal_alignment:
  default_fps: 5.0
  max_sequence_gap_frames: 5
  missing_value_strategy: "explicit_null"
  sort_by: "frame_id"

execution_budget:
  batch_size: 100
  max_workers: 1
  memory_limit_mb: 500
```

### Operational Presets
`FeatureConfig.get_preset(preset_name)` provides pre-validated parameter sets:
1. **`default`:** Full feature groups enabled, float32 precision, batch size 100, 500 MB memory limit.
2. **`fast_cpu`:** Visual quality disabled (eliminates image re-reading), batch size 200, 250 MB memory limit.
3. **`high_precision`:** All groups enabled, 6 decimal places, float64 precision.
4. **`minimal`:** Detection and tracking groups only, batch size 50, 150 MB memory limit.

---

## 5. Feature Schema Registry (Schema v1.0)

The schema defines 25+ standardized fields categorized across 5 functional domains:

| Field Name | Type | Value Range | Required? | Group | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `frame_id` | `int` | $\ge 0$ | **Yes** | Metadata | Sequential zero-indexed frame identifier |
| `timestamp_sec` | `float` | $\ge 0.0$ | **Yes** | Metadata | Elapsed video timestamp in seconds |
| `track_id` | `int` | $\ge 0$ | **Yes** | Tracking | Unique persistent track identifier |
| `class_id` | `int` | $\ge 0$ | **Yes** | Detection | Class integer identifier |
| `class_name` | `str` | Non-empty | **Yes** | Detection | Standardized class label (e.g. `car`, `truck`) |
| `confidence` | `float` | $[0.0, 1.0]$ | **Yes** | Detection | Raw YOLO detection confidence |
| `bbox_x1` | `float` | $\ge 0.0$ | **Yes** | Detection | Normalized bounding box top-left X coordinate |
| `bbox_y1` | `float` | $\ge 0.0$ | **Yes** | Detection | Normalized bounding box top-left Y coordinate |
| `bbox_x2` | `float` | $> \text{bbox\_x1}$ | **Yes** | Detection | Normalized bounding box bottom-right X coordinate |
| `bbox_y2` | `float` | $> \text{bbox\_y1}$ | **Yes** | Detection | Normalized bounding box bottom-right Y coordinate |
| `bbox_width` | `float` | $> 0.0$ | **Yes** | Detection | Bounding box width |
| `bbox_height` | `float` | $> 0.0$ | **Yes** | Detection | Bounding box height |
| `bbox_area` | `float` | $> 0.0$ | **Yes** | Detection | Bounding box area |
| `bbox_aspect_ratio`| `float` | $> 0.0$ | **Yes** | Detection | Ratio of box width to height ($w/h$) |
| `velocity_x` | `float` | Real number | Optional | Motion | Horizontal centroid displacement per second |
| `velocity_y` | `float` | Real number | Optional | Motion | Vertical centroid displacement per second |
| `speed` | `float` | $\ge 0.0$ | Optional | Motion | Centroid Euclidean velocity magnitude |
| `acceleration` | `float` | Real number | Optional | Motion | Temporal derivative of Euclidean velocity |
| `displacement_ratio`| `float`| $\ge 0.0$ | Optional | Motion | Centroid displacement relative to box diagonal |
| `blur_score` | `float` | $\ge 0.0$ | Optional | Visual Quality | Laplacian variance sharpness measure |
| `brightness_score` | `float` | $[0.0, 255.0]$ | Optional | Visual Quality | Mean image luminance |
| `contrast_score` | `float` | $\ge 0.0$ | Optional | Visual Quality | Standard deviation of grayscale pixel intensities |
| `track_age` | `int` | $\ge 1$ | Optional | Tracking | Number of consecutive frames track has existed |
| `consecutive_hits` | `int` | $\ge 0$ | Optional | Tracking | Unbroken count of active detection associations |
| `consecutive_lost` | `int` | $\ge 0$ | Optional | Tracking | Count of recent frames where track was missed |
| `observation_ratio`| `float`| $[0.0, 1.0]$ | Optional | Tracking | Fraction of track lifespan with active detections |
| `tracking_state` | `str` | Enum | Optional | Tracking | Operational state (`tracked`, `lost`, `removed`) |

---

## 6. Multi-Artifact Integration & Validation Engine

The `validate_input_artifacts` engine inspects real upstream artifacts on disk before feature computation:

```python
from src.features.feature_schema import validate_input_artifacts

result = validate_input_artifacts(
    detection_file="datasets/processed/driving_sample/detections/batch_runs/run_.../detection_results.json",
    tracking_file="datasets/processed/driving_sample/tracking/multi_object/mot_.../multi_object_tracking.json",
    metadata_file="datasets/processed/driving_sample/metadata.csv",
    strict=True,
)
print("Artifacts Valid:", result.is_valid)
```

### Validation Rules
1. **File Existence & Non-Empty:** Checks file path resolution and non-zero byte size.
2. **JSON/CSV Schema Conformance:** Confirms root object schemas and required top-level keys (`schema_version`, `frames`).
3. **Monotonic Frame Ordering:** Validates strictly ascending `frame_id` sequences without gaps or duplicates.
4. **Dataset ID Consistency:** Ensures cross-artifact synchronization to the same underlying dataset (`driving_sample`).
5. **Frame Identifier Normalization:** Handles both numeric and formatted string identifiers (`frame_000001.jpg`, `frame_id: 1`).

---

## 7. Environment Verification Engine (`FeatureEnvironmentVerifier`)

The verifier executes diagnostic runtime checks:
- **Python Version:** Validates runtime $\ge 3.9$ (verified: `3.13.3`).
- **Hardware Audit:** Checks CPU logical and physical core counts, and system RAM availability.
- **Scientific Stack:** Verifies PyTorch (CPU-only), NumPy, OpenCV, SciPy, PyYAML, and Pandas.
- **Configuration Integrity:** Audits `feature_config.yaml` against schema constraints.
- **Artifact Availability:** Confirms presence of real detection, tracking, metadata, and image frames.

### CLI Usage

```powershell
# Plain-text diagnostic report
.\venv\Scripts\python.exe -m src.features.feature_environment

# Machine-readable JSON telemetry
.\venv\Scripts\python.exe -m src.features.feature_environment --json

# Validate custom configuration
.\venv\Scripts\python.exe -m src.features.feature_environment --config config/feature_config.yaml
```

Sample CLI output:
```
======================================================================
PERCEPTGUARD AI — FEATURE ENGINEERING ENVIRONMENT REPORT
======================================================================
  Environment Status         : [PASS]
----------------------------------------------------------------------
  1. Runtime & Host Platform :
     • Python Version        : 3.13.3
     • Operating System      : Windows 11 (AMD64)
     • CPU Cores (Log/Phys)  : 8 / 4
     • Host RAM (Avail/Total): 3.25 GB / 7.84 GB
----------------------------------------------------------------------
  2. Core Scientific & Vision Libraries :
     • PyTorch               : 2.15.0+cpu (CPU Allocation: True)
     • CUDA Available        : False
     • NumPy                 : 2.5.4
     • OpenCV                : 5.0.0
     • SciPy                 : 1.18.1
     • PyYAML                : 6.0.3
     • Pandas Available      : True
----------------------------------------------------------------------
  3. Feature Engineering Configuration :
     • Config Valid          : True
     • Config Path           : .../config/feature_config.yaml
----------------------------------------------------------------------
  4. Upstream Artifact Verification :
     • Detection Artifact    : FOUND
     • Tracking Artifact     : FOUND
     • Dataset Metadata      : FOUND
     • Preprocessed Frames   : FOUND
     • Artifact Integrity   : VALID
======================================================================
```

---

## 8. Verification & Quality Assurance

Automated unit and integration test suite (`tests/features/`):
- `test_feature_config.py`: 26 tests (immutability, validation rules, YAML serialization, operational presets, CLI).
- `test_feature_schema.py`: 25 tests (field definitions, record validation, geometry boundaries, multi-artifact validation).
- `test_feature_environment.py`: 15 tests (hardware profiling, dependency detection, missing library handling, CLI output).

**Test Results:** **66 / 66 Passed (100% Pass Rate)**
**Regression Suite:** Full repository suite passing with zero regressions across Days 1–4.
