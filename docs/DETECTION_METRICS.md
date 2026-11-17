# PerceptGuard AI – Advanced YOLO Detection Metrics & Perception Analysis

## 1. Executive Overview & Technical Scope

This document specifies the architecture, mathematical formulations, data contracts, and operational CLI for the **Advanced YOLO Detection Metrics & Perception Analysis Engine** (`src/analysis/detection_metrics.py`) in **PerceptGuard AI**.

The metrics engine evaluates YOLO batch detection outputs from Task 4 (`detection_results.json`), calculating detection volumes, confidence probability distributions, semantic class frequencies, sequence temporal consistency, and exploratory heuristic reliability indicators.

### Key Architectural Guarantees:
1. **Decoupled Evaluation:** Operates purely on serialized detection records. Strictly zero model weights, zero PyTorch deserialization, and zero forward inference passes occur during analysis.
2. **Honest Ground-Truth Accounting:** Does not fabricate precision, recall, F1-score, or mAP when ground-truth labels are absent. Accurately marks ground-truth evaluation as `"unavailable"` and documents the annotation requirement.
3. **Parametric Confidence Profiling:** Computes mean, median, standard deviation, and standardized histogram distributions across 5 intervals, separating softmax confidence from localization accuracy.
4. **Sequence Temporal Consistency:** Quantifies frame-to-frame delta variation ($|\Delta N_t|$) and volatility across chronological frames without assuming object identity tracking.
5. **Transparent Heuristic Reliability Indicator:** Formulates an interpretable, normalized frame-level signal ($R_{\text{heuristic}} \in [0.0, 1.0]$) combining detection volume stability and confidence health.
6. **Multi-File Versioned Output Architecture (Schema v1.0):** Emits `detection_metrics.json`, `class_metrics.json`, `frame_metrics.json`, `analysis_summary.json`, and `analysis_metadata.json`.

---

## 2. System Architecture & Evaluation Pipeline

```
┌────────────────────────────────────────┐
│  Task 4 detection_results.json (v1.0)  │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│      JSON Schema & Data Validator      │
│ - Validates schema_version == "1.0"    │
│ - Verifies required keys & frame arrays│
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│       Core Analytical Aggregators      │
│ 1. Detection Volume & Frame Completeness│
│ 2. Confidence Statistical Distribution │
│ 3. Class-Wise Frequencies & Ranges     │
│ 4. Frame Telemetry & Processing Speeds │
│ 5. Temporal Consistency & Volatility   │
│ 6. Heuristic Reliability Scorer        │
│ 7. Ground-Truth IoU Bipartite Matching │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Atomic Multi-File Persistence                       │
│ datasets/processed/<dataset>/analysis/<run_id>/                        │
│   ├── detection_metrics.json                                           │
│   ├── class_metrics.json                                               │
│   ├── frame_metrics.json                                               │
│   ├── analysis_summary.json                                            │
│   └── analysis_metadata.json                                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations & Metric Definitions

### 3.1 Detection Volume Statistics
- **Total Frames Processed ($N_{\text{total}}$):** Number of evaluated frames in sequence.
- **Successful Frames ($N_{\text{success}}$):** Frames where `processing_status == "success"`.
- **Frames With Detections ($N_{\text{det}}$):** Successful frames with detection count $\ge 1$.
- **Frames Without Detections ($N_{\text{zero}}$):** Successful frames with detection count $= 0$.
- **Average Detections Per Frame:**
  $$\bar{D} = \frac{\sum_{t=1}^{N_{\text{success}}} D_t}{N_{\text{success}}}$$

### 3.2 Confidence Analysis
Softmax confidence values $c_i \in (0, 1]$ are analyzed across all $M$ detections:
- **Mean Confidence:** $\mu_c = \frac{1}{M} \sum_{i=1}^M c_i$
- **Median Confidence:** $\tilde{c} = \text{median}(c_1, \dots, c_M)$
- **Confidence Standard Deviation:**
  $$\sigma_c = \sqrt{\frac{1}{M-1} \sum_{i=1}^M (c_i - \mu_c)^2}$$
- **Low-Confidence Percentage:** Share of detections where $c_i < \tau_{\text{conf}}$ (default $\tau_{\text{conf}} = 0.30$):
  $$P_{\text{low}} = \frac{|\{c_i \mid c_i < \tau_{\text{conf}}\}|}{M} \times 100\%$$
- **Distribution Intervals:** Standard 5-bin histogram: $[0.0, 0.2)$, $[0.2, 0.4)$, $[0.4, 0.6)$, $[0.6, 0.8)$, $[0.8, 1.0]$.

### 3.3 Temporal Consistency Analysis
Evaluated across chronologically ordered video frames $t = 1 \dots T$:
- **Frame-to-Frame Detection Count Variation:**
  $$|\Delta D_t| = |D_t - D_{t-1}|$$
- **Mean Detection Count Variation:**
  $$\overline{\Delta D} = \frac{1}{T-1} \sum_{t=2}^T |\Delta D_t|$$
- **Temporal Stability Score ($S_{\text{temporal}} \in [0.0, 1.0]$):**
  $$S_{\text{temporal}} = \max\left(0.0, \min\left(1.0, 1.0 - \frac{\overline{\Delta D}}{\max(1.0, \bar{D} + 1.0)}\right)\right)$$
- **Documentation Note:** These metrics measure sequence-level frequency volatility. They do not claim object-level trajectory consistency, as unique tracking IDs are introduced in multi-object tracking (ByteTrack).

### 3.4 Heuristic Detection Reliability Indicator
An exploratory frame-level heuristic reliability signal $R_{\text{heuristic}} \in [0.0, 1.0]$:
$$R_{\text{heuristic}} = \begin{cases}
0.0, & \text{if } \text{status} = \text{"failed"} \\
0.85, & \text{if } D_t = 0 \text{ and status} = \text{"success"} \\
\max\left(0.05, \min\left(1.0, \mu_{c, t} - \text{penalty}_{\text{low}} - \text{penalty}_{\Delta}\right)\right), & \text{if } D_t > 0
\end{cases}$$
where:
- $\text{penalty}_{\text{low}} = \frac{|\{c_i < \tau_{\text{conf}}\}|}{D_t} \times 0.25$
- $\text{penalty}_{\Delta} = \min(0.20, |\Delta D_t| \times 0.05)$
- **Strict Scientific Disclaimer:** This indicator is a heuristic proxy reflecting detector confidence and temporal stability. It must **never** be cited as true ground-truth perception accuracy.

---

## 4. Ground-Truth Evaluation Protocol

When a ground-truth JSON annotation file is provided via `--ground-truth <path>`:
1. **Bounding Box IoU:**
   $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$
2. **Greedy Bipartite Matching:**
   Predictions per frame are sorted by confidence descending. For each prediction, the available ground-truth box of matching class with maximum $\text{IoU} \ge \tau_{\text{IoU}}$ (default 0.50) is matched as a True Positive (TP).
3. **Metrics Calculation:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Current Ground-Truth Status on `driving_sample`:
The current `driving_sample` dataset is an unannotated monocular driving sequence. Consequently:
- Ground-truth evaluation status is marked **`"unavailable"`**.
- Precision, recall, and F1 are set to `null` with explicit disclaimer.
- Ground-truth evaluation functions are fully implemented and verified via automated test suites.

---

## 5. Output Directory & Versioned Artifacts

All evaluation runs are saved under:
`datasets/processed/<dataset>/analysis/<run_id>/`

```
datasets/processed/driving_sample/analysis/metrics_20261117_180149_789f26/
├── detection_metrics.json    # Aggregated volume, confidence, temporal, and heuristic metrics
├── class_metrics.json        # Class-wise breakdowns and frequencies
├── frame_metrics.json        # Frame-level telemetry and temporal deltas
├── analysis_summary.json     # High-level operational summary
└── analysis_metadata.json    # Environment, parameter settings, and metric definitions
```

---

## 6. Command Line Interface (CLI)

```powershell
# Evaluate actual Task 4 detection output
python -m src.analysis.detection_metrics --input datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json --overwrite

# Custom confidence and IoU thresholds
python -m src.analysis.detection_metrics --input <path> --confidence-threshold 0.35 --iou-threshold 0.60

# Evaluate with ground-truth annotations
python -m src.analysis.detection_metrics --input <path> --ground-truth path/to/annotations.json

# Partial smoke analysis
python -m src.analysis.detection_metrics --input <path> --max-frames 10 --verbose
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *(Required)* | Path to `detection_results.json` generated by Task 4. |
| `--output-dir` | `str` | `None` | Custom destination directory for analysis artifacts. |
| `--confidence-threshold`| `float`| `0.30` | Cutoff for low-confidence categorization. |
| `--iou-threshold` | `float`| `0.50` | Minimum IoU overlap for ground-truth matching. |
| `--ground-truth` | `str` | `None` | Path to ground-truth annotations JSON. |
| `--max-frames` | `int` | `None` | Frame cap for smoke analysis (marks status `WARNING`). |
| `--overwrite` | `flag` | `False` | Overwrite destination directory if it already exists. |
| `--verbose` | `flag` | `False` | Enable DEBUG diagnostic logging. |

---

## 7. Automated Test Suite (30 Tests)

The engine is verified by [tests/analysis/test_detection_metrics.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/analysis/test_detection_metrics.py) and [tests/test_detection_metrics.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/test_detection_metrics.py):

| Test ID | Test Method | Scope |
| :---: | :--- | :--- |
| `01` | `test_01_valid_detection_input` | Valid detection JSON parsing. |
| `02` | `test_02_empty_detection_input` | Empty detection list handling. |
| `03` | `test_03_zero_detection_frames` | Accurate accounting of zero-detection frames. |
| `04` | `test_04_confidence_statistics` | Mean, median, std dev, and low-conf percentage. |
| `05` | `test_05_class_wise_statistics` | Class frequency and confidence aggregation. |
| `06` | `test_06_frame_wise_statistics` | Frame-level telemetry and field preservation. |
| `07` | `test_07_temporal_variation` | Delta counts and stability scoring. |
| `08` | `test_08_missing_timestamps` | Graceful handling of missing timestamps. |
| `09` | `test_09_invalid_confidence_values`| Filtering of out-of-bounds confidences. |
| `10` | `test_10_invalid_bounding_boxes` | Safe handling of malformed boxes in IoU. |
| `11` | `test_11_missing_required_fields` | Rejection of missing top-level schema keys. |
| `12` | `test_12_duplicate_frame_ids` | Deterministic handling of duplicate IDs. |
| `13` | `test_13_failed_frames` | Proper accounting of failed frames. |
| `14` | `test_14_missing_ground_truth` | Status reported as 'unavailable' with disclaimer. |
| `15` | `test_15_valid_ground_truth_matching`| Greedy bipartite matching against ground truth. |
| `16` | `test_16_iou_calculation` | Mathematical accuracy of box IoU. |
| `17` | `test_17_precision_and_recall` | Precision and recall edge cases. |
| `18` | `test_18_f1_score` | Harmonic mean calculation for F1. |
| `19` | `test_19_output_json_validation` | Validation of all 5 output JSON files. |
| `20` | `test_20_schema_versioning` | Rejection of unsupported schema versions. |
| `21` | `test_21_cli_execution` | CLI runner entrypoint execution. |
| `22` | `test_22_partial_analysis` | Sliced runs assign status `WARNING`. |
| `23` | `test_23_error_reporting` | CLI error propagation and code 1 on failure. |
| `24` | `test_24_deterministic_output` | Identical inputs produce identical outputs. |
| `25` | `test_25_empty_dataset` | Zero-handling on empty dataset. |
| `26` | `test_26_confidence_threshold_behavior`| Sensitivity to varying confidence thresholds. |
| `27` | `test_27_class_filtering` | Class separation across distinct COCO classes. |
| `28` | `test_28_invalid_json_syntax` | Syntax-corrupted JSON rejection. |
| `29` | `test_29_unsupported_schema` | Rejection of incompatible schema numbers. |
| `30` | `test_30_real_task4_artifact_integration` | End-to-end integration test on real Task 4 data. |

---

## 8. Real-Data Benchmark Telemetry on `driving_sample`

Evaluation performed against actual Task 4 detection output (`run_20261117_115348_e1789b`):

| Evaluation Metric | Measured Benchmark Value |
| :--- | :--- |
| **Dataset Evaluated** | `driving_sample` |
| **Detection Source Run** | `run_20261117_115348_e1789b` |
| **Total Frames Processed** | **200 frames** |
| **Successful Frames** | **200 frames (100.0%)** |
| **Failed Frames** | **0 frames (0.0%)** |
| **Frames With Detections** | 166 frames (83.0%) |
| **Frames Without Detections** | 34 frames (17.0%) |
| **Total Objects Detected** | **211 objects** |
| **Average Objects / Frame** | **1.05 objects** (Min: 0, Max: 4) |
| **Mean Softmax Confidence** | **0.4215** |
| **Median Softmax Confidence** | **0.4195** |
| **Confidence Standard Deviation** | **0.1045** |
| **Confidence Range** | **[0.2522, 0.7449]** |
| **Low-Confidence Detections (< 0.30)** | **33 detections (15.6%)** |
| **Temporal Stability Score** | **0.8166** |
| **Mean Heuristic Reliability** | **0.4619** |
| **Ground-Truth Evaluation Status** | **`unavailable`** (No labels provided in dataset) |
| **Final Evaluation Status** | **`PASS`** |

### Semantic Class Breakdown
- `traffic light`: 159 detections (75.36%, mean conf: 0.4489, bounds: [0.2596, 0.7449])
- `truck`: 31 detections (14.69%, mean conf: 0.3289, bounds: [0.2537, 0.5763])
- `car`: 7 detections (3.32%, mean conf: 0.3359, bounds: [0.2522, 0.5132])
- `potted plant`: 6 detections (2.84%, mean conf: 0.3541)
- `bus`: 3 detections (1.42%, mean conf: 0.3364)
- `surfboard`: 2 detections (0.95%, mean conf: 0.4361)
- `person`: 1 detection (0.47%, mean conf: 0.3705)
- `train`: 1 detection (0.47%, mean conf: 0.3787)
- `vase`: 1 detection (0.47%, mean conf: 0.3392)

