# PerceptGuard AI: Temporal Feature Aggregation

**Status:** Completed  
**Milestone:** Day 5 – Task 5  
**Module:** `src/features/temporal_feature_aggregator.py`  
**Schema Version:** 1.0  
**Hardware Target:** Intel Core i3 / 8 GB RAM / CPU-only inference (No CUDA GPU dependency)  

---

## 1. Executive Summary

Temporal Feature Aggregation is the central synthesis engine of the **PerceptGuard AI** feature engineering pipeline. Its mission is to join, align, and temporally aggregate multi-modal perception streams—**Detection Features (Task 2)**, **Motion & Kinematic Features (Task 3)**, and **Visual Quality Features (Task 4)**—into unified temporal feature matrices across sequential video frames.

In safety-critical autonomous driving applications, isolated instantaneous detections are noisy and prone to intermittent dropouts or false positives. By calculating multi-horizon rolling statistics (means, dynamic ranges, rates of change, and linear trend slopes) over past video frames, PerceptGuard AI provides the downstream recurrent forecasting models (LSTM/GRU) with comprehensive contextual trajectories of perceptual reliability.

```
+-----------------------------------------------------------------------------------+
|               PerceptGuard AI Multi-Source Temporal Feature Aggregator            |
+-----------------------------------------------------------------------------------+
        |
        |---> Detection Features (Task 2): 10 Frame Metrics (Counts, Confidences, BBoxes)
        |---> Motion Features (Task 3):    9 Frame Metrics (Tracks, Displacements, Speeds)
        |---> Visual Features (Task 4):    8 Frame Metrics (Blur, Brightness, Contrasts)
        |
        v
+-----------------------------------------------------------------------------------+
| Multi-Source Alignment & Invariant Verification                                  |
| - Verify dataset_id identity across all sources                                    |
| - Align on stable frame_id primary keys                                            |
| - Preserve strict chronological sequence order (source_frame_index, timestamps)     |
| - Validate spatial dimensions (image_width, image_height)                          |
+-----------------------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------------------+
| Frame-Level Feature Fusion (27 Base Numerical Metrics + Coverage Telemetry)       |
+-----------------------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------------------+
| Trailing Past-Looking Window Aggregation (e.g., w3, w5, w10)                      |
| - STRICT: Zero Future-Frame Leakage Guaranteed (k in [max(0, i - W + 1), i])      |
| - Statistics: Mean, Min, Max, Std, Median, First, Last, Diff                     |
| - Explicit Missing Policy: No Silent Zero Infill; Null-Aware Observation Tracking  |
| - Ordinary Least Squares (OLS) Linear Trend Slopes for Key Reliability Indicators |
+-----------------------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------------------+
| Versioned Schema v1.0 Atomic Output Artifacts                                     |
| 1. temporal_features.json          (Full nested schema)                           |
| 2. aggregated_frame_features.json  (Flat sequence/tabular format)                 |
| 3. temporal_feature_summary.json   (Global alignment & distribution metrics)       |
| 4. feature_metadata.json           (Hardware environment, schema & registry)      |
| 5. feature_errors.json             (Comprehensive anomaly & conflict audit)       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Multi-Source Alignment & Data Fusion

### 2.1 Multi-Modal Input Ingestion

The aggregator ingests three independent upstream feature groups:

1. **Detection Features (Task 2):**
   `detection_count`, `unique_class_count`, `mean_confidence`, `minimum_confidence`, `maximum_confidence`, `confidence_std`, `total_bbox_area`, `mean_bbox_area`, `mean_bbox_width`, `mean_bbox_height`.
2. **Motion Features (Task 3):**
   `observed_track_count`, `moving_track_count`, `stationary_track_count`, `mean_displacement`, `maximum_displacement`, `mean_speed`, `mean_velocity_x`, `mean_velocity_y`, `motion_feature_coverage`.
3. **Visual Quality Features (Task 4):**
   `blur_score`, `mean_brightness`, `median_brightness`, `brightness_std`, `contrast_std`, `contrast_range`, `contrast_percentile_range`, `image_aspect_ratio`.

### 2.2 Alignment & Verification Invariants

- **Primary Join Key:** Aligned on canonical string `frame_id` (e.g. `frame_000001.jpg`).
- **Dataset Consistency:** Evaluates `dataset_id` across all provided streams. Conflicting identifiers raise `DatasetIdMismatchError`.
- **Chronological Ordering:** Frames are sorted primarily by `source_frame_index` (with fallback to natural alphanumeric sorting). Non-monotonic timestamps are flagged.
- **Duplicate Frame Detection:** Duplicate occurrences of any `frame_id` within a single source artifact raise `DuplicateFrameIdError`.
- **Spatial Dimension Agreement:** Resolves `(image_width, image_height)`; discrepancies across sources for the same frame are logged to `feature_errors.json`.
- **Source Availability & Coverage Tracking:**
  $$\text{source\_coverage} = \frac{\sum [\text{detection}, \text{motion}, \text{visual}]}{3}$$
  $$\text{numeric\_feature\_coverage} = \frac{\text{valid\_numeric\_metrics}}{27}$$

---

## 3. Temporal Window Aggregation Architecture

### 3.1 Strict Future-Frame Leakage Prevention

In predictive reliability forecasting, using future frames ($t + \tau$) to construct feature representations at time $t$ introduces target leakage, producing overly optimistic offline evaluations that fail catastrophically in real-time deployment.

To prevent future leakage, the aggregator enforces **trailing / past-looking window alignment**:

$$\mathcal{W}_W(i) = \{j \in \mathbb{N} \mid \max(0, i - W + 1) \le j \le i\}$$

For any frame at chronological index $i$ and window length $W$:
- Frames $j > i$ are strictly excluded from the evaluation slice.
- For early sequence frames ($i < W - 1$), the window dynamically scales to the available history without artificial padding, recording `contributing_frame_count` $\le W$ and true observation coverage.

### 3.2 Aggregate Statistics Formulations

For each numerical feature $Y = [y_1, y_2, \dots, y_k]$ of valid (non-null) observations within the trailing window:

1. **Arithmetic Mean:**
   $$\bar{y} = \frac{1}{k} \sum_{j=1}^k y_j$$
2. **Extrema:**
   $$\min(Y) = \min_{j} y_j, \quad \max(Y) = \max_{j} y_j$$
3. **Population Standard Deviation:**
   $$\sigma = \sqrt{\frac{1}{k} \sum_{j=1}^k (y_j - \bar{y})^2} \quad (\sigma = 0.0 \text{ if } k \le 1)$$
4. **Median:**
   $$\text{Median}(Y) = P_{50}(Y)$$
5. **Boundary Evolution:**
   $$\text{first} = y_1, \quad \text{last} = y_k, \quad \text{diff} = y_k - y_1$$
6. **Observation Tracking:**
   $$\text{valid\_count} = k, \quad \text{missing\_count} = W_{\text{size}} - k, \quad \text{coverage\_ratio} = \frac{k}{W_{\text{size}}}$$

If $k = 0$ (no valid observations in window), all statistical metrics evaluate to `null`. **Missing values are never fabricated or treated as zero.**

---

## 4. Temporal Trend Modeling (Ordinary Least Squares)

For key reliability-indicative signals, the aggregator computes the rate of temporal change (trend slope) over each trailing window using Ordinary Least Squares (OLS) regression against sequence time step $x_j \in [0, 1, \dots, k-1]$:

$$\text{slope} = \frac{\sum_{j=0}^{k-1} (x_j - \bar{x})(y_j - \bar{y})}{\sum_{j=0}^{k-1} (x_j - \bar{x})^2}$$

### 4.1 Tracked Trend Metrics
- `confidence_trend`: Trend slope of `mean_confidence`.
- `blur_trend`: Trend slope of `blur_score`.
- `brightness_trend`: Trend slope of `mean_brightness`.
- `contrast_trend`: Trend slope of `contrast_std`.
- `displacement_trend`: Trend slope of `mean_displacement`.
- `speed_trend`: Trend slope of `mean_speed`.

### 4.2 Status Categorization
- `valid`: Calculated slope where $k \ge \tau_{\text{min}}$ (default $\ge 2$).
  - `direction`: `"increasing"` ($\text{slope} > 10^{-4}$), `"decreasing"` ($\text{slope} < -10^{-4}$), or `"flat"` ($|\text{slope}| \le 10^{-4}$).
- `constant`: All observations identical ($\max(y) - \min(y) < 10^{-9}$), $\text{slope} = 0.0$.
- `insufficient_data`: Fewer than 2 valid observations in the window ($k < 2$).
- `no_data`: Zero valid observations in the window ($k = 0$).

---

## 5. Versioned Artifact Specifications (Schema v1.0)

Output directory: `datasets/processed/<dataset_id>/features/temporal/<run_id>/`

### 5.1 `temporal_features.json`
Complete nested structure containing full frame metadata, fused base features, window statistics, and linear trends:
```json
{
  "schema_version": "1.0",
  "feature_type": "temporal_aggregated",
  "dataset_id": "driving_sample",
  "run_id": "temporal_driving_sample_20261118_151500_e128ab",
  "total_frames": 200,
  "features": [
    {
      "schema_version": "1.0",
      "dataset_id": "driving_sample",
      "frame_id": "frame_000005.jpg",
      "source_frame_index": 4,
      "timestamp_seconds": 0.7976,
      "image_width": 640,
      "image_height": 360,
      "source_availability": {
        "detection": true,
        "motion": true,
        "visual_quality": true
      },
      "feature_coverage": {
        "source_coverage": 1.0,
        "total_sources_available": 3,
        "valid_numeric_feature_count": 27,
        "total_numeric_feature_count": 27,
        "numeric_feature_coverage": 1.0
      },
      "detection_features": {
        "detection_count": 2,
        "mean_confidence": 0.4431,
        "total_bbox_area": 48210.5
      },
      "motion_features": {
        "observed_track_count": 2,
        "mean_displacement": 0.4432,
        "mean_speed": 1.5588
      },
      "visual_quality_features": {
        "blur_score": 451.2031,
        "mean_brightness": 32.1402,
        "contrast_std": 58.2104
      },
      "temporal_windows": {
        "w5": {
          "window_size": 5,
          "window_alignment": "trailing",
          "window_start_frame": "frame_000001.jpg",
          "window_end_frame": "frame_000005.jpg",
          "contributing_frame_count": 5,
          "feature_coverage": 0.9852,
          "feature_statistics": {
            "blur_score": {
              "mean": 454.1203,
              "min": 400.19,
              "max": 512.4,
              "std": 41.201,
              "median": 451.2031,
              "first": 400.19,
              "last": 451.2031,
              "diff": 51.0131,
              "valid_count": 5,
              "missing_count": 0,
              "coverage_ratio": 1.0
            }
          },
          "trends": {
            "confidence_trend": {
              "slope": 0.0215,
              "status": "valid",
              "direction": "increasing",
              "method": "ordinary_least_squares",
              "observation_count": 5
            }
          }
        }
      },
      "processing_status": "success"
    }
  ]
}
```

### 5.2 `aggregated_frame_features.json`
Flattened sequence-ready representation designed for matrix transformation and PyTorch / LSTM sequence modeling. Contains flat column keys such as `det_detection_count`, `mot_mean_displacement`, `vq_blur_score`, `w5_blur_score_mean`, `w5_confidence_trend_slope`.

### 5.3 `temporal_feature_summary.json`
Dataset-level alignment rates, coverage summaries, and execution telemetry.

### 5.4 `feature_metadata.json`
Hardware environment, library versions, parameter registry, and feature documentation.

### 5.5 `feature_errors.json`
Audit log recording any dimensional mismatches, conflicting metadata, or missing frames.

---

## 6. Command-Line Interface (CLI)

```bash
# Execute temporal feature aggregation across detection, motion, and visual quality
python -m src.features.temporal_feature_aggregator \
  --detection datasets/processed/driving_sample/features/detection/det_features_driving_sample_20261118_141511_9e022a/frame_detection_features.json \
  --motion datasets/processed/driving_sample/features/motion/motion_features_driving_sample_20261118_143109_bb4a56/frame_motion_features.json \
  --visual datasets/processed/driving_sample/features/visual_quality/run_official/visual_quality_features.json \
  --output datasets/processed/driving_sample/features/temporal/run_official \
  --window-sizes 3,5,10 \
  --overwrite \
  --verbose
```

### Available CLI Flags:
- `--detection, -d`: Path to detection features JSON artifact or run directory.
- `--motion, -m`: Path to motion features JSON artifact or run directory.
- `--visual, -v`: Path to visual quality features JSON artifact or run directory.
- `--config, -c`: Path to `config/feature_config.yaml`.
- `--output, -o`: Custom output destination directory.
- `--window-sizes, -w`: Comma-separated temporal window lengths (default: `3,5,10`).
- `--max-frames`: Cap on total chronological frames to process.
- `--start-frame`: Start frame index filter (inclusive).
- `--end-frame`: End frame index filter (inclusive).
- `--allow-partial`: Permit processing when 1 or 2 sources are missing.
- `--overwrite`: Overwrite destination directory if non-empty.
- `--verbose`: Enable debug logging and stack traces.
