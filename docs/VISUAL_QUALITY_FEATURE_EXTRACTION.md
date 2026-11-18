# PerceptGuard AI: Visual Quality Feature Extraction

**Status:** Completed  
**Milestone:** Day 5 – Task 4  
**Module:** `src/features/visual_quality_extractor.py`  
**Schema Version:** 1.0  
**Hardware Target:** Intel Core i3 / 8 GB RAM / CPU-only inference (No CUDA GPU dependency)  

---

## 1. Executive Summary

Visual Quality Feature Extraction constitutes the fourth phase of the **PerceptGuard AI** Day 5 Feature Engineering pipeline. Its mission is to extract objective optical and image quality features directly from raw image pixel buffers without re-running compute-heavy YOLO object detection or ByteTrack tracking.

In autonomous driving systems, sensor degradation (motion blur, lens flare, adverse night lighting, low-contrast fog/rain) directly impacts downstream object detector reliability. By isolating optical image characteristics into standardized numerical representations, PerceptGuard AI provides the predictive LSTM/GRU models with direct representations of perceptual signal-to-noise ratio.

```
+-----------------------------------------------------------------------------+
|               PerceptGuard AI Visual Quality Feature Pipeline               |
+-----------------------------------------------------------------------------+
       |
       v
 [ Frame Images (BGR/Gray/BGRA) ] ---> [ Dimensions & Readability Validation ]
       |                                                    |
       +--------------------+-------------------------------+
                            |
                            v
       [ Color Conversion & Buffer Management (del img) ]
                            |
       +--------------------+-------------------------------+
       |                    |                               |
       v                    v                               v
[ Blur Score ]     [ Brightness Score ]            [ Contrast Score ]
- Laplacian Var     - Mean / Median / Min / Max     - RMS Std Deviation
- Heuristic Tier    - Brightness Std                - Dynamic Range (Max - Min)
                    - Heuristic Tier                - Percentile Range (P95 - P5)
                                                    - Normalized Contrast
                                                    - Heuristic Tier
       |                    |                               |
       +--------------------+-------------------------------+
                            |
                            v
   [ Geometry & Channels: Width, Height, Aspect Ratio, BGR Means ]
                            |
                            v
           [ Schema v1.0 Atomic Output Artifacts ]
           1. visual_quality_features.json
           2. visual_quality_summary.json
           3. visual_quality_metadata.json
           4. visual_quality_errors.json
```

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Optical Blur Measurement (Laplacian Variance)

The variance of the Laplacian operator is the standard computer vision metric for estimating optical edge sharpness. Rapid spatial intensity transitions produce large second-order derivative magnitudes, while blur suppresses high spatial frequencies.

Given a grayscale image $I(x, y)$, the discrete Laplacian is defined by convolving $I$ with the standard second-derivative kernel:

$$\nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

In OpenCV, this is computed using `cv2.Laplacian(gray, cv2.CV_64F)`. The blur score is the statistical population variance across all $N$ pixels:

$$\bar{L} = \frac{1}{N} \sum_{x, y} \nabla^2 I(x, y)$$

$$\text{blur\_score} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x, y} (\nabla^2 I(x, y) - \bar{L})^2$$

- **Numeric Type:** `cv2.CV_64F` prevents 8-bit integer truncation and signed integer overflow.
- **Physical Interpretation:** Higher values indicate sharp high-frequency edges; values approaching zero indicate severe optical defocus, motion blur, or uniform regions.

### 2.2 Grayscale Brightness Statistics

Pixel intensities are extracted across the grayscale representation $I(x, y) \in [0, 255]$:

1. **Mean Brightness:**
   $$\mu_I = \frac{1}{N} \sum_{x, y} I(x, y)$$
2. **Median Brightness:**
   $$\text{Median}(I) = P_{50}(I)$$
3. **Brightness Standard Deviation:**
   $$\sigma_I = \sqrt{\frac{1}{N} \sum_{x, y} (I(x, y) - \mu_I)^2}$$
4. **Minimum / Maximum:**
   $$\min(I), \quad \max(I)$$

### 2.3 Contrast Metrics

Contrast measures the distribution and dynamic range of luminance across the frame:

1. **RMS Contrast (`contrast_std`):**
   Root-mean-square contrast is defined as the standard deviation of pixel intensities:
   $$\text{RMS Contrast} = \sigma_I = \sqrt{\frac{1}{N} \sum_{x, y} (I(x, y) - \mu_I)^2}$$
2. **Dynamic Range (`contrast_range`):**
   $$\Delta I = \max(I) - \min(I)$$
3. **Robust Percentile Range (`contrast_percentile_range`):**
   To resist single hot or dead sensor pixels:
   $$\Delta_{90} = P_{95}(I) - P_{5}(I)$$
4. **Normalized Contrast (`normalized_contrast`):**
   Coefficient of variation with numerical epsilon ($\epsilon = 10^{-6}$):
   $$\hat{C} = \frac{\sigma_I}{\mu_I + \epsilon}$$

### 2.4 Channel Metrics & Geometry

- **Image Geometry:**
  $$\text{aspect\_ratio} = \frac{\text{width}}{\text{height}}, \quad \text{pixel\_count} = \text{width} \times \text{height}$$
- **Independent Color Channel Means (BGR):**
  $$\mu_B = \frac{1}{N} \sum B(x, y), \quad \mu_G = \frac{1}{N} \sum G(x, y), \quad \mu_R = \frac{1}{N} \sum R(x, y)$$
  *(For 1-channel grayscale inputs, `color_channel_means` evaluates to `null` to avoid fabricating artificial color data).*

---

## 3. Heuristic Classification Thresholds

> [!NOTE]
> Heuristic quality categories are rule-based operational tags for screening and exploratory analytics. They do **not** represent ground-truth reliability labels or detector accuracy predictions.

Thresholds are fully configurable via `FeatureConfig` or custom dictionary overrides:

| Category | Metric | Condition | Default Cutoff | Classification Tag |
| :--- | :--- | :--- | :--- | :--- |
| **Blur** | `blur_score` | $\text{score} > \tau_{\text{sharp}}$ | $> 300.0$ | `sharp` |
| | | $\tau_{\text{blurry}} \le \text{score} \le \tau_{\text{sharp}}$ | $100.0 \le \text{score} \le 300.0$ | `moderate` |
| | | $\text{score} < \tau_{\text{blurry}}$ | $< 100.0$ | `blurry` |
| **Brightness** | `mean_brightness` | $\mu_I < \tau_{\text{dark}}$ | $< 60.0$ | `dark` |
| | | $\tau_{\text{dark}} \le \mu_I \le \tau_{\text{bright}}$ | $60.0 \le \mu_I \le 180.0$ | `normal` |
| | | $\mu_I > \tau_{\text{bright}}$ | $> 180.0$ | `bright` |
| **Contrast** | `contrast_std` | $\sigma_I < \tau_{\text{low}}$ | $< 30.0$ | `low` |
| | | $\tau_{\text{low}} \le \sigma_I \le \tau_{\text{high}}$ | $30.0 \le \sigma_I \le 70.0$ | `normal` |
| | | $\sigma_I > \tau_{\text{high}}$ | $> 70.0$ | `high` |

---

## 4. Software Architecture & Invariant Guarantees

### 4.1 Strict Input Validation & Fault Tolerance

The `VisualQualityExtractor` incorporates multi-stage defensive validation:
1. **File Existence & Non-Zero Byte Verification:** Detects 0-byte placeholders (`EmptyImageFileError`) and missing files (`FileNotFoundExtractionError`).
2. **Format Verification:** Validates extensions against whitelist (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
3. **OpenCV Memory Decode:** Rejects corrupt headers or truncated byte streams (`CorruptedImageFileError`).
4. **Dimension & Channel Invariants:** Rejects $0 \times 0$, negative, or empty numpy arrays (`InvalidImageDimensionError`).
5. **Fail-Soft Batch Operation:** In directory scanning mode, corrupted frames do not terminate execution. Failures are captured in `visual_quality_errors.json`, preserving throughput.

### 4.2 Memory & CPU Optimization (Edge / Intel Core i3)

- **Explicit Memory Deallocation:** Heavy image arrays (`img`, `gray`) are explicitly deleted (`del img`) after computing scalars, preventing garbage-collection backpressure.
- **Single Pass Statistics:** Brightness mean, median, min, max, and std are derived in a unified vector operation.
- **Peak RAM Footprint:** Ingestion and extraction on 200 real $640 \times 360$ frames consumes $< 25$ MB RSS delta.
- **Throughput:** ~16 ms per frame (~62 FPS) on single-thread CPU execution.

---

## 5. Artifact Output Schema (Version 1.0)

Every extraction run produces four atomic JSON artifacts in `datasets/processed/<dataset_id>/features/visual_quality/<run_id>/`:

### 5.1 `visual_quality_features.json`
Contains frame-level visual quality feature records:
```json
{
  "schema_version": "1.0",
  "feature_type": "visual_quality",
  "dataset_id": "driving_sample",
  "run_id": "vq_driving_sample_20261118_143801_a9f1b2",
  "generated_at": "2026-11-18T14:38:05.123456+00:00",
  "total_frames": 200,
  "features": [
    {
      "dataset_id": "driving_sample",
      "frame_id": "frame_000001.jpg",
      "frame_filename": "frame_000001.jpg",
      "image_path": "C:/Users/.../frame_000001.jpg",
      "extracted_frame_id": "frame_000001",
      "source_frame_index": 0,
      "timestamp_seconds": 0.0,
      "image_width": 640,
      "image_height": 360,
      "image_aspect_ratio": 1.7778,
      "image_pixel_count": 230400,
      "channels": 3,
      "blur_score": 444.2051,
      "blur_category": "sharp",
      "mean_brightness": 32.8565,
      "median_brightness": 0.0,
      "brightness_std": 58.1032,
      "minimum_brightness": 0.0,
      "maximum_brightness": 255.0,
      "brightness_category": "dark",
      "contrast_std": 58.1032,
      "contrast_range": 255.0,
      "contrast_percentile_range": 146.0,
      "normalized_contrast": 1.7684,
      "contrast_category": "normal",
      "grayscale_mean": 32.8565,
      "grayscale_std": 58.1032,
      "color_channel_means": {
        "b": 32.1411,
        "g": 32.613,
        "r": 33.6202
      },
      "processing_status": "success",
      "error_message": null
    }
  ]
}
```

### 5.2 `visual_quality_summary.json`
High-level distribution and statistical telemetry across all evaluated frames:
```json
{
  "schema_version": "1.0",
  "run_id": "vq_driving_sample_20261118_143801_a9f1b2",
  "dataset_id": "driving_sample",
  "source_image_dir": "datasets/processed/driving_sample/preprocessed_frames",
  "metadata_source": "datasets/processed/driving_sample/metadata.csv",
  "extracted_at": "2026-11-18T14:38:05.123456+00:00",
  "is_partial": false,
  "processed_frame_count": 200,
  "valid_frame_count": 200,
  "failed_frame_count": 0,
  "extraction_duration_seconds": 3.2412,
  "validation_status": "PASS",
  "blur_statistics": {
    "mean": 499.9142,
    "median": 472.105,
    "std": 124.5123,
    "min": 198.4012,
    "max": 892.1245
  },
  "brightness_statistics": {
    "mean": 32.2104,
    "median": 0.0,
    "std": 57.8123,
    "min": 0.0,
    "max": 255.0
  },
  "contrast_statistics": {
    "mean": 58.4812,
    "median": 58.1032,
    "std": 3.4121,
    "min": 45.1203,
    "max": 72.4102
  },
  "quality_category_counts": {
    "blur": { "sharp": 188, "moderate": 12, "blurry": 0 },
    "brightness": { "dark": 200, "normal": 0, "bright": 0 },
    "contrast": { "low": 0, "normal": 196, "high": 4 }
  }
}
```

### 5.3 `visual_quality_metadata.json`
Complete execution environment, hardware audit, library versions, and configuration thresholds.

### 5.4 `visual_quality_errors.json`
Comprehensive error log detailing any rejected frames, missing paths, or corrupted payloads.

---

## 6. Command-Line Interface (CLI)

```bash
# Execute visual quality extraction on preprocessed frames with Day 2 metadata
python -m src.features.visual_quality_extractor \
  --input datasets/processed/driving_sample/preprocessed_frames \
  --metadata datasets/processed/driving_sample/metadata.csv \
  --output datasets/processed/driving_sample/features/visual_quality/run_official \
  --dataset-id driving_sample \
  --overwrite \
  --verbose
```

### Available CLI Flags:
- `--input, -i`: Path to directory containing frame images (required).
- `--metadata, -m`: Path to Day 2 `metadata.csv` (optional).
- `--config, -c`: Path to `config/feature_config.yaml` (optional).
- `--output, -o`: Custom output directory (optional).
- `--dataset-id`: Dataset identifier name (optional, defaults to directory inference).
- `--max-frames`: Cap on total frames to process.
- `--start-frame`: Starting source frame index filter.
- `--end-frame`: Ending source frame index filter.
- `--overwrite`: Overwrite destination directory if non-empty.
- `--verbose, -v`: Enable verbose debug logging and stack traces.
