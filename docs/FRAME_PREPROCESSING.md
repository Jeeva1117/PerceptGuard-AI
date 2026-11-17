# Frame Preprocessing Pipeline Specification

## 1. Overview & Architectural Role

The **Frame Preprocessing Pipeline** (`src/video_processing/frame_preprocessor.py`) is Stage 3 of the **PerceptGuard AI** perception architecture. It bridges raw video frame extraction (Stage 2) and downstream perception models (YOLOv8 object detection and ByteTrack tracking).

Its primary objective is transforming heterogeneous, raw extracted driving frames into validated, geometrically consistent, aspect-ratio-preserved, and diagnostic-annotated images while strictly preventing memory bloat and preserving the original extracted dataset.

```
datasets/processed/driving_sample/frames/ (Raw Extracted Frames)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                     FramePreprocessor                       │
│  - Multi-point image validation & corruption detection       │
│  - BGR color space management & grayscale conversion        │
│  - Visual quality diagnostics (Blur, Brightness, Contrast)  │
│  - Aspect-ratio-preserving letterbox canvas resizing        │
│  - Atomic disk persistence with verified encoding           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
datasets/processed/driving_sample/preprocessed_frames/ (640x360 px)
  ├── frame_000001.jpg (Letterboxed, Aspect Preserved)
  ├── frame_000002.jpg (Letterboxed, Aspect Preserved)
  └── ...
```

---

## 2. Multi-Point Image Validation Rules

To prevent corrupted inputs from poisoning downstream perception models, every candidate frame undergoes strict diagnostic inspection before any transformation:

1. **Filesystem Integrity:** Verifies that the path exists and is a regular file (rejects directory paths and missing symlinks).
2. **Byte Feasibility:** Rejects 0-byte or truncated files (`stat().st_size > 0`).
3. **OpenCV Header & Container Parsing:** Decodes stream via `cv2.imread(..., cv2.IMREAD_UNCHANGED)`. Validates that the returned numpy array is non-None and non-empty.
4. **Spatial Geometry Bounds:** Rejects zero or negative spatial dimensions ($W > 0, H > 0$).
5. **Channel Layout Verification:** Validates that channel count $C \in \{1, 3, 4\}$. 4-channel BGRA images are automatically normalized to 3-channel BGR.

Invalid frames are recorded with structured error descriptions in `PreprocessedFrameRecord` rather than silently skipped.

---

## 3. Resizing & Aspect-Ratio Preservation Strategy

Autonomous driving object detection models (such as YOLOv8) require consistent spatial dimensions for batch tensor processing. However, direct spatial stretching introduces severe aspect ratio distortion (e.g., squishing tall trucks or stretching oncoming vehicles), altering visual aspect ratios and compromising geometric feature extraction.

### Letterbox Resizing Formulation
Given source dimensions $(W_{\text{orig}}, H_{\text{orig}})$ and target dimensions $(W_t, H_t)$:

1. **Proportional Scale Factor:**
   \[
   s = \min\left(\frac{W_t}{W_{\text{orig}}}, \frac{H_t}{H_{\text{orig}}}\right)
   \]
2. **Scaled Dimensions:**
   \[
   W_{\text{new}} = \text{round}(W_{\text{orig}} \cdot s), \quad H_{\text{new}} = \text{round}(H_{\text{orig}} \cdot s)
   \]
3. **Interpolation Selection:**
   - Downscaling ($s < 1.0$): `cv2.INTER_AREA` (resamples pixel area to eliminate moiré and aliasing artifacts).
   - Upscaling ($s \ge 1.0$): `cv2.INTER_LINEAR` (bilinear interpolation).
4. **Centered Canvas Placement:**
   A canvas of exact dimensions $(H_t, W_t)$ is initialized with `padding_color` (default: black `(0, 0, 0)`). The scaled content is centered:
   \[
   \text{pad}_{\text{top}} = \frac{H_t - H_{\text{new}}}{2}, \quad \text{pad}_{\text{left}} = \frac{W_t - W_{\text{new}}}{2}
   \]
   \[
   \text{Canvas}[\text{pad}_{\text{top}} : \text{pad}_{\text{top}} + H_{\text{new}}, \; \text{pad}_{\text{left}} : \text{pad}_{\text{left}} + W_{\text{new}}] = I_{\text{scaled}}
   \]

### Application to BDD100K Driving Video
- **Source Resolution:** $720 \times 1280$ pixels (Portrait 9:16)
- **Target Resolution:** $640 \times 360$ pixels (Landscape 16:9)
- **Calculated Scale:** $s = \min(640/720, 360/1280) = 0.28125$
- **Scaled Dimensions:** $202 \times 360$ pixels
- **Letterbox Padding:** $\text{pad}_{\text{left}} = 219$ px, $\text{pad}_{\text{right}} = 219$ px, $\text{pad}_{\text{top}} = 0$ px.
- **Result:** Exact $640 \times 360$ dimensions, zero vehicle distortion, centered content.

---

## 4. Color-Channel Convention

| Stage / Component | Convention | Justification |
| :--- | :--- | :--- |
| **OpenCV Disk I/O** | **BGR** | OpenCV `imread()` and `imwrite()` interact with image containers in native BGR format. Preserving BGR prevents red/blue color channel inversion on disk. |
| **Visual Quality Diagnostics** | **Grayscale (GRAY)** | Converted via `cv2.COLOR_BGR2GRAY` for luminance and edge gradient calculations. |
| **YOLOv8 / PyTorch Tensor Input** | **RGB** | Standard PyTorch vision models expect normalized RGB tensors ($[0.0, 1.0]$). Available via `preprocessor.convert_color(img, "RGB")`. |

---

## 5. Visual Quality Diagnostics

To support downstream predictive reliability modeling, three visual quality indicators are extracted for each frame:

### A. Blur Score (Laplacian Variance)
\[
\text{Score}_{\text{blur}} = \text{Var}\left(\nabla^2 I_{\text{gray}}\right) = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)
\]
- **Interpretation:** Evaluates the density of sharp edge gradients in the scene.
  - $\text{Score} < 50.0$: Heavy optical blur, motion smear, or dense fog.
  - $\text{Score} > 150.0$: Sharp, high-contrast urban road scene.

### B. Brightness Score (Mean Luminance)
\[
\text{Score}_{\text{bright}} = \frac{1}{W \cdot H} \sum_{x,y} I_{\text{gray}}(x, y)
\]
- **Interpretation:** Range $[0.0, 255.0]$.
  - $\text{Score} < 40.0$: Deep darkness, unlit night, unlit tunnel.
  - $\text{Score} > 220.0$: Severe direct sun glare or camera saturation.
  - Nominal daylight: $80.0 - 160.0$.

### C. Contrast Score (RMS Contrast)
\[
\text{Score}_{\text{contrast}} = \sqrt{\frac{1}{W \cdot H} \sum_{x,y} \left(I_{\text{gray}}(x, y) - \mu\right)^2}
\]
- **Interpretation:** Standard deviation of pixel intensities. Low values denote low dynamic range or uniform haze; high values denote strong shadow-highlight differentiation.

> [!IMPORTANT]
> **Heuristic Quality vs. Detection Reliability Distinction:**  
> These metrics measure raw visual scene characteristics, NOT ground-truth object detection reliability. A sharp image can still suffer detection failure due to occlusion or small target scale; conversely, a slightly blurry scene may still yield correct detections for large proximate vehicles. Future reliability targets $R_{t+h}$ are computed strictly via IoU correctness against ground truth.

---

## 6. Configuration Reference

Defined in [config/video_config.yaml](file:///c:/Users/acer/Desktop/PerceptGuard-AI/config/video_config.yaml):

```yaml
preprocessing:
  input_directory: datasets/processed/driving_sample/frames
  output_directory: datasets/processed/driving_sample/preprocessed_frames
  target_width: 640
  target_height: 360
  preserve_aspect_ratio: true
  padding_color: [0, 0, 0]
  color_mode: BGR
  image_format: jpg
  jpeg_quality: 95
  calculate_blur: true
  calculate_brightness: true
  calculate_contrast: true
  blur_threshold: null
  brightness_threshold_low: 40.0
  brightness_threshold_high: 220.0
  overwrite: true
```

---

## 7. Directory Organization & Immutability

```
datasets/processed/driving_sample/
├── frames/                         # IMMUTABLE: Raw Task 3 extracted frames (720x1280)
│   ├── frame_000001.jpg
│   └── ...
└── preprocessed_frames/            # PROCESSED: Standardized Task 4 outputs (640x360)
    ├── frame_000001.jpg
    └── ...
```

---

## 8. Verification & Test Suite

### Automated Unit & Integration Tests
Automated test suite covers 17 synthetic unit tests and 1 real-world integration test in [tests/test_frame_preprocessor.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/test_frame_preprocessor.py):

```powershell
python tests/test_frame_preprocessor.py
```

```text
test_01_valid_image_loading ... ok
test_02_missing_image_handling ... ok
test_03_corrupted_image_handling ... ok
test_04_empty_image_handling ... ok
test_05_invalid_image_dimensions ... ok
test_06_image_resizing_direct ... ok
test_07_aspect_ratio_preservation_letterbox ... ok
test_08_bgr_rgb_conversion ... ok
test_09_grayscale_conversion ... ok
test_10_blur_score_calculation ... ok
test_11_brightness_score_calculation ... ok
test_12_contrast_score_calculation ... ok
test_13_output_image_saving ... ok
test_14_failed_image_write_handling ... ok
test_15_original_input_image_preservation ... ok
test_16_batch_directory_processing ... ok
test_17_invalid_configuration_handling ... ok
test_18_real_dataset_frames_integration ... ok

----------------------------------------------------------------------
Ran 18 tests in 0.880s
OK
```

### Full Driving Dataset Preprocessing Benchmark
Executed across all 200 Task 3 extracted frames (`datasets/processed/driving_sample/frames/`):

- **Total Input Frames:** `200`
- **Successfully Preprocessed Frames:** `200` (100% success rate)
- **Failed Frames:** `0`
- **Source Dimensions:** `720 × 1280` px
- **Target Dimensions:** `640 × 360` px (Letterboxed, aspect ratio strictly preserved)
- **Mean Blur Sharpness (Laplacian Var):** `436.75`
- **Mean Brightness (Mean Grayscale):** `102.05`
- **Mean Contrast (RMS Std Dev):** `62.04`
- **Execution Wall Time:** `9.41 seconds` (Throughput: **21.3 frames/s** on Intel Core i3 CPU)
- **Processed Storage Footprint:** `7.02 MB` in `datasets/processed/driving_sample/preprocessed_frames/`
- **Data Governance:** Strictly excluded from Git version control via `.gitignore`.

