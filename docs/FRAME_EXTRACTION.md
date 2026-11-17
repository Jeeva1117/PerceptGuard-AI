# Frame Extraction Engine Specification

## 1. Overview & Architectural Role

The **Frame Extraction Engine** (`src/video_processing/frame_extractor.py`) serves as Stage 2 of the **PerceptGuard AI** perception pipeline. Its primary responsibility is the deterministic, resource-bounded extraction and time-sampling of raw autonomous-driving video frames into high-fidelity image sequences suitable for downstream object detection (YOLOv8) and multi-object tracking (ByteTrack).

```
Raw Video (.mp4)
       │
       ▼
┌──────────────────────────────────────────────┐
│                VideoLoader                   │
│  - Stream validation & OpenCV VideoCapture   │
│  - Hardware-bounded header inspection        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│               FrameExtractor                 │
│  - Deterministic time-based frame sampling   │
│  - Single-frame in-memory decoding           │
│  - Fast OpenCV grab() frame-skipping         │
│  - Zero-padded sequential image persistence  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
datasets/processed/driving_sample/frames/
  ├── frame_000001.jpg (t = 0.000s, source_idx = 0)
  ├── frame_000002.jpg (t = 0.199s, source_idx = 6)
  └── ...
```

---

## 2. Temporal Sampling Strategy

### Mathematical Formulation

Given:
- Source video frame rate: \(FPS_{\text{src}} > 0\) (e.g., \(30.09\) FPS)
- Target extraction frame rate: \(FPS_{\text{target}} > 0\) (e.g., \(5.0\) FPS)
- Total frames in source container: \(N_{\text{src}} \ge 1\) (e.g., \(1,204\) frames)
- Configured maximum frames: \(M_{\text{max}}\) (optional)

#### Subsampling Case (\(0 < FPS_{\text{target}} < FPS_{\text{src}}\))
The sampling step in frame space is:
\[
S = \frac{FPS_{\text{src}}}{FPS_{\text{target}}}
\]
For each extracted frame index \(k = 0, 1, 2, \dots\):
\[
\text{source\_frame\_index}_k = \text{round}(k \cdot S)
\]

#### Guarantees
1. **Monotonicity & Zero Duplication:** Since \(S > 1.0\), \(\text{round}((k+1)S) > \text{round}(kS)\) strictly. Duplicate source frames are mathematically impossible.
2. **Boundary Preservation:** The sequence begins at \(k = 0\) (\(\text{source\_frame\_index} = 0\)) and terminates when \(\text{source\_frame\_index} \ge N_{\text{src}}\) or \(k \ge M_{\text{max}}\).
3. **Oversampling Safety Guard:** If \(FPS_{\text{target}} \ge FPS_{\text{src}}\), duplicate frame upsampling is strictly disallowed. The engine clamps the sampling step to \(S = 1.0\) (extracting each sequential frame exactly once) and logs an informative operational warning.

---

## 3. Timestamp Calculation Convention

Every extracted frame carries a physically grounded timestamp in seconds:
\[
t(\text{source\_frame\_index}) = \frac{\text{source\_frame\_index}}{FPS_{\text{src}}}
\]

### Rationale
Calculating timestamps from the exact discrete source frame index and container frame rate provides absolute determinism and numerical stability across platforms, eliminating non-monotonic timestamp drift caused by container presentation timestamp (PTS) quantization.

---

## 4. Resource & Memory Management

Designed specifically for constrained CPU runtimes (Intel Core i3, 8 GB RAM):

| Optimization Strategy | Implementation Details |
| :--- | :--- |
| **Stream-Decoupled Skipping** | Uses OpenCV `VideoCapture.grab()` for skipped source frames. `grab()` advances the demuxer position without triggering expensive YUV-to-BGR color conversion or full decompression. |
| **Single-Frame Buffering** | Only one decoded frame (`numpy.ndarray`) exists in memory at any point in time. Image buffers are immediately garbage collected after `cv2.imwrite()`. |
| **Idempotent Resource Teardown** | File descriptors are guaranteed to be closed via Python context manager protocol (`__enter__` / `__exit__`), explicit `release_resources()`, and finalizer fallback (`__del__`). |
| **Zero Heap Accumulation** | The engine accumulates only lightweight dataclass metadata (`ExtractedFrameInfo`), consuming \(< 1\) MB of heap memory across thousands of frames. |

---

## 5. Configuration Options

Configured via [config/video_config.yaml](file:///c:/Users/acer/Desktop/PerceptGuard-AI/config/video_config.yaml):

```yaml
frame_extraction:
  output_directory: datasets/processed/driving_sample/frames
  extraction_fps: 5.0
  max_frames: 300
  image_format: jpg
  filename_prefix: frame_
  filename_padding: 6
  jpeg_quality: 95
  overwrite: true
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `output_directory` | `Path` | `datasets/processed/driving_sample/frames` | Directory where extracted images are persisted. |
| `extraction_fps` | `float` | `5.0` | Target temporal sampling rate in frames per second. |
| `max_frames` | `int` | `None` / `300` | Optional upper bound on total extracted frames. |
| `filename_prefix` | `str` | `"frame_"` | Prefix applied to sequential image filenames. |
| `filename_padding` | `int` | `6` | Zero-padding width for extracted frame indices. |
| `image_format` | `str` | `"jpg"` | Image format (`jpg`, `jpeg`, `png`, `webp`). |
| `jpeg_quality` | `int` | `95` | Compression quality for JPEG output (1–100). |
| `overwrite` | `bool` | `True` | Overwrite existing files if already present. |

---

## 6. Directory Structure & Naming Convention

Extracted images are stored in `datasets/processed/`:

```
datasets/
├── raw/
│   └── converted_026c7465-309f6d33.mp4   # Unmodified original BDD100K driving sample
└── processed/
    └── driving_sample/
        └── frames/
            ├── frame_000001.jpg           # Extracted ID 1, Source Index 0,  t = 0.000s
            ├── frame_000002.jpg           # Extracted ID 2, Source Index 6,  t = 0.199s
            ├── frame_000003.jpg           # Extracted ID 3, Source Index 12, t = 0.399s
            └── ...
```

> [!NOTE]
> The **extracted frame ID** (`1`, `2`, `3`...) and **source video frame index** (`0`, `6`, `12`...) are distinct concepts maintained independently in the `ExtractedFrameInfo` record.

---

## 7. Custom Exception Hierarchy

All extraction exceptions inherit from `VideoProcessingError`:

- `FrameExtractionError`: Base exception for extraction failures.
- `InvalidExtractionConfigError`: Raised on invalid configuration (negative FPS, padding out of range [1, 12], unsupported format).
- `FrameSaveError`: Raised when directory creation fails, an empty frame is supplied, or `cv2.imwrite` returns False.
- `VideoReadError`: Raised when unrecoverable decoding errors occur.

---

## 8. Usage Example

```python
from pathlib import Path
from src.video_processing import FrameExtractor

video_path = Path("datasets/raw/converted_026c7465-309f6d33.mp4")
output_dir = Path("datasets/processed/driving_sample/frames")

with FrameExtractor(
    video_source=video_path,
    output_dir=output_dir,
    extraction_fps=5.0,
    filename_prefix="frame_",
    filename_padding=6,
    jpeg_quality=95,
) as extractor:
    summary = extractor.extract_frames()
    print(f"Extracted {summary.extracted_frames_count} frames in {summary.elapsed_time_seconds:.2f}s")
    print(f"Time span: {summary.first_timestamp_seconds:.3f}s -> {summary.last_timestamp_seconds:.3f}s")
```

---

## 9. Verification & Benchmark Telemetry

### Unit Test Execution
Automated suite covers 12 synthetic unit tests and 1 real-world video integration test in [tests/test_frame_extractor.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/test_frame_extractor.py):

```powershell
python tests/test_frame_extractor.py
```

```text
Ran 13 tests in 1.242s
OK
```

### Real Autonomous-Driving Video Benchmark
Executed on actual BDD100K driving sample:

- **Source File:** `datasets/raw/converted_026c7465-309f6d33.mp4`
- **Source Resolution:** `720 × 1280 px`
- **Source FPS:** `30.09`
- **Total Source Frames:** `1,204`
- **Configured Extraction FPS:** `5.0`
- **Extracted Frames Count:** `200`
- **Extraction Execution Time:** `3.12 seconds` (Throughput: **64.2 frames/s** on Intel Core i3 CPU)
- **First Frame Timestamp:** `0.0000 s` (`frame_000001.jpg`, source index 0)
- **Last Frame Timestamp:** `39.8139 s` (`frame_000200.jpg`, source index 1198)
- **Output Storage Footprint:** `45.41 MB`
- **Data Governance:** Verified that `datasets/processed/*` is strictly ignored by `.gitignore` and excluded from GitHub version control.
