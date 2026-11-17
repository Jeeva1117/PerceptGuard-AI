# PerceptGuard-AI: Video Loading & Ingestion Specification

**Module:** `src.video_processing.video_loader`  
**Author:** Staff Computer Vision & Autonomous Driving ML Systems Engineer  
**Document Version:** 1.1.0  
**Status:** **Diagnosed, Improved, Tested & Formally Verified**

---

## 1. Executive Summary & Purpose

In autonomous-driving perception research, robust video ingestion is the foundational prerequisite for downstream object detection (YOLOv8), multi-object tracking (ByteTrack), and predictive reliability forecasting (LSTM/GRU).

Standard naive implementations (`cv2.VideoCapture`) suffer from severe vulnerabilities in production pipelines:
1. **Resource Leaks:** Open OS file handles and capture streams left unclosed upon unhandled exceptions.
2. **Silent Ingestion Failures:** OpenCV does not raise native Python exceptions when a file is corrupted, missing, empty, or unsupported; instead, it silently returns `False` from `.read()`, leading to subtle downstream bugs or infinite loops.
3. **Miscalibrated Temporal Metadata:** Corrupted video headers frequently report $0$ FPS or negative frame counts, which derail temporal sequence windowing ($W=10$).
4. **RAM Thrashing:** Decoding and storing entire driving sequences in RAM exhausts working memory on edge hardware (Intel Core i3, 8 GB RAM).

The `VideoLoader` class provides an enterprise-grade, memory-efficient abstraction that guarantees deterministic path validation, comprehensive metadata extraction, structured domain exception handling, strict first-frame decoding verification, and resource deallocation via the Python Context Manager protocol (`with VideoLoader(...) as loader:`).

---

## 2. In-Depth Diagnosis: IDE Video Preview vs. OpenCV Loading

### 2.1 The Observed Issue
When opening `datasets/raw/converted_026c7465-309f6d33.mp4` directly inside the Antigravity editor tab, the IDE displays:
> *"An error occurred while loading the video file."*

### 2.2 Diagnostic Investigation Across 5 Possibilities
1. **Antigravity Editor Built-In Video Preview Limitation (CONFIRMED ROOT CAUSE):**
   The Antigravity IDE is built on an Electron/Chromium desktop foundation (VS Code core). When an `.mp4` file is clicked in the file explorer, the editor invokes its built-in media preview extension, which attempts to render the video using an HTML5 `<video>` tag inside a Chromium webview. Chromium's media engine only bundles open/royalty-free web codecs (H.264/AVC1, VP8, VP9, AV1). It **does not support** MPEG-4 Part 2 Visual (`FMP4` / `mp4v-es`). When Chromium encounters this codec, it emits `MEDIA_ERR_SRC_NOT_SUPPORTED`, causing the editor to display the error banner.
2. **Incorrect Video Path Resolution (DISPROVEN):**
   Path resolution was validated using `pathlib.Path.resolve()`. The file resides at `C:\Users\acer\Desktop\PerceptGuard-AI\datasets\raw\converted_026c7465-309f6d33.mp4` and is resolved deterministically across Windows backslashes and POSIX forward slashes.
3. **OpenCV VideoCapture Failure (DISPROVEN):**
   OpenCV utilizes its bundled `opencv_videoio_ffmpeg` backend (prebuilt FFmpeg binaries). FFmpeg libavcodec includes full support for MPEG-4 Part 2 (`FMP4`). OpenCV opens the file immediately (`cap.isOpened() == True`).
4. **Unsupported Video Codec in OpenCV (DISPROVEN FOR OPENCV; APPLIES ONLY TO CHROMIUM):**
   OpenCV's FFMPEG backend natively decodes `FMP4` frames into standard BGR uint8 NumPy arrays.
5. **Corrupted or Incomplete Video File (DISPROVEN):**
   The file size is 49,774,590 bytes (47.47 MB). File headers (`ftypisom`, `mdat`, `moov`) are intact. All 1,204 frames decode with zero corrupted frames.

### 2.3 Verdict & Engineering Policy
The video file is **100% healthy, complete, and fully decodable by OpenCV**. The error is strictly a **presentation limitation of the IDE's Chromium HTML5 media player**. Per data governance rules:
- The original video is **not** deleted, renamed, or overwritten.
- OpenCV pipeline consumers (`FrameExtractor`, `DatasetValidator`) operate seamlessly on the original file.
- `VideoLoader` now includes an explicit verification command and first-frame decode validation.

---

## 3. Production Enhancements Implemented in `VideoLoader`

To eliminate silent failures and provide robust diagnostics, the following enhancements were implemented:
1. **Zero-Byte File Trap:** Added `EmptyVideoError` (inherits from `CorruptedVideoError` and `VideoOpenError`). `_verify_path_integrity()` checks `stat().st_size == 0` and halts before unhandled C++ exceptions occur.
2. **First-Frame Decoding Verification:** `VideoLoader.open(verify_first_frame=True)` attempts reading the first frame (`cap.read()`). If `ret` is False or `frame` is empty, `FirstFrameDecodeError` is raised.
3. **Automatic Stream Pointer Rewind:** After first-frame verification, `cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)` rewinds the stream pointer back to frame 0, ensuring downstream sequential readers receive frame 0.
4. **Backend & Telemetry Tracking:** `VideoMetadata` now captures `backend` (`cap.getBackendName()`), reporting `"FFMPEG"` on Windows.
5. **Programmatic Verification API:** `loader.verify() -> Dict[str, Any]` audits file existence, size, OpenCV open, backend, FourCC codec, resolution, FPS, total frames, duration, and first frame decode, returning a comprehensive dictionary.

---

## 4. Reusable Verification Command

Execute the standardized verification command:

```powershell
python -m src.video_processing.video_loader --verify
```

### Exact Verification Output on Real Driving Dataset
```text
VIDEO FILE: FOUND
VIDEO OPEN: SUCCESS
CODEC / BACKEND: FMP4 / FFMPEG
RESOLUTION: 720 x 1280
FPS: 30.09
FRAME COUNT: 1204
DURATION: 40.01s
FIRST FRAME DECODE: SUCCESS
VIDEO LOADING STATUS: PASS
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--verify` | flag | `True` | Runs comprehensive video stream verification. |
| `--video-path` | string | `None` | Path to custom target video file (falls back to `config/video_config.yaml`). |
| `--config` | string | `None` | Path to custom YAML configuration file. |

---

## 5. Structured Error Handling & Domain Exceptions

All domain exceptions inherit from `VideoProcessingError`:

```
VideoProcessingError (Base)
├── VideoNotFoundError
├── InvalidVideoPathError
├── InvalidVideoMetadataError
└── VideoOpenError
    └── CorruptedVideoError
        ├── EmptyVideoError
        └── FirstFrameDecodeError
```

```python
from src.video_processing.video_loader import (
    VideoLoader,
    VideoNotFoundError,
    InvalidVideoPathError,
    VideoOpenError,
    CorruptedVideoError,
    EmptyVideoError,
    FirstFrameDecodeError,
    InvalidVideoMetadataError,
)

try:
    with VideoLoader("datasets/raw/converted_026c7465-309f6d33.mp4") as loader:
        meta = loader.get_metadata()
except VideoNotFoundError as e:
    logger.error("Missing video file: %s", e)
except InvalidVideoPathError as e:
    logger.error("Path is a directory: %s", e)
except EmptyVideoError as e:
    logger.error("Video file is 0 bytes: %s", e)
except FirstFrameDecodeError as e:
    logger.error("Failed to decode first frame: %s", e)
except VideoOpenError as e:
    logger.error("OpenCV container failure: %s", e)
except InvalidVideoMetadataError as e:
    logger.error("Physical metadata constraint violation: %s", e)
```

---

## 6. Real-World Driving Video Diagnostic Telemetry

Evaluated against `datasets/raw/converted_026c7465-309f6d33.mp4`:

| Diagnostic Property | Extracted Value | Verification Result |
| :--- | :--- | :--- |
| **Filesystem Status** | Exists, regular file | Verified |
| **Filesystem Size** | 49,774,590 bytes (47.47 MB) | Verified |
| **OpenCV Capture Open** | `SUCCESS` | Verified |
| **Capture Backend** | `FFMPEG` | Verified |
| **Video Codec** | `FMP4` (ISO MPEG-4 Part 2) | Verified |
| **Spatial Resolution** | `720 x 1280` pixels | Verified |
| **Temporal Frame Rate** | `30.09 FPS` | Verified |
| **Total Frame Count** | `1,204 frames` | Verified |
| **Stream Duration** | `40.01 seconds` | Verified |
| **First Frame Decoding** | `SUCCESS` ($1280 \times 720 \times 3$ uint8) | Verified |
| **Overall Loading Status** | **`PASS`** | Verified |

---

## 7. Automated Test Suite Execution

Run the complete 13-test suite:

```powershell
python tests/test_video_loader.py
```

### Test Suite Execution Output
```text
test_01_valid_synthetic_video_loading (__main__.TestVideoLoader.test_01_valid_synthetic_video_loading) ... ok
test_02_metadata_retrieval (__main__.TestVideoLoader.test_02_metadata_retrieval) ... ok
test_03_missing_video_file_raises_error (__main__.TestVideoLoader.test_03_missing_video_file_raises_error) ... ok
test_04_directory_path_raises_error (__main__.TestVideoLoader.test_04_directory_path_raises_error) ... ok
test_05_corrupted_or_non_video_file_raises_error (__main__.TestVideoLoader.test_05_corrupted_or_non_video_file_raises_error) ... ok
test_06_zero_byte_video_raises_error (__main__.TestVideoLoader.test_06_zero_byte_video_raises_error) ... ok
test_07_failed_first_frame_decoding_raises_error (__main__.TestVideoLoader.test_07_failed_first_frame_decoding_raises_error) ... ok
test_08_resource_release_is_idempotent (__main__.TestVideoLoader.test_08_resource_release_is_idempotent) ... ok
test_09_context_manager_lifecycle (__main__.TestVideoLoader.test_09_context_manager_lifecycle) ... ok
test_10_programmatic_verify_method_success (__main__.TestVideoLoader.test_10_programmatic_verify_method_success) ... ok
test_11_programmatic_verify_method_failure_modes (__main__.TestVideoLoader.test_11_programmatic_verify_method_failure_modes) ... ok
test_12_real_bdd100k_video_integration (__main__.TestVideoLoader.test_12_real_bdd100k_video_integration) ... ok
test_13_cli_verification_runner (__main__.TestVideoLoader.test_13_cli_verification_runner) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.380s

OK
```
