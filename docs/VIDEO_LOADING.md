# PerceptGuard-AI: Video Loading & Ingestion Specification

**Module:** `src.video_processing.video_loader`  
**Author:** Staff Computer Vision & Autonomous Driving ML Systems Engineer  
**Document Version:** 1.0.0  
**Status:** **Implemented, Tested & Verified (Day 2 – Task 2 Baseline)**

---

## 1. Executive Summary & Purpose

In autonomous-driving perception research, robust video ingestion is the foundational prerequisite for downstream object detection (YOLOv8), multi-object tracking (ByteTrack), and predictive reliability forecasting (LSTM/GRU).

Standard naive implementations (`cv2.VideoCapture`) suffer from severe vulnerabilities in production pipelines:
1. **Resource Leaks:** Open OS file handles and capture streams left unclosed upon unhandled exceptions.
2. **Silent Ingestion Failures:** OpenCV does not raise native Python exceptions when a file is corrupted, missing, or unsupported; instead, it silently returns `False` from `.read()`, leading to subtle downstream bugs or infinite loops.
3. **Miscalibrated Temporal Metadata:** Corrupted video headers frequently report $0$ FPS or negative frame counts, which derail temporal sequence windowing ($W=10$).
4. **RAM Thrashing:** Decoding and storing entire driving sequences in RAM exhausts working memory on edge hardware (Intel Core i3, 8 GB RAM).

The `VideoLoader` class provides an enterprise-grade, memory-efficient abstraction that guarantees deterministic path validation, comprehensive metadata extraction, structured domain exception handling, and strict resource deallocation via the Python Context Manager protocol (`with VideoLoader(...) as loader:`).

---

## 2. Dependencies & Runtime Environment

- **Python Runtime:** Python 3.10+ (Tested on Python 3.13.3 64-bit AMD64)
- **OpenCV:** `opencv-python>=4.8.0` (Verified: `5.0.0.93`)
- **NumPy:** `numpy>=1.24.0` (Verified: `2.5.4`)
- **PyYAML:** `pyyaml>=6.0.0` (Verified: `6.0.3`)
- **Testing:** `pytest>=8.0.0` / standard library `unittest`

---

## 3. Module Architecture & File Locations

```
PerceptGuard-AI/
├── config/
│   └── video_config.yaml           # Centralized video ingestion parameters & thresholds
├── docs/
│   └── VIDEO_LOADING.md            # Technical specification & API documentation
├── outputs/
│   └── logs/                       # Execution and processing logs (.gitkeep)
├── src/
│   └── video_processing/           # Video ingestion and decoding package
│       ├── __init__.py             # Public package exports
│       └── video_loader.py         # VideoLoader class & VideoMetadata dataclass
└── tests/
    └── test_video_loader.py        # Automated unit and real-video integration test suite
```

---

## 4. API Reference & Usage Guide

### 4.1 Data Contract: `VideoMetadata`

The `VideoMetadata` dataclass encapsulates all physical stream characteristics:

```python
@dataclass(frozen=True)
class VideoMetadata:
    file_path: Path             # Absolute resolved Path to video
    filename: str               # File basename (e.g. 'converted_026c7465-309f6d33.mp4')
    width: int                  # Spatial horizontal frame resolution in pixels
    height: int                 # Spatial vertical frame resolution in pixels
    fps: float                  # Temporal sampling rate (frames per second)
    total_frames: int           # Total frame index count in the sequence
    duration_seconds: float     # Total duration (total_frames / fps)
    codec: str                  # FourCC video container codec identifier
    is_valid: bool              # Diagnostic validity flag
```

### 4.2 Standard Context Manager Pattern (Recommended)

```python
from pathlib import Path
from src.video_processing.video_loader import VideoLoader

video_path = Path("datasets/raw/converted_026c7465-309f6d33.mp4")

# Automatically opens, validates, and guarantees release of VideoCapture file handles
with VideoLoader(video_path) as loader:
    metadata = loader.get_metadata()
    print(f"Loaded: {metadata.filename}")
    print(f"Resolution: {metadata.width}x{metadata.height} @ {metadata.fps:.2f} FPS")
    print(f"Total Frames: {metadata.total_frames} ({metadata.duration_seconds:.2f}s)")
    print(f"Codec: {metadata.codec}")
```

### 4.3 Explicit Resource Management Pattern

```python
from src.video_processing.video_loader import VideoLoader

loader = VideoLoader("datasets/raw/converted_026c7465-309f6d33.mp4")
try:
    loader.open()
    metadata = loader.get_metadata()
    # Pipeline execution...
finally:
    loader.release()  # Idempotent cleanup
```

---

## 5. Structured Error Handling & Domain Exceptions

All exceptions inherit from `VideoProcessingError` to enable granular catch blocks:

```python
from src.video_processing.video_loader import (
    VideoLoader,
    VideoNotFoundError,
    InvalidVideoPathError,
    VideoOpenError,
    InvalidVideoMetadataError,
)

try:
    with VideoLoader("datasets/raw/target_video.mp4") as loader:
        meta = loader.get_metadata()
except VideoNotFoundError as e:
    logger.error("Missing file: %s", e)
except InvalidVideoPathError as e:
    logger.error("Path is a directory or invalid: %s", e)
except VideoOpenError as e:
    logger.error("Corrupted or unsupported video stream: %s", e)
except InvalidVideoMetadataError as e:
    logger.error("Corrupted video headers (e.g. 0 FPS or 0 frames): %s", e)
```

---

## 6. Real-World Driving Video Integration Test Results

The module was evaluated against the verified BDD100K daytime urban driving sequence (`datasets/raw/converted_026c7465-309f6d33.mp4`):

| Diagnostic Metric | Extracted Value | Validation Status |
| :--- | :--- | :--- |
| **Filename** | `converted_026c7465-309f6d33.mp4` | Verified |
| **Spatial Resolution** | **720 × 1280** pixels | Verified |
| **Temporal Frame Rate** | **30.09 FPS** | Verified |
| **Total Sequential Frames** | **1,204 frames** | Verified |
| **Calculated Duration** | **40.01 seconds** | Verified |
| **Underlying Codec** | `FMP4` (ISO MPEG-4 Part 14) | Verified |
| **Memory Consumption** | `< 25 MB` (Stream header inspection only) | Verified (Lightweight CPU) |
| **Resource Cleanup** | Verified (Stream closed immediately) | Verified |

---

## 7. Automated Test Suite Execution

Run the complete test suite using either `unittest` or `pytest`:

```powershell
# Using Python standard unittest
python tests/test_video_loader.py

# Using pytest
pytest tests/test_video_loader.py -v
```

### Test Suite Execution Output
```text
tests/test_video_loader.py::TestVideoLoader::test_01_valid_synthetic_video_loading PASSED [ 12%]
tests/test_video_loader.py::TestVideoLoader::test_02_metadata_retrieval PASSED             [ 25%]
tests/test_video_loader.py::TestVideoLoader::test_03_missing_video_file_raises_error PASSED [ 37%]
tests/test_video_loader.py::TestVideoLoader::test_04_directory_path_raises_error PASSED   [ 50%]
tests/test_video_loader.py::TestVideoLoader::test_05_corrupted_or_non_video_file_raises_error PASSED [ 62%]
tests/test_video_loader.py::TestVideoLoader::test_06_resource_release_is_idempotent PASSED [ 75%]
tests/test_video_loader.py::TestVideoLoader::test_07_context_manager_lifecycle PASSED     [ 87%]
tests/test_video_loader.py::TestVideoLoader::test_08_real_bdd100k_video_integration PASSED [100%]

============================== 8 passed in 0.73s ==============================
```

---

## 8. Next Milestones (Strict Boundaries)

Per development phase boundaries, frame decoding, frame extraction, image resizing, blur/sharpness calculation, YOLO inference, and multi-object tracking are deferred to subsequent tasks (Day 2 Task 3 & Day 3).

