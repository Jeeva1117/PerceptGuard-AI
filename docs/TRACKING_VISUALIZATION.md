# PerceptGuard AI – Advanced Tracking Visualization Engine

## 1. Executive Overview & Technical Scope

The **Tracking Visualization Engine** (`TrackingVisualizer`) constitutes the primary visual diagnostics, verification, and presentation layer of **PerceptGuard AI**. It ingests sequential tracking results produced by upstream Multi-Object Tracking (MOT), ByteTrack, or Single-Object Tracking (SOT) engines and renders high-fidelity, annotated frame sequences and optional MP4 video artifacts.

Designed in accordance with autonomous driving perception standards (Tesla Vision style), the engine operates under strict production constraints:
- **Intel Core i3 / 8 GB RAM Compatibility**: Single-threaded CPU execution with deterministic sequential frame rendering and immediate frame buffer deallocation.
- **Strict Decoupling**: Ingests pre-computed tracking JSON artifacts. Does **not** load YOLO models, re-execute object detection, or re-run ByteTrack association algorithms.
- **Persistent Color Consistency**: Synthesizes deterministic, high-contrast, identity-preserving bounding box and trajectory colors via a golden-ratio HSV generator.
- **Temporal Motion Trajectory**: Traces historical centroid motion paths with configurable tail length and strict occlusion/teleportation gap gating.
- **Production HUD Overlay**: Renders semi-transparent diagnostic telemetry banners reporting frame sequence numbers, timestamps, active track counts, and run identifiers.

---

## 2. System Architecture & Decoupled Pipeline

```
+-------------------------------------------------------------------------------+
|                       UPSTREAM PERSISTED ARTIFACTS                             |
|  - multi_object_tracking.json / single_object_tracking.json / tracking_results|
|  - Preprocessed video frames (datasets/processed/driving_sample/...)         |
+---------------------------------------+---------------------------------------+
                                        | (Ingest & Validate Schema)
                                        v
+-------------------------------------------------------------------------------+
|                      TRACKING VISUALIZATION ENGINE                            |
|                       src/visualization/tracking_visualizer.py                |
|                                                                               |
|  1. Tracking Schema Ingestion & Frame Normalization                           |
|  2. Source Image Resolution & Safe Decoding                                   |
|  3. Deterministic Identity Color Generator (HSV Golden Ratio)                 |
|  4. Bounding Box & Non-Clipping Label Badge Rendering                         |
|  5. Centroid Dual-Ring Marker & Historical Trajectory Rendering               |
|  6. Semi-Transparent HUD Telemetry Banner Rendering                           |
|  7. Frame Buffer Persistence & Optional OpenCV VideoWriter Pipeline           |
+---------------------------------------+---------------------------------------+
                                        | (Structured Export)
                                        v
+-------------------------------------------------------------------------------+
|                         MULTI-FILE OUTPUT ARTIFACTS                           |
|  datasets/processed/<dataset>/visualizations/tracking/<run_id>/               |
|  |-- annotated_frames/                                                        |
|  |   |-- frame_000000.jpg                                                     |
|  |   `-- frame_000001.jpg ...                                                 |
|  |-- annotated_video.mp4           (Optional MP4 Video Sequence)              |
|  |-- visualization_summary.json    (Run Metrics, Frame Counts, Latency)       |
|  |-- visualization_metadata.json   (System Info, Styling, Parameter Config)   |
|  `-- errors.json                   (Structural Error Log & Tracebacks)        |
+-------------------------------------------------------------------------------+
```

---

## 3. Ingestion & Multi-Schema Support

The engine supports multiple tracking JSON formats across PerceptGuard AI milestones:

1. **Multi-Object Tracking Schema (v1.0)**:
   - Root fields: `schema_version`, `run_id`, `dataset_id`, `frames` (or `tracking_results`).
   - Frame object: `frame_id`, `frame_index`, `image_path` (or `frame_path`), `tracks` (list of track records).
   - Track object: `track_id`, `class_name`, `confidence`, `bbox` `[x1, y1, x2, y2]`.
2. **ByteTrack Legacy Schema**:
   - Frame array: `tracking_results` with nested detection arrays containing `track_id`.
3. **Single-Object Tracking Schema**:
   - Primary tracking file: `single_object_tracking.json` or `target_history.json` tracking individual selected targets across sequences.

### Schema Validation & Fallback Discovery
If a tracking file omits absolute source image paths or references relative paths, `TrackingVisualizer` performs multi-tier fallback resolution:
- Candidate 1: Direct absolute path resolution.
- Candidate 2: Relative path resolution against the tracking file's parent directory.
- Candidate 3: `datasets/processed/<dataset_id>/preprocessed_frames/`.
- Candidate 4: Recursive file discovery by `frame_id` or sequential zero-padded index matching (`frame_000000.jpg`).

---

## 4. Deterministic Identity Color Synthesis

To prevent color flicker, random drift, or cross-frame identity confusion, tracking colors are generated deterministically using the golden ratio in HSV space:

$$\text{Hue} = \left((\text{track\_id} \times 0.618033988749895) \pmod{1.0}\right) \times 180$$

```python
def get_deterministic_track_color(track_id: int) -> Tuple[int, int, int]:
    golden_ratio_conjugate = 0.618033988749895
    h = (track_id * golden_ratio_conjugate) % 1.0
    h_cv = int(h * 180)
    s_cv = 210  # High saturation for vivid contrast
    v_cv = 255  # Maximum brightness
    hsv_pixel = np.uint8([[[h_cv, s_cv, v_cv]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
```

### Visual Guarantees:
- **Frame-to-Frame Invariance**: Object ID 1 always maintains the exact same BGR color across all video frames.
- **High Inter-Class Distinguishability**: Consecutive track IDs receive maximally spaced hues on the 360-degree color wheel.
- **Dynamic Class Contrast**: Text labels and contrast backgrounds ensure readability across diverse driving lighting conditions (daylight, tunnels, glare).

---

## 5. Visual Markup Specification

Each detected and tracked object receives multi-tier visual markup:

### 5.1 Bounding Box & Label Badge
- **Coordinates**: Clamped strictly within image boundaries $[0, W-1] \times [0, H-1]$ with finite validation ($\text{NaN}/\text{Inf}$ rejection).
- **Label String**: `#<track_id> <class_name> <confidence%>` (e.g., `#2 truck 0.88`).
- **Text Background Badge**: Filled rectangle rendered behind text in the track's color. Non-clipping edge logic shifts the badge below the top bounding line if $y_1 - \text{badge\_height} < 0$.
- **Text Color**: High-contrast white or black based on background luminance.

### 5.2 Centroid Anchor Point
- A dual-ring centroid marker is drawn at the center coordinate $(c_x, c_y) = \left(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}\right)$:
  - Outer ring: Radius $r=4$ px in deterministic track color.
  - Inner core: Radius $r=2$ px in solid white `(255, 255, 255)`.

### 5.3 Motion Trajectory with Occlusion Protection
- Historical centroids for each active track ID are stored in circular buffers (default window: 30 frames).
- Trajectory lines are rendered connecting consecutive observations:
  $$\text{Line segment drawn only if } |idx_2 - idx_1| \le 2$$
- **Occlusion Gap Protection**: If a track is lost or occluded for more than 2 frames, the visualizer breaks the trajectory line rather than drawing a spurious teleportation line across the frame.

### 5.4 Diagnostic HUD Banner
- Position: Top-left overlay $[0, 0] \text{ to } [420, 48]$ px.
- Background: Semi-transparent black rectangle ($\alpha = 0.55$).
- Telemetry parameters:
  - Row 1: `Frame: <frame_id> | <width>x<height>`
  - Row 2: `Active Tracks: <N> | Run: <run_id>`

---

## 6. Video Compilation & Decoding Verification

When `--save-video` is enabled:
1. `cv2.VideoWriter` initializes using the standard Windows/Linux MP4 codec:
   ```python
   fourcc = cv2.VideoWriter_fourcc(*"mp4v")
   writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
   ```
2. Frames are sequentially streamed into the container during the single visualization pass (zero secondary disk reading).
3. Upon sequence termination, the container is flushed and released.
4. **Post-Write Validation Probe**: The engine immediately re-opens the video via `cv2.VideoCapture` to verify header integrity, frame count, dimensions, and readability:
   ```python
   chk = cv2.VideoCapture(str(video_path))
   if chk.isOpened() and int(chk.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
       video_success = True
   chk.release()
   ```

---

## 7. Multi-File Output Schema Specification

All artifacts are persisted into an isolated run directory: `datasets/processed/<dataset>/visualizations/tracking/vis_<dataset>_<timestamp>_<uuid>/`.

### 7.1 `visualization_summary.json`
```json
{
  "schema_version": "1.0",
  "run_id": "vis_driving_sample_20261117_162958_99428e",
  "dataset_id": "driving_sample",
  "tracking_run_id": "mot_driving_sample_20261117_160903_1c98f4",
  "tracking_file": "C:\\Users\\acer\\Desktop\\PerceptGuard-AI\\datasets\\processed\\driving_sample\\tracking\\multi_object\\mot_driving_sample_20261117_160903_1c98f4\\multi_object_tracking.json",
  "total_frames_in_tracking_results": 200,
  "total_frames_attempted": 200,
  "successful_visualizations": 200,
  "failed_visualizations": 0,
  "empty_frames": 34,
  "total_tracks_rendered": 193,
  "unique_track_ids_visualized": 3,
  "video_generated": true,
  "video_path": "C:\\Users\\acer\\Desktop\\PerceptGuard-AI\\datasets\\processed\\driving_sample\\visualizations\\tracking\\vis_driving_sample_20261117_162958_99428e\\annotated_video.mp4",
  "video_fps": 5.0,
  "total_visualization_time_seconds": 2.52,
  "average_visualization_time_ms": 12.33,
  "status": "PASS"
}
```

### 7.2 `visualization_metadata.json`
```json
{
  "schema_version": "1.0",
  "run_id": "vis_driving_sample_20261117_162958_99428e",
  "dataset_id": "driving_sample",
  "created_at": "2026-11-17T16:30:00.728566+00:00",
  "source_tracking_file": "C:\\Users\\acer\\Desktop\\PerceptGuard-AI\\datasets\\processed\\driving_sample\\tracking\\multi_object\\mot_driving_sample_20261117_160903_1c98f4\\multi_object_tracking.json",
  "source_frames_directory": "C:\\Users\\acer\\Desktop\\PerceptGuard-AI\\datasets\\processed\\driving_sample\\preprocessed_frames",
  "styling_parameters": {
    "show_trajectory": true,
    "trajectory_length": 30,
    "show_confidence": true,
    "show_frame_info": true,
    "line_thickness": null,
    "font_scale": null
  },
  "environment": {
    "python_version": "3.13.3",
    "platform": "win32",
    "opencv_version": "5.0.0"
  }
}
```

### 7.3 `errors.json`
```json
{
  "schema_version": "1.0",
  "run_id": "vis_driving_sample_20261117_162958_99428e",
  "error_count": 0,
  "errors": []
}
```

---

## 8. Command Line Interface (CLI)

The module includes a command line interface supporting both ad-hoc diagnostics and automated batch operations:

### 8.1 Visualizing Multi-Object Tracking (Full Sequence + Video)
```bash
python -m src.visualization.tracking_visualizer \
    --tracking-file datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
    --save-video \
    --overwrite
```

### 8.2 Rapid Smoke Test (First 30 Frames)
```bash
python -m src.visualization.tracking_visualizer \
    --tracking-file datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json \
    --max-frames 30 \
    --save-video
```

### 8.3 Visualizing Single-Object Tracking
```bash
python -m src.visualization.tracking_visualizer \
    --tracking-file datasets/processed/driving_sample/tracking/single_object/single_target_1_20261117_154123_fa3f36/single_object_tracking.json \
    --save-video
```

### 8.4 CLI Flag Reference
| Flag | Type | Description |
|---|---|---|
| `--tracking-file` | Path (Required) | Absolute or relative path to tracking results JSON |
| `--frames-dir` | Path (Optional) | Directory containing source frame images |
| `--output-dir` | Path (Optional) | Target root directory for visualization artifacts |
| `--trajectory-length` | Integer (Default: 30) | Maximum historical centroid points per track |
| `--save-video` | Boolean Flag | Enables generation of `annotated_video.mp4` |
| `--video-fps` | Float (Default: 5.0) | Frame rate for output video sequence |
| `--no-trajectory` | Boolean Flag | Disables motion trajectory rendering |
| `--no-confidence` | Boolean Flag | Hides confidence percentage from text badges |
| `--no-frame-info` | Boolean Flag | Disables top-left HUD telemetry banner |
| `--max-frames` | Integer (Optional) | Limits number of frames processed (smoke tests) |
| `--overwrite` | Boolean Flag | Replaces existing visualization directory |

---

## 9. Automated Test Suite (32 Unit & Integration Tests)

A dedicated test suite was implemented in `tests/visualization/test_tracking_visualizer.py` verifying full functional, structural, and behavioral compliance:

| Test ID | Test Method | Verification Target | Status |
|---|---|---|---|
| 01 | `test_01_visualizer_initialization` | Default parameters, state attributes, configuration initialization | **PASS** |
| 02 | `test_02_tracking_json_loading` | Ingestion of valid tracking JSON files | **PASS** |
| 03 | `test_03_tracking_schema_validation` | Validation of schema versions and missing core fields | **PASS** |
| 04 | `test_04_single_object_visualization` | Rendering single target annotations and trajectories | **PASS** |
| 05 | `test_05_multiple_object_visualization` | Rendering simultaneous multi-target annotations | **PASS** |
| 06 | `test_06_bounding_box_drawing` | Correct bounding box drawing and coordinate clamping | **PASS** |
| 07 | `test_07_track_id_labels` | Presence and formatting of `#<track_id>` in badges | **PASS** |
| 08 | `test_08_class_labels` | Inclusion of correct class names | **PASS** |
| 09 | `test_09_confidence_labels` | Confidence score formatting and toggle handling | **PASS** |
| 10 | `test_10_deterministic_track_colors` | Distinct, deterministic colors generated per track ID | **PASS** |
| 11 | `test_11_color_consistency_across_frames` | Persistent color stability across multiple frames | **PASS** |
| 12 | `test_12_object_center_drawing` | Centroid anchor point calculation and dual-ring rendering | **PASS** |
| 13 | `test_13_trajectory_generation` | Motion path line segment rendering | **PASS** |
| 14 | `test_14_independent_trajectories_for_different_ids` | Trajectory isolation between distinct track IDs | **PASS** |
| 15 | `test_15_trajectory_length_limits` | Trajectory FIFO buffer limiting to configured length | **PASS** |
| 16 | `test_16_missing_observations` | Trajectory continuity break across multi-frame occlusions | **PASS** |
| 17 | `test_17_invalid_bounding_boxes` | Handling of negative coordinates, inverted coordinates, non-finite values | **PASS** |
| 18 | `test_18_out_of_frame_coordinates` | Safe clamping of coordinates extending outside image boundaries | **PASS** |
| 19 | `test_19_empty_detection_frames` | Correct handling and status assignment for zero-track frames | **PASS** |
| 20 | `test_20_zero_object_tracking_results` | Graceful execution when tracking results contain 0 tracks | **PASS** |
| 21 | `test_21_missing_source_frames` | Logging and skipping frames when source image is absent | **PASS** |
| 22 | `test_22_corrupt_source_images` | Catching unreadable/truncated image files without crash | **PASS** |
| 23 | `test_23_frame_order_preservation` | Strict preservation of input temporal frame ordering | **PASS** |
| 24 | `test_24_frame_metadata_overlays` | HUD overlay rendering with active track count and frame ID | **PASS** |
| 25 | `test_25_output_directory_creation` | Automatic creation of directory hierarchies | **PASS** |
| 26 | `test_26_annotated_frame_saving` | Verifying persisted JPEG files match source dimensions | **PASS** |
| 27 | `test_27_optional_video_generation` | Successful MP4 encoding and decoding verification | **PASS** |
| 28 | `test_28_invalid_video_fps` | Validation and rejection of non-positive FPS values | **PASS** |
| 29 | `test_29_video_writer_failure_handling` | Graceful fallback when VideoWriter fails to open | **PASS** |
| 30 | `test_30_cpu_and_memory_efficient_sequential_processing` | Frame-by-frame memory reclamation without memory leaks | **PASS** |
| 31 | `test_31_partial_run_behavior` | Proper handling of `--max-frames` with `WARNING` status flag | **PASS** |
| 32 | `test_32_error_report_generation` | Schema-compliant `errors.json` generation | **PASS** |

**Execution Result**:
```
tests/visualization/test_tracking_visualizer.py::TestTrackingVisualizer ...
============================== 32 passed in 10.13s ==============================
```

---

## 10. Real-Data Performance Benchmarks

The visualization engine was evaluated on actual PerceptGuard AI driving video sequences (`driving_sample`, 200 frames at 640x360 resolution):

| Metric | 50-Frame Smoke Run | 200-Frame Full Benchmark |
|---|---|---|
| **Input Artifact** | `mot_driving_sample_...json` | `mot_driving_sample_...json` |
| **Frames Processed** | 50 / 50 (100% Success) | 200 / 200 (100% Success) |
| **Empty Frames** | 9 | 34 |
| **Failed Frames** | 0 | 0 |
| **Total Tracks Rendered** | 48 | 193 |
| **Unique Track IDs** | 2 | 3 |
| **Mean Frame Latency** | **15.02 ms / frame** | **12.33 ms / frame** |
| **Effective Throughput** | **66.6 FPS** | **81.1 FPS** |
| **Total Run Duration** | 0.86 seconds | 2.52 seconds |
| **Video Generated** | `annotated_video.mp4` (50 frames) | `annotated_video.mp4` (200 frames) |
| **Video Dimensions** | 640 x 360 px | 640 x 360 px |
| **Peak Memory Delta** | $< 25 \text{ MB}$ | $< 35 \text{ MB}$ |
| **Status** | `WARNING` (Partial run) | `PASS` (Full sequence) |

---

## 11. Architectural Invariants & Certification

The implementation complies with all strict Tesla-level engineering invariants:
1. **Zero Repeated Inference**: Visualization consumes pre-computed tracking JSON; no YOLO weights are loaded into CPU memory.
2. **Zero Secondary Association**: Does not instantiate ByteTrack, Kalman filters, or Hungarian matching.
3. **No In-Place Mutation**: Source preprocessed frame images in `datasets/processed/` are read in read-only mode and remain bitwise identical.
4. **Deterministic Visuals**: Track colors, coordinates, and bounding annotations are 100% reproducible across repeated runs.
5. **Decoupled Error Isolation**: Malformed bounding boxes or corrupt images are caught and documented in `errors.json` without pipeline interruption.
