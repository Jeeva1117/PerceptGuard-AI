# PerceptGuard AI – Single Image Object Detection Engine

## 1. Executive Overview & Technical Scope

This document specifies the design, interface contracts, validation protocols, and operational workflows for the **Single Image Object Detection Engine** (`src/detection/single_image_detector.py`) in **PerceptGuard AI**.

As the core inference foundation for frame-level autonomous perception, this module is engineered to accept a single camera frame or driving image, validate input integrity, reuse the pre-initialized YOLO model loader, execute deterministic CPU inference, extract and calibrate 2D bounding boxes and class semantics, profile operational latencies, and optionally persist versioned, machine-readable JSON detection records.

---

## 2. System Architecture & Component Interaction

The `SingleImageDetector` decouples input acquisition, model inference, and output serialization into a cohesive pipeline:

```
                            ┌────────────────────────────┐
                            │    Input Image (.jpg/.png) │
                            └─────────────┬──────────────┘
                                          │
                                          ▼
                            ┌────────────────────────────┐
                            │     Image Validation       │
                            │  - Path & Existence check  │
                            │  - File size & extension   │
                            │  - In-memory OpenCV decode │
                            │  - Color space BGR standard│
                            └─────────────┬──────────────┘
                                          │ (Decoded BGR matrix)
                                          ▼
                            ┌────────────────────────────┐
                            │    YOLO Inference Engine   │
                            │  - Reuses YOLOModelLoader  │
                            │  - Applies YOLOConfig      │
                            │  - Single CPU pass (FP32)  │
                            │  - Ultralytics predict()   │
                            └─────────────┬──────────────┘
                                          │ (Raw detection tensors)
                                          ▼
                            ┌────────────────────────────┐
                            │  Extraction & Calibration  │
                            │  - xyxy coordinate clamp   │
                            │  - Confidence boundary     │
                            │  - Class catalog mapping   │
                            │  - Zero-detection safety   │
                            └─────────────┬──────────────┘
                                          │
                                          ▼
                            ┌────────────────────────────┐
                            │  Structured Result Object  │
                            │  - SingleImageDetection    │
                            │    Result (Schema v1.0)    │
                            │  - Atomic JSON persistence │
                            └────────────────────────────┘
```

---

## 3. Input Image Validation & Format Standardization

Input validation is encapsulated in `SingleImageDetector.validate_image()`:

| Check | Failure Condition | Raised Exception |
| :--- | :--- | :--- |
| **Path Check** | Non-existent image file | `ImageNotFoundError` |
| **File Type** | Directory passed as image | `ImageValidationError` |
| **Extension** | Extension not in `{'.jpg', '.jpeg', '.png'}` | `UnsupportedImageFormatError` |
| **File Size** | Byte length is 0 (empty file) | `ImageValidationError` |
| **Decoding** | Non-image bytes or truncated stream | `ImageDecodeError` |
| **Dimensions** | Width $\le 0$ or Height $\le 0$ | `ImageValidationError` |
| **Channels** | Channels $\notin \{1, 3, 4\}$ | `ImageValidationError` |

### 3.1 Color Space Standardization
- **Grayscale (1-channel):** Automatically normalized to 3-channel BGR via `cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)` to satisfy YOLO tensor input requirements.
- **Alpha Transparency (4-channel BGRA):** Converted to 3-channel BGR via `cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)`.
- **Standard (3-channel BGR):** Passed directly without modification.

---

## 4. Model & Configuration Integration

The detector strictly reuses the pre-existing, verified perception infrastructure:
- **Model Architecture:** YOLO11 Nano (`yolo11n.pt`, 2.6M parameters, 11.6 GFLOPs).
- **Execution Device:** Default device `"cpu"`, enforcing FP32 single precision.
- **Loader Reuse:** If an existing `YOLOModelLoader` instance is provided, it is injected directly, avoiding duplicate weight deserialization or memory leaks.
- **Parameter Propagation:** Hyperparameters defined in `config/yolo_config.yaml` (`confidence_threshold: 0.25`, `iou_threshold: 0.45`, `image_size: 640`, `max_detections: 100`, `agnostic_nms: false`, `classes: null`) are propagated into the inference call.

---

## 5. Detection Extraction, Calibration & Coordinates

For each detected object, bounding box coordinates are calibrated:
- **Coordinate Convention:** $[x_1, y_1, x_2, y_2]$ representing top-left $(x_1, y_1)$ and bottom-right $(x_2, y_2)$.
- **Finiteness Guarantee:** All coordinates are asserted finite (`math.isfinite`).
- **Boundary Clamping:** Coordinates are clamped within the physical image boundary $[0.0, \text{width}]$ and $[0.0, \text{height}]$.
- **Degenerate Box Rejection:** Boxes where $x_2 \le x_1$ or $y_2 \le y_1$ after clamping are discarded with diagnostic logging.
- **Zero Detections:** Frames with zero detected objects are honestly classified as `processing_status: "success"` with `detection_count: 0` and `detections: []`.

---

## 6. Standardized Detection Response Schema (v1.0)

Detection outputs are returned as a `SingleImageDetectionResult` dataclass and can be serialized to RFC 8259-compliant JSON:

```json
{
  "schema_version": "1.0",
  "image": {
    "filename": "frame_000001.jpg",
    "image_path": "C:\\Users\\acer\\Desktop\\PerceptGuard-AI\\datasets\\processed\\driving_sample\\preprocessed_frames\\frame_000001.jpg",
    "width": 640,
    "height": 360,
    "channels": 3
  },
  "model": {
    "name": "yolo11n",
    "task": "detect",
    "device": "cpu"
  },
  "inference": {
    "image_size": 640,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "max_detections": 100,
    "inference_time_ms": 45.50
  },
  "performance": {
    "image_decode_time_ms": 2.55,
    "inference_time_ms": 45.50,
    "serialization_time_ms": 0.50,
    "total_time_ms": 48.68
  },
  "result": {
    "processing_status": "success",
    "detection_count": 2,
    "detections": [
      {
        "detection_index": 0,
        "class_id": 9,
        "class_name": "traffic light",
        "confidence": 0.4381,
        "bbox_xyxy": [219.5, 238.1, 276.4, 359.1]
      },
      {
        "detection_index": 1,
        "class_id": 7,
        "class_name": "truck",
        "confidence": 0.2961,
        "bbox_xyxy": [219.5, 1.2, 350.9, 355.9]
      }
    ]
  }
}
```

---

## 7. Performance Profiling Instrumentation

Timing is captured using Python's monotonic high-resolution timer (`time.perf_counter()`):
- `image_decode_time_ms`: File reading and in-memory OpenCV matrix construction.
- `inference_time_ms`: Direct YOLO neural network forward pass and NMS post-processing.
- `serialization_time_ms`: JSON string generation and atomic file write.
- `total_time_ms`: Complete lifecycle elapsed time.

### Benchmark Telemetry on Intel Core i3 (CPU):
- **Cold-Start Load + Inference:** $\approx 3,640\text{ ms}$ (includes initial weights parsing and graph construction)
- **Warm Inference Latency:** $\approx 45.5\text{ ms}$ ($\approx 22.0\text{ FPS}$)
- **Total Warm Execution:** $\approx 48.7\text{ ms}$

---

## 8. CLI & Programmatic Usage

### 8.1 CLI Usage
```powershell
# Standard detection run
python -m src.detection.single_image_detector --image datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg

# Save output JSON with explicit overwrite
python -m src.detection.single_image_detector --image datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg --save-json --overwrite

# Execute with specific perception preset
python -m src.detection.single_image_detector --image datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg --preset high_precision
```

### 8.2 Programmatic Usage
```python
from src.detection.single_image_detector import SingleImageDetector

# Initialize detector
detector = SingleImageDetector()

# Run detection
result = detector.detect(
    image_path="datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg",
    save_json=True,
    overwrite=True,
)

print(f"Detections found: {result.detection_count}")
for obj in result.detections:
    print(f" - {obj.class_name} (conf={obj.confidence}): {obj.bbox_xyxy}")
```

---

## 9. Automated Testing & Verification

Implemented in [tests/detection/test_single_image_detector.py](file:///c:/Users/acer/Desktop/PerceptGuard-AI/tests/detection/test_single_image_detector.py) covering 26 unit and integration tests:
- Image validation (valid JPEG, PNG, missing paths, bad extensions, corrupt files, empty files).
- Dimension and channel standardization (grayscale, RGBA).
- Loader injection and reuse.
- Single, multiple, and zero detection handling.
- Coordinate boundary clamping and non-finite rejection.
- Monotonic latency measurement.
- JSON serialization, atomic writes, and overwrite protection.
- Real-model integration test on genuine driving frame `frame_000001.jpg`.

---

## 10. Known Limitations & Research Boundaries

1. **Static 2D Boxes:** Single image detection produces instantaneous spatial bounding boxes; temporal associations and velocity vectors require multi-object tracking (ByteTrack in Day 3 Task 5).
2. **Confidence Calibration:** Detector confidence reflects categorical softmax probability, not future localization reliability ($R_{t+h}$). Future tasks will train an LSTM meta-forecaster on extracted feature sequences.

