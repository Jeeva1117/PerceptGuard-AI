# PerceptGuard AI – Advanced YOLO Model Configuration Architecture

## 1. Executive Summary & Architectural Overview

The **YOLO Model Configuration Subsystem** in PerceptGuard AI establishes a high-assurance, reproducible, and hardware-aware configuration layer for object detection in autonomous-driving perception pipelines.

In autonomous systems, hardcoding detection hyperparameters within inference loops is a known anti-pattern that creates technical debt, impairs parameter sweeping, and risks safety regression. This subsystem decouples configuration from implementation, enforcing strict boundary checks, stride compliance, CPU thread allocation, class filtering, and execution reproducibility.

```
                    ┌────────────────────────────┐
                    │    config/yolo_config.yaml │
                    └─────────────┬──────────────┘
                                  │ (safe_load & schema parse)
                                  ▼
                    ┌────────────────────────────┐
                    │      YOLOConfig Dataclass  │
                    │  ├── ModelConfig           │
                    │  ├── RuntimeConfig         │
                    │  ├── InferenceConfig       │
                    │  ├── PerformanceConfig     │
                    │  └── ValidationConfig      │
                    └─────────────┬──────────────┘
                                  │ validate()
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │  Strict Validation Engine                        │
        │  ├── Stride Conformity (imgsz % 32 == 0)         │
        │  ├── Numerical Bounds (conf, iou ∈ (0, 1])       │
        │  ├── CPU Safety (FP32 forced, CUDA guard)        │
        │  ├── Class Bounds (COCO IDs [0..79], unique)     │
        │  └── Presets (balanced, high_precision, etc.)    │
        └─────────────────────────┬────────────────────────┘
                                  │ apply_runtime_optimizations()
                                  ▼
                    ┌────────────────────────────┐
                    │      YOLOModelLoader       │
                    │  (Lazy Load & CPU Predict) │
                    └────────────────────────────┘
```

---

## 2. Configuration Schema & Hierarchy

Configurations are stored in `config/yolo_config.yaml` structured across 5 core functional domains:

```yaml
version: "1.0.0"

# 1. Model Checkpoint & Architecture Definition
model:
  name: "yolo11n"
  architecture: "yolo11"
  weights_path: "models/yolo/yolo11n.pt"
  task: "detect"
  pretrained: true

# 2. Runtime Compute & Reproducibility Settings
runtime:
  device: "cpu"
  half_precision: false
  deterministic: true
  seed: 42

# 3. Detection & Inference Hyperparameters
inference:
  image_size: 640
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
  agnostic_nms: false
  classes: null
  verbose: false

# 4. Performance & Hardware Constraints
performance:
  warmup_iterations: 1
  max_allowed_latency_ms: 500.0
  batch_size: 1
  cpu_threads: null

# 5. Validation Boundaries & Platform Guardrails
validation:
  strict_mode: true
  validate_weights: true
  validate_parameters: true
  min_confidence: 0.01
  max_confidence: 1.0
  min_iou: 0.01
  max_iou: 1.0
  min_image_size: 32
  max_image_size: 1920
  supported_tasks:
    - "detect"
  supported_devices:
    - "cpu"
  supported_models:
    - "yolo11n"
    - "yolo11s"
    - "yolo11m"
    - "yolov8n"
    - "yolov8s"
```

---

## 3. Parameter Descriptions & Engineering Bounds

| Parameter | Type | Default | Valid Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model.name` | `str` | `"yolo11n"` | Must match supported model list | Identifier for model variant |
| `model.weights_path` | `str`/`Path` | `"models/yolo/yolo11n.pt"` | Existing file path | Local filesystem path to model weights |
| `model.task` | `str` | `"detect"` | Exactly `"detect"` | Vision task type (2D bounding box detection) |
| `runtime.device` | `str` | `"cpu"` | `"cpu"` (or `"cuda"` if hardware present) | Compute device for inference |
| `runtime.half_precision`| `bool` | `false` | Must be `false` on CPU | Disables FP16 convolutions on CPU to avoid crashes |
| `runtime.seed` | `int` | `42` | $[0, 2^{32}-1]$ | Seed for deterministic PyTorch and NumPy operations |
| `inference.image_size` | `int` | `640` | $[32, 1920]$, multiple of 32 | Square letterbox image input dimension |
| `inference.confidence_threshold` | `float` | `0.25` | $(0.0, 1.0]$ | Confidence score floor for object retention |
| `inference.iou_threshold`| `float` | `0.45` | $(0.0, 1.0]$ | NMS Intersection-over-Union overlap cutoff |
| `inference.max_detections` | `int` | `100` | $\ge 1$, integer | Maximum bounding boxes retained per frame |
| `inference.agnostic_nms`| `bool` | `false` | `true` or `false` | Class-agnostic non-maximum suppression |
| `inference.classes` | `List[int]` | `null` | Unique list of integers in $[0, 79]$ | Optional COCO class filter (e.g. `[0, 2, 7]`) |
| `performance.cpu_threads`| `int` | `null` | $\ge 1$ or `null` | Explicit PyTorch CPU thread count override |
| `performance.max_allowed_latency_ms` | `float` | `500.0` | $> 0.0$ | Timeout safety threshold for single-frame inference |

---

## 4. Parameter Validation Engine

Parameter validation is implemented in `YOLOConfig.validate()` and enforces the following fail-safe criteria:

1. **Downsampling Stride Conformity:**
   YOLO architectures utilize 5 downsampling stages in the backbone/SPPF, requiring input tensor dimensions to be exact multiples of 32 ($2^5$). Non-stride dimensions (e.g. 645) trigger a `YOLOConfigError`.
2. **Strict Numerical Bounds:**
   Confidence thresholds, IoU thresholds, and max detections are strictly type-checked to reject non-numerics, booleans (`isinstance(True, int)` trap), NaNs, infinities, and out-of-range values.
3. **Hardware & Precision Policy:**
   CPU execution enforces single-precision (FP32). Passing `half_precision: true` while `device: cpu` raises `YOLOConfigError` before any model initialization can occur.
4. **COCO Class Filtering Integrity:**
   If specified, class IDs must be non-negative integers $\le 79$ (standard COCO-80 maximum index). Duplicate class IDs (e.g., `[2, 5, 2]`) are rejected.
5. **Non-Destructive Weight Verification:**
   When `validate_weights: true`, existence, regular file status, and non-zero byte size are verified without mutating or downloading weights.

---

## 5. Perception Presets Catalog

For rapid operational switching across perception scenarios, PerceptGuard AI provides 4 documented presets:

| Preset Name | Confidence | IoU | Image Size | Max Det | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `balanced` *(Default)* | `0.25` | `0.45` | `640` | `100` | Standard urban driving perception |
| `high_precision` | `0.50` | `0.50` | `640` | `50` | False-positive reduction, bright highway conditions |
| `high_recall` | `0.10` | `0.40` | `640` | `200` | Nighttime, dense traffic, small object detection |
| `cpu_lightweight` | `0.30` | `0.50` | `320` | `50` | Constrained CPU nodes (~23 ms latency, >40 FPS) |

---

## 6. Programmatic & CLI Usage

### 6.1 Programmatic API
```python
from src.detection.yolo_config import load_yolo_config, YOLOConfig
from src.detection.yolo_model_loader import YOLOModelLoader

# 1. Load default validated configuration
config = load_yolo_config()

# 2. Apply a perception preset
high_precision_cfg = config.with_preset("high_precision")

# 3. Apply class filtering (vehicles only: car=2, bus=5, truck=7)
high_precision_cfg.inference.classes = [2, 5, 7]
high_precision_cfg.validate()

# 4. Initialize loader with configuration
loader = YOLOModelLoader(config=high_precision_cfg)
results = loader.predict_single_frame("datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg")
```

### 6.2 CLI Verification & Preset Testing
```powershell
# Run verification with default configuration
python -m src.detection.yolo_model_loader --verify

# Run verification with a specific preset
python -m src.detection.yolo_model_loader --verify --preset high_precision
python -m src.detection.yolo_model_loader --verify --preset cpu_lightweight
```

---

## 7. Automated Test Suite

Configuration tests are implemented in `tests/detection/test_yolo_config.py` and `tests/test_yolo_config.py` (23 tests):

- **Schema & File Parsing:** Valid YAML loading, missing file, malformed syntax, non-dict root.
- **Boundary Verification:** Confidence bounds, IoU bounds, image size bounds, stride of 32 constraint, max detections.
- **Hardware & Device Safety:** CPU device check, CUDA unavailable trap, FP16 CPU rejection.
- **Class Filtering:** Negative IDs, out-of-range IDs, duplicate class IDs.
- **Serialization & Reproducibility:** `to_dict()`, `to_yaml()`, `to_metadata_dict()`.
- **Presets & Loader Integration:** Preset overrides, `YOLOModelLoader` compatibility, real weight integration, and live driving frame inference.

```powershell
python -m pytest tests/detection/test_yolo_config.py -v
```

---

## 8. Known Limitations & Research Boundary

1. **Hardware Specificity:** Real-time throughput (>30 FPS) at 640px requires GPU acceleration; CPU throughput achieves ~20.6 FPS on 640px and ~43.3 FPS on 320px (`cpu_lightweight`).
2. **COCO Class Mapping:** Pretrained classes are restricted to the 80 COCO categories; domain-specific autonomous driving classes (e.g., emergency vehicles, construction barriers) will be addressed in future fine-tuning stages.
