# PerceptGuard AI – Advanced YOLO Environment & Pretrained Model Setup

## 1. Executive Overview & Technical Scope

This document specifies the architecture, hardware adaptation, model selection, configuration schema, verification protocols, and operational guidelines for the **YOLO Object Detection Environment** in **PerceptGuard AI**.

As the perception foundation for PerceptGuard AI, the object detection layer is tasked with generating real-time bounding boxes, class semantics, and confidence scores from sequential driving video frames. These detections serve as the primary upstream inputs for multi-object tracking (ByteTrack) and predictive reliability forecasting (LSTM/GRU).

In accordance with strict safety-conscious autonomous driving engineering standards and CPU hardware constraints, this subsystem is engineered to run deterministically on commodity CPU architectures without requiring dedicated GPU acceleration.

---

## 2. Hardware Environment & Host Runtime Specifications

The detection subsystem has been profiled, deployed, and validated on the following host environment:

| Attribute | Specification | Operational Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home Single Language (64-bit AMD64) | Production Verified |
| **Processor (CPU)** | Intel Core i3 Architecture (`Intel64 Family 6 Model 190 Stepping 0`) | Verified (8 Logical Cores) |
| **System Memory (RAM)**| 8.00 GB Physical RAM | Profiled (< 300 MB Footprint) |
| **CUDA / GPU Acceleration** | Not Present (`torch.cuda.is_available() == False`) | Deterministic CPU Fallback |
| **Python Runtime** | Python 3.13.3 (64-bit AMD64) | Production Verified |
| **Deep Learning Framework** | PyTorch `2.15.0+cpu` | Official CPU Wheel Verified |
| **Computer Vision Engine** | OpenCV `5.0.0.93` | Headless BGR Matrix Decoding |
| **Detection Framework** | Ultralytics `8.4.215` (YOLO11) | Official Release Verified |
| **Scientific Numerical Stack**| NumPy `2.5.4` | C-Accelerated Vector Operations |

---

## 3. CPU-Only Execution Strategy & Architecture Rationale

Standard deep-learning computer vision pipelines assume the presence of high-throughput CUDA accelerators. However, real-world edge perception nodes, test harnesses, and cost-constrained autonomous driving workstations must often operate within CPU-only environments.

PerceptGuard AI addresses this via a **CPU-First Inference Architecture**:

### 3.1 Precision Governance (FP32 Enforcement)
- Standard PyTorch CPU backends do not provide native hardware acceleration for FP16 (half-precision) 2D convolutions unless specific AVX512-BF16 / AMX instruction sets are available. Forcing `half=True` on commodity CPUs results in runtime errors or severe emulation latency.
- The configuration rigorously enforces `half_precision: false` (standard FP32 single precision).
- Deprecated parameter flags (such as passing explicit `half=False` in Ultralytics `predict()`) are omitted to avoid engine deprecation warnings.

### 3.2 Threading & Resource Allocation
- PyTorch and OpenCV dynamically utilize multi-threaded CPU primitives (`torch.set_num_threads`, OpenMP).
- On the host 8-thread CPU, inference pipelines operate without CPU thrashing, maintaining low context-switch overhead.

### 3.3 Latency & Memory Envelope
- **Cold Load Time:** $\le 100\text{ ms}$ (measured at $84.78\text{ ms}$).
- **Warm Inference Latency:** $\approx 48.5\text{ ms}$ on $640 \times 360$ letterboxed tensors ($\approx 20.6\text{ FPS}$).
- **Memory Consumption:** Sub-300 MB inference memory resident set size (RSS), leaving ample headroom for downstream tracking and sequence forecasting buffers.

---

## 4. Pretrained Model Selection Justification

PerceptGuard AI selects **YOLO11n (`yolo11n.pt`)** as the primary detection backbone.

```
+-----------------------------------------------------------------------------------------------+
| Model      | Parameters | FLOPs (640) | Weights Size | CPU Latency (Host) | Target FPS Status |
+-----------------------------------------------------------------------------------------------+
| YOLO11n    | 2.6M       | 11.6 GFLOPs | 5.35 MB      | ~48.5 ms           | ~20.6 FPS (PASS)  |
| YOLO11s    | 9.4M       | 21.5 GFLOPs | 18.2 MB      | ~115.0 ms          | ~8.7 FPS  (FAIL)  |
| YOLO11m    | 20.1M      | 68.0 GFLOPs | 40.5 MB      | ~240.0 ms          | ~4.1 FPS  (FAIL)  |
| YOLO11x    | 56.9M      | 194.9 GFLOPs| 114.0 MB     | ~620.0 ms          | ~1.6 FPS  (FAIL)  |
+-----------------------------------------------------------------------------------------------+
```

### 4.1 Technical Justification Matrix
1. **Lightweight Parameter Envelope:** With only 2.6 million parameters, YOLO11n minimizes CPU cache misses and DRAM bus saturation on Intel Core i3 systems.
2. **COCO Pretrained Coverage:** Directly recognizes all 8 key autonomous driving traffic classes without requiring immediate retraining:
   - `person` (Class ID 0)
   - `bicycle` (Class ID 1)
   - `car` (Class ID 2)
   - `motorcycle` (Class ID 3)
   - `bus` (Class ID 5)
   - `truck` (Class ID 7)
   - `traffic light` (Class ID 9)
   - `stop sign` (Class ID 11)
3. **Reproducibility & Weight Verification:**
   - **Weight File:** `models/yolo/yolo11n.pt`
   - **Size:** $5,613,764\text{ bytes}$ ($5.35\text{ MB}$)
   - **Cryptographic Hash (SHA-256):**
     `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`

---

## 5. Configuration Architecture (`config/yolo_config.yaml`)

Configuration is fully decoupled into YAML format with strict schema validation:

```yaml
model:
  name: "yolo11n"
  weights_path: "models/yolo/yolo11n.pt"
  task: "detect"
  device: "cpu"
  half_precision: false

inference:
  image_size: 640
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
  agnostic_nms: false
  classes: null

performance:
  warmup_iterations: 1
  max_allowed_latency_ms: 500.0

dataset:
  sample_frame_path: "datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg"
```

### 5.1 Parameter Specifications
- `model.weights_path`: Relative or absolute path to local model weights.
- `model.device`: Computation device (`"cpu"` enforced for deterministic baseline).
- `model.half_precision`: Must remain `false` for CPU backends.
- `inference.image_size`: Maximum letterbox dimension ($640\text{ px}$).
- `inference.confidence_threshold`: Detection acceptance threshold ($0.25$).
- `inference.iou_threshold`: Non-Maximum Suppression (NMS) overlap threshold ($0.45$).
- `performance.max_allowed_latency_ms`: Safety timeout ceiling ($500.0\text{ ms}$).

---

## 6. Directory Governance & Weight Management

```
PerceptGuard-AI/
├── config/
│   └── yolo_config.yaml        # Tracked in Git
├── models/
│   ├── .gitkeep                # Tracked in Git
│   └── yolo/
│       ├── .gitkeep            # Tracked in Git
│       └── yolo11n.pt          # STRICTLY IGNORED in Git (.gitignore)
```

### 6.1 Data Governance Compliance
- Binary model checkpoints (`*.pt`, `*.pth`, `*.onnx`, `*.engine`) are excluded by `.gitignore` to prevent repository bloat.
- Directory structure is preserved in Git using `models/yolo/.gitkeep`.
- Automated scripts download weights securely to `models/yolo/` if missing.

---

## 7. Verification & Telemetry Command

A command-line verification utility is built directly into the detection module:

```powershell
python -m src.detection.yolo_model_loader --verify
```

### 7.1 Verified Execution Output
```text
============================================================
PERCEPTGUARD AI – YOLO ENVIRONMENT CHECK
============================================================
Python version: 3.13.3
PyTorch version: 2.15.0+cpu
Ultralytics version: 8.4.215
OpenCV version: 5.0.0
CPU information: Intel64 Family 6 Model 190 Stepping 0, GenuineIntel
CUDA available: False
Selected device: cpu
YOLO model: yolo11n
Model weights: yolo11n.pt
Model task: detect
Model loading: SUCCESS (84.78 ms)
Dataset frame: frame_000001.jpg
Frame decoding: SUCCESS (640x360 px)
Inference status: SUCCESS (48.51 ms)
Inference time: 48.51 ms
Detections returned: 2
============================================================
OVERALL STATUS: PASS
============================================================
```

---

## 8. Automated Test Suite Execution

The detection environment test suite is implemented in `tests/test_yolo_model_loader.py` and features 18 comprehensive tests:

1. **Unit Tests (15 tests):**
   - Config loading, validation traps (invalid conf, invalid device, invalid task).
   - Missing weight exception handling (`YOLOModelNotFoundError`).
   - Mocked model loading, parameter caching, and prediction error propagation.
   - Half-precision CPU safety validation.
   - Non-existent and unreadable image decoding traps.
2. **Integration Tests (3 tests):**
   - Live loading of `yolo11n.pt` on CPU.
   - Live single-frame inference on `frame_000001.jpg` ($640 \times 360$).
   - Full CLI verification pipeline execution (`--verify`).

### 8.1 Running Tests
```powershell
# Run the YOLO model loader test suite
python -m pytest tests/test_yolo_model_loader.py -v

# Run the complete project test suite (112 tests)
python -m pytest -q
```

**Result:** `112 passed in 21.06s` (100% pass rate across Day 1, Day 2, and Day 3 Task 1).

---

## 9. Troubleshooting & Common Issues

| Issue / Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| `RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'` | Attempting to use FP16 half precision on CPU. | Ensure `half_precision: false` in `yolo_config.yaml`. |
| `WARNING 'half' is deprecated...` | Passing deprecated `half=False` kwarg to Ultralytics. | Omit `half` parameter when calling `model.predict()`. |
| `YOLOModelNotFoundError: Weights file not found` | Weights have not been downloaded to `models/yolo/`. | Download `yolo11n.pt` into `models/yolo/`. |
| `YOLOInferenceError: Failed to decode image` | Frame path corrupted or invalid image format. | Verify image path and check decodability with `cv2.imread()`. |
| High CPU Latency (> 200 ms) | System running heavy background processes or wrong model. | Ensure `yolo11n.pt` (nano) is used, not `yolo11x.pt`. |

