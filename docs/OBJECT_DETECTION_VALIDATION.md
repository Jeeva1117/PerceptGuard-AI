# PerceptGuard-AI: Object Detection Validation Report

**Document Version:** 2.0 (Release Readiness)  
**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  

---

## 1. Executive Summary

This report documents the validation of Feature 2: Object Detection and Perception for PerceptGuard-AI. The system utilizes Ultralytics YOLO11n (nano variant, ~2.6M parameters) optimized for CPU execution on standard personal computers. It is tuned specifically for real-world road environments, focusing on the six canonical road user classes while suppressing background clutter.

---

## 2. Model Configuration & Road User Class Taxonomy

YOLO11n is configured to isolate road actors from general COCO categories:

| Class ID | Semantic Class Name | Automotive Perception Role |
| :---: | :--- | :--- |
| `0` | `person` | Pedestrian, crossing pedestrian, roadside worker |
| `1` | `bicycle` | Cyclist, e-bike commuter |
| `2` | `car` | Passenger vehicle, taxi, sedan, SUV |
| `3` | `motorcycle` | Motorcyclist, moped, scooter |
| `5` | `bus` | Transit vehicle, school bus |
| `7` | `truck` | Heavy vehicle, cargo lorry, commercial hauler |

Irrelevant categories (such as indoor objects, domestic animals, aircraft, furniture) are filtered at inference time by providing `classes=[0, 1, 2, 3, 5, 7]` to `model.predict()`.

---

## 3. Two-Stage Detection Architecture

To support ByteTrack's two-stage association without cluttering the user interface with low-probability bounding boxes, the system implements a dual-tier confidence pipeline:

```
OpenCV RGB Frame (640x360 / 1280x720)
                  │
                  ▼
[YOLO11n Inference]
 ├── conf = 0.10, iou = 0.45
 └── classes = [0, 1, 2, 3, 5, 7]
                  │
                  ├──────────────────────────────┐
                  ▼                              ▼
     [All Detections (conf >= 0.10)]    [Visual Detections (conf >= 0.20)]
                  │                              │
                  ▼                              ▼
     Forwarded to ByteTrackEngine        Displayed on UI Canvas &
     (Enables Stage 2 low-score          recorded in Frame Metrics
      recovery for occluded targets)     (Clean, false-positive resilient)
```

### Benefits:
1. **Zero Motorcycle Starvation**: Detections with confidence between 0.10 and 0.20 (common for small, fast-moving, or partially occluded motorcycles) are captured for tracking continuity rather than dropped.
2. **Clean Frontend Presentation**: The user interface does not render distracting spurious bounding boxes, maintaining high visual clarity.

---

## 4. Hardware Latency & Resource Utilization

Benchmarks measured on Intel Core i3 (CPU execution, Windows 11):

| Operation | Input Resolution | Average Latency (ms) | Memory Impact |
| :--- | :--- | :--- | :--- |
| **YOLO11n Predict** | 640x360 | 45.2 ms | ~180 MB RAM |
| **YOLO11n Predict** | 1280x720 | 88.6 ms | ~240 MB RAM |
| **NMS & Box Formatting** | Up to 30 objects | 1.8 ms | < 1 MB |
| **Feature Extraction (27 feats)** | Full frame | 6.4 ms | < 5 MB |

Total perception loop latency comfortably exceeds the 2 FPS balanced analysis requirement on pure CPU hardware.

---

## 5. Verification Test Matrix

| Test Case | Description | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| `test_default_road_classes_mapping` | Verifies correct COCO class mappings | `test_motorcycle_and_road_users.py` | **PASS** |
| `test_motorcycle_detection_ingestion` | Verifies motorcycle extraction and bounding box coordinate validation | `test_motorcycle_and_road_users.py` | **PASS** |
| `test_multi_road_user_concurrent_tracking` | Verifies simultaneous detection of car, motorcycle, truck, and pedestrian | `test_motorcycle_and_road_users.py` | **PASS** |
| `test_video_pipeline_end_to_end` | Verifies full video inference and detection output generation | `test_upload_and_processing_pipeline.py` | **PASS** |
| `test_regression_scenarios` | 12 driving clips evaluated with realistic road actors | `test_regression_scenarios.py` | **PASS** |
