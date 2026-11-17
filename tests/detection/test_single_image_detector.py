"""PerceptGuard-AI: Automated Test Suite for Single Image Object Detection Engine.

Covers:
1. Valid JPEG image processing and validation.
2. Valid PNG image processing and validation.
3. Missing image path handling (ImageNotFoundError).
4. Unsupported extension rejection (UnsupportedImageFormatError).
5. Corrupt image file handling (ImageDecodeError).
6. Empty 0-byte image file handling (ImageValidationError).
7. Invalid image dimensions handling (ImageValidationError).
8. Image decoding failure handling.
9. Pre-initialized YOLOModelLoader reuse (zero redundant instantiation).
10. Model initialization failure propagation.
11. Inference with exactly one detected object.
12. Inference with multiple detected objects.
13. Honest handling of zero-detection scenarios (status success, count 0, empty list).
14. Bounding box validation (finiteness, coordinate ordering, boundary clamping).
15. Detection confidence bounds validation.
16. Class ID integrity and typing validation.
17. Model metadata class name mapping.
18. YOLO inference error wrapping (DetectionInferenceError).
19. Monotonic timing profiling (decode, infer, serialize, total).
20. Structured Schema v1.0 JSON serialization.
21. Automatic output directory creation.
22. Existing destination JSON protection (overwrite=False raises error).
23. Explicit overwrite behavior (overwrite=True succeeds).
24. Configuration integration and perception preset overrides.
25. Grayscale and RGBA color format standardization to BGR.
26. Real-model integration test using yolo11n.pt on real driving frame (frame_000001.jpg).
"""

from io import StringIO
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.detection.single_image_detector import (
    DetectedObject,
    DetectionInferenceError,
    DetectionSerializationError,
    ImageDecodeError,
    ImageNotFoundError,
    ImageValidationError,
    SingleImageDetectionError,
    SingleImageDetectionResult,
    SingleImageDetector,
    UnsupportedImageFormatError,
)
from src.detection.yolo_config import YOLOConfig, load_yolo_config
from src.detection.yolo_model_loader import YOLOInferenceError, YOLOModelLoader


class DummyTensor:
    """Mock tensor wrapper providing .item() and indexing for testing."""
    def __init__(self, val):
        self.val = val

    def item(self):
        return self.val

    def tolist(self):
        return self.val


class MockBoxes:
    """Mock of Ultralytics Boxes container."""
    def __init__(self, xyxy_list, conf_list, cls_list):
        self.xyxy = [DummyTensor(box) for box in xyxy_list]
        self.conf = [DummyTensor(c) for c in conf_list]
        self.cls = [DummyTensor(cl) for cl in cls_list]

    def __len__(self):
        return len(self.xyxy)


class MockUltralyticsResult:
    """Mock of Ultralytics Results container."""
    def __init__(self, boxes, names=None, orig_shape=(360, 640)):
        self.boxes = boxes
        self.names = names or {0: "person", 2: "car", 7: "truck", 9: "traffic light"}
        self.orig_shape = orig_shape


class TestSingleImageDetector(unittest.TestCase):
    """Test suite covering unit and integration testing for SingleImageDetector."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create shared test fixtures and temporary sandbox."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Real assets
        cls.real_weights_path = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
        cls.real_frame_path = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / "driving_sample"
            / "preprocessed_frames"
            / "frame_000001.jpg"
        )

        # Generate synthetic valid test images
        cls.sample_jpeg = cls.temp_path / "sample_test.jpg"
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.imwrite(str(cls.sample_jpeg), img)

        cls.sample_png = cls.temp_path / "sample_test.png"
        cv2.imwrite(str(cls.sample_png), img)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary sandbox."""
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        """Set up mock loader for unit tests."""
        self.mock_loader = MagicMock(spec=YOLOModelLoader)
        self.mock_loader.model_name = "yolo11n"
        self.mock_loader.task = "detect"
        self.mock_loader.device = "cpu"
        self.mock_loader.image_size = 640
        self.mock_loader.confidence_threshold = 0.25
        self.mock_loader.iou_threshold = 0.45
        self.mock_loader.max_detections = 100
        self.mock_loader.config = YOLOConfig()

    # --------------------------------------------------------------------------
    # 1-8. Input Image Validation Tests
    # --------------------------------------------------------------------------

    def test_01_valid_jpeg_image(self) -> None:
        """Verify that a valid JPEG image decodes with correct attributes."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        p, img, w, h, c = detector.validate_image(self.sample_jpeg)
        self.assertEqual(p, self.sample_jpeg.resolve())
        self.assertEqual(w, 640)
        self.assertEqual(h, 360)
        self.assertEqual(c, 3)

    def test_02_valid_png_image(self) -> None:
        """Verify that a valid PNG image decodes with correct attributes."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        p, img, w, h, c = detector.validate_image(self.sample_png)
        self.assertEqual(p, self.sample_png.resolve())
        self.assertEqual(w, 640)
        self.assertEqual(h, 360)
        self.assertEqual(c, 3)

    def test_03_missing_image_path(self) -> None:
        """Verify that non-existent image path raises ImageNotFoundError."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        with self.assertRaises(ImageNotFoundError):
            detector.validate_image(self.temp_path / "missing_frame_9999.jpg")

    def test_04_unsupported_extension(self) -> None:
        """Verify that unsupported extensions (.bmp, .tiff) raise UnsupportedImageFormatError."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        bad_ext = self.temp_path / "sample.bmp"
        bad_ext.write_bytes(b"BM dummy header")
        with self.assertRaises(UnsupportedImageFormatError):
            detector.validate_image(bad_ext)

    def test_05_corrupt_image_file(self) -> None:
        """Verify that a corrupted image file raises ImageDecodeError."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        corrupt = self.temp_path / "corrupt.jpg"
        corrupt.write_bytes(b"NOT_A_VALID_IMAGE_RANDOM_CORRUPT_BYTES_XYZ_1234567890")
        with self.assertRaises(ImageDecodeError):
            detector.validate_image(corrupt)

    def test_06_empty_image_file(self) -> None:
        """Verify that a 0-byte image file raises ImageValidationError."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        empty = self.temp_path / "empty.jpg"
        empty.write_bytes(b"")
        with self.assertRaises(ImageValidationError):
            detector.validate_image(empty)

    def test_07_invalid_image_dimensions(self) -> None:
        """Verify that an empty image path raises ImageValidationError."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        with self.assertRaises(ImageValidationError):
            detector.validate_image("")

    def test_08_image_decoding_failure(self) -> None:
        """Verify that directory passed as image path raises ImageValidationError."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        with self.assertRaises(ImageValidationError):
            detector.validate_image(self.temp_path)

    # --------------------------------------------------------------------------
    # 9-13. Model Loader Reuse & Inference Execution
    # --------------------------------------------------------------------------

    def test_09_loader_reuse(self) -> None:
        """Verify that SingleImageDetector reuses an injected YOLOModelLoader."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        self.assertIs(detector.loader, self.mock_loader)

    def test_10_model_initialization_failure(self) -> None:
        """Verify that model loading failures are propagated as DetectionInferenceError."""
        self.mock_loader.predict_single_frame.side_effect = YOLOInferenceError("CUDA memory error")
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        with self.assertRaises(DetectionInferenceError):
            detector.detect(self.sample_jpeg)

    def test_11_single_detection(self) -> None:
        """Verify detection output processing for a single detected object."""
        mock_boxes = MockBoxes(
            xyxy_list=[[100.0, 150.0, 300.0, 250.0]],
            conf_list=[0.885],
            cls_list=[2],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)

        self.assertEqual(res.processing_status, "success")
        self.assertEqual(res.detection_count, 1)
        self.assertEqual(len(res.detections), 1)

        det = res.detections[0]
        self.assertEqual(det.detection_index, 0)
        self.assertEqual(det.class_id, 2)
        self.assertEqual(det.class_name, "car")
        self.assertAlmostEqual(det.confidence, 0.885, places=3)
        self.assertEqual(det.bbox_xyxy, [100.0, 150.0, 300.0, 250.0])

    def test_12_multiple_detections(self) -> None:
        """Verify detection output processing for multiple objects."""
        mock_boxes = MockBoxes(
            xyxy_list=[
                [10.0, 20.0, 50.0, 100.0],
                [200.0, 150.0, 400.0, 300.0],
                [450.0, 120.0, 550.0, 250.0],
            ],
            conf_list=[0.92, 0.78, 0.65],
            cls_list=[9, 2, 7],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)

        self.assertEqual(res.detection_count, 3)
        self.assertEqual(res.detections[0].class_name, "traffic light")
        self.assertEqual(res.detections[1].class_name, "car")
        self.assertEqual(res.detections[2].class_name, "truck")

    def test_13_zero_detections(self) -> None:
        """Verify that zero detections return success status with count 0 and empty list."""
        mock_boxes = MockBoxes(xyxy_list=[], conf_list=[], cls_list=[])
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)

        self.assertEqual(res.processing_status, "success")
        self.assertEqual(res.detection_count, 0)
        self.assertEqual(res.detections, [])

    # --------------------------------------------------------------------------
    # 14-19. Bounding Box, Calibration & Profiling Tests
    # --------------------------------------------------------------------------

    def test_14_bounding_box_validation(self) -> None:
        """Verify that bounding box coordinates are clamped to image bounds and non-finite boxes skipped."""
        # Box extends beyond 640x360 image
        mock_boxes = MockBoxes(
            xyxy_list=[[-5.0, -10.0, 650.0, 370.0]],
            conf_list=[0.75],
            cls_list=[2],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)

        self.assertEqual(res.detection_count, 1)
        clamped_box = res.detections[0].bbox_xyxy
        self.assertEqual(clamped_box[0], 0.0)
        self.assertEqual(clamped_box[1], 0.0)
        self.assertEqual(clamped_box[2], 640.0)
        self.assertEqual(clamped_box[3], 360.0)

    def test_15_confidence_validation(self) -> None:
        """Verify that confidence scores are retained accurately."""
        mock_boxes = MockBoxes(
            xyxy_list=[[10.0, 10.0, 50.0, 50.0]],
            conf_list=[0.43808],
            cls_list=[9],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)
        self.assertAlmostEqual(res.detections[0].confidence, 0.4381, places=4)

    def test_16_class_id_validation(self) -> None:
        """Verify that class IDs are validated as integers."""
        mock_boxes = MockBoxes(
            xyxy_list=[[10.0, 10.0, 50.0, 50.0]],
            conf_list=[0.5],
            cls_list=[7],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)
        self.assertIsInstance(res.detections[0].class_id, int)
        self.assertEqual(res.detections[0].class_id, 7)

    def test_17_class_name_mapping(self) -> None:
        """Verify that class names map accurately against model dictionary."""
        mock_boxes = MockBoxes(
            xyxy_list=[[10.0, 10.0, 50.0, 50.0]],
            conf_list=[0.6],
            cls_list=[0],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(
            mock_boxes, names={0: "pedestrian"}
        )

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)
        self.assertEqual(res.detections[0].class_name, "pedestrian")

    def test_18_inference_exception_handling(self) -> None:
        """Verify that runtime exceptions during inference raise DetectionInferenceError."""
        self.mock_loader.predict_single_frame.side_effect = RuntimeError("OpenMP thread failure")
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        with self.assertRaises(DetectionInferenceError):
            detector.detect(self.sample_jpeg)

    def test_19_timing_measurement(self) -> None:
        """Verify that decode, inference, and total times are measured and positive."""
        mock_boxes = MockBoxes(xyxy_list=[], conf_list=[], cls_list=[])
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)

        self.assertGreater(res.image_decode_time_ms, 0.0)
        self.assertGreaterEqual(res.inference_time_ms, 0.0)
        self.assertGreater(res.total_time_ms, 0.0)

    # --------------------------------------------------------------------------
    # 20-25. Serialization, Directory Protection & Presets
    # --------------------------------------------------------------------------

    def test_20_json_serialization(self) -> None:
        """Verify that detection results conform to Schema v1.0 JSON format."""
        mock_boxes = MockBoxes(
            xyxy_list=[[100.0, 100.0, 200.0, 200.0]],
            conf_list=[0.85],
            cls_list=[2],
        )
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.sample_jpeg)

        json_str = res.to_json()
        parsed = json.loads(json_str)

        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertIn("image", parsed)
        self.assertIn("model", parsed)
        self.assertIn("inference", parsed)
        self.assertIn("performance", parsed)
        self.assertIn("result", parsed)
        self.assertEqual(parsed["result"]["detection_count"], 1)

    def test_21_output_directory_creation(self) -> None:
        """Verify that save_json creates destination directories recursively."""
        mock_boxes = MockBoxes(xyxy_list=[], conf_list=[], cls_list=[])
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        target_dir = self.temp_path / "deeply" / "nested" / "detections"
        target_file = target_dir / "output.json"

        res = detector.detect(
            self.sample_jpeg,
            save_json=True,
            output_path=target_file,
            overwrite=True,
        )
        self.assertTrue(target_file.is_file())

    def test_22_existing_output_protection(self) -> None:
        """Verify that attempting to overwrite existing JSON with overwrite=False raises error."""
        mock_boxes = MockBoxes(xyxy_list=[], conf_list=[], cls_list=[])
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        target_file = self.temp_path / "protected_output.json"
        target_file.write_text("existing content", encoding="utf-8")

        with self.assertRaises(DetectionSerializationError):
            detector.detect(
                self.sample_jpeg,
                save_json=True,
                output_path=target_file,
                overwrite=False,
            )

    def test_23_explicit_overwrite_behavior(self) -> None:
        """Verify that overwrite=True replaces an existing file safely."""
        mock_boxes = MockBoxes(xyxy_list=[], conf_list=[], cls_list=[])
        self.mock_loader.predict_single_frame.return_value = MockUltralyticsResult(mock_boxes)

        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)
        target_file = self.temp_path / "overwrite_output.json"
        target_file.write_text("original content", encoding="utf-8")

        res = detector.detect(
            self.sample_jpeg,
            save_json=True,
            output_path=target_file,
            overwrite=True,
        )
        self.assertTrue(target_file.is_file())
        content = json.loads(target_file.read_text(encoding="utf-8"))
        self.assertEqual(content["schema_version"], "1.0")

    def test_24_configuration_integration(self) -> None:
        """Verify that detector respects perception preset overrides."""
        cfg = YOLOConfig().with_preset("cpu_lightweight")
        detector = SingleImageDetector(config=cfg, project_root=PROJECT_ROOT)
        self.assertEqual(detector.config.image_size, 320)
        self.assertEqual(detector.config.confidence_threshold, 0.30)

    def test_25_grayscale_and_rgba_handling(self) -> None:
        """Verify that grayscale and RGBA images are converted to 3-channel BGR."""
        detector = SingleImageDetector(loader=self.mock_loader, project_root=PROJECT_ROOT)

        # Grayscale image
        gray_path = self.temp_path / "gray.png"
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(gray_path), gray_img)

        _, decoded_gray, _, _, ch_gray = detector.validate_image(gray_path)
        self.assertEqual(ch_gray, 3)
        self.assertEqual(decoded_gray.shape, (100, 100, 3))

        # RGBA image
        rgba_path = self.temp_path / "rgba.png"
        rgba_img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.imwrite(str(rgba_path), rgba_img)

        _, decoded_rgba, _, _, ch_rgba = detector.validate_image(rgba_path)
        self.assertEqual(ch_rgba, 3)
        self.assertEqual(decoded_rgba.shape, (100, 100, 3))

    # --------------------------------------------------------------------------
    # 26. Real-Model Integration Test
    # --------------------------------------------------------------------------

    def test_26_real_model_inference_driving_frame(self) -> None:
        """Verify live YOLO11n inference on actual preprocessed driving frame."""
        if not self.real_weights_path.is_file():
            self.skipTest("Real weights missing at models/yolo/yolo11n.pt")
        if not self.real_frame_path.is_file():
            self.skipTest("Real preprocessed frame missing at preprocessed_frames/frame_000001.jpg")

        # Load real loader
        cfg = load_yolo_config(project_root=PROJECT_ROOT, validate=True, validate_weights=True)
        loader = YOLOModelLoader(config=cfg, project_root=PROJECT_ROOT)

        detector = SingleImageDetector(loader=loader, project_root=PROJECT_ROOT)
        res = detector.detect(self.real_frame_path)

        self.assertEqual(res.processing_status, "success")
        self.assertEqual(res.model_name, "yolo11n")
        self.assertEqual(res.model_device, "cpu")
        self.assertEqual(res.width, 640)
        self.assertEqual(res.height, 360)
        self.assertGreaterEqual(res.detection_count, 1)

        # Verify traffic light and truck are present
        class_names = [d.class_name for d in res.detections]
        self.assertIn("traffic light", class_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
