"""PerceptGuard-AI: Unit and Integration Test Suite for YOLOModelLoader.

Covers:
1. Default configuration loading and custom YAML parsing.
2. Valid configuration attributes and path resolution.
3. Parameter boundary validation (confidence, IoU, image size).
4. Task and device constraint validation (task='detect', device='cpu').
5. Half-precision rejection on CPU.
6. Missing model weights handling (YOLOModelNotFoundError).
7. Directory passed as weights path handling.
8. Zero-byte weights file handling (YOLOModelLoadError).
9. Mocked model loading lifecycle and task verification.
10. Mocked task mismatch detection.
11. Lazy loading invariant verification.
12. Single-frame prediction input validation.
13. Real model weights loading integration test.
14. Real single-frame inference smoke test on driving dataset.
15. End-to-end CLI verification runner.
"""

from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import ultralytics

from src.detection.yolo_model_loader import (
    YOLOConfigError,
    YOLOEnvironmentError,
    YOLOInferenceError,
    YOLOModelInfo,
    YOLOModelLoadError,
    YOLOModelLoader,
    YOLOModelNotFoundError,
    verify_yolo_environment,
)


class TestYOLOModelLoader(unittest.TestCase):
    """Test suite covering unit and integration testing for YOLOModelLoader."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate project fixtures and real weights."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)
        cls.real_weights_path = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
        cls.real_frame_path = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / "driving_sample"
            / "preprocessed_frames"
            / "frame_000001.jpg"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary sandbox."""
        cls.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # Configuration and Parameter Validation Tests (Unit Tests)
    # --------------------------------------------------------------------------

    def test_01_default_configuration_loading(self) -> None:
        """Verify that default settings load cleanly from config/yolo_config.yaml."""
        loader = YOLOModelLoader(project_root=PROJECT_ROOT)
        self.assertEqual(loader.model_name, "yolo11n")
        self.assertEqual(loader.task, "detect")
        self.assertEqual(loader.device, "cpu")
        self.assertEqual(loader.image_size, 640)
        self.assertEqual(loader.confidence_threshold, 0.25)
        self.assertEqual(loader.iou_threshold, 0.45)
        self.assertFalse(loader.is_loaded)

    def test_02_custom_yaml_configuration(self) -> None:
        """Verify that a custom YAML file correctly overrides default parameters."""
        custom_yaml = self.temp_path / "custom_yolo.yaml"
        custom_yaml.write_text(
            "yolo:\n"
            "  model_name: custom_nano\n"
            "  confidence_threshold: 0.50\n"
            "  iou_threshold: 0.60\n"
            "  image_size: 320\n",
            encoding="utf-8",
        )
        loader = YOLOModelLoader(config_path=custom_yaml, project_root=PROJECT_ROOT)
        self.assertEqual(loader.model_name, "custom_nano")
        self.assertEqual(loader.confidence_threshold, 0.50)
        self.assertEqual(loader.iou_threshold, 0.60)
        self.assertEqual(loader.image_size, 320)

    def test_03_invalid_confidence_threshold_raises_error(self) -> None:
        """Verify that confidence thresholds <= 0.0 or > 1.0 raise YOLOConfigError."""
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(conf_threshold=0.0, project_root=PROJECT_ROOT)
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(conf_threshold=1.5, project_root=PROJECT_ROOT)

    def test_04_invalid_iou_threshold_raises_error(self) -> None:
        """Verify that IoU thresholds <= 0.0 or > 1.0 raise YOLOConfigError."""
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(iou_threshold=-0.1, project_root=PROJECT_ROOT)
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(iou_threshold=1.1, project_root=PROJECT_ROOT)

    def test_05_invalid_image_size_raises_error(self) -> None:
        """Verify that extreme image sizes raise YOLOConfigError."""
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(image_size=16, project_root=PROJECT_ROOT)
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(image_size=4096, project_root=PROJECT_ROOT)

    def test_06_unsupported_task_raises_error(self) -> None:
        """Verify that non-detection tasks raise YOLOConfigError."""
        bad_yaml = self.temp_path / "bad_task.yaml"
        bad_yaml.write_text("yolo:\n  task: segment\n", encoding="utf-8")
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(config_path=bad_yaml, project_root=PROJECT_ROOT)

    def test_07_cuda_selection_on_cpu_environment_raises_error(self) -> None:
        """Verify that requesting CUDA device when unavailable raises YOLOConfigError."""
        if not torch.cuda.is_available():
            with self.assertRaises(YOLOConfigError):
                YOLOModelLoader(device="cuda", project_root=PROJECT_ROOT)

    def test_08_half_precision_on_cpu_raises_error(self) -> None:
        """Verify that half_precision=True on CPU device raises YOLOConfigError."""
        bad_yaml = self.temp_path / "bad_fp16.yaml"
        bad_yaml.write_text("yolo:\n  device: cpu\n  half_precision: true\n", encoding="utf-8")
        with self.assertRaises(YOLOConfigError):
            YOLOModelLoader(config_path=bad_yaml, project_root=PROJECT_ROOT)

    # --------------------------------------------------------------------------
    # Weights Filesystem Validation Tests
    # --------------------------------------------------------------------------

    def test_09_missing_weights_file_raises_error(self) -> None:
        """Verify that non-existent weights path raises YOLOModelNotFoundError."""
        loader = YOLOModelLoader(
            weights_path="models/yolo/non_existent_weights_12345.pt",
            project_root=PROJECT_ROOT,
        )
        with self.assertRaises(YOLOModelNotFoundError):
            loader.load_model()

    def test_10_directory_weights_path_raises_error(self) -> None:
        """Verify that passing a directory as weights path raises YOLOModelNotFoundError."""
        loader = YOLOModelLoader(
            weights_path=self.temp_path,
            project_root=PROJECT_ROOT,
        )
        with self.assertRaises(YOLOModelNotFoundError):
            loader.load_model()

    def test_11_empty_weights_file_raises_error(self) -> None:
        """Verify that a 0-byte weights file raises YOLOModelLoadError."""
        empty_pt = self.temp_path / "empty_weights.pt"
        empty_pt.touch()
        loader = YOLOModelLoader(weights_path=empty_pt, project_root=PROJECT_ROOT)
        with self.assertRaises(YOLOModelLoadError):
            loader.load_model()

    # --------------------------------------------------------------------------
    # Mocked Model Loading and Lifecycle Tests
    # --------------------------------------------------------------------------

    def test_12_mocked_model_loading_success(self) -> None:
        """Verify model loading lifecycle using mocked Ultralytics YOLO instance."""
        fake_pt = self.temp_path / "mocked_weights.pt"
        fake_pt.write_bytes(b"MOCKED_WEIGHTS_BINARY_CONTENT")

        mock_instance = MagicMock()
        mock_instance.task = "detect"
        mock_instance.names = {0: "person", 1: "bicycle", 2: "car"}

        with patch("src.detection.yolo_model_loader.YOLO", return_value=mock_instance):
            loader = YOLOModelLoader(weights_path=fake_pt, project_root=PROJECT_ROOT)
            self.assertFalse(loader.is_loaded)

            model = loader.load_model()
            self.assertTrue(loader.is_loaded)
            self.assertEqual(model, mock_instance)

            info = loader.get_model_info()
            self.assertEqual(info.model_name, "yolo11n")
            self.assertEqual(info.task, "detect")
            self.assertEqual(info.device, "cpu")
            self.assertEqual(info.num_classes, 3)
            self.assertEqual(info.class_names[2], "car")
            self.assertTrue(info.is_loaded)
            self.assertIsInstance(info.to_dict(), dict)

    def test_13_mocked_model_task_mismatch_raises_error(self) -> None:
        """Verify that loading a non-detection model raises YOLOModelLoadError."""
        fake_pt = self.temp_path / "mocked_classify.pt"
        fake_pt.write_bytes(b"MOCKED_CLASSIFICATION_WEIGHTS")

        mock_instance = MagicMock()
        mock_instance.task = "classify"

        with patch("src.detection.yolo_model_loader.YOLO", return_value=mock_instance):
            loader = YOLOModelLoader(weights_path=fake_pt, project_root=PROJECT_ROOT)
            with self.assertRaises(YOLOModelLoadError):
                loader.load_model()

    def test_14_lazy_loading_invariant(self) -> None:
        """Verify that creating a YOLOModelLoader does not eagerly load weights into RAM."""
        loader = YOLOModelLoader(project_root=PROJECT_ROOT)
        self.assertFalse(loader.is_loaded)
        self.assertIsNone(loader._model)
        self.assertIsNone(loader._model_info)

    def test_15_predict_single_frame_invalid_inputs(self) -> None:
        """Verify that invalid frame inputs raise YOLOInferenceError."""
        fake_pt = self.temp_path / "mock_weights_for_pred.pt"
        fake_pt.write_bytes(b"MOCKED_WEIGHTS")

        mock_instance = MagicMock()
        mock_instance.task = "detect"
        mock_instance.names = {0: "car"}

        with patch("src.detection.yolo_model_loader.YOLO", return_value=mock_instance):
            loader = YOLOModelLoader(weights_path=fake_pt, project_root=PROJECT_ROOT)

            # Test 1: Empty numpy array
            with self.assertRaises(YOLOInferenceError):
                loader.predict_single_frame(np.array([]))

            # Test 2: Non-existent image file
            with self.assertRaises(YOLOInferenceError):
                loader.predict_single_frame("non_existent_image_9999.jpg")

    # --------------------------------------------------------------------------
    # Integration Tests (Real Local Model & Real Dataset Frame)
    # --------------------------------------------------------------------------

    def test_16_real_model_loading_integration(self) -> None:
        """Integration test: Verify loading real yolo11n.pt weights on CPU."""
        if not self.real_weights_path.is_file():
            self.skipTest(f"Real weights not found at '{self.real_weights_path}'. Skipping integration test.")

        loader = YOLOModelLoader(weights_path=self.real_weights_path, project_root=PROJECT_ROOT)
        model = loader.load_model()

        self.assertIsNotNone(model)
        self.assertTrue(loader.is_loaded)
        info = loader.get_model_info()

        self.assertEqual(info.task, "detect")
        self.assertEqual(info.device, "cpu")
        self.assertEqual(info.num_classes, 80)
        self.assertIn("car", info.class_names.values())
        self.assertIn("person", info.class_names.values())
        self.assertGreater(info.file_size_bytes, 5_000_000)

    def test_17_real_dataset_single_frame_smoke_inference(self) -> None:
        """Integration test: Run CPU smoke inference on real preprocessed driving frame."""
        if not self.real_weights_path.is_file():
            self.skipTest(f"Real weights not found at '{self.real_weights_path}'. Skipping smoke test.")
        if not self.real_frame_path.is_file():
            self.skipTest(f"Real frame not found at '{self.real_frame_path}'. Skipping smoke test.")

        loader = YOLOModelLoader(weights_path=self.real_weights_path, project_root=PROJECT_ROOT)
        result = loader.predict_single_frame(self.real_frame_path)

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "boxes"))
        self.assertEqual(result.orig_shape, (360, 640))
        # Ensure boxes container is accessible and valid
        boxes = result.boxes
        self.assertIsNotNone(boxes)
        # Even if detections count is 0, the result structure must be valid
        self.assertGreaterEqual(len(boxes), 0)

    def test_18_cli_environment_verification(self) -> None:
        """Verify that verify_yolo_environment executes and produces valid diagnostics."""
        if not self.real_weights_path.is_file() or not self.real_frame_path.is_file():
            self.skipTest("Real weights or test frame missing. Skipping CLI verification test.")

        captured = StringIO()
        with patch("sys.stdout", captured):
            passed = verify_yolo_environment(frame_path=self.real_frame_path)

        output = captured.getvalue()
        self.assertTrue(passed)
        self.assertIn("PERCEPTGUARD AI – YOLO ENVIRONMENT CHECK", output)
        self.assertIn("Selected device: cpu", output)
        self.assertIn("Model loading: SUCCESS", output)
        self.assertIn("Frame decoding: SUCCESS", output)
        self.assertIn("Inference status: SUCCESS", output)
        self.assertIn("OVERALL STATUS: PASS", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
