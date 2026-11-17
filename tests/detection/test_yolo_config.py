"""PerceptGuard-AI: Comprehensive Test Suite for Advanced YOLO Model Configuration.

Validates:
1. Valid YAML loading and schema parsing.
2. Missing configuration file exception handling.
3. Malformed YAML syntax detection.
4. Missing or invalid root mapping.
5. Invalid model name handling.
6. Missing weights file when strict weights validation is active.
7. Image size validation (positivity, stride of 32, range bounds).
8. Confidence threshold boundary and type validation.
9. IoU threshold boundary and type validation.
10. Maximum detections validation (positivity, non-bool, integer).
11. Compute device constraint verification (CPU enforcement, CUDA availability check).
12. Numerical precision guardrails (FP16 rejection on CPU).
13. Class filtering ID validation (types, bounds, negative IDs).
14. Duplicate class ID rejection.
15. Type safety across performance and validation sections.
16. Project-relative path resolution for weights and configurations.
17. Configuration serialization and roundtrip fidelity (to_dict, to_yaml, to_metadata_dict).
18. Default configuration defaults and bounds.
19. Strict mode validation and error messaging.
20. Seamless compatibility with existing YOLOModelLoader.
21. Operational perception presets (balanced, high_precision, high_recall, cpu_lightweight).
22. Real local weights integration test with yolo11n.pt.
23. Real single-frame driving perception inference with custom configuration.
"""

from copy import deepcopy
from io import StringIO
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
import yaml

from src.detection.yolo_config import (
    PRESETS,
    InferenceConfig,
    ModelConfig,
    PerformanceConfig,
    RuntimeConfig,
    ValidationConfig,
    YOLOConfig,
    YOLOConfigError,
    YOLOEnvironmentError,
    YOLOModelNotFoundError,
    load_yolo_config,
)
from src.detection.yolo_model_loader import YOLOModelLoader


class TestYOLOModelConfiguration(unittest.TestCase):
    """Test suite covering unit and integration testing for YOLO model configuration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate project fixtures, weights, and test frames."""
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
    # 1-5. File, Schema & Model Architecture Tests
    # --------------------------------------------------------------------------

    def test_01_valid_yaml_loading(self) -> None:
        """Verify that a valid YAML configuration loads cleanly into a typed YOLOConfig object."""
        cfg_file = self.temp_path / "valid_cfg.yaml"
        cfg_file.write_text(
            "version: '1.0.0'\n"
            "model:\n"
            "  name: 'yolo11n'\n"
            "  architecture: 'yolo11'\n"
            "  weights_path: 'models/yolo/yolo11n.pt'\n"
            "  task: 'detect'\n"
            "runtime:\n"
            "  device: 'cpu'\n"
            "  half_precision: false\n"
            "inference:\n"
            "  image_size: 640\n"
            "  confidence_threshold: 0.25\n"
            "  iou_threshold: 0.45\n"
            "  max_detections: 100\n",
            encoding="utf-8",
        )
        cfg = YOLOConfig.from_yaml(cfg_file, project_root=PROJECT_ROOT)
        self.assertEqual(cfg.model.name, "yolo11n")
        self.assertEqual(cfg.model.architecture, "yolo11")
        self.assertEqual(cfg.runtime.device, "cpu")
        self.assertFalse(cfg.runtime.half_precision)
        self.assertEqual(cfg.inference.image_size, 640)
        self.assertAlmostEqual(cfg.inference.confidence_threshold, 0.25)
        self.assertAlmostEqual(cfg.inference.iou_threshold, 0.45)
        self.assertEqual(cfg.inference.max_detections, 100)

    def test_02_missing_configuration_file(self) -> None:
        """Verify that a non-existent YAML configuration path raises YOLOConfigError."""
        non_existent = self.temp_path / "does_not_exist_9999.yaml"
        with self.assertRaises(YOLOConfigError):
            YOLOConfig.from_yaml(non_existent, project_root=PROJECT_ROOT)

    def test_03_malformed_yaml(self) -> None:
        """Verify that malformed YAML syntax raises YOLOConfigError."""
        malformed = self.temp_path / "malformed.yaml"
        malformed.write_text("model: [unclosed list\n  key: value", encoding="utf-8")
        with self.assertRaises(YOLOConfigError):
            YOLOConfig.from_yaml(malformed, project_root=PROJECT_ROOT)

    def test_04_missing_or_invalid_root_mapping(self) -> None:
        """Verify that non-dictionary root structures raise YOLOConfigError."""
        list_root = self.temp_path / "list_root.yaml"
        list_root.write_text("- item1\n- item2\n", encoding="utf-8")
        with self.assertRaises(YOLOConfigError):
            YOLOConfig.from_yaml(list_root, project_root=PROJECT_ROOT)

    def test_05_invalid_model_name(self) -> None:
        """Verify that empty or non-string model name raises YOLOConfigError."""
        cfg = YOLOConfig()
        cfg.model.name = ""
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    # --------------------------------------------------------------------------
    # 6-10. Weights & Hyperparameter Boundaries
    # --------------------------------------------------------------------------

    def test_06_missing_weights_file(self) -> None:
        """Verify that missing weights file raises YOLOModelNotFoundError when validate_weights is True."""
        cfg = YOLOConfig()
        cfg.model.weights_path = "models/yolo/non_existent_weights_xyz.pt"
        cfg.validation.validate_weights = True
        with self.assertRaises(YOLOModelNotFoundError):
            cfg.validate(project_root=PROJECT_ROOT)

    def test_07_invalid_image_size(self) -> None:
        """Verify that image sizes not divisible by 32 or out of bounds raise YOLOConfigError."""
        # Non-stride (not divisible by 32)
        cfg = YOLOConfig()
        cfg.inference.image_size = 645
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Below minimum stride bound (e.g. 16 < 32)
        cfg.inference.image_size = 16
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Above maximum limit (> 1920)
        cfg.inference.image_size = 2048
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Non-integer / boolean
        cfg.inference.image_size = True  # type: ignore[assignment]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_08_invalid_confidence_threshold(self) -> None:
        """Verify that confidence thresholds outside (0.0, 1.0] or non-numeric raise YOLOConfigError."""
        cfg = YOLOConfig()

        # Zero or negative
        cfg.inference.confidence_threshold = 0.0
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        cfg.inference.confidence_threshold = -0.05
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Greater than 1.0
        cfg.inference.confidence_threshold = 1.01
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Boolean
        cfg.inference.confidence_threshold = True  # type: ignore[assignment]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # String
        cfg.inference.confidence_threshold = "0.5"  # type: ignore[assignment]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_09_invalid_iou_threshold(self) -> None:
        """Verify that IoU thresholds outside (0.0, 1.0] or non-numeric raise YOLOConfigError."""
        cfg = YOLOConfig()

        # Negative
        cfg.inference.iou_threshold = -0.1
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Greater than 1.0
        cfg.inference.iou_threshold = 1.5
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Boolean
        cfg.inference.iou_threshold = False  # type: ignore[assignment]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_10_invalid_max_detections(self) -> None:
        """Verify that max_detections <= 0 or non-integers raise YOLOConfigError."""
        cfg = YOLOConfig()

        # Zero
        cfg.inference.max_detections = 0
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Negative
        cfg.inference.max_detections = -10
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Boolean
        cfg.inference.max_detections = True  # type: ignore[assignment]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    # --------------------------------------------------------------------------
    # 11-15. Hardware, Precision, Class Filtering & Types
    # --------------------------------------------------------------------------

    def test_11_invalid_device(self) -> None:
        """Verify that unsupported devices or CUDA requests when unavailable raise YOLOConfigError."""
        cfg = YOLOConfig()
        cfg.runtime.device = "tpu"
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        if not torch.cuda.is_available():
            cfg.runtime.device = "cuda"
            with self.assertRaises(YOLOConfigError):
                cfg.validate()

    def test_12_invalid_precision_on_cpu(self) -> None:
        """Verify that half_precision=True on CPU device raises YOLOConfigError."""
        cfg = YOLOConfig()
        cfg.runtime.device = "cpu"
        cfg.runtime.half_precision = True
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_13_invalid_class_ids(self) -> None:
        """Verify that negative class IDs, out-of-range IDs, and non-integers raise YOLOConfigError."""
        cfg = YOLOConfig()

        # Negative class ID
        cfg.inference.classes = [0, 2, -1]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Out-of-range (> 79 for COCO-80)
        cfg.inference.classes = [0, 2, 85]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

        # Non-integer / boolean element
        cfg.inference.classes = [0, True, 2]  # type: ignore[list-item]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_14_duplicate_class_ids(self) -> None:
        """Verify that duplicate class IDs in inference.classes raise YOLOConfigError."""
        cfg = YOLOConfig()
        cfg.inference.classes = [0, 2, 5, 2]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_15_incorrect_parameter_types(self) -> None:
        """Verify that incorrect types across performance, runtime, and validation raise YOLOConfigError."""
        cfg = YOLOConfig()

        # Performance batch size non-int
        cfg.performance.batch_size = "one"  # type: ignore[assignment]
        with self.assertRaises(YOLOConfigError):
            cfg.validate()
        cfg.performance.batch_size = 1

        # Performance latency non-numeric
        cfg.performance.max_allowed_latency_ms = -5.0
        with self.assertRaises(YOLOConfigError):
            cfg.validate()
        cfg.performance.max_allowed_latency_ms = 500.0

        # Runtime seed negative
        cfg.runtime.seed = -10
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    # --------------------------------------------------------------------------
    # 16-20. Path Resolution, Serialization, Presets & Loader Compatibility
    # --------------------------------------------------------------------------

    def test_16_project_relative_path_resolution(self) -> None:
        """Verify that weights_path resolves accurately relative to project_root."""
        cfg = YOLOConfig()
        cfg.model.weights_path = "models/yolo/yolo11n.pt"
        resolved = cfg.resolve_weights_path(PROJECT_ROOT)
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved, (PROJECT_ROOT / "models/yolo/yolo11n.pt").resolve())

    def test_17_configuration_serialization(self) -> None:
        """Verify that to_dict, to_yaml, and to_metadata_dict serialize properly."""
        cfg = YOLOConfig()
        d = cfg.to_dict()
        self.assertIn("model", d)
        self.assertIn("runtime", d)
        self.assertIn("inference", d)
        self.assertIn("performance", d)
        self.assertIn("validation", d)

        # YAML roundtrip
        yaml_out = self.temp_path / "serialized.yaml"
        yaml_str = cfg.to_yaml(yaml_out)
        self.assertTrue(yaml_out.is_file())
        reloaded = YOLOConfig.from_yaml(yaml_out, project_root=PROJECT_ROOT)
        self.assertEqual(reloaded.model.name, cfg.model.name)
        self.assertEqual(reloaded.inference.confidence_threshold, cfg.inference.confidence_threshold)

        # Metadata dictionary
        meta = cfg.to_metadata_dict()
        self.assertIn("metadata", meta)
        self.assertIn("torch_version", meta["metadata"])

    def test_18_default_configuration_behavior(self) -> None:
        """Verify default configuration values against production baselines."""
        cfg = YOLOConfig()
        self.assertEqual(cfg.model_name, "yolo11n")
        self.assertEqual(cfg.device, "cpu")
        self.assertFalse(cfg.half_precision)
        self.assertEqual(cfg.image_size, 640)
        self.assertEqual(cfg.confidence_threshold, 0.25)
        self.assertEqual(cfg.iou_threshold, 0.45)
        self.assertEqual(cfg.max_detections, 100)
        self.assertFalse(cfg.agnostic_nms)
        self.assertIsNone(cfg.classes)

    def test_19_invalid_configuration_rejection(self) -> None:
        """Verify that strict mode rejects uncalibrated and unsupported parameters."""
        cfg = YOLOConfig()
        cfg.model.task = "invalid_task"
        with self.assertRaises(YOLOConfigError):
            cfg.validate()

    def test_20_compatibility_with_model_loader(self) -> None:
        """Verify that YOLOModelLoader directly accepts YOLOConfig instance."""
        cfg = YOLOConfig()
        cfg.inference.confidence_threshold = 0.35
        cfg.inference.iou_threshold = 0.55
        cfg.inference.image_size = 320

        loader = YOLOModelLoader(config=cfg, project_root=PROJECT_ROOT)
        self.assertEqual(loader.confidence_threshold, 0.35)
        self.assertEqual(loader.iou_threshold, 0.55)
        self.assertEqual(loader.image_size, 320)
        self.assertEqual(loader.device, "cpu")

    def test_21_perception_presets(self) -> None:
        """Verify that presets correctly override hyperparameters."""
        base_cfg = YOLOConfig()

        # High Precision Preset
        hp_cfg = base_cfg.with_preset("high_precision")
        self.assertEqual(hp_cfg.confidence_threshold, 0.50)
        self.assertEqual(hp_cfg.iou_threshold, 0.50)
        self.assertEqual(hp_cfg.max_detections, 50)

        # High Recall Preset
        hr_cfg = base_cfg.with_preset("high_recall")
        self.assertEqual(hr_cfg.confidence_threshold, 0.10)
        self.assertEqual(hr_cfg.iou_threshold, 0.40)
        self.assertEqual(hr_cfg.max_detections, 200)

        # CPU Lightweight Preset
        cl_cfg = base_cfg.with_preset("cpu_lightweight")
        self.assertEqual(cl_cfg.confidence_threshold, 0.30)
        self.assertEqual(cl_cfg.image_size, 320)
        self.assertEqual(cl_cfg.max_detections, 50)

        # Unknown preset
        with self.assertRaises(YOLOConfigError):
            base_cfg.with_preset("non_existent_preset")

    # --------------------------------------------------------------------------
    # 22-23. Real Weights & Single-Frame Driving Integration Tests
    # --------------------------------------------------------------------------

    def test_22_real_model_weights_integration(self) -> None:
        """Verify that production config/yolo_config.yaml loads real weights on CPU."""
        if not self.real_weights_path.is_file():
            self.skipTest("Real YOLO11n weights not found at models/yolo/yolo11n.pt")

        cfg = load_yolo_config(project_root=PROJECT_ROOT, validate=True, validate_weights=True)
        self.assertEqual(cfg.model_name, "yolo11n")
        self.assertEqual(cfg.device, "cpu")
        self.assertFalse(cfg.half_precision)

        loader = YOLOModelLoader(config=cfg, project_root=PROJECT_ROOT)
        model = loader.load_model()
        self.assertIsNotNone(model)
        self.assertTrue(loader.is_loaded)

    def test_23_single_frame_inference_with_config(self) -> None:
        """Verify that configured loader runs inference on real driving frame."""
        if not self.real_weights_path.is_file():
            self.skipTest("Real weights missing.")
        if not self.real_frame_path.is_file():
            self.skipTest("Real preprocessed frame missing.")

        cfg = load_yolo_config(project_root=PROJECT_ROOT, validate=True, validate_weights=True)
        loader = YOLOModelLoader(config=cfg, project_root=PROJECT_ROOT)
        res = loader.predict_single_frame(str(self.real_frame_path))
        self.assertIsNotNone(res)
        boxes = getattr(res, "boxes", None)
        self.assertIsNotNone(boxes)
        # Should detect traffic light and truck
        self.assertGreaterEqual(len(boxes), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

