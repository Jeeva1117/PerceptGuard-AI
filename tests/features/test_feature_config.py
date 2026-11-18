"""Automated unit and integration test suite for PerceptGuard AI Feature Configuration.

Validates:
- Default YAML configuration loading and parameter boundaries.
- Flat and nested dictionary deserialization.
- Immutability enforcement (frozen dataclass).
- Rejection of invalid types, schema versions, precisions, and resource constraints.
- Operational presets (default, fast_cpu, high_precision, minimal).
- YAML serialization roundtrip.
- CLI argument validation and error code returns.
"""

from pathlib import Path
import tempfile
import unittest

from src.features.feature_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    FEATURE_PRESETS,
    FeatureConfig,
    FeatureConfigError,
    FeatureConfigFileNotFoundError,
    InvalidFeatureConfigError,
    main,
)


class TestFeatureConfig(unittest.TestCase):
    """Test suite for FeatureConfig and configuration manager."""

    def test_01_default_configuration_loading(self) -> None:
        cfg = FeatureConfig.from_yaml()
        self.assertEqual(cfg.schema_version, "1.0")
        self.assertEqual(cfg.feature_set_name, "standard_driving_features")
        self.assertTrue(cfg.include_detection_features)
        self.assertTrue(cfg.include_motion_features)
        self.assertTrue(cfg.include_visual_quality_features)
        self.assertTrue(cfg.include_tracking_features)
        self.assertEqual(cfg.sort_by, "source_frame_index")
        self.assertEqual(cfg.float_precision, "float32")
        self.assertEqual(cfg.decimal_places, 4)
        self.assertEqual(cfg.missing_value_strategy, "explicit_null")

    def test_02_loading_from_nested_dict(self) -> None:
        nested_data = {
            "schema_version": "1.0",
            "feature_set_name": "custom_set",
            "inputs": {
                "detection_results_path": "path/to/det.json",
                "tracking_results_path": "path/to/trk.json",
            },
            "outputs": {
                "output_dir": "custom/features",
                "save_format": "json",
            },
            "feature_groups": {
                "include_detection_features": True,
                "include_motion_features": False,
                "include_visual_quality_features": False,
                "include_tracking_features": True,
            },
            "precision": {
                "float_precision": "float64",
                "decimal_places": 6,
            },
            "processing": {
                "batch_size": 50,
                "max_workers": 2,
                "memory_limit_mb": 400,
            },
        }
        cfg = FeatureConfig.from_dict(nested_data)
        self.assertEqual(cfg.feature_set_name, "custom_set")
        self.assertEqual(cfg.detection_results_path, "path/to/det.json")
        self.assertFalse(cfg.include_motion_features)
        self.assertEqual(cfg.float_precision, "float64")
        self.assertEqual(cfg.decimal_places, 6)
        self.assertEqual(cfg.batch_size, 50)

    def test_03_immutability_enforcement(self) -> None:
        cfg = FeatureConfig.from_yaml()
        with self.assertRaises((TypeError, AttributeError)):
            cfg.decimal_places = 8  # type: ignore

    def test_04_missing_configuration_file_raises_error(self) -> None:
        with self.assertRaises(FeatureConfigFileNotFoundError):
            FeatureConfig.from_yaml("non_existent_config_path_123.yaml")

    def test_05_directory_path_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(InvalidFeatureConfigError):
                FeatureConfig.from_yaml(tmpdir)

    def test_06_malformed_yaml_content_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("schema_version: [unclosed list")
            p = Path(f.name)
        try:
            with self.assertRaises(InvalidFeatureConfigError):
                FeatureConfig.from_yaml(p)
        finally:
            if p.exists():
                p.unlink()

    def test_07_empty_yaml_file_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")
            p = Path(f.name)
        try:
            with self.assertRaises(InvalidFeatureConfigError):
                FeatureConfig.from_yaml(p)
        finally:
            if p.exists():
                p.unlink()

    def test_08_non_dict_yaml_content_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            p = Path(f.name)
        try:
            with self.assertRaises(InvalidFeatureConfigError):
                FeatureConfig.from_yaml(p)
        finally:
            if p.exists():
                p.unlink()

    def test_09_unsupported_schema_version_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(schema_version="99.0")

    def test_10_invalid_float_precision_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(float_precision="int32")

    def test_11_invalid_decimal_places_boundary_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(decimal_places=0)
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(decimal_places=9)

    def test_12_invalid_missing_value_strategy_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(missing_value_strategy="interpolate_random")

    def test_13_invalid_sort_by_field_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(sort_by="random_hash")

    def test_14_non_positive_default_fps_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(default_fps=0.0)
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(default_fps=-5.0)

    def test_15_negative_max_sequence_gap_frames_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(max_sequence_gap_frames=-1)

    def test_16_non_positive_batch_size_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(batch_size=0)

    def test_17_non_positive_max_workers_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(max_workers=0)

    def test_18_insufficient_memory_limit_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(memory_limit_mb=20)

    def test_19_all_feature_groups_disabled_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig(
                include_detection_features=False,
                include_motion_features=False,
                include_visual_quality_features=False,
                include_tracking_features=False,
            )

    def test_20_serialization_to_dict_and_yaml_roundtrip(self) -> None:
        cfg = FeatureConfig.from_yaml()
        d = cfg.to_dict(nested=True)
        self.assertIn("inputs", d)
        self.assertIn("precision", d)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            p = Path(f.name)
        try:
            cfg.to_yaml(p)
            reloaded = FeatureConfig.from_yaml(p)
            self.assertEqual(cfg.feature_set_name, reloaded.feature_set_name)
            self.assertEqual(cfg.batch_size, reloaded.batch_size)
            self.assertEqual(cfg.decimal_places, reloaded.decimal_places)
        finally:
            if p.exists():
                p.unlink()

    def test_21_operational_presets(self) -> None:
        for preset_name in ["default", "fast_cpu", "high_precision", "minimal"]:
            preset = FeatureConfig.from_preset(preset_name)
            self.assertIsInstance(preset, FeatureConfig)

        fast_cpu = FeatureConfig.from_preset("fast_cpu")
        self.assertFalse(fast_cpu.include_visual_quality_features)
        self.assertEqual(fast_cpu.batch_size, 200)

        high_prec = FeatureConfig.from_preset("high_precision")
        self.assertEqual(high_prec.decimal_places, 6)
        self.assertEqual(high_prec.float_precision, "float64")

    def test_22_unknown_preset_rejection(self) -> None:
        with self.assertRaises(InvalidFeatureConfigError):
            FeatureConfig.from_preset("non_existent_preset_xyz")

    def test_23_cli_validation_success(self) -> None:
        ret = main(["--validate"])
        self.assertEqual(ret, 0)

    def test_24_cli_validation_preset(self) -> None:
        ret = main(["--preset", "fast_cpu", "--validate"])
        self.assertEqual(ret, 0)

    def test_25_cli_missing_config_returns_code_1(self) -> None:
        ret = main(["--config", "non_existent_file.yaml", "--validate"])
        self.assertEqual(ret, 1)

    def test_26_cli_invalid_config_returns_code_1(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("batch_size: -10\n")
            p = Path(f.name)
        try:
            ret = main(["--config", str(p), "--validate"])
            self.assertEqual(ret, 1)
        finally:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    unittest.main()
