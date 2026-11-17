"""PerceptGuard AI: Unit and Integration Tests for TrackerConfig Architecture.

Validates:
- ByteTrack YAML configuration loading and schema parsing
- Parameter type, boundary, and domain constraints
- Configuration immutability and serialization
- Preset factories and CLI entrypoint
"""

from pathlib import Path
import tempfile
from typing import Any, Dict
import unittest

import yaml

from src.tracking.tracker_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    REQUIRED_TRACKER_FIELDS,
    SUPPORTED_TRACKER_TYPES,
    TRACKER_PRESETS,
    TrackerConfig,
    TrackerConfigError,
    TrackerConfigFileNotFoundError,
    main,
)


class TestTrackerConfig(unittest.TestCase):
    """Test suite for TrackerConfig validation, loading, and immutability."""

    def setUp(self) -> None:
        self.valid_payload: Dict[str, Any] = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.80,
            "fuse_score": True,
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. Valid Configuration Loading & Parsing
    # --------------------------------------------------------------------------

    def test_01_valid_dict_instantiation(self) -> None:
        """Test instantiation and attribute verification from valid dictionary."""
        cfg = TrackerConfig.from_dict(self.valid_payload)
        self.assertEqual(cfg.tracker_type, "bytetrack")
        self.assertEqual(cfg.track_high_thresh, 0.25)
        self.assertEqual(cfg.track_low_thresh, 0.10)
        self.assertEqual(cfg.new_track_thresh, 0.25)
        self.assertEqual(cfg.track_buffer, 30)
        self.assertEqual(cfg.match_thresh, 0.80)
        self.assertTrue(cfg.fuse_score)

    def test_02_valid_yaml_file_loading(self) -> None:
        """Test loading configuration from a physical YAML file."""
        test_file = self.temp_path / "valid_tracker.yaml"
        test_file.write_text(yaml.dump(self.valid_payload), encoding="utf-8")

        cfg = TrackerConfig.from_yaml(test_file)
        self.assertEqual(cfg.tracker_type, "bytetrack")
        self.assertEqual(cfg.track_buffer, 30)
        self.assertEqual(cfg.config_path, test_file.resolve())

    def test_03_default_configuration_loading(self) -> None:
        """Test loading default repository configuration without explicit path."""
        cfg = TrackerConfig.from_yaml()
        self.assertEqual(cfg.tracker_type, "bytetrack")
        self.assertGreater(cfg.track_high_thresh, cfg.track_low_thresh)
        self.assertTrue(cfg.is_frozen)

    # --------------------------------------------------------------------------
    # 2. File and Path Validation
    # --------------------------------------------------------------------------

    def test_04_missing_configuration_file(self) -> None:
        """Test that non-existent YAML path raises TrackerConfigFileNotFoundError."""
        missing = self.temp_path / "does_not_exist.yaml"
        with self.assertRaises(TrackerConfigFileNotFoundError):
            TrackerConfig.from_yaml(missing)

    def test_05_malformed_yaml_content(self) -> None:
        """Test that malformed YAML syntax raises TrackerConfigError."""
        bad_file = self.temp_path / "corrupt.yaml"
        bad_file.write_text("tracker_type: [unclosed list\n  foo: bar", encoding="utf-8")
        with self.assertRaises(TrackerConfigError):
            TrackerConfig.from_yaml(bad_file)

    def test_06_non_dict_yaml_content(self) -> None:
        """Test that YAML containing a scalar or list raises TrackerConfigError."""
        list_file = self.temp_path / "list.yaml"
        list_file.write_text("- item1\n- item2", encoding="utf-8")
        with self.assertRaises(TrackerConfigError):
            TrackerConfig.from_yaml(list_file)

    def test_07_directory_as_config_path(self) -> None:
        """Test that passing a directory path raises TrackerConfigError."""
        with self.assertRaises(TrackerConfigError):
            TrackerConfig.from_yaml(self.temp_path)

    # --------------------------------------------------------------------------
    # 3. Schema & Required Parameters Validation
    # --------------------------------------------------------------------------

    def test_08_missing_required_fields(self) -> None:
        """Test that omitting any required field raises TrackerConfigError."""
        for field_name in REQUIRED_TRACKER_FIELDS:
            incomplete = dict(self.valid_payload)
            del incomplete[field_name]
            with self.assertRaises(TrackerConfigError) as ctx:
                TrackerConfig.from_dict(incomplete)
            self.assertIn("Missing required tracking parameters", str(ctx.exception))

    def test_09_invalid_tracker_type(self) -> None:
        """Test that unsupported tracker types raise TrackerConfigError."""
        invalid_types = ["botsort", "deepsort", "ocsort", "norfair", 123, None]
        for t in invalid_types:
            payload = dict(self.valid_payload, tracker_type=t)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    # --------------------------------------------------------------------------
    # 4. Threshold Range & Type Validations
    # --------------------------------------------------------------------------

    def test_10_invalid_track_high_thresh(self) -> None:
        """Test out-of-range or invalid types for track_high_thresh."""
        invalid_values = [0.0, -0.1, 1.05, 2.0, "0.5", True, None]
        for val in invalid_values:
            payload = dict(self.valid_payload, track_high_thresh=val)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    def test_11_invalid_track_low_thresh(self) -> None:
        """Test out-of-range or invalid types for track_low_thresh."""
        invalid_values = [-0.01, 1.0, 1.5, "0.1", True, None]
        for val in invalid_values:
            payload = dict(self.valid_payload, track_low_thresh=val)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    def test_12_low_thresh_greater_than_or_equal_to_high_thresh(self) -> None:
        """Test constraint: track_low_thresh must be strictly less than track_high_thresh."""
        # Equal
        payload_equal = dict(self.valid_payload, track_high_thresh=0.25, track_low_thresh=0.25)
        with self.assertRaises(TrackerConfigError) as ctx:
            TrackerConfig.from_dict(payload_equal)
        self.assertIn("must be strictly less than", str(ctx.exception))

        # Greater
        payload_greater = dict(self.valid_payload, track_high_thresh=0.20, track_low_thresh=0.30)
        with self.assertRaises(TrackerConfigError) as ctx:
            TrackerConfig.from_dict(payload_greater)
        self.assertIn("must be strictly less than", str(ctx.exception))

    def test_13_invalid_new_track_thresh(self) -> None:
        """Test out-of-range or invalid types for new_track_thresh."""
        invalid_values = [0.0, -0.5, 1.1, "0.25", False, None]
        for val in invalid_values:
            payload = dict(self.valid_payload, new_track_thresh=val)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    def test_14_invalid_track_buffer(self) -> None:
        """Test out-of-range or non-integer track_buffer values."""
        invalid_buffers = [0, -1, 1001, 30.5, "30", True, None]
        for b in invalid_buffers:
            payload = dict(self.valid_payload, track_buffer=b)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    def test_15_invalid_match_thresh(self) -> None:
        """Test out-of-range or invalid types for match_thresh."""
        invalid_values = [0.0, -0.2, 1.1, "0.8", True, None]
        for val in invalid_values:
            payload = dict(self.valid_payload, match_thresh=val)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    def test_16_invalid_fuse_score(self) -> None:
        """Test non-boolean fuse_score values."""
        invalid_values = ["true", "False", 1, 0, None, [True]]
        for val in invalid_values:
            payload = dict(self.valid_payload, fuse_score=val)
            with self.assertRaises(TrackerConfigError):
                TrackerConfig.from_dict(payload)

    # --------------------------------------------------------------------------
    # 5. Immutability, Serialization & Presets
    # --------------------------------------------------------------------------

    def test_17_immutability_enforcement(self) -> None:
        """Test that modifying a frozen TrackerConfig raises TypeError."""
        cfg = TrackerConfig.from_dict(self.valid_payload, freeze=True)
        self.assertTrue(cfg.is_frozen)
        with self.assertRaises(TypeError):
            cfg.track_buffer = 45
        with self.assertRaises(TypeError):
            cfg.track_high_thresh = 0.50

    def test_18_unfrozen_instance_allows_modification(self) -> None:
        """Test that unfrozen TrackerConfig permits updates prior to freeze."""
        cfg = TrackerConfig.from_dict(self.valid_payload, freeze=False)
        self.assertFalse(cfg.is_frozen)
        cfg.track_buffer = 45
        self.assertEqual(cfg.track_buffer, 45)
        cfg.freeze()
        self.assertTrue(cfg.is_frozen)
        with self.assertRaises(TypeError):
            cfg.track_buffer = 50

    def test_19_to_dict_and_to_yaml_serialization(self) -> None:
        """Test round-trip serialization to dict and YAML string/file."""
        cfg = TrackerConfig.from_dict(self.valid_payload)
        d = cfg.to_dict()
        self.assertEqual(d["tracker_type"], "bytetrack")
        self.assertEqual(d["track_buffer"], 30)

        out_file = self.temp_path / "exported.yaml"
        yaml_str = cfg.to_yaml(out_file)
        self.assertTrue(out_file.exists())
        self.assertIn("tracker_type: bytetrack", yaml_str)

        # Reload from exported
        reloaded = TrackerConfig.from_yaml(out_file)
        self.assertEqual(reloaded.to_dict(), cfg.to_dict())

    def test_20_operational_presets(self) -> None:
        """Test loading all predefined operational presets."""
        for preset_name in TRACKER_PRESETS.keys():
            cfg = TrackerConfig.from_preset(preset_name)
            self.assertEqual(cfg.tracker_type, "bytetrack")
            self.assertTrue(cfg.is_frozen)

    def test_21_unknown_preset_rejection(self) -> None:
        """Test that requesting an unknown preset name raises TrackerConfigError."""
        with self.assertRaises(TrackerConfigError):
            TrackerConfig.from_preset("non_existent_preset")

    def test_22_strict_keys_mode(self) -> None:
        """Test strict mode rejection of unrecognized parameter keys."""
        payload_with_extra = dict(self.valid_payload, unknown_param=42)
        # Non-strict ignores extra keys in from_dict (or accepts raw dict)
        # Strict mode raises TrackerConfigError
        with self.assertRaises(TrackerConfigError) as ctx:
            TrackerConfig.from_dict(payload_with_extra, strict_keys=True)
        self.assertIn("Unsupported or unrecognized", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 6. CLI Entrypoint
    # --------------------------------------------------------------------------

    def test_23_cli_validate_success(self) -> None:
        """Test CLI with --validate returns exit code 0."""
        exit_code = main(["--validate"])
        self.assertEqual(exit_code, 0)

    def test_24_cli_preset_validation(self) -> None:
        """Test CLI validating a named preset returns exit code 0."""
        exit_code = main(["--preset", "cpu_fast"])
        self.assertEqual(exit_code, 0)

    def test_25_cli_missing_file_returns_error_code(self) -> None:
        """Test CLI with non-existent config path returns code 2."""
        exit_code = main(["--config", str(self.temp_path / "missing.yaml")])
        self.assertEqual(exit_code, 2)

    def test_26_cli_invalid_content_returns_error_code(self) -> None:
        """Test CLI with invalid configuration content returns code 1."""
        bad_file = self.temp_path / "bad.yaml"
        bad_file.write_text(yaml.dump({"tracker_type": "invalid"}), encoding="utf-8")
        exit_code = main(["--config", str(bad_file)])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
