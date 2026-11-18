"""PerceptGuard AI: Feature Engineering Configuration Manager.

Provides strongly typed, validated, and immutable configuration contracts for
multidimensional feature extraction and sequence generation pipelines.

Architectural Principles:
- CPU-First Compatibility: Enforces memory-bounded sequential processing limits.
- Strict Invariant Enforcement: Validates schema versions, field constraints,
  feature group toggles, and missing-value policies.
- Hierarchical & Flat Interoperability: Seamlessly parses nested YAML specifications
  and produces flattened execution-ready parameter sets.
- Preset Architecture: Offers production presets (default, fast_cpu, high_precision, minimal).
- CLI Introspection: Validates configurations via command-line invocation.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger("PerceptGuard.FeatureConfig")

DEFAULT_CONFIG_RELATIVE_PATH = Path("config/feature_config.yaml")

SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1", 1, 1.0}
SUPPORTED_FLOAT_PRECISIONS = {"float16", "float32", "float64"}
SUPPORTED_MISSING_VALUE_STRATEGIES = {"explicit_null", "drop_incomplete"}
SUPPORTED_SORT_FIELDS = {"source_frame_index", "frame_id", "timestamp_seconds"}


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class FeatureConfigError(Exception):
    """Base exception for all feature configuration errors."""
    pass


class FeatureConfigFileNotFoundError(FeatureConfigError):
    """Raised when the specified feature configuration YAML file is not found."""
    pass


class InvalidFeatureConfigError(FeatureConfigError):
    """Raised when feature configuration fails schema or boundary validation."""
    pass


# ==============================================================================
# Core Configuration Contract
# ==============================================================================

@dataclass(frozen=True)
class FeatureConfig:
    """Strongly typed, immutable configuration for Feature Engineering."""

    # Schema & Identification
    schema_version: str = "1.0"
    feature_set_name: str = "standard_driving_features"

    # Input Artifacts
    detection_results_path: Optional[str] = None
    tracking_results_path: Optional[str] = None
    dataset_metadata_path: Optional[str] = None
    preprocessed_frames_dir: Optional[str] = None

    # Output Storage
    output_dir: str = "datasets/processed/driving_sample/features"
    save_format: str = "json"
    compress_output: bool = False

    # Feature Group Toggles
    include_detection_features: bool = True
    include_motion_features: bool = True
    include_visual_quality_features: bool = True
    include_tracking_features: bool = True

    # Temporal Alignment & Frame Ordering
    sort_by: str = "source_frame_index"
    ascending: bool = True
    require_timestamps: bool = False
    default_fps: float = 5.0
    time_unit: str = "seconds"
    max_sequence_gap_frames: int = 2

    # Missing-Value Policy
    missing_value_strategy: str = "explicit_null"
    fail_on_missing_required: bool = True
    fail_on_invalid_bbox: bool = False

    # Numeric Precision
    float_precision: str = "float32"
    decimal_places: int = 4
    normalize_coordinates: bool = False

    # CPU Execution Budgets
    batch_size: int = 100
    max_workers: int = 1
    memory_limit_mb: int = 500
    stream_processing: bool = True

    # Diagnostics & Telemetry
    log_level: str = "INFO"
    strict_validation: bool = True

    def __post_init__(self) -> None:
        """Validate all parameters and invariant constraints upon instantiation."""
        # 1. Schema Version Validation
        if str(self.schema_version) not in SUPPORTED_SCHEMA_VERSIONS:
            raise InvalidFeatureConfigError(
                f"Unsupported schema_version '{self.schema_version}'. Supported: {SUPPORTED_SCHEMA_VERSIONS}"
            )

        # 2. Precision & Rounding Validation
        if self.float_precision not in SUPPORTED_FLOAT_PRECISIONS:
            raise InvalidFeatureConfigError(
                f"Invalid float_precision '{self.float_precision}'. Must be one of {SUPPORTED_FLOAT_PRECISIONS}"
            )
        if not (1 <= self.decimal_places <= 8):
            raise InvalidFeatureConfigError(
                f"decimal_places must be between 1 and 8 (inclusive), got {self.decimal_places}"
            )

        # 3. Missing Value Strategy
        if self.missing_value_strategy not in SUPPORTED_MISSING_VALUE_STRATEGIES:
            raise InvalidFeatureConfigError(
                f"Invalid missing_value_strategy '{self.missing_value_strategy}'. Must be one of {SUPPORTED_MISSING_VALUE_STRATEGIES}"
            )

        # 4. Temporal Alignment
        if self.sort_by not in SUPPORTED_SORT_FIELDS:
            raise InvalidFeatureConfigError(
                f"Invalid sort_by '{self.sort_by}'. Must be one of {SUPPORTED_SORT_FIELDS}"
            )
        if self.default_fps <= 0.0:
            raise InvalidFeatureConfigError(f"default_fps must be positive, got {self.default_fps}")
        if self.max_sequence_gap_frames < 0:
            raise InvalidFeatureConfigError(
                f"max_sequence_gap_frames must be >= 0, got {self.max_sequence_gap_frames}"
            )

        # 5. CPU Execution Budgets
        if self.batch_size < 1:
            raise InvalidFeatureConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_workers < 1:
            raise InvalidFeatureConfigError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.memory_limit_mb < 50:
            raise InvalidFeatureConfigError(f"memory_limit_mb must be >= 50, got {self.memory_limit_mb}")

        # 6. Feature Group Enablement Guard
        if not any([
            self.include_detection_features,
            self.include_motion_features,
            self.include_visual_quality_features,
            self.include_tracking_features,
        ]):
            raise InvalidFeatureConfigError(
                "At least one feature group (detection, motion, visual_quality, tracking) must be enabled."
            )

    # --------------------------------------------------------------------------
    # Serialization and Factory Methods
    # --------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureConfig":
        """Instantiate FeatureConfig from a dictionary (supports both nested and flat layouts)."""
        if not isinstance(data, dict):
            raise InvalidFeatureConfigError(f"Configuration data must be a dictionary, got {type(data).__name__}")

        flat: Dict[str, Any] = {}

        # Top-level keys
        for key in ["schema_version", "feature_set_name"]:
            if key in data:
                flat[key] = data[key]

        # Nested: inputs
        inputs = data.get("inputs", {})
        if isinstance(inputs, dict):
            if "detection_results_path" in inputs:
                flat["detection_results_path"] = inputs["detection_results_path"]
            if "tracking_results_path" in inputs:
                flat["tracking_results_path"] = inputs["tracking_results_path"]
            if "dataset_metadata_path" in inputs:
                flat["dataset_metadata_path"] = inputs["dataset_metadata_path"]
            if "preprocessed_frames_dir" in inputs:
                flat["preprocessed_frames_dir"] = inputs["preprocessed_frames_dir"]

        # Nested: outputs
        outputs = data.get("outputs", {})
        if isinstance(outputs, dict):
            if "output_dir" in outputs:
                flat["output_dir"] = outputs["output_dir"]
            if "save_format" in outputs:
                flat["save_format"] = outputs["save_format"]
            if "compress_output" in outputs:
                flat["compress_output"] = outputs["compress_output"]

        # Nested: feature_groups
        f_groups = data.get("feature_groups", {})
        if isinstance(f_groups, dict):
            for fg in [
                "include_detection_features",
                "include_motion_features",
                "include_visual_quality_features",
                "include_tracking_features",
            ]:
                if fg in f_groups:
                    flat[fg] = f_groups[fg]

        # Nested: temporal_alignment
        t_align = data.get("temporal_alignment", {})
        if isinstance(t_align, dict):
            for k in ["sort_by", "ascending", "require_timestamps", "default_fps", "time_unit", "max_sequence_gap_frames"]:
                if k in t_align:
                    flat[k] = t_align[k]

        # Nested: missing_value_policy
        mv = data.get("missing_value_policy", {})
        if isinstance(mv, dict):
            if "strategy" in mv:
                flat["missing_value_strategy"] = mv["strategy"]
            if "fail_on_missing_required" in mv:
                flat["fail_on_missing_required"] = mv["fail_on_missing_required"]
            if "fail_on_invalid_bbox" in mv:
                flat["fail_on_invalid_bbox"] = mv["fail_on_invalid_bbox"]

        # Nested: precision
        prec = data.get("precision", {})
        if isinstance(prec, dict):
            for k in ["float_precision", "decimal_places", "normalize_coordinates"]:
                if k in prec:
                    flat[k] = prec[k]

        # Nested: processing
        proc = data.get("processing", {})
        if isinstance(proc, dict):
            for k in ["batch_size", "max_workers", "memory_limit_mb", "stream_processing"]:
                if k in proc:
                    flat[k] = proc[k]

        # Nested: logging
        log_cfg = data.get("logging", {})
        if isinstance(log_cfg, dict):
            if "log_level" in log_cfg:
                flat["log_level"] = log_cfg["log_level"]
            if "strict_validation" in log_cfg:
                flat["strict_validation"] = log_cfg["strict_validation"]

        # Also accept any direct flat parameters that override nested keys
        known_fields = cls.__dataclass_fields__.keys()
        for k, v in data.items():
            if k in known_fields and not isinstance(v, dict):
                flat[k] = v

        try:
            return cls(**flat)
        except TypeError as ex:
            raise InvalidFeatureConfigError(f"Failed to instantiate FeatureConfig: {ex}") from ex

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "FeatureConfig":
        """Load and validate FeatureConfig from a YAML file.

        Args:
            path: Path to YAML configuration file. If None, resolves default path.
        """
        if path is None:
            resolved_path = _project_root / DEFAULT_CONFIG_RELATIVE_PATH
        else:
            p = Path(path)
            resolved_path = p if p.is_absolute() else _project_root / p

        if not resolved_path.exists():
            raise FeatureConfigFileNotFoundError(
                f"Feature configuration file not found at: '{resolved_path}'"
            )

        if resolved_path.is_dir():
            raise InvalidFeatureConfigError(
                f"Specified configuration path is a directory, not a file: '{resolved_path}'"
            )

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as ex:
            raise InvalidFeatureConfigError(
                f"Failed to parse YAML file '{resolved_path}': {ex}"
            ) from ex

        if content is None:
            raise InvalidFeatureConfigError(f"Configuration file '{resolved_path}' is empty.")

        if not isinstance(content, dict):
            raise InvalidFeatureConfigError(
                f"Configuration file '{resolved_path}' must contain a YAML mapping/dict, got {type(content).__name__}"
            )

        return cls.from_dict(content)

    @classmethod
    def from_preset(cls, preset_name: str) -> "FeatureConfig":
        """Instantiate a predefined configuration preset."""
        preset_key = preset_name.strip().lower()
        if preset_key not in FEATURE_PRESETS:
            raise InvalidFeatureConfigError(
                f"Unknown feature preset '{preset_name}'. Supported presets: {list(FEATURE_PRESETS.keys())}"
            )
        return FEATURE_PRESETS[preset_key]

    def to_dict(self, nested: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary (either nested YAML schema format or flat)."""
        flat = asdict(self)
        if not nested:
            return flat

        return {
            "schema_version": self.schema_version,
            "feature_set_name": self.feature_set_name,
            "inputs": {
                "detection_results_path": self.detection_results_path,
                "tracking_results_path": self.tracking_results_path,
                "dataset_metadata_path": self.dataset_metadata_path,
                "preprocessed_frames_dir": self.preprocessed_frames_dir,
            },
            "outputs": {
                "output_dir": self.output_dir,
                "save_format": self.save_format,
                "compress_output": self.compress_output,
            },
            "feature_groups": {
                "include_detection_features": self.include_detection_features,
                "include_motion_features": self.include_motion_features,
                "include_visual_quality_features": self.include_visual_quality_features,
                "include_tracking_features": self.include_tracking_features,
            },
            "temporal_alignment": {
                "sort_by": self.sort_by,
                "ascending": self.ascending,
                "require_timestamps": self.require_timestamps,
                "default_fps": self.default_fps,
                "time_unit": self.time_unit,
                "max_sequence_gap_frames": self.max_sequence_gap_frames,
            },
            "missing_value_policy": {
                "strategy": self.missing_value_strategy,
                "fail_on_missing_required": self.fail_on_missing_required,
                "fail_on_invalid_bbox": self.fail_on_invalid_bbox,
            },
            "precision": {
                "float_precision": self.float_precision,
                "decimal_places": self.decimal_places,
                "normalize_coordinates": self.normalize_coordinates,
            },
            "processing": {
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
                "memory_limit_mb": self.memory_limit_mb,
                "stream_processing": self.stream_processing,
            },
            "logging": {
                "log_level": self.log_level,
                "strict_validation": self.strict_validation,
            },
        }

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to a YAML formatted string or persist to file."""
        data = self.to_dict(nested=True)
        yaml_str = yaml.dump(data, sort_keys=False, indent=2)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str


# ==============================================================================
# Operational Presets
# ==============================================================================

FEATURE_PRESETS: Dict[str, FeatureConfig] = {
    "default": FeatureConfig(
        feature_set_name="default_balanced",
        include_detection_features=True,
        include_motion_features=True,
        include_visual_quality_features=True,
        include_tracking_features=True,
        batch_size=100,
        decimal_places=4,
    ),
    "fast_cpu": FeatureConfig(
        feature_set_name="fast_cpu_lightweight",
        include_detection_features=True,
        include_motion_features=True,
        include_visual_quality_features=False,  # Skip image decoding for speed
        include_tracking_features=True,
        batch_size=200,
        decimal_places=3,
        float_precision="float32",
    ),
    "high_precision": FeatureConfig(
        feature_set_name="high_precision_scientific",
        include_detection_features=True,
        include_motion_features=True,
        include_visual_quality_features=True,
        include_tracking_features=True,
        batch_size=50,
        decimal_places=6,
        float_precision="float64",
        strict_validation=True,
    ),
    "minimal": FeatureConfig(
        feature_set_name="minimal_detection_tracking",
        include_detection_features=True,
        include_motion_features=False,
        include_visual_quality_features=False,
        include_tracking_features=True,
        batch_size=200,
        decimal_places=4,
    ),
}


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI for validating and inspecting FeatureConfig."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Feature Engineering Configuration Validator."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file to validate (defaults to config/feature_config.yaml).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(FEATURE_PRESETS.keys()),
        help="Inspect a built-in feature configuration preset.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the configuration and exit with code 0 on success, code 1 on failure.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        if args.preset:
            cfg = FeatureConfig.from_preset(args.preset)
            logger.info("Loaded preset '%s' successfully.", args.preset)
        else:
            cfg = FeatureConfig.from_yaml(args.config)
            cfg_path = args.config or str(_project_root / DEFAULT_CONFIG_RELATIVE_PATH)
            logger.info("Loaded configuration from '%s' successfully.", cfg_path)

        print("\n" + "=" * 65)
        print("PERCEPTGUARD AI — FEATURE CONFIGURATION VALIDATION")
        print("=" * 65)
        print(f"  Schema Version             : {cfg.schema_version}")
        print(f"  Feature Set Name           : {cfg.feature_set_name}")
        print(f"  Detection Features Enabled : {cfg.include_detection_features}")
        print(f"  Motion Features Enabled    : {cfg.include_motion_features}")
        print(f"  Visual Quality Enabled     : {cfg.include_visual_quality_features}")
        print(f"  Tracking Features Enabled  : {cfg.include_tracking_features}")
        print(f"  Sort Order                 : {cfg.sort_by} (ascending={cfg.ascending})")
        print(f"  Missing Value Policy       : {cfg.missing_value_strategy}")
        print(f"  Numeric Precision          : {cfg.float_precision} (decimals: {cfg.decimal_places})")
        print(f"  CPU Batch Size / Workers   : {cfg.batch_size} / {cfg.max_workers}")
        print(f"  Memory Limit               : {cfg.memory_limit_mb} MB")
        print("  Status                     : [VALID]")
        print("=" * 65 + "\n")
        return 0

    except FeatureConfigError as ex:
        logger.error("Feature configuration validation failed: %s", ex)
        return 1
    except Exception as ex:
        logger.error("Unexpected error validating configuration: %s", ex, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
