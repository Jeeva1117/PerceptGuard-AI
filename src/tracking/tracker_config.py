"""PerceptGuard AI: Production Multi-Object Tracking Configuration Architecture.

Senior Staff-level, production-grade configuration loader and validator for ByteTrack
multi-object tracking in autonomous-driving predictive perception pipelines.

Key Architectural Guarantees:
- Strongly typed, immutable-friendly dataclass representation.
- Safe YAML deserialization with boundary and type validation.
- Complete domain constraint enforcement (track thresholds, buffer bounds, matcher cost).
- Safe path resolution relative to PerceptGuard AI repository root.
- Standalone CLI validation entrypoint (`python -m src.tracking.tracker_config --validate`).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml

logger = logging.getLogger("PerceptGuard.TrackerConfig")

# ==============================================================================
# Domain Exceptions
# ==============================================================================


class TrackingEnvironmentError(Exception):
    """Base exception for all tracking environment and configuration failures."""

    pass


class TrackerConfigError(TrackingEnvironmentError):
    """Raised when tracker configuration parameters are missing, malformed, or out of bounds."""

    pass


class TrackerConfigFileNotFoundError(TrackerConfigError, FileNotFoundError):
    """Raised when tracker configuration file does not exist at specified path."""

    pass


# ==============================================================================
# Configuration Schema Constants
# ==============================================================================

SUPPORTED_TRACKER_TYPES: Set[str] = {"bytetrack"}

REQUIRED_TRACKER_FIELDS: Set[str] = {
    "tracker_type",
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "track_buffer",
    "match_thresh",
    "fuse_score",
}

DEFAULT_CONFIG_RELATIVE_PATH: str = "config/tracker_config.yaml"

TRACKER_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
    },
    "cpu_fast": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.30,
        "track_low_thresh": 0.15,
        "new_track_thresh": 0.35,
        "track_buffer": 20,
        "match_thresh": 0.70,
        "fuse_score": False,
    },
    "high_retention": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.20,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.20,
        "track_buffer": 60,
        "match_thresh": 0.85,
        "fuse_score": True,
    },
    "strict_identity": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.40,
        "track_low_thresh": 0.15,
        "new_track_thresh": 0.45,
        "track_buffer": 15,
        "match_thresh": 0.60,
        "fuse_score": True,
    },
}


# ==============================================================================
# Tracker Configuration Dataclass
# ==============================================================================


@dataclass
class TrackerConfig:
    """ByteTrack Multi-Object Tracking Configuration Container.

    Represents validated, CPU-compatible configuration settings for ByteTrack.

    Attributes:
        tracker_type (str): Backend tracker algorithm identifier. Must be 'bytetrack'.
        track_high_thresh (float): First-stage high-confidence association threshold in (0.0, 1.0].
        track_low_thresh (float): Second-stage low-confidence association threshold in [0.0, 1.0).
            Must satisfy track_low_thresh < track_high_thresh.
        new_track_thresh (float): Minimum confidence required to initialize a new tracklet in (0.0, 1.0].
        track_buffer (int): Frame buffer retention period for lost tracks in [1, 1000].
        match_thresh (float): Maximum association cost gate (IoU distance) in (0.0, 1.0].
        fuse_score (bool): Whether to fuse detection score with spatial IoU similarity during association.
        config_path (Optional[Path]): Source path where this configuration was loaded from.
        frozen (bool): When True, prevents attribute modification to guarantee runtime immutability.
    """

    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    config_path: Optional[Path] = None
    _frozen: bool = field(default=False, repr=False)

    def __setattr__(self, key: str, value: Any) -> None:
        """Enforce post-initialization immutability when frozen."""
        if getattr(self, "_frozen", False) and key != "_frozen":
            raise TypeError(
                f"TrackerConfig instance is immutable (frozen). Cannot modify '{key}'."
            )
        super().__setattr__(key, value)

    def freeze(self) -> "TrackerConfig":
        """Freeze configuration to prevent subsequent parameter modifications."""
        object.__setattr__(self, "_frozen", True)
        return self

    @property
    def is_frozen(self) -> bool:
        """Return True if this configuration instance is frozen."""
        return self._frozen

    # --------------------------------------------------------------------------
    # Serialization & Inspection
    # --------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a clean dictionary representation."""
        return {
            "tracker_type": self.tracker_type,
            "track_high_thresh": float(self.track_high_thresh),
            "track_low_thresh": float(self.track_low_thresh),
            "new_track_thresh": float(self.new_track_thresh),
            "track_buffer": int(self.track_buffer),
            "match_thresh": float(self.match_thresh),
            "fuse_score": bool(self.fuse_score),
        }

    def to_yaml(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to a YAML formatted string or file.

        Args:
            output_path: Optional destination file path.

        Returns:
            Formatted YAML string.
        """
        payload = self.to_dict()
        yaml_str = yaml.dump(payload, default_flow_style=False, sort_keys=False)
        if output_path is not None:
            dest = Path(output_path).resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(yaml_str, encoding="utf-8")
        return yaml_str

    # --------------------------------------------------------------------------
    # Validation Architecture
    # --------------------------------------------------------------------------

    def validate(self, strict_keys: bool = False, raw_keys: Optional[Set[str]] = None) -> None:
        """Validate all configuration parameters against schema and physics boundaries.

        Args:
            strict_keys: If True and raw_keys is supplied, rejects unrecognized parameters.
            raw_keys: Set of keys parsed directly from the raw YAML dictionary.

        Raises:
            TrackerConfigError: If any parameter is missing, typed incorrectly, or out of range.
        """
        # 1. Unrecognized parameters check (when raw_keys provided)
        if strict_keys and raw_keys is not None:
            unknown = raw_keys - REQUIRED_TRACKER_FIELDS
            if unknown:
                raise TrackerConfigError(
                    f"Unsupported or unrecognized tracking parameters: {sorted(unknown)}. "
                    f"Allowed parameters are: {sorted(REQUIRED_TRACKER_FIELDS)}"
                )

        # 2. tracker_type validation
        if not isinstance(self.tracker_type, str):
            raise TrackerConfigError(
                f"'tracker_type' must be a string, got {type(self.tracker_type).__name__}: {self.tracker_type}"
            )
        normalized_tracker = self.tracker_type.strip().lower()
        if normalized_tracker not in SUPPORTED_TRACKER_TYPES:
            raise TrackerConfigError(
                f"Unsupported 'tracker_type': '{self.tracker_type}'. "
                f"Supported trackers: {sorted(SUPPORTED_TRACKER_TYPES)}"
            )

        # 3. Numeric type and range validations
        # track_high_thresh
        if isinstance(self.track_high_thresh, bool) or not isinstance(
            self.track_high_thresh, (int, float)
        ):
            raise TrackerConfigError(
                f"'track_high_thresh' must be a float, got {type(self.track_high_thresh).__name__}"
            )
        high_th = float(self.track_high_thresh)
        if not (0.0 < high_th <= 1.0):
            raise TrackerConfigError(
                f"'track_high_thresh' must be in range (0.0, 1.0], got {high_th}"
            )

        # track_low_thresh
        if isinstance(self.track_low_thresh, bool) or not isinstance(
            self.track_low_thresh, (int, float)
        ):
            raise TrackerConfigError(
                f"'track_low_thresh' must be a float, got {type(self.track_low_thresh).__name__}"
            )
        low_th = float(self.track_low_thresh)
        if not (0.0 <= low_th < 1.0):
            raise TrackerConfigError(
                f"'track_low_thresh' must be in range [0.0, 1.0), got {low_th}"
            )

        # Bounding relation between low and high
        if low_th >= high_th:
            raise TrackerConfigError(
                f"'track_low_thresh' ({low_th}) must be strictly less than 'track_high_thresh' ({high_th}). "
                "Low-confidence second stage association requires track_low_thresh < track_high_thresh."
            )

        # new_track_thresh
        if isinstance(self.new_track_thresh, bool) or not isinstance(
            self.new_track_thresh, (int, float)
        ):
            raise TrackerConfigError(
                f"'new_track_thresh' must be a float, got {type(self.new_track_thresh).__name__}"
            )
        new_th = float(self.new_track_thresh)
        if not (0.0 < new_th <= 1.0):
            raise TrackerConfigError(
                f"'new_track_thresh' must be in range (0.0, 1.0], got {new_th}"
            )

        # track_buffer
        if isinstance(self.track_buffer, bool) or not isinstance(self.track_buffer, int):
            raise TrackerConfigError(
                f"'track_buffer' must be an integer, got {type(self.track_buffer).__name__}"
            )
        if not (1 <= self.track_buffer <= 1000):
            raise TrackerConfigError(
                f"'track_buffer' must be an integer in range [1, 1000], got {self.track_buffer}"
            )

        # match_thresh
        if isinstance(self.match_thresh, bool) or not isinstance(
            self.match_thresh, (int, float)
        ):
            raise TrackerConfigError(
                f"'match_thresh' must be a float, got {type(self.match_thresh).__name__}"
            )
        match_th = float(self.match_thresh)
        if not (0.0 < match_th <= 1.0):
            raise TrackerConfigError(
                f"'match_thresh' must be in range (0.0, 1.0], got {match_th}"
            )

        # fuse_score
        if not isinstance(self.fuse_score, bool):
            raise TrackerConfigError(
                f"'fuse_score' must be a boolean (True/False), got {type(self.fuse_score).__name__}"
            )

    # --------------------------------------------------------------------------
    # Factory Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config_path: Optional[Union[str, Path]] = None,
        strict_keys: bool = False,
        freeze: bool = True,
    ) -> "TrackerConfig":
        """Instantiate and validate TrackerConfig from a raw dictionary.

        Args:
            data: Raw configuration dictionary.
            config_path: Optional file path provenance.
            strict_keys: If True, rejects unrecognized parameter keys.
            freeze: If True, freezes instance after validation.

        Returns:
            Validated TrackerConfig instance.
        """
        if not isinstance(data, dict):
            raise TrackerConfigError(
                f"Configuration payload must be a dictionary, got {type(data).__name__}"
            )

        raw_keys = set(data.keys())

        # Check for missing required fields
        missing = REQUIRED_TRACKER_FIELDS - raw_keys
        if missing:
            raise TrackerConfigError(
                f"Missing required tracking parameters: {sorted(missing)}. "
                f"Required schema fields: {sorted(REQUIRED_TRACKER_FIELDS)}"
            )

        resolved_path = Path(config_path).resolve() if config_path else None

        instance = cls(
            tracker_type=data["tracker_type"],
            track_high_thresh=data["track_high_thresh"],
            track_low_thresh=data["track_low_thresh"],
            new_track_thresh=data["new_track_thresh"],
            track_buffer=data["track_buffer"],
            match_thresh=data["match_thresh"],
            fuse_score=data["fuse_score"],
            config_path=resolved_path,
        )

        instance.validate(strict_keys=strict_keys, raw_keys=raw_keys)
        if freeze:
            instance.freeze()
        return instance

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
        strict_keys: bool = False,
        freeze: bool = True,
    ) -> "TrackerConfig":
        """Load, parse, and validate tracker configuration from a YAML file.

        Args:
            config_path: Path to YAML file. If None, defaults to `config/tracker_config.yaml`.
            project_root: Repository root path for relative path resolution.
            strict_keys: If True, rejects unrecognized parameter keys.
            freeze: If True, freezes instance after validation.

        Returns:
            Validated TrackerConfig instance.

        Raises:
            TrackerConfigFileNotFoundError: If the YAML file does not exist.
            TrackerConfigError: If YAML is malformed or parameters fail validation.
        """
        root = Path(project_root).resolve() if project_root else _project_root

        if config_path is None:
            resolved_path = (root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
        else:
            p = Path(config_path)
            resolved_path = p.resolve() if p.is_absolute() else (root / p).resolve()

        if not resolved_path.exists():
            raise TrackerConfigFileNotFoundError(
                f"Tracker configuration file not found at: {resolved_path}"
            )
        if not resolved_path.is_file():
            raise TrackerConfigError(
                f"Tracker configuration path is not a regular file: {resolved_path}"
            )

        try:
            content = resolved_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise TrackerConfigError(
                f"Malformed YAML in tracker configuration '{resolved_path}': {e}"
            ) from e
        except Exception as e:
            raise TrackerConfigError(
                f"Failed to read tracker configuration '{resolved_path}': {e}"
            ) from e

        if not isinstance(data, dict):
            raise TrackerConfigError(
                f"Tracker configuration file '{resolved_path}' must deserialize into a dictionary, "
                f"got {type(data).__name__}"
            )

        return cls.from_dict(
            data=data,
            config_path=resolved_path,
            strict_keys=strict_keys,
            freeze=freeze,
        )

    @classmethod
    def from_preset(
        cls,
        preset_name: str = "default",
        freeze: bool = True,
    ) -> "TrackerConfig":
        """Instantiate TrackerConfig from a predefined operational preset.

        Available presets:
        - 'default': Balanced ByteTrack settings for autonomous driving video perception.
        - 'cpu_fast': Lightweight settings for higher frame rates on Intel Core i3 CPU.
        - 'high_retention': Extended memory buffer for high-occlusion environments.
        - 'strict_identity': Conservative thresholding minimizing track ID switches.

        Args:
            preset_name: Key of the preset to load.
            freeze: If True, freezes instance.

        Returns:
            Validated TrackerConfig instance.
        """
        norm_name = preset_name.strip().lower()
        if norm_name not in TRACKER_PRESETS:
            raise TrackerConfigError(
                f"Unknown tracker preset: '{preset_name}'. "
                f"Available presets: {sorted(TRACKER_PRESETS.keys())}"
            )
        preset_data = deepcopy(TRACKER_PRESETS[norm_name])
        return cls.from_dict(preset_data, freeze=freeze)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for tracker configuration validation."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: ByteTrack Configuration Validator"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to tracker YAML configuration file (default: config/tracker_config.yaml)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the configuration and print status report",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(TRACKER_PRESETS.keys()),
        help="Validate a predefined operational preset",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict rejection of unrecognized YAML keys",
    )

    args = parser.parse_args(argv)

    try:
        if args.preset:
            print(f"[INFO] Validating preset: '{args.preset}'...")
            config = TrackerConfig.from_preset(args.preset)
        else:
            target_path = args.config or DEFAULT_CONFIG_RELATIVE_PATH
            print(f"[INFO] Validating tracker configuration from: '{target_path}'...")
            config = TrackerConfig.from_yaml(args.config, strict_keys=args.strict)

        print("[SUCCESS] Tracker configuration is VALID.")
        print("-" * 50)
        for k, v in config.to_dict().items():
            print(f"  {k:20s}: {v}")
        print("-" * 50)
        print(f"  Frozen (Immutable)  : {config.is_frozen}")
        if config.config_path:
            print(f"  Source Path         : {config.config_path}")
        return 0

    except TrackerConfigFileNotFoundError as e:
        print(f"[ERROR] Configuration file not found: {e}", file=sys.stderr)
        return 2
    except TrackerConfigError as e:
        print(f"[ERROR] Tracker configuration validation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected validation error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
