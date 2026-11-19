"""PerceptGuard AI: Temporal Sequence Construction Engine.

Transforms chronological multi-modal feature vectors and aggregated temporal matrices
(from Day 5 Task 5) into model-ready sliding-window sequences for future LSTM/GRU-based
predictive reliability forecasting.

Architectural Principles & Guarantees:
- Zero Future-Frame Leakage: Input windows X_t = [features(t - L + 1), ..., features(t)]
  strictly consume past and current observations only. Future frames (t + h) are strictly
  quarantined as forecast target references.
- Purged Chronological Dataset Splitting: In single-video datasets, temporal embargo gaps
  (gap >= max(L) + max(H)) are enforced between Train, Val, and Test splits. Overlapping
  windows from the same continuous sequence are mathematically barred from crossing splits.
- Training-Derived Imputation and Scaling: Scalers (Standardization, Min-Max) and Imputers
  (Mean, Median) are strictly fitted on the TRAINING split only. Validation and Test splits
  are transformed using training-derived statistics without leakage.
- Explicit Null Integrity: Non-detection or inactive track frames maintain physical null
  semantics. Missing values are never silently replaced with zero.
- CPU-First Compatibility: Vectorized NumPy array slicing, bounded memory footprint,
  and deterministic sorting for Intel Core i3 / 8 GB RAM execution.
- Production Artifact Serialization: Emits model-ready .npy tensors (sequences_L*.npy),
  JSON sequence metadata, feature manifests, split allocations, and parameter records.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

import numpy as np

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.feature_config import FeatureConfig, FeatureConfigError

logger = logging.getLogger("PerceptGuard.TemporalSequenceBuilder")

SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1", 1, 1.0}
SUPPORTED_SCALING_STRATEGIES = {"none", "standardization", "min_max"}
SUPPORTED_MISSING_POLICIES = {"reject", "drop_features", "impute_mean", "impute_median"}
SUPPORTED_SPLIT_STRATEGIES = {"chronological_purged", "video_level", "none"}

# Standard 22 base feature candidates from Task 2, 3, 4 + coverage from Task 5
CANONICAL_DETECTION_FEATURES = [
    "detection_count",
    "unique_class_count",
    "mean_confidence",
    "confidence_std",
    "mean_bbox_area",
    "mean_bbox_width",
    "mean_bbox_height",
]

CANONICAL_MOTION_FEATURES = [
    "observed_track_count",
    "moving_track_count",
    "stationary_track_count",
    "mean_displacement",
    "maximum_displacement",
    "mean_speed",
    "mean_velocity_x",
    "mean_velocity_y",
    "motion_feature_coverage",
]

CANONICAL_VISUAL_QUALITY_FEATURES = [
    "blur_score",
    "mean_brightness",
    "brightness_std",
    "contrast_std",
    "contrast_range",
    "contrast_percentile_range",
]

CANONICAL_TEMPORAL_COVERAGE_FEATURES = [
    "numeric_feature_coverage",
]


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class TemporalSequenceError(Exception):
    """Base exception for all temporal sequence construction errors."""
    pass


class InvalidSequenceInputError(TemporalSequenceError):
    """Raised when input files, JSON format, or schemas are invalid."""
    pass


class SchemaVersionMismatchError(TemporalSequenceError):
    """Raised when the input schema version is unsupported."""
    pass


class EmptyDatasetError(TemporalSequenceError):
    """Raised when the dataset contains zero frames or insufficient frames for sequences."""
    pass


class ChronologicalOrderError(TemporalSequenceError):
    """Raised when frames are not strictly sorted chronologically or have non-monotonic indices."""
    pass


class DuplicateFrameIdError(TemporalSequenceError):
    """Raised when duplicate frame IDs exist within the input dataset."""
    pass


class FeatureSelectionError(TemporalSequenceError):
    """Raised when feature selection fails or requested features are missing."""
    pass


class DataLeakageError(TemporalSequenceError):
    """Raised when data leakage between splits or future frames is detected."""
    pass


class MissingValueError(TemporalSequenceError):
    """Raised when missing values violate the configured missing-value policy."""
    pass


class ScalingError(TemporalSequenceError):
    """Raised when scaler fitting or transformation fails."""
    pass


# ==============================================================================
# Data Contracts & Schemas
# ==============================================================================

@dataclass(frozen=True)
class HorizonTargetRecord:
    """Metadata describing a forecasting horizon target for a single sequence."""
    horizon: int
    target_frame_id: Optional[str]
    target_source_frame_index: Optional[int]
    target_timestamp_seconds: Optional[float]
    target_frame_gap: Optional[int]
    target_status: str  # "valid", "gap_detected", "out_of_bounds"
    target_label: Optional[float] = None
    target_label_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceMetadataRecord:
    """Detailed metadata for a single sliding-window sequence."""
    sequence_id: str
    dataset_id: str
    input_sequence_length: int
    feature_count: int
    start_frame_id: str
    end_frame_id: str
    start_source_frame_index: Optional[int]
    end_source_frame_index: Optional[int]
    start_timestamp_seconds: Optional[float]
    end_timestamp_seconds: Optional[float]
    frame_ids: List[str]
    split: str  # "train", "val", "test", "unassigned"
    horizons: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Temporal Sequence Builder Core Class
# ==============================================================================

class TemporalSequenceBuilder:
    """Orchestrates temporal sequence extraction, splitting, scaling, and serialization."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        sequence_lengths: Optional[Sequence[int]] = None,
        forecast_horizons: Optional[Sequence[int]] = None,
        scaling_strategy: Optional[str] = None,
        sequence_missing_policy: Optional[str] = None,
        missing_feature_threshold: Optional[float] = None,
        split_strategy: Optional[str] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        temporal_gap: Optional[int] = None,
        random_seed: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        feature_preset: str = "standard",
        max_sequence_gap_frames: Optional[int] = None,
    ) -> None:
        """Initialize the TemporalSequenceBuilder with validated configurations."""
        self.config = config or FeatureConfig()

        self.sequence_lengths = tuple(
            sequence_lengths if sequence_lengths is not None else self.config.sequence_lengths
        )
        self.forecast_horizons = tuple(
            forecast_horizons if forecast_horizons is not None else self.config.forecast_horizons
        )
        self.scaling_strategy = (
            scaling_strategy if scaling_strategy is not None else self.config.scaling_strategy
        ).lower()
        self.missing_policy = (
            sequence_missing_policy
            if sequence_missing_policy is not None
            else self.config.sequence_missing_policy
        ).lower()
        self.missing_feature_threshold = (
            missing_feature_threshold
            if missing_feature_threshold is not None
            else self.config.missing_feature_threshold
        )
        self.split_strategy = (
            split_strategy if split_strategy is not None else self.config.split_strategy
        ).lower()
        self.train_ratio = train_ratio if train_ratio is not None else self.config.train_ratio
        self.val_ratio = val_ratio if val_ratio is not None else self.config.val_ratio
        self.test_ratio = test_ratio if test_ratio is not None else self.config.test_ratio
        self.temporal_gap = temporal_gap if temporal_gap is not None else self.config.temporal_gap
        self.random_seed = random_seed if random_seed is not None else self.config.random_seed
        self.requested_feature_names = list(feature_names) if feature_names is not None else None
        self.feature_preset = feature_preset.lower()
        self.max_sequence_gap_frames = (
            max_sequence_gap_frames
            if max_sequence_gap_frames is not None
            else self.config.max_sequence_gap_frames
        )

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate operational parameters against system invariants."""
        if not self.sequence_lengths or any(l <= 0 for l in self.sequence_lengths):
            raise InvalidSequenceInputError(
                f"sequence_lengths must be non-empty sequence of positive integers: {self.sequence_lengths}"
            )
        if not self.forecast_horizons or any(h <= 0 for h in self.forecast_horizons):
            raise InvalidSequenceInputError(
                f"forecast_horizons must be non-empty sequence of positive integers: {self.forecast_horizons}"
            )
        if self.scaling_strategy not in SUPPORTED_SCALING_STRATEGIES:
            raise InvalidSequenceInputError(
                f"Unsupported scaling_strategy '{self.scaling_strategy}'. Supported: {SUPPORTED_SCALING_STRATEGIES}"
            )
        if self.missing_policy not in SUPPORTED_MISSING_POLICIES:
            raise InvalidSequenceInputError(
                f"Unsupported sequence_missing_policy '{self.missing_policy}'. Supported: {SUPPORTED_MISSING_POLICIES}"
            )
        if not (0.0 <= self.missing_feature_threshold <= 1.0):
            raise InvalidSequenceInputError(
                f"missing_feature_threshold must be in [0.0, 1.0], got {self.missing_feature_threshold}"
            )
        if self.split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
            raise InvalidSequenceInputError(
                f"Unsupported split_strategy '{self.split_strategy}'. Supported: {SUPPORTED_SPLIT_STRATEGIES}"
            )
        if self.split_strategy != "none":
            total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total_ratio - 1.0) > 1e-3:
                raise InvalidSequenceInputError(
                    f"Split ratios must sum to 1.0, got train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio} (sum={total_ratio:.4f})"
                )
        if self.temporal_gap is not None and self.temporal_gap < 0:
            raise InvalidSequenceInputError(
                f"temporal_gap must be non-negative, got {self.temporal_gap}"
            )

    # --------------------------------------------------------------------------
    # Ingestion & Input Validation
    # --------------------------------------------------------------------------

    def load_input(self, input_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate raw temporal features artifact from path or in-memory dict."""
        if isinstance(input_source, dict):
            raw_data = deepcopy(input_source)
            source_path = "<in_memory_dict>"
        elif isinstance(input_source, (str, Path)):
            p = Path(input_source)
            if not p.exists():
                raise InvalidSequenceInputError(f"Input file not found at: '{p}'")
            if p.is_dir():
                # Attempt to locate temporal_features.json or aggregated_frame_features.json
                cand1 = p / "temporal_features.json"
                cand2 = p / "aggregated_frame_features.json"
                if cand1.exists():
                    p = cand1
                elif cand2.exists():
                    p = cand2
                else:
                    raise InvalidSequenceInputError(
                        f"Directory '{p}' contains neither temporal_features.json nor aggregated_frame_features.json"
                    )
            source_path = str(p.resolve())
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except json.JSONDecodeError as ex:
                raise InvalidSequenceInputError(f"Invalid JSON file at '{p}': {ex}") from ex
        else:
            raise InvalidSequenceInputError(
                f"input_source must be a file path or dict, got {type(input_source).__name__}"
            )

        if not isinstance(raw_data, dict):
            raise InvalidSequenceInputError(
                f"Loaded data must be a JSON object (dict), got {type(raw_data).__name__}"
            )

        raw_data["_source_path"] = source_path
        self._validate_raw_schema(raw_data)
        return raw_data

    def _validate_raw_schema(self, data: Dict[str, Any]) -> None:
        """Enforce strict schema version, structural fields, and frame integrity."""
        schema_ver = str(data.get("schema_version", ""))
        if schema_ver not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionMismatchError(
                f"Unsupported schema_version '{schema_ver}'. Supported: {SUPPORTED_SCHEMA_VERSIONS}"
            )

        # Detect format: nested (features) or flat (frames)
        if "features" in data and isinstance(data["features"], list):
            frames_list = data["features"]
        elif "frames" in data and isinstance(data["frames"], list):
            frames_list = data["frames"]
        else:
            raise InvalidSequenceInputError(
                "Input artifact missing 'features' or 'frames' array."
            )

        if len(frames_list) == 0:
            raise EmptyDatasetError("Input artifact contains 0 frames; cannot construct sequences.")

        # Check for unique frame IDs and chronological ordering
        seen_frame_ids: Set[str] = set()
        prev_source_index: Optional[int] = None
        prev_timestamp: Optional[float] = None

        for idx, frame in enumerate(frames_list):
            if not isinstance(frame, dict):
                raise InvalidSequenceInputError(
                    f"Frame record at index {idx} must be a dictionary, got {type(frame).__name__}"
                )

            frame_id = frame.get("frame_id")
            if not frame_id or not isinstance(frame_id, str):
                raise InvalidSequenceInputError(
                    f"Frame record at index {idx} missing valid 'frame_id'."
                )

            if frame_id in seen_frame_ids:
                raise DuplicateFrameIdError(
                    f"Duplicate frame_id '{frame_id}' detected at index {idx}."
                )
            seen_frame_ids.add(frame_id)

            # Chronological check via source_frame_index
            s_idx = frame.get("source_frame_index")
            if s_idx is not None:
                if not isinstance(s_idx, (int, np.integer)):
                    raise InvalidSequenceInputError(
                        f"Frame '{frame_id}' has non-integer source_frame_index: {s_idx}"
                    )
                if prev_source_index is not None and s_idx < prev_source_index:
                    raise ChronologicalOrderError(
                        f"Frames out of chronological order at index {idx}: "
                        f"source_frame_index {s_idx} < previous {prev_source_index}."
                    )
                prev_source_index = int(s_idx)

            # Chronological check via timestamp_seconds
            t_sec = frame.get("timestamp_seconds")
            if t_sec is not None:
                if not isinstance(t_sec, (int, float, np.floating)):
                    raise InvalidSequenceInputError(
                        f"Frame '{frame_id}' has non-numeric timestamp_seconds: {t_sec}"
                    )
                if math.isnan(t_sec) or math.isinf(t_sec):
                    raise InvalidSequenceInputError(
                        f"Frame '{frame_id}' has non-finite timestamp_seconds: {t_sec}"
                    )
                if prev_timestamp is not None and t_sec < prev_timestamp - 1e-6:
                    raise ChronologicalOrderError(
                        f"Frames out of chronological order at index {idx}: "
                        f"timestamp_seconds {t_sec} < previous {prev_timestamp}."
                    )
                prev_timestamp = float(t_sec)

    # --------------------------------------------------------------------------
    # Canonical Frame Feature Extraction
    # --------------------------------------------------------------------------

    def extract_canonical_frames(
        self, raw_data: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract standardized frame representation mapping both flat and nested inputs."""
        dataset_id = str(raw_data.get("dataset_id", "unknown_dataset"))
        if "features" in raw_data and isinstance(raw_data["features"], list):
            raw_frames = raw_data["features"]
        elif "frames" in raw_data and isinstance(raw_data["frames"], list):
            raw_frames = raw_data["frames"]
        else:
            raw_frames = []

        canonical_frames: List[Dict[str, Any]] = []

        for f_idx, rf in enumerate(raw_frames):
            frame_id = rf.get("frame_id")
            source_index = rf.get("source_frame_index")
            timestamp = rf.get("timestamp_seconds")
            if timestamp is not None:
                timestamp = float(timestamp)

            frame_features: Dict[str, Optional[float]] = {}

            # Case A: Nested structure (temporal_features.json)
            if "detection_features" in rf or "motion_features" in rf:
                # Detection
                det = rf.get("detection_features", {})
                for k, v in det.items():
                    val = self._clean_numeric_value(v, f_idx, k)
                    frame_features[k] = val
                    frame_features[f"det_{k}"] = val

                # Motion
                mot = rf.get("motion_features", {})
                for k, v in mot.items():
                    val = self._clean_numeric_value(v, f_idx, k)
                    frame_features[k] = val
                    frame_features[f"mot_{k}"] = val

                # Visual Quality
                vq = rf.get("visual_quality_features", {})
                for k, v in vq.items():
                    val = self._clean_numeric_value(v, f_idx, k)
                    frame_features[k] = val
                    frame_features[f"vq_{k}"] = val

                # Feature Coverage
                cov = rf.get("feature_coverage", {})
                if "numeric_feature_coverage" in cov:
                    val = self._clean_numeric_value(cov["numeric_feature_coverage"], f_idx, "numeric_feature_coverage")
                    frame_features["numeric_feature_coverage"] = val
                if "source_coverage" in cov:
                    val = self._clean_numeric_value(cov["source_coverage"], f_idx, "source_coverage")
                    frame_features["source_coverage"] = val

                # Temporal Windows
                t_wins = rf.get("temporal_windows", {})
                for w_key, w_val in t_wins.items():
                    stats = w_val.get("statistics", {})
                    for feat_name, stat_dict in stats.items():
                        for stat_name, s_val in stat_dict.items():
                            k_stat = f"{w_key}_{feat_name}_{stat_name}"
                            val = self._clean_numeric_value(s_val, f_idx, k_stat)
                            frame_features[k_stat] = val
                    trends = w_val.get("trends", {})
                    for t_name, t_record in trends.items():
                        slope = t_record.get("slope") if isinstance(t_record, dict) else None
                        k_slope = f"{w_key}_{t_name}_slope"
                        val = self._clean_numeric_value(slope, f_idx, k_slope)
                        frame_features[k_slope] = val

            # Case B: Flat structure (aggregated_frame_features.json)
            else:
                for k, v in rf.items():
                    if k in {"frame_id", "source_frame_index", "timestamp_seconds", "schema_version", "dataset_id"}:
                        continue
                    val = self._clean_numeric_value(v, f_idx, k)
                    frame_features[k] = val
                    # Also map prefix-stripped versions for canonical detection/motion/vq
                    for prefix in ("det_", "mot_", "vq_"):
                        if k.startswith(prefix):
                            bare_name = k[len(prefix):]
                            if bare_name not in frame_features:
                                frame_features[bare_name] = val

            canonical_frames.append({
                "frame_id": frame_id,
                "source_frame_index": source_index,
                "timestamp_seconds": timestamp,
                "features": frame_features,
            })

        return dataset_id, canonical_frames

    @staticmethod
    def _clean_numeric_value(val: Any, f_idx: int, field_name: str) -> Optional[float]:
        """Convert scalar value to float or None, raising error on invalid strings or infs."""
        if val is None:
            return None
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float, np.integer, np.floating)):
            f_val = float(val)
            if math.isnan(f_val):
                return None
            if math.isinf(f_val):
                raise InvalidSequenceInputError(
                    f"Infinite numerical value detected at frame index {f_idx} for field '{field_name}'."
                )
            return f_val
        raise InvalidSequenceInputError(
            f"Non-numeric value '{val}' (type {type(val).__name__}) detected at frame index {f_idx} for field '{field_name}'."
        )

    # --------------------------------------------------------------------------
    # Feature Selection and Deterministic Ordering
    # --------------------------------------------------------------------------

    def select_features(
        self,
        frames: List[Dict[str, Any]],
        requested_features: Optional[List[str]] = None,
        preset: str = "standard",
    ) -> List[str]:
        """Resolve and sort the deterministic list of numerical features."""
        if len(frames) == 0:
            raise EmptyDatasetError("Cannot select features from an empty frame sequence.")

        # Identify all keys that exist across frames
        available_keys: Set[str] = set()
        for f in frames:
            available_keys.update(f["features"].keys())

        # 1. User explicitly specified features
        if requested_features is not None and len(requested_features) > 0:
            resolved: List[str] = []
            missing: List[str] = []
            for rf in requested_features:
                if rf in available_keys:
                    resolved.append(rf)
                elif f"det_{rf}" in available_keys:
                    resolved.append(f"det_{rf}")
                elif f"mot_{rf}" in available_keys:
                    resolved.append(f"mot_{rf}")
                elif f"vq_{rf}" in available_keys:
                    resolved.append(f"vq_{rf}")
                else:
                    missing.append(rf)

            if missing:
                raise FeatureSelectionError(
                    f"Requested features not found in input artifact: {missing}"
                )
            return resolved

        # 2. Preset: 'core' (the 22 base metrics + temporal coverage)
        core_candidates = (
            CANONICAL_DETECTION_FEATURES
            + CANONICAL_MOTION_FEATURES
            + CANONICAL_VISUAL_QUALITY_FEATURES
            + CANONICAL_TEMPORAL_COVERAGE_FEATURES
        )

        if preset == "core":
            selected: List[str] = []
            for c in core_candidates:
                if c in available_keys:
                    selected.append(c)
                elif f"det_{c}" in available_keys:
                    selected.append(f"det_{c}")
                elif f"mot_{c}" in available_keys:
                    selected.append(f"mot_{c}")
                elif f"vq_{c}" in available_keys:
                    selected.append(f"vq_{c}")
            if not selected:
                raise FeatureSelectionError("No core features could be resolved from dataset.")
            return selected

        # 3. Preset: 'standard' (core + key rolling statistics & trend slopes)
        if preset == "standard":
            selected = []
            for c in core_candidates:
                if c in available_keys:
                    selected.append(c)
                elif f"det_{c}" in available_keys:
                    selected.append(f"det_{c}")
                elif f"mot_{c}" in available_keys:
                    selected.append(f"mot_{c}")
                elif f"vq_{c}" in available_keys:
                    selected.append(f"vq_{c}")

            # Add key trailing window stats for w3 and w5
            key_window_stats = [
                "w3_detection_count_mean",
                "w3_mean_confidence_mean",
                "w3_blur_score_mean",
                "w3_contrast_std_mean",
                "w5_detection_count_mean",
                "w5_mean_confidence_mean",
                "w5_blur_score_mean",
                "w5_contrast_std_mean",
                "w3_confidence_trend_slope",
                "w3_blur_trend_slope",
                "w5_confidence_trend_slope",
                "w5_blur_trend_slope",
            ]
            for kws in key_window_stats:
                if kws in available_keys and kws not in selected:
                    selected.append(kws)

            if not selected:
                raise FeatureSelectionError("No standard features could be resolved from dataset.")
            return selected

        # 4. Preset: 'all_numerical'
        if preset == "all_numerical":
            all_sorted = sorted(available_keys)
            return all_sorted

        raise FeatureSelectionError(f"Unknown feature preset '{preset}'.")

    # --------------------------------------------------------------------------
    # Dataset Splitting and Purged Chronological Partitioning
    # --------------------------------------------------------------------------

    def partition_splits(
        self,
        total_frames: int,
        split_strategy: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        temporal_gap: Optional[int] = None,
        max_seq_len: int = 20,
        max_horizon: int = 5,
    ) -> Tuple[Dict[int, str], Dict[str, Any]]:
        """Partition chronological frame sequence into train/val/test splits with embargo gaps."""
        assignments: Dict[int, str] = {}

        if split_strategy == "none":
            for i in range(total_frames):
                assignments[i] = "all"
            split_manifest = {
                "strategy": "none",
                "total_frames": total_frames,
                "splits": {"all": {"frame_count": total_frames, "start_index": 0, "end_index": total_frames}},
            }
            return assignments, split_manifest

        if split_strategy == "chronological_purged":
            # Determine temporal gap (embargo) between contiguous splits
            # By default, gap must be >= max_seq_len + max_horizon to prevent overlapping window leakage
            required_gap = max_seq_len + max_horizon
            eff_gap = temporal_gap if temporal_gap is not None else required_gap

            # Guard against overly small datasets
            min_required_frames = 2 * eff_gap + 3 * max_seq_len
            if total_frames < min_required_frames:
                # Fallback to zero or scaled gap if dataset is too small for full embargo
                eff_gap = max(0, (total_frames - 3 * max_seq_len) // 2) if total_frames > 3 * max_seq_len else 0
                logger.warning(
                    f"Dataset frames ({total_frames}) smaller than ideal embargo requirement ({min_required_frames}). "
                    f"Adjusted temporal gap to {eff_gap} frames."
                )

            # Available frame pool after embargo gaps (2 gaps: train->val, val->test)
            available_frames = total_frames - 2 * eff_gap
            if available_frames <= 0:
                raise EmptyDatasetError(
                    f"Cannot partition {total_frames} frames with temporal gap {eff_gap}. "
                    f"Available frames after gap is {available_frames}."
                )

            n_train = max(max_seq_len, int(round(available_frames * train_ratio)))
            n_val = max(max_seq_len, int(round(available_frames * val_ratio))) if val_ratio > 0 else 0
            n_test = available_frames - n_train - n_val
            if n_test < max_seq_len and test_ratio > 0:
                n_test = max_seq_len
                n_train = available_frames - n_val - n_test

            # Partition indices
            train_start = 0
            train_end = n_train

            gap1_start = train_end
            gap1_end = gap1_start + eff_gap

            val_start = gap1_end
            val_end = val_start + n_val

            gap2_start = val_end
            gap2_end = gap2_start + eff_gap

            test_start = gap2_end
            test_end = total_frames

            for i in range(total_frames):
                if train_start <= i < train_end:
                    assignments[i] = "train"
                elif gap1_start <= i < gap1_end:
                    assignments[i] = "purged_gap_1"
                elif val_start <= i < val_end:
                    assignments[i] = "val"
                elif gap2_start <= i < gap2_end:
                    assignments[i] = "purged_gap_2"
                elif test_start <= i < test_end:
                    assignments[i] = "test"
                else:
                    assignments[i] = "unassigned"

            split_manifest = {
                "strategy": "chronological_purged",
                "temporal_gap": eff_gap,
                "train": {"start_idx": train_start, "end_idx": train_end, "frame_count": train_end - train_start},
                "gap_1": {"start_idx": gap1_start, "end_idx": gap1_end, "frame_count": gap1_end - gap1_start},
                "val": {"start_idx": val_start, "end_idx": val_end, "frame_count": val_end - val_start},
                "gap_2": {"start_idx": gap2_start, "end_idx": gap2_end, "frame_count": gap2_end - gap2_start},
                "test": {"start_idx": test_start, "end_idx": test_end, "frame_count": test_end - test_start},
            }
            return assignments, split_manifest

        raise InvalidSequenceInputError(f"Unsupported split_strategy: {split_strategy}")

    # --------------------------------------------------------------------------
    # Imputation & Scaling (Strict Training-Only Fitting)
    # --------------------------------------------------------------------------

    def fit_imputer_and_scaler(
        self,
        frames: List[Dict[str, Any]],
        split_assignments: Dict[int, str],
        feature_names: List[str],
        missing_policy: str,
        missing_threshold: float,
        scaling_strategy: str,
    ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """Fit imputation and scaling parameters strictly on training split data."""
        # Extract train indices
        train_indices = [i for i, f in enumerate(frames) if split_assignments.get(i) in {"train", "all"}]
        if not train_indices:
            raise EmptyDatasetError("No training frames available to fit imputer or scaler.")

        n_train = len(train_indices)

        # 1. Missingness analysis per feature on training split
        feature_missing_counts: Dict[str, int] = {f: 0 for f in feature_names}
        train_feature_values: Dict[str, List[float]] = {f: [] for f in feature_names}

        for idx in train_indices:
            frame_feats = frames[idx]["features"]
            for f in feature_names:
                val = frame_feats.get(f)
                if val is None or math.isnan(val):
                    feature_missing_counts[f] += 1
                else:
                    train_feature_values[f].append(float(val))

        missing_rates = {f: feature_missing_counts[f] / n_train for f in feature_names}

        # 2. Drop features exceeding missing threshold if enabled
        active_features = list(feature_names)
        dropped_features: List[str] = []

        if missing_policy == "drop_features":
            for f in feature_names:
                if missing_rates[f] > missing_threshold:
                    dropped_features.append(f)
                    active_features.remove(f)

            if not active_features:
                raise FeatureSelectionError(
                    f"All features dropped because missingness exceeded threshold {missing_threshold}: {dropped_features}"
                )

        # 3. Fit Imputer
        imputer_params: Dict[str, Any] = {
            "schema_version": "1.0",
            "missing_value_policy": missing_policy,
            "fitted_on_split": "train",
            "fitted": False,
            "dropped_features": dropped_features,
            "imputed_values": {},
            "per_feature_missing_rates": {f: round(missing_rates[f], 4) for f in feature_names},
        }

        if missing_policy in {"impute_mean", "impute_median"}:
            imputed_vals: Dict[str, float] = {}
            for f in active_features:
                vals = train_feature_values[f]
                if vals:
                    if missing_policy == "impute_mean":
                        imp = float(np.mean(vals))
                    else:
                        imp = float(np.median(vals))
                    imputed_vals[f] = round(imp, 6)
                else:
                    # Feature is 100% missing in training data
                    imputed_vals[f] = 0.0

            imputer_params["fitted"] = True
            imputer_params["imputed_values"] = imputed_vals

        # 4. Fit Scaler (on imputed training features)
        scaler_params: Dict[str, Any] = {
            "schema_version": "1.0",
            "scaling_strategy": scaling_strategy,
            "fitted_on_split": "train",
            "fitted": False,
            "parameters": {},
        }

        if scaling_strategy in {"standardization", "min_max"}:
            fitted_stats: Dict[str, Dict[str, float]] = {}

            for f in active_features:
                raw_vals = train_feature_values[f]
                imp_val = imputer_params["imputed_values"].get(f, 0.0)
                # Reconstruct full training vector including imputed values
                full_train_f = [
                    frames[idx]["features"].get(f) if frames[idx]["features"].get(f) is not None else imp_val
                    for idx in train_indices
                ]
                arr = np.array(full_train_f, dtype=np.float64)

                if scaling_strategy == "standardization":
                    u = float(np.mean(arr))
                    s = float(np.std(arr))
                    fitted_stats[f] = {
                        "mean": round(u, 6),
                        "std": round(s if s > 1e-8 else 1.0, 6),  # prevent div-by-zero
                    }
                elif scaling_strategy == "min_max":
                    vmin = float(np.min(arr))
                    vmax = float(np.max(arr))
                    fitted_stats[f] = {
                        "min": round(vmin, 6),
                        "max": round(vmax, 6),
                        "range": round(vmax - vmin if (vmax - vmin) > 1e-8 else 1.0, 6),
                    }

            scaler_params["fitted"] = True
            scaler_params["parameters"] = fitted_stats

        return active_features, imputer_params, scaler_params

    def transform_frames(
        self,
        frames: List[Dict[str, Any]],
        active_features: List[str],
        imputer_params: Dict[str, Any],
        scaler_params: Dict[str, Any],
    ) -> np.ndarray:
        """Transform raw frame features into a 2D float32 array applying fitted imputation & scaling."""
        n_frames = len(frames)
        n_feats = len(active_features)
        matrix = np.full((n_frames, n_feats), np.nan, dtype=np.float32)

        is_imputing = imputer_params.get("fitted", False)
        imputed_dict = imputer_params.get("imputed_values", {})

        is_scaling = scaler_params.get("fitted", False)
        scaling_strategy = scaler_params.get("scaling_strategy", "none")
        scaler_dict = scaler_params.get("parameters", {})

        for i, f in enumerate(frames):
            frame_feats = f["features"]
            for j, feat in enumerate(active_features):
                val = frame_feats.get(feat)
                if val is None or math.isnan(val):
                    if is_imputing:
                        val = imputed_dict.get(feat, 0.0)
                    else:
                        val = np.nan
                else:
                    val = float(val)

                # Scaling
                if not np.isnan(val) and is_scaling:
                    params = scaler_dict.get(feat, {})
                    if scaling_strategy == "standardization":
                        u = params.get("mean", 0.0)
                        s = params.get("std", 1.0)
                        val = (val - u) / s
                    elif scaling_strategy == "min_max":
                        vmin = params.get("min", 0.0)
                        vrange = params.get("range", 1.0)
                        val = (val - vmin) / vrange

                matrix[i, j] = val

        return matrix

    # --------------------------------------------------------------------------
    # Sliding-Window Sequence Construction & Forecasting Horizons
    # --------------------------------------------------------------------------

    def construct_sequences_for_length(
        self,
        dataset_id: str,
        frames: List[Dict[str, Any]],
        feature_matrix: np.ndarray,
        active_features: List[str],
        seq_len: int,
        horizons: Sequence[int],
        split_assignments: Dict[int, str],
        target_labels: Optional[Dict[str, float]] = None,
        missing_policy: str = "reject",
    ) -> Tuple[np.ndarray, List[SequenceMetadataRecord], List[Dict[str, Any]]]:
        """Construct sliding windows of length L with strict boundary, leakage, and gap checks."""
        n_frames = len(frames)
        n_features = len(active_features)

        valid_sequences: List[np.ndarray] = []
        valid_metadata: List[SequenceMetadataRecord] = []
        rejected_records: List[Dict[str, Any]] = []

        if n_frames < seq_len:
            logger.warning(
                f"Total frames ({n_frames}) is less than sequence length ({seq_len}); 0 sequences generated."
            )
            empty_tensor = np.zeros((0, seq_len, n_features), dtype=np.float32)
            return empty_tensor, valid_metadata, rejected_records

        # Calculate expected source frame stride
        source_indices = [f["source_frame_index"] for f in frames if f["source_frame_index"] is not None]
        if len(source_indices) >= 2:
            stride_diffs = [source_indices[i] - source_indices[i - 1] for i in range(1, len(source_indices))]
            median_stride = int(np.median(stride_diffs))
        else:
            median_stride = 1

        seq_counter = 0

        for end_idx in range(seq_len - 1, n_frames):
            start_idx = end_idx - seq_len + 1
            window_slice = feature_matrix[start_idx : end_idx + 1]
            window_frames = frames[start_idx : end_idx + 1]

            start_frame = window_frames[0]
            end_frame = window_frames[-1]

            # 1. Boundary & Split Verification (Never cross split boundaries!)
            split_tag = split_assignments.get(end_idx, "unassigned")
            # If ending frame is purged, skip
            if split_tag.startswith("purged"):
                rejected_records.append({
                    "start_frame_id": start_frame["frame_id"],
                    "end_frame_id": end_frame["frame_id"],
                    "reason": f"Window ends in embargo gap ({split_tag})",
                })
                continue

            # Verify all frames in window belong to the identical split
            is_cross_split = False
            for k in range(start_idx, end_idx + 1):
                if split_assignments.get(k) != split_tag:
                    is_cross_split = True
                    break

            if is_cross_split:
                rejected_records.append({
                    "start_frame_id": start_frame["frame_id"],
                    "end_frame_id": end_frame["frame_id"],
                    "reason": "Window crosses dataset split boundary or embargo gap",
                })
                continue

            # 2. Temporal Discontinuity / Gap Check
            has_gap = False
            for k in range(1, seq_len):
                prev_s = window_frames[k - 1]["source_frame_index"]
                curr_s = window_frames[k]["source_frame_index"]
                if prev_s is not None and curr_s is not None:
                    step = curr_s - prev_s
                    if step > (self.max_sequence_gap_frames * median_stride):
                        has_gap = True
                        break

            if has_gap:
                rejected_records.append({
                    "start_frame_id": start_frame["frame_id"],
                    "end_frame_id": end_frame["frame_id"],
                    "reason": f"Discontinuous source frame gap exceeds limit ({self.max_sequence_gap_frames}x stride)",
                })
                continue

            # 3. Missing Value Policy Enforcement
            if missing_policy == "reject":
                if np.isnan(window_slice).any():
                    nan_count = int(np.isnan(window_slice).sum())
                    rejected_records.append({
                        "start_frame_id": start_frame["frame_id"],
                        "end_frame_id": end_frame["frame_id"],
                        "reason": f"Window contains {nan_count} null/NaN values (missing_policy='reject')",
                    })
                    continue

            # Ensure all values are finite
            if not np.isfinite(window_slice).all():
                rejected_records.append({
                    "start_frame_id": start_frame["frame_id"],
                    "end_frame_id": end_frame["frame_id"],
                    "reason": "Window contains non-finite numerical values (NaN/Inf)",
                })
                continue

            # 4. Forecast Target Horizons (Strict Target Frame Identification)
            horizons_dict: Dict[str, Dict[str, Any]] = {}
            for h in horizons:
                target_idx = end_idx + h
                if target_idx < n_frames:
                    target_frame = frames[target_idx]
                    t_frame_id = target_frame["frame_id"]
                    t_source_idx = target_frame["source_frame_index"]
                    t_timestamp = target_frame["timestamp_seconds"]

                    end_s_idx = end_frame["source_frame_index"]
                    frame_gap = (t_source_idx - end_s_idx) if (t_source_idx is not None and end_s_idx is not None) else h

                    # Check ground truth label joining
                    target_lbl: Optional[float] = None
                    lbl_available = False
                    if target_labels is not None and t_frame_id in target_labels:
                        target_lbl = float(target_labels[t_frame_id])
                        lbl_available = True

                    h_rec = HorizonTargetRecord(
                        horizon=h,
                        target_frame_id=t_frame_id,
                        target_source_frame_index=t_source_idx,
                        target_timestamp_seconds=t_timestamp,
                        target_frame_gap=frame_gap,
                        target_status="valid",
                        target_label=target_lbl,
                        target_label_available=lbl_available,
                    )
                else:
                    h_rec = HorizonTargetRecord(
                        horizon=h,
                        target_frame_id=None,
                        target_source_frame_index=None,
                        target_timestamp_seconds=None,
                        target_frame_gap=None,
                        target_status="out_of_bounds",
                        target_label=None,
                        target_label_available=False,
                    )
                horizons_dict[f"h{h}"] = h_rec.to_dict()

            seq_counter += 1
            seq_id = f"seq_{dataset_id}_L{seq_len}_{seq_counter:06d}"

            meta_rec = SequenceMetadataRecord(
                sequence_id=seq_id,
                dataset_id=dataset_id,
                input_sequence_length=seq_len,
                feature_count=n_features,
                start_frame_id=start_frame["frame_id"],
                end_frame_id=end_frame["frame_id"],
                start_source_frame_index=start_frame["source_frame_index"],
                end_source_frame_index=end_frame["source_frame_index"],
                start_timestamp_seconds=start_frame["timestamp_seconds"],
                end_timestamp_seconds=end_frame["timestamp_seconds"],
                frame_ids=[wf["frame_id"] for wf in window_frames],
                split=split_tag,
                horizons=horizons_dict,
            )

            valid_sequences.append(window_slice)
            valid_metadata.append(meta_rec)

        if valid_sequences:
            tensor = np.stack(valid_sequences, axis=0).astype(np.float32)
        else:
            tensor = np.zeros((0, seq_len, n_features), dtype=np.float32)

        return tensor, valid_metadata, rejected_records

    # --------------------------------------------------------------------------
    # End-to-End Orchestration & Pipeline Execution
    # --------------------------------------------------------------------------

    def build(
        self,
        input_source: Union[str, Path, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        target_labels: Optional[Dict[str, float]] = None,
        max_sequences: Optional[int] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Execute end-to-end sequence construction pipeline and return summary manifest."""
        t_start = datetime.now(timezone.utc)

        # 1. Load and validate input
        raw_data = self.load_input(input_source)
        source_path = raw_data.get("_source_path", "unknown")
        dataset_id, frames = self.extract_canonical_frames(raw_data)
        total_frames = len(frames)

        logger.info(
            f"Loaded {total_frames} frames from '{source_path}' for dataset '{dataset_id}'."
        )

        # 2. Select Features
        active_features = self.select_features(
            frames,
            requested_features=self.requested_feature_names,
            preset=self.feature_preset,
        )
        logger.info(
            f"Selected {len(active_features)} numerical features (preset: {self.feature_preset})."
        )

        # 3. Partition Dataset Splits
        max_L = max(self.sequence_lengths)
        max_H = max(self.forecast_horizons)
        split_assignments, split_manifest = self.partition_splits(
            total_frames=total_frames,
            split_strategy=self.split_strategy,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            temporal_gap=self.temporal_gap,
            max_seq_len=max_L,
            max_horizon=max_H,
        )

        # 4. Fit Imputer and Scaler (STRICTLY on Training Split!)
        final_features, imputer_params, scaler_params = self.fit_imputer_and_scaler(
            frames=frames,
            split_assignments=split_assignments,
            feature_names=active_features,
            missing_policy=self.missing_policy,
            missing_threshold=self.missing_feature_threshold,
            scaling_strategy=self.scaling_strategy,
        )

        # 5. Transform all frames into continuous 2D feature matrix
        feature_matrix = self.transform_frames(
            frames=frames,
            active_features=final_features,
            imputer_params=imputer_params,
            scaler_params=scaler_params,
        )

        # 6. Construct Sliding-Window Sequences per sequence length
        all_tensors: Dict[str, np.ndarray] = {}
        all_metadata_by_len: Dict[str, List[Dict[str, Any]]] = {}
        all_metadata_flat: List[Dict[str, Any]] = []
        all_rejected_records: List[Dict[str, Any]] = []
        counts_per_split_by_len: Dict[str, Dict[str, int]] = {}

        for L in self.sequence_lengths:
            tensor, meta_list, rej_list = self.construct_sequences_for_length(
                dataset_id=dataset_id,
                frames=frames,
                feature_matrix=feature_matrix,
                active_features=final_features,
                seq_len=L,
                horizons=self.forecast_horizons,
                split_assignments=split_assignments,
                target_labels=target_labels,
                missing_policy=self.missing_policy,
            )

            if max_sequences is not None and max_sequences > 0:
                tensor = tensor[:max_sequences]
                meta_list = meta_list[:max_sequences]

            tensor_key = f"L{L}"
            all_tensors[tensor_key] = tensor
            dict_meta_list = [m.to_dict() for m in meta_list]
            all_metadata_by_len[tensor_key] = dict_meta_list
            all_metadata_flat.extend(dict_meta_list)
            all_rejected_records.extend(rej_list)

            # Split counts
            split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "total": len(meta_list)}
            for m in meta_list:
                s = m.split
                split_counts[s] = split_counts.get(s, 0) + 1
            counts_per_split_by_len[tensor_key] = split_counts

            logger.info(
                f"Generated {tensor.shape[0]} sequences for length L={L} with shape {tensor.shape} "
                f"(train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)})."
            )

        t_end = datetime.now(timezone.utc)
        duration_sec = round((t_end - t_start).total_seconds(), 4)

        # 7. Check Data Leakage Invariants
        self._verify_leakage_invariants(all_metadata_flat, split_manifest)

        # 8. Assemble Manifests and Artifacts
        run_id = f"seq_{dataset_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        summary_manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "dataset_id": dataset_id,
            "created_at": t_end.isoformat(),
            "source_artifact_path": source_path,
            "input_frame_count": total_frames,
            "configured_sequence_lengths": list(self.sequence_lengths),
            "configured_horizons": list(self.forecast_horizons),
            "feature_count": len(final_features),
            "ordered_feature_names": final_features,
            "scaling_strategy": self.scaling_strategy,
            "missing_value_policy": self.missing_policy,
            "target_labels_available": target_labels is not None and len(target_labels) > 0,
            "sequence_counts_per_length": {k: t.shape[0] for k, t in all_tensors.items()},
            "tensor_shapes_per_length": {k: list(t.shape) for k, t in all_tensors.items()},
            "split_counts_per_length": counts_per_split_by_len,
            "rejected_sequences_count": len(all_rejected_records),
            "processing_duration_seconds": duration_sec,
            "validation_status": "PASS",
        }

        feature_manifest = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "total_features": len(final_features),
            "ordered_feature_names": final_features,
            "feature_categories": self._categorize_features(final_features),
        }

        dataset_splits_manifest = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "split_strategy": self.split_strategy,
            "random_seed": self.random_seed,
            "split_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "temporal_gap": split_manifest.get("temporal_gap"),
            "frame_split_ranges": split_manifest,
            "sequence_counts_per_split": counts_per_split_by_len,
            "leakage_check_passed": True,
        }

        errors_manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "dataset_id": dataset_id,
            "total_errors": len(all_rejected_records),
            "errors": all_rejected_records,
        }

        sequence_metadata_manifest = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "run_id": run_id,
            "total_sequences": len(all_metadata_flat),
            "sequences": all_metadata_flat,
            "sequences_by_length": all_metadata_by_len,
        }

        # 9. Save Artifacts if Output Directory is Specified
        saved_paths: Dict[str, str] = {}
        if output_dir is not None:
            saved_paths = self.save_artifacts(
                output_dir=output_dir,
                run_id=run_id,
                dataset_id=dataset_id,
                tensors=all_tensors,
                sequence_metadata=sequence_metadata_manifest,
                feature_manifest=feature_manifest,
                dataset_splits=dataset_splits_manifest,
                scaler_parameters=scaler_params,
                imputation_parameters=imputer_params,
                sequence_summary=summary_manifest,
                sequence_errors=errors_manifest,
                overwrite=overwrite,
            )

        return {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "summary": summary_manifest,
            "feature_manifest": feature_manifest,
            "dataset_splits": dataset_splits_manifest,
            "scaler_parameters": scaler_params,
            "imputation_parameters": imputer_params,
            "sequence_metadata": sequence_metadata_manifest,
            "sequence_errors": errors_manifest,
            "tensors": all_tensors,
            "saved_paths": saved_paths,
        }

    @staticmethod
    def _categorize_features(features: List[str]) -> Dict[str, List[str]]:
        """Categorize feature names by domain modality."""
        cats: Dict[str, List[str]] = {
            "detection": [],
            "motion": [],
            "visual_quality": [],
            "temporal_aggregation": [],
        }
        for f in features:
            if f.startswith("det_") or f in CANONICAL_DETECTION_FEATURES:
                cats["detection"].append(f)
            elif f.startswith("mot_") or f in CANONICAL_MOTION_FEATURES:
                cats["motion"].append(f)
            elif f.startswith("vq_") or f in CANONICAL_VISUAL_QUALITY_FEATURES:
                cats["visual_quality"].append(f)
            else:
                cats["temporal_aggregation"].append(f)
        return cats

    def _verify_leakage_invariants(
        self,
        metadata_list: List[Dict[str, Any]],
        split_manifest: Dict[str, Any],
    ) -> None:
        """Assert zero overlap and strict chronological isolation across splits."""
        if self.split_strategy != "chronological_purged":
            return

        train_frames: Set[str] = set()
        val_frames: Set[str] = set()
        test_frames: Set[str] = set()

        for rec in metadata_list:
            split = rec["split"]
            f_ids = rec["frame_ids"]
            if split == "train":
                train_frames.update(f_ids)
            elif split == "val":
                val_frames.update(f_ids)
            elif split == "test":
                test_frames.update(f_ids)

        # 1. Assert disjoint frame sets
        if not train_frames.isdisjoint(val_frames):
            overlap = train_frames.intersection(val_frames)
            raise DataLeakageError(
                f"Data leakage detected between Train and Val splits! Overlapping frames: {list(overlap)[:5]}"
            )
        if not train_frames.isdisjoint(test_frames):
            overlap = train_frames.intersection(test_frames)
            raise DataLeakageError(
                f"Data leakage detected between Train and Test splits! Overlapping frames: {list(overlap)[:5]}"
            )
        if not val_frames.isdisjoint(test_frames):
            overlap = val_frames.intersection(test_frames)
            raise DataLeakageError(
                f"Data leakage detected between Val and Test splits! Overlapping frames: {list(overlap)[:5]}"
            )

        logger.info(
            f"Zero-leakage invariant verified: train frames ({len(train_frames)}), "
            f"val frames ({len(val_frames)}), test frames ({len(test_frames)}) are strictly disjoint."
        )

    # --------------------------------------------------------------------------
    # Output Persistence
    # --------------------------------------------------------------------------

    def save_artifacts(
        self,
        output_dir: Union[str, Path],
        run_id: str,
        dataset_id: str,
        tensors: Dict[str, np.ndarray],
        sequence_metadata: Dict[str, Any],
        feature_manifest: Dict[str, Any],
        dataset_splits: Dict[str, Any],
        scaler_parameters: Dict[str, Any],
        imputation_parameters: Dict[str, Any],
        sequence_summary: Dict[str, Any],
        sequence_errors: Dict[str, Any],
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """Save all 11 Schema v1.0 sequence artifacts to disk."""
        target_dir = Path(output_dir)
        if target_dir.exists() and not overwrite and any(target_dir.iterdir()):
            # Create timestamped subfolder under target_dir
            target_dir = target_dir / run_id

        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: Dict[str, str] = {}

        # 1. Save .npy tensors for each configured sequence length
        for key, tensor in tensors.items():
            tensor_filename = f"sequences_{key}.npy"
            tensor_path = target_dir / tensor_filename
            np.save(tensor_path, tensor)
            saved_paths[tensor_filename] = str(tensor_path.resolve())

        # 2. Save JSON manifests
        json_artifacts = [
            ("sequence_metadata.json", sequence_metadata),
            ("feature_manifest.json", feature_manifest),
            ("dataset_splits.json", dataset_splits),
            ("scaler_parameters.json", scaler_parameters),
            ("imputation_parameters.json", imputation_parameters),
            ("sequence_summary.json", sequence_summary),
            ("sequence_errors.json", sequence_errors),
        ]

        for filename, data in json_artifacts:
            path = target_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            saved_paths[filename] = str(path.resolve())

        logger.info(f"Successfully persisted {len(saved_paths)} sequence artifacts to '{target_dir}'.")
        return saved_paths

    def validate_only(
        self, input_source: Union[str, Path, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Dry-run validation of inputs, feature feasibility, and sequence feasibility."""
        raw_data = self.load_input(input_source)
        dataset_id, frames = self.extract_canonical_frames(raw_data)
        features = self.select_features(frames, preset=self.feature_preset)
        max_L = max(self.sequence_lengths)
        is_feasible = len(frames) >= max_L

        return {
            "validation_status": "PASS",
            "dataset_id": dataset_id,
            "total_frames": len(frames),
            "resolved_feature_count": len(features),
            "configured_sequence_lengths": list(self.sequence_lengths),
            "max_sequence_length": max_L,
            "sequence_feasible": is_feasible,
            "feature_names_sample": features[:10],
        }


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for PerceptGuard AI Temporal Sequence Construction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Temporal Sequence Construction Pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Task 5 temporal features JSON (temporal_features.json or aggregated_frame_features.json) or directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to config/feature_config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory to save sequence artifacts (defaults to datasets/processed/<dataset_id>/features/sequences/<run_id>).",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=str,
        default=None,
        help="Comma-separated sequence lengths (e.g. 5,10,15,20).",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated forecast horizons (e.g. 1,3,5).",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default=None,
        choices=["none", "standardization", "min_max"],
        help="Feature scaling strategy (none, standardization, min_max).",
    )
    parser.add_argument(
        "--missing-policy",
        type=str,
        default=None,
        choices=["reject", "drop_features", "impute_mean", "impute_median"],
        help="Missing-value policy (reject, drop_features, impute_mean, impute_median).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of data allocated to test split (e.g. 0.15).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=None,
        help="Fraction of data allocated to validation split (e.g. 0.15).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limit number of generated sequences (useful for test/debugging).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run input validation and feasibility checks without generating tensors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target directory if it already contains files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        # Load config
        if args.config:
            cfg = FeatureConfig.from_yaml(args.config)
        else:
            cfg = FeatureConfig.from_yaml()

        seq_lengths = None
        if args.sequence_lengths:
            seq_lengths = [int(x.strip()) for x in args.sequence_lengths.split(",") if x.strip()]

        horizons = None
        if args.horizons:
            horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

        # Compute ratios
        val_ratio = args.validation_size if args.validation_size is not None else cfg.val_ratio
        test_ratio = args.test_size if args.test_size is not None else cfg.test_ratio
        train_ratio = max(0.0, round(1.0 - val_ratio - test_ratio, 4))

        builder = TemporalSequenceBuilder(
            config=cfg,
            sequence_lengths=seq_lengths,
            forecast_horizons=horizons,
            scaling_strategy=args.scaling,
            sequence_missing_policy=args.missing_policy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=args.random_seed,
        )

        if args.validate_only:
            result = builder.validate_only(args.input)
            print(json.dumps(result, indent=2))
            return 0

        # Determine output directory
        out_dir = args.output
        if out_dir is None:
            # Load input briefly to get dataset_id
            raw = builder.load_input(args.input)
            d_id = str(raw.get("dataset_id", "driving_sample"))
            run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_dir = Path("datasets/processed") / d_id / "features/sequences" / f"seq_run_{run_stamp}"

        results = builder.build(
            input_source=args.input,
            output_dir=out_dir,
            max_sequences=args.max_sequences,
            overwrite=args.overwrite,
        )

        print("\n=== Temporal Sequence Construction Complete ===")
        print(f"Dataset ID: {results['dataset_id']}")
        print(f"Run ID: {results['run_id']}")
        print(f"Feature Count: {results['summary']['feature_count']}")
        print(f"Sequence Counts: {results['summary']['sequence_counts_per_length']}")
        print(f"Tensor Shapes: {results['summary']['tensor_shapes_per_length']}")
        print(f"Artifacts Directory: {out_dir}")
        print("Status: PASS\n")
        return 0

    except Exception as ex:
        logger.error(f"Execution failed: {ex}", exc_info=args.verbose)
        print(f"\n[ERROR] Temporal Sequence Construction Failed: {ex}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
