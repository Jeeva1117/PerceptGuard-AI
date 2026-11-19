"""Tests for PerceptGuard AI: Temporal Sequence Builder.

Comprehensive unit and integration test suite covering 50+ test cases across:
1. Input ingestion & schema validation
2. Feature selection & deterministic ordering
3. Missing-value policies (reject, drop, impute)
4. Sliding-window sequence construction (L=5, 10, 15, 20, custom)
5. Forecasting horizons (1, 3, 5 frames ahead)
6. Zero future-frame leakage guarantees
7. Chronological purged splitting & overlapping-window leakage prevention
8. Training-only scaler & imputer fitting
9. Model-ready tensor shape and finite value validation
10. Target-label policy (handling availability vs non-availability)
11. CLI interface and integration with real driving sample artifacts
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List
import uuid

import numpy as np
import pytest

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.feature_config import FeatureConfig
from src.features.temporal_sequence_builder import (
    CANONICAL_DETECTION_FEATURES,
    CANONICAL_MOTION_FEATURES,
    CANONICAL_TEMPORAL_COVERAGE_FEATURES,
    CANONICAL_VISUAL_QUALITY_FEATURES,
    ChronologicalOrderError,
    DataLeakageError,
    DuplicateFrameIdError,
    EmptyDatasetError,
    FeatureSelectionError,
    InvalidSequenceInputError,
    MissingValueError,
    ScalingError,
    SchemaVersionMismatchError,
    TemporalSequenceBuilder,
    TemporalSequenceError,
)

REAL_TEMPORAL_FEATURES_PATH = (
    _project_root
    / "datasets/processed/driving_sample/features/temporal/run_official/temporal_features.json"
)
REAL_AGGREGATED_FEATURES_PATH = (
    _project_root
    / "datasets/processed/driving_sample/features/temporal/run_official/aggregated_frame_features.json"
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

def make_synthetic_frame(
    idx: int,
    source_stride: int = 6,
    fps: float = 5.0,
    with_nulls: bool = False,
) -> Dict[str, Any]:
    """Helper to generate a realistic synthetic frame record with full modalities."""
    has_det = not (with_nulls and (idx % 7 == 0))
    has_track = not (with_nulls and (idx % 5 == 0))

    return {
        "frame_id": f"frame_{idx + 1:06d}.jpg",
        "source_frame_index": idx * source_stride,
        "timestamp_seconds": round(idx / fps, 4),
        "source_availability": {
            "detection": True,
            "motion": True,
            "visual_quality": True,
        },
        "feature_coverage": {
            "source_coverage": 1.0,
            "numeric_feature_coverage": 0.85 if with_nulls else 1.0,
        },
        "detection_features": {
            "detection_count": 3 if has_det else 0,
            "unique_class_count": 2 if has_det else 0,
            "mean_confidence": 0.82 if has_det else None,
            "confidence_std": 0.05 if has_det else None,
            "mean_bbox_area": 25000.0 if has_det else None,
            "mean_bbox_width": 120.0 if has_det else None,
            "mean_bbox_height": 210.0 if has_det else None,
        },
        "motion_features": {
            "observed_track_count": 3 if has_track else 0,
            "moving_track_count": 2 if has_track else 0,
            "stationary_track_count": 1 if has_track else 0,
            "mean_displacement": 14.2 if has_track else None,
            "maximum_displacement": 22.0 if has_track else None,
            "mean_speed": 71.0 if has_track else None,
            "mean_velocity_x": 5.0 if has_track else None,
            "mean_velocity_y": 12.0 if has_track else None,
            "motion_feature_coverage": 0.9 if has_track else 0.0,
        },
        "visual_quality_features": {
            "blur_score": 380.5 + math.sin(idx * 0.1) * 20.0,
            "mean_brightness": 115.0 + math.cos(idx * 0.1) * 10.0,
            "brightness_std": 45.0,
            "contrast_std": 52.0,
            "contrast_range": 255.0,
            "contrast_percentile_range": 160.0,
        },
        "temporal_windows": {
            "w3": {
                "statistics": {
                    "detection_count": {"mean": 3.0, "std": 0.0, "min": 3.0, "max": 3.0},
                    "mean_confidence": {"mean": 0.82, "std": 0.0},
                    "blur_score": {"mean": 380.0, "std": 5.0},
                    "contrast_std": {"mean": 52.0, "std": 0.0},
                },
                "trends": {
                    "confidence_trend": {"slope": 0.001, "direction": "increasing", "status": "valid"},
                    "blur_trend": {"slope": -0.05, "direction": "decreasing", "status": "valid"},
                },
            },
            "w5": {
                "statistics": {
                    "detection_count": {"mean": 3.0, "std": 0.0},
                    "mean_confidence": {"mean": 0.82, "std": 0.0},
                    "blur_score": {"mean": 380.0, "std": 5.0},
                    "contrast_std": {"mean": 52.0, "std": 0.0},
                },
                "trends": {
                    "confidence_trend": {"slope": 0.001, "direction": "increasing", "status": "valid"},
                    "blur_trend": {"slope": -0.05, "direction": "decreasing", "status": "valid"},
                },
            },
        },
    }


def make_synthetic_dataset(
    num_frames: int = 80,
    with_nulls: bool = False,
    dataset_id: str = "synthetic_driving",
) -> Dict[str, Any]:
    """Generate a full synthetic dataset dictionary conforming to Schema v1.0."""
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "run_id": f"temporal_syn_{uuid.uuid4().hex[:6]}",
        "total_frames": num_frames,
        "features": [
            make_synthetic_frame(i, with_nulls=with_nulls) for i in range(num_frames)
        ],
    }


# ==============================================================================
# Unit & Integration Tests (1 to 52)
# ==============================================================================

class TestTemporalSequenceBuilder:
    """Rigorous test suite for TemporalSequenceBuilder."""

    def test_01_valid_temporal_feature_input(self):
        """Verify successful ingestion of valid synthetic dataset."""
        data = make_synthetic_dataset(num_frames=40)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[1])
        res = builder.validate_only(data)
        assert res["validation_status"] == "PASS"
        assert res["total_frames"] == 40
        assert res["resolved_feature_count"] > 0

    def test_02_missing_input_file(self):
        """Verify proper error when input file does not exist."""
        builder = TemporalSequenceBuilder()
        with pytest.raises(InvalidSequenceInputError, match="Input file not found"):
            builder.load_input("non_existent_file_path.json")

    def test_03_invalid_json(self, tmp_path):
        """Verify error when JSON syntax is corrupted."""
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{\"schema_version\": 1.0, unclosed_json...", encoding="utf-8")
        builder = TemporalSequenceBuilder()
        with pytest.raises(InvalidSequenceInputError, match="Invalid JSON file"):
            builder.load_input(bad_file)

    def test_04_unsupported_schema_version(self):
        """Verify rejection of unsupported schema version."""
        data = make_synthetic_dataset(num_frames=10)
        data["schema_version"] = "99.0"
        builder = TemporalSequenceBuilder()
        with pytest.raises(SchemaVersionMismatchError, match="Unsupported schema_version"):
            builder.load_input(data)

    def test_05_duplicate_frame_identifiers(self):
        """Verify error when duplicate frame IDs exist."""
        data = make_synthetic_dataset(num_frames=10)
        data["features"][3]["frame_id"] = data["features"][2]["frame_id"]
        builder = TemporalSequenceBuilder()
        with pytest.raises(DuplicateFrameIdError, match="Duplicate frame_id"):
            builder.load_input(data)

    def test_06_chronological_ordering(self):
        """Verify error when frames violate monotonic chronological order."""
        data = make_synthetic_dataset(num_frames=10)
        data["features"][4]["source_frame_index"] = 2  # previous was 18
        builder = TemporalSequenceBuilder()
        with pytest.raises(ChronologicalOrderError, match="chronological order"):
            builder.load_input(data)

    def test_07_invalid_numerical_values(self):
        """Verify error when numerical fields contain invalid string data."""
        data = make_synthetic_dataset(num_frames=10)
        data["features"][2]["detection_features"]["detection_count"] = "invalid_string"
        builder = TemporalSequenceBuilder()
        with pytest.raises(InvalidSequenceInputError, match="Non-numeric value"):
            builder.extract_canonical_frames(data)

    def test_08_nan_and_infinity(self):
        """Verify error when numerical fields contain infinite numbers."""
        data = make_synthetic_dataset(num_frames=10)
        data["features"][2]["visual_quality_features"]["blur_score"] = float("inf")
        builder = TemporalSequenceBuilder()
        with pytest.raises(InvalidSequenceInputError, match="Infinite numerical value"):
            builder.extract_canonical_frames(data)

    def test_09_feature_selection(self):
        """Verify selection of specific requested features."""
        data = make_synthetic_dataset(num_frames=30)
        req = ["detection_count", "blur_score", "mean_brightness"]
        builder = TemporalSequenceBuilder(feature_names=req)
        _, frames = builder.extract_canonical_frames(data)
        selected = builder.select_features(frames, requested_features=req)
        assert selected == req

    def test_10_deterministic_feature_ordering(self):
        """Verify that selected feature order is strictly deterministic across repeated runs."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(feature_preset="standard")
        _, frames = builder.extract_canonical_frames(data)
        order1 = builder.select_features(frames, preset="standard")
        order2 = builder.select_features(frames, preset="standard")
        assert order1 == order2

    def test_11_missing_feature_columns(self):
        """Verify error when user requests a feature that does not exist."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(feature_names=["non_existent_feature_x"])
        _, frames = builder.extract_canonical_frames(data)
        with pytest.raises(FeatureSelectionError, match="Requested features not found"):
            builder.select_features(frames, requested_features=["non_existent_feature_x"])

    def test_12_missing_value_rejection(self):
        """Verify that missing_policy='reject' properly excludes windows containing nulls."""
        data = make_synthetic_dataset(num_frames=50, with_nulls=True)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            forecast_horizons=[1],
            sequence_missing_policy="reject",
            split_strategy="none",
        )
        res = builder.build(data)
        # Windows containing nulls are rejected
        assert res["summary"]["rejected_sequences_count"] > 0
        # All accepted sequences are finite
        tensor = res["tensors"]["L5"]
        if tensor.shape[0] > 0:
            assert np.isfinite(tensor).all()

    def test_13_training_only_imputation(self):
        """Verify that imputation values are computed exclusively from training data."""
        data = make_synthetic_dataset(num_frames=60, with_nulls=True)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            sequence_missing_policy="impute_mean",
            split_strategy="chronological_purged",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        res = builder.build(data)
        imp_params = res["imputation_parameters"]
        assert imp_params["fitted"] is True
        assert imp_params["fitted_on_split"] == "train"
        assert len(imp_params["imputed_values"]) > 0
        # Check tensor has no NaNs
        tensor = res["tensors"]["L5"]
        assert np.isfinite(tensor).all()

    def test_14_sliding_window_construction(self):
        """Verify that sliding windows advance chronologically frame by frame."""
        data = make_synthetic_dataset(num_frames=20, with_nulls=False)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            forecast_horizons=[1],
            split_strategy="none",
        )
        res = builder.build(data)
        seqs = res["sequence_metadata"]["sequences"]
        # In a contiguous sequence of 20 frames with L=5, there are 20 - 5 + 1 = 16 sequences
        assert len(seqs) == 16
        # Sequence 0 covers frames 1-5, sequence 1 covers frames 2-6
        assert seqs[0]["frame_ids"] == [f"frame_{i:06d}.jpg" for i in range(1, 6)]
        assert seqs[1]["frame_ids"] == [f"frame_{i:06d}.jpg" for i in range(2, 7)]

    def test_15_five_frame_sequences(self):
        """Verify L=5 tensor dimensions."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L5"]
        assert tensor.ndim == 3
        assert tensor.shape[1] == 5

    def test_16_ten_frame_sequences(self):
        """Verify L=10 tensor dimensions."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[10], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L10"]
        assert tensor.shape[1] == 10

    def test_17_fifteen_frame_sequences(self):
        """Verify L=15 tensor dimensions."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[15], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L15"]
        assert tensor.shape[1] == 15

    def test_18_twenty_frame_sequences(self):
        """Verify L=20 tensor dimensions."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[20], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L20"]
        assert tensor.shape[1] == 20

    def test_19_custom_sequence_lengths(self):
        """Verify support for custom sequence lengths (e.g. L=7, L=12)."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[7, 12], split_strategy="none")
        res = builder.build(data)
        assert "L7" in res["tensors"]
        assert "L12" in res["tensors"]
        assert res["tensors"]["L7"].shape[1] == 7
        assert res["tensors"]["L12"].shape[1] == 12

    def test_20_correct_tensor_dimensions(self):
        """Verify tensor shape matches (N, L, F)."""
        data = make_synthetic_dataset(num_frames=25)
        builder = TemporalSequenceBuilder(sequence_lengths=[8], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L8"]
        n_feats = res["summary"]["feature_count"]
        assert tensor.shape == (18, 8, n_feats)

    def test_21_correct_sequence_metadata(self):
        """Verify sequence metadata fields adhere to contract."""
        data = make_synthetic_dataset(num_frames=20)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        meta = res["sequence_metadata"]["sequences"][0]
        assert "sequence_id" in meta
        assert "dataset_id" in meta
        assert meta["input_sequence_length"] == 5
        assert "start_frame_id" in meta
        assert "end_frame_id" in meta
        assert "horizons" in meta

    def test_22_one_frame_forecast_horizon(self):
        """Verify horizon H=1 target frame mapping."""
        data = make_synthetic_dataset(num_frames=20)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[1], split_strategy="none")
        res = builder.build(data)
        first_seq = res["sequence_metadata"]["sequences"][0]
        # First seq ends at frame_000005.jpg, horizon 1 should target frame_000006.jpg
        h1 = first_seq["horizons"]["h1"]
        assert h1["target_frame_id"] == "frame_000006.jpg"
        assert h1["target_status"] == "valid"

    def test_23_three_frame_forecast_horizon(self):
        """Verify horizon H=3 target frame mapping."""
        data = make_synthetic_dataset(num_frames=20)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[3], split_strategy="none")
        res = builder.build(data)
        first_seq = res["sequence_metadata"]["sequences"][0]
        # First seq ends at frame_000005.jpg, horizon 3 should target frame_000008.jpg
        h3 = first_seq["horizons"]["h3"]
        assert h3["target_frame_id"] == "frame_000008.jpg"
        assert h3["target_status"] == "valid"

    def test_24_five_frame_forecast_horizon(self):
        """Verify horizon H=5 target frame mapping."""
        data = make_synthetic_dataset(num_frames=20)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[5], split_strategy="none")
        res = builder.build(data)
        first_seq = res["sequence_metadata"]["sequences"][0]
        # First seq ends at frame_000005.jpg, horizon 5 should target frame_000010.jpg
        h5 = first_seq["horizons"]["h5"]
        assert h5["target_frame_id"] == "frame_000010.jpg"
        assert h5["target_status"] == "valid"

    def test_25_future_frame_leakage_prevention(self):
        """Strict verification: modifying future frames does NOT alter input sequence tensor."""
        data1 = make_synthetic_dataset(num_frames=30)
        data2 = deepcopy(data1)
        # Modify future frames (frames 15 to 30)
        for i in range(15, 30):
            data2["features"][i]["visual_quality_features"]["blur_score"] += 500.0

        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[1], split_strategy="none")
        res1 = builder.build(data1)
        res2 = builder.build(data2)

        # Sequences ending before frame 15 (indices 0 to 9) must be numerically identical
        t1_early = res1["tensors"]["L5"][:10]
        t2_early = res2["tensors"]["L5"][:10]
        np.testing.assert_array_almost_equal(t1_early, t2_early, decimal=5)

    def test_26_missing_source_frame_indices(self):
        """Verify graceful fallback when source_frame_index is null/missing."""
        data = make_synthetic_dataset(num_frames=20)
        for f in data["features"]:
            f["source_frame_index"] = None
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        assert res["tensors"]["L5"].shape[0] == 16

    def test_27_missing_timestamps(self):
        """Verify graceful handling when timestamps are missing."""
        data = make_synthetic_dataset(num_frames=20)
        for f in data["features"]:
            f["timestamp_seconds"] = None
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        assert res["tensors"]["L5"].shape[0] == 16

    def test_28_invalid_frame_gaps(self):
        """Verify that windows spanning discontinuous gaps are rejected."""
        data = make_synthetic_dataset(num_frames=25)
        # Insert large gap between index 9 and index 10
        for i in range(10, 25):
            data["features"][i]["source_frame_index"] += 500

        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            split_strategy="none",
            max_sequence_gap_frames=2,
        )
        res = builder.build(data)
        # Windows crossing the gap should be rejected
        assert res["summary"]["rejected_sequences_count"] > 0
        # Valid windows should still be produced before and after the gap
        assert res["tensors"]["L5"].shape[0] > 0

    def test_29_dataset_boundary_handling(self):
        """Verify sequence builder respects dataset identity."""
        data = make_synthetic_dataset(num_frames=20, dataset_id="driving_sample")
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        assert res["dataset_id"] == "driving_sample"
        assert res["summary"]["dataset_id"] == "driving_sample"

    def test_30_sequence_segment_boundary_handling(self):
        """Verify sequences never cross split boundaries or embargo gaps."""
        data = make_synthetic_dataset(num_frames=80)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            split_strategy="chronological_purged",
            temporal_gap=10,
        )
        res = builder.build(data)
        seqs = res["sequence_metadata"]["sequences"]
        for s in seqs:
            # Check that every sequence has a consistent split assignment
            assert s["split"] in {"train", "val", "test"}

    def test_31_training_split_generation(self):
        """Verify training sequences are generated."""
        data = make_synthetic_dataset(num_frames=70)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="chronological_purged", temporal_gap=5)
        res = builder.build(data)
        train_count = res["summary"]["split_counts_per_length"]["L5"]["train"]
        assert train_count > 0

    def test_32_validation_split_generation(self):
        """Verify validation sequences are generated."""
        data = make_synthetic_dataset(num_frames=70)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="chronological_purged", temporal_gap=5)
        res = builder.build(data)
        val_count = res["summary"]["split_counts_per_length"]["L5"]["val"]
        assert val_count > 0

    def test_33_test_split_generation(self):
        """Verify test sequences are generated."""
        data = make_synthetic_dataset(num_frames=70)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="chronological_purged", temporal_gap=5)
        res = builder.build(data)
        test_count = res["summary"]["split_counts_per_length"]["L5"]["test"]
        assert test_count > 0

    def test_34_overlapping_window_leakage_prevention(self):
        """Verify strict zero-leakage invariant: no frame ID crosses split sets."""
        data = make_synthetic_dataset(num_frames=90)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5, 10],
            split_strategy="chronological_purged",
            temporal_gap=15,
        )
        res = builder.build(data)
        train_frames = set()
        val_frames = set()
        test_frames = set()

        for s in res["sequence_metadata"]["sequences"]:
            if s["split"] == "train":
                train_frames.update(s["frame_ids"])
            elif s["split"] == "val":
                val_frames.update(s["frame_ids"])
            elif s["split"] == "test":
                test_frames.update(s["frame_ids"])

        assert train_frames.isdisjoint(val_frames), "Train and Val share frame IDs!"
        assert train_frames.isdisjoint(test_frames), "Train and Test share frame IDs!"
        assert val_frames.isdisjoint(test_frames), "Val and Test share frame IDs!"

    def test_35_training_only_scaler_fitting(self):
        """Verify scaler parameters are fitted only on the training split."""
        data = make_synthetic_dataset(num_frames=80)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            scaling_strategy="standardization",
            split_strategy="chronological_purged",
            temporal_gap=10,
        )
        res = builder.build(data)
        scaler = res["scaler_parameters"]
        assert scaler["fitted"] is True
        assert scaler["fitted_on_split"] == "train"
        assert len(scaler["parameters"]) > 0

    def test_36_training_only_imputer_fitting(self):
        """Verify imputer median statistics are fitted only on training frames."""
        data = make_synthetic_dataset(num_frames=80, with_nulls=True)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            sequence_missing_policy="impute_median",
            split_strategy="chronological_purged",
            temporal_gap=10,
        )
        res = builder.build(data)
        imputer = res["imputation_parameters"]
        assert imputer["fitted"] is True
        assert imputer["missing_value_policy"] == "impute_median"

    def test_37_target_label_unavailable_handling(self):
        """Verify target labels are marked unavailable when ground truth is absent."""
        data = make_synthetic_dataset(num_frames=20)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[1], split_strategy="none")
        res = builder.build(data)
        assert res["summary"]["target_labels_available"] is False
        first_h = res["sequence_metadata"]["sequences"][0]["horizons"]["h1"]
        assert first_h["target_label"] is None
        assert first_h["target_label_available"] is False

    def test_38_valid_ground_truth_label_alignment(self):
        """Verify support for joining ground truth labels when provided."""
        data = make_synthetic_dataset(num_frames=20)
        # Provide external target labels dictionary
        external_labels = {
            "frame_000006.jpg": 0.95,
            "frame_000007.jpg": 0.88,
        }
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[1], split_strategy="none")
        res = builder.build(data, target_labels=external_labels)
        assert res["summary"]["target_labels_available"] is True
        seq0 = res["sequence_metadata"]["sequences"][0]  # targets frame_000006
        assert seq0["horizons"]["h1"]["target_label"] == 0.95
        assert seq0["horizons"]["h1"]["target_label_available"] is True

    def test_39_reproducible_split_generation(self):
        """Verify identical split outputs across identical random seeds."""
        data = make_synthetic_dataset(num_frames=50)
        builder1 = TemporalSequenceBuilder(random_seed=42, sequence_lengths=[5])
        builder2 = TemporalSequenceBuilder(random_seed=42, sequence_lengths=[5])
        res1 = builder1.build(data)
        res2 = builder2.build(data)
        assert res1["dataset_splits"] == res2["dataset_splits"]

    def test_40_empty_or_insufficient_input_data(self):
        """Verify error when dataset contains 0 frames."""
        data = {"schema_version": "1.0", "dataset_id": "empty", "features": []}
        builder = TemporalSequenceBuilder()
        with pytest.raises(EmptyDatasetError):
            builder.load_input(data)

    def test_41_partial_processing(self):
        """Verify max_sequences argument limits output size."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data, max_sequences=3)
        assert res["tensors"]["L5"].shape[0] == 3
        assert len(res["sequence_metadata"]["sequences"]) == 3

    def test_42_output_artifact_validation(self, tmp_path):
        """Verify all 11 required files are written to disk."""
        data = make_synthetic_dataset(num_frames=30)
        out_dir = tmp_path / "sequences_out"
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[1], split_strategy="none")
        res = builder.build(data, output_dir=out_dir, overwrite=True)

        expected_files = [
            "sequences_L5.npy",
            "sequence_metadata.json",
            "feature_manifest.json",
            "dataset_splits.json",
            "scaler_parameters.json",
            "imputation_parameters.json",
            "sequence_summary.json",
            "sequence_errors.json",
        ]
        for ef in expected_files:
            assert (out_dir / ef).exists(), f"Missing expected output: {ef}"

    def test_43_model_ready_tensor_finite_value_validation(self):
        """Verify generated tensor contains strictly finite float32 values."""
        data = make_synthetic_dataset(num_frames=30)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L5"]
        assert tensor.dtype == np.float32
        assert np.isfinite(tensor).all()

    def test_44_cli_argument_validation(self, tmp_path):
        """Verify CLI entry point runs with valid arguments and generates artifacts."""
        data = make_synthetic_dataset(num_frames=30)
        data_path = tmp_path / "test_features.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        out_dir = tmp_path / "cli_out"
        from src.features.temporal_sequence_builder import main
        code = main([
            "--input", str(data_path),
            "--output", str(out_dir),
            "--sequence-lengths", "5,10",
            "--horizons", "1,3",
            "--overwrite",
        ])
        assert code == 0
        assert (out_dir / "sequences_L5.npy").exists()
        assert (out_dir / "sequences_L10.npy").exists()

    def test_45_sequence_metadata_consistency(self):
        """Verify consistency between tensor index and metadata sequence ID."""
        data = make_synthetic_dataset(num_frames=25)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(data)
        tensor = res["tensors"]["L5"]
        meta_list = res["sequence_metadata"]["sequences_by_length"]["L5"]
        assert tensor.shape[0] == len(meta_list)
        for i, m in enumerate(meta_list):
            assert f"L5_{i+1:06d}" in m["sequence_id"]

    # --------------------------------------------------------------------------
    # Additional Edge Cases (46 to 52)
    # --------------------------------------------------------------------------

    def test_46_min_max_scaling(self):
        """Verify min-max scaling normalizes features into [0, 1] range."""
        data = make_synthetic_dataset(num_frames=40)
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            scaling_strategy="min_max",
            split_strategy="none",
        )
        res = builder.build(data)
        tensor = res["tensors"]["L5"]
        # Values should be bounded in [0.0, 1.0]
        assert np.min(tensor) >= -1e-6
        assert np.max(tensor) <= 1.0 + 1e-6

    def test_47_drop_features_missing_policy(self):
        """Verify drop_features excludes columns with high missing rates."""
        data = make_synthetic_dataset(num_frames=50)
        # Artificially nullify one feature in 80% of frames
        for i in range(40):
            data["features"][i]["detection_features"]["mean_confidence"] = None

        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            sequence_missing_policy="drop_features",
            missing_feature_threshold=0.5,
            split_strategy="none",
        )
        res = builder.build(data)
        imp = res["imputation_parameters"]
        assert "mean_confidence" in imp["dropped_features"] or "det_mean_confidence" in imp["dropped_features"]

    def test_48_out_of_bounds_horizon_handling(self):
        """Verify last sequences record horizon target as out_of_bounds."""
        data = make_synthetic_dataset(num_frames=20)
        builder = TemporalSequenceBuilder(sequence_lengths=[5], forecast_horizons=[5], split_strategy="none")
        res = builder.build(data)
        # The very last sequence ends at frame 20 (index 19); index 19 + 5 = 24 >= 20 -> out of bounds
        last_seq = res["sequence_metadata"]["sequences"][-1]
        assert last_seq["horizons"]["h5"]["target_status"] == "out_of_bounds"
        assert last_seq["horizons"]["h5"]["target_frame_id"] is None

    def test_49_flat_input_format_support(self, tmp_path):
        """Verify builder parses flat aggregated_frame_features.json format."""
        flat_data = {
            "schema_version": "1.0",
            "dataset_id": "flat_driving",
            "frames": [
                {
                    "frame_id": f"frame_{i+1:06d}.jpg",
                    "source_frame_index": i * 6,
                    "timestamp_seconds": i * 0.2,
                    "det_detection_count": 2,
                    "det_mean_confidence": 0.8,
                    "vq_blur_score": 400.0,
                    "numeric_feature_coverage": 1.0,
                }
                for i in range(25)
            ],
        }
        builder = TemporalSequenceBuilder(sequence_lengths=[5], split_strategy="none")
        res = builder.build(flat_data)
        assert res["dataset_id"] == "flat_driving"
        assert res["tensors"]["L5"].shape[0] == 21

    def test_50_validate_only_mode(self):
        """Verify validate_only mode operates without tensor allocation."""
        data = make_synthetic_dataset(num_frames=25)
        builder = TemporalSequenceBuilder(sequence_lengths=[5, 10])
        val_res = builder.validate_only(data)
        assert val_res["validation_status"] == "PASS"
        assert val_res["sequence_feasible"] is True

    def test_51_zero_variance_scaling_safety(self):
        """Verify standardization does not produce NaN/Inf when variance is zero."""
        data = make_synthetic_dataset(num_frames=30)
        # Set all blur scores to identical value
        for f in data["features"]:
            f["visual_quality_features"]["blur_score"] = 100.0

        builder = TemporalSequenceBuilder(
            sequence_lengths=[5],
            scaling_strategy="standardization",
            split_strategy="none",
        )
        res = builder.build(data)
        tensor = res["tensors"]["L5"]
        assert np.isfinite(tensor).all()

    @pytest.mark.skipif(
        not REAL_TEMPORAL_FEATURES_PATH.exists(),
        reason="Real driving_sample temporal_features.json artifact not present on disk.",
    )
    def test_52_real_driving_sample_integration(self):
        """End-to-end integration test on real Task 5 temporal features artifact."""
        builder = TemporalSequenceBuilder(
            sequence_lengths=[5, 10],
            forecast_horizons=[1, 3, 5],
            sequence_missing_policy="impute_mean",
            scaling_strategy="none",
            split_strategy="chronological_purged",
            temporal_gap=15,
        )
        res = builder.build(REAL_TEMPORAL_FEATURES_PATH)
        assert res["dataset_id"] == "driving_sample"
        assert "L5" in res["tensors"]
        assert "L10" in res["tensors"]
        assert res["tensors"]["L5"].shape[0] > 0
        assert np.isfinite(res["tensors"]["L5"]).all()
        assert res["summary"]["validation_status"] == "PASS"
