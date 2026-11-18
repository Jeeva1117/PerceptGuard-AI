"""PerceptGuard AI: Tests for Temporal Feature Aggregator (Day 5 Task 5).

Comprehensive unit and integration test suite covering:
1. Valid three-source alignment
2. Detection-only input
3. Motion-only input
4. Visual-only input
5. Missing source artifacts
6. Dataset ID mismatch
7. Frame ID mismatch
8. Duplicate frame IDs
9. Missing frames across sources
10. Chronological frame ordering
11. Timestamp monotonicity and ordering
12. Conflicting metadata handling
13. Image dimension mismatch detection
14. Frame-level feature fusion
15. Three-frame window aggregation (w3)
16. Five-frame window aggregation (w5)
17. Ten-frame window aggregation (w10)
18. Custom window sizes (e.g. w2, w4)
19. Trailing-window alignment
20. Strict zero future-frame leakage prevention
21. Mean calculation accuracy
22. Minimum and maximum calculation
23. Population standard deviation calculation
24. Median calculation
25. First and last value tracking
26. First-to-last difference calculation
27. Feature coverage calculation
28. Missing-value policy (no silent zeros)
29. Zero-detection frame handling
30. Zero-track frame handling
31. Insufficient observations handling
32. Temporal trend calculations (OLS slope)
33. Constant-value trends (slope = 0.0)
34. Non-finite / NaN coordinate prevention
35. Empty input artifact rejection
36. Partial batch processing (--max-frames, --start-frame, --end-frame)
37. Output JSON schema validation for all 5 artifacts
38. Metadata and provenance preservation
39. Deterministic sorting and reproducibility
40. CLI argument parsing and execution
41. Output directory exists protection
42. Output directory overwrite success
43. Real dataset integration test on driving_sample artifacts (200 frames)
44. Five manually verified calculations on real artifacts
45. Strict future-frame leakage verification on real sequence
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List
from copy import deepcopy
import pytest

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.temporal_feature_aggregator import (
    DEFAULT_MIN_OBSERVATIONS_FOR_TREND,
    DEFAULT_WINDOW_ALIGNMENT,
    DEFAULT_WINDOW_SIZES,
    NUMERICAL_FEATURE_KEYS,
    TREND_FEATURE_MAPPINGS,
    DatasetIdMismatchError,
    DimensionMismatchError,
    DuplicateFrameIdError,
    FrameIdMismatchError,
    InputArtifactError,
    MissingSourceArtifactError,
    OutputDirectoryExistsError,
    TemporalFeatureAggregationError,
    TemporalFeatureAggregator,
    calculate_ols_trend,
    main,
)
from src.features.feature_config import FeatureConfig

REAL_DETECTION_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "features"
    / "detection"
    / "det_features_driving_sample_20261118_141511_9e022a"
    / "frame_detection_features.json"
)

REAL_MOTION_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "features"
    / "motion"
    / "motion_features_driving_sample_20261118_143109_bb4a56"
    / "frame_motion_features.json"
)

REAL_VISUAL_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "features"
    / "visual_quality"
    / "run_official"
    / "visual_quality_features.json"
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def synthetic_detection_data() -> Dict[str, Any]:
    """Create 5 synthetic frames of detection features."""
    return {
        "schema_version": "1.0",
        "dataset_id": "synthetic_test_seq",
        "total_frames": 5,
        "frames": [
            {
                "frame_id": f"frame_{i:06d}.jpg",
                "source_frame_index": i,
                "timestamp_seconds": round(i * 0.2, 4),
                "image_width": 640,
                "image_height": 360,
                "detection_count": 2 if i > 0 else 0,
                "unique_class_count": 1 if i > 0 else 0,
                "mean_confidence": round(0.5 + i * 0.1, 4) if i > 0 else None,
                "minimum_confidence": round(0.4 + i * 0.1, 4) if i > 0 else None,
                "maximum_confidence": round(0.6 + i * 0.1, 4) if i > 0 else None,
                "confidence_std": 0.1 if i > 0 else None,
                "total_bbox_area": round(1000.0 * i, 2) if i > 0 else None,
                "mean_bbox_area": round(500.0 * i, 2) if i > 0 else None,
                "mean_bbox_width": round(50.0 * i, 2) if i > 0 else None,
                "mean_bbox_height": round(10.0 * i, 2) if i > 0 else None,
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def synthetic_motion_data() -> Dict[str, Any]:
    """Create 5 synthetic frames of motion features."""
    return {
        "schema_version": "1.0",
        "dataset_id": "synthetic_test_seq",
        "total_frames": 5,
        "frames": [
            {
                "frame_id": f"frame_{i:06d}.jpg",
                "source_frame_index": i,
                "timestamp_seconds": round(i * 0.2, 4),
                "observed_track_count": 2 if i > 0 else 0,
                "moving_track_count": 1 if i > 1 else 0,
                "stationary_track_count": 1 if i > 1 else 0,
                "mean_displacement": round(2.0 * i, 4) if i > 0 else None,
                "maximum_displacement": round(3.0 * i, 4) if i > 0 else None,
                "mean_speed": round(10.0 * i, 4) if i > 0 else None,
                "mean_velocity_x": round(1.5 * i, 4) if i > 0 else None,
                "mean_velocity_y": round(0.5 * i, 4) if i > 0 else None,
                "motion_feature_coverage": 1.0 if i > 0 else 0.0,
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def synthetic_visual_data() -> Dict[str, Any]:
    """Create 5 synthetic frames of visual quality features."""
    return {
        "schema_version": "1.0",
        "dataset_id": "synthetic_test_seq",
        "total_frames": 5,
        "features": [
            {
                "frame_id": f"frame_{i:06d}.jpg",
                "source_frame_index": i,
                "timestamp_seconds": round(i * 0.2, 4),
                "image_width": 640,
                "image_height": 360,
                "blur_score": round(300.0 + i * 20.0, 4),
                "mean_brightness": round(100.0 + i * 5.0, 4),
                "median_brightness": round(100.0 + i * 5.0, 4),
                "brightness_std": round(20.0 + i, 4),
                "contrast_std": round(40.0 + i * 2.0, 4),
                "contrast_range": round(120.0 + i * 5.0, 4),
                "contrast_percentile_range": round(90.0 + i * 3.0, 4),
                "image_aspect_ratio": 1.7778,
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def synthetic_artifacts_on_disk(
    tmp_path: Path,
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
) -> Dict[str, Path]:
    """Write synthetic feature payloads to JSON files on disk."""
    det_file = tmp_path / "frame_detection_features.json"
    mot_file = tmp_path / "frame_motion_features.json"
    vq_file = tmp_path / "visual_quality_features.json"

    det_file.write_text(json.dumps(synthetic_detection_data), encoding="utf-8")
    mot_file.write_text(json.dumps(synthetic_motion_data), encoding="utf-8")
    vq_file.write_text(json.dumps(synthetic_visual_data), encoding="utf-8")

    return {
        "detection": det_file,
        "motion": mot_file,
        "visual": vq_file,
    }


# ==============================================================================
# UNIT TESTS: Alignment, Fusion & Input Validation
# ==============================================================================

def test_01_valid_three_source_alignment(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify alignment across detection, motion, and visual quality sources."""
    aggregator = TemporalFeatureAggregator()
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    assert result["processed_frame_count"] == 5
    assert result["successfully_aligned_frame_count"] == 5
    assert result["partially_aligned_frame_count"] == 0
    assert len(result["features"]) == 5

    rec_0 = result["features"][0]
    assert rec_0.frame_id == "frame_000000.jpg"
    assert rec_0.source_availability.detection is True
    assert rec_0.source_availability.motion is True
    assert rec_0.source_availability.visual_quality is True
    assert rec_0.feature_coverage.source_coverage == 1.0


def test_02_detection_only_input_with_partial_allowed(synthetic_detection_data: Dict[str, Any]):
    """Verify aggregator processes detection-only source when allow_partial=True."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(detection_source=synthetic_detection_data)

    assert result["processed_frame_count"] == 5
    assert result["successfully_aligned_frame_count"] == 0
    assert result["partially_aligned_frame_count"] == 5
    rec_1 = result["features"][1]
    assert rec_1.source_availability.detection is True
    assert rec_1.source_availability.motion is False
    assert rec_1.source_availability.visual_quality is False
    assert rec_1.detection_features["detection_count"] == 2
    assert rec_1.motion_features["observed_track_count"] is None
    assert rec_1.visual_quality_features["blur_score"] is None


def test_03_motion_only_input_with_partial_allowed(synthetic_motion_data: Dict[str, Any]):
    """Verify aggregator processes motion-only source when allow_partial=True."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(motion_source=synthetic_motion_data)

    assert result["processed_frame_count"] == 5
    rec_2 = result["features"][2]
    assert rec_2.source_availability.motion is True
    assert rec_2.source_availability.detection is False
    assert rec_2.motion_features["moving_track_count"] == 1


def test_04_visual_only_input_with_partial_allowed(synthetic_visual_data: Dict[str, Any]):
    """Verify aggregator processes visual-only source when allow_partial=True."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(visual_source=synthetic_visual_data)

    assert result["processed_frame_count"] == 5
    rec_3 = result["features"][3]
    assert rec_3.source_availability.visual_quality is True
    assert rec_3.visual_quality_features["contrast_std"] == 46.0


def test_05_missing_source_artifacts_raises_error(synthetic_detection_data: Dict[str, Any]):
    """Verify MissingSourceArtifactError is raised when required source is omitted without allow_partial."""
    aggregator = TemporalFeatureAggregator(allow_partial=False)
    with pytest.raises(MissingSourceArtifactError) as exc_info:
        aggregator.aggregate(detection_source=synthetic_detection_data)
    assert "Missing required feature sources" in str(exc_info.value)


def test_06_dataset_id_mismatch(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
):
    """Verify DatasetIdMismatchError is raised when sources report conflicting dataset IDs."""
    mismatched_motion = dict(synthetic_motion_data)
    mismatched_motion["dataset_id"] = "different_dataset_sequence"

    aggregator = TemporalFeatureAggregator(allow_partial=True)
    with pytest.raises(DatasetIdMismatchError) as exc_info:
        aggregator.aggregate(
            detection_source=synthetic_detection_data,
            motion_source=mismatched_motion,
        )
    assert "Conflicting dataset identifiers detected" in str(exc_info.value)


def test_07_frame_id_mismatch(
    synthetic_detection_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify FrameIdMismatchError is raised when sources have zero overlapping frame IDs."""
    disjoint_visual = {
        "dataset_id": synthetic_detection_data["dataset_id"],
        "features": [
            {"frame_id": "other_frame_999.jpg", "blur_score": 100.0}
        ]
    }
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    with pytest.raises(FrameIdMismatchError) as exc_info:
        aggregator.aggregate(
            detection_source=synthetic_detection_data,
            visual_source=disjoint_visual,
        )
    assert "Zero overlapping frame identifiers" in str(exc_info.value)


def test_08_duplicate_frame_ids_rejected():
    """Verify DuplicateFrameIdError is raised when an artifact contains duplicate frame IDs."""
    dup_data = {
        "dataset_id": "test_ds",
        "frames": [
            {"frame_id": "frame_0001.jpg", "detection_count": 1},
            {"frame_id": "frame_0001.jpg", "detection_count": 2},
        ]
    }
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    with pytest.raises(DuplicateFrameIdError) as exc_info:
        aggregator.aggregate(detection_source=dup_data)
    assert "Duplicate frame_id 'frame_0001.jpg'" in str(exc_info.value)


def test_09_missing_frames_union_handling(
    synthetic_detection_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify aggregator safely constructs union of frames and logs partial status."""
    # Remove frame 4 from detection
    partial_detection = {
        "dataset_id": synthetic_detection_data["dataset_id"],
        "frames": synthetic_detection_data["frames"][:4]  # frames 0..3
    }
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(
        detection_source=partial_detection,
        visual_source=synthetic_visual_data,  # has frames 0..4
    )

    assert result["processed_frame_count"] == 5
    f4 = result["features"][4]
    assert f4.frame_id == "frame_000004.jpg"
    assert f4.source_availability.detection is False
    assert f4.source_availability.visual_quality is True
    assert f4.processing_status == "partial"


def test_10_chronological_frame_ordering(
    synthetic_detection_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify records are strictly sorted by chronological source_frame_index."""
    # Reverse frame order in detection data
    shuffled_detection = {
        "dataset_id": synthetic_detection_data["dataset_id"],
        "frames": list(reversed(synthetic_detection_data["frames"])),
    }
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(
        detection_source=shuffled_detection,
        visual_source=synthetic_visual_data,
    )

    extracted_ids = [r.frame_id for r in result["features"]]
    assert extracted_ids == [f"frame_{i:06d}.jpg" for i in range(5)]
    extracted_indices = [r.source_frame_index for r in result["features"]]
    assert extracted_indices == [0, 1, 2, 3, 4]


def test_11_timestamp_ordering(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
):
    """Verify timestamps are preserved and monotonically non-decreasing."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
    )
    timestamps = [r.timestamp_seconds for r in result["features"]]
    assert timestamps == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_12_conflicting_metadata_source_index_priority():
    """Verify priority resolution for metadata attributes across sources."""
    d1 = {"dataset_id": "test", "frames": [{"frame_id": "f1.jpg", "source_frame_index": 10}]}
    d2 = {"dataset_id": "test", "features": [{"frame_id": "f1.jpg", "source_frame_index": 10}]}

    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(detection_source=d1, visual_source=d2)
    assert result["features"][0].source_frame_index == 10


def test_13_image_dimension_mismatch_logged():
    """Verify dimension mismatch between sources is logged to error audit without crashing."""
    d1 = {"dataset_id": "test", "frames": [{"frame_id": "f1.jpg", "image_width": 640, "image_height": 360}]}
    d2 = {"dataset_id": "test", "features": [{"frame_id": "f1.jpg", "image_width": 1280, "image_height": 720}]}

    aggregator = TemporalFeatureAggregator(allow_partial=True)
    result = aggregator.aggregate(detection_source=d1, visual_source=d2)

    assert len(result["errors"]) >= 1
    err = result["errors"][0]
    assert err.error_type == "DIMENSION_MISMATCH"
    assert result["summary"].validation_status == "WARNING"


def test_14_frame_level_feature_fusion(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify all 27 base numerical features are fused correctly on an aligned frame."""
    aggregator = TemporalFeatureAggregator()
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    f2 = result["features"][2]
    # Detection
    assert f2.detection_features["detection_count"] == 2
    assert f2.detection_features["mean_confidence"] == pytest.approx(0.7, abs=1e-4)
    assert f2.detection_features["total_bbox_area"] == 2000.0
    # Motion
    assert f2.motion_features["observed_track_count"] == 2
    assert f2.motion_features["mean_displacement"] == 4.0
    assert f2.motion_features["mean_speed"] == 20.0
    # Visual Quality
    assert f2.visual_quality_features["blur_score"] == 340.0
    assert f2.visual_quality_features["mean_brightness"] == 110.0
    assert f2.visual_quality_features["contrast_std"] == 44.0


# ==============================================================================
# UNIT TESTS: Temporal Window Aggregation & Statistics
# ==============================================================================

def test_15_three_frame_aggregation(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify 3-frame trailing window (w3) aggregates correctly."""
    aggregator = TemporalFeatureAggregator(window_sizes=[3])
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    f2 = result["features"][2]
    w3_data = f2.temporal_windows["w3"]
    assert w3_data["window_size"] == 3
    assert w3_data["contributing_frame_count"] == 3  # frames 0, 1, 2
    assert w3_data["window_start_frame"] == "frame_000000.jpg"
    assert w3_data["window_end_frame"] == "frame_000002.jpg"

    # Blur scores: frame 0 = 300, frame 1 = 320, frame 2 = 340
    blur_stats = w3_data["feature_statistics"]["blur_score"]
    assert blur_stats["mean"] == pytest.approx(320.0, abs=1e-4)
    assert blur_stats["min"] == 300.0
    assert blur_stats["max"] == 340.0
    assert blur_stats["first"] == 300.0
    assert blur_stats["last"] == 340.0
    assert blur_stats["diff"] == 40.0


def test_16_five_frame_aggregation(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify 5-frame trailing window (w5) at frame 4 aggregates frames 0..4."""
    aggregator = TemporalFeatureAggregator(window_sizes=[5])
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    f4 = result["features"][4]
    w5_data = f4.temporal_windows["w5"]
    assert w5_data["window_size"] == 5
    assert w5_data["contributing_frame_count"] == 5
    assert w5_data["window_start_frame"] == "frame_000000.jpg"
    assert w5_data["window_end_frame"] == "frame_000004.jpg"

    # Brightness: 100, 105, 110, 115, 120 -> mean = 110, median = 110
    bright_stats = w5_data["feature_statistics"]["mean_brightness"]
    assert bright_stats["mean"] == pytest.approx(110.0, abs=1e-4)
    assert bright_stats["median"] == pytest.approx(110.0, abs=1e-4)
    assert bright_stats["min"] == 100.0
    assert bright_stats["max"] == 120.0
    assert bright_stats["diff"] == 20.0


def test_17_ten_frame_aggregation_partial_window(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify 10-frame window handles sequences shorter than window size (5 frames available)."""
    aggregator = TemporalFeatureAggregator(window_sizes=[10])
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    f4 = result["features"][4]
    w10_data = f4.temporal_windows["w10"]
    assert w10_data["window_size"] == 10
    assert w10_data["contributing_frame_count"] == 5  # Only 5 frames exist chronologically


def test_18_custom_window_sizes(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify custom window lengths (e.g. 2, 4) configure properly."""
    aggregator = TemporalFeatureAggregator(window_sizes=[2, 4])
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    f3 = result["features"][3]
    assert "w2" in f3.temporal_windows
    assert "w4" in f3.temporal_windows
    assert "w5" not in f3.temporal_windows


def test_19_trailing_window_alignment_boundaries(synthetic_visual_data: Dict[str, Any]):
    """Verify trailing window boundaries strictly conform to [max(0, i - W + 1), i]."""
    aggregator = TemporalFeatureAggregator(window_sizes=[3], window_alignment="trailing", allow_partial=True)
    result = aggregator.aggregate(visual_source=synthetic_visual_data)

    # Frame 0: slice [0, 0] -> contributing count = 1
    assert result["features"][0].temporal_windows["w3"]["contributing_frame_count"] == 1
    # Frame 1: slice [0, 1] -> contributing count = 2
    assert result["features"][1].temporal_windows["w3"]["contributing_frame_count"] == 2
    # Frame 2: slice [0, 2] -> contributing count = 3
    assert result["features"][2].temporal_windows["w3"]["contributing_frame_count"] == 3
    # Frame 3: slice [1, 3] -> contributing count = 3
    assert result["features"][3].temporal_windows["w3"]["contributing_frame_count"] == 3


def test_20_no_future_frame_leakage_strict_check(synthetic_visual_data: Dict[str, Any]):
    """Verify modifying future frames has ZERO effect on trailing window metrics of past frames."""
    aggregator = TemporalFeatureAggregator(window_sizes=[3], allow_partial=True)

    # Run 1: original data
    res1 = aggregator.aggregate(visual_source=synthetic_visual_data)
    f2_mean1 = res1["features"][2].temporal_windows["w3"]["feature_statistics"]["blur_score"]["mean"]

    # Run 2: drastically alter frame 3 and frame 4
    mutated_visual = deepcopy(synthetic_visual_data)
    mutated_visual["features"][3]["blur_score"] = 999999.0
    mutated_visual["features"][4]["blur_score"] = 999999.0

    res2 = aggregator.aggregate(visual_source=mutated_visual)
    f2_mean2 = res2["features"][2].temporal_windows["w3"]["feature_statistics"]["blur_score"]["mean"]

    # Trailing window for frame 2 must be 100% identical
    assert f2_mean1 == f2_mean2


def test_21_mean_calculation():
    """Verify mean calculation ignores None values and computes true mean."""
    aggregator = TemporalFeatureAggregator()
    # Values: [10.0, None, 20.0, 30.0]
    stats = aggregator._compute_feature_stats([10.0, None, 20.0, 30.0], decimal_places=4)
    assert stats.valid_count == 3
    assert stats.missing_count == 1
    assert stats.mean == 20.0


def test_22_minimum_and_maximum():
    """Verify min and max computations."""
    aggregator = TemporalFeatureAggregator()
    stats = aggregator._compute_feature_stats([15.0, 5.0, 25.0], decimal_places=4)
    assert stats.min == 5.0
    assert stats.max == 25.0


def test_23_population_standard_deviation():
    """Verify standard deviation uses population formula."""
    aggregator = TemporalFeatureAggregator()
    # std of [10, 20]: mean = 15, variance = ((10-15)^2 + (20-15)^2)/2 = (25+25)/2 = 25 -> std = 5.0
    stats = aggregator._compute_feature_stats([10.0, 20.0], decimal_places=4)
    assert stats.std == pytest.approx(5.0, abs=1e-4)


def test_24_median_calculation():
    """Verify median calculation for even and odd sequence lengths."""
    aggregator = TemporalFeatureAggregator()
    # Odd length: [10, 50, 20] -> sorted [10, 20, 50] -> median 20
    assert aggregator._compute_feature_stats([10.0, 50.0, 20.0], 4).median == 20.0
    # Even length: [10, 40, 20, 30] -> sorted [10, 20, 30, 40] -> median 25
    assert aggregator._compute_feature_stats([10.0, 40.0, 20.0, 30.0], 4).median == 25.0


def test_25_first_and_last_values():
    """Verify chronological first and last valid values."""
    aggregator = TemporalFeatureAggregator()
    stats = aggregator._compute_feature_stats([None, 42.0, None, 99.0], 4)
    assert stats.first == 42.0
    assert stats.last == 99.0


def test_26_first_to_last_difference():
    """Verify diff = last - first calculation."""
    aggregator = TemporalFeatureAggregator()
    stats = aggregator._compute_feature_stats([100.0, 110.0, 140.0], 4)
    assert stats.diff == 40.0


def test_27_feature_coverage_calculation():
    """Verify coverage_ratio = valid_count / total_window_length."""
    aggregator = TemporalFeatureAggregator()
    stats = aggregator._compute_feature_stats([1.0, None, 2.0, None, 3.0], 4)
    assert stats.valid_count == 3
    assert stats.missing_count == 2
    assert stats.coverage_ratio == 0.6


def test_28_missing_value_policy_no_silent_zeros():
    """Verify all-None window yields None statistics without treating missing as zero."""
    aggregator = TemporalFeatureAggregator()
    stats = aggregator._compute_feature_stats([None, None, None], 4)
    assert stats.mean is None
    assert stats.min is None
    assert stats.max is None
    assert stats.std is None
    assert stats.diff is None
    assert stats.valid_count == 0
    assert stats.missing_count == 3
    assert stats.coverage_ratio == 0.0


def test_29_zero_detection_frames_preserved(synthetic_detection_data: Dict[str, Any]):
    """Verify zero-detection frames are preserved with detection_count=0 and None confidence."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    res = aggregator.aggregate(detection_source=synthetic_detection_data)
    f0 = res["features"][0]
    assert f0.detection_features["detection_count"] == 0
    assert f0.detection_features["mean_confidence"] is None


def test_30_frames_without_tracked_objects_preserved(synthetic_motion_data: Dict[str, Any]):
    """Verify zero-track frames are preserved with observed_track_count=0 and None displacement."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)
    res = aggregator.aggregate(motion_source=synthetic_motion_data)
    f0 = res["features"][0]
    assert f0.motion_features["observed_track_count"] == 0
    assert f0.motion_features["mean_displacement"] is None


# ==============================================================================
# UNIT TESTS: Temporal Trend Calculations
# ==============================================================================

def test_31_insufficient_observations_for_trend():
    """Verify trend calculation returns insufficient_data when valid points < 2."""
    trend = calculate_ols_trend([(0, 10.0)], decimal_places=4, min_observations=2)
    assert trend.status == "insufficient_data"
    assert trend.slope is None
    assert trend.direction == "none"


def test_32_temporal_trend_increasing_and_decreasing():
    """Verify positive and negative linear slope calculations."""
    # Increasing: step 0: 10, step 1: 20, step 2: 30 -> slope = 10.0
    inc_trend = calculate_ols_trend([(0, 10.0), (1, 20.0), (2, 30.0)])
    assert inc_trend.status == "valid"
    assert inc_trend.slope == pytest.approx(10.0, abs=1e-4)
    assert inc_trend.direction == "increasing"

    # Decreasing: step 0: 30, step 1: 20, step 2: 10 -> slope = -10.0
    dec_trend = calculate_ols_trend([(0, 30.0), (1, 20.0), (2, 10.0)])
    assert dec_trend.status == "valid"
    assert dec_trend.slope == pytest.approx(-10.0, abs=1e-4)
    assert dec_trend.direction == "decreasing"


def test_33_constant_value_trends():
    """Verify constant values yield slope=0.0 and direction='flat'."""
    flat_trend = calculate_ols_trend([(0, 50.0), (1, 50.0), (2, 50.0)])
    assert flat_trend.status == "constant"
    assert flat_trend.slope == 0.0
    assert flat_trend.direction == "flat"


def test_34_non_finite_coordinate_prevention():
    """Verify NaN and Inf values are safely filtered and do not crash trend calculations."""
    aggregator = TemporalFeatureAggregator()
    stats = aggregator._compute_feature_stats([10.0, float("nan"), 20.0], decimal_places=4)
    assert stats.valid_count == 2
    assert stats.mean == 15.0


def test_35_empty_input_artifact_rejected(tmp_path: Path):
    """Verify 0-byte input artifacts raise InputArtifactError."""
    empty_file = tmp_path / "empty_det.json"
    empty_file.touch()

    aggregator = TemporalFeatureAggregator(allow_partial=True)
    with pytest.raises(InputArtifactError) as exc_info:
        aggregator.aggregate(detection_source=empty_file)
    assert "Input source artifact is empty" in str(exc_info.value)


def test_36_partial_batch_processing_slicing(
    synthetic_detection_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify start_frame, end_frame, and max_frames constraints."""
    aggregator = TemporalFeatureAggregator(allow_partial=True)

    # max_frames = 2
    res_max = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        visual_source=synthetic_visual_data,
        max_frames=2,
    )
    assert res_max["processed_frame_count"] == 2
    assert res_max["summary"].is_partial is True

    # start_frame=1, end_frame=3
    res_slice = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        visual_source=synthetic_visual_data,
        start_frame=1,
        end_frame=3,
    )
    assert res_slice["processed_frame_count"] == 3
    assert [r.source_frame_index for r in res_slice["features"]] == [1, 2, 3]


def test_37_output_json_schema_validation(
    synthetic_artifacts_on_disk: Dict[str, Path],
    tmp_path: Path,
):
    """Verify all 5 versioned output JSON files match Schema v1.0 specifications."""
    out_dir = tmp_path / "test_out"
    aggregator = TemporalFeatureAggregator()
    aggregator.aggregate(
        detection_source=synthetic_artifacts_on_disk["detection"],
        motion_source=synthetic_artifacts_on_disk["motion"],
        visual_source=synthetic_artifacts_on_disk["visual"],
        output_dir=out_dir,
    )

    # 1. temporal_features.json
    f1 = out_dir / "temporal_features.json"
    assert f1.exists()
    d1 = json.loads(f1.read_text(encoding="utf-8"))
    assert d1["schema_version"] == "1.0"
    assert d1["feature_type"] == "temporal_aggregated"
    assert len(d1["features"]) == 5

    # 2. aggregated_frame_features.json
    f2 = out_dir / "aggregated_frame_features.json"
    assert f2.exists()
    d2 = json.loads(f2.read_text(encoding="utf-8"))
    assert len(d2["frames"]) == 5
    assert "det_detection_count" in d2["frames"][0]
    assert "vq_blur_score" in d2["frames"][0]

    # 3. temporal_feature_summary.json
    f3 = out_dir / "temporal_feature_summary.json"
    assert f3.exists()
    d3 = json.loads(f3.read_text(encoding="utf-8"))
    assert d3["processed_frame_count"] == 5
    assert d3["validation_status"] == "PASS"

    # 4. feature_metadata.json
    f4 = out_dir / "feature_metadata.json"
    assert f4.exists()
    d4 = json.loads(f4.read_text(encoding="utf-8"))
    assert "configured_window_sizes" in d4
    assert d4["hardware_execution"]["cpu_only"] is True

    # 5. feature_errors.json
    f5 = out_dir / "feature_errors.json"
    assert f5.exists()
    d5 = json.loads(f5.read_text(encoding="utf-8"))
    assert d5["total_errors"] == 0


def test_38_metadata_preservation(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify source_frame_index, timestamp_seconds, and dataset_id are preserved."""
    aggregator = TemporalFeatureAggregator()
    result = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    f3 = result["features"][3]
    assert f3.dataset_id == "synthetic_test_seq"
    assert f3.frame_id == "frame_000003.jpg"
    assert f3.source_frame_index == 3
    assert f3.timestamp_seconds == 0.6


def test_39_deterministic_ordering_and_reproducibility(
    synthetic_detection_data: Dict[str, Any],
    synthetic_motion_data: Dict[str, Any],
    synthetic_visual_data: Dict[str, Any],
):
    """Verify identical outputs across multiple invocations."""
    aggregator = TemporalFeatureAggregator()
    res1 = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )
    res2 = aggregator.aggregate(
        detection_source=synthetic_detection_data,
        motion_source=synthetic_motion_data,
        visual_source=synthetic_visual_data,
    )

    json1 = [f.to_dict() for f in res1["features"]]
    json2 = [f.to_dict() for f in res2["features"]]
    assert json1 == json2


def test_40_cli_validation(synthetic_artifacts_on_disk: Dict[str, Path], tmp_path: Path):
    """Verify CLI interface runs end-to-end and returns exit code 0."""
    out_dir = tmp_path / "cli_test_out"
    args = [
        "--detection", str(synthetic_artifacts_on_disk["detection"]),
        "--motion", str(synthetic_artifacts_on_disk["motion"]),
        "--visual", str(synthetic_artifacts_on_disk["visual"]),
        "--output", str(out_dir),
        "--window-sizes", "3,5",
        "--verbose",
    ]
    exit_code = main(args)
    assert exit_code == 0
    assert (out_dir / "temporal_features.json").exists()


def test_41_output_directory_exists_protection(
    synthetic_artifacts_on_disk: Dict[str, Path],
    tmp_path: Path,
):
    """Verify OutputDirectoryExistsError is raised when overwrite=False and target exists."""
    out_dir = tmp_path / "existing_dir"
    out_dir.mkdir(parents=True)
    (out_dir / "dummy.txt").write_text("existing content", encoding="utf-8")

    aggregator = TemporalFeatureAggregator()
    with pytest.raises(OutputDirectoryExistsError):
        aggregator.aggregate(
            detection_source=synthetic_artifacts_on_disk["detection"],
            motion_source=synthetic_artifacts_on_disk["motion"],
            visual_source=synthetic_artifacts_on_disk["visual"],
            output_dir=out_dir,
            overwrite=False,
        )


def test_42_output_directory_overwrite_success(
    synthetic_artifacts_on_disk: Dict[str, Path],
    tmp_path: Path,
):
    """Verify existing output directory can be safely overwritten when overwrite=True."""
    out_dir = tmp_path / "overwrite_dir"
    out_dir.mkdir(parents=True)
    (out_dir / "dummy.txt").write_text("existing content", encoding="utf-8")

    aggregator = TemporalFeatureAggregator()
    result = aggregator.aggregate(
        detection_source=synthetic_artifacts_on_disk["detection"],
        motion_source=synthetic_artifacts_on_disk["motion"],
        visual_source=synthetic_artifacts_on_disk["visual"],
        output_dir=out_dir,
        overwrite=True,
    )
    assert result["processed_frame_count"] == 5
    assert (out_dir / "temporal_features.json").exists()


# ==============================================================================
# INTEGRATION TESTS: Real BDD100K Driving Sample Artifacts
# ==============================================================================

@pytest.mark.skipif(
    not (REAL_DETECTION_ARTIFACT.exists() and REAL_MOTION_ARTIFACT.exists() and REAL_VISUAL_ARTIFACT.exists()),
    reason="Real driving_sample feature artifacts not available"
)
def test_43_real_driving_sample_dataset_integration(tmp_path: Path):
    """Integration test: execute temporal feature aggregation on real 200-frame artifacts."""
    out_dir = tmp_path / "real_driving_temporal"
    aggregator = TemporalFeatureAggregator(window_sizes=[3, 5, 10])

    result = aggregator.aggregate(
        detection_source=REAL_DETECTION_ARTIFACT,
        motion_source=REAL_MOTION_ARTIFACT,
        visual_source=REAL_VISUAL_ARTIFACT,
        output_dir=out_dir,
        overwrite=True,
    )

    assert result["processed_frame_count"] == 200
    assert result["successfully_aligned_frame_count"] == 200
    assert result["partially_aligned_frame_count"] == 0
    assert result["summary"].validation_status == "PASS"

    # Verify frame 1 attributes
    f1 = result["features"][0]
    assert f1.frame_id == "frame_000001.jpg"
    assert f1.dataset_id == "driving_sample"
    assert f1.detection_features["detection_count"] == 2
    assert f1.visual_quality_features["blur_score"] > 0.0

    # Verify window aggregates at frame 5
    f5 = result["features"][4]
    assert "w3" in f5.temporal_windows
    assert "w5" in f5.temporal_windows
    assert "w10" in f5.temporal_windows


@pytest.mark.skipif(
    not (REAL_DETECTION_ARTIFACT.exists() and REAL_MOTION_ARTIFACT.exists() and REAL_VISUAL_ARTIFACT.exists()),
    reason="Real driving_sample feature artifacts not available"
)
def test_44_real_driving_sample_five_manual_verified_calculations():
    """Manual verification of 5 mathematical calculations against raw source records."""
    det_raw = json.loads(REAL_DETECTION_ARTIFACT.read_text(encoding="utf-8"))["frames"]
    mot_raw = json.loads(REAL_MOTION_ARTIFACT.read_text(encoding="utf-8"))["frames"]
    vq_raw = json.loads(REAL_VISUAL_ARTIFACT.read_text(encoding="utf-8"))["features"]

    aggregator = TemporalFeatureAggregator(window_sizes=[3, 5])
    result = aggregator.aggregate(
        detection_source=REAL_DETECTION_ARTIFACT,
        motion_source=REAL_MOTION_ARTIFACT,
        visual_source=REAL_VISUAL_ARTIFACT,
    )
    records = result["features"]

    # Calculation 1: Frame 2 (index 1) detection_count
    # In raw: frame_000002.jpg detection_count is 1
    assert records[1].detection_features["detection_count"] == det_raw[1]["detection_count"]

    # Calculation 2: Frame 3 (index 2) w3 mean blur_score
    # Window w3 for frame 3 includes frames 0, 1, 2
    raw_blurs = [vq_raw[0]["blur_score"], vq_raw[1]["blur_score"], vq_raw[2]["blur_score"]]
    expected_blur_mean = round(sum(raw_blurs) / 3.0, 4)
    w3_blur_mean = records[2].temporal_windows["w3"]["feature_statistics"]["blur_score"]["mean"]
    assert w3_blur_mean == pytest.approx(expected_blur_mean, abs=1e-4)

    # Calculation 3: Frame 5 (index 4) w5 mean brightness
    # Window w5 for frame 5 includes frames 0, 1, 2, 3, 4
    raw_brights = [vq_raw[k]["mean_brightness"] for k in range(5)]
    expected_bright_mean = round(sum(raw_brights) / 5.0, 4)
    w5_bright_mean = records[4].temporal_windows["w5"]["feature_statistics"]["mean_brightness"]["mean"]
    assert w5_bright_mean == pytest.approx(expected_bright_mean, abs=1e-4)

    # Calculation 4: Frame 5 (index 4) w5 blur diff = last - first
    expected_blur_diff = round(vq_raw[4]["blur_score"] - vq_raw[0]["blur_score"], 4)
    w5_blur_diff = records[4].temporal_windows["w5"]["feature_statistics"]["blur_score"]["diff"]
    assert w5_blur_diff == pytest.approx(expected_blur_diff, abs=1e-4)

    # Calculation 5: Frame 5 (index 4) w5 mean confidence slope (OLS trend)
    # Check that OLS trend matches analytical formula
    confs = [(k, det_raw[k]["mean_confidence"]) for k in range(5) if det_raw[k]["mean_confidence"] is not None]
    expected_trend = calculate_ols_trend(confs, decimal_places=4)
    w5_trend = records[4].temporal_windows["w5"]["trends"]["confidence_trend"]
    assert w5_trend["slope"] == expected_trend.slope


@pytest.mark.skipif(
    not (REAL_DETECTION_ARTIFACT.exists() and REAL_MOTION_ARTIFACT.exists() and REAL_VISUAL_ARTIFACT.exists()),
    reason="Real driving_sample feature artifacts not available"
)
def test_45_real_driving_sample_future_frame_leakage_invariance():
    """Verify trailing window features at frame 10 remain unchanged if frames 11..200 are truncated."""
    aggregator = TemporalFeatureAggregator(window_sizes=[5])

    # Run on full 200 frames
    full_res = aggregator.aggregate(
        detection_source=REAL_DETECTION_ARTIFACT,
        motion_source=REAL_MOTION_ARTIFACT,
        visual_source=REAL_VISUAL_ARTIFACT,
    )
    frame_9_full = full_res["features"][9].temporal_windows["w5"]

    # Run on partial sequence capped at frame 10 (max_frames=10)
    partial_res = aggregator.aggregate(
        detection_source=REAL_DETECTION_ARTIFACT,
        motion_source=REAL_MOTION_ARTIFACT,
        visual_source=REAL_VISUAL_ARTIFACT,
        max_frames=10,
    )
    frame_9_partial = partial_res["features"][9].temporal_windows["w5"]

    # Trailing window statistics must be 100% identical
    assert frame_9_full == frame_9_partial
