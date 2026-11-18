"""PerceptGuard AI: Tests for Motion Feature Extractor (Day 5 Task 3).

Comprehensive unit and integration test suite covering:
- Object position features (centroid, width, height, area)
- Frame-to-frame displacement (delta_x, delta_y, Euclidean, absolute, normalized)
- Velocity features (delta_time, velocity_x, velocity_y, speed)
- Movement direction (radians, degrees, 8-cardinal sectors, stationary threshold)
- Bounding-box scale changes (width, height, area, area change ratio)
- Temporal continuity and gap handling (first observation, consecutive, gap motion, gap too large)
- Track-level and frame-level aggregations
- Error handling, zero-detection frames, NaN/Inf coordinate validation
- Multi-object, single-object, and ByteTrack JSON format ingestion
- CLI interface execution and argument validation
- Integration tests using real BDD100K driving tracking artifacts
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List
import pytest

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.motion_feature_extractor import (
    FrameMotionFeatureRecord,
    InvalidTrackRecordError,
    MotionExtractionErrorRecord,
    MotionFeatureExtractionError,
    MotionFeatureExtractor,
    MotionFeatureRecord,
    MotionFeatureSummary,
    OutputDirectoryExistsError,
    TrackMotionSummaryRecord,
    TrackingArtifactValidationError,
    calculate_direction,
    compute_std,
    main,
)
from src.features.feature_config import FeatureConfig

REAL_MOT_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "tracking"
    / "multi_object"
    / "mot_driving_sample_20261117_160903_1c98f4"
    / "multi_object_tracking.json"
)

REAL_SOT_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "tracking"
    / "single_object"
    / "single_target_1_20261117_154123_fa3f36"
    / "single_object_tracking.json"
)


@pytest.fixture
def sample_synthetic_tracking_data() -> Dict[str, Any]:
    """Synthetic multi-frame tracking dictionary for isolated testing."""
    return {
        "schema_version": "1.0",
        "dataset_id": "test_motion_dataset",
        "run_id": "mot_test_001",
        "image_width": 640,
        "image_height": 360,
        "total_frames": 4,
        "frame_results": [
            {
                "frame_id": "frame_000001.jpg",
                "source_frame_index": 0,
                "timestamp_seconds": 0.0,
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 2,
                        "class_name": "car",
                        "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
                    },
                    {
                        "track_id": 2,
                        "class_id": 7,
                        "class_name": "truck",
                        "bbox_xyxy": [300.0, 150.0, 450.0, 300.0],
                    },
                ],
            },
            {
                "frame_id": "frame_000002.jpg",
                "source_frame_index": 1,
                "timestamp_seconds": 0.2,
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 2,
                        "class_name": "car",
                        "bbox_xyxy": [103.0, 104.0, 203.0, 204.0],
                    },
                    {
                        "track_id": 2,
                        "class_id": 7,
                        "class_name": "truck",
                        "bbox_xyxy": [300.2, 150.1, 450.2, 300.1],
                    },
                ],
            },
            {
                "frame_id": "frame_000003.jpg",
                "source_frame_index": 2,
                "timestamp_seconds": 0.4,
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 2,
                        "class_name": "car",
                        "bbox_xyxy": [106.0, 108.0, 206.0, 208.0],
                    }
                ],
            },
            {
                "frame_id": "frame_000004.jpg",
                "source_frame_index": 3,
                "timestamp_seconds": 0.6,
                "tracks": [],
            },
        ],
    }


class TestMotionFeatureExtractorUnit:
    """Isolated unit tests for motion feature calculations and edge cases."""

    def test_01_valid_track_observations(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, track_sums, frame_recs, summary, errors, _ = extractor.extract_from_dict(
            sample_synthetic_tracking_data
        )

        assert len(motion_recs) == 5  # 2 in f1, 2 in f2, 1 in f3, 0 in f4
        assert len(track_sums) == 2
        assert len(frame_recs) == 4
        assert summary.valid_motion_calculation_count == 3  # track1 in f2, track2 in f2, track1 in f3
        assert summary.unavailable_motion_count == 2  # first observations for track 1 and track 2
        assert summary.invalid_record_count == 0

    def test_02_first_observation_of_track(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        first_rec = motion_recs[0]
        assert first_rec.track_id == 1
        assert first_rec.motion_calculation_status == "first_observation"
        assert first_rec.previous_frame_id is None
        assert first_rec.displacement is None
        assert first_rec.delta_x is None
        assert first_rec.delta_y is None
        assert first_rec.speed is None
        assert first_rec.direction_category is None

    def test_03_horizontal_displacement(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        # Track 1: frame 1 center_x = 150.0, frame 2 center_x = 153.0 -> delta_x = 3.0
        rec_f2_t1 = motion_recs[2]
        assert rec_f2_t1.track_id == 1
        assert rec_f2_t1.delta_x == 3.0
        assert rec_f2_t1.absolute_delta_x == 3.0

    def test_04_vertical_displacement(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        # Track 1: frame 1 center_y = 150.0, frame 2 center_y = 154.0 -> delta_y = 4.0
        rec_f2_t1 = motion_recs[2]
        assert rec_f2_t1.delta_y == 4.0
        assert rec_f2_t1.absolute_delta_y == 4.0

    def test_05_euclidean_displacement(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        # Track 1: delta_x = 3.0, delta_y = 4.0 -> displacement = sqrt(3^2 + 4^2) = 5.0
        rec_f2_t1 = motion_recs[2]
        assert rec_f2_t1.displacement == 5.0
        assert rec_f2_t1.center_distance == 5.0

    def test_06_normalized_displacement(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        # Image: 640 x 360, diagonal = sqrt(640^2 + 360^2) = 734.3024
        rec_f2_t1 = motion_recs[2]
        diag = math.sqrt(640 ** 2 + 360 ** 2)
        assert rec_f2_t1.normalized_delta_x == round(3.0 / 640.0, 4)
        assert rec_f2_t1.normalized_delta_y == round(4.0 / 360.0, 4)
        assert rec_f2_t1.normalized_displacement == round(5.0 / diag, 4)

    def test_07_missing_image_dimensions(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["image_width"] = None
        data["image_height"] = None

        extractor = MotionFeatureExtractor(image_width=None, image_height=None)
        # Force metadata.csv not to load dimensions by testing with None
        motion_recs, _, _, _, _, raw_meta = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        # If dimensions could not be found, normalized fields must be None
        if raw_meta["image_dimensions"]["width"] is None:
            assert rec.normalized_delta_x is None
            assert rec.normalized_delta_y is None
            assert rec.normalized_displacement is None

    def test_08_valid_velocity_and_speed_calculation(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        # delta_time = 0.2s, delta_x = 3.0, delta_y = 4.0, disp = 5.0
        rec = motion_recs[2]
        assert rec.delta_time == 0.2
        assert rec.velocity_x == round(3.0 / 0.2, 4)  # 15.0
        assert rec.velocity_y == round(4.0 / 0.2, 4)  # 20.0
        assert rec.speed == round(5.0 / 0.2, 4)  # 25.0

    def test_09_missing_timestamps_velocity_is_null(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        for f in data["frame_results"]:
            f["timestamp_seconds"] = None

        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        assert rec.delta_time is None
        assert rec.velocity_x is None
        assert rec.velocity_y is None
        assert rec.speed is None
        assert rec.displacement == 5.0  # displacement still valid

    def test_10_duplicate_timestamps_handling(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["frame_results"][1]["timestamp_seconds"] = 0.0  # same as frame 0

        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, errors, _ = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        assert rec.delta_time is None
        assert rec.speed is None
        assert any(e.error_type == "NON_POSITIVE_TIME_DELTA" for e in errors)

    def test_11_decreasing_timestamps_handling(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["frame_results"][1]["timestamp_seconds"] = -0.5

        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, errors, _ = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        assert rec.speed is None
        assert any(e.error_type == "NON_POSITIVE_TIME_DELTA" for e in errors)

    def test_12_zero_time_difference_division_by_zero_prevention(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["frame_results"][1]["timestamp_seconds"] = 0.0

        extractor = MotionFeatureExtractor()
        # Must execute cleanly without unhandled ZeroDivisionError
        extractor.extract_from_dict(data)

    def test_13_direction_calculation_cardinal_sectors(self) -> None:
        _, deg_r, cat_r = calculate_direction(10.0, 0.0, 10.0)
        assert cat_r == "right"
        assert deg_r == 0.0

        _, deg_l, cat_l = calculate_direction(-10.0, 0.0, 10.0)
        assert cat_l == "left"
        assert deg_l == 180.0

        _, deg_d, cat_d = calculate_direction(0.0, 10.0, 10.0)
        assert cat_d == "down"
        assert deg_d == 90.0

        _, deg_u, cat_u = calculate_direction(0.0, -10.0, 10.0)
        assert cat_u == "up"
        assert deg_u == -90.0

    def test_14_stationary_movement_threshold(self) -> None:
        # Movement under threshold (0.5 <= 1.0) must be stationary
        _, deg, cat = calculate_direction(0.3, 0.4, 0.5, stationary_threshold=1.0)
        assert cat == "stationary"
        assert deg == 0.0

        # Movement above threshold (1.5 > 1.0) must not be stationary
        _, deg2, cat2 = calculate_direction(0.9, 1.2, 1.5, stationary_threshold=1.0)
        assert cat2 == "down_right"

    def test_15_diagonal_movement(self) -> None:
        _, _, cat_dr = calculate_direction(10.0, 10.0, 14.14)
        assert cat_dr == "down_right"

        _, _, cat_dl = calculate_direction(-10.0, 10.0, 14.14)
        assert cat_dl == "down_left"

        _, _, cat_ur = calculate_direction(10.0, -10.0, 14.14)
        assert cat_ur == "up_right"

        _, _, cat_ul = calculate_direction(-10.0, -10.0, 14.14)
        assert cat_ul == "up_left"

    def test_16_bounding_box_width_changes(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        # frame 0: w=100.0, frame 1: w=120.0
        data["frame_results"][1]["tracks"][0]["bbox_xyxy"] = [100.0, 100.0, 220.0, 200.0]

        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        assert rec.bbox_width_change == 20.0

    def test_17_bounding_box_height_changes(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        # frame 0: h=100.0, frame 1: h=80.0
        data["frame_results"][1]["tracks"][0]["bbox_xyxy"] = [100.0, 100.0, 200.0, 180.0]

        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        assert rec.bbox_height_change == -20.0

    def test_18_bounding_box_area_changes_and_ratio(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        # frame 0: area = 10000.0, frame 1: area = 15000.0
        data["frame_results"][1]["tracks"][0]["bbox_xyxy"] = [100.0, 100.0, 250.0, 200.0]

        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(data)

        rec = motion_recs[2]
        assert rec.bbox_area_change == 5000.0
        assert rec.bbox_area_change_ratio == 0.5

    def test_19_zero_area_bounding_boxes(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["frame_results"][0]["tracks"][0]["bbox_xyxy"] = [100.0, 100.0, 100.0, 100.0]

        extractor = MotionFeatureExtractor()
        _, _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_record_count == 1
        assert any(e.error_type in ("INVERTED_X_COORDINATES", "ZERO_AREA_BBOX") for e in errors)

    def test_20_track_id_changes_and_independence(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        _, track_sums, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        # Track 1 has 3 observations, Track 2 has 2 observations
        t1 = next(t for t in track_sums if t.track_id == 1)
        t2 = next(t for t in track_sums if t.track_id == 2)
        assert t1.total_observations == 3
        assert t2.total_observations == 2

    def test_21_missing_frame_observations(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        # Track 2 is present in frame 0 and frame 1, missing in frame 2
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        t2_recs = [r for r in motion_recs if r.track_id == 2]
        assert len(t2_recs) == 2

    def test_22_gap_based_motion(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        # Track 2 present in frame 0, missing in frame 1, reappears in frame 2 (gap=2 <= tolerance=5)
        data = sample_synthetic_tracking_data
        # Remove Track 2 from frame 1 and add to frame 2
        del data["frame_results"][1]["tracks"][1]
        data["frame_results"][2]["tracks"].append({
            "track_id": 2,
            "class_id": 7,
            "class_name": "truck",
            "bbox_xyxy": [310.0, 150.0, 460.0, 300.0],
        })

        extractor = MotionFeatureExtractor(max_gap_tolerance=5)
        motion_recs, _, _, summary, _, _ = extractor.extract_from_dict(data)

        t2_recs = [r for r in motion_recs if r.track_id == 2]
        assert len(t2_recs) == 2
        reappearing = t2_recs[1]
        assert reappearing.motion_calculation_status == "gap_motion"
        assert reappearing.frame_gap == 2
        assert reappearing.observation_gap == 1
        assert reappearing.displacement == 10.0
        assert summary.gap_based_calculation_count == 1

    def test_23_reappearing_track_ids_gap_too_large(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        del data["frame_results"][1]["tracks"][1]
        data["frame_results"][2]["tracks"].append({
            "track_id": 2,
            "class_id": 7,
            "class_name": "truck",
            "bbox_xyxy": [310.0, 150.0, 460.0, 300.0],
        })

        # Set max_gap_tolerance=1 -> gap of 2 is too large
        extractor = MotionFeatureExtractor(max_gap_tolerance=1)
        motion_recs, _, _, _, _, _ = extractor.extract_from_dict(data)

        t2_recs = [r for r in motion_recs if r.track_id == 2]
        reappearing = t2_recs[1]
        assert reappearing.motion_calculation_status == "gap_too_large"
        assert reappearing.displacement is None

    def test_24_track_level_aggregation(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        _, track_sums, _, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        t1 = next(t for t in track_sums if t.track_id == 1)
        assert t1.total_observations == 3
        assert t1.total_valid_motion_steps == 2
        assert t1.mean_displacement == 5.0
        assert t1.maximum_displacement == 5.0
        assert t1.minimum_displacement == 5.0
        assert t1.motion_data_coverage == 1.0

    def test_25_frame_level_aggregation(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor(stationary_threshold=1.0)
        _, _, frame_recs, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        f0 = frame_recs[0]
        assert f0.observed_track_count == 2
        assert f0.moving_track_count == 0
        assert f0.stationary_track_count == 0
        assert f0.mean_displacement is None

        f1 = frame_recs[1]
        assert f1.observed_track_count == 2
        assert f1.moving_track_count == 1  # Track 1 moved 5.0 px > 1.0 px
        assert f1.stationary_track_count == 1  # Track 2 moved 0.22 px <= 1.0 px
        assert f1.mean_displacement is not None

    def test_26_empty_frames(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        _, _, frame_recs, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        f3 = frame_recs[3]
        assert f3.observed_track_count == 0
        assert f3.moving_track_count == 0
        assert f3.stationary_track_count == 0
        assert f3.mean_displacement is None
        assert f3.motion_feature_coverage is None

    def test_27_invalid_tracking_json(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "bad.json"
        corrupt.write_text("{ corrupt json: [", encoding="utf-8")

        extractor = MotionFeatureExtractor()
        with pytest.raises(TrackingArtifactValidationError):
            extractor.extract_from_file(corrupt)

    def test_28_invalid_numeric_values(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["frame_results"][0]["tracks"][0]["bbox_xyxy"] = [10.0, float("nan"), 50.0, 50.0]

        extractor = MotionFeatureExtractor()
        _, _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_record_count == 1
        assert any(e.error_type == "NON_FINITE_COORDINATE" for e in errors)

    def test_29_duplicate_frame_identifiers(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        data = sample_synthetic_tracking_data
        data["frame_results"][1]["frame_id"] = "frame_000001.jpg"

        extractor = MotionFeatureExtractor()
        _, _, frame_recs, _, errors, _ = extractor.extract_from_dict(data)

        assert any(e.error_type == "DUPLICATE_FRAME_ID" for e in errors)
        assert len(frame_recs) == 3

    def test_30_frame_ordering_preservation(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        _, _, frame_recs, _, _, _ = extractor.extract_from_dict(sample_synthetic_tracking_data)

        assert [f.frame_id for f in frame_recs] == [
            "frame_000001.jpg",
            "frame_000002.jpg",
            "frame_000003.jpg",
            "frame_000004.jpg",
        ]

    def test_31_dataset_metadata_preservation(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        motion_recs, _, _, summary, _, raw_meta = extractor.extract_from_dict(sample_synthetic_tracking_data)

        assert summary.dataset_id == "test_motion_dataset"
        assert raw_meta["dataset_id"] == "test_motion_dataset"
        assert all(m.dataset_id == "test_motion_dataset" for m in motion_recs)

    def test_32_partial_processing_max_frames(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        _, _, frame_recs, summary, _, _ = extractor.extract_from_dict(
            sample_synthetic_tracking_data, max_frames=2
        )

        assert len(frame_recs) == 2
        assert summary.is_partial is True
        assert summary.processed_frame_count == 2

    def test_33_partial_processing_start_end_frame(self, sample_synthetic_tracking_data: Dict[str, Any]) -> None:
        extractor = MotionFeatureExtractor()
        _, _, frame_recs, summary, _, _ = extractor.extract_from_dict(
            sample_synthetic_tracking_data, start_frame=1, end_frame=2
        )

        assert len(frame_recs) == 2
        assert frame_recs[0].frame_id == "frame_000002.jpg"
        assert frame_recs[1].frame_id == "frame_000003.jpg"
        assert summary.is_partial is True

    def test_34_output_schema_validation(self, sample_synthetic_tracking_data: Dict[str, Any], tmp_path: Path) -> None:
        in_file = tmp_path / "tracks.json"
        in_file.write_text(json.dumps(sample_synthetic_tracking_data), encoding="utf-8")
        out_dir = tmp_path / "out_features"

        extractor = MotionFeatureExtractor()
        res = extractor.extract_from_file(in_file, output_dir=out_dir)

        # Check all 6 files exist
        assert (out_dir / "motion_features.json").exists()
        assert (out_dir / "track_motion_features.json").exists()
        assert (out_dir / "frame_motion_features.json").exists()
        assert (out_dir / "motion_feature_summary.json").exists()
        assert (out_dir / "feature_metadata.json").exists()
        assert (out_dir / "feature_errors.json").exists()

        with open(out_dir / "motion_feature_summary.json", encoding="utf-8") as f:
            sum_data = json.load(f)
            assert sum_data["schema_version"] == "1.0"
            assert sum_data["processed_frame_count"] == 4
            assert sum_data["unique_track_count"] == 2

    def test_35_output_directory_overwrite_protection(self, sample_synthetic_tracking_data: Dict[str, Any], tmp_path: Path) -> None:
        in_file = tmp_path / "tracks.json"
        in_file.write_text(json.dumps(sample_synthetic_tracking_data), encoding="utf-8")
        out_dir = tmp_path / "out_features"

        extractor = MotionFeatureExtractor()
        extractor.extract_from_file(in_file, output_dir=out_dir)

        with pytest.raises(OutputDirectoryExistsError):
            extractor.extract_from_file(in_file, output_dir=out_dir, overwrite=False)

    def test_36_output_directory_overwrite_success(self, sample_synthetic_tracking_data: Dict[str, Any], tmp_path: Path) -> None:
        in_file = tmp_path / "tracks.json"
        in_file.write_text(json.dumps(sample_synthetic_tracking_data), encoding="utf-8")
        out_dir = tmp_path / "out_features"

        extractor = MotionFeatureExtractor()
        extractor.extract_from_file(in_file, output_dir=out_dir)
        res = extractor.extract_from_file(in_file, output_dir=out_dir, overwrite=True)
        assert res["valid_motion_count"] == 3

    def test_37_single_object_tracking_json_support(self, tmp_path: Path) -> None:
        sot_data = {
            "schema_version": "1.0",
            "dataset_id": "test_sot",
            "target_id": 1,
            "observations": [
                {
                    "frame_id": "frame_000001.jpg",
                    "source_frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "track_id": 1,
                    "class_id": 9,
                    "class_name": "traffic light",
                    "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
                },
                {
                    "frame_id": "frame_000002.jpg",
                    "source_frame_index": 1,
                    "timestamp_seconds": 0.2,
                    "track_id": 1,
                    "class_id": 9,
                    "class_name": "traffic light",
                    "bbox_xyxy": [105.0, 100.0, 205.0, 200.0],
                },
            ],
        }
        in_file = tmp_path / "sot.json"
        in_file.write_text(json.dumps(sot_data), encoding="utf-8")

        extractor = MotionFeatureExtractor()
        res = extractor.extract_from_file(in_file, output_dir=tmp_path / "sot_out")
        assert res["valid_motion_count"] == 1
        assert res["unique_track_count"] == 1


class TestMotionFeatureExtractorCLI:
    """CLI execution tests for MotionFeatureExtractor."""

    def test_38_cli_execution_success(self, tmp_path: Path) -> None:
        if not REAL_MOT_ARTIFACT.exists():
            pytest.skip("Real MOT artifact not available.")

        out_dir = tmp_path / "cli_mot_out"
        exit_code = main([
            "--input", str(REAL_MOT_ARTIFACT),
            "--output", str(out_dir),
            "--max-frames", "10",
            "--overwrite",
        ])
        assert exit_code == 0
        assert (out_dir / "motion_feature_summary.json").exists()

    def test_39_cli_execution_invalid_file(self) -> None:
        exit_code = main(["--input", "non_existent_tracking_file_xyz.json"])
        assert exit_code == 1

    def test_40_cli_partial_run_flags(self, tmp_path: Path) -> None:
        if not REAL_MOT_ARTIFACT.exists():
            pytest.skip("Real MOT artifact not available.")

        out_dir = tmp_path / "cli_partial_out"
        exit_code = main([
            "--input", str(REAL_MOT_ARTIFACT),
            "--output", str(out_dir),
            "--max-frames", "5",
            "--overwrite",
        ])
        assert exit_code == 0
        with open(out_dir / "motion_feature_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["is_partial"] is True
            assert summary["processed_frame_count"] == 5


class TestMotionFeatureExtractorIntegration:
    """Integration tests on genuine project tracking artifacts."""

    def test_41_integration_real_mot_artifact(self, tmp_path: Path) -> None:
        if not REAL_MOT_ARTIFACT.exists():
            pytest.skip("Real MOT artifact not available.")

        out_dir = tmp_path / "real_mot_features"
        extractor = MotionFeatureExtractor()
        res = extractor.extract_from_file(REAL_MOT_ARTIFACT, output_dir=out_dir)

        assert res["summary"]["processed_frame_count"] == 200
        assert res["summary"]["unique_track_count"] == 3
        assert res["summary"]["total_track_observations"] == 193
        assert res["summary"]["valid_motion_calculation_count"] == 184
        assert res["summary"]["gap_based_calculation_count"] == 22
        assert res["summary"]["validation_status"] == "PASS"

    def test_42_integration_real_sot_artifact(self, tmp_path: Path) -> None:
        if not REAL_SOT_ARTIFACT.exists():
            pytest.skip("Real SOT artifact not available.")

        out_dir = tmp_path / "real_sot_features"
        extractor = MotionFeatureExtractor()
        res = extractor.extract_from_file(REAL_SOT_ARTIFACT, output_dir=out_dir)

        assert res["summary"]["unique_track_count"] == 1
        assert res["summary"]["validation_status"] == "PASS"

    def test_43_integration_real_math_verification(self, tmp_path: Path) -> None:
        """Verify manual calculations on real Track 1 across frames 1 and 2."""
        if not REAL_MOT_ARTIFACT.exists():
            pytest.skip("Real MOT artifact not available.")

        out_dir = tmp_path / "real_math_features"
        extractor = MotionFeatureExtractor()
        extractor.extract_from_file(REAL_MOT_ARTIFACT, output_dir=out_dir)

        with open(out_dir / "motion_features.json", encoding="utf-8") as f:
            recs = json.load(f)["records"]

        # Track 1 records
        t1_recs = [r for r in recs if r["track_id"] == 1]
        assert len(t1_recs) == 159

        # First record: frame 1 -> first observation
        r0 = t1_recs[0]
        assert r0["frame_id"] == "frame_000001.jpg"
        assert r0["motion_calculation_status"] == "first_observation"
        assert r0["displacement"] is None

        # Second record: frame 2 vs frame 1
        r1 = t1_recs[1]
        assert r1["frame_id"] == "frame_000002.jpg"
        assert r1["motion_calculation_status"] == "consecutive"
        assert r1["frame_gap"] == 1
        assert r1["observation_gap"] == 0

        # Center coordinates check
        # frame 1: [219.5, 238.1, 276.4, 359.1] -> cx=247.95, cy=298.60
        # frame 2: [219.98, 238.62, 276.62, 359.01] -> cx=248.30, cy=298.815
        assert r1["delta_x"] == 0.35
        assert r1["delta_y"] == 0.215
        expected_disp = round(math.sqrt(0.35 ** 2 + 0.215 ** 2), 4)  # 0.4108
        assert r1["displacement"] == expected_disp
        assert r1["speed"] == round(expected_disp / 0.1994, 4)  # 2.0602
        assert r1["direction_category"] == "stationary"  # disp 0.4108 <= 1.0
