"""PerceptGuard AI: Tests for Detection Feature Extractor (Day 5 Task 2).

Comprehensive unit and integration test suite covering:
- Detection-level feature extraction (identity, confidence, bounding box geometry, normalized geometry)
- Frame-level detection feature aggregations and zero-detection frame handling
- Coordinate, confidence, class, and dimension validation
- Missing, inverted, zero-area, NaN, and Infinity handling
- Duplicate detection index and duplicate frame identifier handling
- Single-image and batch detection JSON formats
- Partial sequence processing (start_frame, end_frame, max_frames)
- Directory overwrite protections and persistence of all 5 versioned artifacts
- CLI interface execution and argument validation
- Real-data integration tests using genuine Day 3/Day 4 detection artifacts
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

from src.features.detection_feature_extractor import (
    DetectionExtractionErrorRecord,
    DetectionFeatureExtractionError,
    DetectionFeatureExtractor,
    DetectionFeatureRecord,
    DetectionFeatureSummary,
    FrameDetectionFeatureRecord,
    InputArtifactValidationError,
    OutputDirectoryExistsError,
    categorize_confidence,
    compute_std,
    main,
)
from src.features.feature_config import FeatureConfig

REAL_DETECTION_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "detections"
    / "batch_runs"
    / "run_20261117_115348_e1789b"
    / "detection_results.json"
)

REAL_SINGLE_IMAGE_ARTIFACT = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "detections"
    / "single_image"
    / "frame_000001_detections.json"
)


@pytest.fixture
def sample_raw_batch_data() -> Dict[str, Any]:
    """Synthetic multi-frame detection payload for isolated unit tests."""
    return {
        "schema_version": "1.0",
        "dataset_id": "test_dataset",
        "run_id": "run_test_001",
        "total_frames": 3,
        "frame_results": [
            {
                "image_filename": "frame_000001.jpg",
                "extracted_frame_id": 1,
                "source_frame_index": 0,
                "timestamp_seconds": 0.0,
                "image_width": 640,
                "image_height": 360,
                "detection_count": 2,
                "detections": [
                    {
                        "detection_index": 0,
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.85,
                        "bbox_xyxy": [100.0, 150.0, 200.0, 250.0],
                    },
                    {
                        "detection_index": 1,
                        "class_id": 7,
                        "class_name": "truck",
                        "confidence": 0.45,
                        "bbox_xyxy": [300.0, 100.0, 450.0, 300.0],
                    },
                ],
            },
            {
                "image_filename": "frame_000002.jpg",
                "extracted_frame_id": 2,
                "source_frame_index": 1,
                "timestamp_seconds": 0.2,
                "image_width": 640,
                "image_height": 360,
                "detection_count": 1,
                "detections": [
                    {
                        "detection_index": 0,
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.32,
                        "bbox_xyxy": [50.0, 80.0, 100.0, 200.0],
                    }
                ],
            },
            {
                "image_filename": "frame_000003.jpg",
                "extracted_frame_id": 3,
                "source_frame_index": 2,
                "timestamp_seconds": 0.4,
                "image_width": 640,
                "image_height": 360,
                "detection_count": 0,
                "detections": [],
            },
        ],
    }


class TestDetectionFeatureExtractorUnit:
    """Isolated unit tests for detection feature extraction logic."""

    def test_01_valid_detection_extraction(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        det_recs, frame_recs, summary, errors, _ = extractor.extract_from_dict(sample_raw_batch_data)

        assert len(det_recs) == 3
        assert len(frame_recs) == 3
        assert summary.valid_detection_count == 3
        assert summary.invalid_detection_count == 0
        assert summary.validation_status == "PASS"

        first = det_recs[0]
        assert first.dataset_id == "test_dataset"
        assert first.frame_id == "frame_000001.jpg"
        assert first.class_name == "car"
        assert first.confidence == 0.85
        assert first.confidence_category == "high"

    def test_02_multiple_detections_in_one_frame(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        det_recs, frame_recs, _, _, _ = extractor.extract_from_dict(sample_raw_batch_data)

        frame1_dets = [d for d in det_recs if d.frame_id == "frame_000001.jpg"]
        assert len(frame1_dets) == 2
        assert frame_recs[0].detection_count == 2
        assert frame_recs[0].unique_class_count == 2
        assert frame_recs[0].class_wise_detection_counts == {"car": 1, "truck": 1}

    def test_03_single_detection_in_frame(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        det_recs, frame_recs, _, _, _ = extractor.extract_from_dict(sample_raw_batch_data)

        frame2 = frame_recs[1]
        assert frame2.detection_count == 1
        assert frame2.unique_class_count == 1
        assert frame2.mean_confidence == 0.32
        assert frame2.minimum_confidence == 0.32
        assert frame2.maximum_confidence == 0.32
        assert frame2.confidence_std == 0.0

    def test_04_zero_detection_frames(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        _, frame_recs, summary, _, _ = extractor.extract_from_dict(sample_raw_batch_data)

        frame3 = frame_recs[2]
        assert frame3.detection_count == 0
        assert frame3.unique_class_count == 0
        assert frame3.mean_confidence is None
        assert frame3.minimum_confidence is None
        assert frame3.maximum_confidence is None
        assert frame3.confidence_std is None
        assert frame3.total_bbox_area is None
        assert frame3.mean_bbox_area is None
        assert frame3.maximum_bbox_area is None
        assert frame3.mean_bbox_width is None
        assert frame3.mean_bbox_height is None
        assert frame3.class_wise_detection_counts == {}
        assert summary.zero_detection_frame_count == 1

    def test_05_confidence_calculations_and_categories(self) -> None:
        assert categorize_confidence(0.95) == "high"
        assert categorize_confidence(0.75) == "high"
        assert categorize_confidence(0.749) == "medium"
        assert categorize_confidence(0.40) == "medium"
        assert categorize_confidence(0.399) == "low"
        assert categorize_confidence(0.01) == "low"

        # Statistical test
        confs = [0.85, 0.45]
        mean_c = sum(confs) / 2.0
        std_c = compute_std(confs, mean_c)
        assert round(mean_c, 4) == 0.65
        assert round(std_c, 4) == 0.2

    def test_06_bounding_box_calculations(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        det_recs, _, _, _, _ = extractor.extract_from_dict(sample_raw_batch_data)

        # bbox: [100.0, 150.0, 200.0, 250.0]
        car = det_recs[0]
        assert car.bbox_width == 100.0
        assert car.bbox_height == 100.0
        assert car.bbox_area == 10000.0
        assert car.bbox_center_x == 150.0
        assert car.bbox_center_y == 200.0

    def test_07_normalized_bounding_box_calculations(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        det_recs, _, _, _, _ = extractor.extract_from_dict(sample_raw_batch_data)

        # image size: 640 x 360
        car = det_recs[0]
        assert car.normalized_center_x == round(150.0 / 640.0, 4)
        assert car.normalized_center_y == round(200.0 / 360.0, 4)
        assert car.normalized_width == round(100.0 / 640.0, 4)
        assert car.normalized_height == round(100.0 / 360.0, 4)
        assert car.normalized_bbox_area == round(10000.0 / (640.0 * 360.0), 4)

    def test_08_missing_image_dimensions(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["image_width"] = None
        data["frame_results"][0]["image_height"] = None

        extractor = DetectionFeatureExtractor()
        det_recs, _, _, errors, _ = extractor.extract_from_dict(data)

        car = det_recs[0]
        assert car.normalized_center_x is None
        assert car.normalized_center_y is None
        assert car.normalized_width is None
        assert car.normalized_height is None
        assert car.normalized_bbox_area is None

    def test_09_invalid_bounding_boxes_inverted_x(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["bbox_xyxy"] = [250.0, 100.0, 200.0, 200.0]

        extractor = DetectionFeatureExtractor()
        det_recs, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "INVERTED_X_COORDINATES" for e in errors)
        # Invalid detection must not be added to valid records
        assert not any(d.class_name == "car" for d in det_recs)

    def test_10_invalid_bounding_boxes_inverted_y(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["bbox_xyxy"] = [100.0, 250.0, 200.0, 200.0]

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "INVERTED_Y_COORDINATES" for e in errors)

    def test_11_invalid_bounding_boxes_negative_coords(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["bbox_xyxy"] = [-10.0, 50.0, 100.0, 150.0]

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "NEGATIVE_COORDINATES" for e in errors)

    def test_12_invalid_bounding_boxes_zero_area(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["bbox_xyxy"] = [100.0, 100.0, 100.0, 100.0]

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type in ("INVERTED_X_COORDINATES", "ZERO_AREA_BBOX") for e in errors)

    def test_13_invalid_confidence_negative(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["confidence"] = -0.15

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "CONFIDENCE_OUT_OF_RANGE" for e in errors)

    def test_14_invalid_confidence_exceeds_one(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["confidence"] = 1.25

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "CONFIDENCE_OUT_OF_RANGE" for e in errors)

    def test_15_invalid_confidence_non_numeric(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["confidence"] = "invalid_string"

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "INVALID_CONFIDENCE" for e in errors)

    def test_16_missing_class_information(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["class_id"] = -1
        data["frame_results"][0]["detections"][1]["class_name"] = "  "

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 2
        assert any(e.error_type == "INVALID_CLASS_ID" for e in errors)
        assert any(e.error_type == "INVALID_CLASS_NAME" for e in errors)

    def test_17_duplicate_detection_indices(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][1]["detection_index"] = 0

        extractor = DetectionFeatureExtractor()
        _, _, _, errors, _ = extractor.extract_from_dict(data)

        assert any(e.error_type == "DUPLICATE_DETECTION_INDEX" for e in errors)

    def test_18_duplicate_frame_identifiers(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][1]["image_filename"] = "frame_000001.jpg"
        data["frame_results"][1]["extracted_frame_id"] = 1

        extractor = DetectionFeatureExtractor()
        _, frame_recs, _, errors, _ = extractor.extract_from_dict(data)

        assert any(e.error_type == "DUPLICATE_FRAME_ID" for e in errors)
        # Duplicate frame skipped
        assert len(frame_recs) == 2

    def test_19_invalid_json_handling(self, tmp_path: Path) -> None:
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{ unclosed json: ", encoding="utf-8")

        extractor = DetectionFeatureExtractor()
        with pytest.raises(InputArtifactValidationError):
            extractor.extract_from_file(corrupt_file)

    def test_20_missing_input_file_handling(self) -> None:
        extractor = DetectionFeatureExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_file("non_existent_file_xyz123.json")

    def test_21_empty_input_file_handling(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")

        extractor = DetectionFeatureExtractor()
        with pytest.raises(InputArtifactValidationError):
            extractor.extract_from_file(empty_file)

    def test_22_nan_confidence_handling(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["confidence"] = float("nan")

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "NON_FINITE_CONFIDENCE" for e in errors)

    def test_23_infinity_coordinate_handling(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["bbox_xyxy"] = [10.0, 10.0, float("inf"), 50.0]

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "NON_FINITE_COORDINATE" for e in errors)

    def test_24_nan_coordinate_handling(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        data = sample_raw_batch_data
        data["frame_results"][0]["detections"][0]["bbox_xyxy"] = [10.0, float("nan"), 50.0, 50.0]

        extractor = DetectionFeatureExtractor()
        _, _, summary, errors, _ = extractor.extract_from_dict(data)

        assert summary.invalid_detection_count == 1
        assert any(e.error_type == "NON_FINITE_COORDINATE" for e in errors)

    def test_25_frame_ordering_preservation(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        _, frame_recs, _, _, _ = extractor.extract_from_dict(sample_raw_batch_data)

        assert [f.frame_id for f in frame_recs] == [
            "frame_000001.jpg",
            "frame_000002.jpg",
            "frame_000003.jpg",
        ]

    def test_26_dataset_metadata_preservation(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        det_recs, _, summary, _, raw_meta = extractor.extract_from_dict(sample_raw_batch_data)

        assert summary.dataset_id == "test_dataset"
        assert raw_meta["dataset_id"] == "test_dataset"
        assert all(d.dataset_id == "test_dataset" for d in det_recs)

    def test_27_partial_batch_processing_max_frames(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        _, frame_recs, summary, _, _ = extractor.extract_from_dict(
            sample_raw_batch_data, max_frames=2
        )

        assert len(frame_recs) == 2
        assert summary.is_partial is True
        assert summary.processed_frame_count == 2

    def test_28_partial_batch_processing_start_end_frame(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor = DetectionFeatureExtractor()
        _, frame_recs, summary, _, _ = extractor.extract_from_dict(
            sample_raw_batch_data, start_frame=1, end_frame=2
        )

        assert len(frame_recs) == 2
        assert frame_recs[0].frame_id == "frame_000002.jpg"
        assert frame_recs[1].frame_id == "frame_000003.jpg"
        assert summary.is_partial is True

    def test_29_output_json_schema_validation(self, sample_raw_batch_data: Dict[str, Any], tmp_path: Path) -> None:
        in_file = tmp_path / "detections.json"
        in_file.write_text(json.dumps(sample_raw_batch_data), encoding="utf-8")

        out_dir = tmp_path / "features_out"
        extractor = DetectionFeatureExtractor()
        res = extractor.extract_from_file(in_file, output_dir=out_dir)

        # Check all 5 artifacts
        assert (out_dir / "detection_features.json").exists()
        assert (out_dir / "frame_detection_features.json").exists()
        assert (out_dir / "detection_feature_summary.json").exists()
        assert (out_dir / "feature_metadata.json").exists()
        assert (out_dir / "feature_errors.json").exists()

        # Validate summary schema
        with open(out_dir / "detection_feature_summary.json", encoding="utf-8") as f:
            sum_data = json.load(f)
            assert sum_data["schema_version"] == "1.0"
            assert sum_data["processed_frame_count"] == 3
            assert sum_data["valid_detection_count"] == 3

    def test_30_output_directory_overwrite_protection(self, sample_raw_batch_data: Dict[str, Any], tmp_path: Path) -> None:
        in_file = tmp_path / "detections.json"
        in_file.write_text(json.dumps(sample_raw_batch_data), encoding="utf-8")
        out_dir = tmp_path / "features_out"

        extractor = DetectionFeatureExtractor()
        extractor.extract_from_file(in_file, output_dir=out_dir)

        # Second extraction without overwrite must fail
        with pytest.raises(OutputDirectoryExistsError):
            extractor.extract_from_file(in_file, output_dir=out_dir, overwrite=False)

    def test_31_output_directory_overwrite_success(self, sample_raw_batch_data: Dict[str, Any], tmp_path: Path) -> None:
        in_file = tmp_path / "detections.json"
        in_file.write_text(json.dumps(sample_raw_batch_data), encoding="utf-8")
        out_dir = tmp_path / "features_out"

        extractor = DetectionFeatureExtractor()
        extractor.extract_from_file(in_file, output_dir=out_dir)
        # With overwrite=True, succeeds
        res = extractor.extract_from_file(in_file, output_dir=out_dir, overwrite=True)
        assert res["valid_detection_count"] == 3

    def test_32_single_image_detection_artifact_support(self, tmp_path: Path) -> None:
        single_img_data = {
            "schema_version": "1.0",
            "image": {
                "filename": "single_test.jpg",
                "width": 640,
                "height": 360,
            },
            "result": {
                "processing_status": "success",
                "detection_count": 1,
                "detections": [
                    {
                        "detection_index": 0,
                        "class_id": 9,
                        "class_name": "traffic light",
                        "confidence": 0.65,
                        "bbox_xyxy": [100.0, 100.0, 150.0, 200.0],
                    }
                ],
            },
        }
        in_file = tmp_path / "single_image_det.json"
        in_file.write_text(json.dumps(single_img_data), encoding="utf-8")

        extractor = DetectionFeatureExtractor()
        res = extractor.extract_from_file(in_file, output_dir=tmp_path / "single_out")
        assert res["processed_frame_count"] == 1
        assert res["valid_detection_count"] == 1

    def test_33_custom_decimal_precision_rounding(self, sample_raw_batch_data: Dict[str, Any]) -> None:
        extractor_2dec = DetectionFeatureExtractor(decimal_places=2)
        det_recs_2, _, _, _, _ = extractor_2dec.extract_from_dict(sample_raw_batch_data)
        # normalized_center_x for car: 150/640 = 0.234375 -> 0.23
        assert det_recs_2[0].normalized_center_x == 0.23

        extractor_6dec = DetectionFeatureExtractor(decimal_places=6)
        det_recs_6, _, _, _, _ = extractor_6dec.extract_from_dict(sample_raw_batch_data)
        assert det_recs_6[0].normalized_center_x == 0.234375


class TestDetectionFeatureExtractorCLI:
    """Tests for Detection Feature Extractor CLI."""

    def test_34_cli_execution_success(self, tmp_path: Path) -> None:
        if not REAL_DETECTION_ARTIFACT.exists():
            pytest.skip("Real detection artifact not available on disk.")

        out_dir = tmp_path / "cli_out"
        exit_code = main([
            "--input", str(REAL_DETECTION_ARTIFACT),
            "--output", str(out_dir),
            "--max-frames", "10",
            "--overwrite",
        ])
        assert exit_code == 0
        assert (out_dir / "detection_feature_summary.json").exists()

    def test_35_cli_execution_invalid_file(self) -> None:
        exit_code = main(["--input", "does_not_exist_file.json"])
        assert exit_code == 1

    def test_36_cli_partial_run_flags(self, tmp_path: Path) -> None:
        if not REAL_DETECTION_ARTIFACT.exists():
            pytest.skip("Real detection artifact not available on disk.")

        out_dir = tmp_path / "cli_partial_out"
        exit_code = main([
            "--input", str(REAL_DETECTION_ARTIFACT),
            "--output", str(out_dir),
            "--max-frames", "5",
            "--overwrite",
        ])
        assert exit_code == 0
        with open(out_dir / "detection_feature_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["is_partial"] is True
            assert summary["processed_frame_count"] == 5


class TestDetectionFeatureExtractorIntegration:
    """Integration tests on genuine project detection artifacts."""

    def test_37_integration_real_artifact_batch_runs(self, tmp_path: Path) -> None:
        if not REAL_DETECTION_ARTIFACT.exists():
            pytest.skip("Real detection artifact not available on disk.")

        out_dir = tmp_path / "real_run_features"
        extractor = DetectionFeatureExtractor()
        res = extractor.extract_from_file(REAL_DETECTION_ARTIFACT, output_dir=out_dir)

        assert res["processed_frame_count"] == 200
        assert res["valid_detection_count"] == 211
        assert res["summary"]["zero_detection_frame_count"] == 34
        assert res["summary"]["invalid_detection_count"] == 0
        assert res["summary"]["validation_status"] == "PASS"

    def test_38_integration_real_artifact_empty_frames_verification(self, tmp_path: Path) -> None:
        if not REAL_DETECTION_ARTIFACT.exists():
            pytest.skip("Real detection artifact not available on disk.")

        out_dir = tmp_path / "real_empty_features"
        extractor = DetectionFeatureExtractor()
        extractor.extract_from_file(REAL_DETECTION_ARTIFACT, output_dir=out_dir)

        with open(out_dir / "frame_detection_features.json", encoding="utf-8") as f:
            frames_data = json.load(f)

        empty_frames = [f for f in frames_data["frames"] if f["detection_count"] == 0]
        assert len(empty_frames) == 34
        for ef in empty_frames:
            assert ef["unique_class_count"] == 0
            assert ef["mean_confidence"] is None
            assert ef["confidence_std"] is None
            assert ef["total_bbox_area"] is None
            assert ef["class_wise_detection_counts"] == {}

    def test_39_integration_real_artifact_bounding_box_sample_math(self, tmp_path: Path) -> None:
        """Verify mathematical calculations on real frame 1, detection 0."""
        if not REAL_DETECTION_ARTIFACT.exists():
            pytest.skip("Real detection artifact not available on disk.")

        out_dir = tmp_path / "real_math_features"
        extractor = DetectionFeatureExtractor()
        extractor.extract_from_file(REAL_DETECTION_ARTIFACT, output_dir=out_dir)

        with open(out_dir / "detection_features.json", encoding="utf-8") as f:
            dets = json.load(f)["records"]

        # First detection: traffic light, bbox: [219.5, 238.1, 276.4, 359.1] in 640x360
        d0 = dets[0]
        assert d0["frame_id"] == "frame_000001.jpg"
        assert d0["class_name"] == "traffic light"
        assert d0["confidence"] == 0.4381
        assert d0["confidence_category"] == "medium"
        assert d0["x_min"] == 219.5
        assert d0["y_min"] == 238.1
        assert d0["x_max"] == 276.4
        assert d0["y_max"] == 359.1
        assert d0["bbox_width"] == round(276.4 - 219.5, 4)  # 56.9
        assert d0["bbox_height"] == round(359.1 - 238.1, 4)  # 121.0
        assert d0["bbox_area"] == round(56.9 * 121.0, 4)  # 6884.9
        assert d0["bbox_center_x"] == round((219.5 + 276.4) / 2.0, 4)  # 247.95
        assert d0["bbox_center_y"] == round((238.1 + 359.1) / 2.0, 4)  # 298.6
        assert d0["normalized_center_x"] == round(247.95 / 640.0, 4)  # 0.3874
        assert d0["normalized_center_y"] == round(298.6 / 360.0, 4)  # 0.8294
        assert d0["normalized_width"] == round(56.9 / 640.0, 4)  # 0.0889
        assert d0["normalized_height"] == round(121.0 / 360.0, 4)  # 0.3361
        assert d0["normalized_bbox_area"] == round(6884.9 / (640.0 * 360.0), 4)  # 0.0299
