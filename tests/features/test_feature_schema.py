"""Automated unit and integration test suite for PerceptGuard AI Feature Schema.

Validates:
- FeatureFieldDescriptor definitions and metadata.
- FeatureSchemaDefinition queries (required, optional, group breakdowns).
- FeatureRecord dataclass instantiation and serialization.
- Strict and permissive record-level field validations (confidence, geometry, kinematics).
- Upstream artifact validation (detection results, tracking outputs, metadata).
- Rejection of corrupt JSON, non-monotonic frame orders, and duplicate frame IDs.
- Real-data integration with existing Day 2–Day 4 outputs.
"""

from pathlib import Path
import tempfile
import unittest

from src.features.feature_schema import (
    FEATURE_FIELD_REGISTRY,
    FEATURE_SCHEMA_VERSION,
    FeatureFieldDescriptor,
    FeatureRecord,
    FeatureSchemaDefinition,
    FeatureSchemaError,
    IncompatibleArtifactError,
    InvalidFeatureRecordError,
    MissingRequiredFieldError,
    ValidationResult,
    validate_feature_record,
    validate_input_artifacts,
)


class TestFeatureSchema(unittest.TestCase):
    """Test suite for FeatureSchemaDefinition, FeatureRecord, and Validation engines."""

    def _create_valid_record_dict(self) -> dict:
        return {
            "frame_id": "frame_000001.jpg",
            "source_frame_index": 1,
            "dataset_id": "driving_sample",
            "timestamp_seconds": 0.0,
            "confidence": 0.85,
            "class_id": 2,
            "class_name": "car",
            "bbox_x1": 100.0,
            "bbox_y1": 120.0,
            "bbox_x2": 250.0,
            "bbox_y2": 280.0,
            "bbox_width": 150.0,
            "bbox_height": 160.0,
            "bbox_area": 24000.0,
            "frame_detection_count": 3,
            "center_x": 175.0,
            "center_y": 200.0,
            "displacement_pixels": 4.5,
            "velocity_pixels_per_second": 22.5,
            "movement_direction_radians": 0.7854,
            "movement_direction_degrees": 45.0,
            "blur_score": 350.2,
            "brightness": 110.5,
            "contrast": 55.4,
            "track_id": 1,
            "track_observation_count": 5,
            "is_visible": True,
            "observation_ratio": 1.0,
            "consecutive_lost_frames": 0,
            "reacquisition_count": 0,
            "tracking_state": "TRACKING",
        }

    # --------------------------------------------------------------------------
    # Schema Registry & Descriptors
    # --------------------------------------------------------------------------

    def test_01_schema_version_and_registry_count(self) -> None:
        self.assertEqual(FeatureSchemaDefinition.SCHEMA_VERSION, "1.0")
        fields = FeatureSchemaDefinition.get_all_fields()
        self.assertGreaterEqual(len(fields), 25)

    def test_02_field_descriptor_properties(self) -> None:
        desc = FeatureSchemaDefinition.get_descriptor("confidence")
        self.assertIsNotNone(desc)
        self.assertEqual(desc.data_type, "float")
        self.assertEqual(desc.feature_group, "detection")
        self.assertTrue(desc.is_required)
        self.assertFalse(desc.is_nullable)

    def test_03_query_required_and_optional_fields(self) -> None:
        req_names = FeatureSchemaDefinition.get_required_field_names()
        opt_names = FeatureSchemaDefinition.get_optional_field_names()

        self.assertIn("frame_id", req_names)
        self.assertIn("confidence", req_names)
        self.assertIn("track_id", req_names)
        self.assertIn("bbox_x1", req_names)

        self.assertIn("displacement_pixels", opt_names)
        self.assertIn("blur_score", opt_names)
        self.assertIn("timestamp_seconds", opt_names)

    def test_04_query_fields_by_group(self) -> None:
        det_fields = FeatureSchemaDefinition.get_fields_by_group("detection")
        mot_fields = FeatureSchemaDefinition.get_fields_by_group("motion")
        vis_fields = FeatureSchemaDefinition.get_fields_by_group("visual_quality")
        trk_fields = FeatureSchemaDefinition.get_fields_by_group("tracking")

        self.assertGreater(len(det_fields), 0)
        self.assertGreater(len(mot_fields), 0)
        self.assertGreater(len(vis_fields), 0)
        self.assertGreater(len(trk_fields), 0)

    # --------------------------------------------------------------------------
    # Record-Level Validation
    # --------------------------------------------------------------------------

    def test_05_valid_record_passes_validation(self) -> None:
        rec = self._create_valid_record_dict()
        res = validate_feature_record(rec)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)

    def test_06_missing_required_field_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        del rec["confidence"]
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("confidence" in e for e in res.errors))

    def test_07_null_required_field_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["track_id"] = None
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("track_id" in e for e in res.errors))

    def test_08_optional_fields_as_none_passes_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["displacement_pixels"] = None
        rec["velocity_pixels_per_second"] = None
        rec["blur_score"] = None
        rec["brightness"] = None
        rec["contrast"] = None
        res = validate_feature_record(rec)
        self.assertTrue(res.is_valid)

    def test_09_invalid_confidence_range_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["confidence"] = 1.25
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("confidence" in e for e in res.errors))

        rec["confidence"] = -0.1
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)

    def test_10_non_finite_coordinates_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["bbox_x1"] = float("nan")
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("non-finite" in e for e in res.errors))

    def test_11_inverted_bounding_box_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["bbox_x1"] = 250.0
        rec["bbox_x2"] = 100.0  # x2 < x1
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("x2 <= x1" in e for e in res.errors))

    def test_12_invalid_tracking_state_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["tracking_state"] = "FLYING_AWAY"
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("tracking_state" in e for e in res.errors))

    def test_13_invalid_observation_ratio_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["observation_ratio"] = 1.5
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)

    def test_14_invalid_brightness_range_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["brightness"] = 300.0
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)

    def test_15_negative_track_id_fails_validation(self) -> None:
        rec = self._create_valid_record_dict()
        rec["track_id"] = -5
        res = validate_feature_record(rec)
        self.assertFalse(res.is_valid)

    def test_16_feature_record_dataclass_validation(self) -> None:
        rec_obj = FeatureRecord(
            frame_id="frame_000001.jpg",
            source_frame_index=1,
            dataset_id="driving_sample",
            confidence=0.90,
            class_id=2,
            class_name="car",
            bbox_x1=10.0,
            bbox_y1=10.0,
            bbox_x2=50.0,
            bbox_y2=50.0,
            bbox_width=40.0,
            bbox_height=40.0,
            bbox_area=1600.0,
            frame_detection_count=1,
            center_x=30.0,
            center_y=30.0,
            track_id=1,
        )
        res = validate_feature_record(rec_obj)
        self.assertTrue(res.is_valid)

    def test_17_validation_result_raise_if_invalid(self) -> None:
        res = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        with self.assertRaises(InvalidFeatureRecordError):
            res.raise_if_invalid()

    # --------------------------------------------------------------------------
    # Input Artifacts Integration Validation
    # --------------------------------------------------------------------------

    def test_18_missing_detection_file_fails_artifact_validation(self) -> None:
        res = validate_input_artifacts(detection_file="non_existent_det_path.json")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("does not exist" in e for e in res.errors))

    def test_19_empty_artifact_file_fails_validation(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            res = validate_input_artifacts(detection_file=p)
            self.assertFalse(res.is_valid)
            self.assertTrue(any("is empty" in e for e in res.errors))
        finally:
            if p.exists():
                p.unlink()

    def test_20_malformed_json_artifact_fails_validation(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{ broken json")
            p = Path(f.name)
        try:
            res = validate_input_artifacts(tracking_file=p)
            self.assertFalse(res.is_valid)
            self.assertTrue(any("invalid JSON" in e for e in res.errors))
        finally:
            if p.exists():
                p.unlink()

    def test_21_unsupported_schema_version_fails_validation(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"schema_version": "88.0", "frame_results": []}')
            p = Path(f.name)
        try:
            res = validate_input_artifacts(detection_file=p)
            self.assertFalse(res.is_valid)
            self.assertTrue(any("unsupported schema_version" in e for e in res.errors))
        finally:
            if p.exists():
                p.unlink()

    def test_22_duplicate_frame_id_fails_validation(self) -> None:
        data = {
            "schema_version": "1.0",
            "frame_results": [
                {"frame_id": "frame_001.jpg", "source_frame_index": 1},
                {"frame_id": "frame_001.jpg", "source_frame_index": 2},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            import json
            json.dump(data, f)
            p = Path(f.name)
        try:
            res = validate_input_artifacts(detection_file=p)
            self.assertFalse(res.is_valid)
            self.assertTrue(any("Duplicate frame" in e for e in res.errors))
        finally:
            if p.exists():
                p.unlink()

    def test_23_non_monotonic_frame_ordering_fails_validation(self) -> None:
        data = {
            "schema_version": "1.0",
            "frame_results": [
                {"frame_id": "frame_002.jpg", "source_frame_index": 5},
                {"frame_id": "frame_001.jpg", "source_frame_index": 2},  # 2 < 5
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            import json
            json.dump(data, f)
            p = Path(f.name)
        try:
            res = validate_input_artifacts(detection_file=p)
            self.assertFalse(res.is_valid)
            self.assertTrue(any("Non-monotonic frame ordering" in e for e in res.errors))
        finally:
            if p.exists():
                p.unlink()

    def test_24_mismatched_dataset_ids_fails_validation(self) -> None:
        d1 = {"schema_version": "1.0", "dataset_id": "set_A", "frame_results": []}
        d2 = {"schema_version": "1.0", "dataset_id": "set_B", "frame_results": []}

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f1:
            import json
            json.dump(d1, f1)
            p1 = Path(f1.name)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f2:
            json.dump(d2, f2)
            p2 = Path(f2.name)

        try:
            res = validate_input_artifacts(detection_file=p1, tracking_file=p2)
            self.assertFalse(res.is_valid)
            self.assertTrue(any("Incompatible dataset IDs" in e for e in res.errors))
        finally:
            if p1.exists():
                p1.unlink()
            if p2.exists():
                p2.unlink()

    def test_25_real_project_artifacts_validation(self) -> None:
        real_det = Path("datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json")
        real_mot = Path("datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json")
        real_meta = Path("datasets/processed/driving_sample/metadata.csv")

        if not (real_det.exists() and real_mot.exists() and real_meta.exists()):
            self.skipTest("Real project artifacts not available for integration test.")

        res = validate_input_artifacts(
            detection_file=real_det,
            tracking_file=real_mot,
            metadata_file=real_meta,
            strict=False,
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)


if __name__ == "__main__":
    unittest.main()
