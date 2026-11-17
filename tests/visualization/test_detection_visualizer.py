"""PerceptGuard AI: Automated Test Suite for Detection Visualizer.

Covers:
1. Valid detection JSON parsing.
2. Malformed JSON handling.
3. Missing JSON file (DetectionJSONValidationError).
4. Unsupported schema version (UnsupportedSchemaVersionError).
5. Missing frame results handling.
6. Missing source image on disk.
7. Corrupt source image on disk (SourceImageDecodeError).
8. Valid bounding box coordinates validation.
9. Invalid bounding box geometry (x2 <= x1, y2 <= y1).
10. Out-of-bounds bounding box clamping.
11. Non-finite coordinates rejection (NaN / Inf).
12. Invalid confidence scores rejection (<0.0, >1.0, non-finite).
13. Missing class name fallback generation.
14. Zero-detection frame handling (successful visualization, unannotated copy).
15. Multiple detections rendered on a single frame.
16. Deterministic class color mapping consistency.
17. Label text rendering inside canvas boundaries.
18. Image dimension preservation (exact width/height matching).
19. Output directory auto-creation.
20. Existing output directory overwrite protection.
21. Explicit overwrite parameter behavior.
22. Failed visualization reporting into error catalog.
23. Summary statistics accuracy.
24. Partial run status (WARNING) vs full run status (PASS).
25. Batch visualization ordering.
26. Invariant: Zero YOLO model initialization.
27. Invariant: Zero repeated inference calls.
28. Invariant: Source images remain completely unaltered (checksum verified).
29. Real-data integration test on genuine Task 4 detection artifacts.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.visualization.detection_visualizer import (
    BoundingBoxAnnotation,
    DEFAULT_COLOR_PALETTE,
    DetectionJSONValidationError,
    DetectionVisualizer,
    FrameVisualizationRecord,
    InvalidBoundingBoxError,
    OutputPersistenceError,
    SourceImageDecodeError,
    SourceImageNotFoundError,
    UnsupportedSchemaVersionError,
    VisualizationError,
    VisualizationErrorRecord,
    VisualizationSummary,
    generate_deterministic_palette,
)


class TestDetectionVisualizer(unittest.TestCase):
    """Unit and integration test suite for DetectionVisualizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create sandbox temporary directory and locate real project artifacts."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Real project artifacts from Day 2 and Day 3 Task 4
        cls.real_frames_dir = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / "driving_sample"
            / "preprocessed_frames"
        )
        cls.real_detection_dir = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / "driving_sample"
            / "detections"
            / "batch_runs"
            / "run_20261117_115348_e1789b"
        )
        cls.real_detection_file = cls.real_detection_dir / "detection_results.json"

        # Synthetic test assets
        cls.synthetic_images_dir = cls.temp_path / "mock_images"
        cls.synthetic_images_dir.mkdir(parents=True, exist_ok=True)

        cls.sample_img_w = 640
        cls.sample_img_h = 360
        cls.test_frame_1 = cls.synthetic_images_dir / "frame_000001.jpg"
        cls.test_frame_2 = cls.synthetic_images_dir / "frame_000002.jpg"
        cls.test_frame_3 = cls.synthetic_images_dir / "frame_000003.jpg"

        for p in [cls.test_frame_1, cls.test_frame_2, cls.test_frame_3]:
            img = np.full((cls.sample_img_h, cls.sample_img_w, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(p), img)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary test artifacts."""
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def setUp(self) -> None:
        """Instantiate a fresh visualizer before each test."""
        self.visualizer = DetectionVisualizer(project_root=PROJECT_ROOT)

    def _create_mock_json(
        self,
        frame_results: list,
        schema_version: str = "1.0",
        run_id: str = "test_run_001",
        dataset_id: str = "test_dataset",
    ) -> Path:
        """Helper to create a synthetic detection_results.json."""
        payload = {
            "schema_version": schema_version,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "total_frames": len(frame_results),
            "frame_results": frame_results,
        }
        json_path = self.temp_path / f"det_{run_id}_{len(frame_results)}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return json_path

    # ==========================================================================
    # 1. JSON Schema & Input Validation
    # ==========================================================================

    def test_01_valid_detection_json(self) -> None:
        """Verify successful loading and parsing of valid detection JSON."""
        mock_file = self._create_mock_json([
            {
                "image_filename": "frame_000001.jpg",
                "image_path": str(self.test_frame_1),
                "extracted_frame_id": 1,
                "processing_status": "success",
                "detection_count": 0,
                "detections": [],
            }
        ])
        data = self.visualizer.load_and_validate_detection_json(mock_file)
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["dataset_id"], "test_dataset")
        self.assertEqual(len(data["frame_results"]), 1)

    def test_02_malformed_json(self) -> None:
        """Verify that syntax-corrupted JSON raises DetectionJSONValidationError."""
        corrupt_path = self.temp_path / "corrupt.json"
        corrupt_path.write_text("{ unquoted_key: invalid_json, ", encoding="utf-8")

        with self.assertRaises(DetectionJSONValidationError) as ctx:
            self.visualizer.load_and_validate_detection_json(corrupt_path)
        self.assertIn("Malformed detection JSON", str(ctx.exception))

    def test_03_missing_json_file(self) -> None:
        """Verify that non-existent detection file raises DetectionJSONValidationError."""
        missing = self.temp_path / "non_existent_file.json"
        with self.assertRaises(DetectionJSONValidationError) as ctx:
            self.visualizer.load_and_validate_detection_json(missing)
        self.assertIn("not found", str(ctx.exception))

    def test_04_unsupported_schema_version(self) -> None:
        """Verify that schema versions other than 1.0 raise UnsupportedSchemaVersionError."""
        mock_file = self._create_mock_json([], schema_version="2.5", run_id="v2_run")
        with self.assertRaises(UnsupportedSchemaVersionError) as ctx:
            self.visualizer.load_and_validate_detection_json(mock_file)
        self.assertIn("Unsupported schema version '2.5'", str(ctx.exception))

    def test_05_missing_frame_results(self) -> None:
        """Verify that JSON missing 'frame_results' key raises DetectionJSONValidationError."""
        bad_json = self.temp_path / "bad_schema.json"
        with open(bad_json, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0", "run_id": "r1", "dataset_id": "d1"}, f)

        with self.assertRaises(DetectionJSONValidationError) as ctx:
            self.visualizer.load_and_validate_detection_json(bad_json)
        self.assertIn("Missing required field 'frame_results'", str(ctx.exception))

    # ==========================================================================
    # 2. Source Image Resolution & Reading
    # ==========================================================================

    def test_06_missing_source_image(self) -> None:
        """Verify that missing source image results in a failed frame record."""
        frame_dict = {
            "image_filename": "completely_missing_frame.jpg",
            "image_path": str(self.temp_path / "completely_missing_frame.jpg"),
            "extracted_frame_id": 999,
            "detection_count": 0,
            "detections": [],
        }
        rec = self.visualizer.visualize_single_frame(frame_dict)
        self.assertEqual(rec.status, "failed")
        self.assertIn("could not be resolved", str(rec.error_message))

    def test_07_corrupt_source_image(self) -> None:
        """Verify that unreadable or corrupted source image raises SourceImageDecodeError."""
        corrupt_img_path = self.temp_path / "corrupt_img.jpg"
        corrupt_img_path.write_bytes(b"NOT_AN_IMAGE_RANDOM_BYTES_123456789")

        frame_dict = {
            "image_filename": "corrupt_img.jpg",
            "image_path": str(corrupt_img_path),
            "detection_count": 0,
            "detections": [],
        }
        rec = self.visualizer.visualize_single_frame(frame_dict)
        self.assertEqual(rec.status, "failed")
        self.assertIn("failed to decode", str(rec.error_message))

    # ==========================================================================
    # 3. Bounding Box & Coordinate Validation
    # ==========================================================================

    def test_08_valid_bounding_box(self) -> None:
        """Verify that a valid bounding box produces a valid BoundingBoxAnnotation."""
        det = {
            "detection_index": 0,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.88,
            "bbox_xyxy": [100.0, 50.0, 250.0, 180.0],
        }
        ann = self.visualizer.validate_and_create_annotation(det, 640, 360)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.bbox_xyxy, [100.0, 50.0, 250.0, 180.0])
        self.assertEqual(ann.class_name, "car")
        self.assertEqual(ann.confidence, 0.88)
        self.assertEqual(ann.color_bgr, DEFAULT_COLOR_PALETTE[2])

    def test_09_invalid_bounding_box_geometry(self) -> None:
        """Verify that inverted boxes (x2 <= x1 or y2 <= y1) are rejected."""
        # x2 < x1
        det1 = {"bbox_xyxy": [200.0, 50.0, 100.0, 180.0], "confidence": 0.8, "class_id": 2}
        self.assertIsNone(self.visualizer.validate_and_create_annotation(det1, 640, 360))

        # y2 == y1 (zero height)
        det2 = {"bbox_xyxy": [100.0, 50.0, 200.0, 50.0], "confidence": 0.8, "class_id": 2}
        self.assertIsNone(self.visualizer.validate_and_create_annotation(det2, 640, 360))

    def test_10_out_of_bounds_bounding_box_clamping(self) -> None:
        """Verify that boxes extending beyond boundaries are clamped to [0, W] and [0, H]."""
        det = {
            "class_id": 7,
            "class_name": "truck",
            "confidence": 0.75,
            "bbox_xyxy": [-20.0, -10.0, 700.0, 400.0],
        }
        ann = self.visualizer.validate_and_create_annotation(det, 640, 360)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.bbox_xyxy, [0.0, 0.0, 640.0, 360.0])

    def test_11_non_finite_coordinates(self) -> None:
        """Verify that NaN or Inf coordinates are safely rejected."""
        for val in [float("nan"), float("inf"), float("-inf")]:
            det = {
                "class_id": 2,
                "confidence": 0.9,
                "bbox_xyxy": [10.0, 10.0, val, 100.0],
            }
            ann = self.visualizer.validate_and_create_annotation(det, 640, 360)
            self.assertIsNone(ann)

    def test_12_invalid_confidence(self) -> None:
        """Verify that confidence out of bounds (<0.0, >1.0, NaN) is rejected."""
        for conf in [-0.5, 1.5, float("nan"), "invalid_str"]:
            det = {
                "class_id": 2,
                "confidence": conf,
                "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
            }
            ann = self.visualizer.validate_and_create_annotation(det, 640, 360)
            self.assertIsNone(ann)

    def test_13_missing_class_name(self) -> None:
        """Verify fallback class name generation when class_name is empty or omitted."""
        det = {
            "class_id": 9,
            "confidence": 0.5,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }
        ann = self.visualizer.validate_and_create_annotation(det, 640, 360)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.class_name, "class_9")

    # ==========================================================================
    # 4. Rendering & Visual Contracts
    # ==========================================================================

    def test_14_zero_detection_frame(self) -> None:
        """Verify that zero-detection frames are visualized cleanly without errors."""
        dest_img = self.temp_path / "zero_det_out.jpg"
        frame_dict = {
            "image_filename": "frame_000001.jpg",
            "image_path": str(self.test_frame_1),
            "extracted_frame_id": 1,
            "processing_status": "success",
            "detection_count": 0,
            "detections": [],
        }
        rec = self.visualizer.visualize_single_frame(
            frame_dict,
            output_image_path=dest_img,
            overwrite=True,
        )
        self.assertEqual(rec.status, "success")
        self.assertEqual(rec.boxes_rendered, 0)
        self.assertTrue(dest_img.is_file())
        self.assertGreater(dest_img.stat().st_size, 0)

    def test_15_multiple_detections(self) -> None:
        """Verify accurate rendering when a frame contains multiple detections."""
        dest_img = self.temp_path / "multi_det_out.jpg"
        frame_dict = {
            "image_filename": "frame_000002.jpg",
            "image_path": str(self.test_frame_2),
            "extracted_frame_id": 2,
            "detection_count": 2,
            "detections": [
                {
                    "detection_index": 0,
                    "class_id": 9,
                    "class_name": "traffic light",
                    "confidence": 0.438,
                    "bbox_xyxy": [219.0, 238.0, 276.0, 359.0],
                },
                {
                    "detection_index": 1,
                    "class_id": 7,
                    "class_name": "truck",
                    "confidence": 0.296,
                    "bbox_xyxy": [219.0, 1.0, 350.0, 355.0],
                },
            ],
        }
        rec = self.visualizer.visualize_single_frame(
            frame_dict,
            output_image_path=dest_img,
            overwrite=True,
        )
        self.assertEqual(rec.status, "success")
        self.assertEqual(rec.boxes_rendered, 2)
        self.assertTrue(dest_img.is_file())

    def test_16_class_based_deterministic_colors(self) -> None:
        """Verify that the exact same class_id always gets the exact same color."""
        pal1 = generate_deterministic_palette(80)
        pal2 = generate_deterministic_palette(80)
        self.assertEqual(pal1, pal2)

        # Check distinctness for key autonomous classes
        self.assertNotEqual(pal1[0], pal1[2])  # person != car
        self.assertNotEqual(pal1[2], pal1[7])  # car != truck
        self.assertNotEqual(pal1[7], pal1[9])  # truck != traffic light

    def test_17_label_rendering(self) -> None:
        """Verify that label text and banner are drawn without clipping."""
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        # Annotation near top edge (y1 = 2) should place label inside box
        ann = BoundingBoxAnnotation(
            bbox_xyxy=[10.0, 2.0, 100.0, 50.0],
            class_id=2,
            class_name="car",
            confidence=0.92,
            color_bgr=(255, 128, 0),
        )
        annotated, count = self.visualizer.render_annotations(img, [ann])
        self.assertEqual(count, 1)
        # Verify that canvas was modified
        self.assertFalse(np.array_equal(img, annotated))

    def test_18_image_dimension_preservation(self) -> None:
        """Verify that output annotated image dimensions exactly match input dimensions."""
        dest_img = self.temp_path / "dim_preserve.jpg"
        frame_dict = {
            "image_filename": "frame_000001.jpg",
            "image_path": str(self.test_frame_1),
            "detection_count": 1,
            "detections": [{
                "class_id": 2, "class_name": "car", "confidence": 0.8,
                "bbox_xyxy": [50.0, 50.0, 150.0, 150.0]
            }],
        }
        self.visualizer.visualize_single_frame(frame_dict, output_image_path=dest_img)
        loaded = cv2.imread(str(dest_img))
        self.assertEqual(loaded.shape[1], self.sample_img_w)
        self.assertEqual(loaded.shape[0], self.sample_img_h)

    # ==========================================================================
    # 5. Output Management & Overwrite Protection
    # ==========================================================================

    def test_19_output_directory_creation(self) -> None:
        """Verify that non-existent output directories are automatically created."""
        nested_out = self.temp_path / "deep" / "nested" / "vis_out"
        mock_file = self._create_mock_json([
            {"image_filename": "frame_000001.jpg", "image_path": str(self.test_frame_1), "detection_count": 0, "detections": []}
        ])
        summary, _, _, out_root = self.visualizer.visualize_batch(
            detection_file=mock_file,
            output_dir=nested_out,
            overwrite=True,
        )
        self.assertTrue((out_root / "annotated_frames").is_dir())
        self.assertTrue((out_root / "visualization_summary.json").is_file())

    def test_20_existing_output_protection(self) -> None:
        """Verify that OutputPersistenceError is raised if output dir exists and overwrite=False."""
        existing_out = self.temp_path / "protected_vis_dir"
        existing_out.mkdir(parents=True, exist_ok=True)
        mock_file = self._create_mock_json([])

        with self.assertRaises(OutputPersistenceError) as ctx:
            self.visualizer.visualize_batch(
                detection_file=mock_file,
                output_dir=existing_out,
                overwrite=False,
            )
        self.assertIn("already exists and overwrite=False", str(ctx.exception))

    def test_21_explicit_overwrite(self) -> None:
        """Verify that existing output directory is successfully reused when overwrite=True."""
        reused_out = self.temp_path / "reused_vis_dir"
        reused_out.mkdir(parents=True, exist_ok=True)
        mock_file = self._create_mock_json([
            {"image_filename": "frame_000001.jpg", "image_path": str(self.test_frame_1), "detection_count": 0, "detections": []}
        ])
        summary, _, _, out_root = self.visualizer.visualize_batch(
            detection_file=mock_file,
            output_dir=reused_out,
            overwrite=True,
        )
        self.assertEqual(summary.successful_visualizations, 1)

    def test_22_failed_visualization_reporting(self) -> None:
        """Verify that failed frames are cataloged into visualization_errors.json."""
        out_dir = self.temp_path / "errors_vis_dir"
        mock_file = self._create_mock_json([
            {"image_filename": "non_existent_image_xyz.jpg", "detection_count": 0, "detections": []}
        ])
        summary, _, errors, _ = self.visualizer.visualize_batch(
            detection_file=mock_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(summary.failed_visualizations, 1)
        self.assertEqual(len(errors), 1)
        err_json = json.loads((out_dir / "visualization_errors.json").read_text(encoding="utf-8"))
        self.assertEqual(err_json["error_count"], 1)

    def test_23_summary_count_accuracy(self) -> None:
        """Verify mathematical accuracy of all summary metrics and aggregations."""
        out_dir = self.temp_path / "summary_vis_dir"
        mock_file = self._create_mock_json([
            # 1 frame with 2 detections
            {
                "image_filename": "frame_000001.jpg", "image_path": str(self.test_frame_1),
                "detection_count": 2,
                "detections": [
                    {"class_id": 2, "class_name": "car", "confidence": 0.8, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
                    {"class_id": 9, "class_name": "traffic light", "confidence": 0.7, "bbox_xyxy": [60.0, 60.0, 100.0, 100.0]}
                ]
            },
            # 1 frame with 0 detections
            {
                "image_filename": "frame_000002.jpg", "image_path": str(self.test_frame_2),
                "detection_count": 0, "detections": []
            }
        ])
        summary, _, _, _ = self.visualizer.visualize_batch(
            detection_file=mock_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(summary.total_frames_attempted, 2)
        self.assertEqual(summary.successful_visualizations, 2)
        self.assertEqual(summary.failed_visualizations, 0)
        self.assertEqual(summary.frames_with_detections, 1)
        self.assertEqual(summary.frames_without_detections, 1)
        self.assertEqual(summary.total_boxes_rendered, 2)
        self.assertEqual(summary.final_status, "PASS")

    def test_24_partial_run_status(self) -> None:
        """Verify that slicing or max_frames results in status WARNING."""
        out_dir = self.temp_path / "partial_vis_dir"
        mock_file = self._create_mock_json([
            {"image_filename": "frame_000001.jpg", "image_path": str(self.test_frame_1), "detection_count": 0, "detections": []},
            {"image_filename": "frame_000002.jpg", "image_path": str(self.test_frame_2), "detection_count": 0, "detections": []},
            {"image_filename": "frame_000003.jpg", "image_path": str(self.test_frame_3), "detection_count": 0, "detections": []},
        ])
        summary, _, _, _ = self.visualizer.visualize_batch(
            detection_file=mock_file,
            output_dir=out_dir,
            max_frames=1,
            overwrite=True,
        )
        self.assertEqual(summary.total_frames_attempted, 1)
        self.assertEqual(summary.final_status, "WARNING")

    def test_25_batch_visualization_ordering(self) -> None:
        """Verify sequential execution preserves frame ordering from detection results."""
        out_dir = self.temp_path / "order_vis_dir"
        mock_file = self._create_mock_json([
            {"image_filename": "frame_000001.jpg", "image_path": str(self.test_frame_1), "detection_count": 0, "detections": []},
            {"image_filename": "frame_000002.jpg", "image_path": str(self.test_frame_2), "detection_count": 0, "detections": []},
        ])
        _, records, _, _ = self.visualizer.visualize_batch(
            detection_file=mock_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(records[0].image_filename, "frame_000001.jpg")
        self.assertEqual(records[1].image_filename, "frame_000002.jpg")

    # ==========================================================================
    # 6. Architectural Invariants
    # ==========================================================================

    def test_26_no_yolo_model_initialization(self) -> None:
        """Verify that YOLO model loader is never called during visualization."""
        with patch("src.detection.yolo_model_loader.YOLOModelLoader.load_model") as mock_load:
            dest_img = self.temp_path / "no_model_out.jpg"
            frame_dict = {
                "image_filename": "frame_000001.jpg",
                "image_path": str(self.test_frame_1),
                "detection_count": 0,
                "detections": [],
            }
            self.visualizer.visualize_single_frame(frame_dict, output_image_path=dest_img)
            mock_load.assert_not_called()

    def test_27_no_repeated_inference(self) -> None:
        """Verify that SingleImageDetector.detect is never called during visualization."""
        with patch("src.detection.single_image_detector.SingleImageDetector.detect") as mock_detect:
            dest_img = self.temp_path / "no_infer_out.jpg"
            frame_dict = {
                "image_filename": "frame_000001.jpg",
                "image_path": str(self.test_frame_1),
                "detection_count": 0,
                "detections": [],
            }
            self.visualizer.visualize_single_frame(frame_dict, output_image_path=dest_img)
            mock_detect.assert_not_called()

    def test_28_source_image_unchanged(self) -> None:
        """Verify that source image bytes remain strictly unchanged after visualization."""
        before_hash = hashlib.sha256(self.test_frame_1.read_bytes()).hexdigest()
        dest_img = self.temp_path / "hash_check_out.jpg"
        frame_dict = {
            "image_filename": "frame_000001.jpg",
            "image_path": str(self.test_frame_1),
            "detection_count": 1,
            "detections": [{
                "class_id": 2, "class_name": "car", "confidence": 0.8,
                "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]
            }],
        }
        self.visualizer.visualize_single_frame(frame_dict, output_image_path=dest_img)
        after_hash = hashlib.sha256(self.test_frame_1.read_bytes()).hexdigest()
        self.assertEqual(before_hash, after_hash)

    # ==========================================================================
    # 7. Real Artifact Integration Test
    # ==========================================================================

    def test_29_real_artifact_integration(self) -> None:
        """Integration test using actual Task 4 detection JSON and real Day 2 driving frames."""
        if not self.real_detection_file.is_file():
            self.skipTest(f"Real detection file not found: {self.real_detection_file}")

        real_out = self.temp_path / "real_vis_integration"
        summary, records, errors, out_root = self.visualizer.visualize_batch(
            detection_file=self.real_detection_file,
            output_dir=real_out,
            max_frames=5,
            overwrite=True,
        )
        self.assertEqual(summary.total_frames_attempted, 5)
        self.assertEqual(summary.successful_visualizations, 5)
        self.assertEqual(summary.failed_visualizations, 0)
        self.assertEqual(len(errors), 0)
        self.assertEqual(summary.final_status, "WARNING")  # partial smoke run
        self.assertTrue((out_root / "annotated_frames" / "frame_000001_annotated.jpg").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
