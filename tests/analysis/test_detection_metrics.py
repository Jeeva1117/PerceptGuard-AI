"""PerceptGuard AI: Automated Test Suite for Detection Metrics & Analysis Engine.

Covers 30 required verification categories:
1. Valid detection input parsing.
2. Empty detection input handling.
3. Zero-detection frames aggregation.
4. Confidence statistics (mean, median, min, max, std dev).
5. Class-wise statistics aggregation.
6. Frame-wise metrics calculation.
7. Temporal variation metrics.
8. Missing timestamps handled safely.
9. Invalid confidence values filtered.
10. Invalid bounding boxes handled safely.
11. Missing required fields rejected.
12. Duplicate frame identifiers handled safely.
13. Failed frame records handling.
14. Missing ground-truth reported as 'unavailable'.
15. Valid ground-truth bipartite matching.
16. Box IoU mathematical calculation.
17. Precision and recall calculation.
18. F1-score calculation.
19. Output JSON schema and file structure.
20. Schema versioning validation.
21. CLI runner execution.
22. Partial analysis status (WARNING).
23. Error handling and domain exceptions.
24. Deterministic metric calculation.
25. Empty frame results list.
26. Confidence threshold adjustment sensitivity.
27. Multi-class distribution metrics.
28. Malformed JSON syntax rejection.
29. Unsupported schema version rejection.
30. Real Task 4 detection output integration test.
"""

from copy import deepcopy
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

from src.analysis.detection_metrics import (
    ClassMetricRecord,
    DetectionMetrics,
    DetectionMetricsError,
    FrameMetricRecord,
    GroundTruthEvaluationError,
    GroundTruthEvaluationResult,
    InvalidDetectionDataError,
    MetricsAnalysisSummary,
    MetricsPersistenceError,
    TemporalConsistencyMetrics,
    UnsupportedSchemaError,
    compute_box_iou,
    main,
)


class TestDetectionMetrics(unittest.TestCase):
    """Automated test suite covering unit and integration testing for DetectionMetrics."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create sandbox directory and locate real project assets."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        cls.real_detection_file = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / "driving_sample"
            / "detections"
            / "batch_runs"
            / "run_20261117_115348_e1789b"
            / "detection_results.json"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary test sandbox."""
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def setUp(self) -> None:
        """Initialize fresh DetectionMetrics engine before each test."""
        self.engine = DetectionMetrics(
            confidence_threshold=0.30,
            iou_threshold=0.50,
            project_root=PROJECT_ROOT,
        )

    def _create_mock_json(
        self,
        frame_results: list,
        schema_version: str = "1.0",
        run_id: str = "run_test_01",
        dataset_id: str = "driving_test",
    ) -> Path:
        """Helper to create synthetic detection_results.json."""
        payload = {
            "schema_version": schema_version,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "total_frames": len(frame_results),
            "frame_results": frame_results,
        }
        json_path = self.temp_path / f"{run_id}_{len(frame_results)}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return json_path

    # ==========================================================================
    # 1. Structural Validation & Schema Parsing
    # ==========================================================================

    def test_01_valid_detection_input(self) -> None:
        """Verify successful loading and validation of valid detection JSON."""
        mock_file = self._create_mock_json([
            {
                "image_filename": "frame_000001.jpg",
                "extracted_frame_id": 1,
                "processing_status": "success",
                "detection_count": 0,
                "detections": [],
            }
        ])
        data = self.engine.load_and_validate_input(mock_file)
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["dataset_id"], "driving_test")
        self.assertEqual(len(data["frame_results"]), 1)

    def test_02_empty_detection_input(self) -> None:
        """Verify handling of detection file with empty frame list."""
        mock_file = self._create_mock_json([])
        summary, out_root = self.engine.analyze_detections(mock_file, overwrite=True)
        self.assertEqual(summary.total_frames_processed, 0)
        self.assertEqual(summary.total_detections, 0)
        self.assertEqual(summary.final_status, "PASS")

    def test_03_zero_detection_frames(self) -> None:
        """Verify that zero-detection frames are accounted correctly as successful."""
        mock_file = self._create_mock_json([
            {"image_filename": "frame_01.jpg", "processing_status": "success", "detection_count": 0, "detections": []},
            {"image_filename": "frame_02.jpg", "processing_status": "success", "detection_count": 0, "detections": []},
        ])
        summary, _ = self.engine.analyze_detections(mock_file, overwrite=True)
        self.assertEqual(summary.frames_without_detections, 2)
        self.assertEqual(summary.frames_with_detections, 0)
        self.assertEqual(summary.successful_frames, 2)
        self.assertEqual(summary.failed_frames, 0)

    # ==========================================================================
    # 2. Confidence Statistics & Distribution
    # ==========================================================================

    def test_04_confidence_statistics(self) -> None:
        """Verify mean, median, min, max, std dev, and low-conf percentage."""
        frames = [
            {
                "image_filename": "f1.jpg",
                "processing_status": "success",
                "detection_count": 3,
                "detections": [
                    {"class_id": 2, "class_name": "car", "confidence": 0.20, "bbox_xyxy": [1, 2, 3, 4]},
                    {"class_id": 2, "class_name": "car", "confidence": 0.50, "bbox_xyxy": [1, 2, 3, 4]},
                    {"class_id": 7, "class_name": "truck", "confidence": 0.80, "bbox_xyxy": [1, 2, 3, 4]},
                ],
            }
        ]
        conf_stats = self.engine.compute_confidence_analysis(frames)
        self.assertAlmostEqual(conf_stats["mean_confidence"], 0.50, places=3)
        self.assertAlmostEqual(conf_stats["median_confidence"], 0.50, places=3)
        self.assertEqual(conf_stats["min_confidence"], 0.20)
        self.assertEqual(conf_stats["max_confidence"], 0.80)
        self.assertGreater(conf_stats["confidence_std_dev"], 0.0)
        self.assertEqual(conf_stats["low_confidence_count"], 1)  # 0.20 < 0.30
        self.assertAlmostEqual(conf_stats["low_confidence_percentage"], 33.33, places=1)

    def test_05_class_wise_statistics(self) -> None:
        """Verify class-wise metrics aggregation and percentages."""
        frames = [
            {
                "processing_status": "success",
                "detections": [
                    {"class_id": 9, "class_name": "traffic light", "confidence": 0.40},
                    {"class_id": 9, "class_name": "traffic light", "confidence": 0.60},
                    {"class_id": 2, "class_name": "car", "confidence": 0.90},
                ],
            }
        ]
        classes = self.engine.compute_class_metrics(frames)
        self.assertIn("traffic light", classes)
        self.assertIn("car", classes)
        self.assertEqual(classes["traffic light"].detection_count, 2)
        self.assertAlmostEqual(classes["traffic light"].mean_confidence, 0.50, places=3)
        self.assertAlmostEqual(classes["traffic light"].percentage_of_total_detections, 66.67, places=1)
        self.assertEqual(classes["car"].detection_count, 1)

    def test_06_frame_wise_statistics(self) -> None:
        """Verify per-frame metric calculations and field preservation."""
        frames = [
            {
                "image_filename": "frame_000001.jpg",
                "extracted_frame_id": 1,
                "source_frame_index": 0,
                "timestamp_seconds": 0.0,
                "processing_status": "success",
                "inference_time_ms": 45.2,
                "detection_count": 2,
                "detections": [
                    {"class_id": 2, "confidence": 0.60},
                    {"class_id": 2, "confidence": 0.80},
                ],
            }
        ]
        records, _ = self.engine.compute_frame_and_temporal_metrics(frames)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r.frame_identifier, "frame_000001.jpg")
        self.assertEqual(r.extracted_frame_id, 1)
        self.assertEqual(r.timestamp_seconds, 0.0)
        self.assertAlmostEqual(r.mean_confidence, 0.70, places=3)
        self.assertEqual(r.max_confidence, 0.80)
        self.assertEqual(r.min_confidence, 0.60)
        self.assertEqual(r.processing_duration_ms, 45.2)

    # ==========================================================================
    # 3. Temporal Consistency
    # ==========================================================================

    def test_07_temporal_variation(self) -> None:
        """Verify frame-to-frame delta counts and temporal stability score."""
        frames = [
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 2, "detections": [{"confidence": 0.5}, {"confidence": 0.5}]},
            {"image_filename": "f2.jpg", "processing_status": "success", "detection_count": 4, "detections": [{"confidence": 0.6}] * 4},
            {"image_filename": "f3.jpg", "processing_status": "success", "detection_count": 1, "detections": [{"confidence": 0.7}]},
        ]
        records, temp_metrics = self.engine.compute_frame_and_temporal_metrics(frames)
        # diffs: |4 - 2| = 2, |1 - 4| = 3. Mean count var = (2 + 3) / 2 = 2.5
        self.assertEqual(temp_metrics.mean_detection_count_variation, 2.5)
        self.assertEqual(temp_metrics.max_detection_count_variation, 3)
        self.assertGreaterEqual(temp_metrics.stability_score, 0.0)
        self.assertLessEqual(temp_metrics.stability_score, 1.0)

    def test_08_missing_timestamps(self) -> None:
        """Verify that absent timestamps do not crash the engine and are recorded as None."""
        frames = [
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 0, "detections": []}
        ]
        records, _ = self.engine.compute_frame_and_temporal_metrics(frames)
        self.assertIsNone(records[0].timestamp_seconds)

    # ==========================================================================
    # 4. Data Traps & Error Handling
    # ==========================================================================

    def test_09_invalid_confidence_values(self) -> None:
        """Verify that non-numeric, negative, or >1.0 confidences are filtered out."""
        frames = [
            {
                "processing_status": "success",
                "detection_count": 3,
                "detections": [
                    {"confidence": -0.5},
                    {"confidence": 1.5},
                    {"confidence": "not_a_number"},
                    {"confidence": 0.75},
                ],
            }
        ]
        conf_stats = self.engine.compute_confidence_analysis(frames)
        self.assertEqual(conf_stats["min_confidence"], 0.75)
        self.assertEqual(conf_stats["max_confidence"], 0.75)

    def test_10_invalid_bounding_boxes(self) -> None:
        """Verify that malformed bounding boxes do not crash IoU calculation."""
        self.assertEqual(compute_box_iou([10, 20], [10, 20, 30, 40]), 0.0)
        self.assertEqual(compute_box_iou([10, 20, 30, 40], []), 0.0)

    def test_11_missing_required_fields(self) -> None:
        """Verify that missing top-level schema keys raise InvalidDetectionDataError."""
        bad_file = self.temp_path / "missing_fields.json"
        bad_file.write_text(json.dumps({"schema_version": "1.0", "run_id": "r1"}), encoding="utf-8")
        with self.assertRaises(InvalidDetectionDataError) as ctx:
            self.engine.load_and_validate_input(bad_file)
        self.assertIn("Missing required field", str(ctx.exception))

    def test_12_duplicate_frame_ids(self) -> None:
        """Verify that duplicate frame entries are handled deterministically."""
        frames = [
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 1, "detections": [{"confidence": 0.5}]},
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 1, "detections": [{"confidence": 0.5}]},
        ]
        stats = self.engine.compute_detection_statistics(frames)
        self.assertEqual(stats["total_frames_processed"], 2)
        self.assertEqual(stats["total_detections"], 2)

    def test_13_failed_frames(self) -> None:
        """Verify that failed frame records are counted in failed_frames."""
        frames = [
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 1, "detections": [{"confidence": 0.5}]},
            {"image_filename": "f2.jpg", "processing_status": "failed", "detection_count": 0, "detections": []},
        ]
        stats = self.engine.compute_detection_statistics(frames)
        self.assertEqual(stats["successful_frames"], 1)
        self.assertEqual(stats["failed_frames"], 1)

    # ==========================================================================
    # 5. Ground-Truth Evaluation
    # ==========================================================================

    def test_14_missing_ground_truth(self) -> None:
        """Verify that missing ground-truth annotations return status 'unavailable'."""
        gt_res = self.engine.evaluate_ground_truth([], ground_truth_file=None)
        self.assertEqual(gt_res.status, "unavailable")
        self.assertIsNone(gt_res.precision)
        self.assertIsNone(gt_res.recall)
        self.assertIsNone(gt_res.f1_score)
        self.assertIn("No ground-truth annotations", gt_res.disclaimer)

    def test_15_valid_ground_truth_matching(self) -> None:
        """Verify greedy bipartite matching against synthetic ground-truth annotations."""
        gt_payload = {
            "frames": [
                {
                    "image_filename": "f1.jpg",
                    "annotations": [
                        {"class_name": "car", "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
                        {"class_name": "truck", "bbox_xyxy": [100.0, 100.0, 200.0, 200.0]},
                    ],
                }
            ]
        }
        gt_path = self.temp_path / "mock_gt.json"
        gt_path.write_text(json.dumps(gt_payload), encoding="utf-8")

        frames = [
            {
                "image_filename": "f1.jpg",
                "detections": [
                    {"class_name": "car", "confidence": 0.90, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},  # exact TP
                    {"class_name": "traffic light", "confidence": 0.80, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},  # FP
                ],
            }
        ]
        gt_res = self.engine.evaluate_ground_truth(frames, ground_truth_file=gt_path)
        self.assertEqual(gt_res.status, "available")
        self.assertEqual(gt_res.true_positives, 1)   # car matched
        self.assertEqual(gt_res.false_positives, 1)  # traffic light
        self.assertEqual(gt_res.false_negatives, 1)  # truck missed
        self.assertAlmostEqual(gt_res.precision, 0.50, places=2)
        self.assertAlmostEqual(gt_res.recall, 0.50, places=2)
        self.assertAlmostEqual(gt_res.f1_score, 0.50, places=2)

    def test_16_iou_calculation(self) -> None:
        """Verify mathematical accuracy of box IoU calculation."""
        # Perfect overlap
        self.assertEqual(compute_box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        # Disjoint boxes
        self.assertEqual(compute_box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        # Partial overlap (half overlap in x, full in y)
        # Box A: [0, 0, 20, 10] (area 200), Box B: [10, 0, 30, 10] (area 200), Inter: [10, 0, 20, 10] (area 100)
        # Union = 200 + 200 - 100 = 300, IoU = 100/300 = 0.3333
        self.assertAlmostEqual(compute_box_iou([0, 0, 20, 10], [10, 0, 30, 10]), 1.0 / 3.0, places=4)

    def test_17_precision_and_recall(self) -> None:
        """Verify precision and recall edge cases."""
        # 100% precision, 50% recall
        gt_payload = {
            "frames": [
                {
                    "image_filename": "f.jpg",
                    "annotations": [
                        {"class_name": "car", "bbox_xyxy": [0, 0, 10, 10]},
                        {"class_name": "car", "bbox_xyxy": [20, 20, 30, 30]},
                    ],
                }
            ]
        }
        gt_path = self.temp_path / "mock_prec_rec_gt.json"
        gt_path.write_text(json.dumps(gt_payload), encoding="utf-8")

        frames = [
            {
                "image_filename": "f.jpg",
                "detections": [
                    {"class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
                ],
            }
        ]
        res = self.engine.evaluate_ground_truth(frames, ground_truth_file=gt_path)
        self.assertEqual(res.precision, 1.0)
        self.assertEqual(res.recall, 0.5)

    def test_18_f1_score(self) -> None:
        """Verify F1 harmonic mean calculation."""
        # Precision 1.0, Recall 0.5 -> F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 0.6667
        gt_payload = {
            "frames": [
                {"image_filename": "f.jpg", "annotations": [{"class_name": "c", "bbox_xyxy": [0, 0, 10, 10]}, {"class_name": "c", "bbox_xyxy": [20, 20, 30, 30]}]}
            ]
        }
        gt_path = self.temp_path / "mock_f1_gt.json"
        gt_path.write_text(json.dumps(gt_payload), encoding="utf-8")
        frames = [{"image_filename": "f.jpg", "detections": [{"class_name": "c", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]}]
        res = self.engine.evaluate_ground_truth(frames, ground_truth_file=gt_path)
        self.assertAlmostEqual(res.f1_score, 0.6667, places=4)

    # ==========================================================================
    # 6. Artifact Serialization & Multi-File Architecture
    # ==========================================================================

    def test_19_output_json_validation(self) -> None:
        """Verify that all 5 analysis JSON artifacts are created with Schema v1.0."""
        out_dir = self.temp_path / "artifacts_test"
        mock_file = self._create_mock_json([
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 1, "detections": [{"confidence": 0.6, "class_name": "car"}]}
        ])
        summary, out_root = self.engine.analyze_detections(mock_file, output_dir=out_dir, overwrite=True)

        expected_files = [
            "detection_metrics.json",
            "class_metrics.json",
            "frame_metrics.json",
            "analysis_summary.json",
            "analysis_metadata.json",
        ]
        for fname in expected_files:
            fpath = out_root / fname
            self.assertTrue(fpath.is_file(), f"Missing expected output artifact: {fname}")
            data = json.loads(fpath.read_text(encoding="utf-8"))
            self.assertEqual(data.get("schema_version"), "1.0")

    def test_20_schema_versioning(self) -> None:
        """Verify that unsupported schema versions raise UnsupportedSchemaError."""
        bad_file = self._create_mock_json([], schema_version="2.0", run_id="v2_run")
        with self.assertRaises(UnsupportedSchemaError) as ctx:
            self.engine.load_and_validate_input(bad_file)
        self.assertIn("Unsupported schema version '2.0'", str(ctx.exception))

    # ==========================================================================
    # 7. CLI & Operational Execution
    # ==========================================================================

    def test_21_cli_execution(self) -> None:
        """Verify CLI runner entrypoint execution."""
        mock_file = self._create_mock_json([
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 0, "detections": []}
        ])
        out_dir = self.temp_path / "cli_out"
        exit_code = main(["--input", str(mock_file), "--output-dir", str(out_dir), "--overwrite"])
        self.assertEqual(exit_code, 0)

    def test_22_partial_analysis(self) -> None:
        """Verify that max_frames results in status WARNING."""
        mock_file = self._create_mock_json([
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 0, "detections": []},
            {"image_filename": "f2.jpg", "processing_status": "success", "detection_count": 0, "detections": []},
        ])
        summary, _ = self.engine.analyze_detections(mock_file, max_frames=1, overwrite=True)
        self.assertEqual(summary.total_frames_processed, 1)
        self.assertEqual(summary.final_status, "WARNING")

    def test_23_error_reporting(self) -> None:
        """Verify that non-existent input file causes CLI to return code 1."""
        missing = self.temp_path / "does_not_exist_file.json"
        exit_code = main(["--input", str(missing)])
        self.assertEqual(exit_code, 1)

    def test_24_deterministic_output(self) -> None:
        """Verify that identical input data produces identical metric values."""
        frames = [
            {"image_filename": "f1.jpg", "processing_status": "success", "detection_count": 2, "detections": [{"confidence": 0.4}, {"confidence": 0.8}]},
        ]
        s1 = self.engine.compute_confidence_analysis(frames)
        s2 = self.engine.compute_confidence_analysis(frames)
        self.assertEqual(s1, s2)

    def test_25_empty_dataset(self) -> None:
        """Verify zero-handling when dataset contains 0 frames."""
        stats = self.engine.compute_detection_statistics([])
        self.assertEqual(stats["total_frames_processed"], 0)
        self.assertEqual(stats["average_detections_per_frame"], 0.0)

    def test_26_confidence_threshold_behavior(self) -> None:
        """Verify that varying confidence threshold alters low_confidence_count."""
        frames = [
            {"processing_status": "success", "detections": [{"confidence": 0.35}]}
        ]
        e_low = DetectionMetrics(confidence_threshold=0.30)
        s_low = e_low.compute_confidence_analysis(frames)
        self.assertEqual(s_low["low_confidence_count"], 0)  # 0.35 >= 0.30

        e_high = DetectionMetrics(confidence_threshold=0.40)
        s_high = e_high.compute_confidence_analysis(frames)
        self.assertEqual(s_high["low_confidence_count"], 1)  # 0.35 < 0.40

    def test_27_class_filtering(self) -> None:
        """Verify distinct class separation across multiple semantic classes."""
        frames = [
            {
                "processing_status": "success",
                "detections": [
                    {"class_id": 0, "class_name": "person", "confidence": 0.8},
                    {"class_id": 2, "class_name": "car", "confidence": 0.9},
                    {"class_id": 5, "class_name": "bus", "confidence": 0.7},
                ],
            }
        ]
        classes = self.engine.compute_class_metrics(frames)
        self.assertEqual(len(classes), 3)
        self.assertIn("person", classes)
        self.assertIn("car", classes)
        self.assertIn("bus", classes)

    def test_28_invalid_json_syntax(self) -> None:
        """Verify that corrupted JSON text raises InvalidDetectionDataError."""
        bad_json = self.temp_path / "bad_syntax.json"
        bad_json.write_text("{ unquoted_key: val, ", encoding="utf-8")
        with self.assertRaises(InvalidDetectionDataError):
            self.engine.load_and_validate_input(bad_json)

    def test_29_unsupported_schema(self) -> None:
        """Verify that unsupported schema raises UnsupportedSchemaError."""
        bad_schema = self._create_mock_json([], schema_version="9.9")
        with self.assertRaises(UnsupportedSchemaError):
            self.engine.load_and_validate_input(bad_schema)

    # ==========================================================================
    # 8. Real Artifact Integration Test
    # ==========================================================================

    def test_30_real_task4_artifact_integration(self) -> None:
        """Integration test on genuine Task 4 detection output (run_20261117_115348_e1789b)."""
        if not self.real_detection_file.is_file():
            self.skipTest(f"Real detection results not found at: {self.real_detection_file}")

        out_dir = self.temp_path / "real_task4_integration"
        summary, out_root = self.engine.analyze_detections(
            detection_file=self.real_detection_file,
            output_dir=out_dir,
            overwrite=True,
        )
        # Real task 4 dataset has exactly 200 frames, 211 detections, 166 with det, 34 without det
        self.assertEqual(summary.total_frames_processed, 200)
        self.assertEqual(summary.successful_frames, 200)
        self.assertEqual(summary.failed_frames, 0)
        self.assertEqual(summary.total_detections, 211)
        self.assertEqual(summary.frames_with_detections, 166)
        self.assertEqual(summary.frames_without_detections, 34)
        self.assertEqual(summary.final_status, "PASS")
        self.assertEqual(summary.ground_truth_status, "unavailable")
        self.assertTrue((out_root / "detection_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)

