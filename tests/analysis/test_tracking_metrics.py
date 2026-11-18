"""Automated unit and integration test suite for PerceptGuard AI Tracking Metrics Engine.

Validates:
- TrackingMetricsAnalyzer initialization and parameter boundaries.
- Multi-schema tracking ingestion (MOT, SOT, ByteTrack).
- Ingestion error handling (corrupted JSON, missing files, unsupported schema).
- Mathematical correctness of track-level metrics (duration, observations, ratios, gaps, conf stats).
- Kinematic calculations (cumulative displacement, mean displacement, velocity).
- Frame-level metrics (detection counts, active tracks, new/lost transitions, empty frames).
- Class-level metrics breakdown.
- Ground-Truth Rule compliance: MOTA, IDF1, HOTA = NOT AVAILABLE when GT is absent.
- Multi-file artifact persistence and overwrite safety.
- CLI execution and error codes.
- Real-data integration with actual driving sample tracking outputs.
"""

import json
import math
from pathlib import Path
import tempfile
import unittest

from src.analysis.tracking_metrics import (
    ClassTrackingMetrics,
    FrameTrackingMetrics,
    GroundTruthEvaluation,
    InvalidTrackingDataError,
    TrackObservationMetrics,
    TrackingAnalysisSummary,
    TrackingMetricsAnalyzer,
    TrackingMetricsError,
    TrackingMetricsPersistenceError,
    UnsupportedTrackingSchemaError,
)


class TestTrackingMetricsAnalyzer(unittest.TestCase):
    """Test suite for TrackingMetricsAnalyzer."""

    def setUp(self) -> None:
        self.analyzer = TrackingMetricsAnalyzer(min_confidence_threshold=0.1, gap_tolerance_frames=2)

    def _create_mock_tracking_dict(self) -> dict:
        """Helper to create a standard, valid tracking output dictionary."""
        return {
            "schema_version": "1.0",
            "run_id": "mot_test_run_001",
            "dataset_id": "test_dataset",
            "tracker_type": "bytetrack",
            "frame_results": [
                {
                    "frame_id": "frame_000001.jpg",
                    "source_frame_index": 1,
                    "timestamp_seconds": 0.0,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.80,
                            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
                        },
                        {
                            "track_id": 2,
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": 0.60,
                            "bbox_xyxy": [100.0, 100.0, 130.0, 180.0],
                        },
                    ],
                },
                {
                    "frame_id": "frame_000002.jpg",
                    "source_frame_index": 2,
                    "timestamp_seconds": 0.1,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.90,
                            "bbox_xyxy": [14.0, 13.0, 54.0, 53.0],  # dx=4, dy=3 => dist=5.0
                        },
                    ],
                },
                {
                    "frame_id": "frame_000003.jpg",
                    "source_frame_index": 3,
                    "timestamp_seconds": 0.2,
                    "tracks": [],  # Empty frame for Track 1 and Track 2
                },
                {
                    "frame_id": "frame_000004.jpg",
                    "source_frame_index": 4,
                    "timestamp_seconds": 0.3,
                    "tracks": [],  # Second empty frame (gap=2 frames for Track 1)
                },
                {
                    "frame_id": "frame_000005.jpg",
                    "source_frame_index": 5,
                    "timestamp_seconds": 0.4,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.70,
                            "bbox_xyxy": [14.0, 13.0, 54.0, 53.0],  # Reacquired Track 1
                        },
                    ],
                },
            ],
        }

    # --------------------------------------------------------------------------
    # Initialization & Parameter Tests
    # --------------------------------------------------------------------------

    def test_01_analyzer_initialization_defaults(self) -> None:
        analyzer = TrackingMetricsAnalyzer()
        self.assertEqual(analyzer.min_confidence_threshold, 0.1)
        self.assertEqual(analyzer.gap_tolerance_frames, 2)
        self.assertEqual(len(analyzer.errors), 0)

    def test_02_invalid_confidence_threshold_rejection(self) -> None:
        with self.assertRaises(TrackingMetricsError):
            TrackingMetricsAnalyzer(min_confidence_threshold=-0.1)
        with self.assertRaises(TrackingMetricsError):
            TrackingMetricsAnalyzer(min_confidence_threshold=1.5)

    def test_03_invalid_gap_tolerance_rejection(self) -> None:
        with self.assertRaises(TrackingMetricsError):
            TrackingMetricsAnalyzer(gap_tolerance_frames=0)
        with self.assertRaises(TrackingMetricsError):
            TrackingMetricsAnalyzer(gap_tolerance_frames=-2)

    # --------------------------------------------------------------------------
    # Schema Ingestion & Error Handling
    # --------------------------------------------------------------------------

    def test_04_missing_tracking_file_raises_error(self) -> None:
        with self.assertRaises(InvalidTrackingDataError):
            self.analyzer.analyze(tracking_source="non_existent_file_path.json")

    def test_05_malformed_json_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{ invalid json [")
            p = Path(f.name)
        try:
            with self.assertRaises(InvalidTrackingDataError):
                self.analyzer.analyze(tracking_source=p)
        finally:
            if p.exists():
                p.unlink()

    def test_06_unsupported_schema_version_raises_error(self) -> None:
        bad_schema = {"schema_version": "99.0", "frame_results": []}
        with self.assertRaises(UnsupportedTrackingSchemaError):
            self.analyzer.analyze(tracking_source=bad_schema)

    def test_07_invalid_frames_container_raises_error(self) -> None:
        bad_data = {"schema_version": "1.0", "frame_results": "not_a_list"}
        with self.assertRaises(InvalidTrackingDataError):
            self.analyzer.analyze(tracking_source=bad_data)

    # --------------------------------------------------------------------------
    # Track-Level Metrics Tests
    # --------------------------------------------------------------------------

    def test_08_track_level_observation_count_and_span(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        # Track 1 observed in frames 1, 2, 5 => total 3 observations, span = 5 - 1 + 1 = 5
        t1 = next(t for t in track_metrics if t.track_id == 1)
        self.assertEqual(t1.total_observed_frames, 3)
        self.assertEqual(t1.frame_span, 5)
        self.assertEqual(t1.observation_ratio, 0.6)  # 3 / 5 = 0.6

        # Track 2 observed only in frame 1 => total 1 observation, span = 1
        t2 = next(t for t in track_metrics if t.track_id == 2)
        self.assertEqual(t2.total_observed_frames, 1)
        self.assertEqual(t2.frame_span, 1)
        self.assertEqual(t2.observation_ratio, 1.0)

    def test_09_track_gap_analysis(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        t1 = next(t for t in track_metrics if t.track_id == 1)
        # Observed at index 1, 2, 5. Gaps: index 2 to 5 has 2 missing frames (3 and 4)
        self.assertEqual(t1.observation_gaps_count, 1)
        self.assertEqual(t1.max_gap_frames, 2)
        self.assertEqual(t1.mean_gap_frames, 2.0)
        self.assertEqual(t1.reacquisition_count, 1)  # Gap >= tolerance (2)

    def test_10_track_confidence_statistics(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        t1 = next(t for t in track_metrics if t.track_id == 1)
        # Confs for Track 1: [0.80, 0.90, 0.70] => mean=0.80, min=0.70, max=0.90, std=0.10
        self.assertAlmostEqual(t1.mean_confidence, 0.80, places=4)
        self.assertAlmostEqual(t1.min_confidence, 0.70, places=4)
        self.assertAlmostEqual(t1.max_confidence, 0.90, places=4)
        self.assertAlmostEqual(t1.confidence_std_dev, 0.10, places=2)

    def test_11_track_geometry_metrics(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        t1 = next(t for t in track_metrics if t.track_id == 1)
        # Box width: 50-10=40, 54-14=40, 54-14=40 => mean width=40.0
        # Box height: 50-10=40, 53-13=40, 53-13=40 => mean height=40.0
        # Area: 1600.0
        self.assertEqual(t1.mean_bbox_width, 40.0)
        self.assertEqual(t1.mean_bbox_height, 40.0)
        self.assertEqual(t1.mean_bbox_area, 1600.0)

    def test_12_track_kinematics_displacement_and_velocity(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        t1 = next(t for t in track_metrics if t.track_id == 1)
        # Center 1: (30.0, 30.0), Center 2: (34.0, 33.0) => dist = 5.0
        # Center 3: (34.0, 33.0) => dist = 0.0
        # Cumulative displacement = 5.0
        self.assertEqual(t1.cumulative_displacement_pixels, 5.0)
        self.assertEqual(t1.mean_displacement_pixels, 2.5)  # 5.0 / 2 steps = 2.5
        # Total duration = 0.4s - 0.0s = 0.4s => velocity = 5.0 / 0.4 = 12.5 px/s
        self.assertEqual(t1.mean_velocity_pixels_per_second, 12.5)

    # --------------------------------------------------------------------------
    # Frame-Level Metrics Tests
    # --------------------------------------------------------------------------

    def test_13_frame_level_active_track_counts(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        self.assertEqual(len(frame_metrics), 5)
        self.assertEqual(frame_metrics[0].active_track_count, 2)
        self.assertEqual(frame_metrics[0].active_track_ids, [1, 2])
        self.assertEqual(frame_metrics[1].active_track_count, 1)
        self.assertEqual(frame_metrics[1].active_track_ids, [1])
        self.assertEqual(frame_metrics[2].active_track_count, 0)
        self.assertTrue(frame_metrics[2].is_empty_frame)
        self.assertEqual(frame_metrics[3].active_track_count, 0)
        self.assertTrue(frame_metrics[3].is_empty_frame)
        self.assertEqual(frame_metrics[4].active_track_count, 1)

    def test_14_frame_level_new_and_lost_transitions(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        # Frame 1: new tracks = [1, 2] (2), lost = 0
        self.assertEqual(frame_metrics[0].new_tracks_count, 2)
        self.assertEqual(frame_metrics[0].lost_tracks_count, 0)

        # Frame 2: active = [1], lost = [2] (1), new = 0
        self.assertEqual(frame_metrics[1].new_tracks_count, 0)
        self.assertEqual(frame_metrics[1].lost_tracks_count, 1)

        # Frame 3: active = [], lost = [1] (1), new = 0
        self.assertEqual(frame_metrics[2].new_tracks_count, 0)
        self.assertEqual(frame_metrics[2].lost_tracks_count, 1)

    # --------------------------------------------------------------------------
    # Class-Level Metrics Tests
    # --------------------------------------------------------------------------

    def test_15_class_level_breakdown(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        self.assertEqual(len(class_metrics), 2)
        car_metric = next(c for c in class_metrics if c.class_name == "car")
        person_metric = next(c for c in class_metrics if c.class_name == "person")

        self.assertEqual(car_metric.unique_track_count, 1)
        self.assertEqual(car_metric.total_observations, 3)
        self.assertAlmostEqual(car_metric.mean_confidence, 0.80, places=4)

        self.assertEqual(person_metric.unique_track_count, 1)
        self.assertEqual(person_metric.total_observations, 1)
        self.assertAlmostEqual(person_metric.mean_confidence, 0.60, places=4)

    # --------------------------------------------------------------------------
    # Summary Statistics & Aggregations
    # --------------------------------------------------------------------------

    def test_16_summary_metrics_aggregation(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        self.assertEqual(summary.total_frames_analyzed, 5)
        self.assertEqual(summary.frames_with_detections, 3)
        self.assertEqual(summary.frames_without_detections, 2)
        self.assertEqual(summary.total_track_observations, 4)  # 2 in f1, 1 in f2, 1 in f5
        self.assertEqual(summary.total_unique_tracks, 2)
        self.assertEqual(summary.max_simultaneous_active_tracks, 2)
        self.assertEqual(summary.average_active_tracks_per_frame, 0.8)  # 4 / 5 = 0.8
        self.assertEqual(summary.status, "PASS")

    def test_17_duration_distribution_percentiles(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        dist = summary.track_duration_distribution
        # Track 1 has 3 frames, Track 2 has 1 frame
        self.assertEqual(dist["min_frames"], 1)
        self.assertEqual(dist["max_frames"], 3)
        self.assertEqual(dist["mean_frames"], 2.0)
        self.assertEqual(dist["median_frames"], 2.0)

    # --------------------------------------------------------------------------
    # Ground-Truth Rule Compliance Tests
    # --------------------------------------------------------------------------

    def test_18_ground_truth_rule_when_annotations_absent(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        gt = summary.ground_truth_evaluation
        self.assertFalse(gt["ground_truth_available"])
        self.assertIsNone(gt["mota"])
        self.assertIsNone(gt["motp"])
        self.assertIsNone(gt["idf1"])
        self.assertIsNone(gt["hota"])
        self.assertIsNone(gt["confirmed_id_switches"])
        self.assertEqual(gt["status"], "NOT AVAILABLE")

    def test_19_observation_ratio_not_conflated_with_accuracy(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        # Ensure summary does NOT contain fake accuracy keys
        sum_dict = summary.to_dict()
        self.assertNotIn("tracking_accuracy", sum_dict)
        self.assertNotIn("mota_accuracy", sum_dict)

    # --------------------------------------------------------------------------
    # Edge Cases & Fault Tolerance
    # --------------------------------------------------------------------------

    def test_20_zero_detection_sequence(self) -> None:
        empty_data = {
            "schema_version": "1.0",
            "run_id": "empty_run",
            "dataset_id": "empty_set",
            "frame_results": [
                {"frame_id": "f1.jpg", "source_frame_index": 1, "tracks": []},
                {"frame_id": "f2.jpg", "source_frame_index": 2, "tracks": []},
            ],
        }
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(empty_data)

        self.assertEqual(summary.total_frames_analyzed, 2)
        self.assertEqual(summary.frames_with_detections, 0)
        self.assertEqual(summary.frames_without_detections, 2)
        self.assertEqual(summary.total_unique_tracks, 0)
        self.assertEqual(len(track_metrics), 0)
        self.assertEqual(summary.status, "PASS")

    def test_21_invalid_bounding_box_handled_safely(self) -> None:
        data = {
            "schema_version": "1.0",
            "frame_results": [
                {
                    "frame_id": "f1.jpg",
                    "source_frame_index": 1,
                    "tracks": [
                        {"track_id": 1, "bbox_xyxy": [50.0, 50.0, 10.0, 10.0]},  # Inverted box
                    ],
                }
            ],
        }
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)
        # Should record error without crashing
        self.assertGreater(len(self.analyzer.errors), 0)
        self.assertEqual(summary.total_unique_tracks, 0)

    def test_22_non_finite_coordinates_handled_safely(self) -> None:
        data = {
            "schema_version": "1.0",
            "frame_results": [
                {
                    "frame_id": "f1.jpg",
                    "source_frame_index": 1,
                    "tracks": [
                        {"track_id": 1, "bbox_xyxy": [float("nan"), 10.0, 50.0, 50.0]},
                    ],
                }
            ],
        }
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)
        self.assertGreater(len(self.analyzer.errors), 0)

    def test_23_out_of_order_frame_indices_sorted(self) -> None:
        data = {
            "schema_version": "1.0",
            "frame_results": [
                {"frame_id": "f2.jpg", "source_frame_index": 2, "tracks": [{"track_id": 1, "bbox_xyxy": [1, 1, 10, 10]}]},
                {"frame_id": "f1.jpg", "source_frame_index": 1, "tracks": [{"track_id": 1, "bbox_xyxy": [1, 1, 10, 10]}]},
            ],
        }
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)
        # Frames should be sorted by source_frame_index
        self.assertEqual(frame_metrics[0].source_frame_index, 1)
        self.assertEqual(frame_metrics[1].source_frame_index, 2)

    def test_24_missing_timestamps_handled_gracefully(self) -> None:
        data = {
            "schema_version": "1.0",
            "frame_results": [
                {"frame_id": "f1.jpg", "source_frame_index": 1, "tracks": [{"track_id": 1, "bbox_xyxy": [1, 1, 10, 10]}]},
                {"frame_id": "f2.jpg", "source_frame_index": 2, "tracks": [{"track_id": 1, "bbox_xyxy": [5, 5, 15, 15]}]},
            ],
        }
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)
        t1 = track_metrics[0]
        self.assertIsNone(t1.first_timestamp_seconds)
        self.assertIsNone(t1.mean_velocity_pixels_per_second)
        self.assertEqual(t1.cumulative_displacement_pixels, round(math.hypot(4.5, 4.5), 2))

    def test_25_max_frames_smoke_test_sets_warning_status(self) -> None:
        data = self._create_mock_tracking_dict()  # 5 frames total
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data, max_frames=2)
        self.assertEqual(summary.total_frames_analyzed, 2)
        self.assertEqual(summary.status, "WARNING")

    # --------------------------------------------------------------------------
    # Artifact Persistence Tests
    # --------------------------------------------------------------------------

    def test_26_save_analysis_artifacts(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = self.analyzer.save_analysis_artifacts(
                output_dir=tmpdir,
                summary=summary,
                track_metrics=track_metrics,
                frame_metrics=frame_metrics,
                class_metrics=class_metrics,
            )

            self.assertTrue((out_path / "tracking_analysis_summary.json").exists())
            self.assertTrue((out_path / "track_metrics.json").exists())
            self.assertTrue((out_path / "frame_metrics.json").exists())
            self.assertTrue((out_path / "class_metrics.json").exists())
            self.assertTrue((out_path / "metadata.json").exists())
            self.assertTrue((out_path / "errors.json").exists())

    def test_27_overwrite_protection(self) -> None:
        data = self._create_mock_tracking_dict()
        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.analyzer.save_analysis_artifacts(
                output_dir=tmpdir,
                summary=summary,
                track_metrics=track_metrics,
                frame_metrics=frame_metrics,
                class_metrics=class_metrics,
            )

            # Attempt saving without overwrite should raise error
            with self.assertRaises(TrackingMetricsPersistenceError):
                self.analyzer.save_analysis_artifacts(
                    output_dir=tmpdir,
                    summary=summary,
                    track_metrics=track_metrics,
                    frame_metrics=frame_metrics,
                    class_metrics=class_metrics,
                    overwrite=False,
                )

            # Saving with overwrite should succeed
            res = self.analyzer.save_analysis_artifacts(
                output_dir=tmpdir,
                summary=summary,
                track_metrics=track_metrics,
                frame_metrics=frame_metrics,
                class_metrics=class_metrics,
                overwrite=True,
            )
            self.assertTrue(res.exists())

    # --------------------------------------------------------------------------
    # Real Artifact Integration Tests
    # --------------------------------------------------------------------------

    def test_28_real_mot_artifact_ingestion(self) -> None:
        real_mot_file = Path("datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json")
        if not real_mot_file.exists():
            self.skipTest(f"Real MOT artifact {real_mot_file} not found on disk.")

        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(real_mot_file)

        self.assertEqual(summary.total_frames_analyzed, 200)
        self.assertEqual(summary.total_unique_tracks, 3)
        self.assertEqual(summary.total_track_observations, 193)
        self.assertEqual(len(track_metrics), 3)
        self.assertEqual(len(frame_metrics), 200)
        self.assertEqual(summary.status, "PASS")

    def test_29_real_single_object_artifact_ingestion(self) -> None:
        real_sot_file = Path("datasets/processed/driving_sample/tracking/single_object/single_target_1_20261117_154123_fa3f36/single_object_tracking.json")
        if not real_sot_file.exists():
            self.skipTest(f"Real SOT artifact {real_sot_file} not found on disk.")

        summary, track_metrics, frame_metrics, class_metrics = self.analyzer.analyze(real_sot_file)

        self.assertEqual(summary.total_frames_analyzed, 200)
        self.assertEqual(summary.total_unique_tracks, 1)
        self.assertEqual(len(track_metrics), 1)
        self.assertEqual(summary.status, "PASS")

    def test_30_cli_execution_with_real_artifact(self) -> None:
        from src.analysis.tracking_metrics import main

        real_mot_file = Path("datasets/processed/driving_sample/tracking/multi_object/mot_driving_sample_20261117_160903_1c98f4/multi_object_tracking.json")
        if not real_mot_file.exists():
            self.skipTest(f"Real MOT artifact {real_mot_file} not found.")

        with tempfile.TemporaryDirectory() as tmpdir:
            ret = main(["--tracking-file", str(real_mot_file), "--output-dir", tmpdir, "--max-frames", "20", "--overwrite"])
            self.assertEqual(ret, 0)
            self.assertTrue((Path(tmpdir) / "tracking_analysis_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
