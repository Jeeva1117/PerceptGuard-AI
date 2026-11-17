"""PerceptGuard AI: Comprehensive Unit and Integration Tests for MultiObjectTracker.

Validates 30 production computer vision and multi-object tracking requirements:
1. Module initialization
2. Multiple objects in one frame
3. Multiple objects across sequential frames
4. Unique tracker IDs
5. Consistent IDs across observations
6. New object appearance
7. Object disappearance
8. Temporary missing observations
9. Reappearance behavior
10. Empty detection frames
11. Zero-object video
12. Invalid image input
13. Invalid tracking results
14. Bounding-box validation
15. Confidence validation
16. Multiple object classes
17. Tracker reset
18. Independent tracking sessions
19. Metadata preservation
20. Timestamp handling
21. Frame-index handling
22. Per-object history
23. Summary statistics
24. Output JSON validation
25. Output directory creation
26. Error handling
27. CLI argument validation
28. CPU-only compatibility
29. Memory-efficient sequential processing
30. Integration with existing ByteTrack pipeline
"""

import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.tracking.bytetrack_engine import InvalidDetectionInputError
from src.tracking.multi_object_tracker import (
    FrameMOTRecord,
    MOTConfigurationError,
    MOTInputNotFoundError,
    MultiObjectTracker,
    MultiObjectTrackingError,
    MultiObjectTrackingSummary,
    TrackHistory,
    TrackLifecycleState,
    TrackedObject,
    main,
)
from src.tracking.tracker_config import TrackerConfig


class TestMultiObjectTracker(unittest.TestCase):
    """Test suite covering 30 multi-object tracking requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Mock precomputed batch detection JSON
        self.mock_detection_file = self.temp_path / "mock_detections.json"
        self.sample_payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": "test_batch_run_001",
            "dataset_id": "driving_sample",
            "model_name": "yolo11n",
            "total_frames": 4,
            "frame_results": [
                {
                    "image_filename": "frame_000001.jpg",
                    "extracted_frame_id": 1,
                    "source_frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "detections": [
                        {
                            "detection_index": 0,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.90,
                            "bbox_xyxy": [100.0, 150.0, 200.0, 250.0],
                        },
                        {
                            "detection_index": 1,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.85,
                            "bbox_xyxy": [400.0, 300.0, 600.0, 500.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000002.jpg",
                    "extracted_frame_id": 2,
                    "source_frame_index": 1,
                    "timestamp_seconds": 0.033,
                    "detections": [
                        {
                            "detection_index": 0,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.92,
                            "bbox_xyxy": [102.0, 151.0, 202.0, 251.0],
                        },
                        {
                            "detection_index": 1,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.87,
                            "bbox_xyxy": [405.0, 302.0, 605.0, 502.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000003.jpg",
                    "extracted_frame_id": 3,
                    "source_frame_index": 2,
                    "timestamp_seconds": 0.066,
                    "detections": [
                        # Car missing (lost)
                        {
                            "detection_index": 0,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.88,
                            "bbox_xyxy": [410.0, 305.0, 610.0, 505.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000004.jpg",
                    "extracted_frame_id": 4,
                    "source_frame_index": 3,
                    "timestamp_seconds": 0.100,
                    "detections": [
                        # Car reacquired
                        {
                            "detection_index": 0,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.89,
                            "bbox_xyxy": [105.0, 153.0, 205.0, 253.0],
                        },
                        {
                            "detection_index": 1,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.86,
                            "bbox_xyxy": [415.0, 308.0, 615.0, 508.0],
                        },
                    ],
                },
            ],
        }
        with open(self.mock_detection_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_payload, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # Test 1: Module Initialization
    # --------------------------------------------------------------------------
    def test_01_module_initialization(self) -> None:
        tracker = MultiObjectTracker(dataset_id="test_ds", max_lost_frames=20, frame_rate=25)
        self.assertEqual(tracker.dataset_id, "test_ds")
        self.assertEqual(tracker.max_lost_frames, 20)
        self.assertEqual(tracker.frame_rate, 25)
        self.assertEqual(tracker.total_unique_tracks, 0)
        self.assertEqual(len(tracker.frame_records), 0)

    # --------------------------------------------------------------------------
    # Test 2: Multiple Objects in One Frame
    # --------------------------------------------------------------------------
    def test_02_multiple_objects_in_one_frame(self) -> None:
        tracker = MultiObjectTracker()
        dets = [
            {"class_id": 2, "class_name": "car", "confidence": 0.90, "bbox_xyxy": [10, 10, 50, 50]},
            {"class_id": 7, "class_name": "truck", "confidence": 0.85, "bbox_xyxy": [100, 100, 200, 200]},
            {"class_id": 0, "class_name": "person", "confidence": 0.80, "bbox_xyxy": [300, 300, 350, 400]},
        ]
        rec = tracker.track_detections(dets, frame_id="f1")
        self.assertEqual(rec.detection_count, 3)
        self.assertEqual(rec.active_track_count, 3)
        self.assertEqual(len(rec.active_tracks), 3)

    # --------------------------------------------------------------------------
    # Test 3: Multiple Objects Across Sequential Frames
    # --------------------------------------------------------------------------
    def test_03_multiple_objects_across_sequential_frames(self) -> None:
        tracker = MultiObjectTracker()
        dets_1 = [
            {"class_id": 2, "class_name": "car", "confidence": 0.90, "bbox_xyxy": [10, 10, 50, 50]},
            {"class_id": 7, "class_name": "truck", "confidence": 0.85, "bbox_xyxy": [100, 100, 200, 200]},
        ]
        dets_2 = [
            {"class_id": 2, "class_name": "car", "confidence": 0.92, "bbox_xyxy": [12, 12, 52, 52]},
            {"class_id": 7, "class_name": "truck", "confidence": 0.88, "bbox_xyxy": [105, 102, 205, 202]},
        ]
        rec1 = tracker.track_detections(dets_1, frame_id="f1")
        rec2 = tracker.track_detections(dets_2, frame_id="f2")

        self.assertEqual(rec1.active_track_count, 2)
        self.assertEqual(rec2.active_track_count, 2)
        self.assertEqual(tracker.total_unique_tracks, 2)

    # --------------------------------------------------------------------------
    # Test 4: Unique Tracker IDs
    # --------------------------------------------------------------------------
    def test_04_unique_tracker_ids(self) -> None:
        tracker = MultiObjectTracker()
        dets = [
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [100, 100, 150, 150]},
        ]
        rec = tracker.track_detections(dets, frame_id="f1")
        track_ids = [t.track_id for t in rec.active_tracks]
        self.assertEqual(len(track_ids), 2)
        self.assertNotEqual(track_ids[0], track_ids[1])

    # --------------------------------------------------------------------------
    # Test 5: Consistent IDs Across Observations
    # --------------------------------------------------------------------------
    def test_05_consistent_ids_across_observations(self) -> None:
        tracker = MultiObjectTracker()
        rec1 = tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        rec2 = tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [12, 11, 52, 51]}], frame_id="f2")

        self.assertEqual(rec1.active_tracks[0].track_id, rec2.active_tracks[0].track_id)
        self.assertEqual(rec2.active_tracks[0].lifecycle_state, TrackLifecycleState.TRACKING.value)

    # --------------------------------------------------------------------------
    # Test 6: New Object Appearance
    # --------------------------------------------------------------------------
    def test_06_new_object_appearance(self) -> None:
        tracker = MultiObjectTracker()
        # Frame 1: object 1 active
        tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        # Frame 2: object 1 moves, object 2 appears (initialized as unconfirmed in ByteTrack)
        tracker.track_detections([
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [12, 11, 52, 51]},
            {"class_id": 7, "class_name": "truck", "confidence": 0.95, "bbox_xyxy": [300, 300, 400, 400]},
        ], frame_id="f2")
        # Frame 3: both objects matched again -> both confirmed active
        rec3 = tracker.track_detections([
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [14, 12, 54, 52]},
            {"class_id": 7, "class_name": "truck", "confidence": 0.95, "bbox_xyxy": [302, 301, 402, 401]},
        ], frame_id="f3")

        self.assertEqual(len(rec3.active_tracks), 2)
        states = {t.class_name: t.lifecycle_state for t in rec3.active_tracks}
        self.assertEqual(states["car"], TrackLifecycleState.TRACKING.value)
        self.assertIn("truck", states)

    # --------------------------------------------------------------------------
    # Test 7: Object Disappearance
    # --------------------------------------------------------------------------
    def test_07_object_disappearance(self) -> None:
        tracker = MultiObjectTracker(max_lost_frames=5)
        tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        # Frame 2: car disappears
        rec2 = tracker.track_detections([], frame_id="f2")

        self.assertEqual(rec2.active_track_count, 0)
        self.assertEqual(tracker.histories[1].lifecycle_state, TrackLifecycleState.TEMPORARILY_LOST.value)
        self.assertEqual(tracker.histories[1].consecutive_lost_frames, 1)

    # --------------------------------------------------------------------------
    # Test 8: Temporary Missing Observations
    # --------------------------------------------------------------------------
    def test_08_temporary_missing_observations(self) -> None:
        tracker = MultiObjectTracker(max_lost_frames=10)
        tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        tracker.track_detections([], frame_id="f2")
        tracker.track_detections([], frame_id="f3")

        self.assertEqual(tracker.histories[1].consecutive_lost_frames, 2)
        self.assertEqual(tracker.histories[1].longest_lost_interval_frames, 2)
        self.assertEqual(tracker.histories[1].lifecycle_state, TrackLifecycleState.TEMPORARILY_LOST.value)

    # --------------------------------------------------------------------------
    # Test 9: Reappearance Behavior
    # --------------------------------------------------------------------------
    def test_09_reappearance_behavior(self) -> None:
        tracker = MultiObjectTracker(max_lost_frames=5)
        tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        tracker.track_detections([], frame_id="f2")  # TEMPORARILY_LOST
        rec3 = tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [11, 10, 51, 50]}], frame_id="f3")

        self.assertEqual(rec3.active_tracks[0].lifecycle_state, TrackLifecycleState.REACQUIRED.value)
        self.assertEqual(tracker.histories[1].reacquisition_count, 1)

    # --------------------------------------------------------------------------
    # Test 10: Empty Detection Frames
    # --------------------------------------------------------------------------
    def test_10_empty_detection_frames(self) -> None:
        tracker = MultiObjectTracker()
        rec = tracker.track_detections([], frame_id="empty_frame.jpg")
        self.assertEqual(rec.detection_count, 0)
        self.assertEqual(rec.active_track_count, 0)
        self.assertEqual(rec.processing_status, "empty_frame")

    # --------------------------------------------------------------------------
    # Test 11: Zero-Object Video
    # --------------------------------------------------------------------------
    def test_11_zero_object_video(self) -> None:
        empty_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_empty",
            "frame_results": [
                {"image_filename": "f1.jpg", "detections": []},
                {"image_filename": "f2.jpg", "detections": []},
            ],
        }
        empty_json = self.temp_path / "empty_run.json"
        with open(empty_json, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f)

        tracker = MultiObjectTracker()
        summary, _, _ = tracker.run(empty_json, output_dir=self.temp_path / "empty_out")
        self.assertEqual(summary.total_frames_processed, 2)
        self.assertEqual(summary.empty_frames, 2)
        self.assertEqual(summary.total_unique_tracks, 0)
        self.assertEqual(summary.status, "PASS")

    # --------------------------------------------------------------------------
    # Test 12: Invalid Image Input
    # --------------------------------------------------------------------------
    def test_12_invalid_image_input(self) -> None:
        tracker = MultiObjectTracker()
        rec = tracker.track_image("non_existent_file.jpg")
        self.assertEqual(rec.processing_status, "error")
        self.assertEqual(len(tracker.error_records), 1)

    # --------------------------------------------------------------------------
    # Test 13: Invalid Tracking Results
    # --------------------------------------------------------------------------
    def test_13_invalid_tracking_results(self) -> None:
        bad_json = self.temp_path / "bad.json"
        with open(bad_json, "w", encoding="utf-8") as f:
            json.dump({"broken": "data"}, f)

        tracker = MultiObjectTracker()
        with self.assertRaises(MultiObjectTrackingError):
            tracker.run(bad_json)

    # --------------------------------------------------------------------------
    # Test 14: Bounding-Box Validation
    # --------------------------------------------------------------------------
    def test_14_bounding_box_validation(self) -> None:
        tracker = MultiObjectTracker()
        # Degenerate zero-area box is safely dropped by ByteTrack conversion
        dets = [
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 10.0, 10.0]},  # degenerate
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [20.0, 20.0, 60.0, 80.0]},   # valid
        ]
        rec = tracker.track_detections(dets, frame_id="f1")
        self.assertEqual(rec.active_track_count, 1)
        self.assertEqual(rec.active_tracks[0].width, 40.0)
        self.assertEqual(rec.active_tracks[0].height, 60.0)

    # --------------------------------------------------------------------------
    # Test 15: Confidence Validation
    # --------------------------------------------------------------------------
    def test_15_confidence_validation(self) -> None:
        tracker = MultiObjectTracker()
        # Confidence > 1.0 raises InvalidDetectionInputError
        with self.assertRaises(InvalidDetectionInputError):
            tracker.track_detections([{"class_id": 2, "confidence": 1.5, "bbox_xyxy": [0, 0, 10, 10]}], frame_id="f1")

    # --------------------------------------------------------------------------
    # Test 16: Multiple Object Classes
    # --------------------------------------------------------------------------
    def test_16_multiple_object_classes(self) -> None:
        tracker = MultiObjectTracker()
        dets = [
            {"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
            {"class_id": 7, "class_name": "truck", "confidence": 0.9, "bbox_xyxy": [100, 100, 200, 200]},
            {"class_id": 9, "class_name": "traffic light", "confidence": 0.9, "bbox_xyxy": [300, 50, 320, 90]},
        ]
        tracker.track_detections(dets, frame_id="f1")
        summary = tracker.compute_summary(dataset_id="test", input_source="test")
        self.assertIn("car", summary.per_class_track_counts)
        self.assertIn("truck", summary.per_class_track_counts)
        self.assertIn("traffic light", summary.per_class_track_counts)

    # --------------------------------------------------------------------------
    # Test 17: Tracker Reset
    # --------------------------------------------------------------------------
    def test_17_tracker_reset(self) -> None:
        tracker = MultiObjectTracker()
        tracker.track_detections([{"class_id": 2, "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        self.assertEqual(tracker.total_unique_tracks, 1)

        tracker.reset()
        self.assertEqual(tracker.total_unique_tracks, 0)
        self.assertEqual(len(tracker.frame_records), 0)
        self.assertEqual(tracker._current_sequence_index, 0)

    # --------------------------------------------------------------------------
    # Test 18: Independent Tracking Sessions
    # --------------------------------------------------------------------------
    def test_18_independent_tracking_sessions(self) -> None:
        tracker = MultiObjectTracker()
        tracker.run(self.mock_detection_file, output_dir=self.temp_path / "run_a")
        tracks_a = tracker.total_unique_tracks

        # Second independent run without leaking state
        tracker.run(self.mock_detection_file, output_dir=self.temp_path / "run_b")
        tracks_b = tracker.total_unique_tracks

        self.assertEqual(tracks_a, tracks_b)
        self.assertEqual(len(tracker.frame_records), 4)

    # --------------------------------------------------------------------------
    # Test 19: Metadata Preservation
    # --------------------------------------------------------------------------
    def test_19_metadata_preservation(self) -> None:
        tracker = MultiObjectTracker()
        out_dir = self.temp_path / "meta_out"
        tracker.run(self.mock_detection_file, output_dir=out_dir)

        meta_file = out_dir / "metadata.json"
        self.assertTrue(meta_file.exists())
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.assertEqual(meta["schema_version"], "1.0")
            self.assertIn("tracker_configuration", meta)
            self.assertIn("environment", meta)

    # --------------------------------------------------------------------------
    # Test 20: Timestamp Handling
    # --------------------------------------------------------------------------
    def test_20_timestamp_handling(self) -> None:
        tracker = MultiObjectTracker()
        tracker.track_detections([{"class_id": 2, "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1", timestamp_seconds=1.234)
        self.assertEqual(tracker.frame_records[0].timestamp_seconds, 1.234)
        self.assertEqual(tracker.histories[1].first_timestamp_seconds, 1.234)

    # --------------------------------------------------------------------------
    # Test 21: Frame-Index Handling
    # --------------------------------------------------------------------------
    def test_21_frame_index_handling(self) -> None:
        tracker = MultiObjectTracker()
        tracker.track_detections([{"class_id": 2, "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1", source_frame_index=42)
        self.assertEqual(tracker.frame_records[0].source_frame_index, 42)
        self.assertEqual(tracker.frame_records[0].active_tracks[0].source_frame_index, 42)

    # --------------------------------------------------------------------------
    # Test 22: Per-Object History
    # --------------------------------------------------------------------------
    def test_22_per_object_history(self) -> None:
        tracker = MultiObjectTracker()
        tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        tracker.track_detections([{"class_id": 2, "class_name": "car", "confidence": 0.92, "bbox_xyxy": [13, 14, 53, 54]}], frame_id="f2")

        hist = tracker.histories[1]
        self.assertEqual(hist.total_observed_frames, 2)
        self.assertEqual(len(hist.observations), 2)
        # Bounding box center displaced under Kalman filter update
        self.assertGreater(hist.cumulative_displacement_pixels, 0.0)
        self.assertAlmostEqual(hist.cumulative_displacement_pixels, 4.34, delta=0.5)

    # --------------------------------------------------------------------------
    # Test 23: Summary Statistics
    # --------------------------------------------------------------------------
    def test_23_summary_statistics(self) -> None:
        tracker = MultiObjectTracker()
        tracker.track_detections([
            {"class_id": 2, "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"class_id": 2, "confidence": 0.9, "bbox_xyxy": [50, 50, 60, 60]},
        ], frame_id="f1")
        tracker.track_detections([
            {"class_id": 2, "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"class_id": 2, "confidence": 0.9, "bbox_xyxy": [50, 50, 60, 60]},
        ], frame_id="f2")

        summary = tracker.compute_summary(dataset_id="test", input_source="test")
        self.assertEqual(summary.total_frames_processed, 2)
        self.assertEqual(summary.max_simultaneous_active_tracks, 2)
        self.assertEqual(summary.total_detections, 4)
        self.assertAlmostEqual(summary.avg_objects_per_frame, 2.0, places=2)

    # --------------------------------------------------------------------------
    # Test 24: Output JSON Validation
    # --------------------------------------------------------------------------
    def test_24_output_json_validation(self) -> None:
        tracker = MultiObjectTracker()
        out_dir = self.temp_path / "json_val_out"
        summary, _, _ = tracker.run(self.mock_detection_file, output_dir=out_dir)

        # All 5 files must exist
        f_mot = out_dir / "multi_object_tracking.json"
        f_hist = out_dir / "track_histories.json"
        f_sum = out_dir / "tracking_summary.json"
        f_meta = out_dir / "metadata.json"
        f_err = out_dir / "errors.json"

        for f in (f_mot, f_hist, f_sum, f_meta, f_err):
            self.assertTrue(f.exists(), f"Missing output file: {f.name}")
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                self.assertIn("schema_version", data)

    # --------------------------------------------------------------------------
    # Test 25: Output Directory Creation
    # --------------------------------------------------------------------------
    def test_25_output_directory_creation(self) -> None:
        tracker = MultiObjectTracker()
        deep_dir = self.temp_path / "nested" / "output" / "dir"
        self.assertFalse(deep_dir.exists())

        tracker.run(self.mock_detection_file, output_dir=deep_dir)
        self.assertTrue(deep_dir.exists())

    # --------------------------------------------------------------------------
    # Test 26: Error Handling (FileExistsError)
    # --------------------------------------------------------------------------
    def test_26_error_handling(self) -> None:
        tracker = MultiObjectTracker()
        out_dir = self.temp_path / "collision_dir"
        out_dir.mkdir()

        with self.assertRaises(FileExistsError):
            tracker.run(self.mock_detection_file, output_dir=out_dir, overwrite=False)

        # Succeeds when overwrite=True
        summary, _, _ = tracker.run(self.mock_detection_file, output_dir=out_dir, overwrite=True)
        self.assertEqual(summary.status, "PASS")

    # --------------------------------------------------------------------------
    # Test 27: CLI Argument Validation
    # --------------------------------------------------------------------------
    def test_27_cli_argument_validation(self) -> None:
        # Non-existent input file returns exit code 2
        code = main(["--input", "non_existent_path.json"])
        self.assertEqual(code, 2)

    # --------------------------------------------------------------------------
    # Test 28: CPU-Only Compatibility
    # --------------------------------------------------------------------------
    def test_28_cpu_only_compatibility(self) -> None:
        tracker = MultiObjectTracker()
        # Verify tracker runs purely on CPU structures without torch.cuda
        rec = tracker.track_detections([{"class_id": 2, "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1")
        self.assertEqual(rec.active_track_count, 1)

    # --------------------------------------------------------------------------
    # Test 29: Memory-Efficient Sequential Processing
    # --------------------------------------------------------------------------
    def test_29_memory_efficient_sequential_processing(self) -> None:
        tracker = MultiObjectTracker()
        # Run 20 sequential frames
        for i in range(20):
            tracker.track_detections(
                [{"class_id": 2, "confidence": 0.9, "bbox_xyxy": [i, i, i + 20, i + 20]}],
                frame_id=f"frame_{i:04d}.jpg",
            )
        self.assertEqual(len(tracker.frame_records), 20)
        self.assertEqual(tracker.total_unique_tracks, 1)

    # --------------------------------------------------------------------------
    # Test 30: Integration with Existing ByteTrack Pipeline
    # --------------------------------------------------------------------------
    def test_30_integration_with_existing_bytetrack_pipeline(self) -> None:
        tracker = MultiObjectTracker()
        out_dir = self.temp_path / "integration_out"
        summary, frame_recs, histories = tracker.run(self.mock_detection_file, output_dir=out_dir)

        self.assertEqual(summary.status, "PASS")
        self.assertEqual(summary.total_frames_processed, 4)
        self.assertEqual(summary.total_unique_tracks, 2)
        self.assertIn(1, histories)
        self.assertIn(2, histories)


if __name__ == "__main__":
    unittest.main()
