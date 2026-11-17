"""PerceptGuard AI: Comprehensive Unit and Integration Tests for SingleObjectTracker.

Validates 30 production computer vision and object tracking requirements:
1. Tracker initialization (default and custom params)
2. Target ID selection
3. Class-based selection
4. Multiple candidate handling & disambiguation rules
5. Target not found handling (TargetNotFoundError)
6. Initial target observation (INITIALIZED state)
7. Consecutive target observations (TRACKING state)
8. Persistent tracking ID continuity across multiple frames
9. Temporary tracking loss (TEMPORARILY_LOST state)
10. Target reacquisition (REACQUIRED state)
11. Target termination (TERMINATED state when lost > max_lost_frames)
12. Complete state machine transitions (UNINITIALIZED -> INITIALIZED -> TRACKING -> TEMPORARILY_LOST -> REACQUIRED -> TERMINATED)
13. Bounding box coordinate extraction
14. Center-coordinate calculation (cx, cy)
15. Bounding box dimensions (w, h)
16. Frame displacement calculation (Euclidean distance)
17. Missing timestamps handling (timestamp=None)
18. Timestamp-based duration and pixel velocity calculation
19. Missing observations recorded in history
20. Confidence statistics (mean, std dev)
21. Visibility percentage calculation
22. Invalid bounding boxes handling
23. Duplicate frame identifiers handling
24. Out-of-order frame natural sorting
25. Invalid tracking records handling
26. JSON serialization (to_dict() schema completeness)
27. Output directory protection (FileExistsError when exists and overwrite=False)
28. CLI argument validation (mutually exclusive args and invalid target ID)
29. Partial run reporting (status WARNING when terminated early)
30. End-to-end single-object tracking on mock and real tracking data
"""

import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import patch

from src.tracking.bytetrack_engine import TrackRecord
from src.tracking.single_object_tracker import (
    AmbiguousTargetSelectionError,
    InvalidTargetIdError,
    SingleObjectTracker,
    SingleObjectTrackingError,
    SingleObjectTrackingState,
    SingleObjectTrackingSummary,
    TargetNotFoundError,
    TargetObservation,
    TargetSelectionRule,
    main,
)


class TestSingleObjectTracker(unittest.TestCase):
    """Test suite covering 30 single-object tracking requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Helper to construct sample tracking results json
        self.mock_tracking_results_file = self.temp_path / "mock_tracking_results.json"
        self.sample_payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": "test_track_run_001",
            "dataset_id": "test_driving",
            "model_name": "yolo11n",
            "total_frames": 5,
            "frame_results": [
                {
                    "image_filename": "frame_000001.jpg",
                    "extracted_frame_id": 1,
                    "source_frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 9,
                            "class_name": "traffic light",
                            "confidence": 0.85,
                            "bbox_xyxy": [100.0, 150.0, 140.0, 250.0],
                        },
                        {
                            "track_id": 2,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.90,
                            "bbox_xyxy": [300.0, 400.0, 500.0, 600.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000002.jpg",
                    "extracted_frame_id": 2,
                    "source_frame_index": 1,
                    "timestamp_seconds": 0.033,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 9,
                            "class_name": "traffic light",
                            "confidence": 0.87,
                            "bbox_xyxy": [102.0, 151.0, 142.0, 251.0],
                        },
                        {
                            "track_id": 2,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.91,
                            "bbox_xyxy": [305.0, 402.0, 505.0, 602.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000003.jpg",
                    "extracted_frame_id": 3,
                    "source_frame_index": 2,
                    "timestamp_seconds": 0.066,
                    "tracks": [
                        # Track 1 temporarily occluded / missing
                        {
                            "track_id": 2,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.89,
                            "bbox_xyxy": [310.0, 405.0, 510.0, 605.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000004.jpg",
                    "extracted_frame_id": 4,
                    "source_frame_index": 3,
                    "timestamp_seconds": 0.100,
                    "tracks": [
                        # Track 1 reacquired
                        {
                            "track_id": 1,
                            "class_id": 9,
                            "class_name": "traffic light",
                            "confidence": 0.84,
                            "bbox_xyxy": [105.0, 152.0, 145.0, 252.0],
                        },
                    ],
                },
                {
                    "image_filename": "frame_000005.jpg",
                    "extracted_frame_id": 5,
                    "source_frame_index": 4,
                    "timestamp_seconds": 0.133,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 9,
                            "class_name": "traffic light",
                            "confidence": 0.88,
                            "bbox_xyxy": [106.0, 153.0, 146.0, 253.0],
                        },
                    ],
                },
            ],
        }
        with open(self.mock_tracking_results_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_payload, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # Test 1: Tracker Initialization
    # --------------------------------------------------------------------------
    def test_01_tracker_initialization_default_and_custom(self) -> None:
        # Default initialization
        tracker = SingleObjectTracker()
        self.assertIsNone(tracker.target_id)
        self.assertIsNone(tracker.target_class)
        self.assertEqual(tracker.selection_rule, TargetSelectionRule.HIGHEST_CONFIDENCE)
        self.assertEqual(tracker.max_lost_frames, 30)
        self.assertEqual(tracker.state, SingleObjectTrackingState.UNINITIALIZED)
        self.assertEqual(len(tracker.history), 0)

        # Custom initialization
        custom_tracker = SingleObjectTracker(
            target_id=2,
            target_class="truck",
            selection_rule=TargetSelectionRule.LARGEST_AREA,
            max_lost_frames=15,
            dataset_id="custom_dataset",
        )
        self.assertEqual(custom_tracker.target_id, 2)
        self.assertEqual(custom_tracker.target_class, "truck")
        self.assertEqual(custom_tracker.selection_rule, TargetSelectionRule.LARGEST_AREA)
        self.assertEqual(custom_tracker.max_lost_frames, 15)
        self.assertEqual(custom_tracker.dataset_id, "custom_dataset")

        # Invalid target ID
        with self.assertRaises(InvalidTargetIdError):
            SingleObjectTracker(target_id=-1)
        with self.assertRaises(InvalidTargetIdError):
            SingleObjectTracker(target_id=0)

    # --------------------------------------------------------------------------
    # Test 2: Target ID Selection
    # --------------------------------------------------------------------------
    def test_02_target_id_selection(self) -> None:
        tracker = SingleObjectTracker(target_id=2)
        candidates = [
            {"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.8},
            {"track_id": 2, "class_id": 7, "class_name": "truck", "confidence": 0.95},
        ]
        selected_id = tracker.select_target_from_candidates(candidates)
        self.assertEqual(selected_id, 2)
        self.assertEqual(tracker.target_class_id, 7)
        self.assertEqual(tracker.target_class_name, "truck")

    # --------------------------------------------------------------------------
    # Test 3: Class-based Selection
    # --------------------------------------------------------------------------
    def test_03_class_based_selection(self) -> None:
        tracker = SingleObjectTracker(target_class="traffic light")
        candidates = [
            {"track_id": 10, "class_id": 9, "class_name": "traffic light", "confidence": 0.8},
            {"track_id": 12, "class_id": 2, "class_name": "car", "confidence": 0.9},
        ]
        selected_id = tracker.select_target_from_candidates(candidates)
        self.assertEqual(selected_id, 10)
        self.assertEqual(tracker.target_id, 10)
        self.assertEqual(tracker.target_class_name, "traffic light")

    # --------------------------------------------------------------------------
    # Test 4: Multiple Candidate Handling & Disambiguation Rules
    # --------------------------------------------------------------------------
    def test_04_multiple_candidate_disambiguation_rules(self) -> None:
        candidates = [
            {"track_id": 101, "class_id": 2, "class_name": "car", "confidence": 0.70, "bbox_xyxy": [0, 0, 100, 100]}, # area 10000
            {"track_id": 102, "class_id": 2, "class_name": "car", "confidence": 0.95, "bbox_xyxy": [0, 0, 40, 40]},    # area 1600
            {"track_id": 103, "class_id": 2, "class_name": "car", "confidence": 0.85, "bbox_xyxy": [0, 0, 50, 50]},    # area 2500
        ]

        # Highest confidence rule
        t_conf = SingleObjectTracker(target_class="car", selection_rule=TargetSelectionRule.HIGHEST_CONFIDENCE)
        self.assertEqual(t_conf.select_target_from_candidates(candidates), 102)

        # Largest area rule
        t_area = SingleObjectTracker(target_class="car", selection_rule=TargetSelectionRule.LARGEST_AREA)
        self.assertEqual(t_area.select_target_from_candidates(candidates), 101)

        # First detected rule
        t_first = SingleObjectTracker(target_class="car", selection_rule=TargetSelectionRule.FIRST_DETECTED)
        self.assertEqual(t_first.select_target_from_candidates(candidates), 101)

    # --------------------------------------------------------------------------
    # Test 5: Target Not Found Handling
    # --------------------------------------------------------------------------
    def test_05_target_not_found_handling(self) -> None:
        tracker_id = SingleObjectTracker(target_id=999)
        candidates = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.8}]
        with self.assertRaises(TargetNotFoundError):
            tracker_id.select_target_from_candidates(candidates)

        tracker_cls = SingleObjectTracker(target_class="bicycle")
        with self.assertRaises(TargetNotFoundError):
            tracker_cls.select_target_from_candidates(candidates)

        # Empty candidates
        with self.assertRaises(TargetNotFoundError):
            tracker_id.select_target_from_candidates([])

    # --------------------------------------------------------------------------
    # Test 6: Initial Target Observation (INITIALIZED State)
    # --------------------------------------------------------------------------
    def test_06_initial_target_observation_state(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]
        obs = tracker.step(tracks, frame_id="frame_0001.jpg", source_frame_index=0, timestamp_seconds=0.0)

        self.assertEqual(obs.tracking_state, SingleObjectTrackingState.INITIALIZED.value)
        self.assertTrue(obs.is_visible)
        self.assertEqual(obs.track_id, 1)
        self.assertIsNone(obs.frame_displacement_pixels)
        self.assertIsNone(obs.velocity_pixels_per_second)

    # --------------------------------------------------------------------------
    # Test 7: Consecutive Target Observations (TRACKING State)
    # --------------------------------------------------------------------------
    def test_07_consecutive_target_observations_state(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks_1 = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]
        tracks_2 = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.88, "bbox_xyxy": [12.0, 20.0, 52.0, 80.0]}]

        obs1 = tracker.step(tracks_1, frame_id="frame_0001.jpg", timestamp_seconds=0.0)
        obs2 = tracker.step(tracks_2, frame_id="frame_0002.jpg", timestamp_seconds=0.05)

        self.assertEqual(obs1.tracking_state, SingleObjectTrackingState.INITIALIZED.value)
        self.assertEqual(obs2.tracking_state, SingleObjectTrackingState.TRACKING.value)
        self.assertTrue(obs2.is_visible)
        self.assertAlmostEqual(obs2.frame_displacement_pixels, 2.0, places=2)

    # --------------------------------------------------------------------------
    # Test 8: Persistent Tracking ID Continuity Across Multiple Frames
    # --------------------------------------------------------------------------
    def test_08_persistent_tracking_id_continuity(self) -> None:
        tracker = SingleObjectTracker(target_id=5)
        for i in range(5):
            tracks = [
                {"track_id": 5, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [i * 10, i * 10, (i + 1) * 20, (i + 1) * 20]},
                {"track_id": 6, "class_id": 2, "class_name": "car", "confidence": 0.95, "bbox_xyxy": [100, 100, 200, 200]},
            ]
            obs = tracker.step(tracks, frame_id=f"frame_{i:04d}.jpg", timestamp_seconds=i * 0.1)
            self.assertEqual(obs.track_id, 5)

        self.assertEqual(len(tracker.history), 5)
        for obs in tracker.history:
            self.assertEqual(obs.track_id, 5)

    # --------------------------------------------------------------------------
    # Test 9: Temporary Tracking Loss (TEMPORARILY_LOST State)
    # --------------------------------------------------------------------------
    def test_09_temporary_tracking_loss_state(self) -> None:
        tracker = SingleObjectTracker(target_id=1, max_lost_frames=5)
        tracks_present = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]
        tracker.step(tracks_present, frame_id="frame_0001.jpg")

        # Frame where target is missing
        obs_missing = tracker.step([], frame_id="frame_0002.jpg")
        self.assertEqual(obs_missing.tracking_state, SingleObjectTrackingState.TEMPORARILY_LOST.value)
        self.assertFalse(obs_missing.is_visible)
        self.assertEqual(obs_missing.consecutive_lost_frames, 1)
        self.assertIsNone(obs_missing.bbox_xyxy)

    # --------------------------------------------------------------------------
    # Test 10: Target Reacquisition (REACQUIRED State)
    # --------------------------------------------------------------------------
    def test_10_target_reacquisition_state(self) -> None:
        tracker = SingleObjectTracker(target_id=1, max_lost_frames=5)
        tracks = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]

        tracker.step(tracks, frame_id="frame_0001.jpg")
        tracker.step([], frame_id="frame_0002.jpg")  # TEMPORARILY_LOST
        obs_reacquired = tracker.step(tracks, frame_id="frame_0003.jpg")

        self.assertEqual(obs_reacquired.tracking_state, SingleObjectTrackingState.REACQUIRED.value)
        self.assertTrue(obs_reacquired.is_visible)
        self.assertEqual(tracker._reacquisitions, 1)

    # --------------------------------------------------------------------------
    # Test 11: Target Termination (TERMINATED State)
    # --------------------------------------------------------------------------
    def test_11_target_termination_state(self) -> None:
        tracker = SingleObjectTracker(target_id=1, max_lost_frames=2)
        tracks = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]

        tracker.step(tracks, frame_id="frame_0001.jpg")  # INITIALIZED
        tracker.step([], frame_id="frame_0002.jpg")      # TEMPORARILY_LOST (lost 1)
        tracker.step([], frame_id="frame_0003.jpg")      # TEMPORARILY_LOST (lost 2)
        obs_term = tracker.step([], frame_id="frame_0004.jpg")  # TERMINATED (lost 3 > max_lost_frames 2)

        self.assertEqual(obs_term.tracking_state, SingleObjectTrackingState.TERMINATED.value)
        self.assertEqual(tracker.state, SingleObjectTrackingState.TERMINATED)

    # --------------------------------------------------------------------------
    # Test 12: Complete State Machine Transitions
    # --------------------------------------------------------------------------
    def test_12_complete_state_machine_transitions(self) -> None:
        tracker = SingleObjectTracker(target_id=1, max_lost_frames=1)
        self.assertEqual(tracker.state, SingleObjectTrackingState.UNINITIALIZED)

        tracks = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]

        # 1. INITIALIZED
        tracker.step(tracks, frame_id="f1")
        self.assertEqual(tracker.state, SingleObjectTrackingState.INITIALIZED)

        # 2. TRACKING
        tracker.step(tracks, frame_id="f2")
        self.assertEqual(tracker.state, SingleObjectTrackingState.TRACKING)

        # 3. TEMPORARILY_LOST
        tracker.step([], frame_id="f3")
        self.assertEqual(tracker.state, SingleObjectTrackingState.TEMPORARILY_LOST)

        # 4. REACQUIRED
        tracker.step(tracks, frame_id="f4")
        self.assertEqual(tracker.state, SingleObjectTrackingState.REACQUIRED)

        # 5. TEMPORARILY_LOST (lost 1)
        tracker.step([], frame_id="f5")
        self.assertEqual(tracker.state, SingleObjectTrackingState.TEMPORARILY_LOST)

        # 6. TERMINATED (lost 2 > 1)
        tracker.step([], frame_id="f6")
        self.assertEqual(tracker.state, SingleObjectTrackingState.TERMINATED)

    # --------------------------------------------------------------------------
    # Test 13: Bounding Box Coordinate Extraction
    # --------------------------------------------------------------------------
    def test_13_bounding_box_coordinate_extraction(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        bbox = [12.5, 34.2, 56.7, 78.9]
        tracks = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": bbox}]
        obs = tracker.step(tracks, frame_id="f1")

        self.assertEqual(obs.bbox_xyxy, bbox)

    # --------------------------------------------------------------------------
    # Test 14: Center-Coordinate Calculation (cx, cy)
    # --------------------------------------------------------------------------
    def test_14_center_coordinate_calculation(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0, 20.0, 30.0, 60.0]}]
        obs = tracker.step(tracks, frame_id="f1")

        self.assertEqual(obs.bbox_center_x, 20.0)  # (10 + 30) / 2
        self.assertEqual(obs.bbox_center_y, 40.0)  # (20 + 60) / 2

    # --------------------------------------------------------------------------
    # Test 15: Bounding Box Dimensions (w, h)
    # --------------------------------------------------------------------------
    def test_15_bounding_box_dimensions(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0, 20.0, 35.0, 65.0]}]
        obs = tracker.step(tracks, frame_id="f1")

        self.assertEqual(obs.bbox_width, 25.0)   # 35 - 10
        self.assertEqual(obs.bbox_height, 45.0)  # 65 - 20

    # --------------------------------------------------------------------------
    # Test 16: Frame Displacement Calculation (Euclidean Distance)
    # --------------------------------------------------------------------------
    def test_16_frame_displacement_calculation(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        # Point 1: cx=10, cy=10 (bbox: [0, 0, 20, 20])
        tracks_1 = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}]
        # Point 2: cx=13, cy=14 (bbox: [3, 4, 23, 24]) -> displacement = sqrt(3^2 + 4^2) = 5.0
        tracks_2 = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [3.0, 4.0, 23.0, 24.0]}]

        tracker.step(tracks_1, frame_id="f1")
        obs2 = tracker.step(tracks_2, frame_id="f2")

        self.assertAlmostEqual(obs2.frame_displacement_pixels, 5.0, places=2)
        self.assertAlmostEqual(tracker._cumulative_displacement, 5.0, places=2)

    # --------------------------------------------------------------------------
    # Test 17: Missing Timestamps Handling (timestamp=None)
    # --------------------------------------------------------------------------
    def test_17_missing_timestamps_handling(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks_1 = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}]
        tracks_2 = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]}]

        tracker.step(tracks_1, frame_id="f1", timestamp_seconds=None)
        obs2 = tracker.step(tracks_2, frame_id="f2", timestamp_seconds=None)

        self.assertIsNotNone(obs2.frame_displacement_pixels)
        self.assertIsNone(obs2.velocity_pixels_per_second)

    # --------------------------------------------------------------------------
    # Test 18: Timestamp-based Duration and Pixel Velocity Calculation
    # --------------------------------------------------------------------------
    def test_18_timestamp_based_duration_and_velocity(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks_1 = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}]
        # Displacement = 10 px in x direction (cx moves from 10 to 20), dt = 0.5s -> velocity = 20 px/s
        tracks_2 = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0, 0.0, 30.0, 20.0]}]

        tracker.step(tracks_1, frame_id="f1", timestamp_seconds=1.0)
        obs2 = tracker.step(tracks_2, frame_id="f2", timestamp_seconds=1.5)

        self.assertAlmostEqual(obs2.frame_displacement_pixels, 10.0, places=2)
        self.assertAlmostEqual(obs2.velocity_pixels_per_second, 20.0, places=2)

    # --------------------------------------------------------------------------
    # Test 19: Missing Observations Recorded in History
    # --------------------------------------------------------------------------
    def test_19_missing_observations_recorded_in_history(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracker.step([{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}], frame_id="f1")
        tracker.step([], frame_id="f2")

        self.assertEqual(len(tracker.history), 2)
        self.assertFalse(tracker.history[1].is_visible)
        self.assertIsNone(tracker.history[1].confidence)
        self.assertIsNone(tracker.history[1].bbox_xyxy)

    # --------------------------------------------------------------------------
    # Test 20: Confidence Statistics (Mean, Std Dev)
    # --------------------------------------------------------------------------
    def test_20_confidence_statistics(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        confs = [0.80, 0.90, 0.70]
        for idx, conf in enumerate(confs):
            tracker.step(
                [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": conf, "bbox_xyxy": [0, 0, 10, 10]}],
                frame_id=f"f{idx}",
            )

        summary = tracker.compute_summary(dataset_id="test")
        self.assertAlmostEqual(summary.mean_confidence, 0.80, places=3)
        self.assertAlmostEqual(summary.confidence_std_dev, 0.0816, places=3)

    # --------------------------------------------------------------------------
    # Test 21: Visibility Percentage Calculation
    # --------------------------------------------------------------------------
    def test_21_visibility_percentage_calculation(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        # 3 visible, 1 missing -> 75.0%
        tracks = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]
        tracker.step(tracks, frame_id="f1")
        tracker.step(tracks, frame_id="f2")
        tracker.step([], frame_id="f3")
        tracker.step(tracks, frame_id="f4")

        summary = tracker.compute_summary(dataset_id="test")
        self.assertEqual(summary.total_frames_processed, 4)
        self.assertEqual(summary.visible_frame_count, 3)
        self.assertEqual(summary.missing_frame_count, 1)
        self.assertAlmostEqual(summary.visibility_percentage, 75.0, places=1)

    # --------------------------------------------------------------------------
    # Test 22: Invalid Bounding Boxes Handling
    # --------------------------------------------------------------------------
    def test_22_invalid_bounding_boxes_handling(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        # Inverted coordinates: x2 < x1, y2 < y1
        tracks = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [100.0, 100.0, 50.0, 50.0]}]
        obs = tracker.step(tracks, frame_id="f1")

        self.assertEqual(obs.bbox_width, 0.0)
        self.assertEqual(obs.bbox_height, 0.0)

        # Truncated or malformed bounding box in candidate
        bad_tracks = [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0]}]
        obs2 = tracker.step(bad_tracks, frame_id="f2")
        self.assertEqual(obs2.bbox_xyxy, [0.0, 0.0, 0.0, 0.0])

    # --------------------------------------------------------------------------
    # Test 23: Duplicate Frame Identifiers Handling
    # --------------------------------------------------------------------------
    def test_23_duplicate_frame_identifiers_handling(self) -> None:
        # Create a payload with duplicate frame entries
        dup_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_dup",
            "frame_results": [
                {
                    "image_filename": "frame_000001.jpg",
                    "tracks": [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}],
                },
                {
                    "image_filename": "frame_000001.jpg",  # duplicate
                    "tracks": [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}],
                },
                {
                    "image_filename": "frame_000002.jpg",
                    "tracks": [{"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [5, 5, 15, 15]}],
                },
            ],
        }
        dup_file = self.temp_path / "dup_tracking.json"
        with open(dup_file, "w", encoding="utf-8") as f:
            json.dump(dup_payload, f)

        tracker = SingleObjectTracker(target_id=1)
        summary, history = tracker.run_from_tracking_results(dup_file, output_dir=self.temp_path / "dup_out")

        # Deduplication must result in exactly 2 unique frames processed
        self.assertEqual(summary.total_frames_processed, 2)
        self.assertEqual(len(history), 2)

    # --------------------------------------------------------------------------
    # Test 24: Out-of-Order Frame Natural Sorting
    # --------------------------------------------------------------------------
    def test_24_out_of_order_frame_natural_sorting(self) -> None:
        shuffled_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_sort",
            "frame_results": [
                {"image_filename": "frame_10.jpg", "tracks": [{"track_id": 1, "class_id": 2, "bbox_xyxy": [0, 0, 10, 10]}]},
                {"image_filename": "frame_2.jpg", "tracks": [{"track_id": 1, "class_id": 2, "bbox_xyxy": [0, 0, 10, 10]}]},
                {"image_filename": "frame_1.jpg", "tracks": [{"track_id": 1, "class_id": 2, "bbox_xyxy": [0, 0, 10, 10]}]},
            ],
        }
        shuffled_file = self.temp_path / "shuffled_tracking.json"
        with open(shuffled_file, "w", encoding="utf-8") as f:
            json.dump(shuffled_payload, f)

        tracker = SingleObjectTracker(target_id=1)
        _, history = tracker.run_from_tracking_results(shuffled_file, output_dir=self.temp_path / "sort_out")

        sorted_frame_names = [o.frame_id for o in history]
        self.assertEqual(sorted_frame_names, ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"])

    # --------------------------------------------------------------------------
    # Test 25: Invalid Tracking Records Handling
    # --------------------------------------------------------------------------
    def test_25_invalid_tracking_records_handling(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        # Track record with string confidence and missing class fields
        bad_track = {"track_id": "1", "confidence": "0.92"}
        obs = tracker.step([bad_track], frame_id="f1")

        self.assertEqual(obs.track_id, 1)
        self.assertEqual(obs.confidence, 0.92)
        self.assertEqual(obs.class_id, 0)
        self.assertEqual(obs.class_name, "unknown")

    # --------------------------------------------------------------------------
    # Test 26: JSON Serialization (to_dict Schema Completeness)
    # --------------------------------------------------------------------------
    def test_26_json_serialization_schema_completeness(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        tracks = [{"track_id": 1, "class_id": 9, "class_name": "traffic light", "confidence": 0.85, "bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]
        obs = tracker.step(tracks, frame_id="f1", timestamp_seconds=0.0)
        d_obs = obs.to_dict()

        expected_obs_keys = {
            "frame_id", "source_frame_index", "timestamp_seconds", "track_id",
            "class_id", "class_name", "confidence", "bbox_xyxy", "bbox_center_x",
            "bbox_center_y", "bbox_width", "bbox_height", "frame_displacement_pixels",
            "velocity_pixels_per_second", "tracking_state", "is_visible", "consecutive_lost_frames"
        }
        self.assertTrue(expected_obs_keys.issubset(set(d_obs.keys())))

        summary = tracker.compute_summary(dataset_id="test")
        d_sum = summary.to_dict()
        expected_sum_keys = {
            "schema_version", "run_id", "dataset_id", "target_id", "target_class",
            "selection_method", "selection_rule", "total_frames_processed",
            "visible_frame_count", "missing_frame_count", "visibility_percentage",
            "reacquisition_count", "longest_consecutive_missing_frames", "final_state", "status"
        }
        self.assertTrue(expected_sum_keys.issubset(set(d_sum.keys())))

    # --------------------------------------------------------------------------
    # Test 27: Output Directory Protection
    # --------------------------------------------------------------------------
    def test_27_output_directory_protection(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        out_dir = self.temp_path / "protected_dir"
        out_dir.mkdir()

        # Without overwrite -> raises FileExistsError
        with self.assertRaises(FileExistsError):
            tracker.run_from_tracking_results(self.mock_tracking_results_file, output_dir=out_dir, overwrite=False)

        # With overwrite -> succeeds
        summary, _ = tracker.run_from_tracking_results(self.mock_tracking_results_file, output_dir=out_dir, overwrite=True)
        self.assertEqual(summary.total_frames_processed, 5)

    # --------------------------------------------------------------------------
    # Test 28: CLI Argument Validation
    # --------------------------------------------------------------------------
    def test_28_cli_argument_validation(self) -> None:
        # Mutually exclusive --target-id and --target-class
        exit_code = main(["--target-id", "1", "--target-class", "car"])
        self.assertEqual(exit_code, 1)

        # Invalid target ID <= 0
        exit_code = main(["--target-id", "0"])
        self.assertEqual(exit_code, 1)

        exit_code = main(["--target-id", "-5"])
        self.assertEqual(exit_code, 1)

    # --------------------------------------------------------------------------
    # Test 29: Partial Run Reporting (Status WARNING)
    # --------------------------------------------------------------------------
    def test_29_partial_run_reporting(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        summary, _ = tracker.run_from_tracking_results(
            self.mock_tracking_results_file,
            output_dir=self.temp_path / "partial_out",
            max_frames=2,
        )
        self.assertEqual(summary.total_frames_processed, 2)
        self.assertEqual(summary.status, "WARNING")

    # --------------------------------------------------------------------------
    # Test 30: End-to-End Single-Object Tracking (Schema v1.0 Persistence)
    # --------------------------------------------------------------------------
    def test_30_end_to_end_single_object_tracking(self) -> None:
        tracker = SingleObjectTracker(target_id=1)
        e2e_out = self.temp_path / "e2e_run"
        summary, history = tracker.run_from_tracking_results(
            self.mock_tracking_results_file,
            output_dir=e2e_out,
        )

        self.assertEqual(summary.status, "PASS")
        self.assertEqual(summary.total_frames_processed, 5)
        self.assertEqual(summary.visible_frame_count, 4)
        self.assertEqual(summary.missing_frame_count, 1)
        self.assertEqual(summary.reacquisition_count, 1)

        # Check all 4 schema JSON files exist and are valid JSON
        f_tracking = e2e_out / "single_object_tracking.json"
        f_history = e2e_out / "target_history.json"
        f_summary = e2e_out / "tracking_summary.json"
        f_metadata = e2e_out / "tracking_metadata.json"

        for f in (f_tracking, f_history, f_summary, f_metadata):
            self.assertTrue(f.exists(), f"File {f.name} was not generated.")

        with open(f_tracking, "r", encoding="utf-8") as f:
            trk_json = json.load(f)
            self.assertEqual(trk_json["schema_version"], "1.0")
            self.assertEqual(len(trk_json["observations"]), 5)

        with open(f_history, "r", encoding="utf-8") as f:
            hist_json = json.load(f)
            self.assertEqual(hist_json["schema_version"], "1.0")
            self.assertEqual(hist_json["visible_count"], 4)
            self.assertEqual(len(hist_json["trajectory"]), 4)

        with open(f_summary, "r", encoding="utf-8") as f:
            sum_json = json.load(f)
            self.assertEqual(sum_json["schema_version"], "1.0")
            self.assertEqual(sum_json["target_id"], 1)

        with open(f_metadata, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
            self.assertEqual(meta_json["schema_version"], "1.0")
            self.assertIn("target_selection", meta_json)
            self.assertIn("tracker_configuration", meta_json)


if __name__ == "__main__":
    unittest.main()
