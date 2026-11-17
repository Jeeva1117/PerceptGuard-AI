"""PerceptGuard AI: Unit and Integration Tests for ByteTrackEngine.

Validates:
- Engine initialization and configuration validation
- Detection-to-adapter conversion (xyxy to xywh, types, degenerate box handling)
- Single, multiple, and zero detection processing
- Tracking ID assignment, persistence, and new track creation
- Tracker reset, session isolation, and frame counters
- Resilient error handling for invalid bounding boxes, confidences, and formats
"""

from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
import numpy as np

from src.tracking.bytetrack_engine import (
    ByteTrackEngine,
    InvalidDetectionInputError,
    TrackRecord,
    TrackingEngineError,
)
from src.tracking.tracker_config import TrackerConfig, TrackerConfigError


class TestByteTrackEngine(unittest.TestCase):
    """Test suite for ByteTrackEngine integration and data association."""

    def setUp(self) -> None:
        self.config = TrackerConfig.from_preset("default")
        self.engine = ByteTrackEngine(config=self.config)

    # --------------------------------------------------------------------------
    # 1. Initialization and Configuration Validation
    # --------------------------------------------------------------------------

    def test_01_default_initialization(self) -> None:
        """Test tracker initialization with default config."""
        engine = ByteTrackEngine()
        self.assertEqual(engine.frame_count, 0)
        self.assertEqual(engine.unique_track_count, 0)
        self.assertEqual(engine.config.tracker_type, "bytetrack")

    def test_02_initialization_with_custom_config(self) -> None:
        """Test tracker initialization with customized TrackerConfig."""
        custom_cfg = TrackerConfig(track_high_thresh=0.35, track_buffer=45)
        engine = ByteTrackEngine(config=custom_cfg)
        self.assertEqual(engine.config.track_high_thresh, 0.35)
        self.assertEqual(engine.config.track_buffer, 45)

    def test_03_initialization_with_dict(self) -> None:
        """Test tracker initialization with raw dictionary config."""
        d = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.30,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.30,
            "track_buffer": 25,
            "match_thresh": 0.75,
            "fuse_score": False,
        }
        engine = ByteTrackEngine(config=d)
        self.assertEqual(engine.config.match_thresh, 0.75)
        self.assertFalse(engine.config.fuse_score)

    def test_04_invalid_config_type_rejection(self) -> None:
        """Test rejection of unsupported configuration types."""
        with self.assertRaises(TrackerConfigError):
            ByteTrackEngine(config=12345)  # type: ignore

    # --------------------------------------------------------------------------
    # 2. Detection Ingestion & Bounding Box Conversions
    # --------------------------------------------------------------------------

    def test_05_single_detection_processing(self) -> None:
        """Test update with a single valid detection record."""
        dets = [
            {
                "bbox_xyxy": [100.0, 150.0, 200.0, 300.0],
                "confidence": 0.85,
                "class_id": 2,
                "class_name": "car",
            }
        ]
        tracks = self.engine.update(dets, frame_id="frame_000001.jpg")
        self.assertEqual(len(tracks), 1)
        self.assertIsInstance(tracks[0], TrackRecord)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].class_name, "car")
        self.assertEqual(self.engine.frame_count, 1)
        self.assertEqual(self.engine.unique_track_count, 1)

    def test_06_multiple_detections_processing(self) -> None:
        """Test update with multiple simultaneous detections."""
        dets = [
            {"bbox_xyxy": [50.0, 50.0, 120.0, 180.0], "confidence": 0.90, "class_id": 2, "class_name": "car"},
            {"bbox_xyxy": [300.0, 200.0, 450.0, 320.0], "confidence": 0.82, "class_id": 7, "class_name": "truck"},
            {"bbox_xyxy": [500.0, 100.0, 560.0, 220.0], "confidence": 0.75, "class_id": 0, "class_name": "person"},
        ]
        tracks = self.engine.update(dets, frame_id="frame_000001.jpg")
        self.assertEqual(len(tracks), 3)
        track_ids = {t.track_id for t in tracks}
        self.assertEqual(len(track_ids), 3)
        self.assertEqual(self.engine.unique_track_count, 3)

    def test_07_empty_detections_handling(self) -> None:
        """Test update with empty detection list preserves state and increments frame count."""
        tracks = self.engine.update([], frame_id="frame_empty.jpg")
        self.assertEqual(tracks, [])
        self.assertEqual(self.engine.frame_count, 1)
        self.assertEqual(self.engine.unique_track_count, 0)

    def test_08_degenerate_zero_area_box_filtering(self) -> None:
        """Test that boxes with zero width or zero height are safely dropped."""
        dets = [
            {"bbox_xyxy": [100.0, 100.0, 100.0, 200.0], "confidence": 0.9, "class_id": 2},  # zero width
            {"bbox_xyxy": [100.0, 100.0, 200.0, 100.0], "confidence": 0.9, "class_id": 2},  # zero height
            {"bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "confidence": 0.9, "class_id": 2, "class_name": "car"},  # valid
        ]
        tracks = self.engine.update(dets)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].class_name, "car")

    # --------------------------------------------------------------------------
    # 3. Track Persistence & Identity Continuity
    # --------------------------------------------------------------------------

    def test_09_track_persistence_across_adjacent_frames(self) -> None:
        """Test that an object moving slightly keeps the exact same track ID."""
        frame1 = [{"bbox_xyxy": [200.0, 150.0, 300.0, 250.0], "confidence": 0.88, "class_id": 2, "class_name": "car"}]
        tracks1 = self.engine.update(frame1, frame_id=1)
        self.assertEqual(len(tracks1), 1)
        first_id = tracks1[0].track_id

        # Adjacent frame with minor displacement
        frame2 = [{"bbox_xyxy": [204.0, 152.0, 304.0, 252.0], "confidence": 0.85, "class_id": 2, "class_name": "car"}]
        tracks2 = self.engine.update(frame2, frame_id=2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, first_id)
        self.assertEqual(self.engine.unique_track_count, 1)

    def test_10_new_track_creation_on_disjoint_object(self) -> None:
        """Test that a new distant object receives a distinct persistent ID upon confirmation."""
        # Frame 1: Object 1 detected and activated
        frame1 = [{"bbox_xyxy": [50.0, 50.0, 150.0, 150.0], "confidence": 0.9, "class_id": 2, "class_name": "car"}]
        tracks1 = self.engine.update(frame1)
        id1 = tracks1[0].track_id

        # Frame 2: Object 1 moves, Object 2 appears (starts as unconfirmed tracklet)
        frame2 = [
            {"bbox_xyxy": [52.0, 51.0, 152.0, 151.0], "confidence": 0.88, "class_id": 2, "class_name": "car"},
            {"bbox_xyxy": [450.0, 300.0, 580.0, 420.0], "confidence": 0.92, "class_id": 7, "class_name": "truck"},
        ]
        tracks2 = self.engine.update(frame2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, id1)

        # Frame 3: Both objects continue; Object 2 is confirmed and active
        frame3 = [
            {"bbox_xyxy": [54.0, 52.0, 154.0, 152.0], "confidence": 0.87, "class_id": 2, "class_name": "car"},
            {"bbox_xyxy": [452.0, 301.0, 582.0, 421.0], "confidence": 0.91, "class_id": 7, "class_name": "truck"},
        ]
        tracks3 = self.engine.update(frame3)
        self.assertEqual(len(tracks3), 2)
        track_map = {t.track_id: t.class_name for t in tracks3}
        self.assertIn(id1, track_map)
        self.assertEqual(track_map[id1], "car")
        self.assertEqual(self.engine.unique_track_count, 2)

    def test_11_lost_track_retention_and_recovery(self) -> None:
        """Test that track survives temporary 1-frame disappearance within track_buffer."""
        frame1 = [{"bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "confidence": 0.9, "class_id": 2, "class_name": "car"}]
        tracks1 = self.engine.update(frame1)
        orig_id = tracks1[0].track_id

        # Frame 2: occluded / zero detection
        tracks2 = self.engine.update([])
        self.assertEqual(len(tracks2), 0)

        # Frame 3: reappears near Kalman predicted position
        frame3 = [{"bbox_xyxy": [102.0, 101.0, 202.0, 201.0], "confidence": 0.85, "class_id": 2, "class_name": "car"}]
        tracks3 = self.engine.update(frame3)
        self.assertEqual(len(tracks3), 1)
        self.assertEqual(tracks3[0].track_id, orig_id)

    # --------------------------------------------------------------------------
    # 4. State Management, Reset & Session Isolation
    # --------------------------------------------------------------------------

    def test_12_tracker_reset_lifecycle(self) -> None:
        """Test that reset() clears all internal track pools and restarts track IDs."""
        dets = [{"bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "confidence": 0.9, "class_id": 2}]
        self.engine.update(dets)
        self.assertEqual(self.engine.frame_count, 1)
        self.assertEqual(self.engine.unique_track_count, 1)

        self.engine.reset()
        self.assertEqual(self.engine.frame_count, 0)
        self.assertEqual(self.engine.unique_track_count, 0)

        # After reset, first track ID should start back at 1
        tracks_after = self.engine.update(dets)
        self.assertEqual(tracks_after[0].track_id, 1)

    def test_13_session_isolation_between_engines(self) -> None:
        """Test that independent ByteTrackEngine instances maintain isolated states."""
        engine_a = ByteTrackEngine()
        engine_b = ByteTrackEngine()

        dets_a = [{"bbox_xyxy": [50.0, 50.0, 150.0, 150.0], "confidence": 0.9, "class_id": 2}]
        dets_b = [{"bbox_xyxy": [300.0, 300.0, 400.0, 400.0], "confidence": 0.9, "class_id": 7}]

        tracks_a = engine_a.update(dets_a)
        tracks_b = engine_b.update(dets_b)

        self.assertEqual(tracks_a[0].track_id, 1)
        self.assertEqual(tracks_b[0].track_id, 1)
        self.assertEqual(engine_a.frame_count, 1)
        self.assertEqual(engine_b.frame_count, 1)

    # --------------------------------------------------------------------------
    # 5. Invalid Input & Error Trapping
    # --------------------------------------------------------------------------

    def test_14_invalid_bbox_format_raises_error(self) -> None:
        """Test that non-4-element bounding box raises InvalidDetectionInputError."""
        invalid_dets = [{"bbox_xyxy": [100.0, 200.0], "confidence": 0.8, "class_id": 2}]
        with self.assertRaises(InvalidDetectionInputError):
            self.engine.update(invalid_dets)

    def test_15_invalid_confidence_raises_error(self) -> None:
        """Test that out-of-range confidence scores raise InvalidDetectionInputError."""
        invalid_dets = [{"bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "confidence": 1.5, "class_id": 2}]
        with self.assertRaises(InvalidDetectionInputError):
            self.engine.update(invalid_dets)

    def test_16_raw_tuple_array_detection_input(self) -> None:
        """Test support for raw [x1, y1, x2, y2, conf, cls] array input."""
        raw_dets = [[120.0, 140.0, 220.0, 240.0, 0.89, 2]]
        tracks = self.engine.update(raw_dets)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

    def test_17_track_record_to_dict_serialization(self) -> None:
        """Test TrackRecord serialization to dictionary format."""
        rec = TrackRecord(
            track_id=42,
            class_id=7,
            class_name="truck",
            confidence=0.87654,
            bbox_xyxy=[100.123, 200.456, 300.789, 400.012],
            detection_index=3,
            track_status="active",
        )
        d = rec.to_dict()
        self.assertEqual(d["track_id"], 42)
        self.assertEqual(d["class_id"], 7)
        self.assertEqual(d["confidence"], 0.8765)
        self.assertEqual(d["bbox_xyxy"], [100.12, 200.46, 300.79, 400.01])
        self.assertEqual(d["detection_index"], 3)


if __name__ == "__main__":
    unittest.main()
