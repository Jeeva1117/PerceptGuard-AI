"""PerceptGuard AI: Comprehensive Road User and Motorcycle Tracking Validation Suite.

Validates:
- Multi-class road user tracking: car, motorcycle, truck, bus, bicycle, pedestrian
- Motorcycle identity persistence across low-confidence dips (ByteTrack Stage 2 recovery)
- Semantic class stability (no random flipping to car or numeric label)
- Temporary occlusion and re-identification (LOST -> REIDENTIFIED transition)
- Session isolation between videos
"""

import unittest
from typing import Any, Dict, List
import numpy as np

from src.tracking.bytetrack_engine import (
    ByteTrackEngine,
    DEFAULT_ROAD_CLASSES,
    TrackLifecycle,
    TrackRecord,
)
from src.tracking.tracker_config import TrackerConfig


class TestMotorcycleAndRoadUsers(unittest.TestCase):
    """Test suite for motorcycle, vulnerable road user, and multi-class tracking."""

    def setUp(self) -> None:
        self.config = TrackerConfig(
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
        )
        self.engine = ByteTrackEngine(config=self.config)

    def test_01_default_road_classes_mapping(self) -> None:
        """Verify default road classes map standard COCO indices accurately."""
        self.assertEqual(DEFAULT_ROAD_CLASSES[0], "person")
        self.assertEqual(DEFAULT_ROAD_CLASSES[1], "bicycle")
        self.assertEqual(DEFAULT_ROAD_CLASSES[2], "car")
        self.assertEqual(DEFAULT_ROAD_CLASSES[3], "motorcycle")
        self.assertEqual(DEFAULT_ROAD_CLASSES[5], "bus")
        self.assertEqual(DEFAULT_ROAD_CLASSES[7], "truck")

    def test_02_motorcycle_detection_ingestion_and_class_resolution(self) -> None:
        """Test motorcycle detection ingestion with explicit and inferred class names."""
        # Frame 1: Explicit motorcycle detection
        dets_frame1 = [
            {
                "bbox_xyxy": [200.0, 300.0, 260.0, 420.0],
                "confidence": 0.88,
                "class_id": 3,
                "class_name": "motorcycle",
            }
        ]
        tracks = self.engine.update(dets_frame1, frame_id="frame_000001.jpg")
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].class_id, 3)
        self.assertEqual(tracks[0].class_name, "motorcycle")
        self.assertEqual(tracks[0].track_lifecycle, TrackLifecycle.TENTATIVE.value)

    def test_03_motorcycle_track_continuity_and_confidence_dip(self) -> None:
        """Test that motorcycle maintains track ID even when confidence dips into Stage 2 (0.10 - 0.24)."""
        # Frame 1: High confidence detection
        tracks1 = self.engine.update(
            [{"bbox_xyxy": [150.0, 200.0, 210.0, 320.0], "confidence": 0.82, "class_id": 3, "class_name": "motorcycle"}],
            frame_id=1,
        )
        self.assertEqual(len(tracks1), 1)
        assigned_id = tracks1[0].track_id

        # Frame 2: Moderate confidence, moving forward
        tracks2 = self.engine.update(
            [{"bbox_xyxy": [155.0, 210.0, 215.0, 330.0], "confidence": 0.70, "class_id": 3, "class_name": "motorcycle"}],
            frame_id=2,
        )
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, assigned_id)

        # Frame 3: Confirmed state (3rd observation)
        tracks3 = self.engine.update(
            [{"bbox_xyxy": [160.0, 220.0, 220.0, 340.0], "confidence": 0.65, "class_id": 3, "class_name": "motorcycle"}],
            frame_id=3,
        )
        self.assertEqual(len(tracks3), 1)
        self.assertEqual(tracks3[0].track_id, assigned_id)
        self.assertEqual(tracks3[0].track_lifecycle, TrackLifecycle.CONFIRMED.value)

        # Frame 4: Severe confidence dip to 0.18 (e.g. motion blur or partial occlusion)
        tracks4 = self.engine.update(
            [{"bbox_xyxy": [165.0, 230.0, 225.0, 350.0], "confidence": 0.18, "class_id": 3, "class_name": "motorcycle"}],
            frame_id=4,
        )
        self.assertEqual(len(tracks4), 1)
        self.assertEqual(tracks4[0].track_id, assigned_id, "Track ID must persist during Stage 2 association")
        self.assertEqual(tracks4[0].class_name, "motorcycle", "Class label must remain motorcycle")

    def test_04_motorcycle_class_stability(self) -> None:
        """Verify motorcycle class does not flip when detection lacks explicit class_name."""
        # Frame 1: Motorcycle with raw array [x1, y1, x2, y2, conf, cls_id=3]
        raw_det = [[100.0, 100.0, 160.0, 220.0, 0.75, 3]]
        tracks1 = self.engine.update(raw_det, frame_id=1)
        self.assertEqual(len(tracks1), 1)
        self.assertEqual(tracks1[0].class_name, "motorcycle")

        # Frame 2: Subsequent frame with lower score
        raw_det2 = [[104.0, 105.0, 164.0, 225.0, 0.40, 3]]
        tracks2 = self.engine.update(raw_det2, frame_id=2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].class_name, "motorcycle")

    def test_05_multi_road_user_concurrent_tracking(self) -> None:
        """Test concurrent tracking of motorcycle, car, truck, and pedestrian."""
        dets = [
            {"bbox_xyxy": [50.0, 100.0, 120.0, 220.0], "confidence": 0.85, "class_id": 2, "class_name": "car"},
            {"bbox_xyxy": [200.0, 150.0, 250.0, 250.0], "confidence": 0.78, "class_id": 3, "class_name": "motorcycle"},
            {"bbox_xyxy": [350.0, 80.0, 500.0, 300.0], "confidence": 0.90, "class_id": 7, "class_name": "truck"},
            {"bbox_xyxy": [550.0, 200.0, 580.0, 290.0], "confidence": 0.72, "class_id": 0, "class_name": "person"},
        ]
        tracks = self.engine.update(dets, frame_id=1)
        self.assertEqual(len(tracks), 4)

        classes_found = {t.class_name for t in tracks}
        self.assertIn("car", classes_found)
        self.assertIn("motorcycle", classes_found)
        self.assertIn("truck", classes_found)
        self.assertIn("person", classes_found)

    def test_06_motorcycle_reidentification_after_occlusion(self) -> None:
        """Test that a motorcycle lost for 1 frame is re-identified when it reappears."""
        # Frames 1-3: Establish track
        for f in range(3):
            tracks = self.engine.update(
                [{"bbox_xyxy": [100.0 + f * 5, 100.0 + f * 5, 150.0 + f * 5, 200.0 + f * 5], "confidence": 0.85, "class_id": 3, "class_name": "motorcycle"}],
                frame_id=f + 1,
            )
        self.assertEqual(len(tracks), 1)
        original_id = tracks[0].track_id

        # Frame 4: Occluded (0 detections)
        tracks_empty = self.engine.update([], frame_id=4)
        self.assertEqual(len(tracks_empty), 0)

        # Frame 5: Re-emerges near predicted Kalman location
        tracks_reemerged = self.engine.update(
            [{"bbox_xyxy": [120.0, 120.0, 170.0, 220.0], "confidence": 0.80, "class_id": 3, "class_name": "motorcycle"}],
            frame_id=5,
        )
        self.assertEqual(len(tracks_reemerged), 1)
        self.assertEqual(tracks_reemerged[0].track_id, original_id)
        self.assertEqual(tracks_reemerged[0].track_lifecycle, TrackLifecycle.REIDENTIFIED.value)
        self.assertEqual(tracks_reemerged[0].class_name, "motorcycle")

    def test_07_engine_reset_session_isolation(self) -> None:
        """Verify engine reset wipes all tracks and classes for clean multi-video operation."""
        self.engine.update(
            [{"bbox_xyxy": [100.0, 100.0, 150.0, 200.0], "confidence": 0.85, "class_id": 3, "class_name": "motorcycle"}],
            frame_id=1,
        )
        self.assertEqual(self.engine.unique_track_count, 1)

        self.engine.reset()
        self.assertEqual(self.engine.frame_count, 0)
        self.assertEqual(self.engine.unique_track_count, 0)
        self.assertEqual(len(self.engine._track_classes), 0)


if __name__ == "__main__":
    unittest.main()
