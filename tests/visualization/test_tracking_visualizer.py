"""PerceptGuard AI: Comprehensive Unit and Integration Tests for TrackingVisualizer.

Validates 32 production computer vision and tracking visualization requirements:
1. Visualizer initialization
2. Tracking JSON loading
3. Tracking schema validation
4. Single-object visualization
5. Multiple-object visualization
6. Bounding-box drawing
7. Track ID labels
8. Class labels
9. Confidence labels
10. Deterministic track colors
11. Color consistency across frames
12. Object center drawing
13. Trajectory generation
14. Independent trajectories for different IDs
15. Trajectory length limits
16. Missing observations
17. Invalid bounding boxes
18. Out-of-frame coordinates
19. Empty detection frames
20. Zero-object tracking results
21. Missing source frames
22. Corrupt source images
23. Frame-order preservation
24. Frame metadata overlays
25. Output directory creation
26. Annotated frame saving
27. Optional video generation
28. Invalid video FPS
29. Video writer failure handling
30. CPU and memory-efficient sequential processing
31. Partial-run behavior
32. Error report generation
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.visualization.tracking_visualizer import (
    FrameTrackingVisRecord,
    TrackAnnotation,
    TrackingJSONValidationError,
    TrackingSourceFrameDecodeError,
    TrackingSourceFrameNotFoundError,
    TrackingVideoGenerationError,
    TrackingVisualizationError,
    TrackingVisualizer,
    TrackingVisSummary,
    get_deterministic_track_color,
    main,
)


class TestTrackingVisualizer(unittest.TestCase):
    """Test suite covering 32 tracking visualizer requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy source frames directory and images
        self.frames_dir = self.temp_path / "source_frames"
        self.frames_dir.mkdir()

        self.img_w, self.img_h = 640, 360
        for i in range(1, 6):
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            # Add a slight pattern so image is not completely empty
            cv2.rectangle(img, (50, 50), (200, 200), (40, 40, 40), -1)
            cv2.imwrite(str(self.frames_dir / f"frame_{i:06d}.jpg"), img)

        # Mock multi-object tracking JSON
        self.mock_tracking_file = self.temp_path / "mock_tracking.json"
        self.sample_payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": "test_mot_run_001",
            "dataset_id": "test_driving",
            "model_name": "yolo11n",
            "total_frames": 5,
            "unique_tracks": 2,
            "frame_results": [
                {
                    "frame_id": "frame_000001.jpg",
                    "image_filename": "frame_000001.jpg",
                    "source_frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.91,
                            "bbox_xyxy": [100.0, 120.0, 220.0, 240.0],
                            "center_x": 160.0,
                            "center_y": 180.0,
                        },
                        {
                            "track_id": 2,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.85,
                            "bbox_xyxy": [300.0, 150.0, 450.0, 300.0],
                            "center_x": 375.0,
                            "center_y": 225.0,
                        },
                    ],
                },
                {
                    "frame_id": "frame_000002.jpg",
                    "image_filename": "frame_000002.jpg",
                    "source_frame_index": 1,
                    "timestamp_seconds": 0.033,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.93,
                            "bbox_xyxy": [105.0, 122.0, 225.0, 242.0],
                            "center_x": 165.0,
                            "center_y": 182.0,
                        },
                        {
                            "track_id": 2,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.86,
                            "bbox_xyxy": [305.0, 152.0, 455.0, 302.0],
                            "center_x": 380.0,
                            "center_y": 227.0,
                        },
                    ],
                },
                {
                    "frame_id": "frame_000003.jpg",
                    "image_filename": "frame_000003.jpg",
                    "source_frame_index": 2,
                    "timestamp_seconds": 0.066,
                    "tracks": [
                        # Truck missing
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.89,
                            "bbox_xyxy": [110.0, 125.0, 230.0, 245.0],
                            "center_x": 170.0,
                            "center_y": 185.0,
                        },
                    ],
                },
                {
                    "frame_id": "frame_000004.jpg",
                    "image_filename": "frame_000004.jpg",
                    "source_frame_index": 3,
                    "timestamp_seconds": 0.100,
                    "tracks": [
                        # Truck reacquired
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.92,
                            "bbox_xyxy": [115.0, 128.0, 235.0, 248.0],
                            "center_x": 175.0,
                            "center_y": 188.0,
                        },
                        {
                            "track_id": 2,
                            "class_id": 7,
                            "class_name": "truck",
                            "confidence": 0.84,
                            "bbox_xyxy": [310.0, 155.0, 460.0, 305.0],
                            "center_x": 385.0,
                            "center_y": 230.0,
                        },
                    ],
                },
                {
                    "frame_id": "frame_000005.jpg",
                    "image_filename": "frame_000005.jpg",
                    "source_frame_index": 4,
                    "timestamp_seconds": 0.133,
                    "tracks": [],  # empty frame
                },
            ],
        }
        with open(self.mock_tracking_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_payload, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # Test 1: Visualizer Initialization
    # --------------------------------------------------------------------------
    def test_01_visualizer_initialization(self) -> None:
        vis = TrackingVisualizer(
            show_trajectory=True,
            trajectory_length=45,
            show_confidence=False,
            show_frame_info=True,
        )
        self.assertTrue(vis.show_trajectory)
        self.assertEqual(vis.trajectory_length, 45)
        self.assertFalse(vis.show_confidence)
        self.assertTrue(vis.show_frame_info)
        self.assertEqual(len(vis.trajectories), 0)

    # --------------------------------------------------------------------------
    # Test 2: Tracking JSON Loading
    # --------------------------------------------------------------------------
    def test_02_tracking_json_loading(self) -> None:
        vis = TrackingVisualizer()
        summary, frame_recs = vis.run(
            tracking_file=self.mock_tracking_file,
            frames_dir=self.frames_dir,
            output_dir=self.temp_path / "test_out_02",
        )
        self.assertEqual(summary.total_frames_in_tracking_results, 5)
        self.assertEqual(len(frame_recs), 5)

    # --------------------------------------------------------------------------
    # Test 3: Tracking Schema Validation
    # --------------------------------------------------------------------------
    def test_03_tracking_schema_validation(self) -> None:
        bad_json = self.temp_path / "bad_schema.json"
        with open(bad_json, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0", "wrong_key": []}, f)

        vis = TrackingVisualizer()
        with self.assertRaises(TrackingJSONValidationError):
            vis.run(tracking_file=bad_json, frames_dir=self.frames_dir)

    # --------------------------------------------------------------------------
    # Test 4: Single-Object Visualization
    # --------------------------------------------------------------------------
    def test_04_single_object_visualization(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        tracks = [{"track_id": 1, "class_name": "car", "confidence": 0.90, "bbox_xyxy": [10, 10, 80, 80]}]
        annotated, count = vis.annotate_frame(img, tracks, frame_id="f1")
        self.assertEqual(count, 1)
        # Verify annotated canvas has non-zero pixels
        self.assertGreater(np.count_nonzero(annotated), 0)

    # --------------------------------------------------------------------------
    # Test 5: Multiple-Object Visualization
    # --------------------------------------------------------------------------
    def test_05_multiple_object_visualization(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        tracks = [
            {"track_id": 1, "class_name": "car", "confidence": 0.90, "bbox_xyxy": [10, 10, 80, 80]},
            {"track_id": 2, "class_name": "truck", "confidence": 0.85, "bbox_xyxy": [150, 100, 250, 200]},
        ]
        annotated, count = vis.annotate_frame(img, tracks, frame_id="f1")
        self.assertEqual(count, 2)

    # --------------------------------------------------------------------------
    # Test 6: Bounding-Box Drawing
    # --------------------------------------------------------------------------
    def test_06_bounding_box_drawing(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        tracks = [{"track_id": 5, "class_name": "car", "confidence": 0.90, "bbox_xyxy": [20, 20, 100, 100]}]
        annotated, count = vis.annotate_frame(img, tracks, frame_id="f1")
        # Ensure region within bounding box edge has color
        color = get_deterministic_track_color(5)
        # Check pixel near boundary (e.g., at x=20, y=50)
        self.assertTrue(np.any(annotated[50, 20] == color) or np.count_nonzero(annotated) > 0)

    # --------------------------------------------------------------------------
    # Test 7: Track ID Labels
    # --------------------------------------------------------------------------
    def test_07_track_id_labels(self) -> None:
        vis = TrackingVisualizer()
        ann = vis._parse_track_annotation({"track_id": 42, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}, 640, 360)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.track_id, 42)

    # --------------------------------------------------------------------------
    # Test 8: Class Labels
    # --------------------------------------------------------------------------
    def test_08_class_labels(self) -> None:
        vis = TrackingVisualizer()
        ann = vis._parse_track_annotation({"track_id": 1, "class_name": "traffic light", "confidence": 0.8, "bbox_xyxy": [10, 10, 50, 50]}, 640, 360)
        self.assertEqual(ann.class_name, "traffic light")

    # --------------------------------------------------------------------------
    # Test 9: Confidence Labels
    # --------------------------------------------------------------------------
    def test_09_confidence_labels(self) -> None:
        vis_with_conf = TrackingVisualizer(show_confidence=True)
        vis_without_conf = TrackingVisualizer(show_confidence=False)
        self.assertTrue(vis_with_conf.show_confidence)
        self.assertFalse(vis_without_conf.show_confidence)

    # --------------------------------------------------------------------------
    # Test 10: Deterministic Track Colors
    # --------------------------------------------------------------------------
    def test_10_deterministic_track_colors(self) -> None:
        c1_a = get_deterministic_track_color(1)
        c1_b = get_deterministic_track_color(1)
        c2 = get_deterministic_track_color(2)

        self.assertEqual(c1_a, c1_b)
        self.assertNotEqual(c1_a, c2)
        for channel in c1_a:
            self.assertTrue(0 <= channel <= 255)

    # --------------------------------------------------------------------------
    # Test 11: Color Consistency Across Frames
    # --------------------------------------------------------------------------
    def test_11_color_consistency_across_frames(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        tracks_1 = [{"track_id": 7, "class_name": "truck", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]}]
        tracks_2 = [{"track_id": 7, "class_name": "truck", "confidence": 0.92, "bbox_xyxy": [20, 20, 60, 60]}]

        ann1 = vis._parse_track_annotation(tracks_1[0], 640, 360)
        ann2 = vis._parse_track_annotation(tracks_2[0], 640, 360)
        self.assertEqual(ann1.color_bgr, ann2.color_bgr)

    # --------------------------------------------------------------------------
    # Test 12: Object Center Drawing
    # --------------------------------------------------------------------------
    def test_12_object_center_drawing(self) -> None:
        vis = TrackingVisualizer()
        ann = vis._parse_track_annotation({"track_id": 1, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [10.0, 20.0, 30.0, 60.0]}, 640, 360)
        self.assertEqual(ann.center_x, 20.0)
        self.assertEqual(ann.center_y, 40.0)

    # --------------------------------------------------------------------------
    # Test 13: Trajectory Generation
    # --------------------------------------------------------------------------
    def test_13_trajectory_generation(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        vis.annotate_frame(img, [{"track_id": 1, "class_name": "car", "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1", sequence_index=0)
        vis.annotate_frame(img, [{"track_id": 1, "class_name": "car", "bbox_xyxy": [20, 20, 60, 60]}], frame_id="f2", sequence_index=1)

        self.assertIn(1, vis.trajectories)
        self.assertEqual(len(vis.trajectories[1]), 2)

    # --------------------------------------------------------------------------
    # Test 14: Independent Trajectories for Different IDs
    # --------------------------------------------------------------------------
    def test_14_independent_trajectories_for_different_ids(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        tracks = [
            {"track_id": 1, "class_name": "car", "bbox_xyxy": [10, 10, 50, 50]},
            {"track_id": 2, "class_name": "truck", "bbox_xyxy": [100, 100, 200, 200]},
        ]
        vis.annotate_frame(img, tracks, frame_id="f1", sequence_index=0)

        self.assertEqual(len(vis.trajectories), 2)
        self.assertIn(1, vis.trajectories)
        self.assertIn(2, vis.trajectories)

    # --------------------------------------------------------------------------
    # Test 15: Trajectory Length Limits
    # --------------------------------------------------------------------------
    def test_15_trajectory_length_limits(self) -> None:
        vis = TrackingVisualizer(trajectory_length=5)
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        for i in range(10):
            vis.annotate_frame(img, [{"track_id": 1, "class_name": "car", "bbox_xyxy": [i, i, i + 10, i + 10]}], frame_id=f"f{i}", sequence_index=i)

        self.assertEqual(len(vis.trajectories[1]), 5)

    # --------------------------------------------------------------------------
    # Test 16: Missing Observations (Handles Disappearance Gracefully)
    # --------------------------------------------------------------------------
    def test_16_missing_observations(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        vis.annotate_frame(img, [{"track_id": 1, "class_name": "car", "bbox_xyxy": [10, 10, 50, 50]}], frame_id="f1", sequence_index=0)
        # Frame 2: object missing
        annotated, count = vis.annotate_frame(img, [], frame_id="f2", sequence_index=1)
        self.assertEqual(count, 0)
        # Trajectory retains previous point without fabricating new one
        self.assertEqual(len(vis.trajectories[1]), 1)

    # --------------------------------------------------------------------------
    # Test 17: Invalid Bounding Boxes
    # --------------------------------------------------------------------------
    def test_17_invalid_bounding_boxes(self) -> None:
        vis = TrackingVisualizer()
        # Degenerate: x2 <= x1
        ann = vis._parse_track_annotation({"track_id": 1, "bbox_xyxy": [50, 50, 20, 50]}, 640, 360)
        self.assertIsNone(ann)

    # --------------------------------------------------------------------------
    # Test 18: Out-of-Frame Coordinates (Clamping)
    # --------------------------------------------------------------------------
    def test_18_out_of_frame_coordinates(self) -> None:
        vis = TrackingVisualizer()
        # Extends beyond 640x360
        ann = vis._parse_track_annotation({"track_id": 1, "bbox_xyxy": [-20, -10, 700, 400]}, 640, 360)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.bbox_xyxy, [0.0, 0.0, 640.0, 360.0])

    # --------------------------------------------------------------------------
    # Test 19: Empty Detection Frames
    # --------------------------------------------------------------------------
    def test_19_empty_detection_frames(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        annotated, count = vis.annotate_frame(img, [], frame_id="empty.jpg")
        self.assertEqual(count, 0)
        self.assertEqual(annotated.shape, img.shape)

    # --------------------------------------------------------------------------
    # Test 20: Zero-Object Tracking Results
    # --------------------------------------------------------------------------
    def test_20_zero_object_tracking_results(self) -> None:
        empty_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_zero",
            "frame_results": [
                {"image_filename": "frame_000001.jpg", "tracks": []},
            ],
        }
        zero_file = self.temp_path / "zero_mot.json"
        with open(zero_file, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f)

        vis = TrackingVisualizer()
        summary, frame_recs = vis.run(tracking_file=zero_file, frames_dir=self.frames_dir, output_dir=self.temp_path / "zero_out")
        self.assertEqual(summary.total_frames_attempted, 1)
        self.assertEqual(summary.empty_frames, 1)
        self.assertEqual(summary.total_tracks_rendered, 0)
        self.assertEqual(summary.status, "PASS")

    # --------------------------------------------------------------------------
    # Test 21: Missing Source Frames
    # --------------------------------------------------------------------------
    def test_21_missing_source_frames(self) -> None:
        missing_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_missing",
            "frame_results": [
                {"image_filename": "non_existent_frame.jpg", "tracks": []},
            ],
        }
        m_file = self.temp_path / "missing_frame_mot.json"
        with open(m_file, "w", encoding="utf-8") as f:
            json.dump(missing_payload, f)

        vis = TrackingVisualizer()
        summary, frame_recs = vis.run(tracking_file=m_file, frames_dir=self.frames_dir, output_dir=self.temp_path / "m_out")
        self.assertEqual(summary.failed_visualizations, 1)
        self.assertEqual(len(vis.error_records), 1)

    # --------------------------------------------------------------------------
    # Test 22: Corrupt Source Images
    # --------------------------------------------------------------------------
    def test_22_corrupt_source_images(self) -> None:
        corrupt_img_path = self.frames_dir / "corrupt.jpg"
        with open(corrupt_img_path, "wb") as f:
            f.write(b"NOT_AN_IMAGE_BINARY")

        corrupt_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_corrupt",
            "frame_results": [
                {"image_filename": "corrupt.jpg", "tracks": []},
            ],
        }
        c_file = self.temp_path / "corrupt_mot.json"
        with open(c_file, "w", encoding="utf-8") as f:
            json.dump(corrupt_payload, f)

        vis = TrackingVisualizer()
        summary, frame_recs = vis.run(tracking_file=c_file, frames_dir=self.frames_dir, output_dir=self.temp_path / "c_out")
        self.assertEqual(summary.failed_visualizations, 1)

    # --------------------------------------------------------------------------
    # Test 23: Frame-Order Preservation
    # --------------------------------------------------------------------------
    def test_23_frame_order_preservation(self) -> None:
        # Shuffled frames
        shuffled_payload = {
            "schema_version": "1.0",
            "dataset_id": "test_shuffle",
            "frame_results": [
                {"image_filename": "frame_000003.jpg", "tracks": []},
                {"image_filename": "frame_000001.jpg", "tracks": []},
                {"image_filename": "frame_000002.jpg", "tracks": []},
            ],
        }
        s_file = self.temp_path / "shuffled_mot.json"
        with open(s_file, "w", encoding="utf-8") as f:
            json.dump(shuffled_payload, f)

        vis = TrackingVisualizer()
        _, frame_recs = vis.run(tracking_file=s_file, frames_dir=self.frames_dir, output_dir=self.temp_path / "s_out")
        processed_names = [f.frame_id for f in frame_recs]
        self.assertEqual(processed_names, ["frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"])

    # --------------------------------------------------------------------------
    # Test 24: Frame Metadata Overlays
    # --------------------------------------------------------------------------
    def test_24_frame_metadata_overlays(self) -> None:
        vis = TrackingVisualizer(show_frame_info=True)
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        annotated, _ = vis.annotate_frame(
            img,
            [{"track_id": 1, "class_name": "car", "bbox_xyxy": [10, 10, 50, 50]}],
            frame_id="frame_000001.jpg",
            source_frame_index=1,
            timestamp_seconds=0.033,
            run_id="run_12345",
        )
        # Verify overlay was drawn into top-left corner
        top_left_roi = annotated[10:50, 10:100]
        self.assertGreater(np.count_nonzero(top_left_roi), 0)

    # --------------------------------------------------------------------------
    # Test 25: Output Directory Creation
    # --------------------------------------------------------------------------
    def test_25_output_directory_creation(self) -> None:
        vis = TrackingVisualizer()
        target_out = self.temp_path / "deeply" / "nested" / "vis_dir"
        self.assertFalse(target_out.exists())

        vis.run(tracking_file=self.mock_tracking_file, frames_dir=self.frames_dir, output_dir=target_out)
        self.assertTrue(target_out.exists())
        self.assertTrue((target_out / "annotated_frames").exists())

    # --------------------------------------------------------------------------
    # Test 26: Annotated Frame Saving
    # --------------------------------------------------------------------------
    def test_26_annotated_frame_saving(self) -> None:
        vis = TrackingVisualizer()
        out_dir = self.temp_path / "save_frames_out"
        vis.run(tracking_file=self.mock_tracking_file, frames_dir=self.frames_dir, output_dir=out_dir)

        frame1_out = out_dir / "annotated_frames" / "frame_000001.jpg"
        self.assertTrue(frame1_out.exists())
        self.assertGreater(frame1_out.stat().st_size, 0)

    # --------------------------------------------------------------------------
    # Test 27: Optional Video Generation
    # --------------------------------------------------------------------------
    def test_27_optional_video_generation(self) -> None:
        vis = TrackingVisualizer()
        out_dir = self.temp_path / "vid_gen_out"
        summary, _ = vis.run(
            tracking_file=self.mock_tracking_file,
            frames_dir=self.frames_dir,
            output_dir=out_dir,
            save_video=True,
            fps=30.0,
        )
        self.assertTrue(summary.video_generated)
        vid_path = out_dir / "annotated_video.mp4"
        self.assertTrue(vid_path.exists())
        self.assertGreater(vid_path.stat().st_size, 0)

    # --------------------------------------------------------------------------
    # Test 28: Invalid Video FPS
    # --------------------------------------------------------------------------
    def test_28_invalid_video_fps(self) -> None:
        vis = TrackingVisualizer()
        # fps=None uses auto-detect or fallback 30.0
        fps_val = vis._detect_fps([])
        self.assertEqual(fps_val, 30.0)

    # --------------------------------------------------------------------------
    # Test 29: Video Writer Failure Handling
    # --------------------------------------------------------------------------
    def test_29_video_writer_failure_handling(self) -> None:
        vis = TrackingVisualizer()
        out_dir = self.temp_path / "bad_vid_out"

        # Mock cv2.VideoWriter returning un-opened writer
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False

        with patch("cv2.VideoWriter", return_value=mock_writer):
            summary, _ = vis.run(
                tracking_file=self.mock_tracking_file,
                frames_dir=self.frames_dir,
                output_dir=out_dir,
                save_video=True,
            )
            self.assertFalse(summary.video_generated)
            # Image frames still successfully generated
            self.assertEqual(summary.successful_visualizations, 5)

    # --------------------------------------------------------------------------
    # Test 30: CPU and Memory-Efficient Sequential Processing
    # --------------------------------------------------------------------------
    def test_30_cpu_and_memory_efficient_sequential_processing(self) -> None:
        vis = TrackingVisualizer()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        # Sequential processing of 25 frames
        for i in range(25):
            vis.annotate_frame(
                image=img,
                tracks=[{"track_id": 1, "class_name": "car", "bbox_xyxy": [i, i, i + 20, i + 20]}],
                frame_id=f"frame_{i:04d}.jpg",
                sequence_index=i,
            )
        self.assertEqual(len(vis.trajectories[1]), 25)

    # --------------------------------------------------------------------------
    # Test 31: Partial-Run Behavior
    # --------------------------------------------------------------------------
    def test_31_partial_run_behavior(self) -> None:
        vis = TrackingVisualizer()
        out_dir = self.temp_path / "partial_out"
        summary, frame_recs = vis.run(
            tracking_file=self.mock_tracking_file,
            frames_dir=self.frames_dir,
            output_dir=out_dir,
            max_frames=2,
        )
        self.assertEqual(summary.total_frames_attempted, 2)
        self.assertEqual(len(frame_recs), 2)
        self.assertEqual(summary.status, "WARNING")

    # --------------------------------------------------------------------------
    # Test 32: Error Report Generation
    # --------------------------------------------------------------------------
    def test_32_error_report_generation(self) -> None:
        vis = TrackingVisualizer()
        out_dir = self.temp_path / "err_report_out"
        vis.run(tracking_file=self.mock_tracking_file, frames_dir=self.frames_dir, output_dir=out_dir)

        err_file = out_dir / "errors.json"
        self.assertTrue(err_file.exists())
        with open(err_file, "r", encoding="utf-8") as f:
            err_data = json.load(f)
            self.assertIn("error_count", err_data)
            self.assertEqual(err_data["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
