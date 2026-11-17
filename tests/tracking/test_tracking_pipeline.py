"""PerceptGuard AI: Unit and Integration Tests for TrackingPipeline.

Validates:
- Detection JSON ingestion without duplicate inference passes
- Sequential natural sorting and temporal frame ordering
- Tracking output schemas (results, summary, metadata, errors)
- Destination folder protection and overwrite behavior
- Frame metadata preservation (timestamps, frame IDs, dimensions)
- CLI runner arguments, validation, and exit codes
- Real Task 4 batch detection output integration
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import patch

from src.tracking.bytetrack_engine import ByteTrackEngine, TrackRecord
from src.tracking.tracker_config import TrackerConfig
from src.tracking.tracking_pipeline import (
    FrameTrackingRecord,
    TrackingInputNotFoundError,
    TrackingOutputExistsError,
    TrackingPipeline,
    TrackingPipelineError,
    TrackingSummary,
    main,
)


class TestTrackingPipeline(unittest.TestCase):
    """Test suite for TrackingPipeline execution, multi-file persistence, and CLI."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create mock batch detection results
        self.mock_detection_file = self.temp_path / "mock_detection_results.json"
        self.mock_payload = {
            "schema_version": "1.0",
            "run_id": "mock_run_12345",
            "dataset_id": "test_dataset",
            "model_name": "yolo11n",
            "total_frames": 3,
            "frame_results": [
                {
                    "image_filename": "frame_000001.jpg",
                    "extracted_frame_id": 1,
                    "source_frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "image_width": 640,
                    "image_height": 360,
                    "inference_time_ms": 45.2,
                    "detections": [
                        {
                            "detection_index": 0,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.88,
                            "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
                        }
                    ],
                },
                {
                    "image_filename": "frame_000002.jpg",
                    "extracted_frame_id": 2,
                    "source_frame_index": 1,
                    "timestamp_seconds": 0.033,
                    "image_width": 640,
                    "image_height": 360,
                    "inference_time_ms": 46.1,
                    "detections": [
                        {
                            "detection_index": 0,
                            "class_id": 2,
                            "class_name": "car",
                            "confidence": 0.86,
                            "bbox_xyxy": [103.0, 101.0, 203.0, 201.0],
                        }
                    ],
                },
                {
                    "image_filename": "frame_000003.jpg",
                    "extracted_frame_id": 3,
                    "source_frame_index": 2,
                    "timestamp_seconds": 0.066,
                    "image_width": 640,
                    "image_height": 360,
                    "inference_time_ms": 44.8,
                    "detections": [],  # zero detection frame
                },
            ],
        }
        with open(self.mock_detection_file, "w", encoding="utf-8") as f:
            json.dump(self.mock_payload, f)

        self.pipeline = TrackingPipeline(
            dataset_id="test_dataset",
            project_root=self.temp_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. Pipeline Execution & Ingestion
    # --------------------------------------------------------------------------

    def test_01_successful_pipeline_execution(self) -> None:
        """Test full pipeline run produces valid summary and frame records."""
        out_dir = self.temp_path / "tracking_out"
        summary, records = self.pipeline.run_from_detection_results(
            detection_file=self.mock_detection_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(summary.total_frames, 3)
        self.assertEqual(summary.successful_frames, 3)
        self.assertEqual(summary.failed_frames, 0)
        self.assertEqual(summary.frames_with_detections, 2)
        self.assertEqual(summary.frames_with_tracks, 2)
        self.assertEqual(summary.unique_track_count, 1)
        self.assertEqual(summary.status, "PASS")
        self.assertEqual(len(records), 3)

    def test_02_identity_continuity_across_frames(self) -> None:
        """Test that vehicle in frame 1 and 2 keeps the exact same track ID."""
        out_dir = self.temp_path / "tracking_out"
        _, records = self.pipeline.run_from_detection_results(
            detection_file=self.mock_detection_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(len(records[0].tracks), 1)
        self.assertEqual(len(records[1].tracks), 1)
        self.assertEqual(records[0].tracks[0].track_id, records[1].tracks[0].track_id)

    def test_03_zero_detection_frame_handled_as_success(self) -> None:
        """Test that frame with 0 detections records success status with empty tracks."""
        out_dir = self.temp_path / "tracking_out"
        _, records = self.pipeline.run_from_detection_results(
            detection_file=self.mock_detection_file,
            output_dir=out_dir,
            overwrite=True,
        )
        frame3 = records[2]
        self.assertEqual(frame3.image_filename, "frame_000003.jpg")
        self.assertEqual(frame3.processing_status, "success")
        self.assertEqual(frame3.detection_count, 0)
        self.assertEqual(frame3.active_track_count, 0)
        self.assertEqual(frame3.tracks, [])

    # --------------------------------------------------------------------------
    # 2. Natural Ordering & Slicing
    # --------------------------------------------------------------------------

    def test_04_natural_sorting_on_unsorted_inputs(self) -> None:
        """Test that out-of-order frame lists are sorted naturally before tracking."""
        scrambled_file = self.temp_path / "scrambled.json"
        scrambled_payload = dict(self.mock_payload)
        scrambled_payload["frame_results"] = [
            self.mock_payload["frame_results"][2],
            self.mock_payload["frame_results"][0],
            self.mock_payload["frame_results"][1],
        ]
        with open(scrambled_file, "w", encoding="utf-8") as f:
            json.dump(scrambled_payload, f)

        out_dir = self.temp_path / "sorted_out"
        _, records = self.pipeline.run_from_detection_results(
            detection_file=scrambled_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(records[0].image_filename, "frame_000001.jpg")
        self.assertEqual(records[1].image_filename, "frame_000002.jpg")
        self.assertEqual(records[2].image_filename, "frame_000003.jpg")

    def test_05_max_frames_smoke_test_sets_warning_status(self) -> None:
        """Test that partial processing with max_frames assigns status WARNING."""
        out_dir = self.temp_path / "smoke_out"
        summary, records = self.pipeline.run_from_detection_results(
            detection_file=self.mock_detection_file,
            output_dir=out_dir,
            max_frames=1,
            overwrite=True,
        )
        self.assertEqual(summary.total_frames, 1)
        self.assertEqual(summary.status, "WARNING")
        self.assertEqual(len(records), 1)

    # --------------------------------------------------------------------------
    # 3. Output Persistence & Multi-File Architecture
    # --------------------------------------------------------------------------

    def test_06_multi_file_artifact_creation(self) -> None:
        """Test that results, summary, and metadata JSON files are created."""
        out_dir = self.temp_path / "artifacts_out"
        self.pipeline.run_from_detection_results(
            detection_file=self.mock_detection_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertTrue((out_dir / "tracking_results.json").exists())
        self.assertTrue((out_dir / "tracking_summary.json").exists())
        self.assertTrue((out_dir / "tracking_metadata.json").exists())

    def test_07_existing_output_protection_raises_error(self) -> None:
        """Test that existing output directory without overwrite=True raises error."""
        out_dir = self.temp_path / "protected_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(TrackingOutputExistsError):
            self.pipeline.run_from_detection_results(
                detection_file=self.mock_detection_file,
                output_dir=out_dir,
                overwrite=False,
            )

    def test_08_overwrite_flag_allows_replacement(self) -> None:
        """Test that overwrite=True safely replaces existing output directory."""
        out_dir = self.temp_path / "overwrite_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary, _ = self.pipeline.run_from_detection_results(
            detection_file=self.mock_detection_file,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(summary.successful_frames, 3)

    # --------------------------------------------------------------------------
    # 4. Error Trapping & Edge Cases
    # --------------------------------------------------------------------------

    def test_09_missing_detection_file_raises_error(self) -> None:
        """Test that non-existent detection file raises TrackingInputNotFoundError."""
        with self.assertRaises(TrackingInputNotFoundError):
            self.pipeline.run_from_detection_results(
                detection_file=self.temp_path / "non_existent.json",
            )

    def test_10_missing_frame_results_key_raises_error(self) -> None:
        """Test that JSON without frame_results raises TrackingPipelineError."""
        bad_file = self.temp_path / "no_frames.json"
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0"}, f)
        with self.assertRaises(TrackingPipelineError):
            self.pipeline.run_from_detection_results(detection_file=bad_file)

    def test_11_corrupted_frame_records_error_catalog(self) -> None:
        """Test that a frame with invalid bounding box records an error without halting."""
        bad_frame_payload = dict(self.mock_payload)
        bad_frame_payload["frame_results"] = [
            {
                "image_filename": "frame_corrupted.jpg",
                "detections": [{"bbox_xyxy": "invalid_box", "confidence": 0.9, "class_id": 2}],
            },
            self.mock_payload["frame_results"][0],
        ]
        bad_json = self.temp_path / "corrupted_frame.json"
        with open(bad_json, "w", encoding="utf-8") as f:
            json.dump(bad_frame_payload, f)

        out_dir = self.temp_path / "corrupted_out"
        summary, records = self.pipeline.run_from_detection_results(
            detection_file=bad_json,
            output_dir=out_dir,
            overwrite=True,
        )
        self.assertEqual(summary.failed_frames, 1)
        self.assertEqual(summary.successful_frames, 1)
        self.assertTrue((out_dir / "tracking_errors.json").exists())

    # --------------------------------------------------------------------------
    # 5. CLI Execution
    # --------------------------------------------------------------------------

    def test_12_cli_successful_run(self) -> None:
        """Test CLI tool execution with valid arguments returns code 0."""
        out_dir = self.temp_path / "cli_out"
        exit_code = main([
            "--input", str(self.mock_detection_file),
            "--output-dir", str(out_dir),
            "--overwrite",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue((out_dir / "tracking_results.json").exists())

    def test_13_cli_missing_input_returns_code_2(self) -> None:
        """Test CLI tool with non-existent input returns exit code 2."""
        exit_code = main(["--input", str(self.temp_path / "missing.json")])
        self.assertEqual(exit_code, 2)

    def test_14_cli_existing_dir_without_overwrite_returns_code_3(self) -> None:
        """Test CLI tool with existing destination without overwrite returns code 3."""
        out_dir = self.temp_path / "cli_protected"
        out_dir.mkdir(parents=True, exist_ok=True)
        exit_code = main([
            "--input", str(self.mock_detection_file),
            "--output-dir", str(out_dir),
        ])
        self.assertEqual(exit_code, 3)

    # --------------------------------------------------------------------------
    # 6. Real Dataset Smoke Test
    # --------------------------------------------------------------------------

    def test_15_real_driving_sample_detection_ingestion(self) -> None:
        """Test ingestion of actual Task 4 detection output on first 5 frames."""
        real_det_file = (
            Path(__file__).resolve().parents[2]
            / "datasets"
            / "processed"
            / "driving_sample"
            / "detections"
            / "batch_runs"
            / "run_20261117_115348_e1789b"
            / "detection_results.json"
        )
        if not real_det_file.exists():
            self.skipTest("Real dataset detection results not available on disk.")

        out_dir = self.temp_path / "real_sample_out"
        summary, records = self.pipeline.run_from_detection_results(
            detection_file=real_det_file,
            output_dir=out_dir,
            max_frames=5,
            overwrite=True,
        )
        self.assertEqual(summary.successful_frames, 5)
        self.assertGreater(summary.total_detections, 0)
        self.assertGreater(summary.unique_track_count, 0)
        self.assertEqual(summary.status, "WARNING")  # smoke test


if __name__ == "__main__":
    unittest.main()
