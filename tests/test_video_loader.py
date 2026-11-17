"""PerceptGuard-AI: Unit and Integration Test Suite for VideoLoader.

Covers:
1. Valid video loading & metadata retrieval.
2. Missing video file handling (VideoNotFoundError).
3. Directory passed as video path handling (InvalidVideoPathError).
4. Corrupted / non-video file handling (VideoOpenError).
5. Zero-byte video handling (EmptyVideoError / CorruptedVideoError).
6. Failed first-frame decoding handling (FirstFrameDecodeError / CorruptedVideoError).
7. Metadata validation (FPS, total_frames, dimensions, duration).
8. Resource release & is_opened status idempotency.
9. Context manager protocol (__enter__, __exit__).
10. Programmatic verify() method on valid and invalid streams.
11. Real autonomous-driving video integration test (BDD100K sample).
12. CLI verification runner (verify_video_file).
"""

from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from src.video_processing.video_loader import (
    VideoLoader,
    VideoMetadata,
    VideoProcessingError,
    VideoNotFoundError,
    InvalidVideoPathError,
    VideoOpenError,
    CorruptedVideoError,
    EmptyVideoError,
    FirstFrameDecodeError,
    InvalidVideoMetadataError,
    verify_video_file,
)


class TestVideoLoader(unittest.TestCase):
    """Comprehensive test suite for VideoLoader and video verification."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a tiny, temporary synthetic video for deterministic unit tests."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_video_path = Path(cls.temp_dir.name) / "synthetic_test_video.mp4"

        # Parameters for synthetic test clip: 30 frames, 128x128, 30 FPS
        cls.syn_width = 128
        cls.syn_height = 128
        cls.syn_fps = 30.0
        cls.syn_frames = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(cls.temp_video_path),
            fourcc,
            cls.syn_fps,
            (cls.syn_width, cls.syn_height),
        )

        for i in range(cls.syn_frames):
            frame = np.full((cls.syn_height, cls.syn_width, 3), fill_value=i * 5, dtype=np.uint8)
            writer.write(frame)

        writer.release()

        # Path to real acquired BDD100K driving sample
        cls.real_video_path = (
            Path(__file__).resolve().parent.parent / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary test artifacts."""
        cls.temp_dir.cleanup()

    def test_01_valid_synthetic_video_loading(self) -> None:
        """Verify that a valid video can be opened and confirms is_opened()."""
        loader = VideoLoader(self.temp_video_path)
        self.assertFalse(loader.is_opened())
        opened = loader.open(verify_first_frame=True)
        self.assertTrue(opened)
        self.assertTrue(loader.is_opened())
        loader.release()
        self.assertFalse(loader.is_opened())

    def test_02_metadata_retrieval(self) -> None:
        """Verify that extracted metadata fields match ground-truth video parameters."""
        with VideoLoader(self.temp_video_path) as loader:
            meta: VideoMetadata = loader.get_metadata()
            self.assertEqual(meta.filename, self.temp_video_path.name)
            self.assertEqual(meta.width, self.syn_width)
            self.assertEqual(meta.height, self.syn_height)
            self.assertAlmostEqual(meta.fps, self.syn_fps, delta=0.5)
            self.assertEqual(meta.total_frames, self.syn_frames)
            self.assertAlmostEqual(meta.duration_seconds, self.syn_frames / self.syn_fps, delta=0.1)
            self.assertTrue(meta.is_valid)
            self.assertTrue(len(meta.backend) > 0)
            self.assertIsInstance(meta.to_dict(), dict)

    def test_03_missing_video_file_raises_error(self) -> None:
        """Verify that a non-existent path raises VideoNotFoundError."""
        non_existent_path = Path("datasets/raw/non_existent_random_sample.mp4")
        loader = VideoLoader(non_existent_path)
        with self.assertRaises(VideoNotFoundError):
            loader.open()

    def test_04_directory_path_raises_error(self) -> None:
        """Verify that passing a directory instead of a file raises InvalidVideoPathError."""
        dir_path = Path(self.temp_dir.name)
        loader = VideoLoader(dir_path)
        with self.assertRaises(InvalidVideoPathError):
            loader.open()

    def test_05_corrupted_or_non_video_file_raises_error(self) -> None:
        """Verify that opening a corrupted or non-video file raises VideoOpenError."""
        corrupt_file = Path(self.temp_dir.name) / "corrupt_fake.mp4"
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("This is plain text pretending to be an MP4 video file.")

        loader = VideoLoader(corrupt_file)
        with self.assertRaises(VideoOpenError):
            loader.open()

    def test_06_zero_byte_video_raises_error(self) -> None:
        """Verify that a 0-byte video file raises EmptyVideoError (subclass of CorruptedVideoError)."""
        zero_byte_file = Path(self.temp_dir.name) / "empty_zero_byte.mp4"
        zero_byte_file.touch()

        loader = VideoLoader(zero_byte_file)
        with self.assertRaises(EmptyVideoError):
            loader.open()

        # Confirm inheritance hierarchy
        with self.assertRaises(CorruptedVideoError):
            loader.open()

        with self.assertRaises(VideoOpenError):
            loader.open()

    def test_07_failed_first_frame_decoding_raises_error(self) -> None:
        """Verify that failure to decode the first frame raises FirstFrameDecodeError."""
        loader = VideoLoader(self.temp_video_path)

        # Mock cv2.VideoCapture so that isOpened() returns True, but read() returns (False, None)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with self.assertRaises(FirstFrameDecodeError):
                loader.open(verify_first_frame=True)

            # Confirm inheritance hierarchy
            with self.assertRaises(CorruptedVideoError):
                loader.open(verify_first_frame=True)

    def test_08_resource_release_is_idempotent(self) -> None:
        """Verify that release() cleans up cleanly and can be called repeatedly without error."""
        loader = VideoLoader(self.temp_video_path)
        loader.open()
        self.assertTrue(loader.is_opened())
        loader.release()
        self.assertFalse(loader.is_opened())
        # Secondary release call should not raise exception
        loader.release()
        self.assertFalse(loader.is_opened())

    def test_09_context_manager_lifecycle(self) -> None:
        """Verify that context manager automatically opens and releases the resource."""
        loader = VideoLoader(self.temp_video_path)
        self.assertFalse(loader.is_opened())
        with loader as ctx:
            self.assertTrue(ctx.is_opened())
            meta = ctx.get_metadata()
            self.assertTrue(meta.is_valid)
        self.assertFalse(loader.is_opened())

    def test_10_programmatic_verify_method_success(self) -> None:
        """Verify that loader.verify() returns complete PASS telemetry for a valid video."""
        loader = VideoLoader(self.temp_video_path)
        res = loader.verify()

        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["file_exists"])
        self.assertGreater(res["file_size_bytes"], 0)
        self.assertTrue(res["video_open"])
        self.assertTrue(res["first_frame_decode"])
        self.assertEqual(res["width"], self.syn_width)
        self.assertEqual(res["height"], self.syn_height)
        self.assertEqual(res["frame_count"], self.syn_frames)
        self.assertAlmostEqual(res["fps"], self.syn_fps, delta=0.5)
        self.assertAlmostEqual(res["duration_seconds"], self.syn_frames / self.syn_fps, delta=0.1)
        self.assertIsNone(res["error_message"])

    def test_11_programmatic_verify_method_failure_modes(self) -> None:
        """Verify that loader.verify() returns FAIL on missing, empty, and corrupt inputs."""
        # Missing file
        missing_loader = VideoLoader(Path(self.temp_dir.name) / "does_not_exist.mp4")
        res_missing = missing_loader.verify()
        self.assertEqual(res_missing["status"], "FAIL")
        self.assertFalse(res_missing["file_exists"])

        # Zero-byte file
        zero_file = Path(self.temp_dir.name) / "test_zero_verify.mp4"
        zero_file.touch()
        zero_loader = VideoLoader(zero_file)
        res_zero = zero_loader.verify()
        self.assertEqual(res_zero["status"], "FAIL")
        self.assertTrue(res_zero["file_exists"])
        self.assertEqual(res_zero["file_size_bytes"], 0)

        # Corrupt file
        corrupt_file = Path(self.temp_dir.name) / "test_corrupt_verify.mp4"
        corrupt_file.write_text("invalid video stream header", encoding="utf-8")
        corrupt_loader = VideoLoader(corrupt_file)
        res_corrupt = corrupt_loader.verify()
        self.assertEqual(res_corrupt["status"], "FAIL")
        self.assertFalse(res_corrupt["video_open"])

    def test_12_real_bdd100k_video_integration(self) -> None:
        """Integration test on the actual acquired BDD100K driving sample."""
        if not self.real_video_path.is_file():
            self.skipTest(f"Real driving sample not present at '{self.real_video_path}'. Skipping integration test.")

        with VideoLoader(self.real_video_path) as loader:
            meta: VideoMetadata = loader.get_metadata()
            self.assertEqual(meta.filename, "converted_026c7465-309f6d33.mp4")
            self.assertEqual(meta.width, 720)
            self.assertEqual(meta.height, 1280)
            self.assertAlmostEqual(meta.fps, 30.09, delta=0.5)
            self.assertEqual(meta.total_frames, 1204)
            self.assertAlmostEqual(meta.duration_seconds, 40.01, delta=0.5)
            self.assertTrue(meta.is_valid)
            self.assertTrue(len(meta.codec) > 0)
            self.assertEqual(meta.backend, "FFMPEG")

    def test_13_cli_verification_runner(self) -> None:
        """Verify that verify_video_file executes and produces formatted diagnostics."""
        captured_stdout = StringIO()
        with patch("sys.stdout", captured_stdout):
            success = verify_video_file(video_path=self.temp_video_path)

        output = captured_stdout.getvalue()
        self.assertTrue(success)
        self.assertIn("VIDEO FILE: FOUND", output)
        self.assertIn("VIDEO OPEN: SUCCESS", output)
        self.assertIn("RESOLUTION: 128 x 128", output)
        self.assertIn("FIRST FRAME DECODE: SUCCESS", output)
        self.assertIn("VIDEO LOADING STATUS: PASS", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
