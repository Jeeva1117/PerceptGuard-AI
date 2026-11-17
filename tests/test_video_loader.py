"""PerceptGuard-AI: Unit and Integration Test Suite for VideoLoader.

Covers:
1. Valid video loading & metadata retrieval.
2. Missing video file handling (VideoNotFoundError).
3. Directory passed as video path handling (InvalidVideoPathError).
4. Corrupted / non-video file handling.
5. Metadata validation (FPS, total_frames, dimensions, duration).
6. Resource release & is_opened status.
7. Context manager protocol (__enter__, __exit__).
8. Real autonomous-driving video integration test (BDD100K sample).
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from src.video_processing.video_loader import (
    VideoLoader,
    VideoMetadata,
    VideoNotFoundError,
    InvalidVideoPathError,
    VideoOpenError,
    InvalidVideoMetadataError,
)


class TestVideoLoader(unittest.TestCase):
    """Test suite covering unit and integration testing for VideoLoader."""

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
            # Create a simple frame with changing intensity gradient
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
        opened = loader.open()
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

    def test_06_resource_release_is_idempotent(self) -> None:
        """Verify that release() cleans up cleanly and can be called repeatedly without error."""
        loader = VideoLoader(self.temp_video_path)
        loader.open()
        self.assertTrue(loader.is_opened())
        loader.release()
        self.assertFalse(loader.is_opened())
        # Secondary release call should not raise exception
        loader.release()
        self.assertFalse(loader.is_opened())

    def test_07_context_manager_lifecycle(self) -> None:
        """Verify that context manager automatically opens and releases the resource."""
        loader = VideoLoader(self.temp_video_path)
        self.assertFalse(loader.is_opened())
        with loader as ctx:
            self.assertTrue(ctx.is_opened())
            meta = ctx.get_metadata()
            self.assertTrue(meta.is_valid)
        self.assertFalse(loader.is_opened())

    def test_08_real_bdd100k_video_integration(self) -> None:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
