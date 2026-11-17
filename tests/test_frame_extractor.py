"""PerceptGuard-AI: Unit and Integration Test Suite for FrameExtractor.

Covers:
1. Valid video extraction (synthetic fixture).
2. Sequential zero-padded image naming.
3. Automatic output directory creation.
4. Configurable extraction FPS (subsampling).
5. Maximum frame limit enforcement (max_frames).
6. Deterministic timestamp calculation (t = source_idx / source_fps).
7. Independent source frame index and extracted ID preservation.
8. Duplicate frame prevention under oversampling (target_fps >= source_fps).
9. Invalid video path handling (VideoNotFoundError).
10. Failed image write handling (FrameSaveError).
11. Resource release on successful completion.
12. Resource release on failure / exception.
13. Real autonomous-driving video integration test (BDD100K sample).
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_processing.frame_extractor import (
    ExtractedFrameInfo,
    ExtractionSummary,
    FrameExtractionError,
    FrameExtractor,
    FrameSaveError,
    InvalidExtractionConfigError,
)
from src.video_processing.video_loader import (
    VideoLoader,
    VideoNotFoundError,
    VideoOpenError,
)


class TestFrameExtractor(unittest.TestCase):
    """Automated test suite for FrameExtractor."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a tiny, temporary synthetic video fixture for unit tests."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_video_path = Path(cls.temp_dir.name) / "synthetic_fixture.mp4"

        # Synthetic clip: 30 frames, 64x64 resolution, 30.0 FPS (1.0 second duration)
        cls.syn_width = 64
        cls.syn_height = 64
        cls.syn_fps = 30.0
        cls.syn_total_frames = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(cls.temp_video_path),
            fourcc,
            cls.syn_fps,
            (cls.syn_width, cls.syn_height),
        )

        for i in range(cls.syn_total_frames):
            # Synthetic gradient frame
            val = int((i / cls.syn_total_frames) * 255)
            frame = np.full((cls.syn_height, cls.syn_width, 3), val, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Real driving sample path from Day 2 Task 1
        cls.real_video_path = PROJECT_ROOT / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4"

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory and synthetic fixture."""
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        """Create a fresh temporary directory for test outputs."""
        self.test_output_dir = Path(tempfile.mkdtemp(prefix="perceptguard_test_frames_"))

    def tearDown(self) -> None:
        """Remove temporary output files."""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir, ignore_errors=True)

    def test_01_valid_synthetic_video_extraction(self) -> None:
        """Verify that synthetic video is extracted into output directory with valid summary."""
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=10.0,
        ) as extractor:
            summary = extractor.extract_frames()

        self.assertIsInstance(summary, ExtractionSummary)
        self.assertEqual(summary.extracted_frames_count, 10)
        self.assertEqual(len(summary.extracted_frames), 10)
        self.assertEqual(summary.total_source_frames, self.syn_total_frames)

        # Verify all 10 image files actually exist on disk
        saved_files = list(self.test_output_dir.glob("*.jpg"))
        self.assertEqual(len(saved_files), 10)

    def test_02_sequential_image_naming(self) -> None:
        """Verify that filenames follow strict zero-padded sequential convention."""
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=5.0,
            filename_prefix="test_frame_",
            filename_padding=4,
        ) as extractor:
            summary = extractor.extract_frames()

        # 5 FPS on 30 FPS video = 5 frames
        self.assertEqual(summary.extracted_frames_count, 5)
        expected_names = [f"test_frame_{i:04d}.jpg" for i in range(1, 6)]
        actual_names = [record.image_filename for record in summary.extracted_frames]
        self.assertEqual(actual_names, expected_names)

        for name in expected_names:
            self.assertTrue((self.test_output_dir / name).is_file())

    def test_03_output_directory_auto_creation(self) -> None:
        """Verify that nested non-existent output directories are automatically created."""
        nested_dir = self.test_output_dir / "nested" / "deep" / "frames"
        self.assertFalse(nested_dir.exists())

        with FrameExtractor(
            self.temp_video_path,
            output_dir=nested_dir,
            extraction_fps=2.0,
        ) as extractor:
            summary = extractor.extract_frames()

        self.assertTrue(nested_dir.is_dir())
        self.assertGreater(summary.extracted_frames_count, 0)
        self.assertTrue(len(list(nested_dir.glob("*.jpg"))) > 0)

    def test_04_configurable_extraction_fps(self) -> None:
        """Verify that different extraction FPS values yield mathematically expected frame counts."""
        # 1. Target FPS = 15.0 on 30.0 FPS clip -> step = 2.0 -> 15 frames
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir / "fps_15",
            extraction_fps=15.0,
        ) as extractor:
            s15 = extractor.extract_frames()
        self.assertEqual(s15.extracted_frames_count, 15)

        # 2. Target FPS = 6.0 on 30.0 FPS clip -> step = 5.0 -> 6 frames
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir / "fps_6",
            extraction_fps=6.0,
        ) as extractor:
            s6 = extractor.extract_frames()
        self.assertEqual(s6.extracted_frames_count, 6)

        # 3. Target FPS = 1.0 on 30.0 FPS clip -> step = 30.0 -> 1 frame
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir / "fps_1",
            extraction_fps=1.0,
        ) as extractor:
            s1 = extractor.extract_frames()
        self.assertEqual(s1.extracted_frames_count, 1)

    def test_05_max_frames_limit(self) -> None:
        """Verify that max_frames parameter stops extraction early."""
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=30.0,
            max_frames=4,
        ) as extractor:
            summary = extractor.extract_frames()

        self.assertEqual(summary.extracted_frames_count, 4)
        self.assertEqual(len(summary.extracted_frames), 4)

    def test_06_timestamp_calculation(self) -> None:
        """Verify timestamp calculation conforms to t = source_frame_index / source_fps."""
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=10.0,
        ) as extractor:
            summary = extractor.extract_frames()

        for record in summary.extracted_frames:
            expected_ts = record.source_frame_index / self.syn_fps
            self.assertAlmostEqual(record.timestamp_seconds, expected_ts, places=4)

    def test_07_source_frame_index_preservation(self) -> None:
        """Verify that source_frame_index is preserved independently from extracted_frame_id."""
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=5.0,
        ) as extractor:
            summary = extractor.extract_frames()

        # Step = 30 / 5 = 6.0 -> source indices: [0, 6, 12, 18, 24]
        # Extracted IDs: [1, 2, 3, 4, 5]
        expected_source_indices = [0, 6, 12, 18, 24]
        actual_source_indices = [r.source_frame_index for r in summary.extracted_frames]
        actual_extracted_ids = [r.extracted_frame_id for r in summary.extracted_frames]

        self.assertEqual(actual_source_indices, expected_source_indices)
        self.assertEqual(actual_extracted_ids, [1, 2, 3, 4, 5])

    def test_08_duplicate_frame_prevention(self) -> None:
        """Verify that requesting extraction_fps > source_fps never duplicates frames."""
        # Request 60 FPS on a 30 FPS video
        with FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=60.0,
        ) as extractor:
            summary = extractor.extract_frames()

        self.assertEqual(summary.extracted_frames_count, self.syn_total_frames)
        source_indices = [r.source_frame_index for r in summary.extracted_frames]
        # Must be strictly monotonic and equal to set
        self.assertEqual(len(source_indices), len(set(source_indices)))
        self.assertEqual(source_indices, list(range(self.syn_total_frames)))

    def test_09_invalid_video_handling(self) -> None:
        """Verify that non-existent video path raises VideoNotFoundError."""
        bogus_path = PROJECT_ROOT / "datasets" / "raw" / "non_existent_clip_12345.mp4"
        with self.assertRaises(VideoNotFoundError):
            extractor = FrameExtractor(bogus_path, output_dir=self.test_output_dir)
            extractor.extract_frames()

    def test_10_failed_image_write_handling(self) -> None:
        """Verify that saving an empty or corrupt frame buffer raises FrameSaveError."""
        extractor = FrameExtractor(self.temp_video_path, output_dir=self.test_output_dir)
        empty_frame = np.array([])
        with self.assertRaises(FrameSaveError):
            extractor.save_frame(empty_frame, self.test_output_dir / "bad_frame.jpg")

    def test_11_resource_release_after_success(self) -> None:
        """Verify that video resources are cleanly released upon successful extraction."""
        extractor = FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=5.0,
        )
        self.assertFalse(extractor._loader.is_opened())

        summary = extractor.extract_frames()
        self.assertGreater(summary.extracted_frames_count, 0)
        # Resource must be closed after extract_frames()
        self.assertFalse(extractor._loader.is_opened())

    def test_12_resource_release_after_failure(self) -> None:
        """Verify that video resources are cleanly released even when an exception occurs."""
        extractor = FrameExtractor(
            self.temp_video_path,
            output_dir=self.test_output_dir,
            extraction_fps=5.0,
        )
        # Force a failure during save_frame
        original_save = extractor.save_frame

        def broken_save(frame: np.ndarray, path: Path) -> None:
            raise FrameSaveError("Simulated filesystem I/O failure.")

        extractor.save_frame = broken_save

        with self.assertRaises(FrameSaveError):
            extractor.extract_frames()

        # Capture must be closed despite exception
        self.assertFalse(extractor._loader.is_opened())

    def test_13_real_bdd100k_video_integration(self) -> None:
        """Integration test on actual BDD100K driving sample (Task 1 dataset)."""
        if not self.real_video_path.is_file():
            self.skipTest(f"Real driving sample not found at '{self.real_video_path}'")

        real_output_dir = self.test_output_dir / "real_bdd100k_frames"

        # Extract at 5 FPS with limit of 25 frames for fast unit-test execution
        with FrameExtractor(
            self.real_video_path,
            output_dir=real_output_dir,
            extraction_fps=5.0,
            max_frames=25,
        ) as extractor:
            summary = extractor.extract_frames()

        self.assertIsInstance(summary, ExtractionSummary)
        self.assertEqual(summary.extracted_frames_count, 25)
        self.assertEqual(summary.original_fps, 30.09)
        self.assertEqual(summary.total_source_frames, 1204)
        self.assertAlmostEqual(summary.first_timestamp_seconds, 0.0, places=3)
        self.assertGreater(summary.last_timestamp_seconds, 4.0)

        # Verify disk files
        saved_files = sorted(list(real_output_dir.glob("*.jpg")))
        self.assertEqual(len(saved_files), 25)
        self.assertEqual(saved_files[0].name, "frame_000001.jpg")
        self.assertEqual(saved_files[-1].name, "frame_000025.jpg")

        # Verify dimensions of extracted image match video metadata (720x1280)
        sample_img = cv2.imread(str(saved_files[0]))
        self.assertIsNotNone(sample_img)
        self.assertEqual(sample_img.shape, (1280, 720, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
