"""PerceptGuard-AI: Unit and Integration Test Suite for FramePreprocessor.

Covers:
1. Valid image loading and decoding.
2. Missing image handling (is_valid=False).
3. Corrupted image file handling (is_valid=False).
4. Empty (zero-byte) file handling (is_valid=False).
5. Invalid image dimensions handling (InvalidImageDimensionsError).
6. Configurable image resizing (direct scaling).
7. Aspect-ratio preservation with letterbox padding.
8. BGR to RGB color conversion.
9. Grayscale conversion.
10. Blur score calculation (Laplacian variance).
11. Brightness score calculation (mean grayscale intensity).
12. Contrast score calculation (standard deviation).
13. Output image saving to disk.
14. Failed image-write handling (ImageSaveError).
15. Original input image preservation (raw files remain identical).
16. Batch directory processing.
17. Invalid configuration parameter rejection.
18. Real-dataset integration test (Task 3 extracted driving frames).
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

from src.video_processing.frame_preprocessor import (
    CorruptedImageError,
    FramePreprocessor,
    ImageSaveError,
    InvalidColorChannelError,
    InvalidImageDimensionsError,
    InvalidPreprocessingConfigError,
    PreprocessedFrameRecord,
    PreprocessingError,
    PreprocessingSummary,
)


class TestFramePreprocessor(unittest.TestCase):
    """Test suite for FramePreprocessor module."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create temporary test fixtures and synthetic images."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.fixtures_dir = Path(cls.temp_dir.name) / "fixtures"
        cls.fixtures_dir.mkdir(parents=True, exist_ok=True)

        # 1. Valid test image: 100x200 (portrait), distinct color pattern
        cls.valid_img_path = cls.fixtures_dir / "valid_image.jpg"
        test_img = np.zeros((200, 100, 3), dtype=np.uint8)
        test_img[20:80, 20:80] = [0, 255, 0]      # Green box
        test_img[100:160, 20:80] = [0, 0, 255]    # Red box
        cv2.imwrite(str(cls.valid_img_path), test_img)

        # 2. Corrupted image: invalid bytes written with .jpg extension
        cls.corrupted_img_path = cls.fixtures_dir / "corrupted_image.jpg"
        with open(cls.corrupted_img_path, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES_123456789")

        # 3. Empty (0-byte) image
        cls.empty_img_path = cls.fixtures_dir / "empty_image.jpg"
        cls.empty_img_path.touch()

        # Path to Task 3 extracted real frames
        cls.real_frames_dir = PROJECT_ROOT / "datasets" / "processed" / "driving_sample" / "frames"

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory and test fixtures."""
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        """Create fresh output directory for each test."""
        self.output_dir = Path(tempfile.mkdtemp(prefix="perceptguard_test_preproc_"))

    def tearDown(self) -> None:
        """Remove output directory."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_01_valid_image_loading(self) -> None:
        """Verify that a valid image is decoded and validated successfully."""
        preprocessor = FramePreprocessor()
        is_valid, err, img = preprocessor.validate_frame(self.valid_img_path)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (200, 100, 3))

    def test_02_missing_image_handling(self) -> None:
        """Verify that a missing image path returns is_valid=False with meaningful error."""
        preprocessor = FramePreprocessor()
        bogus_path = self.fixtures_dir / "non_existent_file_xyz.jpg"
        is_valid, err, img = preprocessor.validate_frame(bogus_path)
        self.assertFalse(is_valid)
        self.assertIn("not found", err.lower())
        self.assertIsNone(img)

    def test_03_corrupted_image_handling(self) -> None:
        """Verify that corrupted image bytes fail validation gracefully."""
        preprocessor = FramePreprocessor()
        is_valid, err, img = preprocessor.validate_frame(self.corrupted_img_path)
        self.assertFalse(is_valid)
        self.assertIn("failed to decode", err.lower())
        self.assertIsNone(img)

    def test_04_empty_image_handling(self) -> None:
        """Verify that a zero-byte file fails validation gracefully."""
        preprocessor = FramePreprocessor()
        is_valid, err, img = preprocessor.validate_frame(self.empty_img_path)
        self.assertFalse(is_valid)
        self.assertIn("zero bytes", err.lower())
        self.assertIsNone(img)

    def test_05_invalid_image_dimensions(self) -> None:
        """Verify that attempting to resize an empty array raises InvalidImageDimensionsError."""
        preprocessor = FramePreprocessor()
        with self.assertRaises(InvalidImageDimensionsError):
            preprocessor.resize_frame(np.array([]))

    def test_06_image_resizing_direct(self) -> None:
        """Verify direct image resizing without aspect ratio preservation."""
        preprocessor = FramePreprocessor(
            target_width=320,
            target_height=180,
            preserve_aspect_ratio=False,
        )
        img = cv2.imread(str(self.valid_img_path))
        resized = preprocessor.resize_frame(img, 320, 180, preserve_aspect_ratio=False)
        self.assertEqual(resized.shape, (180, 320, 3))

    def test_07_aspect_ratio_preservation_letterbox(self) -> None:
        """Verify aspect-ratio-preserving resize with letterbox padding."""
        preprocessor = FramePreprocessor(
            target_width=300,
            target_height=200,
            preserve_aspect_ratio=True,
            padding_color=(0, 0, 0),
        )
        # Original: W=100, H=200 (aspect 1:2)
        # Target: W=300, H=200
        # Scale: min(300/100, 200/200) = 1.0 -> new_w=100, new_h=200
        # Letterbox pads left and right: (300 - 100) // 2 = 100 px each side
        img = cv2.imread(str(self.valid_img_path))
        resized = preprocessor.resize_frame(img, 300, 200, preserve_aspect_ratio=True, padding_color=(0, 0, 0))

        self.assertEqual(resized.shape, (200, 300, 3))
        # Left padding (x=0 to 99) should be black (0, 0, 0)
        self.assertTrue(np.all(resized[:, 0:90] == 0))
        # Right padding (x=210 to 299) should be black (0, 0, 0)
        self.assertTrue(np.all(resized[:, 210:300] == 0))
        # Center should contain the colored box
        self.assertTrue(np.any(resized[:, 100:200] > 0))

    def test_08_bgr_rgb_conversion(self) -> None:
        """Verify correct BGR to RGB color conversion."""
        preprocessor = FramePreprocessor()
        # BGR blue pixel: [255, 0, 0]
        bgr_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)
        rgb_pixel = preprocessor.convert_color(bgr_pixel, "RGB")
        # In RGB, blue is [0, 0, 255]
        self.assertEqual(list(rgb_pixel[0, 0]), [0, 0, 255])

    def test_09_grayscale_conversion(self) -> None:
        """Verify BGR to Grayscale conversion."""
        preprocessor = FramePreprocessor()
        # 3-channel white image
        white_bgr = np.full((10, 10, 3), 255, dtype=np.uint8)
        gray = preprocessor.convert_color(white_bgr, "GRAY")
        self.assertEqual(gray.shape, (10, 10))
        self.assertEqual(gray[0, 0], 255)

    def test_10_blur_score_calculation(self) -> None:
        """Verify blur score: sharp image has higher Laplacian variance than blurred."""
        preprocessor = FramePreprocessor()
        # Create high-frequency checkerboard (sharp edges)
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[::2, ::2] = 255
        sharp[1::2, 1::2] = 255

        # Create smoothed version
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

        sharp_score = preprocessor.calculate_blur_score(sharp)
        blur_score = preprocessor.calculate_blur_score(blurred)

        self.assertGreater(sharp_score, blur_score)
        self.assertGreater(sharp_score, 1000.0)

    def test_11_brightness_score_calculation(self) -> None:
        """Verify brightness score equals mean pixel intensity."""
        preprocessor = FramePreprocessor()
        dark_img = np.full((50, 50), 30, dtype=np.uint8)
        bright_img = np.full((50, 50), 200, dtype=np.uint8)

        self.assertAlmostEqual(preprocessor.calculate_brightness_score(dark_img), 30.0, places=2)
        self.assertAlmostEqual(preprocessor.calculate_brightness_score(bright_img), 200.0, places=2)

    def test_12_contrast_score_calculation(self) -> None:
        """Verify contrast score: uniform image has 0 contrast, split image has high contrast."""
        preprocessor = FramePreprocessor()
        uniform_img = np.full((50, 50), 128, dtype=np.uint8)
        high_contrast_img = np.zeros((50, 50), dtype=np.uint8)
        high_contrast_img[:, :25] = 255

        self.assertAlmostEqual(preprocessor.calculate_contrast_score(uniform_img), 0.0, places=2)
        self.assertGreater(preprocessor.calculate_contrast_score(high_contrast_img), 100.0)

    def test_13_output_image_saving(self) -> None:
        """Verify single frame preprocessing and saving to disk."""
        preprocessor = FramePreprocessor(
            target_width=320,
            target_height=180,
            preserve_aspect_ratio=True,
        )
        out_file = self.output_dir / "processed_test_01.jpg"
        record = preprocessor.preprocess_frame(self.valid_img_path, output_path=out_file)

        self.assertTrue(record.is_valid)
        self.assertTrue(out_file.is_file())
        self.assertGreater(out_file.stat().st_size, 0)
        self.assertEqual(record.processed_width, 320)
        self.assertEqual(record.processed_height, 180)

        # Check saved image dimensions
        saved_img = cv2.imread(str(out_file))
        self.assertEqual(saved_img.shape, (180, 320, 3))

    def test_14_failed_image_write_handling(self) -> None:
        """Verify that writing to an unwritable location raises ImageSaveError."""
        preprocessor = FramePreprocessor()
        # Invalid directory path on Windows
        bad_output_path = Path("Z:/non_existent_drive_9999/bad_folder/frame.jpg")
        with self.assertRaises(ImageSaveError):
            preprocessor.preprocess_frame(self.valid_img_path, output_path=bad_output_path)

    def test_15_original_input_image_preservation(self) -> None:
        """Verify that preprocessing does not alter or corrupt the original source image."""
        preprocessor = FramePreprocessor(target_width=160, target_height=90)
        original_hash = hash(self.valid_img_path.read_bytes())
        original_mtime = self.valid_img_path.stat().st_mtime

        out_file = self.output_dir / "output.jpg"
        preprocessor.preprocess_frame(self.valid_img_path, output_path=out_file)

        current_hash = hash(self.valid_img_path.read_bytes())
        current_mtime = self.valid_img_path.stat().st_mtime

        self.assertEqual(original_hash, current_hash)
        self.assertEqual(original_mtime, current_mtime)

    def test_16_batch_directory_processing(self) -> None:
        """Verify batch directory processing with mixed valid and invalid files."""
        batch_input_dir = self.fixtures_dir / "batch_input"
        batch_input_dir.mkdir(parents=True, exist_ok=True)

        # Copy 3 valid files and 1 corrupt file
        shutil.copy(self.valid_img_path, batch_input_dir / "frame_001.jpg")
        shutil.copy(self.valid_img_path, batch_input_dir / "frame_002.jpg")
        shutil.copy(self.valid_img_path, batch_input_dir / "frame_003.jpg")
        shutil.copy(self.corrupted_img_path, batch_input_dir / "frame_004_corrupt.jpg")

        batch_output_dir = self.output_dir / "batch_output"
        preprocessor = FramePreprocessor(target_width=320, target_height=180)
        summary = preprocessor.process_directory(batch_input_dir, batch_output_dir)

        self.assertIsInstance(summary, PreprocessingSummary)
        self.assertEqual(summary.total_frames, 4)
        self.assertEqual(summary.successful_frames, 3)
        self.assertEqual(summary.failed_frames, 1)

        saved_files = list(batch_output_dir.glob("*.jpg"))
        self.assertEqual(len(saved_files), 3)

    def test_17_invalid_configuration_handling(self) -> None:
        """Verify that negative dimensions or unsupported formats raise InvalidPreprocessingConfigError."""
        with self.assertRaises(InvalidPreprocessingConfigError):
            FramePreprocessor(target_width=-10, target_height=360)

        with self.assertRaises(InvalidPreprocessingConfigError):
            FramePreprocessor(target_width=640, target_height=0)

        with self.assertRaises(InvalidPreprocessingConfigError):
            FramePreprocessor(image_format="unsupported_bmp_xyz")

        with self.assertRaises(InvalidPreprocessingConfigError):
            FramePreprocessor(jpeg_quality=150)

    def test_18_real_dataset_frames_integration(self) -> None:
        """Integration test on actual BDD100K extracted frames from Task 3."""
        if not self.real_frames_dir.is_dir():
            self.skipTest(f"Real frames directory not found at '{self.real_frames_dir}'")

        sample_frames = sorted(list(self.real_frames_dir.glob("*.jpg")))
        if not sample_frames:
            self.skipTest("No extracted frames found in Task 3 directory.")

        # Test on the first 10 frames for fast integration execution
        test_sample = sample_frames[:10]
        test_batch_dir = self.output_dir / "real_sample_input"
        test_batch_dir.mkdir(parents=True, exist_ok=True)
        for f in test_sample:
            shutil.copy(f, test_batch_dir / f.name)

        real_out_dir = self.output_dir / "real_sample_preprocessed"
        preprocessor = FramePreprocessor(
            target_width=640,
            target_height=360,
            preserve_aspect_ratio=True,
            calculate_blur=True,
            calculate_brightness=True,
            calculate_contrast=True,
        )

        summary = preprocessor.process_directory(test_batch_dir, real_out_dir)

        self.assertEqual(summary.total_frames, 10)
        self.assertEqual(summary.successful_frames, 10)
        self.assertEqual(summary.failed_frames, 0)
        self.assertEqual(summary.target_dimensions, (640, 360))

        # Check that output dimensions match target (640x360)
        sample_out = list(real_out_dir.glob("*.jpg"))[0]
        img = cv2.imread(str(sample_out))
        self.assertEqual(img.shape, (360, 640, 3))

        # Check that quality indicators were populated
        self.assertIsNotNone(summary.mean_blur_score)
        self.assertIsNotNone(summary.mean_brightness_score)
        self.assertIsNotNone(summary.mean_contrast_score)
        self.assertGreater(summary.mean_blur_score, 0.0)
        self.assertGreater(summary.mean_brightness_score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

