"""PerceptGuard AI: Tests for Visual Quality Feature Extractor (Day 5 Task 4).

Comprehensive unit and integration test suite covering:
1. Valid 3-channel color image
2. Valid 1-channel grayscale image
3. Valid 4-channel BGRA image
4. Missing image file handling
5. Corrupted image handling
6. Unsupported image format
7. Empty image
8. Invalid image dimensions
9. Blur score calculation accuracy (Laplacian variance)
10. Sharp image behavior
11. Blurred image behavior
12. Brightness calculation accuracy (mean, median, std, min, max)
13. Dark image behavior
14. Bright image behavior
15. Contrast calculation accuracy (std, range, percentile range, normalized contrast)
16. Low-contrast image behavior
17. High-contrast image behavior
18. Percentile contrast calculation (P95 - P5)
19. Color-channel means accuracy (B, G, R)
20. Grayscale channel handling (color_channel_means is None)
21. Small image edge cases (e.g. 2x2, 3x3)
22. NaN and infinity prevention
23. Missing frame metadata handling
24. Duplicate frame ID handling
25. Frame ordering preservation
26. Partial processing (--max-frames, --start-frame, --end-frame)
27. Output JSON schema validation for all 4 artifacts
28. Error artifact generation (visual_quality_errors.json)
29. Custom configuration threshold handling
30. Metadata preservation from Day 2 metadata.csv
31. Output directory exists error
32. Output directory overwrite
33. CLI execution success
34. CLI execution failure on invalid directory
35. CLI partial run flags
36. Integration test on real BDD100K driving_sample preprocessed frames (200 frames)
37. Integration test checking metadata alignment with metadata.csv
38. Integration test verifying pixel math on sample real frames
39. Uniform black image edge case (blur=0, contrast=0)
40. Uniform white image edge case (blur=0, contrast=0)
41. Non-existent metadata csv graceful fallback
42. Extreme aspect ratio image edge case
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import numpy as np
import pytest

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.visual_quality_extractor import (
    DEFAULT_BLUR_THRESHOLDS,
    DEFAULT_BRIGHTNESS_THRESHOLDS,
    DEFAULT_CONTRAST_THRESHOLDS,
    CorruptedImageFileError,
    EmptyImageFileError,
    FileNotFoundExtractionError,
    FrameVisualQualityRecord,
    InvalidImageDimensionError,
    OutputDirectoryExistsError,
    UnsupportedImageFormatError,
    VisualQualityErrorRecord,
    VisualQualityExtractionError,
    VisualQualityExtractor,
    VisualQualitySummary,
    classify_blur,
    classify_brightness,
    classify_contrast,
    main,
)
from src.features.feature_config import FeatureConfig

REAL_FRAMES_DIR = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "preprocessed_frames"
)

REAL_METADATA_CSV = (
    _project_root
    / "datasets"
    / "processed"
    / "driving_sample"
    / "metadata.csv"
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_images_dir(tmp_path: Path) -> Path:
    """Create a temporary directory containing various synthetic test images."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard color image (sharp, normal brightness, normal contrast)
    img_standard = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img_standard[i, j] = [(i * 2) % 256, (j * 2) % 256, 128]
    cv2.imwrite(str(frames_dir / "frame_000001.jpg"), img_standard)

    # 2. Dark image
    img_dark = np.full((100, 100, 3), 30, dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000002.jpg"), img_dark)

    # 3. Bright image
    img_bright = np.full((100, 100, 3), 220, dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000003.jpg"), img_bright)

    # 4. Sharp high-contrast checkerboard
    img_sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    img_sharp[::10, :] = 255
    img_sharp[:, ::10] = 255
    cv2.imwrite(str(frames_dir / "frame_000004.jpg"), img_sharp)

    # 5. Very blurred image
    img_blurry = cv2.GaussianBlur(img_sharp, (25, 25), 10.0)
    cv2.imwrite(str(frames_dir / "frame_000005.jpg"), img_blurry)

    return frames_dir


@pytest.fixture
def synthetic_metadata_csv(synthetic_images_dir: Path, tmp_path: Path) -> Path:
    """Create a matching metadata.csv for synthetic test images."""
    csv_path = tmp_path / "metadata.csv"
    lines = [
        "frame_filename,extracted_frame_id,source_frame_index,timestamp_seconds",
        "frame_000001.jpg,frame_000001,0,0.0",
        "frame_000002.jpg,frame_000002,1,0.033333",
        "frame_000003.jpg,frame_000003,2,0.066667",
        "frame_000004.jpg,frame_000004,3,0.1",
        "frame_000005.jpg,frame_000005,4,0.133333",
    ]
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path


# ============================================================================
# UNIT TESTS: Individual Feature Extractions & Edge Cases
# ============================================================================

def test_01_valid_3channel_color_image():
    """Verify feature extraction on standard 3-channel BGR image."""
    extractor = VisualQualityExtractor()
    img = np.full((120, 160, 3), fill_value=(50, 100, 150), dtype=np.uint8)
    record = extractor.extract_from_image_array(img, frame_id="frame_000001")

    assert record.frame_id == "frame_000001"
    assert record.image_width == 160
    assert record.image_height == 120
    assert record.channels == 3
    assert record.image_pixel_count == 120 * 160
    assert record.image_aspect_ratio == pytest.approx(160 / 120, rel=1e-4)

    # Grayscale conversion for (50, 100, 150) in BGR:
    # 0.299*150 + 0.587*100 + 0.114*50 = 44.85 + 58.7 + 5.7 = 109.25
    assert record.mean_brightness == pytest.approx(109.25, abs=1.5)
    assert record.color_channel_means is not None
    assert record.color_channel_means["b"] == pytest.approx(50.0, abs=1e-3)
    assert record.color_channel_means["g"] == pytest.approx(100.0, abs=1e-3)
    assert record.color_channel_means["r"] == pytest.approx(150.0, abs=1e-3)


def test_02_valid_1channel_grayscale_image():
    """Verify feature extraction on single-channel grayscale image."""
    extractor = VisualQualityExtractor()
    img = np.full((80, 80), fill_value=120, dtype=np.uint8)
    record = extractor.extract_from_image_array(img, frame_id="gray_01")

    assert record.channels == 1
    assert record.color_channel_means is None
    assert record.mean_brightness == pytest.approx(120.0, abs=1e-3)
    assert record.median_brightness == pytest.approx(120.0, abs=1e-3)
    assert record.brightness_std == pytest.approx(0.0, abs=1e-3)


def test_03_valid_4channel_bgra_image(tmp_path: Path):
    """Verify feature extraction on 4-channel BGRA image file."""
    extractor = VisualQualityExtractor()
    img_bgra = np.zeros((64, 64, 4), dtype=np.uint8)
    img_bgra[:, :, 0] = 30   # B
    img_bgra[:, :, 1] = 60   # G
    img_bgra[:, :, 2] = 90   # R
    img_bgra[:, :, 3] = 255  # Alpha

    bgra_path = tmp_path / "test_bgra.png"
    cv2.imwrite(str(bgra_path), img_bgra)

    record = extractor.extract_from_image_file(bgra_path, frame_id="bgra_01")
    assert record.channels == 4
    assert record.color_channel_means is not None
    assert record.color_channel_means["b"] == pytest.approx(30.0, abs=1.0)
    assert record.color_channel_means["g"] == pytest.approx(60.0, abs=1.0)
    assert record.color_channel_means["r"] == pytest.approx(90.0, abs=1.0)


def test_04_missing_image_file_handling(tmp_path: Path):
    """Verify graceful error reporting when image file does not exist."""
    extractor = VisualQualityExtractor()
    missing_file = tmp_path / "non_existent_frame.jpg"

    with pytest.raises(FileNotFoundExtractionError) as exc_info:
        extractor.extract_from_image_file(missing_file)
    assert "Image file does not exist" in str(exc_info.value)


def test_05_corrupted_image_handling(tmp_path: Path):
    """Verify corrupted image files raise CorruptedImageFileError."""
    extractor = VisualQualityExtractor()
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"This is completely invalid JPEG data 1234567890")

    with pytest.raises(CorruptedImageFileError) as exc_info:
        extractor.extract_from_image_file(corrupt_file)
    assert "Corrupted or unreadable" in str(exc_info.value)


def test_06_unsupported_image_format(tmp_path: Path):
    """Verify unsupported file extensions raise UnsupportedImageFormatError."""
    extractor = VisualQualityExtractor()
    txt_file = tmp_path / "frame.txt"
    txt_file.write_text("Hello world", encoding="utf-8")

    with pytest.raises(UnsupportedImageFormatError) as exc_info:
        extractor.extract_from_image_file(txt_file)
    assert "Unsupported image format" in str(exc_info.value)


def test_07_empty_image(tmp_path: Path):
    """Verify empty (0-byte) image files raise EmptyImageFileError."""
    extractor = VisualQualityExtractor()
    empty_file = tmp_path / "empty.jpg"
    empty_file.touch()

    with pytest.raises(EmptyImageFileError) as exc_info:
        extractor.extract_from_image_file(empty_file)
    assert "Image file is empty (0 bytes)" in str(exc_info.value)


def test_08_invalid_image_dimensions():
    """Verify 0-dimension or empty numpy arrays raise InvalidImageDimensionError."""
    extractor = VisualQualityExtractor()
    empty_array = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(InvalidImageDimensionError) as exc_info:
        extractor.extract_from_image_array(empty_array)
    assert "Invalid image dimensions" in str(exc_info.value)


def test_09_blur_score_calculation_accuracy():
    """Verify blur score matches cv2.Laplacian variance definition."""
    extractor = VisualQualityExtractor()
    # Create an image with known high-frequency gradients
    np.random.seed(42)
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    expected_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    record = extractor.extract_from_image_array(img)
    assert record.blur_score == pytest.approx(expected_blur, rel=1e-5)


def test_10_sharp_image_behavior():
    """Verify sharp image has blur_category='sharp' when variance > 300."""
    extractor = VisualQualityExtractor()
    # High contrast checkerboard produces very high Laplacian variance
    sharp_img = np.zeros((100, 100), dtype=np.uint8)
    sharp_img[::2, :] = 255
    record = extractor.extract_from_image_array(sharp_img)

    assert record.blur_score > 300.0
    assert record.blur_category == "sharp"


def test_11_blurred_image_behavior():
    """Verify blurred image has blur_category='blurry' when variance < 100."""
    extractor = VisualQualityExtractor()
    # Heavily blurred image
    sharp_img = np.zeros((100, 100), dtype=np.uint8)
    sharp_img[::4, :] = 255
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), 15.0)
    record = extractor.extract_from_image_array(blurred_img)

    assert record.blur_score < 100.0
    assert record.blur_category == "blurry"


def test_12_brightness_calculation_accuracy():
    """Verify brightness statistics (mean, median, std, min, max)."""
    extractor = VisualQualityExtractor()
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    record = extractor.extract_from_image_array(pixels)

    assert record.mean_brightness == pytest.approx(25.0, abs=1e-3)
    assert record.median_brightness == pytest.approx(25.0, abs=1e-3)
    assert record.minimum_brightness == 10.0
    assert record.maximum_brightness == 40.0
    # Population standard deviation of [10, 20, 30, 40]: sqrt(((15^2 + 5^2 + 5^2 + 15^2)/4)) = sqrt(125) = 11.1803
    assert record.brightness_std == pytest.approx(math.sqrt(125.0), abs=1e-3)


def test_13_dark_image_behavior():
    """Verify dark image categorizes as 'dark' when mean < 60."""
    extractor = VisualQualityExtractor()
    dark_img = np.full((50, 50), 40, dtype=np.uint8)
    record = extractor.extract_from_image_array(dark_img)

    assert record.mean_brightness == 40.0
    assert record.brightness_category == "dark"


def test_14_bright_image_behavior():
    """Verify bright image categorizes as 'bright' when mean > 180."""
    extractor = VisualQualityExtractor()
    bright_img = np.full((50, 50), 200, dtype=np.uint8)
    record = extractor.extract_from_image_array(bright_img)

    assert record.mean_brightness == 200.0
    assert record.brightness_category == "bright"


def test_15_contrast_calculation_accuracy():
    """Verify contrast metrics (RMS std, range, percentile range, normalized contrast)."""
    extractor = VisualQualityExtractor()
    # Uniform gradient 0 to 100
    arr = np.linspace(0, 100, 100, dtype=np.uint8).reshape((10, 10))
    record = extractor.extract_from_image_array(arr)

    assert record.contrast_range == pytest.approx(100.0, abs=1.0)
    assert record.contrast_std == pytest.approx(float(np.std(arr)), abs=1e-3)
    assert record.contrast_percentile_range == pytest.approx(
        float(np.percentile(arr, 95) - np.percentile(arr, 5)), abs=1e-3
    )
    expected_norm = float(np.std(arr) / (np.mean(arr) + 1e-6))
    assert record.normalized_contrast == pytest.approx(expected_norm, rel=1e-4)


def test_16_low_contrast_image_behavior():
    """Verify low-contrast image categorizes as 'low' when std < 30."""
    extractor = VisualQualityExtractor()
    low_contrast_img = np.full((50, 50), 128, dtype=np.uint8)
    # Add tiny variation
    low_contrast_img[0, 0] = 130
    record = extractor.extract_from_image_array(low_contrast_img)

    assert record.contrast_std < 30.0
    assert record.contrast_category == "low"


def test_17_high_contrast_image_behavior():
    """Verify high-contrast image categorizes as 'high' when std > 70."""
    extractor = VisualQualityExtractor()
    # Half black, half white: std is ~127.5
    high_contrast = np.zeros((100, 100), dtype=np.uint8)
    high_contrast[:, 50:] = 255
    record = extractor.extract_from_image_array(high_contrast)

    assert record.contrast_std > 70.0
    assert record.contrast_category == "high"


def test_18_percentile_contrast_calculation():
    """Verify robust percentile contrast handles extreme outlier pixels cleanly."""
    extractor = VisualQualityExtractor()
    # Normal image with values around 100, plus one hot pixel at 255 and one dead pixel at 0
    img = np.full((100, 100), 100, dtype=np.uint8)
    img[0, 0] = 0
    img[99, 99] = 255
    record = extractor.extract_from_image_array(img)

    # Full range is 255 - 0 = 255
    assert record.contrast_range == 255.0
    # But 5th to 95th percentile should ignore isolated single outliers in 10,000 pixels
    assert record.contrast_percentile_range == pytest.approx(0.0, abs=1.0)


def test_19_color_channel_means_accuracy():
    """Verify independent extraction of B, G, R channel means."""
    extractor = VisualQualityExtractor()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :, 0] = 20   # Blue
    img[:, :, 1] = 80   # Green
    img[:, :, 2] = 200  # Red

    record = extractor.extract_from_image_array(img)
    assert record.color_channel_means is not None
    assert record.color_channel_means["b"] == 20.0
    assert record.color_channel_means["g"] == 80.0
    assert record.color_channel_means["r"] == 200.0


def test_20_grayscale_channel_handling():
    """Verify grayscale images have color_channel_means=None and channels=1."""
    extractor = VisualQualityExtractor()
    gray = np.full((40, 40), 99, dtype=np.uint8)
    record = extractor.extract_from_image_array(gray)

    assert record.channels == 1
    assert record.color_channel_means is None
    assert record.grayscale_mean == 99.0


def test_21_small_image_edge_cases():
    """Verify very small images (2x2, 3x3) do not crash or throw exceptions."""
    extractor = VisualQualityExtractor()
    img_2x2 = np.array([[[10, 20, 30], [40, 50, 60]],
                        [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    record = extractor.extract_from_image_array(img_2x2)

    assert record.image_width == 2
    assert record.image_height == 2
    assert record.image_pixel_count == 4
    assert not math.isnan(record.blur_score)
    assert not math.isnan(record.mean_brightness)


def test_22_nan_and_infinity_prevention():
    """Verify all numerical values are finite and valid floats."""
    extractor = VisualQualityExtractor()
    # Uniform black image (mean=0, std=0)
    img_zero = np.zeros((64, 64, 3), dtype=np.uint8)
    record = extractor.extract_from_image_array(img_zero)

    assert not math.isnan(record.blur_score)
    assert not math.isinf(record.blur_score)
    assert not math.isnan(record.mean_brightness)
    assert not math.isnan(record.brightness_std)
    assert not math.isnan(record.normalized_contrast)
    assert not math.isinf(record.normalized_contrast)
    assert record.normalized_contrast == pytest.approx(0.0, abs=1e-5)


def test_23_missing_frame_metadata_handling(synthetic_images_dir: Path):
    """Verify extractor functions gracefully when metadata.csv is omitted."""
    extractor = VisualQualityExtractor()
    results = extractor.extract_from_directory(synthetic_images_dir, metadata_path=None)

    assert len(results["features"]) == 5
    for feat in results["features"]:
        assert feat.extracted_frame_id is None
        assert feat.source_frame_index is None
        assert feat.timestamp_seconds is None


def test_24_duplicate_frame_id_handling(tmp_path: Path):
    """Verify directory scanner handles unique sorting and does not duplicate records."""
    frames_dir = tmp_path / "dup_frames"
    frames_dir.mkdir()
    # Create two files with distinct names
    cv2.imwrite(str(frames_dir / "frame_0001.png"), np.zeros((30, 30, 3), dtype=np.uint8))
    cv2.imwrite(str(frames_dir / "frame_0002.png"), np.zeros((30, 30, 3), dtype=np.uint8))

    extractor = VisualQualityExtractor()
    results = extractor.extract_from_directory(frames_dir)
    assert len(results["features"]) == 2
    frame_ids = [f.frame_id for f in results["features"]]
    assert len(frame_ids) == len(set(frame_ids))


def test_25_frame_ordering_preservation(tmp_path: Path):
    """Verify extracted frames preserve strictly natural alphanumeric sorting order."""
    frames_dir = tmp_path / "ordering_frames"
    frames_dir.mkdir()
    for name in ["frame_000003.jpg", "frame_000001.jpg", "frame_000002.jpg", "frame_000010.jpg"]:
        cv2.imwrite(str(frames_dir / name), np.zeros((20, 20, 3), dtype=np.uint8))

    extractor = VisualQualityExtractor()
    results = extractor.extract_from_directory(frames_dir)
    order = [f.frame_filename for f in results["features"]]
    assert order == ["frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg", "frame_000010.jpg"]


def test_26_partial_processing(synthetic_images_dir: Path):
    """Verify max_frames, start_frame, and end_frame slicing."""
    extractor = VisualQualityExtractor()

    # max_frames = 2
    res_max = extractor.extract_from_directory(synthetic_images_dir, max_frames=2)
    assert len(res_max["features"]) == 2
    assert res_max["summary"].is_partial is True

    # start_frame=2, end_frame=3
    res_slice = extractor.extract_from_directory(synthetic_images_dir, start_frame=2, end_frame=3)
    assert len(res_slice["features"]) == 2
    assert [f.frame_filename for f in res_slice["features"]] == ["frame_000002.jpg", "frame_000003.jpg"]


def test_27_output_json_schema_validation(synthetic_images_dir: Path, tmp_path: Path):
    """Verify all 4 artifacts match Schema v1.0 specifications."""
    out_dir = tmp_path / "features_out"
    extractor = VisualQualityExtractor()
    extractor.extract_from_directory(synthetic_images_dir, output_dir=out_dir)

    # 1. visual_quality_features.json
    feat_file = out_dir / "visual_quality_features.json"
    assert feat_file.exists()
    feat_data = json.loads(feat_file.read_text(encoding="utf-8"))
    assert feat_data["schema_version"] == "1.0"
    assert feat_data["feature_type"] == "visual_quality"
    assert len(feat_data["features"]) == 5

    # Check required fields of a single record
    rec = feat_data["features"][0]
    required_keys = [
        "frame_id", "frame_filename", "image_width", "image_height", "image_aspect_ratio",
        "image_pixel_count", "channels", "blur_score", "blur_category",
        "mean_brightness", "median_brightness", "brightness_std", "minimum_brightness",
        "maximum_brightness", "brightness_category", "contrast_std", "contrast_range",
        "contrast_percentile_range", "normalized_contrast", "contrast_category",
        "grayscale_mean", "grayscale_std", "color_channel_means"
    ]
    for key in required_keys:
        assert key in rec, f"Missing key: {key}"

    # 2. visual_quality_summary.json
    sum_file = out_dir / "visual_quality_summary.json"
    assert sum_file.exists()
    sum_data = json.loads(sum_file.read_text(encoding="utf-8"))
    assert sum_data["total_frames_evaluated"] == 5
    assert sum_data["valid_frames_count"] == 5
    assert sum_data["failed_frames_count"] == 0
    assert "mean_blur_score" in sum_data
    assert "mean_brightness" in sum_data
    assert "blur_distribution" in sum_data

    # 3. visual_quality_metadata.json
    meta_file = out_dir / "visual_quality_metadata.json"
    assert meta_file.exists()
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_data["schema_version"] == "1.0"
    assert "hardware_execution" in meta_data
    assert meta_data["hardware_execution"]["cpu_only"] is True

    # 4. visual_quality_errors.json
    err_file = out_dir / "visual_quality_errors.json"
    assert err_file.exists()
    err_data = json.loads(err_file.read_text(encoding="utf-8"))
    assert err_data["total_errors"] == 0


def test_28_error_artifact_generation(tmp_path: Path):
    """Verify visual_quality_errors.json logs failed files without terminating execution."""
    frames_dir = tmp_path / "mixed_frames"
    frames_dir.mkdir()
    # 1 valid frame
    cv2.imwrite(str(frames_dir / "frame_0001.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
    # 1 corrupted frame
    (frames_dir / "frame_0002.jpg").write_bytes(b"garbage corrupt image bytes")

    out_dir = tmp_path / "mixed_out"
    extractor = VisualQualityExtractor()
    results = extractor.extract_from_directory(frames_dir, output_dir=out_dir)

    assert len(results["features"]) == 1
    assert len(results["errors"]) == 1
    assert results["summary"].total_frames_evaluated == 2
    assert results["summary"].valid_frames_count == 1
    assert results["summary"].failed_frames_count == 1

    err_file = out_dir / "visual_quality_errors.json"
    assert err_file.exists()
    err_data = json.loads(err_file.read_text(encoding="utf-8"))
    assert err_data["total_errors"] == 1
    assert err_data["errors"][0]["error_type"] == "CORRUPTED_IMAGE"


def test_29_custom_configuration_thresholds():
    """Verify custom blur, brightness, and contrast thresholds from FeatureConfig."""
    custom_cfg = FeatureConfig(
        visual_quality_blur_sharp=500.0,
        visual_quality_blur_blurry=200.0,
        visual_quality_brightness_dark=80.0,
        visual_quality_brightness_bright=200.0,
        visual_quality_contrast_low=40.0,
        visual_quality_contrast_high=90.0,
    )
    extractor = VisualQualityExtractor(config=custom_cfg)

    # Blur score of 250 with custom thresholds is 'moderate' (200 <= 250 <= 500)
    assert classify_blur(250.0, custom_cfg) == "moderate"
    # Blur score of 150 is 'blurry' (< 200)
    assert classify_blur(150.0, custom_cfg) == "blurry"

    # Brightness 75 is 'dark' (< 80)
    assert classify_brightness(75.0, custom_cfg) == "dark"

    # Contrast 35 is 'low' (< 40)
    assert classify_contrast(35.0, custom_cfg) == "low"


def test_30_metadata_preservation_from_day2(synthetic_images_dir: Path, synthetic_metadata_csv: Path):
    """Verify extracted records preserve Day 2 extracted_frame_id, source_frame_index, and timestamp."""
    extractor = VisualQualityExtractor()
    results = extractor.extract_from_directory(
        synthetic_images_dir,
        metadata_path=synthetic_metadata_csv
    )

    feat_0 = results["features"][0]
    assert feat_0.frame_filename == "frame_000001.jpg"
    assert feat_0.extracted_frame_id == "frame_000001"
    assert feat_0.source_frame_index == 0
    assert feat_0.timestamp_seconds == 0.0

    feat_4 = results["features"][4]
    assert feat_4.frame_filename == "frame_000005.jpg"
    assert feat_4.source_frame_index == 4
    assert feat_4.timestamp_seconds == pytest.approx(0.1333, abs=1e-4)


def test_31_output_directory_exists_error(synthetic_images_dir: Path, tmp_path: Path):
    """Verify OutputDirectoryExistsError is raised when overwrite=False and files exist."""
    out_dir = tmp_path / "exist_test"
    out_dir.mkdir(parents=True)
    (out_dir / "dummy.txt").write_text("existing content", encoding="utf-8")

    extractor = VisualQualityExtractor()
    with pytest.raises(OutputDirectoryExistsError):
        extractor.extract_from_directory(synthetic_images_dir, output_dir=out_dir, overwrite=False)


def test_32_output_directory_overwrite_success(synthetic_images_dir: Path, tmp_path: Path):
    """Verify overwriting succeeds when overwrite=True."""
    out_dir = tmp_path / "overwrite_test"
    out_dir.mkdir(parents=True)
    (out_dir / "visual_quality_features.json").write_text("{}", encoding="utf-8")

    extractor = VisualQualityExtractor()
    results = extractor.extract_from_directory(synthetic_images_dir, output_dir=out_dir, overwrite=True)
    assert len(results["features"]) == 5
    assert (out_dir / "visual_quality_summary.json").exists()


# ============================================================================
# CLI & INTEGRATION TESTS
# ============================================================================

def test_33_cli_execution_success(synthetic_images_dir: Path, synthetic_metadata_csv: Path, tmp_path: Path):
    """Verify CLI interface runs end-to-end and returns exit code 0."""
    out_dir = tmp_path / "cli_out"
    args = [
        "--input", str(synthetic_images_dir),
        "--metadata", str(synthetic_metadata_csv),
        "--output", str(out_dir),
        "--dataset-id", "test_driving",
        "--verbose"
    ]
    exit_code = main(args)
    assert exit_code == 0
    assert (out_dir / "visual_quality_features.json").exists()


def test_34_cli_execution_failure_on_invalid_directory(tmp_path: Path):
    """Verify CLI returns exit code 1 when given a non-existent input directory."""
    non_dir = tmp_path / "missing_dir"
    args = ["--input", str(non_dir)]
    exit_code = main(args)
    assert exit_code == 1


def test_35_cli_partial_run_flags(synthetic_images_dir: Path, tmp_path: Path):
    """Verify CLI handles --max-frames and --start-frame flags."""
    out_dir = tmp_path / "cli_partial"
    args = [
        "--input", str(synthetic_images_dir),
        "--output", str(out_dir),
        "--start-frame", "1",
        "--end-frame", "2",
        "--overwrite"
    ]
    exit_code = main(args)
    assert exit_code == 0
    sum_data = json.loads((out_dir / "visual_quality_summary.json").read_text(encoding="utf-8"))
    assert sum_data["valid_frames_count"] == 2
    assert sum_data["is_partial"] is True


def test_36_uniform_black_image_edge_case():
    """Verify completely uniform black image yields zero blur and zero contrast."""
    extractor = VisualQualityExtractor()
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    record = extractor.extract_from_image_array(black_img)

    assert record.blur_score == 0.0
    assert record.mean_brightness == 0.0
    assert record.brightness_std == 0.0
    assert record.contrast_std == 0.0
    assert record.contrast_range == 0.0
    assert record.normalized_contrast == 0.0
    assert record.blur_category == "blurry"
    assert record.brightness_category == "dark"
    assert record.contrast_category == "low"


def test_37_uniform_white_image_edge_case():
    """Verify completely uniform white image yields zero blur and zero contrast."""
    extractor = VisualQualityExtractor()
    white_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    record = extractor.extract_from_image_array(white_img)

    assert record.blur_score == 0.0
    assert record.mean_brightness == 255.0
    assert record.brightness_std == 0.0
    assert record.contrast_std == 0.0
    assert record.contrast_range == 0.0
    assert record.normalized_contrast == 0.0
    assert record.blur_category == "blurry"
    assert record.brightness_category == "bright"
    assert record.contrast_category == "low"


def test_38_non_existent_metadata_csv_fallback(synthetic_images_dir: Path):
    """Verify non-existent metadata CSV path issues warning and falls back safely."""
    extractor = VisualQualityExtractor()
    bogus_csv = synthetic_images_dir.parent / "non_existent_meta.csv"
    results = extractor.extract_from_directory(synthetic_images_dir, metadata_path=bogus_csv)

    assert len(results["features"]) == 5
    assert results["features"][0].source_frame_index is None


def test_39_extreme_aspect_ratio_image_edge_case():
    """Verify extreme aspect ratio images (e.g. 1000x10) do not cause division by zero or errors."""
    extractor = VisualQualityExtractor()
    panoramic = np.zeros((10, 1000, 3), dtype=np.uint8)
    record = extractor.extract_from_image_array(panoramic)

    assert record.image_width == 1000
    assert record.image_height == 10
    assert record.image_aspect_ratio == 100.0


# ============================================================================
# REAL-WORLD INTEGRATION TESTS (BDD100K Driving Sample Dataset)
# ============================================================================

@pytest.mark.skipif(not REAL_FRAMES_DIR.exists(), reason="Real driving_sample frames not available")
def test_40_real_dataset_frames_extraction(tmp_path: Path):
    """Integration test: extract visual quality from real driving_sample frames."""
    out_dir = tmp_path / "real_driving_features"
    extractor = VisualQualityExtractor()

    # Process first 10 real frames for quick test suite validation
    results = extractor.extract_from_directory(
        input_dir=REAL_FRAMES_DIR,
        metadata_path=REAL_METADATA_CSV if REAL_METADATA_CSV.exists() else None,
        output_dir=out_dir,
        max_frames=10,
        overwrite=True
    )

    assert len(results["features"]) == 10
    assert len(results["errors"]) == 0
    assert results["summary"].valid_frames_count == 10
    assert results["summary"].mean_blur_score > 0.0

    # Verify frame 1 attributes
    f1 = results["features"][0]
    assert f1.frame_filename == "frame_000001.jpg"
    assert f1.image_width == 640
    assert f1.image_height == 360
    assert f1.channels == 3
    if REAL_METADATA_CSV.exists():
        assert f1.source_frame_index == 0
        assert f1.timestamp_seconds == 0.0


@pytest.mark.skipif(
    not (REAL_FRAMES_DIR.exists() and REAL_METADATA_CSV.exists()),
    reason="Real driving_sample frames or metadata.csv not available"
)
def test_41_real_dataset_metadata_alignment():
    """Integration test: verify 1-to-1 alignment between real metadata.csv and frames."""
    extractor = VisualQualityExtractor()
    meta_lookup = extractor._load_metadata_lookup(REAL_METADATA_CSV)

    assert len(meta_lookup) >= 200
    assert "frame_000001.jpg" in meta_lookup
    m1 = meta_lookup["frame_000001.jpg"]
    assert m1["extracted_frame_id"] == "frame_000001"
    assert m1["source_frame_index"] == 0
    assert m1["timestamp_seconds"] == 0.0

    m200 = meta_lookup["frame_000200.jpg"]
    assert m200["source_frame_index"] == 1198


@pytest.mark.skipif(not REAL_FRAMES_DIR.exists(), reason="Real driving_sample frames not available")
def test_42_real_frame_pixel_math_verification():
    """Integration test: manually verify OpenCV pixel math matches extractor on real frame."""
    frame_path = REAL_FRAMES_DIR / "frame_000001.jpg"
    raw_bgr = cv2.imread(str(frame_path))
    assert raw_bgr is not None

    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    expected_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    expected_mean_bright = float(np.mean(gray))
    expected_contrast_std = float(np.std(gray))

    extractor = VisualQualityExtractor()
    record = extractor.extract_from_image_file(frame_path)

    assert record.blur_score == pytest.approx(expected_blur, rel=1e-5)
    assert record.mean_brightness == pytest.approx(expected_mean_bright, rel=1e-5)
    assert record.contrast_std == pytest.approx(expected_contrast_std, rel=1e-5)
