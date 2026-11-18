"""PerceptGuard AI: Visual Quality Feature Extractor.

Day 5 Task 4: Modular, CPU-efficient extraction of visual and optical quality
features directly from video frames (OpenCV + NumPy).

Extracted Features:
- Optical Blur:
  - Laplacian variance: var(Laplacian(gray, CV_64F))
  - Heuristic categorization: sharp (> 300), moderate (100-300), blurry (< 100)
- Grayscale Brightness:
  - Mean, median, std, min, max
  - Heuristic categorization: dark (< 60), normal (60-180), bright (> 180)
- Grayscale Contrast:
  - RMS contrast (standard deviation of pixel intensities)
  - Full contrast range (max - min)
  - 90% percentile contrast range (P95 - P5)
  - Normalized contrast: std / (mean + 1e-6)
  - Heuristic categorization: low (< 30), normal (30-70), high (> 70)
- Image Geometry & Channel Metrics:
  - Width, height, aspect ratio (width / height), total pixel count
  - Channel count (1 for gray, 3 for BGR, 4 for BGRA)
  - Grayscale mean and std
  - Color channel means (B, G, R) for color images (None for grayscale)

Artifacts Produced (Schema v1.0):
1. visual_quality_features.json
2. visual_quality_summary.json
3. visual_quality_metadata.json
4. visual_quality_errors.json
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import cv2
import numpy as np

from src.features.feature_config import FeatureConfig
from src.features.feature_schema import FEATURE_SCHEMA_VERSION

logger = logging.getLogger("perceptguard.features.visual_quality")

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default heuristic thresholds
DEFAULT_BLUR_THRESHOLD_BLURRY = 100.0
DEFAULT_BLUR_THRESHOLD_SHARP = 300.0
DEFAULT_BRIGHTNESS_THRESHOLD_DARK = 60.0
DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT = 180.0
DEFAULT_CONTRAST_THRESHOLD_LOW = 30.0
DEFAULT_CONTRAST_THRESHOLD_HIGH = 70.0

DEFAULT_BLUR_THRESHOLDS = {
    "sharp": DEFAULT_BLUR_THRESHOLD_SHARP,
    "blurry": DEFAULT_BLUR_THRESHOLD_BLURRY,
}

DEFAULT_BRIGHTNESS_THRESHOLDS = {
    "dark": DEFAULT_BRIGHTNESS_THRESHOLD_DARK,
    "bright": DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT,
}

DEFAULT_CONTRAST_THRESHOLDS = {
    "low": DEFAULT_CONTRAST_THRESHOLD_LOW,
    "high": DEFAULT_CONTRAST_THRESHOLD_HIGH,
}


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class VisualQualityExtractionError(Exception):
    """Base exception for visual quality extraction operations."""
    pass


class ImageReadError(VisualQualityExtractionError):
    """Raised when an image cannot be read from disk."""
    pass


class FileNotFoundExtractionError(ImageReadError, FileNotFoundError):
    """Raised when an image file does not exist on disk."""
    pass


class CorruptedImageError(VisualQualityExtractionError):
    """Raised when an image file is corrupted or cannot be decoded by OpenCV."""
    pass


class CorruptedImageFileError(CorruptedImageError):
    """Alias for corrupted image files."""
    pass


class EmptyImageFileError(CorruptedImageError):
    """Raised when an image file has 0 bytes."""
    pass


class InvalidImageDimensionError(VisualQualityExtractionError, ValueError):
    """Raised when image has non-positive or empty dimensions."""
    pass


class UnsupportedImageFormatError(VisualQualityExtractionError):
    """Raised when an image format is unsupported."""
    pass


class OutputDirectoryExistsError(VisualQualityExtractionError):
    """Raised when target output directory exists and overwrite=False."""
    pass


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class FrameVisualQualityRecord:
    """Frame-level visual quality feature vector."""
    dataset_id: str
    frame_id: str
    frame_filename: str
    image_path: str
    extracted_frame_id: Optional[str]
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    image_width: int
    image_height: int
    image_aspect_ratio: float
    image_pixel_count: int
    channels: int
    blur_score: float
    blur_category: str
    mean_brightness: float
    median_brightness: float
    brightness_std: float
    minimum_brightness: float
    maximum_brightness: float
    brightness_category: str
    contrast_std: float
    contrast_range: float
    contrast_percentile_range: float
    normalized_contrast: float
    contrast_category: str
    grayscale_mean: float
    grayscale_std: float
    color_channel_means: Optional[Dict[str, float]]
    processing_status: str  # "success", "failed", "unreadable"
    error_message: Optional[str] = None
    _error_record: Optional[Any] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary, excluding internal attributes."""
        d = asdict(self)
        d.pop("_error_record", None)
        return d

    def __iter__(self):
        """Enable tuple unpacking: rec, err = extractor.extract_from_image_file(...)."""
        return iter((self, self._error_record))

    def __getitem__(self, idx: int):
        return (self, self._error_record)[idx]

    def __len__(self) -> int:
        return 2


@dataclass
class VisualQualitySummary:
    """Aggregate execution summary and distribution metrics across processed frames."""
    schema_version: str
    run_id: str
    dataset_id: str
    source_image_dir: str
    metadata_source: Optional[str]
    extracted_at: str
    is_partial: bool
    processed_frame_count: int
    valid_frame_count: int
    failed_frame_count: int
    extraction_duration_seconds: float
    validation_status: str
    blur_statistics: Dict[str, Optional[float]]
    brightness_statistics: Dict[str, Optional[float]]
    contrast_statistics: Dict[str, Optional[float]]
    quality_category_counts: Dict[str, Dict[str, int]]

    @property
    def total_frames_evaluated(self) -> int:
        return self.processed_frame_count

    @property
    def valid_frames_count(self) -> int:
        return self.valid_frame_count

    @property
    def failed_frames_count(self) -> int:
        return self.failed_frame_count

    @property
    def mean_blur_score(self) -> Optional[float]:
        return self.blur_statistics.get("mean")

    @property
    def mean_brightness(self) -> Optional[float]:
        return self.brightness_statistics.get("mean")

    @property
    def mean_contrast_std(self) -> Optional[float]:
        return self.contrast_statistics.get("mean")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary with both canonical and alias convenience fields."""
        d = asdict(self)
        d["total_frames_evaluated"] = self.processed_frame_count
        d["valid_frames_count"] = self.valid_frame_count
        d["failed_frames_count"] = self.failed_frame_count
        d["mean_blur_score"] = self.blur_statistics.get("mean")
        d["mean_brightness"] = self.brightness_statistics.get("mean")
        d["mean_contrast_std"] = self.contrast_statistics.get("mean")
        d["blur_distribution"] = self.quality_category_counts.get("blur", {})
        d["brightness_distribution"] = self.quality_category_counts.get("brightness", {})
        d["contrast_distribution"] = self.quality_category_counts.get("contrast", {})
        return d


@dataclass
class VisualQualityErrorRecord:
    """Audit entry recording corrupted, unreadable, or missing frame failures."""
    frame_id: str
    image_path: str
    error_type: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Helper Functions
# ==============================================================================

def classify_blur(
    score: float,
    thresholds: Optional[Union[Dict[str, float], FeatureConfig]] = None,
) -> str:
    """Classify blur score into heuristic tiers: sharp, moderate, blurry."""
    if isinstance(thresholds, FeatureConfig):
        sharp = thresholds.visual_quality_blur_sharp
        blurry = thresholds.visual_quality_blur_blurry
    elif isinstance(thresholds, dict):
        sharp = thresholds.get("sharp", DEFAULT_BLUR_THRESHOLD_SHARP)
        blurry = thresholds.get("blurry", DEFAULT_BLUR_THRESHOLD_BLURRY)
    else:
        sharp = DEFAULT_BLUR_THRESHOLD_SHARP
        blurry = DEFAULT_BLUR_THRESHOLD_BLURRY

    if score > sharp:
        return "sharp"
    elif score < blurry:
        return "blurry"
    return "moderate"


def classify_brightness(
    mean_val: float,
    thresholds: Optional[Union[Dict[str, float], FeatureConfig]] = None,
) -> str:
    """Classify brightness into heuristic tiers: dark, normal, bright."""
    if isinstance(thresholds, FeatureConfig):
        dark = thresholds.visual_quality_brightness_dark
        bright = thresholds.visual_quality_brightness_bright
    elif isinstance(thresholds, dict):
        dark = thresholds.get("dark", DEFAULT_BRIGHTNESS_THRESHOLD_DARK)
        bright = thresholds.get("bright", DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT)
    else:
        dark = DEFAULT_BRIGHTNESS_THRESHOLD_DARK
        bright = DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT

    if mean_val < dark:
        return "dark"
    elif mean_val > bright:
        return "bright"
    return "normal"


def classify_contrast(
    std_val: float,
    thresholds: Optional[Union[Dict[str, float], FeatureConfig]] = None,
) -> str:
    """Classify contrast into heuristic tiers: low, normal, high."""
    if isinstance(thresholds, FeatureConfig):
        low = thresholds.visual_quality_contrast_low
        high = thresholds.visual_quality_contrast_high
    elif isinstance(thresholds, dict):
        low = thresholds.get("low", DEFAULT_CONTRAST_THRESHOLD_LOW)
        high = thresholds.get("high", DEFAULT_CONTRAST_THRESHOLD_HIGH)
    else:
        low = DEFAULT_CONTRAST_THRESHOLD_LOW
        high = DEFAULT_CONTRAST_THRESHOLD_HIGH

    if std_val < low:
        return "low"
    elif std_val > high:
        return "high"
    return "normal"


def extract_frame_index_from_filename(filename: str) -> Optional[int]:
    """Parse numeric sequence index from filename (e.g. frame_000042.jpg -> 42)."""
    match = re.search(r"\d+", filename)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


# ==============================================================================
# Visual Quality Extractor Engine
# ==============================================================================

class VisualQualityExtractor:
    """Production visual quality extractor operating directly on frame pixels."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        blur_thresholds: Optional[Dict[str, float]] = None,
        brightness_thresholds: Optional[Dict[str, float]] = None,
        contrast_thresholds: Optional[Dict[str, float]] = None,
        decimal_places: Optional[int] = None,
    ) -> None:
        """Initialize visual quality extractor with parameters and thresholds.

        Args:
            config: Optional FeatureConfig instance.
            blur_thresholds: Dict with 'sharp' and 'blurry' cutoffs.
            brightness_thresholds: Dict with 'dark' and 'bright' cutoffs.
            contrast_thresholds: Dict with 'low' and 'high' cutoffs.
            decimal_places: Number of decimal places to round computed floats.
        """
        self.config = config or FeatureConfig()

        if blur_thresholds is not None:
            self.blur_thresholds = dict(blur_thresholds)
        elif self.config is not None:
            self.blur_thresholds = {
                "sharp": getattr(self.config, "visual_quality_blur_sharp", DEFAULT_BLUR_THRESHOLD_SHARP),
                "blurry": getattr(self.config, "visual_quality_blur_blurry", DEFAULT_BLUR_THRESHOLD_BLURRY),
            }
        else:
            self.blur_thresholds = dict(DEFAULT_BLUR_THRESHOLDS)

        if brightness_thresholds is not None:
            self.brightness_thresholds = dict(brightness_thresholds)
        elif self.config is not None:
            self.brightness_thresholds = {
                "dark": getattr(self.config, "visual_quality_brightness_dark", DEFAULT_BRIGHTNESS_THRESHOLD_DARK),
                "bright": getattr(self.config, "visual_quality_brightness_bright", DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT),
            }
        else:
            self.brightness_thresholds = dict(DEFAULT_BRIGHTNESS_THRESHOLDS)

        if contrast_thresholds is not None:
            self.contrast_thresholds = dict(contrast_thresholds)
        elif self.config is not None:
            self.contrast_thresholds = {
                "low": getattr(self.config, "visual_quality_contrast_low", DEFAULT_CONTRAST_THRESHOLD_LOW),
                "high": getattr(self.config, "visual_quality_contrast_high", DEFAULT_CONTRAST_THRESHOLD_HIGH),
            }
        else:
            self.contrast_thresholds = dict(DEFAULT_CONTRAST_THRESHOLDS)

        self.decimal_places = (
            decimal_places
            if decimal_places is not None
            else self.config.decimal_places
        )

    def extract_from_directory(
        self,
        frame_dir: Optional[Union[str, Path]] = None,
        metadata_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        dataset_id: Optional[str] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
        # Keyword aliases
        input_dir: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Process an entire directory of frame images sequentially.

        Args:
            frame_dir: Directory containing extracted/preprocessed image files.
            metadata_file: Optional path to metadata.csv.
            output_dir: Custom destination directory for feature artifacts.
            dataset_id: Dataset identifier string.
            max_frames: Maximum number of frames to evaluate.
            start_frame: Start frame index filter (inclusive).
            end_frame: End frame index filter (inclusive).
            overwrite: If True, replace contents of existing output directory.
            input_dir: Alias for frame_dir.
            metadata_path: Alias for metadata_file.

        Returns:
            Dictionary containing output directory path and summary telemetry.
        """
        t_start = time.perf_counter()

        chosen_dir = input_dir or frame_dir
        if not chosen_dir:
            raise VisualQualityExtractionError("Missing required frame_dir / input_dir argument.")

        in_dir = Path(chosen_dir).resolve()
        if not in_dir.exists() or not in_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: '{in_dir}'")

        chosen_meta = metadata_path or metadata_file
        meta_file = Path(chosen_meta).resolve() if chosen_meta else None
        meta_lookup = self._load_metadata_lookup(meta_file) if meta_file else {}

        d_id = dataset_id or self._infer_dataset_id(in_dir)
        run_id = f"vq_{d_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Scan and sort all image files naturally
        image_files = sorted(
            [
                p for p in in_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ],
            key=lambda p: (extract_frame_index_from_filename(p.name) or 0, p.name)
        )

        total_scanned = len(image_files)
        is_partial = False

        # Apply start_frame and end_frame slicing
        filtered_files: List[Path] = []
        for p in image_files:
            meta_info = meta_lookup.get(p.name, {})
            f_idx = meta_info.get("source_frame_index", extract_frame_index_from_filename(p.name))

            if start_frame is not None and f_idx is not None and f_idx < start_frame:
                continue
            if end_frame is not None and f_idx is not None and f_idx > end_frame:
                continue

            filtered_files.append(p)

        if start_frame is not None or end_frame is not None:
            is_partial = True

        if max_frames is not None and max_frames > 0 and len(filtered_files) > max_frames:
            filtered_files = filtered_files[:max_frames]
            is_partial = True

        records: List[FrameVisualQualityRecord] = []
        errors: List[VisualQualityErrorRecord] = []
        seen_frame_ids: Set[str] = set()

        for img_path in filtered_files:
            fid = img_path.name
            if fid in seen_frame_ids:
                errors.append(
                    VisualQualityErrorRecord(
                        frame_id=fid,
                        image_path=str(img_path),
                        error_type="DUPLICATE_FRAME_ID",
                        message=f"Duplicate frame identifier: '{fid}'. Skipping.",
                    )
                )
                continue
            seen_frame_ids.add(fid)

            # Look up metadata
            meta_info = meta_lookup.get(fid, {})
            ext_fid = meta_info.get("extracted_frame_id")
            src_idx = meta_info.get("source_frame_index")
            ts_sec = meta_info.get("timestamp_seconds")

            # Extract features from image (safe mode: do not raise on corrupted individual frame)
            rec = self.extract_from_image_file(
                image_path=img_path,
                dataset_id=d_id,
                frame_id=fid,
                extracted_frame_id=ext_fid,
                source_frame_index=src_idx,
                timestamp_seconds=ts_sec,
                raise_on_error=False,
            )
            err = rec._error_record
            if rec.processing_status == "success":
                records.append(rec)
            if err:
                errors.append(err)

        # Build summary
        duration = round(time.perf_counter() - t_start, 4)
        summary = self._build_summary(
            run_id=run_id,
            dataset_id=d_id,
            source_image_dir=str(in_dir),
            metadata_source=str(meta_file) if meta_file else None,
            is_partial=is_partial,
            total_evaluated=len(filtered_files),
            records=records,
            errors=errors,
            duration=duration,
        )

        # Resolve output directory
        target_dir = self._resolve_output_dir(output_dir, d_id, run_id)
        if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
            raise OutputDirectoryExistsError(
                f"Output directory '{target_dir}' exists and is not empty. Pass overwrite=True to replace."
            )

        target_dir.mkdir(parents=True, exist_ok=True)

        raw_metadata = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": run_id,
            "dataset_id": d_id,
            "source_image_dir": str(in_dir),
            "metadata_source": str(meta_file) if meta_file else None,
            "extracted_at": summary.extracted_at,
            "is_partial": is_partial,
            "thresholds": {
                "blur": self.blur_thresholds,
                "brightness": self.brightness_thresholds,
                "contrast": self.contrast_thresholds,
            },
            "feature_dictionary": self._get_feature_dictionary(),
            "execution_environment": {
                "python_version": sys.version,
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "platform": sys.platform,
            },
            "hardware_execution": {
                "cpu_only": True,
                "cuda_available": False,
                "device": "cpu",
            },
        }

        # Persist all 4 versioned artifacts
        self._persist_artifacts(
            target_dir=target_dir,
            records=records,
            summary=summary,
            errors=errors,
            raw_metadata=raw_metadata,
        )

        logger.info(
            f"Visual quality extraction completed in {duration}s. "
            f"Frames: {len(records)}, Valid: {summary.valid_frame_count}, "
            f"Failed: {summary.failed_frame_count}, Output: {target_dir}"
        )

        return {
            "output_dir": str(target_dir),
            "run_id": run_id,
            "summary": summary,
            "summary_dict": summary.to_dict(),
            "features": records,
            "records": records,
            "errors": errors,
            "valid_frame_count": summary.valid_frame_count,
            "valid_frames_count": summary.valid_frame_count,
            "failed_frame_count": summary.failed_frame_count,
            "failed_frames_count": summary.failed_frame_count,
            "total_frames_evaluated": summary.processed_frame_count,
            "duration_seconds": duration,
        }

    def extract_from_image_file(
        self,
        image_path: Union[str, Path],
        dataset_id: str = "unknown_dataset",
        frame_id: Optional[str] = None,
        extracted_frame_id: Optional[str] = None,
        source_frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        raise_on_error: bool = True,
    ) -> FrameVisualQualityRecord:
        """Extract visual quality metrics from an individual image file on disk.

        Returns FrameVisualQualityRecord which can also be unpacked as (rec, err).
        """
        p = Path(image_path).resolve()
        fid = frame_id or p.name

        if not p.exists():
            err = VisualQualityErrorRecord(
                frame_id=fid,
                image_path=str(p),
                error_type="FILE_NOT_FOUND",
                message=f"Image file does not exist: '{p}'",
            )
            if raise_on_error:
                raise FileNotFoundExtractionError(err.message)
            return self._build_empty_record(
                d_id=dataset_id, fid=fid, p=p, status="failed", err_msg=err.message, err_rec=err
            )

        if p.stat().st_size == 0:
            err = VisualQualityErrorRecord(
                frame_id=fid,
                image_path=str(p),
                error_type="EMPTY_FILE",
                message=f"Image file is empty (0 bytes): '{p}'",
            )
            if raise_on_error:
                raise EmptyImageFileError(err.message)
            return self._build_empty_record(
                d_id=dataset_id, fid=fid, p=p, status="failed", err_msg=err.message, err_rec=err
            )

        if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            err = VisualQualityErrorRecord(
                frame_id=fid,
                image_path=str(p),
                error_type="UNSUPPORTED_FORMAT",
                message=f"Unsupported image format '{p.suffix}'. Supported: {SUPPORTED_IMAGE_EXTENSIONS}",
            )
            if raise_on_error:
                raise UnsupportedImageFormatError(err.message)
            return self._build_empty_record(
                d_id=dataset_id, fid=fid, p=p, status="failed", err_msg=err.message, err_rec=err
            )

        try:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        except Exception as ex:
            err = VisualQualityErrorRecord(
                frame_id=fid,
                image_path=str(p),
                error_type="CORRUPTED_IMAGE",
                message=f"OpenCV failed to decode image: {ex}",
            )
            if raise_on_error:
                raise CorruptedImageFileError(err.message) from ex
            return self._build_empty_record(
                d_id=dataset_id, fid=fid, p=p, status="failed", err_msg=err.message, err_rec=err
            )

        if img is None or img.size == 0:
            err = VisualQualityErrorRecord(
                frame_id=fid,
                image_path=str(p),
                error_type="CORRUPTED_IMAGE",
                message=f"Corrupted or unreadable image file: '{p}'",
            )
            if raise_on_error:
                raise CorruptedImageFileError(err.message)
            return self._build_empty_record(
                d_id=dataset_id, fid=fid, p=p, status="failed", err_msg=err.message, err_rec=err
            )

        # Process image array in-memory
        rec = self.extract_from_image_array(
            image=img,
            dataset_id=dataset_id,
            frame_id=fid,
            image_path=str(p),
            extracted_frame_id=extracted_frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=timestamp_seconds,
            raise_on_error=raise_on_error,
        )

        del img
        return rec

    def extract_from_image_array(
        self,
        image: np.ndarray,
        dataset_id: str = "unknown_dataset",
        frame_id: str = "in_memory_frame",
        image_path: str = "in_memory",
        extracted_frame_id: Optional[str] = None,
        source_frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        raise_on_error: bool = True,
    ) -> FrameVisualQualityRecord:
        """Calculate visual quality metrics from an in-memory NumPy image array."""
        dec = self.decimal_places

        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
            err = VisualQualityErrorRecord(
                frame_id=frame_id,
                image_path=image_path,
                error_type="INVALID_IMAGE_ARRAY",
                message=f"Invalid image dimensions or empty array: {getattr(image, 'shape', None)}",
            )
            if raise_on_error:
                raise InvalidImageDimensionError(err.message)
            return self._build_empty_record(
                d_id=dataset_id, fid=frame_id, p=Path(image_path), status="failed", err_msg=err.message, err_rec=err
            )

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            err = VisualQualityErrorRecord(
                frame_id=frame_id,
                image_path=image_path,
                error_type="INVALID_DIMENSIONS",
                message=f"Invalid image dimensions: {w}x{h}",
            )
            if raise_on_error:
                raise InvalidImageDimensionError(err.message)
            return self._build_empty_record(
                d_id=dataset_id, fid=frame_id, p=Path(image_path), status="failed", err_msg=err.message, err_rec=err
            )

        channels = 1 if image.ndim == 2 else image.shape[2]
        aspect_ratio = round(float(w) / float(h), dec)
        pixel_count = int(w * h)

        # Convert to grayscale for optical metrics
        color_channel_means: Optional[Dict[str, float]] = None

        if channels == 1:
            gray = image
            color_channel_means = None
        elif channels == 3:
            # OpenCV loads standard images as BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            b_mean = round(float(np.mean(image[:, :, 0])), dec)
            g_mean = round(float(np.mean(image[:, :, 1])), dec)
            r_mean = round(float(np.mean(image[:, :, 2])), dec)
            color_channel_means = {"b": b_mean, "g": g_mean, "r": r_mean}
        elif channels == 4:
            # BGRA image
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            b_mean = round(float(np.mean(image[:, :, 0])), dec)
            g_mean = round(float(np.mean(image[:, :, 1])), dec)
            r_mean = round(float(np.mean(image[:, :, 2])), dec)
            color_channel_means = {"b": b_mean, "g": g_mean, "r": r_mean}
        else:
            gray = image[:, :, 0]
            color_channel_means = None

        # 1. Blur Feature: Laplacian Variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score_raw = float(laplacian.var())
        blur_score = round(blur_score_raw, dec)
        blur_cat = classify_blur(blur_score, self.blur_thresholds)

        # 2. Brightness Features
        gray_float = gray.astype(np.float64)
        mean_b = round(float(np.mean(gray_float)), dec)
        med_b = round(float(np.median(gray_float)), dec)
        std_b = round(float(np.std(gray_float)), dec)
        min_b = round(float(np.min(gray_float)), dec)
        max_b = round(float(np.max(gray_float)), dec)
        bright_cat = classify_brightness(mean_b, self.brightness_thresholds)

        # 3. Contrast Features
        contrast_std = std_b
        contrast_range = round(float(max_b - min_b), dec)
        p95 = float(np.percentile(gray_float, 95))
        p5 = float(np.percentile(gray_float, 5))
        contrast_p95_p5 = round(float(p95 - p5), dec)
        norm_contrast = round(float(contrast_std / (mean_b + 1e-6)), dec)
        contrast_cat = classify_contrast(contrast_std, self.contrast_thresholds)

        # 4. Grayscale Statistics
        gray_mean = mean_b
        gray_std = std_b

        rec = FrameVisualQualityRecord(
            dataset_id=dataset_id,
            frame_id=frame_id,
            frame_filename=Path(image_path).name if image_path != "in_memory" else frame_id,
            image_path=image_path,
            extracted_frame_id=extracted_frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=round(timestamp_seconds, dec) if timestamp_seconds is not None else None,
            image_width=w,
            image_height=h,
            image_aspect_ratio=aspect_ratio,
            image_pixel_count=pixel_count,
            channels=channels,
            blur_score=blur_score,
            blur_category=blur_cat,
            mean_brightness=mean_b,
            median_brightness=med_b,
            brightness_std=std_b,
            minimum_brightness=min_b,
            maximum_brightness=max_b,
            brightness_category=bright_cat,
            contrast_std=contrast_std,
            contrast_range=contrast_range,
            contrast_percentile_range=contrast_p95_p5,
            normalized_contrast=norm_contrast,
            contrast_category=contrast_cat,
            grayscale_mean=gray_mean,
            grayscale_std=gray_std,
            color_channel_means=color_channel_means,
            processing_status="success",
            error_message=None,
            _error_record=None,
        )

        return rec

    def _build_empty_record(
        self,
        d_id: str,
        fid: str,
        p: Path,
        status: str,
        err_msg: str,
        err_rec: Optional[VisualQualityErrorRecord] = None,
    ) -> FrameVisualQualityRecord:
        """Construct an empty record with status='failed' for unreadable frames."""
        return FrameVisualQualityRecord(
            dataset_id=d_id,
            frame_id=fid,
            frame_filename=p.name,
            image_path=str(p),
            extracted_frame_id=None,
            source_frame_index=None,
            timestamp_seconds=None,
            image_width=0,
            image_height=0,
            image_aspect_ratio=0.0,
            image_pixel_count=0,
            channels=0,
            blur_score=0.0,
            blur_category="blurry",
            mean_brightness=0.0,
            median_brightness=0.0,
            brightness_std=0.0,
            minimum_brightness=0.0,
            maximum_brightness=0.0,
            brightness_category="dark",
            contrast_std=0.0,
            contrast_range=0.0,
            contrast_percentile_range=0.0,
            normalized_contrast=0.0,
            contrast_category="low",
            grayscale_mean=0.0,
            grayscale_std=0.0,
            color_channel_means=None,
            processing_status=status,
            error_message=err_msg,
            _error_record=err_rec,
        )

    def _load_metadata_lookup(self, metadata_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load Day 2 metadata.csv into an index keyed by frame filename."""
        lookup: Dict[str, Dict[str, Any]] = {}
        if not metadata_path.exists() or metadata_path.stat().st_size == 0:
            logger.warning(f"Metadata file '{metadata_path}' not found or empty.")
            return lookup

        import csv
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = (
                        row.get("preprocessed_image_filename")
                        or row.get("original_image_filename")
                        or row.get("frame_filename")
                        or row.get("filename")
                    )
                    if not fname:
                        continue
                    fname = fname.strip()
                    try:
                        src_idx = int(row["source_frame_index"]) if "source_frame_index" in row and row["source_frame_index"] != "" else None
                    except (ValueError, TypeError):
                        src_idx = None

                    try:
                        ts = float(row["timestamp_seconds"]) if "timestamp_seconds" in row and row["timestamp_seconds"] != "" else None
                    except (ValueError, TypeError):
                        ts = None

                    raw_ext = row.get("extracted_frame_id")
                    if raw_ext is not None:
                        raw_ext = str(raw_ext).strip()
                        if raw_ext.isdigit() and len(raw_ext) < 6:
                            ext_id = f"frame_{int(raw_ext):06d}"
                        else:
                            ext_id = raw_ext
                    else:
                        ext_id = None

                    lookup[fname] = {
                        "extracted_frame_id": ext_id,
                        "source_frame_index": src_idx,
                        "timestamp_seconds": ts,
                    }
        except Exception as ex:
            logger.warning(f"Failed to parse metadata CSV '{metadata_path}': {ex}")

        return lookup

    def _build_summary(
        self,
        run_id: str,
        dataset_id: str,
        source_image_dir: str,
        metadata_source: Optional[str],
        is_partial: bool,
        total_evaluated: int,
        records: List[FrameVisualQualityRecord],
        errors: List[VisualQualityErrorRecord],
        duration: float,
    ) -> VisualQualitySummary:
        """Aggregate statistical distribution of visual quality metrics."""
        dec = self.decimal_places
        valid_recs = [r for r in records if r.processing_status == "success"]

        valid_count = len(valid_recs)
        failed_count = len(errors)

        blur_scores = [r.blur_score for r in valid_recs]
        bright_scores = [r.mean_brightness for r in valid_recs]
        contrast_scores = [r.contrast_std for r in valid_recs]

        def compute_dist(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {"mean": None, "median": None, "std": None, "min": None, "max": None}
            arr = np.array(values, dtype=np.float64)
            return {
                "mean": round(float(np.mean(arr)), dec),
                "median": round(float(np.median(arr)), dec),
                "std": round(float(np.std(arr)), dec),
                "min": round(float(np.min(arr)), dec),
                "max": round(float(np.max(arr)), dec),
            }

        blur_dist = {"sharp": 0, "moderate": 0, "blurry": 0}
        bright_dist = {"dark": 0, "normal": 0, "bright": 0}
        contrast_dist = {"low": 0, "normal": 0, "high": 0}

        for r in valid_recs:
            blur_dist[r.blur_category] = blur_dist.get(r.blur_category, 0) + 1
            bright_dist[r.brightness_category] = bright_dist.get(r.brightness_category, 0) + 1
            contrast_dist[r.contrast_category] = contrast_dist.get(r.contrast_category, 0) + 1

        val_status = "PASS" if failed_count == 0 and len(errors) == 0 else "WARNING"

        return VisualQualitySummary(
            schema_version=FEATURE_SCHEMA_VERSION,
            run_id=run_id,
            dataset_id=dataset_id,
            source_image_dir=source_image_dir,
            metadata_source=metadata_source,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            is_partial=is_partial,
            processed_frame_count=total_evaluated,
            valid_frame_count=valid_count,
            failed_frame_count=failed_count,
            extraction_duration_seconds=duration,
            validation_status=val_status,
            blur_statistics=compute_dist(blur_scores),
            brightness_statistics=compute_dist(bright_scores),
            contrast_statistics=compute_dist(contrast_scores),
            quality_category_counts={
                "blur": blur_dist,
                "brightness": bright_dist,
                "contrast": contrast_dist,
            },
        )

    def _resolve_output_dir(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_id: str,
        run_id: str,
    ) -> Path:
        """Determine destination path for feature artifacts."""
        if output_dir:
            return Path(output_dir).resolve()
        return (
            Path("datasets")
            / "processed"
            / dataset_id
            / "features"
            / "visual_quality"
            / run_id
        ).resolve()

    def _infer_dataset_id(self, dir_path: Path) -> str:
        """Infer dataset id from parent directory tree."""
        parts = dir_path.resolve().parts
        for i, part in enumerate(parts):
            if part == "processed" and i + 1 < len(parts):
                return parts[i + 1]
        return "driving_sample"

    def _persist_artifacts(
        self,
        target_dir: Path,
        records: List[FrameVisualQualityRecord],
        summary: VisualQualitySummary,
        errors: List[VisualQualityErrorRecord],
        raw_metadata: Dict[str, Any],
    ) -> None:
        """Atomically persist all 4 versioned feature artifacts to target directory."""
        # 1. visual_quality_features.json
        features_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_type": "visual_quality",
            "dataset_id": summary.dataset_id,
            "run_id": summary.run_id,
            "generated_at": summary.extracted_at,
            "total_frames": len(records),
            "features": [r.to_dict() for r in records],
        }
        self._write_json_atomically(target_dir / "visual_quality_features.json", features_payload)

        # 2. visual_quality_summary.json
        self._write_json_atomically(target_dir / "visual_quality_summary.json", summary.to_dict())

        # 3. visual_quality_metadata.json
        self._write_json_atomically(target_dir / "visual_quality_metadata.json", raw_metadata)

        # 4. visual_quality_errors.json
        errors_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "generated_at": summary.extracted_at,
            "total_errors": len(errors),
            "errors": [e.to_dict() for e in errors],
        }
        self._write_json_atomically(target_dir / "visual_quality_errors.json", errors_payload)

    def _write_json_atomically(self, target_path: Path, data: Any) -> None:
        """Write JSON payload to temporary file before atomic rename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(target_path)

    def _get_feature_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Return schema documentation for visual quality features."""
        return {
            "blur_score": {"type": "float", "description": "Variance of the Laplacian filter on grayscale frame."},
            "blur_category": {"type": "str", "description": "Heuristic tier: sharp, moderate, blurry."},
            "mean_brightness": {"type": "float", "description": "Mean pixel intensity of grayscale image [0, 255]."},
            "median_brightness": {"type": "float", "description": "Median pixel intensity of grayscale image."},
            "brightness_std": {"type": "float", "description": "Standard deviation of pixel intensity."},
            "minimum_brightness": {"type": "float", "description": "Minimum pixel intensity [0, 255]."},
            "maximum_brightness": {"type": "float", "description": "Maximum pixel intensity [0, 255]."},
            "brightness_category": {"type": "str", "description": "Heuristic tier: dark, normal, bright."},
            "contrast_std": {"type": "float", "description": "RMS contrast: standard deviation of grayscale intensities."},
            "contrast_range": {"type": "float", "description": "Full dynamic range: maximum - minimum intensity."},
            "contrast_percentile_range": {"type": "float", "description": "Robust contrast range: P95 - P5 intensity."},
            "normalized_contrast": {"type": "float", "description": "contrast_std / (mean_brightness + 1e-6)."},
            "contrast_category": {"type": "str", "description": "Heuristic tier: low, normal, high."},
            "grayscale_mean": {"type": "float", "description": "Mean grayscale intensity."},
            "grayscale_std": {"type": "float", "description": "Standard deviation of grayscale intensity."},
            "color_channel_means": {"type": "dict", "description": "Mean intensities for B, G, R channels."},
            "image_aspect_ratio": {"type": "float", "description": "Image width divided by image height."},
            "image_pixel_count": {"type": "int", "description": "Total image pixel count (width * height)."},
        }


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI for Visual Quality Feature Extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Visual Quality Feature Extractor CLI."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to frame directory (e.g. preprocessed_frames).",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        default=None,
        help="Optional path to metadata.csv file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional path to feature_config.yaml file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom destination directory for feature outputs.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset identifier (e.g. driving_sample).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Starting source frame index (inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ending source frame index (inclusive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose / debug diagnostic output.",
    )

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    cfg = FeatureConfig.from_yaml(args.config) if args.config else None
    extractor = VisualQualityExtractor(config=cfg)

    try:
        result = extractor.extract_from_directory(
            frame_dir=args.input,
            metadata_file=args.metadata,
            output_dir=args.output,
            dataset_id=args.dataset_id,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        sum_dict = result["summary"].to_dict() if hasattr(result["summary"], "to_dict") else result["summary"]
        print("\n" + "=" * 70)
        print("PERCEPTGUARD AI — VISUAL QUALITY FEATURE EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"  Dataset ID            : {sum_dict['dataset_id']}")
        print(f"  Run ID                : {sum_dict['run_id']}")
        print(f"  Status                : [{sum_dict['validation_status']}]")
        print(f"  Is Partial Run        : {sum_dict['is_partial']}")
        print(f"  Processed Frames      : {sum_dict['processed_frame_count']}")
        print(f"  Valid Frames          : {sum_dict['valid_frame_count']}")
        print(f"  Failed Frames         : {sum_dict['failed_frame_count']}")
        print(f"  Mean Blur Score       : {sum_dict['blur_statistics']['mean']}")
        print(f"  Mean Brightness       : {sum_dict['brightness_statistics']['mean']}")
        print(f"  Mean Contrast Std     : {sum_dict['contrast_statistics']['mean']}")
        print(f"  Duration              : {sum_dict['extraction_duration_seconds']}s")
        print(f"  Output Directory      : {result['output_dir']}")
        print("=" * 70 + "\n")
        return 0

    except Exception as ex:
        logger.error(f"Visual quality extraction failed: {ex}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
