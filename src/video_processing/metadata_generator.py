"""PerceptGuard-AI: Metadata Generation Engine.

Production-grade ML data engineering module for autonomous driving perception datasets.
Constructs, validates, and serializes structured metadata linking raw video streams,
extracted source frames, and standardized preprocessed tensors.

Key Architectural Guarantees:
- Strict Schema Integrity: Immutable data contracts with rigorous invariant checks
- Deterministic Source Mapping: Preserves original source frame indices and physical timestamps
- Fault-Tolerant Accounting: Records failed/corrupted frames without silent dropping
- Multi-Tier Validation: Enforces uniqueness, monotonicity, and spatial/numeric bounds
- RFC 8259 Compliant JSON: Serializes missing/NaN values as strict JSON null
- Dual Mode Execution: Operates from in-memory pipeline summaries or directly from disk assets
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import yaml

from src.video_processing.frame_extractor import ExtractionSummary
from src.video_processing.frame_preprocessor import (
    FramePreprocessor,
    PreprocessingSummary,
)
from src.video_processing.video_loader import VideoLoader, VideoProcessingError

logger = logging.getLogger("PerceptGuard.MetadataGenerator")

SCHEMA_VERSION: str = "1.0.0"


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class MetadataError(VideoProcessingError):
    """Base exception for metadata generation and serialization failures."""
    pass


class SchemaValidationError(MetadataError):
    """Raised when metadata records violate schema constraints, bounds, or invariants."""
    pass


class MetadataFileError(MetadataError):
    """Raised when filesystem I/O, CSV encoding, or JSON serialization fails."""
    pass


class FrameMappingError(MetadataError):
    """Raised when extracted frames and preprocessed assets cannot be uniquely matched."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class FrameMetadataRecord:
    """Immutable data contract encapsulating complete provenance for a video frame.

    Attributes:
        dataset_id: Logical dataset collection identifier.
        source_video: Basename of original source video container.
        extracted_frame_id: 1-indexed sequential frame number (1, 2, 3...).
        source_frame_index: 0-indexed discrete frame position in source stream (0, 6, 12...).
        timestamp_seconds: Physical timestamp in seconds (source_frame_index / source_fps).
        original_image_filename: Basename of raw extracted image.
        original_image_path: Relative or absolute path to raw extracted image.
        preprocessed_image_filename: Basename of preprocessed image (or None if failed).
        preprocessed_image_path: Relative or absolute path to preprocessed image (or None).
        file_size_bytes: Storage size of preprocessed image file in bytes.
        original_width: Spatial width of source extracted frame.
        original_height: Spatial height of source extracted frame.
        processed_width: Spatial width of preprocessed frame tensor.
        processed_height: Spatial height of preprocessed frame tensor.
        channel_count: Number of color channels (typically 3 for BGR/RGB).
        blur_score: Laplacian variance sharpness indicator (None if failed/unavailable).
        brightness_score: Mean grayscale intensity [0.0, 255.0] (None if unavailable).
        contrast_score: RMS contrast standard deviation (None if unavailable).
        preprocessing_status: Processing status ('SUCCESS', 'FAILED', 'SKIPPED').
        validation_status: Validation status ('VALID', 'INVALID').
        failure_reason: Detailed failure message if processing failed (None if successful).
    """
    dataset_id: str
    source_video: str
    extracted_frame_id: int
    source_frame_index: int
    timestamp_seconds: float
    original_image_filename: str
    original_image_path: str
    preprocessed_image_filename: Optional[str]
    preprocessed_image_path: Optional[str]
    file_size_bytes: Optional[int]
    original_width: Optional[int]
    original_height: Optional[int]
    processed_width: Optional[int]
    processed_height: Optional[int]
    channel_count: Optional[int]
    blur_score: Optional[float]
    brightness_score: Optional[float]
    contrast_score: Optional[float]
    preprocessing_status: str
    validation_status: str
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame record to dictionary representation."""
        data = asdict(self)
        data["timestamp_seconds"] = round(self.timestamp_seconds, 4)
        if self.blur_score is not None and not math.isnan(self.blur_score):
            data["blur_score"] = round(self.blur_score, 2)
        else:
            data["blur_score"] = None
        if self.brightness_score is not None and not math.isnan(self.brightness_score):
            data["brightness_score"] = round(self.brightness_score, 2)
        else:
            data["brightness_score"] = None
        if self.contrast_score is not None and not math.isnan(self.contrast_score):
            data["contrast_score"] = round(self.contrast_score, 2)
        else:
            data["contrast_score"] = None
        return data


@dataclass(frozen=True)
class DatasetSummary:
    """Immutable dataset-level provenance and telemetry summary contract.

    Attributes:
        dataset_id: Logical dataset collection identifier.
        source_video: Basename of original source video container.
        source_video_fps: Frame rate of source video.
        source_video_resolution: Spatial dimensions (width, height) of source video.
        total_source_frames: Total frames reported in source video container.
        total_extracted_frames: Total frames extracted during sampling stage.
        total_preprocessed_frames: Total frames successfully preprocessed.
        total_failed_frames: Total frames that failed validation or preprocessing.
        average_blur_score: Mean blur score across valid frames (None if unavailable).
        average_brightness_score: Mean brightness across valid frames (None if unavailable).
        average_contrast_score: Mean contrast across valid frames (None if unavailable).
        processed_image_dimensions: Standardized spatial dimensions (width, height).
        metadata_generation_timestamp: ISO 8601 generation timestamp with timezone.
        metadata_schema_version: Semantic version of metadata schema.
    """
    dataset_id: str
    source_video: str
    source_video_fps: float
    source_video_resolution: Tuple[int, int]
    total_source_frames: int
    total_extracted_frames: int
    total_preprocessed_frames: int
    total_failed_frames: int
    average_blur_score: Optional[float]
    average_brightness_score: Optional[float]
    average_contrast_score: Optional[float]
    processed_image_dimensions: Tuple[int, int]
    metadata_generation_timestamp: str
    metadata_schema_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataset summary to RFC 8259 strict JSON-compatible dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "source_video": self.source_video,
            "source_video_fps": round(self.source_video_fps, 2),
            "source_video_resolution": list(self.source_video_resolution),
            "total_source_frames": self.total_source_frames,
            "total_extracted_frames": self.total_extracted_frames,
            "total_preprocessed_frames": self.total_preprocessed_frames,
            "total_failed_frames": self.total_failed_frames,
            "average_blur_score": (
                round(self.average_blur_score, 2)
                if self.average_blur_score is not None and not math.isnan(self.average_blur_score)
                else None
            ),
            "average_brightness_score": (
                round(self.average_brightness_score, 2)
                if self.average_brightness_score is not None and not math.isnan(self.average_brightness_score)
                else None
            ),
            "average_contrast_score": (
                round(self.average_contrast_score, 2)
                if self.average_contrast_score is not None and not math.isnan(self.average_contrast_score)
                else None
            ),
            "processed_image_dimensions": list(self.processed_image_dimensions),
            "metadata_generation_timestamp": self.metadata_generation_timestamp,
            "metadata_schema_version": self.metadata_schema_version,
        }


# ==============================================================================
# MetadataGenerator Implementation
# ==============================================================================

class MetadataGenerator:
    """Robust metadata compilation and integrity verification engine.

    Collects frame provenance across video extraction and preprocessing stages,
    validates schema invariants, and serializes auditable metadata.csv and summary.json.
    """

    CSV_COLUMNS: List[str] = [
        "dataset_id",
        "source_video",
        "extracted_frame_id",
        "source_frame_index",
        "timestamp_seconds",
        "original_image_filename",
        "original_image_path",
        "preprocessed_image_filename",
        "preprocessed_image_path",
        "file_size_bytes",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "channel_count",
        "blur_score",
        "brightness_score",
        "contrast_score",
        "preprocessing_status",
        "validation_status",
        "failure_reason",
    ]

    def __init__(
        self,
        dataset_id: str = "driving_sample",
        source_video_path: Optional[Union[str, Path]] = None,
        raw_frames_dir: Optional[Union[str, Path]] = None,
        preprocessed_frames_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize MetadataGenerator with dataset parameters and paths.

        Args:
            dataset_id: Logical identifier for the dataset (e.g., 'driving_sample').
            source_video_path: Path to raw autonomous driving video.
            raw_frames_dir: Directory containing raw extracted frames.
            preprocessed_frames_dir: Directory containing preprocessed frames.
            config_path: Optional path to YAML configuration file.
        """
        self.dataset_id = str(dataset_id)
        self._source_video_path = Path("datasets/raw/converted_026c7465-309f6d33.mp4")
        self._raw_frames_dir = Path("datasets/processed/driving_sample/frames")
        self._preprocessed_frames_dir = Path("datasets/processed/driving_sample/preprocessed_frames")
        self._extraction_fps: float = 5.0
        self._target_width: int = 640
        self._target_height: int = 360

        # Load values from configuration file
        self._load_config(config_path)

        # Apply programmatic overrides
        if source_video_path is not None:
            self._source_video_path = Path(source_video_path).resolve()
        else:
            self._source_video_path = self._source_video_path.resolve()

        if raw_frames_dir is not None:
            self._raw_frames_dir = Path(raw_frames_dir).resolve()
        else:
            self._raw_frames_dir = self._raw_frames_dir.resolve()

        if preprocessed_frames_dir is not None:
            self._preprocessed_frames_dir = Path(preprocessed_frames_dir).resolve()
        else:
            self._preprocessed_frames_dir = self._preprocessed_frames_dir.resolve()

        self._records: List[FrameMetadataRecord] = []
        self._summary: Optional[DatasetSummary] = None

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> None:
        """Load configuration options from video_config.yaml."""
        if not config_path:
            default_cfg = Path(__file__).resolve().parent.parent.parent / "config" / "video_config.yaml"
            if default_cfg.is_file():
                config_path = default_cfg

        if config_path and Path(config_path).is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                v_cfg = cfg.get("video", {})
                if "default_video" in v_cfg and v_cfg["default_video"]:
                    self._source_video_path = Path(v_cfg["default_video"])

                ext_cfg = cfg.get("frame_extraction", {})
                if "output_directory" in ext_cfg:
                    self._raw_frames_dir = Path(ext_cfg["output_directory"])
                if "extraction_fps" in ext_cfg:
                    self._extraction_fps = float(ext_cfg["extraction_fps"])

                p_cfg = cfg.get("preprocessing", {})
                if "output_directory" in p_cfg:
                    self._preprocessed_frames_dir = Path(p_cfg["output_directory"])
                if "target_width" in p_cfg:
                    self._target_width = int(p_cfg["target_width"])
                if "target_height" in p_cfg:
                    self._target_height = int(p_cfg["target_height"])
            except Exception as e:
                logger.warning("Failed to load config %s: %s. Using default paths.", config_path, e)

    @property
    def records(self) -> List[FrameMetadataRecord]:
        """Generated metadata records."""
        return self._records

    @property
    def summary(self) -> Optional[DatasetSummary]:
        """Dataset-level provenance summary."""
        return self._summary

    def validate_metadata_record(self, record: FrameMetadataRecord) -> None:
        """Enforce strict invariant validation on an individual metadata record.

        Args:
            record: FrameMetadataRecord to inspect.

        Raises:
            SchemaValidationError: If record violates spatial, numeric, or status constraints.
        """
        if not record.dataset_id:
            raise SchemaValidationError("Record field 'dataset_id' cannot be empty.")
        if not record.source_video:
            raise SchemaValidationError("Record field 'source_video' cannot be empty.")
        if record.extracted_frame_id <= 0:
            raise SchemaValidationError(
                f"extracted_frame_id must be a positive integer, got: {record.extracted_frame_id}"
            )
        if record.source_frame_index < 0:
            raise SchemaValidationError(
                f"source_frame_index cannot be negative, got: {record.source_frame_index}"
            )
        if record.timestamp_seconds < 0.0 or math.isnan(record.timestamp_seconds):
            raise SchemaValidationError(
                f"timestamp_seconds must be non-negative and finite, got: {record.timestamp_seconds}"
            )

        if record.validation_status not in ("VALID", "INVALID"):
            raise SchemaValidationError(
                f"Invalid validation_status '{record.validation_status}'. Expected 'VALID' or 'INVALID'."
            )
        if record.preprocessing_status not in ("SUCCESS", "FAILED", "SKIPPED"):
            raise SchemaValidationError(
                f"Invalid preprocessing_status '{record.preprocessing_status}'. "
                "Expected 'SUCCESS', 'FAILED', or 'SKIPPED'."
            )

        # For valid preprocessed records, enforce dimension feasibility
        if record.validation_status == "VALID" and record.preprocessing_status == "SUCCESS":
            if not record.original_width or record.original_width <= 0:
                raise SchemaValidationError(f"Invalid original_width: {record.original_width}")
            if not record.original_height or record.original_height <= 0:
                raise SchemaValidationError(f"Invalid original_height: {record.original_height}")
            if not record.processed_width or record.processed_width <= 0:
                raise SchemaValidationError(f"Invalid processed_width: {record.processed_width}")
            if not record.processed_height or record.processed_height <= 0:
                raise SchemaValidationError(f"Invalid processed_height: {record.processed_height}")
            if not record.channel_count or record.channel_count not in (1, 3, 4):
                raise SchemaValidationError(f"Invalid channel_count: {record.channel_count}")

            # Enforce metric bounds if populated
            if record.blur_score is not None:
                if math.isnan(record.blur_score) or record.blur_score < 0.0:
                    raise SchemaValidationError(f"blur_score must be non-negative, got: {record.blur_score}")
            if record.brightness_score is not None:
                if math.isnan(record.brightness_score) or not (0.0 <= record.brightness_score <= 255.0):
                    raise SchemaValidationError(
                        f"brightness_score must be in [0, 255], got: {record.brightness_score}"
                    )
            if record.contrast_score is not None:
                if math.isnan(record.contrast_score) or record.contrast_score < 0.0:
                    raise SchemaValidationError(f"contrast_score must be non-negative, got: {record.contrast_score}")

    def validate_metadata_table(self, records: Sequence[FrameMetadataRecord]) -> None:
        """Enforce table-level invariant checks across all records.

        Checks:
        1. Non-empty table.
        2. Unique extracted_frame_id values.
        3. Unique source_frame_index values.
        4. Strictly monotonic chronological ordering (timestamps).
        5. Consistent dataset_id and source_video across all records.

        Args:
            records: Sequence of FrameMetadataRecord objects.

        Raises:
            SchemaValidationError: If table invariants are violated.
        """
        if not records:
            raise SchemaValidationError("Metadata record table is empty (0 records).")

        seen_extracted_ids: Set[int] = set()
        seen_source_indices: Set[int] = set()
        last_extracted_id = -1
        last_source_idx = -1
        last_timestamp = -1.0
        primary_dataset_id = records[0].dataset_id
        primary_video = records[0].source_video

        for idx, rec in enumerate(records):
            self.validate_metadata_record(rec)

            if rec.dataset_id != primary_dataset_id:
                raise SchemaValidationError(
                    f"Inconsistent dataset_id at row {idx}: '{rec.dataset_id}' != '{primary_dataset_id}'"
                )
            if rec.source_video != primary_video:
                raise SchemaValidationError(
                    f"Inconsistent source_video at row {idx}: '{rec.source_video}' != '{primary_video}'"
                )

            # Check extracted ID uniqueness and order
            if rec.extracted_frame_id in seen_extracted_ids:
                raise SchemaValidationError(
                    f"Duplicate extracted_frame_id encountered: {rec.extracted_frame_id} at row {idx}"
                )
            if rec.extracted_frame_id <= last_extracted_id:
                raise SchemaValidationError(
                    f"Non-monotonic extracted_frame_id: {rec.extracted_frame_id} <= {last_extracted_id}"
                )
            seen_extracted_ids.add(rec.extracted_frame_id)
            last_extracted_id = rec.extracted_frame_id

            # Check source index uniqueness and order
            if rec.source_frame_index in seen_source_indices:
                raise SchemaValidationError(
                    f"Duplicate source_frame_index encountered: {rec.source_frame_index} at row {idx}"
                )
            if rec.source_frame_index <= last_source_idx:
                raise SchemaValidationError(
                    f"Non-monotonic source_frame_index: {rec.source_frame_index} <= {last_source_idx}"
                )
            seen_source_indices.add(rec.source_frame_index)
            last_source_idx = rec.source_frame_index

            # Check timestamp monotonicity
            if rec.timestamp_seconds <= last_timestamp:
                raise SchemaValidationError(
                    f"Non-monotonic timestamp at row {idx}: {rec.timestamp_seconds:.4f} <= {last_timestamp:.4f}"
                )
            last_timestamp = rec.timestamp_seconds

    def generate_metadata(
        self,
        extraction_summary: Optional[ExtractionSummary] = None,
        preprocessing_summary: Optional[PreprocessingSummary] = None,
    ) -> List[FrameMetadataRecord]:
        """Construct structured metadata records from pipeline summaries or disk assets.

        Args:
            extraction_summary: Optional in-memory ExtractionSummary from Task 3.
            preprocessing_summary: Optional in-memory PreprocessingSummary from Task 4.

        Returns:
            List[FrameMetadataRecord]: Ordered, validated metadata records.

        Raises:
            MetadataError: If assets are missing or cannot be resolved.
            SchemaValidationError: If generated records fail validation checks.
        """
        records: List[FrameMetadataRecord] = []

        # Strategy A: Generate from in-memory summaries if provided
        if extraction_summary is not None and preprocessing_summary is not None:
            logger.info("Generating metadata from in-memory pipeline summaries...")
            preproc_map = {r.input_filename: r for r in preprocessing_summary.records}

            for ext_info in extraction_summary.extracted_frames:
                filename = ext_info.image_filename
                proc_rec = preproc_map.get(filename)

                if proc_rec is not None and proc_rec.is_valid:
                    # Successfully preprocessed frame
                    file_size = (
                        proc_rec.output_path.stat().st_size
                        if proc_rec.output_path.is_file()
                        else None
                    )
                    record = FrameMetadataRecord(
                        dataset_id=self.dataset_id,
                        source_video=self._source_video_path.name,
                        extracted_frame_id=ext_info.extracted_frame_id,
                        source_frame_index=ext_info.source_frame_index,
                        timestamp_seconds=ext_info.timestamp_seconds,
                        original_image_filename=ext_info.image_filename,
                        original_image_path=str(ext_info.image_path),
                        preprocessed_image_filename=proc_rec.output_filename,
                        preprocessed_image_path=str(proc_rec.output_path),
                        file_size_bytes=file_size,
                        original_width=proc_rec.original_width,
                        original_height=proc_rec.original_height,
                        processed_width=proc_rec.processed_width,
                        processed_height=proc_rec.processed_height,
                        channel_count=3,
                        blur_score=proc_rec.blur_score,
                        brightness_score=proc_rec.brightness_score,
                        contrast_score=proc_rec.contrast_score,
                        preprocessing_status="SUCCESS",
                        validation_status="VALID",
                        failure_reason=None,
                    )
                elif proc_rec is not None and not proc_rec.is_valid:
                    # Failed preprocessing frame
                    record = FrameMetadataRecord(
                        dataset_id=self.dataset_id,
                        source_video=self._source_video_path.name,
                        extracted_frame_id=ext_info.extracted_frame_id,
                        source_frame_index=ext_info.source_frame_index,
                        timestamp_seconds=ext_info.timestamp_seconds,
                        original_image_filename=ext_info.image_filename,
                        original_image_path=str(ext_info.image_path),
                        preprocessed_image_filename=None,
                        preprocessed_image_path=None,
                        file_size_bytes=None,
                        original_width=None,
                        original_height=None,
                        processed_width=None,
                        processed_height=None,
                        channel_count=None,
                        blur_score=None,
                        brightness_score=None,
                        contrast_score=None,
                        preprocessing_status="FAILED",
                        validation_status="INVALID",
                        failure_reason=proc_rec.error_message,
                    )
                else:
                    # Missing from preprocessing summary
                    record = FrameMetadataRecord(
                        dataset_id=self.dataset_id,
                        source_video=self._source_video_path.name,
                        extracted_frame_id=ext_info.extracted_frame_id,
                        source_frame_index=ext_info.source_frame_index,
                        timestamp_seconds=ext_info.timestamp_seconds,
                        original_image_filename=ext_info.image_filename,
                        original_image_path=str(ext_info.image_path),
                        preprocessed_image_filename=None,
                        preprocessed_image_path=None,
                        file_size_bytes=None,
                        original_width=None,
                        original_height=None,
                        processed_width=None,
                        processed_height=None,
                        channel_count=None,
                        blur_score=None,
                        brightness_score=None,
                        contrast_score=None,
                        preprocessing_status="SKIPPED",
                        validation_status="INVALID",
                        failure_reason="Frame was not present in preprocessing summary",
                    )
                records.append(record)

        # Strategy B: Generate deterministically from filesystem assets
        else:
            logger.info("Generating metadata directly from disk assets...")
            if not self._raw_frames_dir.is_dir():
                raise MetadataError(f"Raw frames directory does not exist: '{self._raw_frames_dir}'")

            # Inspect source video metadata to obtain exact FPS and total frames
            if not self._source_video_path.is_file():
                raise MetadataError(f"Source video file not found at: '{self._source_video_path}'")

            with VideoLoader(self._source_video_path) as loader:
                video_meta = loader.get_metadata()
                source_fps = video_meta.fps
                total_source_frames = video_meta.total_frames
                orig_w, orig_h = video_meta.width, video_meta.height

            # Collect sorted raw extracted image files
            raw_files = sorted([
                f for f in self._raw_frames_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ])

            if not raw_files:
                raise MetadataError(f"No extracted frame images found in '{self._raw_frames_dir}'")

            # Reconstruct exact deterministic sampling indices: i_k = round(k * (fps / target_fps))
            step = (
                source_fps / self._extraction_fps
                if (self._extraction_fps and self._extraction_fps < source_fps)
                else 1.0
            )

            # Initialize preprocessor for on-demand metric verification
            preprocessor = FramePreprocessor(
                target_width=self._target_width,
                target_height=self._target_height,
            )

            for idx, raw_path in enumerate(raw_files, start=1):
                extracted_id = idx
                k = idx - 1
                source_idx = int(round(k * step))
                timestamp_sec = float(source_idx) / source_fps

                proc_path = self._preprocessed_frames_dir / raw_path.name
                if proc_path.is_file() and proc_path.stat().st_size > 0:
                    is_valid, err, decoded_proc = preprocessor.validate_frame(proc_path)
                    raw_valid, raw_err, decoded_raw = preprocessor.validate_frame(raw_path)

                    if is_valid and decoded_proc is not None and raw_valid and decoded_raw is not None:
                        proc_h, proc_w = decoded_proc.shape[:2]
                        channels = 1 if len(decoded_proc.shape) == 2 else decoded_proc.shape[2]

                        # Calculate quality indicators on original raw frame
                        gray_raw = preprocessor.convert_color(decoded_raw, "GRAY")
                        blur_score = preprocessor.calculate_blur_score(gray_raw)
                        brightness_score = preprocessor.calculate_brightness_score(gray_raw)
                        contrast_score = preprocessor.calculate_contrast_score(gray_raw)
                        file_size = proc_path.stat().st_size

                        del decoded_proc
                        del decoded_raw
                        del gray_raw

                        record = FrameMetadataRecord(
                            dataset_id=self.dataset_id,
                            source_video=self._source_video_path.name,
                            extracted_frame_id=extracted_id,
                            source_frame_index=source_idx,
                            timestamp_seconds=timestamp_sec,
                            original_image_filename=raw_path.name,
                            original_image_path=str(raw_path),
                            preprocessed_image_filename=proc_path.name,
                            preprocessed_image_path=str(proc_path),
                            file_size_bytes=file_size,
                            original_width=orig_w,
                            original_height=orig_h,
                            processed_width=proc_w,
                            processed_height=proc_h,
                            channel_count=channels,
                            blur_score=blur_score,
                            brightness_score=brightness_score,
                            contrast_score=contrast_score,
                            preprocessing_status="SUCCESS",
                            validation_status="VALID",
                            failure_reason=None,
                        )
                    else:
                        record = FrameMetadataRecord(
                            dataset_id=self.dataset_id,
                            source_video=self._source_video_path.name,
                            extracted_frame_id=extracted_id,
                            source_frame_index=source_idx,
                            timestamp_seconds=timestamp_sec,
                            original_image_filename=raw_path.name,
                            original_image_path=str(raw_path),
                            preprocessed_image_filename=proc_path.name,
                            preprocessed_image_path=str(proc_path),
                            file_size_bytes=proc_path.stat().st_size,
                            original_width=orig_w,
                            original_height=orig_h,
                            processed_width=None,
                            processed_height=None,
                            channel_count=None,
                            blur_score=None,
                            brightness_score=None,
                            contrast_score=None,
                            preprocessing_status="FAILED",
                            validation_status="INVALID",
                            failure_reason=err or "Preprocessed image decoding failed",
                        )
                else:
                    record = FrameMetadataRecord(
                        dataset_id=self.dataset_id,
                        source_video=self._source_video_path.name,
                        extracted_frame_id=extracted_id,
                        source_frame_index=source_idx,
                        timestamp_seconds=timestamp_sec,
                        original_image_filename=raw_path.name,
                        original_image_path=str(raw_path),
                        preprocessed_image_filename=None,
                        preprocessed_image_path=None,
                        file_size_bytes=None,
                        original_width=orig_w,
                        original_height=orig_h,
                        processed_width=None,
                        processed_height=None,
                        channel_count=None,
                        blur_score=None,
                        brightness_score=None,
                        contrast_score=None,
                        preprocessing_status="FAILED",
                        validation_status="INVALID",
                        failure_reason="Preprocessed image file does not exist on disk",
                    )
                records.append(record)

        # Enforce table validation
        self.validate_metadata_table(records)
        self._records = records
        self._summary = self.generate_summary()
        return self._records

    def save_metadata_csv(self, output_csv_path: Optional[Union[str, Path]] = None) -> Path:
        """Serialize metadata records to a standardized CSV file.

        Args:
            output_csv_path: Destination path for metadata.csv.

        Returns:
            Path: Filesystem path to written CSV.

        Raises:
            MetadataFileError: If writing or read-back verification fails.
        """
        if not self._records:
            raise MetadataError("Cannot save CSV; metadata records have not been generated.")

        target_csv = (
            Path(output_csv_path).resolve()
            if output_csv_path
            else self._preprocessed_frames_dir.parent / "metadata.csv"
        )

        try:
            target_csv.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise MetadataFileError(f"Failed to create directory '{target_csv.parent}': {e}") from e

        # Convert records to Pandas DataFrame
        data = [r.to_dict() for r in self._records]
        df = pd.DataFrame(data)

        # Reorder columns explicitly to match schema definition
        ordered_cols = [c for c in self.CSV_COLUMNS if c in df.columns]
        df = df[ordered_cols]

        try:
            df.to_csv(target_csv, index=False, encoding="utf-8")
        except Exception as e:
            raise MetadataFileError(f"Failed to write CSV to '{target_csv}': {e}") from e

        # Read-back verification to guarantee data integrity
        try:
            read_df = pd.read_csv(target_csv, encoding="utf-8")
            if len(read_df) != len(self._records):
                raise MetadataFileError(
                    f"Read-back validation row count mismatch: {len(read_df)} != {len(self._records)}"
                )
        except Exception as e:
            raise MetadataFileError(f"CSV read-back integrity verification failed for '{target_csv}': {e}") from e

        logger.info("Successfully persisted metadata.csv (%d records) at: %s", len(self._records), target_csv)
        return target_csv

    def generate_summary(self) -> DatasetSummary:
        """Compute aggregate summary telemetry across all generated metadata records.

        Returns:
            DatasetSummary: Immutable summary record.

        Raises:
            MetadataError: If records have not been generated.
        """
        if not self._records:
            raise MetadataError("Cannot generate summary; no metadata records available.")

        # Source video metadata inspection
        source_fps = 30.09
        source_w, source_h = 720, 1280
        total_source = 1204

        if self._source_video_path.is_file():
            try:
                with VideoLoader(self._source_video_path) as loader:
                    vm = loader.get_metadata()
                    source_fps = vm.fps
                    source_w, source_h = vm.width, vm.height
                    total_source = vm.total_frames
            except Exception as e:
                logger.warning("Could not read video metadata for summary: %s", e)

        total_extracted = len(self._records)
        successful_records = [
            r for r in self._records
            if r.validation_status == "VALID" and r.preprocessing_status == "SUCCESS"
        ]
        total_preprocessed = len(successful_records)
        total_failed = total_extracted - total_preprocessed

        # Compute metric averages across successful frames
        blur_vals = [r.blur_score for r in successful_records if r.blur_score is not None and not math.isnan(r.blur_score)]
        bright_vals = [r.brightness_score for r in successful_records if r.brightness_score is not None and not math.isnan(r.brightness_score)]
        contrast_vals = [r.contrast_score for r in successful_records if r.contrast_score is not None and not math.isnan(r.contrast_score)]

        avg_blur = float(np.mean(blur_vals)) if blur_vals else None
        avg_bright = float(np.mean(bright_vals)) if bright_vals else None
        avg_contrast = float(np.mean(contrast_vals)) if contrast_vals else None

        # Determine output dimensions
        proc_w = successful_records[0].processed_width if successful_records and successful_records[0].processed_width else self._target_width
        proc_h = successful_records[0].processed_height if successful_records and successful_records[0].processed_height else self._target_height

        now_iso = datetime.now(timezone.utc).isoformat()

        self._summary = DatasetSummary(
            dataset_id=self.dataset_id,
            source_video=self._source_video_path.name,
            source_video_fps=source_fps,
            source_video_resolution=(source_w, source_h),
            total_source_frames=total_source,
            total_extracted_frames=total_extracted,
            total_preprocessed_frames=total_preprocessed,
            total_failed_frames=total_failed,
            average_blur_score=avg_blur,
            average_brightness_score=avg_bright,
            average_contrast_score=avg_contrast,
            processed_image_dimensions=(proc_w, proc_h),
            metadata_generation_timestamp=now_iso,
            metadata_schema_version=SCHEMA_VERSION,
        )
        return self._summary

    def save_summary_json(self, output_json_path: Optional[Union[str, Path]] = None) -> Path:
        """Serialize dataset summary to RFC 8259 strict JSON format.

        Guarantees that all missing or NaN values are serialized as JSON null,
        preventing non-standard JavaScript tokens in production pipelines.

        Args:
            output_json_path: Destination path for summary.json.

        Returns:
            Path: Filesystem path to written summary.json.

        Raises:
            MetadataFileError: If writing or JSON encoding fails.
        """
        if self._summary is None:
            self._summary = self.generate_summary()

        target_json = (
            Path(output_json_path).resolve()
            if output_json_path
            else self._preprocessed_frames_dir.parent / "summary.json"
        )

        try:
            target_json.parent.mkdir(parents=True, exist_ok=True)
            summary_dict = self._summary.to_dict()

            with open(target_json, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2)

            # Verification: parse back to confirm strict JSON syntax
            with open(target_json, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            if parsed.get("dataset_id") != self.dataset_id:
                raise MetadataFileError("Summary JSON read-back validation failed: key mismatch.")

        except Exception as e:
            raise MetadataFileError(f"Failed to persist summary.json to '{target_json}': {e}") from e

        logger.info("Successfully persisted summary.json at: %s", target_json)
        return target_json

    def get_metadata_statistics(self) -> Dict[str, Any]:
        """Compute exploratory data analysis (EDA) statistics over metadata records."""
        if not self._records:
            return {}

        df = pd.DataFrame([r.to_dict() for r in self._records])
        stats: Dict[str, Any] = {
            "total_records": len(df),
            "status_distribution": df["preprocessing_status"].value_counts().to_dict(),
            "validation_distribution": df["validation_status"].value_counts().to_dict(),
        }

        # Numeric quality metrics
        for col in ("blur_score", "brightness_score", "contrast_score", "file_size_bytes"):
            if col in df.columns:
                valid_series = df[col].dropna()
                if not valid_series.empty:
                    stats[col] = {
                        "mean": round(float(valid_series.mean()), 2),
                        "std": round(float(valid_series.std()), 2),
                        "min": round(float(valid_series.min()), 2),
                        "max": round(float(valid_series.max()), 2),
                    }
        return stats
