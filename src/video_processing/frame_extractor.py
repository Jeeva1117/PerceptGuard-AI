"""PerceptGuard-AI: Frame Extraction Engine.

Production-grade, CPU-optimized video frame extraction engine for autonomous driving
perception research. Implements deterministic time-based frame sampling, sequential
decoding, resource management, and strict validation.

Key Architectural Guarantees:
- Lightweight CPU execution (Intel Core i3 compatible)
- Processes one frame at a time; never buffers video sequences into memory
- Leverages OpenCV grab() for fast frame skipping without expensive decompression
- Decouples extracted frame ID (1-indexed) from source video frame index (0-indexed)
- Guarantees deterministic timestamp calculation: t = source_frame_index / source_fps
- Zero frame duplication even if extraction FPS exceeds source FPS
- Strict resource release under both normal and exceptional termination
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import yaml

from src.video_processing.video_loader import (
    VideoLoader,
    VideoMetadata,
    VideoOpenError,
    VideoProcessingError,
)

logger = logging.getLogger("PerceptGuard.FrameExtractor")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class FrameExtractionError(VideoProcessingError):
    """Base exception for all frame extraction failures."""
    pass


class InvalidExtractionConfigError(FrameExtractionError):
    """Raised when extraction parameters violate physical bounds or invariants."""
    pass


class FrameSaveError(FrameExtractionError):
    """Raised when OpenCV fails to encode or write a frame to persistent storage."""
    pass


class VideoReadError(FrameExtractionError):
    """Raised when an unrecoverable decoding error occurs during stream consumption."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class ExtractedFrameInfo:
    """Immutable data contract encapsulating an extracted video frame's metadata.

    Attributes:
        extracted_frame_id: 1-indexed sequential extracted ID (1, 2, 3...).
        source_frame_index: 0-indexed position in source video stream (0, 6, 12...).
        timestamp_seconds: Physical time in seconds (source_frame_index / source_fps).
        image_filename: Basename of the saved image file (e.g., 'frame_000001.jpg').
        image_path: Absolute filesystem path to the saved image file.
    """
    extracted_frame_id: int
    source_frame_index: int
    timestamp_seconds: float
    image_filename: str
    image_path: Path

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame record to dictionary format."""
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["timestamp_seconds"] = round(self.timestamp_seconds, 4)
        return data

    def __str__(self) -> str:
        return (
            f"ExtractedFrameInfo(id={self.extracted_frame_id}, "
            f"source_idx={self.source_frame_index}, "
            f"time={self.timestamp_seconds:.3f}s, "
            f"file='{self.image_filename}')"
        )


@dataclass(frozen=True)
class ExtractionSummary:
    """Immutable summary record for a completed frame extraction session.

    Attributes:
        video_path: Absolute path to the source video file.
        original_fps: Source video frame rate.
        target_fps: Configured sampling frame rate.
        total_source_frames: Total frames reported in source container.
        extracted_frames_count: Actual count of frames successfully extracted.
        output_directory: Filesystem directory containing extracted frames.
        first_timestamp_seconds: Timestamp of first extracted frame.
        last_timestamp_seconds: Timestamp of last extracted frame.
        extracted_frames: Ordered tuple of all extracted frame records.
        elapsed_time_seconds: Execution wall-clock duration in seconds.
    """
    video_path: Path
    original_fps: float
    target_fps: float
    total_source_frames: int
    extracted_frames_count: int
    output_directory: Path
    first_timestamp_seconds: Optional[float]
    last_timestamp_seconds: Optional[float]
    extracted_frames: Tuple[ExtractedFrameInfo, ...]
    elapsed_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize extraction summary to dictionary format."""
        return {
            "video_path": str(self.video_path),
            "original_fps": round(self.original_fps, 2),
            "target_fps": round(self.target_fps, 2),
            "total_source_frames": self.total_source_frames,
            "extracted_frames_count": self.extracted_frames_count,
            "output_directory": str(self.output_directory),
            "first_timestamp_seconds": (
                round(self.first_timestamp_seconds, 4)
                if self.first_timestamp_seconds is not None
                else None
            ),
            "last_timestamp_seconds": (
                round(self.last_timestamp_seconds, 4)
                if self.last_timestamp_seconds is not None
                else None
            ),
            "elapsed_time_seconds": round(self.elapsed_time_seconds, 3),
            "sample_filenames": [f.image_filename for f in self.extracted_frames[:5]],
        }

    def __str__(self) -> str:
        return (
            f"ExtractionSummary(video='{self.video_path.name}', "
            f"extracted={self.extracted_frames_count}/{self.total_source_frames} frames, "
            f"target_fps={self.target_fps:.1f}, "
            f"time_span=[{self.first_timestamp_seconds or 0.0:.2f}s - {self.last_timestamp_seconds or 0.0:.2f}s], "
            f"elapsed={self.elapsed_time_seconds:.2f}s)"
        )


# ==============================================================================
# FrameExtractor Engine
# ==============================================================================

class FrameExtractor:
    """Robust, memory-efficient video frame extraction engine.

    Extracts frames sequentially from an autonomous-driving video stream,
    applying deterministic sampling rate conversion, sequential naming,
    integrity verification, and guaranteed resource cleanup.

    Example:
        >>> with FrameExtractor("datasets/raw/sample.mp4", extraction_fps=5.0) as extractor:
        ...     summary = extractor.extract_frames()
        ...     print(summary.extracted_frames_count)
    """

    SUPPORTED_IMAGE_FORMATS: Set[str] = {"jpg", "jpeg", "png", "webp"}

    def __init__(
        self,
        video_source: Union[str, Path, VideoLoader],
        output_dir: Optional[Union[str, Path]] = None,
        extraction_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        filename_prefix: Optional[str] = None,
        filename_padding: Optional[int] = None,
        image_format: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        overwrite: Optional[bool] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize FrameExtractor with video source and optional configurations.

        Args:
            video_source: Video file path (str/Path) or an existing VideoLoader instance.
            output_dir: Destination directory for extracted image files.
            extraction_fps: Desired sampling rate in FPS (e.g. 5.0).
            max_frames: Optional upper limit on extracted frame count.
            filename_prefix: Prefix string for saved image filenames.
            filename_padding: Integer zero-padding width for extracted frame indices.
            image_format: Output image format ('jpg', 'jpeg', 'png', 'webp').
            jpeg_quality: Compression quality for JPEG output (1-100).
            overwrite: If True, existing image files in output directory will be overwritten.
            config_path: Path to YAML configuration file overriding default settings.
        """
        # Determine whether FrameExtractor owns the loader instance
        if isinstance(video_source, VideoLoader):
            self._loader = video_source
            self._owns_loader = False
            self.video_path = self._loader.video_path
        else:
            self.video_path = Path(video_source).resolve()
            self._loader = VideoLoader(self.video_path, config_path=config_path)
            self._owns_loader = True

        # Configuration defaults
        self._output_dir = Path("datasets/processed/driving_sample/frames")
        self._extraction_fps: Optional[float] = 5.0
        self._max_frames: Optional[int] = None
        self._filename_prefix: str = "frame_"
        self._filename_padding: int = 6
        self._image_format: str = "jpg"
        self._jpeg_quality: int = 95
        self._overwrite: bool = True

        # Load values from configuration file
        self._load_config(config_path)

        # Explicit parameter overrides take precedence over config file
        if output_dir is not None:
            self._output_dir = Path(output_dir).resolve()
        else:
            self._output_dir = self._output_dir.resolve()

        if extraction_fps is not None:
            self._extraction_fps = float(extraction_fps)

        if max_frames is not None:
            self._max_frames = int(max_frames)

        if filename_prefix is not None:
            self._filename_prefix = str(filename_prefix)

        if filename_padding is not None:
            self._filename_padding = int(filename_padding)

        if image_format is not None:
            self._image_format = image_format.lower().lstrip(".")

        if jpeg_quality is not None:
            self._jpeg_quality = int(jpeg_quality)

        if overwrite is not None:
            self._overwrite = bool(overwrite)

        self._validate_parameters()
        self._summary: Optional[ExtractionSummary] = None

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> None:
        """Parse configuration parameters from video_config.yaml."""
        if not config_path:
            default_cfg = Path(__file__).resolve().parent.parent.parent / "config" / "video_config.yaml"
            if default_cfg.is_file():
                config_path = default_cfg

        if config_path and Path(config_path).is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                ext_cfg = cfg.get("frame_extraction", {})
                if "output_directory" in ext_cfg:
                    self._output_dir = Path(ext_cfg["output_directory"])
                if "extraction_fps" in ext_cfg:
                    self._extraction_fps = float(ext_cfg["extraction_fps"])
                if "max_frames" in ext_cfg and ext_cfg["max_frames"] is not None:
                    self._max_frames = int(ext_cfg["max_frames"])
                if "filename_prefix" in ext_cfg:
                    self._filename_prefix = str(ext_cfg["filename_prefix"])
                if "filename_padding" in ext_cfg:
                    self._filename_padding = int(ext_cfg["filename_padding"])
                if "image_format" in ext_cfg:
                    self._image_format = str(ext_cfg["image_format"]).lower().lstrip(".")
                if "jpeg_quality" in ext_cfg:
                    self._jpeg_quality = int(ext_cfg["jpeg_quality"])
                if "overwrite" in ext_cfg:
                    self._overwrite = bool(ext_cfg["overwrite"])
            except Exception as e:
                logger.warning("Failed to load config %s: %s. Using default settings.", config_path, e)

    def _validate_parameters(self) -> None:
        """Enforce physical sanity and invariant constraints on parameters."""
        if self._extraction_fps is not None and self._extraction_fps <= 0.0:
            raise InvalidExtractionConfigError(
                f"extraction_fps must be positive, got: {self._extraction_fps}"
            )

        if self._max_frames is not None and self._max_frames <= 0:
            raise InvalidExtractionConfigError(
                f"max_frames must be a positive integer, got: {self._max_frames}"
            )

        if self._filename_padding < 1 or self._filename_padding > 12:
            raise InvalidExtractionConfigError(
                f"filename_padding must be between 1 and 12, got: {self._filename_padding}"
            )

        if self._image_format not in self.SUPPORTED_IMAGE_FORMATS:
            raise InvalidExtractionConfigError(
                f"Unsupported image_format '{self._image_format}'. Supported: {self.SUPPORTED_IMAGE_FORMATS}"
            )

        if not (1 <= self._jpeg_quality <= 100):
            raise InvalidExtractionConfigError(
                f"jpeg_quality must be in range [1, 100], got: {self._jpeg_quality}"
            )

    @property
    def output_directory(self) -> Path:
        """Destination directory for extracted frame images."""
        return self._output_dir

    @property
    def extraction_fps(self) -> Optional[float]:
        """Configured extraction frame rate."""
        return self._extraction_fps

    def _compute_target_source_indices(
        self, source_fps: float, total_source_frames: int
    ) -> List[int]:
        """Compute strictly monotonic, non-duplicate source frame indices.

        Mathematical Sampling Formulation:
        Let S = source_fps / target_fps.
        For k = 0, 1, 2, ...:
            target_source_index = round(k * S)

        Guarantees:
        - When target_fps >= source_fps, upsampling by duplicating frames is forbidden;
          every sequential frame is extracted once (step = 1).
        - When target_fps < source_fps, S > 1.0, ensuring round((k+1)*S) > round(k*S).
          Indices are strictly monotonic with zero duplicates.
        - Stops at min(total_source_frames - 1, max_frames).
        """
        if self._extraction_fps is None or self._extraction_fps <= 0.0:
            step = 1.0
            effective_target_fps = source_fps
        elif self._extraction_fps >= source_fps:
            logger.info(
                "Requested extraction FPS (%.2f) >= source FPS (%.2f). "
                "Clamping to 1:1 extraction without frame duplication.",
                self._extraction_fps,
                source_fps,
            )
            step = 1.0
            effective_target_fps = source_fps
        else:
            step = source_fps / self._extraction_fps
            effective_target_fps = self._extraction_fps

        target_indices: List[int] = []
        k = 0
        last_index = -1

        while True:
            idx = int(round(k * step))
            if idx >= total_source_frames:
                break
            if idx > last_index:
                target_indices.append(idx)
                last_index = idx
                if self._max_frames and len(target_indices) >= self._max_frames:
                    break
            k += 1

        logger.debug(
            "Computed %d target indices from %d total frames (source_fps=%.2f, target_fps=%.2f)",
            len(target_indices),
            total_source_frames,
            source_fps,
            effective_target_fps,
        )
        return target_indices

    def save_frame(self, frame: np.ndarray, filepath: Union[str, Path]) -> None:
        """Write an extracted frame buffer to disk as an image file.

        Args:
            frame: BGR numpy image array.
            filepath: Destination filesystem path.

        Raises:
            FrameSaveError: If directory creation or OpenCV image encoding fails.
        """
        target_path = Path(filepath)

        if frame is None or frame.size == 0:
            raise FrameSaveError(f"Cannot save empty or uninitialized frame to '{target_path}'.")

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Failed to create output directory '{target_path.parent}': {e}"
            logger.error(msg)
            raise FrameSaveError(msg) from e

        if target_path.exists() and not self._overwrite:
            logger.debug("Skipping existing file '%s' (overwrite=False)", target_path.name)
            return

        encode_params: List[int] = []
        if self._image_format in ("jpg", "jpeg"):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        elif self._image_format == "png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif self._image_format == "webp":
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, self._jpeg_quality]

        try:
            success = cv2.imwrite(str(target_path), frame, encode_params)
        except Exception as e:
            msg = f"OpenCV exception during imwrite to '{target_path}': {e}"
            logger.error(msg)
            raise FrameSaveError(msg) from e

        if not success or not target_path.is_file() or target_path.stat().st_size == 0:
            msg = f"cv2.imwrite failed to persist valid image file at '{target_path}'"
            logger.error(msg)
            raise FrameSaveError(msg)

    def extract_frames(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExtractionSummary:
        """Execute sequential frame extraction based on the configured sampling policy.

        Args:
            progress_callback: Optional callable(current_extracted, target_total)
                               invoked periodically for telemetry reporting.

        Returns:
            ExtractionSummary: Immutable audit record of the extraction execution.

        Raises:
            VideoProcessingError: If video stream cannot be opened or decoded.
            FrameSaveError: If image encoding or disk write fails.
        """
        start_time = time.perf_counter()

        # Ensure video stream is open and valid metadata is extracted
        if not self._loader.is_opened():
            self._loader.open()

        metadata: VideoMetadata = self._loader.get_metadata()
        source_fps = metadata.fps
        total_source_frames = metadata.total_frames

        # Ensure destination directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Compute sampling schedule
        target_indices = self._compute_target_source_indices(source_fps, total_source_frames)
        total_targets = len(target_indices)

        if total_targets == 0:
            logger.warning("No target frames computed for video '%s'", self.video_path.name)
            summary = ExtractionSummary(
                video_path=self.video_path,
                original_fps=source_fps,
                target_fps=self._extraction_fps or source_fps,
                total_source_frames=total_source_frames,
                extracted_frames_count=0,
                output_directory=self._output_dir,
                first_timestamp_seconds=None,
                last_timestamp_seconds=None,
                extracted_frames=(),
                elapsed_time_seconds=time.perf_counter() - start_time,
            )
            self._summary = summary
            return summary

        target_set = set(target_indices)
        extracted_records: List[ExtractedFrameInfo] = []

        logger.info(
            "Starting extraction for '%s': target_fps=%.2f (source=%.2f FPS, %d frames), extracting %d frames",
            self.video_path.name,
            self._extraction_fps or source_fps,
            source_fps,
            total_source_frames,
            total_targets,
        )

        extracted_counter = 0
        current_source_idx = 0
        max_target_index = target_indices[-1]

        try:
            while current_source_idx <= max_target_index:
                if current_source_idx in target_set:
                    # Target frame: decode fully into memory
                    ret, frame = self._loader.read_frame()
                    if not ret or frame is None:
                        logger.warning(
                            "End-of-stream reached unexpectedly at source frame index %d",
                            current_source_idx,
                        )
                        break

                    extracted_counter += 1
                    # Documented timestamp convention: t = source_frame_index / source_fps
                    timestamp_sec = float(current_source_idx) / source_fps

                    # Filename: 1-indexed extracted frame ID, zero-padded
                    filename = (
                        f"{self._filename_prefix}"
                        f"{extracted_counter:0{self._filename_padding}d}."
                        f"{self._image_format}"
                    )
                    image_path = self._output_dir / filename

                    # Persist frame to disk
                    self.save_frame(frame, image_path)

                    record = ExtractedFrameInfo(
                        extracted_frame_id=extracted_counter,
                        source_frame_index=current_source_idx,
                        timestamp_seconds=timestamp_sec,
                        image_filename=filename,
                        image_path=image_path,
                    )
                    extracted_records.append(record)

                    # Trigger telemetry callback
                    if progress_callback:
                        progress_callback(extracted_counter, total_targets)

                    # Check max frames termination
                    if self._max_frames and extracted_counter >= self._max_frames:
                        logger.info("Reached configured max_frames limit (%d)", self._max_frames)
                        break

                    # Deliberately release frame from memory
                    del frame
                else:
                    # Non-target frame: skip without decompression for maximum speed
                    grabbed = self._loader.grab_frame()
                    if not grabbed:
                        logger.warning(
                            "End-of-stream reached during grab at source frame index %d",
                            current_source_idx,
                        )
                        break

                current_source_idx += 1

        finally:
            # Ensure resources are released if FrameExtractor owns the loader
            if self._owns_loader:
                self.release_resources()

        elapsed = time.perf_counter() - start_time
        first_ts = extracted_records[0].timestamp_seconds if extracted_records else None
        last_ts = extracted_records[-1].timestamp_seconds if extracted_records else None

        self._summary = ExtractionSummary(
            video_path=self.video_path,
            original_fps=source_fps,
            target_fps=self._extraction_fps or source_fps,
            total_source_frames=total_source_frames,
            extracted_frames_count=len(extracted_records),
            output_directory=self._output_dir,
            first_timestamp_seconds=first_ts,
            last_timestamp_seconds=last_ts,
            extracted_frames=tuple(extracted_records),
            elapsed_time_seconds=elapsed,
        )

        logger.info(
            "Extraction completed successfully: extracted %d/%d frames in %.2fs (%.1f frames/s)",
            len(extracted_records),
            total_targets,
            elapsed,
            (len(extracted_records) / elapsed) if elapsed > 0 else 0.0,
        )

        return self._summary

    def get_extraction_summary(self) -> Optional[ExtractionSummary]:
        """Retrieve summary record from the most recent extraction execution."""
        return self._summary

    def release_resources(self) -> None:
        """Release underlying video capture file descriptors idempotently."""
        if self._loader is not None and self._loader.is_opened():
            self._loader.release()

    def __enter__(self) -> "FrameExtractor":
        """Context manager entry; returns extractor instance."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; guarantees VideoCapture resource cleanup."""
        self.release_resources()

    def __del__(self) -> None:
        """Finalizer fallback to guarantee release of OS file handles."""
        self.release_resources()
