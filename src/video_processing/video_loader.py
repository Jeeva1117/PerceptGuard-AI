"""PerceptGuard-AI: Video Loading and Ingestion Module.

Staff-level, production-grade video loading module leveraging OpenCV.
Performs path validation, OpenCV VideoCapture initialization, metadata extraction,
and strict diagnostic validation for autonomous driving computer vision pipelines.

Constraints:
- Lightweight CPU execution (Intel Core i3 compatible)
- Avoids decoding entire video into memory
- Strictly cleans up all VideoCapture OS file descriptors
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
import cv2
import yaml

logger = logging.getLogger("PerceptGuard.VideoLoader")


# ==============================================================================
# Custom Domain Exceptions
# ==============================================================================

class VideoProcessingError(Exception):
    """Base exception for all video processing failures."""
    pass


class VideoNotFoundError(VideoProcessingError):
    """Raised when the target video file path does not exist on disk."""
    pass


class InvalidVideoPathError(VideoProcessingError):
    """Raised when path is invalid (e.g., directory instead of a regular file)."""
    pass


class VideoOpenError(VideoProcessingError):
    """Raised when OpenCV VideoCapture fails to open the video file."""
    pass


class CorruptedVideoError(VideoProcessingError):
    """Raised when the video file cannot provide valid headers or decodable stream."""
    pass


class InvalidVideoMetadataError(VideoProcessingError):
    """Raised when video metadata violates physical feasibility (e.g., zero FPS)."""
    pass


# ==============================================================================
# Metadata Data Contract
# ==============================================================================

@dataclass(frozen=True)
class VideoMetadata:
    """Immutable data contract encapsulating verified video stream metadata."""
    file_path: Path
    filename: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary with string path representation."""
        data = asdict(self)
        data["file_path"] = str(self.file_path)
        return data

    def __str__(self) -> str:
        return (
            f"VideoMetadata(filename='{self.filename}', "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, "
            f"frames={self.total_frames}, "
            f"duration={self.duration_seconds:.2f}s, "
            f"codec='{self.codec}', "
            f"status={'VALID' if self.is_valid else 'INVALID'})"
        )


# ==============================================================================
# VideoLoader Implementation
# ==============================================================================

class VideoLoader:
    """Robust OpenCV video loader with metadata verification and resource management.

    Supports both explicit open/release semantics and Python context manager protocol:
        with VideoLoader("path/to/video.mp4") as loader:
            metadata = loader.get_metadata()

    Attributes:
        video_path (Path): Path to the video file.
        metadata (Optional[VideoMetadata]): Extracted metadata if opened and validated.
    """

    DEFAULT_SUPPORTED_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        video_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        validate_on_init: bool = False,
    ) -> None:
        """Initialize VideoLoader with video file path and optional configuration.

        Args:
            video_path: Path to the target video file.
            config_path: Optional path to a YAML configuration file.
            validate_on_init: If True, immediately opens, validates, and stores metadata.
        """
        self.video_path = Path(video_path).resolve()
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._supported_extensions = self.DEFAULT_SUPPORTED_EXTENSIONS.copy()
        self._min_fps = 1.0
        self._max_fps = 120.0
        self._min_frames = 1
        self._min_width = 16
        self._min_height = 16

        self._load_config(config_path)

        if validate_on_init:
            self.open()
            self.get_metadata()

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> None:
        """Load configuration options if a valid config file is provided."""
        if not config_path:
            # Fallback to default project config location if present
            default_cfg = Path(__file__).resolve().parent.parent.parent / "config" / "video_config.yaml"
            if default_cfg.is_file():
                config_path = default_cfg

        if config_path and Path(config_path).is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                video_cfg = cfg.get("video", {})
                if "supported_extensions" in video_cfg:
                    self._supported_extensions = {ext.lower() for ext in video_cfg["supported_extensions"]}
                val_cfg = video_cfg.get("validation", {})
                self._min_fps = float(val_cfg.get("min_fps", self._min_fps))
                self._max_fps = float(val_cfg.get("max_fps", self._max_fps))
                self._min_frames = int(val_cfg.get("min_frames", self._min_frames))
                self._min_width = int(val_cfg.get("min_width", self._min_width))
                self._min_height = int(val_cfg.get("min_height", self._min_height))
            except Exception as e:
                logger.warning("Failed to parse config file %s: %s. Using defaults.", config_path, e)

    def _verify_path_integrity(self) -> None:
        """Validate filesystem path existence, file type, and extension."""
        if not self.video_path.exists():
            msg = f"Video file not found at path: '{self.video_path}'"
            logger.error(msg)
            raise VideoNotFoundError(msg)

        if not self.video_path.is_file():
            msg = f"Provided path is a directory, not a video file: '{self.video_path}'"
            logger.error(msg)
            raise InvalidVideoPathError(msg)

        ext = self.video_path.suffix.lower()
        if ext not in self._supported_extensions:
            logger.warning(
                "File extension '%s' is not in supported list (%s). Attempting open anyway.",
                ext,
                self._supported_extensions,
            )

    @staticmethod
    def _decode_fourcc(fourcc_code: float) -> str:
        """Safely decode an integer OpenCV fourcc code into a 4-character string."""
        try:
            int_code = int(fourcc_code)
            if int_code <= 0:
                return "UNKNOWN"
            chars = [chr((int_code >> 8 * i) & 0xFF) for i in range(4)]
            # Filter non-printable ASCII
            codec_str = "".join(c if 32 <= ord(c) <= 126 else "?" for c in chars)
            return codec_str.strip() or "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def open(self) -> bool:
        """Initialize OpenCV VideoCapture and confirm the stream is opened.

        Returns:
            bool: True if video is successfully opened and decodable.

        Raises:
            VideoNotFoundError: If the file does not exist.
            InvalidVideoPathError: If the path is a directory.
            VideoOpenError: If OpenCV fails to open or parse the video container.
        """
        self._verify_path_integrity()

        # If already opened, release before re-opening
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()

        logger.info("Initializing VideoCapture for: %s", self.video_path.name)
        self._capture = cv2.VideoCapture(str(self.video_path))

        if not self._capture.isOpened():
            msg = f"OpenCV failed to open video stream at: '{self.video_path}'"
            logger.error(msg)
            self.release()
            raise VideoOpenError(msg)

        return True

    def is_opened(self) -> bool:
        """Check if OpenCV VideoCapture is currently active and open."""
        return self._capture is not None and self._capture.isOpened()

    def get_metadata(self) -> VideoMetadata:
        """Extract and validate video stream parameters from OpenCV capture.

        Returns:
            VideoMetadata: Verified metadata object.

        Raises:
            VideoOpenError: If video is not currently opened.
            InvalidVideoMetadataError: If metadata attributes fail physical sanity checks.
        """
        if not self.is_opened():
            self.open()

        if self._capture is None:
            raise VideoOpenError(f"Cannot retrieve metadata; VideoCapture is uninitialized for '{self.video_path}'.")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = self._capture.get(cv2.CAP_PROP_FOURCC)
        codec = self._decode_fourcc(fourcc)

        # Validate metadata integrity
        if fps <= 0.0 or fps < self._min_fps or fps > self._max_fps:
            msg = f"Invalid video FPS: {fps:.2f} (must be between {self._min_fps} and {self._max_fps}) for '{self.video_path.name}'"
            logger.error(msg)
            raise InvalidVideoMetadataError(msg)

        if total_frames < self._min_frames:
            msg = f"Invalid video frame count: {total_frames} (must be >= {self._min_frames}) for '{self.video_path.name}'"
            logger.error(msg)
            raise InvalidVideoMetadataError(msg)

        if width < self._min_width or height < self._min_height:
            msg = f"Invalid resolution: {width}x{height} (minimum {self._min_width}x{self._min_height}) for '{self.video_path.name}'"
            logger.error(msg)
            raise InvalidVideoMetadataError(msg)

        duration_seconds = total_frames / fps if fps > 0 else 0.0

        self._metadata = VideoMetadata(
            file_path=self.video_path,
            filename=self.video_path.name,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
            codec=codec,
            is_valid=True,
        )

        logger.info(
            "Extracted VideoMetadata: %dx%d @ %.2f FPS, %d frames (%.2fs), codec='%s'",
            width,
            height,
            fps,
            total_frames,
            duration_seconds,
            codec,
        )
        return self._metadata

    def release(self) -> None:
        """Safely release the underlying OpenCV VideoCapture resource."""
        if self._capture is not None:
            try:
                if self._capture.isOpened():
                    self._capture.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture resource: %s", e)
            finally:
                self._capture = None
                logger.debug("Released VideoCapture resources for: %s", self.video_path.name)

    def __enter__(self) -> "VideoLoader":
        """Context manager entry; opens video stream."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; guarantees VideoCapture resource release."""
        self.release()

    def __del__(self) -> None:
        """Finalizer fallback to ensure OS file handles are never leaked."""
        self.release()

