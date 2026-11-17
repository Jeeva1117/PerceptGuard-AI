"""PerceptGuard-AI: Video Loading and Ingestion Module.

Staff-level, production-grade video loading module leveraging OpenCV.
Performs path validation, OpenCV VideoCapture initialization, metadata extraction,
and strict diagnostic validation for autonomous driving computer vision pipelines.

Constraints:
- Lightweight CPU execution (Intel Core i3 compatible)
- Avoids decoding entire video into memory
- Strictly cleans up all VideoCapture OS file descriptors
- Guaranteed non-destructive operations
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Set, Union

# Ensure project root is in sys.path when invoked directly
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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


class CorruptedVideoError(VideoOpenError):
    """Raised when the video file cannot provide valid headers or decodable stream."""
    pass


class EmptyVideoError(CorruptedVideoError):
    """Raised when the target video file exists but has a size of 0 bytes."""
    pass


class FirstFrameDecodeError(CorruptedVideoError):
    """Raised when OpenCV opens the container but fails to decode the first frame."""
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
    backend: str = "FFMPEG"

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
            f"backend='{self.backend}', "
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
            self.open(verify_first_frame=True)
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
        """Validate filesystem path existence, file type, file size, and extension."""
        if not self.video_path.exists():
            msg = f"Video file not found at path: '{self.video_path}'"
            logger.error(msg)
            raise VideoNotFoundError(msg)

        if not self.video_path.is_file():
            msg = f"Provided path is a directory, not a video file: '{self.video_path}'"
            logger.error(msg)
            raise InvalidVideoPathError(msg)

        if self.video_path.stat().st_size == 0:
            msg = f"Video file is empty (0 bytes): '{self.video_path}'"
            logger.error(msg)
            raise EmptyVideoError(msg)

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

    def open(self, verify_first_frame: bool = True) -> bool:
        """Initialize OpenCV VideoCapture and confirm the stream is opened and decodable.

        Args:
            verify_first_frame: If True, tests decoding the first frame to confirm decodability
                                and rewinds the stream pointer back to 0.

        Returns:
            bool: True if video is successfully opened and decodable.

        Raises:
            VideoNotFoundError: If the file does not exist.
            InvalidVideoPathError: If the path is a directory.
            EmptyVideoError: If the file exists but has 0 bytes.
            VideoOpenError: If OpenCV fails to open or parse the video container.
            FirstFrameDecodeError: If first frame decoding fails when verify_first_frame is True.
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

        if verify_first_frame:
            ret, frame = self._capture.read()
            if not ret or frame is None or frame.size == 0:
                self.release()
                msg = (
                    f"OpenCV opened the container for '{self.video_path.name}', "
                    "but failed to decode the first video frame (stream may be corrupted or codec unsupported)."
                )
                logger.error(msg)
                raise FirstFrameDecodeError(msg)

            # Rewind capture pointer to frame 0
            seek_ok = self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            if not seek_ok or int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) != 0:
                # Fallback: re-open if seek failed on container
                self._capture.release()
                self._capture = cv2.VideoCapture(str(self.video_path))

        return True

    def is_opened(self) -> bool:
        """Check if OpenCV VideoCapture is currently active and open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def capture(self) -> Optional[cv2.VideoCapture]:
        """Expose the underlying OpenCV VideoCapture object."""
        return self._capture

    def read_frame(self) -> tuple[bool, Optional[Any]]:
        """Read the next sequential frame from the video stream."""
        if not self.is_opened() or self._capture is None:
            raise VideoOpenError(f"Video stream '{self.video_path.name}' is not currently opened.")
        return self._capture.read()

    def grab_frame(self) -> bool:
        """Advance the video stream by one frame without decoding image pixels."""
        if not self.is_opened() or self._capture is None:
            raise VideoOpenError(f"Video stream '{self.video_path.name}' is not currently opened.")
        return self._capture.grab()

    def retrieve_frame(self) -> tuple[bool, Optional[Any]]:
        """Decode and return the frame grabbed by the most recent grab_frame() call."""
        if not self.is_opened() or self._capture is None:
            raise VideoOpenError(f"Video stream '{self.video_path.name}' is not currently opened.")
        return self._capture.retrieve()

    def set_frame_position(self, frame_index: int) -> bool:
        """Seek to a specific 0-indexed source frame position."""
        if not self.is_opened() or self._capture is None:
            raise VideoOpenError(f"Video stream '{self.video_path.name}' is not currently opened.")
        return self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))

    def get_metadata(self) -> VideoMetadata:
        """Extract and validate video stream parameters from OpenCV capture.

        Returns:
            VideoMetadata: Verified metadata object.

        Raises:
            VideoOpenError: If video is not currently opened.
            InvalidVideoMetadataError: If metadata attributes fail physical sanity checks.
        """
        if not self.is_opened():
            self.open(verify_first_frame=True)

        if self._capture is None:
            raise VideoOpenError(f"Cannot retrieve metadata; VideoCapture is uninitialized for '{self.video_path}'.")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = self._capture.get(cv2.CAP_PROP_FOURCC)
        codec = self._decode_fourcc(fourcc)
        backend = self._capture.getBackendName()

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
            backend=backend,
        )

        logger.info(
            "Extracted VideoMetadata: %dx%d @ %.2f FPS, %d frames (%.2fs), codec='%s', backend='%s'",
            width,
            height,
            fps,
            total_frames,
            duration_seconds,
            codec,
            backend,
        )
        return self._metadata

    def verify(self) -> Dict[str, Any]:
        """Perform comprehensive, non-destructive verification of the video file.

        Returns:
            Dict[str, Any]: Verification telemetry with exact status indicators.
        """
        results: Dict[str, Any] = {
            "video_path": str(self.video_path),
            "filename": self.video_path.name,
            "file_exists": False,
            "file_size_bytes": 0,
            "video_open": False,
            "backend": "NONE",
            "codec": "UNKNOWN",
            "resolution": "N/A",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration_seconds": 0.0,
            "first_frame_decode": False,
            "status": "FAIL",
            "error_message": None,
        }

        # 1. Path existence and file type
        if not self.video_path.exists():
            results["error_message"] = f"Video file not found at: '{self.video_path}'"
            return results

        if not self.video_path.is_file():
            results["error_message"] = f"Provided path is not a regular file: '{self.video_path}'"
            return results

        file_size = self.video_path.stat().st_size
        results["file_exists"] = True
        results["file_size_bytes"] = file_size

        if file_size == 0:
            results["error_message"] = f"Video file is empty (0 bytes): '{self.video_path}'"
            return results

        # 2. Open via OpenCV VideoCapture
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            if not cap.isOpened():
                results["error_message"] = f"OpenCV failed to open video stream at: '{self.video_path}'"
                return results

            results["video_open"] = True
            results["backend"] = cap.getBackendName()
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = self._decode_fourcc(fourcc)
            results["codec"] = codec

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            results["width"] = w
            results["height"] = h
            results["resolution"] = f"{w} x {h}"
            results["fps"] = round(fps, 2)
            results["frame_count"] = cnt
            duration = (cnt / fps) if fps > 0.0 else 0.0
            results["duration_seconds"] = round(duration, 2)

            # Metadata validation
            if w <= 0 or h <= 0 or fps <= 0.0 or cnt <= 0:
                results["error_message"] = (
                    f"Invalid stream telemetry: {w}x{h} @ {fps:.2f} FPS, {cnt} frames"
                )
                return results

            # 3. First frame decoding verification
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                results["first_frame_decode"] = False
                results["error_message"] = (
                    "OpenCV opened the container, but failed to decode the first video frame."
                )
                return results

            results["first_frame_decode"] = True
            results["status"] = "PASS"
            return results

        except Exception as e:
            results["error_message"] = f"Exception during video verification: {e}"
            return results
        finally:
            cap.release()

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
        self.open(verify_first_frame=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; guarantees VideoCapture resource release."""
        self.release()

    def __del__(self) -> None:
        """Finalizer fallback to ensure OS file handles are never leaked."""
        self.release()


# ==============================================================================
# Verification Runner & CLI
# ==============================================================================

def verify_video_file(
    video_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Execute video verification and print standard diagnostics to stdout.

    Args:
        video_path: Path to target video file (or None to use default).
        config_path: Path to video_config.yaml (or None).

    Returns:
        bool: True if video verification passed, False otherwise.
    """
    resolved_path: Optional[Path] = None
    project_root = Path(__file__).resolve().parents[2]

    if video_path:
        resolved_path = Path(video_path)
        if not resolved_path.is_absolute():
            resolved_path = (project_root / resolved_path).resolve()
        else:
            resolved_path = resolved_path.resolve()
    else:
        # 1. Try config default
        cfg_file = Path(config_path).resolve() if config_path else project_root / "config" / "video_config.yaml"
        if cfg_file.is_file():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                def_v = cfg.get("video", {}).get("default_video")
                if def_v:
                    candidate = (project_root / def_v).resolve() if not Path(def_v).is_absolute() else Path(def_v).resolve()
                    if candidate.is_file():
                        resolved_path = candidate
            except Exception:
                pass

        # 2. Try datasets/raw directory
        if not resolved_path or not resolved_path.is_file():
            raw_dir = project_root / "datasets" / "raw"
            if raw_dir.is_dir():
                known_default = raw_dir / "converted_026c7465-309f6d33.mp4"
                if known_default.is_file():
                    resolved_path = known_default
                else:
                    vids = [
                        f for f in sorted(raw_dir.iterdir())
                        if f.is_file() and f.suffix.lower() in VideoLoader.DEFAULT_SUPPORTED_EXTENSIONS
                    ]
                    if vids:
                        resolved_path = vids[0]

        if not resolved_path:
            resolved_path = project_root / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4"

    loader = VideoLoader(resolved_path, config_path=config_path)
    res = loader.verify()

    file_found_str = "FOUND" if res["file_exists"] else "NOT FOUND"
    video_open_str = "SUCCESS" if res["video_open"] else "FAILED"
    codec_backend_str = (
        f"{res['codec']} / {res['backend']}"
        if res["video_open"]
        else "UNKNOWN / NONE"
    )
    resolution_str = res["resolution"] if res["video_open"] else "N/A"
    fps_str = f"{res['fps']:.2f}" if res["video_open"] and res["fps"] > 0 else "N/A"
    frame_count_str = f"{res['frame_count']}" if res["video_open"] and res["frame_count"] > 0 else "N/A"
    duration_str = f"{res['duration_seconds']:.2f}s" if res["video_open"] and res["duration_seconds"] > 0 else "N/A"
    first_frame_str = "SUCCESS" if res["first_frame_decode"] else "FAILED"
    status_str = res["status"]

    print(f"VIDEO FILE: {file_found_str}")
    print(f"VIDEO OPEN: {video_open_str}")
    print(f"CODEC / BACKEND: {codec_backend_str}")
    print(f"RESOLUTION: {resolution_str}")
    print(f"FPS: {fps_str}")
    print(f"FRAME COUNT: {frame_count_str}")
    print(f"DURATION: {duration_str}")
    print(f"FIRST FRAME DECODE: {first_frame_str}")
    print(f"VIDEO LOADING STATUS: {status_str}")

    return res["status"] == "PASS"


def main() -> None:
    """CLI Entrypoint for video loader and verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI - Video Loader & Verification Utility"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run comprehensive video stream verification on the target file.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to target video file (default: default_video from video_config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config/video_config.yaml)",
    )
    args = parser.parse_args()

    if args.verify or len(sys.argv) == 1:
        success = verify_video_file(video_path=args.video_path, config_path=args.config)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
