"""PerceptGuard-AI: Frame Preprocessing Pipeline.

Production-grade, CPU-optimized image preprocessing module for autonomous driving
perception research. Implements strict image validation, aspect-ratio-preserving
resizing, color-space management, and lightweight visual quality diagnostics.

Key Architectural Guarantees:
- Lightweight CPU execution (Intel Core i3 compatible)
- Processes one image at a time with immediate heap release; zero full-dataset RAM buffering
- Decouples raw extracted frames from processed outputs (raw inputs remain strictly untouched)
- Preserves object aspect ratios via centered letterbox padding (prevents vehicle/lane distortion)
- Provides diagnostic image quality indicators (Laplacian blur, brightness, contrast)
- Distinguishes heuristic quality indicators from ground-truth detection reliability labels
- Comprehensive error handling for missing, corrupted, truncated, or zero-byte images
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import yaml

from src.video_processing.video_loader import VideoProcessingError

logger = logging.getLogger("PerceptGuard.FramePreprocessor")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class PreprocessingError(VideoProcessingError):
    """Base exception for all frame preprocessing failures."""
    pass


class CorruptedImageError(PreprocessingError):
    """Raised when an image file exists but contains corrupted or undecodable data."""
    pass


class InvalidImageDimensionsError(PreprocessingError):
    """Raised when image dimensions violate physical spatial feasibility."""
    pass


class InvalidColorChannelError(PreprocessingError):
    """Raised when image color channels cannot be mapped to the expected format."""
    pass


class ImageSaveError(PreprocessingError):
    """Raised when OpenCV fails to persist the processed frame to storage."""
    pass


class InvalidPreprocessingConfigError(PreprocessingError):
    """Raised when configuration parameters violate physical bounds or invariants."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class PreprocessedFrameRecord:
    """Immutable audit contract for an individual processed frame.

    Attributes:
        input_filename: Original basename of the input extracted frame.
        input_path: Absolute filesystem path of the source image.
        output_filename: Basename of the preprocessed output image.
        output_path: Absolute filesystem path of the saved output image.
        original_width: Spatial width of source image in pixels.
        original_height: Spatial height of source image in pixels.
        processed_width: Spatial width of preprocessed image in pixels.
        processed_height: Spatial height of preprocessed image in pixels.
        blur_score: Laplacian variance sharpness metric (higher = sharper).
        brightness_score: Mean grayscale intensity in range [0.0, 255.0].
        contrast_score: Grayscale standard deviation (RMS contrast).
        is_valid: True if image passed all validation checks and was processed.
        error_message: Diagnostic failure description if processing failed.
    """
    input_filename: str
    input_path: Path
    output_filename: str
    output_path: Path
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    blur_score: Optional[float]
    brightness_score: Optional[float]
    contrast_score: Optional[float]
    is_valid: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame record to dictionary format."""
        data = asdict(self)
        data["input_path"] = str(self.input_path)
        data["output_path"] = str(self.output_path)
        if self.blur_score is not None:
            data["blur_score"] = round(self.blur_score, 2)
        if self.brightness_score is not None:
            data["brightness_score"] = round(self.brightness_score, 2)
        if self.contrast_score is not None:
            data["contrast_score"] = round(self.contrast_score, 2)
        return data

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else f"FAILED: {self.error_message}"
        return (
            f"PreprocessedFrameRecord(file='{self.input_filename}', "
            f"orig={self.original_width}x{self.original_height}, "
            f"proc={self.processed_width}x{self.processed_height}, "
            f"blur={self.blur_score:.1f if self.blur_score is not None else -1}, "
            f"bright={self.brightness_score:.1f if self.brightness_score is not None else -1}, "
            f"status={status})"
        )


@dataclass(frozen=True)
class PreprocessingSummary:
    """Immutable summary record for a batch directory preprocessing execution.

    Attributes:
        input_directory: Directory containing source raw extracted frames.
        output_directory: Directory containing preprocessed output frames.
        total_frames: Total number of input frames discovered.
        successful_frames: Number of frames successfully validated and preprocessed.
        failed_frames: Number of frames that failed validation or processing.
        target_dimensions: Configured (width, height) output dimensions.
        mean_blur_score: Mean Laplacian blur score across valid frames.
        mean_brightness_score: Mean brightness across valid frames.
        mean_contrast_score: Mean contrast across valid frames.
        records: Ordered tuple of all processed frame records.
        elapsed_time_seconds: Total wall-clock execution time in seconds.
    """
    input_directory: Path
    output_directory: Path
    total_frames: int
    successful_frames: int
    failed_frames: int
    target_dimensions: Tuple[int, int]
    mean_blur_score: Optional[float]
    mean_brightness_score: Optional[float]
    mean_contrast_score: Optional[float]
    records: Tuple[PreprocessedFrameRecord, ...]
    elapsed_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessing summary to dictionary format."""
        return {
            "input_directory": str(self.input_directory),
            "output_directory": str(self.output_directory),
            "total_frames": self.total_frames,
            "successful_frames": self.successful_frames,
            "failed_frames": self.failed_frames,
            "target_dimensions": list(self.target_dimensions),
            "mean_blur_score": round(self.mean_blur_score, 2) if self.mean_blur_score is not None else None,
            "mean_brightness_score": round(self.mean_brightness_score, 2) if self.mean_brightness_score is not None else None,
            "mean_contrast_score": round(self.mean_contrast_score, 2) if self.mean_contrast_score is not None else None,
            "elapsed_time_seconds": round(self.elapsed_time_seconds, 3),
            "sample_processed_files": [r.output_filename for r in self.records[:5] if r.is_valid],
        }

    def __str__(self) -> str:
        return (
            f"PreprocessingSummary(total={self.total_frames}, "
            f"successful={self.successful_frames}, "
            f"failed={self.failed_frames}, "
            f"target={self.target_dimensions[0]}x{self.target_dimensions[1]}, "
            f"mean_blur={self.mean_blur_score:.1f if self.mean_blur_score is not None else -1}, "
            f"mean_bright={self.mean_brightness_score:.1f if self.mean_brightness_score is not None else -1}, "
            f"elapsed={self.elapsed_time_seconds:.2f}s)"
        )


# ==============================================================================
# FramePreprocessor Implementation
# ==============================================================================

class FramePreprocessor:
    """Robust, memory-efficient driving frame preprocessing pipeline.

    Applies input validation, aspect-ratio-preserving letterboxing, color-space
    management, and visual quality metric calculation for autonomous driving datasets.
    """

    SUPPORTED_INPUT_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        preserve_aspect_ratio: Optional[bool] = None,
        padding_color: Optional[Tuple[int, int, int]] = None,
        color_mode: Optional[str] = None,
        image_format: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        calculate_blur: Optional[bool] = None,
        calculate_brightness: Optional[bool] = None,
        calculate_contrast: Optional[bool] = None,
        overwrite: Optional[bool] = None,
    ) -> None:
        """Initialize FramePreprocessor with configuration or parameter overrides.

        Args:
            config_path: Optional path to YAML configuration file.
            target_width: Spatial target width in pixels (e.g., 640).
            target_height: Spatial target height in pixels (e.g., 360).
            preserve_aspect_ratio: If True, centers content within letterbox padding.
            padding_color: Tuple of BGR color values for letterbox canvas (default: [0, 0, 0]).
            color_mode: Storage color mode convention ('BGR' for standard OpenCV disk persistence).
            image_format: Saved image format extension ('jpg', 'png', 'webp').
            jpeg_quality: JPEG compression quality factor in range [1, 100].
            calculate_blur: If True, computes Laplacian variance sharpness score.
            calculate_brightness: If True, computes grayscale mean intensity.
            calculate_contrast: If True, computes grayscale standard deviation.
            overwrite: If True, replaces existing preprocessed images on disk.
        """
        # Default parameter values
        self._input_dir = Path("datasets/processed/driving_sample/frames")
        self._output_dir = Path("datasets/processed/driving_sample/preprocessed_frames")
        self._target_width = 640
        self._target_height = 360
        self._preserve_aspect_ratio = True
        self._padding_color: Tuple[int, int, int] = (0, 0, 0)
        self._color_mode = "BGR"
        self._image_format = "jpg"
        self._jpeg_quality = 95
        self._calculate_blur = True
        self._calculate_brightness = True
        self._calculate_contrast = True
        self._overwrite = True

        # Load values from configuration file
        self._load_config(config_path)

        # Apply programmatic overrides
        if target_width is not None:
            self._target_width = int(target_width)
        if target_height is not None:
            self._target_height = int(target_height)
        if preserve_aspect_ratio is not None:
            self._preserve_aspect_ratio = bool(preserve_aspect_ratio)
        if padding_color is not None:
            self._padding_color = tuple(padding_color)  # type: ignore[assignment]
        if color_mode is not None:
            self._color_mode = str(color_mode).upper().strip()
        if image_format is not None:
            self._image_format = str(image_format).lower().lstrip(".")
        if jpeg_quality is not None:
            self._jpeg_quality = int(jpeg_quality)
        if calculate_blur is not None:
            self._calculate_blur = bool(calculate_blur)
        if calculate_brightness is not None:
            self._calculate_brightness = bool(calculate_brightness)
        if calculate_contrast is not None:
            self._calculate_contrast = bool(calculate_contrast)
        if overwrite is not None:
            self._overwrite = bool(overwrite)

        self._validate_parameters()
        self._summary: Optional[PreprocessingSummary] = None

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
                p_cfg = cfg.get("preprocessing", {})
                if "input_directory" in p_cfg:
                    self._input_dir = Path(p_cfg["input_directory"])
                if "output_directory" in p_cfg:
                    self._output_dir = Path(p_cfg["output_directory"])
                if "target_width" in p_cfg:
                    self._target_width = int(p_cfg["target_width"])
                if "target_height" in p_cfg:
                    self._target_height = int(p_cfg["target_height"])
                if "preserve_aspect_ratio" in p_cfg:
                    self._preserve_aspect_ratio = bool(p_cfg["preserve_aspect_ratio"])
                if "padding_color" in p_cfg:
                    self._padding_color = tuple(p_cfg["padding_color"])  # type: ignore[assignment]
                if "color_mode" in p_cfg:
                    self._color_mode = str(p_cfg["color_mode"]).upper().strip()
                if "image_format" in p_cfg:
                    self._image_format = str(p_cfg["image_format"]).lower().lstrip(".")
                if "jpeg_quality" in p_cfg:
                    self._jpeg_quality = int(p_cfg["jpeg_quality"])
                if "calculate_blur" in p_cfg:
                    self._calculate_blur = bool(p_cfg["calculate_blur"])
                if "calculate_brightness" in p_cfg:
                    self._calculate_brightness = bool(p_cfg["calculate_brightness"])
                if "calculate_contrast" in p_cfg:
                    self._calculate_contrast = bool(p_cfg["calculate_contrast"])
                if "overwrite" in p_cfg:
                    self._overwrite = bool(p_cfg["overwrite"])
            except Exception as e:
                logger.warning("Failed to parse config %s: %s. Using default settings.", config_path, e)

    def _validate_parameters(self) -> None:
        """Validate parameter boundaries and invariant constraints."""
        if self._target_width <= 0 or self._target_height <= 0:
            raise InvalidPreprocessingConfigError(
                f"Target dimensions must be strictly positive, got: {self._target_width}x{self._target_height}"
            )

        if not (1 <= self._jpeg_quality <= 100):
            raise InvalidPreprocessingConfigError(
                f"jpeg_quality must be in range [1, 100], got: {self._jpeg_quality}"
            )

        if self._image_format not in {"jpg", "jpeg", "png", "webp"}:
            raise InvalidPreprocessingConfigError(
                f"Unsupported image_format '{self._image_format}'. Supported: jpg, jpeg, png, webp"
            )

        if self._color_mode not in {"BGR", "RGB"}:
            raise InvalidPreprocessingConfigError(
                f"Unsupported color_mode '{self._color_mode}'. Supported: BGR, RGB"
            )

    @property
    def target_dimensions(self) -> Tuple[int, int]:
        """Configured target spatial dimensions (width, height)."""
        return self._target_width, self._target_height

    @property
    def preserve_aspect_ratio(self) -> bool:
        """Configured aspect ratio preservation policy."""
        return self._preserve_aspect_ratio

    def validate_frame(
        self, image_path: Union[str, Path]
    ) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
        """Perform comprehensive integrity checks on an image file.

        Validates:
        1. Filesystem existence and file status.
        2. Non-zero byte file size.
        3. Successful OpenCV image decoding.
        4. Non-empty numpy buffer.
        5. Valid positive spatial dimensions.
        6. Supported channel layout (grayscale, BGR, or BGRA).

        Args:
            image_path: Path to the target image file.

        Returns:
            Tuple of (is_valid, error_message, decoded_image_array).
        """
        path = Path(image_path)

        if not path.exists():
            return False, f"Image file not found at path: '{path}'", None

        if not path.is_file():
            return False, f"Path is a directory, not a regular image file: '{path}'", None

        try:
            file_size = path.stat().st_size
        except OSError as e:
            return False, f"Failed to inspect file size for '{path}': {e}", None

        if file_size == 0:
            return False, f"Image file is empty (zero bytes): '{path}'", None

        try:
            # Load with IMREAD_UNCHANGED to detect true channel configuration
            decoded_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            return False, f"OpenCV exception during decoding of '{path}': {e}", None

        if decoded_image is None or decoded_image.size == 0:
            return False, f"OpenCV failed to decode image data from '{path}' (corrupt or unsupported format)", None

        h, w = decoded_image.shape[:2]
        if h <= 0 or w <= 0:
            return False, f"Invalid decoded image dimensions ({w}x{h}) for '{path}'", None

        channels = 1 if len(decoded_image.shape) == 2 else decoded_image.shape[2]
        if channels not in (1, 3, 4):
            return False, f"Unsupported channel configuration ({channels} channels) for '{path}'", None

        return True, None, decoded_image

    def resize_frame(
        self,
        image: np.ndarray,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        preserve_aspect_ratio: Optional[bool] = None,
        padding_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Resize image to target dimensions with optional aspect-ratio letterboxing.

        Resizing Strategies:
        - Aspect Ratio Preserving (Letterbox): Computes scale factor s = min(tw/w, th/h).
          Scales content proportionally, then centers it within a canvas filled with
          padding_color. This guarantees vehicles, pedestrians, and lane geometry
          maintain their true physical aspect ratio without distortion.
        - Direct Stretch: Resizes directly to (target_width, target_height).

        Interpolation:
        - Uses cv2.INTER_AREA when downscaling (shrinking) to prevent aliasing.
        - Uses cv2.INTER_LINEAR when upscaling.

        Args:
            image: Input image array (BGR or Grayscale).
            target_width: Optional width override.
            target_height: Optional height override.
            preserve_aspect_ratio: Optional aspect ratio preservation override.
            padding_color: Optional padding color override.

        Returns:
            np.ndarray: Resized image array of exact shape (target_height, target_width, channels).

        Raises:
            InvalidImageDimensionsError: If image array is empty or dimensions are invalid.
            InvalidPreprocessingConfigError: If target dimensions are non-positive.
        """
        tw = target_width if target_width is not None else self._target_width
        th = target_height if target_height is not None else self._target_height
        preserve = preserve_aspect_ratio if preserve_aspect_ratio is not None else self._preserve_aspect_ratio
        pad_col = padding_color if padding_color is not None else self._padding_color

        if image is None or image.size == 0:
            raise InvalidImageDimensionsError("Cannot resize empty or uninitialized image buffer.")

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise InvalidImageDimensionsError(f"Invalid source image dimensions: {w}x{h}")

        if tw <= 0 or th <= 0:
            raise InvalidPreprocessingConfigError(f"Target dimensions must be positive, got: {tw}x{th}")

        if not preserve:
            interp = cv2.INTER_AREA if (tw < w and th < h) else cv2.INTER_LINEAR
            return cv2.resize(image, (tw, th), interpolation=interp)

        # Aspect-ratio preserving letterboxing
        scale = min(tw / w, th / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_content = cv2.resize(image, (new_w, new_h), interpolation=interp)

        # Create canvas filled with padding_color
        if len(image.shape) == 3:
            canvas = np.full((th, tw, image.shape[2]), pad_col, dtype=np.uint8)
        else:
            gray_val = pad_col[0] if isinstance(pad_col, (tuple, list)) else int(pad_col)
            canvas = np.full((th, tw), gray_val, dtype=np.uint8)

        # Center placement coordinates
        pad_top = (th - new_h) // 2
        pad_left = (tw - new_w) // 2

        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_content
        return canvas

    def convert_color(
        self, image: np.ndarray, target_color_space: str = "RGB"
    ) -> np.ndarray:
        """Convert image array to specified color space.

        Color Space Architecture:
        - OpenCV Disk / imread / imwrite default: BGR.
        - Deep Learning / YOLOv8 / PyTorch inference default: RGB.
        - Visual Quality / Feature extraction default: GRAY.

        Args:
            image: Input image array.
            target_color_space: Target color space ('RGB', 'BGR', 'GRAY').

        Returns:
            np.ndarray: Converted image array.

        Raises:
            InvalidColorChannelError: If input is empty or target color space is unsupported.
        """
        if image is None or image.size == 0:
            raise InvalidColorChannelError("Cannot convert color of empty or uninitialized image buffer.")

        target = target_color_space.upper().strip()
        is_gray = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)

        if target == "RGB":
            if is_gray:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif target == "BGR":
            if is_gray:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                return image.copy()
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif target in ("GRAY", "GRAYSCALE"):
            if is_gray:
                return image.copy() if len(image.shape) == 2 else image[:, :, 0].copy()
            elif len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        raise InvalidColorChannelError(
            f"Unsupported target color space '{target}'. Supported: RGB, BGR, GRAY"
        )

    @staticmethod
    def calculate_blur_score(gray_image: np.ndarray) -> float:
        """Calculate image sharpness/blur index using Laplacian variance.

        Formula:
            Var(Laplacian(I)) = Var(d^2 I / dx^2 + d^2 I / dy^2)

        Interpretation:
            Higher values (> 100) denote sharp, distinct edge gradients.
            Lower values (< 50) denote optical defocus, heavy motion blur, or fog.

        Note:
            This metric is a visual heuristic indicator, NOT a ground-truth detection
            reliability label. True reliability is evaluated via downstream IoU matching.

        Args:
            gray_image: Grayscale image array.

        Returns:
            float: Laplacian variance score.
        """
        if gray_image is None or gray_image.size == 0:
            return 0.0
        lap = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(lap.var())

    @staticmethod
    def calculate_brightness_score(gray_image: np.ndarray) -> float:
        """Calculate mean grayscale intensity across image pixels.

        Formula:
            mu = (1 / (W * H)) * sum(I(x, y))

        Interpretation:
            Range [0.0, 255.0].
            0.0 = completely black (night without headlights/streetlights).
            255.0 = completely saturated white (severe direct solar glare).
            Typical daytime driving: 80.0 - 160.0.

        Args:
            gray_image: Grayscale image array.

        Returns:
            float: Mean intensity.
        """
        if gray_image is None or gray_image.size == 0:
            return 0.0
        return float(np.mean(gray_image))

    @staticmethod
    def calculate_contrast_score(gray_image: np.ndarray) -> float:
        """Calculate standard deviation of grayscale intensities (RMS Contrast).

        Formula:
            sigma = sqrt((1 / (W * H)) * sum((I(x, y) - mu)^2))

        Interpretation:
            Low values (< 20) indicate washed out, low dynamic range (heavy fog, rain).
            High values (> 60) indicate strong separation between highlights and shadows.

        Args:
            gray_image: Grayscale image array.

        Returns:
            float: Standard deviation of pixel intensities.
        """
        if gray_image is None or gray_image.size == 0:
            return 0.0
        return float(np.std(gray_image))

    def preprocess_frame(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> PreprocessedFrameRecord:
        """Execute complete validation, quality assessment, resizing, and saving for one frame.

        Args:
            image_path: Filesystem path to the input raw extracted frame.
            output_path: Optional destination path for preprocessed image.

        Returns:
            PreprocessedFrameRecord: Immutable record of preprocessing telemetry and status.

        Raises:
            ImageSaveError: If output disk write fails.
        """
        src_path = Path(image_path).resolve()
        out_path = Path(output_path).resolve() if output_path else self._output_dir / src_path.name

        # 1. Image Validation
        is_valid, err_msg, decoded = self.validate_frame(src_path)
        if not is_valid or decoded is None:
            logger.warning("Frame validation failed for '%s': %s", src_path.name, err_msg)
            return PreprocessedFrameRecord(
                input_filename=src_path.name,
                input_path=src_path,
                output_filename=out_path.name,
                output_path=out_path,
                original_width=0,
                original_height=0,
                processed_width=0,
                processed_height=0,
                blur_score=None,
                brightness_score=None,
                contrast_score=None,
                is_valid=False,
                error_message=err_msg,
            )

        orig_h, orig_w = decoded.shape[:2]

        # Handle 4-channel BGRA images if encountered
        if len(decoded.shape) == 3 and decoded.shape[2] == 4:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)

        # 2. Compute Grayscale Image for Quality Indicators
        gray_image = self.convert_color(decoded, "GRAY")

        # 3. Compute Quality Indicators
        blur_score = self.calculate_blur_score(gray_image) if self._calculate_blur else None
        brightness_score = self.calculate_brightness_score(gray_image) if self._calculate_brightness else None
        contrast_score = self.calculate_contrast_score(gray_image) if self._calculate_contrast else None

        # 4. Resize Frame with Aspect Ratio Preservation
        processed_frame = self.resize_frame(decoded)
        proc_h, proc_w = processed_frame.shape[:2]

        # 5. Persist Preprocessed Image to Disk
        if output_path is not None or self._output_dir is not None:
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                msg = f"Failed to create output directory '{out_path.parent}': {e}"
                logger.error(msg)
                raise ImageSaveError(msg) from e

            if not out_path.exists() or self._overwrite:
                encode_params: List[int] = []
                if self._image_format in ("jpg", "jpeg"):
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                elif self._image_format == "png":
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                elif self._image_format == "webp":
                    encode_params = [cv2.IMWRITE_WEBP_QUALITY, self._jpeg_quality]

                try:
                    success = cv2.imwrite(str(out_path), processed_frame, encode_params)
                except Exception as e:
                    msg = f"OpenCV exception during imwrite to '{out_path}': {e}"
                    logger.error(msg)
                    raise ImageSaveError(msg) from e

                if not success or not out_path.is_file() or out_path.stat().st_size == 0:
                    msg = f"cv2.imwrite failed to persist valid image file at '{out_path}'"
                    logger.error(msg)
                    raise ImageSaveError(msg)

        # Explicitly release memory buffers
        del decoded
        del gray_image
        del processed_frame

        return PreprocessedFrameRecord(
            input_filename=src_path.name,
            input_path=src_path,
            output_filename=out_path.name,
            output_path=out_path,
            original_width=orig_w,
            original_height=orig_h,
            processed_width=proc_w,
            processed_height=proc_h,
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            is_valid=True,
            error_message=None,
        )

    def process_directory(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PreprocessingSummary:
        """Preprocess an entire directory of extracted frames sequentially.

        Frames are processed one by one to strictly bound CPU and memory usage.
        Raw input frames remain entirely untouched and unmodified.

        Args:
            input_dir: Directory containing input extracted frames (defaults to configured path).
            output_dir: Destination directory for preprocessed frames.
            progress_callback: Optional telemetry callable(current_index, total_count).

        Returns:
            PreprocessingSummary: Comprehensive execution audit record.
        """
        start_time = time.perf_counter()
        in_path = Path(input_dir).resolve() if input_dir else self._input_dir.resolve()
        out_path = Path(output_dir).resolve() if output_dir else self._output_dir.resolve()

        if not in_path.exists() or not in_path.is_dir():
            raise PreprocessingError(f"Input directory does not exist or is not a directory: '{in_path}'")

        out_path.mkdir(parents=True, exist_ok=True)

        # Collect and sort image files alphabetically to preserve temporal sequence
        all_files = sorted([
            f for f in in_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_INPUT_EXTENSIONS
        ])
        total_files = len(all_files)

        logger.info(
            "Found %d candidate image files in '%s' for preprocessing -> '%s'",
            total_files,
            in_path.name,
            out_path.name,
        )

        records: List[PreprocessedFrameRecord] = []
        valid_blur_scores: List[float] = []
        valid_brightness_scores: List[float] = []
        valid_contrast_scores: List[float] = []

        for idx, file_path in enumerate(all_files, start=1):
            dest_file = out_path / file_path.name
            record = self.preprocess_frame(file_path, output_path=dest_file)
            records.append(record)

            if record.is_valid:
                if record.blur_score is not None:
                    valid_blur_scores.append(record.blur_score)
                if record.brightness_score is not None:
                    valid_brightness_scores.append(record.brightness_score)
                if record.contrast_score is not None:
                    valid_contrast_scores.append(record.contrast_score)

            if progress_callback:
                progress_callback(idx, total_files)

        elapsed = time.perf_counter() - start_time
        successful = sum(1 for r in records if r.is_valid)
        failed = total_files - successful

        mean_blur = float(np.mean(valid_blur_scores)) if valid_blur_scores else None
        mean_brightness = float(np.mean(valid_brightness_scores)) if valid_brightness_scores else None
        mean_contrast = float(np.mean(valid_contrast_scores)) if valid_contrast_scores else None

        self._summary = PreprocessingSummary(
            input_directory=in_path,
            output_directory=out_path,
            total_frames=total_files,
            successful_frames=successful,
            failed_frames=failed,
            target_dimensions=(self._target_width, self._target_height),
            mean_blur_score=mean_blur,
            mean_brightness_score=mean_brightness,
            mean_contrast_score=mean_contrast,
            records=tuple(records),
            elapsed_time_seconds=elapsed,
        )

        logger.info(
            "Preprocessing finished: %d/%d successful in %.2fs (%.1f frames/s). Mean blur=%.1f, brightness=%.1f",
            successful,
            total_files,
            elapsed,
            (total_files / elapsed) if elapsed > 0 else 0.0,
            mean_blur or 0.0,
            mean_brightness or 0.0,
        )

        return self._summary

    def get_processing_summary(self) -> Optional[PreprocessingSummary]:
        """Retrieve audit summary of most recent batch preprocessing execution."""
        return self._summary

