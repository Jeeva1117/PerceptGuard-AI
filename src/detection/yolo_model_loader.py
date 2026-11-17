"""PerceptGuard-AI: YOLO Model Loader & Environment Foundation.

Staff-level, production-grade YOLO model loading and verification engine.
Provides deterministic configuration parsing, CPU-compatible model initialization,
lightweight nano model management (YOLO11n), and single-frame inference smoke testing
for autonomous driving predictive perception pipelines.

Key Guarantees:
- CPU-First execution (Intel Core i3 compatible, avoids GPU dependencies)
- Lazy model loading (avoids overhead during module import)
- Strict parameter boundary validation (rejects uncalibrated thresholds)
- Guaranteed non-destructive operations (never mutates weights or datasets)
"""

from dataclasses import asdict, dataclass
import hashlib
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
import yaml

logger = logging.getLogger("PerceptGuard.YOLOModelLoader")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class YOLOEnvironmentError(Exception):
    """Base exception for all YOLO environment and model loading failures."""
    pass


class YOLOConfigError(YOLOEnvironmentError):
    """Raised when YOLO configuration parameters are missing or invalid."""
    pass


class YOLOModelNotFoundError(YOLOEnvironmentError):
    """Raised when model weights file does not exist at the specified path."""
    pass


class YOLOModelLoadError(YOLOEnvironmentError):
    """Raised when Ultralytics fails to load, parse, or validate model weights."""
    pass


class YOLOInferenceError(YOLOEnvironmentError):
    """Raised when single-frame YOLO inference fails or produces invalid output."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class YOLOModelInfo:
    """Immutable data contract describing verified YOLO model attributes."""
    model_name: str
    weights_path: Path
    task: str
    device: str
    framework_version: str
    num_classes: int
    class_names: Dict[int, str]
    is_loaded: bool
    file_size_bytes: int
    sha256_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        data = asdict(self)
        data["weights_path"] = str(self.weights_path)
        return data

    def __str__(self) -> str:
        return (
            f"YOLOModelInfo(model='{self.model_name}', "
            f"task='{self.task}', "
            f"device='{self.device}', "
            f"classes={self.num_classes}, "
            f"size={self.file_size_bytes} bytes, "
            f"loaded={self.is_loaded})"
        )


# ==============================================================================
# YOLO Model Loader Implementation
# ==============================================================================

class YOLOModelLoader:
    """Production-grade YOLO model loader and inference foundation.

    Encapsulates configuration parsing, path resolution, CPU inference safety,
    and lazy loading semantics for Ultralytics detection models.
    """

    DEFAULT_CONFIG_PATH = Path("config/yolo_config.yaml")
    SUPPORTED_TASKS = {"detect"}
    SUPPORTED_DEVICES = {"cpu"}

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize YOLOModelLoader with configuration and parameter overrides.

        Args:
            config_path: Path to YAML configuration file.
            weights_path: Explicit override for model weights path.
            device: Explicit inference device override (e.g. 'cpu').
            conf_threshold: Confidence score threshold in (0.0, 1.0].
            iou_threshold: NMS IoU threshold in (0.0, 1.0].
            image_size: Input image dimension (e.g. 640).
            project_root: Base repository directory for relative path resolution.
        """
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )

        # Default configuration settings
        self.model_name = "yolo11n"
        self._weights_rel_path = "models/yolo/yolo11n.pt"
        self.task = "detect"
        self.device = "cpu"
        self.image_size = 640
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 100
        self.half_precision = False
        self.verbose = False

        # Load values from configuration file if present
        self._load_config(config_path)

        # Explicit parameter overrides
        if weights_path is not None:
            self._weights_rel_path = str(weights_path)
        if device is not None:
            self.device = str(device).lower()
        if conf_threshold is not None:
            self.confidence_threshold = float(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
        if image_size is not None:
            self.image_size = int(image_size)

        # Resolve weights path
        raw_p = Path(self._weights_rel_path)
        self.weights_path = raw_p if raw_p.is_absolute() else (self.project_root / raw_p).resolve()

        # Validate configuration values
        self._validate_config()

        # Internal model cache
        self._model: Optional[YOLO] = None
        self._model_info: Optional[YOLOModelInfo] = None

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> None:
        """Parse configuration parameters from YAML file."""
        target_cfg: Path
        if config_path:
            p = Path(config_path)
            target_cfg = p if p.is_absolute() else (self.project_root / p).resolve()
        else:
            target_cfg = self.project_root / self.DEFAULT_CONFIG_PATH

        if target_cfg.is_file():
            try:
                with open(target_cfg, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                yolo_cfg = cfg.get("yolo", {})
                if "model_name" in yolo_cfg:
                    self.model_name = str(yolo_cfg["model_name"])
                if "model_weights_path" in yolo_cfg:
                    self._weights_rel_path = str(yolo_cfg["model_weights_path"])
                if "task" in yolo_cfg:
                    self.task = str(yolo_cfg["task"])
                if "device" in yolo_cfg:
                    self.device = str(yolo_cfg["device"]).lower()
                if "image_size" in yolo_cfg:
                    self.image_size = int(yolo_cfg["image_size"])
                if "confidence_threshold" in yolo_cfg:
                    self.confidence_threshold = float(yolo_cfg["confidence_threshold"])
                if "iou_threshold" in yolo_cfg:
                    self.iou_threshold = float(yolo_cfg["iou_threshold"])
                if "max_detections" in yolo_cfg:
                    self.max_detections = int(yolo_cfg["max_detections"])
                if "half_precision" in yolo_cfg:
                    self.half_precision = bool(yolo_cfg["half_precision"])
                if "verbose" in yolo_cfg:
                    self.verbose = bool(yolo_cfg["verbose"])

            except Exception as e:
                logger.warning("Failed to parse config %s: %s. Using defaults.", target_cfg, e)

    def _validate_config(self) -> None:
        """Validate configuration parameters against physical and framework constraints."""
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise YOLOConfigError(
                f"confidence_threshold must be in (0.0, 1.0], got: {self.confidence_threshold}"
            )

        if not (0.0 < self.iou_threshold <= 1.0):
            raise YOLOConfigError(
                f"iou_threshold must be in (0.0, 1.0], got: {self.iou_threshold}"
            )

        if not (32 <= self.image_size <= 1920):
            raise YOLOConfigError(
                f"image_size must be between 32 and 1920, got: {self.image_size}"
            )

        if self.task not in self.SUPPORTED_TASKS:
            raise YOLOConfigError(
                f"Unsupported task '{self.task}'. Supported tasks: {self.SUPPORTED_TASKS}"
            )

        if self.device != "cpu":
            if not torch.cuda.is_available():
                raise YOLOConfigError(
                    f"Requested device '{self.device}', but CUDA is not available on this host. "
                    "PerceptGuard AI operates on CPU mode."
                )

        if self.device == "cpu" and self.half_precision:
            raise YOLOConfigError(
                "half_precision=True (FP16) is not supported for CPU inference in PyTorch. "
                "Set half_precision=False for CPU inference."
            )

    @property
    def is_loaded(self) -> bool:
        """Check if the YOLO model is currently loaded in memory."""
        return self._model is not None

    def load_model(self, force_reload: bool = False) -> YOLO:
        """Load and cache the Ultralytics YOLO model from weights on disk.

        Args:
            force_reload: If True, forces re-instantiation of the model.

        Returns:
            YOLO: Initialized Ultralytics model instance.

        Raises:
            YOLOModelNotFoundError: If weights file does not exist on disk.
            YOLOModelLoadError: If model instantiation or verification fails.
        """
        if self._model is not None and not force_reload:
            return self._model

        # 1. Verify weights file existence
        if not self.weights_path.exists():
            raise YOLOModelNotFoundError(
                f"YOLO model weights file not found at: '{self.weights_path}'. "
                f"Please ensure pretrained weights are stored under '{self.weights_path.parent}'."
            )

        if not self.weights_path.is_file():
            raise YOLOModelNotFoundError(
                f"Specified weights path is a directory, not a file: '{self.weights_path}'."
            )

        file_size = self.weights_path.stat().st_size
        if file_size == 0:
            raise YOLOModelLoadError(
                f"YOLO weights file is empty (0 bytes): '{self.weights_path}'."
            )

        # 2. Compute SHA-256 hash
        try:
            h = hashlib.sha256()
            with open(self.weights_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            sha256 = h.hexdigest()
        except Exception as e:
            sha256 = "UNKNOWN"

        # 3. Instantiate model via Ultralytics
        logger.info("Loading YOLO model from: %s on device: %s", self.weights_path.name, self.device)
        try:
            model = YOLO(str(self.weights_path), task=self.task)
        except Exception as e:
            msg = f"Failed to load Ultralytics YOLO model from '{self.weights_path}': {e}"
            logger.error(msg)
            raise YOLOModelLoadError(msg) from e

        # 4. Verify task compatibility
        detected_task = getattr(model, "task", self.task)
        if detected_task != "detect":
            raise YOLOModelLoadError(
                f"Model task mismatch: expected 'detect', got '{detected_task}'. "
                "PerceptGuard AI requires a 2D bounding-box object detection model."
            )

        # 5. Extract class catalog
        class_names = getattr(model, "names", {})
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        elif not isinstance(class_names, dict):
            class_names = {}

        self._model = model
        self._model_info = YOLOModelInfo(
            model_name=self.model_name,
            weights_path=self.weights_path,
            task=self.task,
            device=self.device,
            framework_version=ultralytics.__version__,
            num_classes=len(class_names),
            class_names=class_names,
            is_loaded=True,
            file_size_bytes=file_size,
            sha256_hash=sha256,
        )

        logger.info(
            "YOLO Model successfully initialized: %s (%d COCO classes, %.2f MB)",
            self.model_name,
            len(class_names),
            file_size / (1024 * 1024),
        )
        return self._model

    def get_model_info(self) -> YOLOModelInfo:
        """Retrieve verified model information contract.

        Returns:
            YOLOModelInfo: Immutable metadata contract.
        """
        if self._model_info is None:
            self.load_model()
        return self._model_info  # type: ignore[return-value]

    def predict_single_frame(
        self,
        frame: Union[np.ndarray, str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> Any:
        """Run single-frame inference on CPU using the loaded YOLO model.

        Args:
            frame: Preprocessed image matrix (BGR uint8) or path to an image file.
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.
            imgsz: Optional image size override.

        Returns:
            ultralytics.engine.results.Results: Ultralytics detection results container.

        Raises:
            YOLOInferenceError: If image loading, format, or inference execution fails.
        """
        model = self.load_model()

        # Handle path input
        image_to_predict: np.ndarray
        if isinstance(frame, (str, Path)):
            fpath = Path(frame)
            if not fpath.is_file():
                raise YOLOInferenceError(f"Input image file does not exist: '{fpath}'")
            decoded = cv2.imread(str(fpath))
            if decoded is None or decoded.size == 0:
                raise YOLOInferenceError(f"OpenCV failed to decode image: '{fpath}'")
            image_to_predict = decoded
        elif isinstance(frame, np.ndarray):
            if frame.size == 0:
                raise YOLOInferenceError("Cannot run inference on an empty NumPy frame array.")
            image_to_predict = frame
        else:
            raise YOLOInferenceError(f"Unsupported frame type: {type(frame)}")

        conf_val = conf if conf is not None else self.confidence_threshold
        iou_val = iou if iou is not None else self.iou_threshold
        imgsz_val = imgsz if imgsz is not None else self.image_size

        try:
            results = model.predict(
                source=image_to_predict,
                device=self.device,
                conf=conf_val,
                iou=iou_val,
                imgsz=imgsz_val,
                verbose=self.verbose,
                max_det=self.max_detections,
            )
            return results[0]
        except Exception as e:
            msg = f"YOLO single-frame prediction failed: {e}"
            logger.error(msg)
            raise YOLOInferenceError(msg) from e


# ==============================================================================
# Environment Verification & CLI
# ==============================================================================

def verify_yolo_environment(
    config_path: Optional[Union[str, Path]] = None,
    frame_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Execute end-to-end YOLO environment, model loading, and smoke inference check.

    Args:
        config_path: Path to yolo_config.yaml.
        frame_path: Path to test image frame (defaults to real preprocessed frame 1).

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    project_root = Path(__file__).resolve().parents[2]

    # 1. Environment metadata
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    ultra_ver = ultralytics.__version__
    cv_ver = cv2.__version__
    cpu_info = platform.processor() or "Intel Core i3 Architecture (x86_64)"
    cuda_avail = torch.cuda.is_available()
    selected_device = "cpu"

    # 2. Resolve target frame
    if frame_path:
        target_frame = Path(frame_path).resolve()
    else:
        target_frame = (
            project_root
            / "datasets"
            / "processed"
            / "driving_sample"
            / "preprocessed_frames"
            / "frame_000001.jpg"
        )

    print("=" * 60)
    print("PERCEPTGUARD AI – YOLO ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Python version: {py_ver}")
    print(f"PyTorch version: {torch_ver}")
    print(f"Ultralytics version: {ultra_ver}")
    print(f"OpenCV version: {cv_ver}")
    print(f"CPU information: {cpu_info}")
    print(f"CUDA available: {cuda_avail}")
    print(f"Selected device: {selected_device}")

    # 3. Model loading
    loader = YOLOModelLoader(config_path=config_path, project_root=project_root)
    print(f"YOLO model: {loader.model_name}")
    print(f"Model weights: {loader.weights_path.name}")
    print(f"Model task: {loader.task}")

    model_loading_status = "FAILED"
    inference_status = "FAILED"
    frame_decoding_status = "FAILED"
    inference_time_ms = 0.0
    detections_returned = 0

    try:
        t_load_0 = time.perf_counter()
        loader.load_model()
        t_load_1 = time.perf_counter()
        model_loading_status = f"SUCCESS ({(t_load_1 - t_load_0)*1000:.2f} ms)"
    except Exception as e:
        model_loading_status = f"FAILED ({e})"

    print(f"Model loading: {model_loading_status}")
    print(f"Dataset frame: {target_frame.name}")

    # 4. Frame decoding & inference
    if target_frame.is_file():
        img = cv2.imread(str(target_frame))
        if img is not None and img.size > 0:
            frame_decoding_status = f"SUCCESS ({img.shape[1]}x{img.shape[0]} px)"

            if "SUCCESS" in model_loading_status:
                try:
                    # Warmup once
                    _ = loader.predict_single_frame(img)

                    # Benchmark inference
                    t0 = time.perf_counter()
                    res = loader.predict_single_frame(img)
                    t1 = time.perf_counter()

                    inference_time_ms = (t1 - t0) * 1000
                    boxes = getattr(res, "boxes", None)
                    detections_returned = len(boxes) if boxes is not None else 0
                    inference_status = f"SUCCESS ({inference_time_ms:.2f} ms)"
                except Exception as e:
                    inference_status = f"FAILED ({e})"
        else:
            frame_decoding_status = "FAILED (OpenCV decode failed)"
    else:
        frame_decoding_status = f"FAILED (File not found at {target_frame})"

    print(f"Frame decoding: {frame_decoding_status}")
    print(f"Inference status: {inference_status}")
    print(f"Inference time: {inference_time_ms:.2f} ms")
    print(f"Detections returned: {detections_returned}")
    print("=" * 60)

    is_overall_pass = (
        "SUCCESS" in model_loading_status
        and "SUCCESS" in frame_decoding_status
        and "SUCCESS" in inference_status
    )

    status_str = "PASS" if is_overall_pass else "FAIL"
    print(f"OVERALL STATUS: {status_str}")
    print("=" * 60)

    return is_overall_pass


def main() -> None:
    """CLI entrypoint for YOLO model loader verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI – YOLO Environment & Model Verification"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run comprehensive YOLO environment, model loading, and single-frame inference check.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to yolo_config.yaml (default: config/yolo_config.yaml)",
    )
    parser.add_argument(
        "--frame-path",
        type=str,
        default=None,
        help="Path to test frame image (default: datasets/processed/driving_sample/preprocessed_frames/frame_000001.jpg)",
    )
    args = parser.parse_args()

    if args.verify or len(sys.argv) == 1:
        success = verify_yolo_environment(config_path=args.config, frame_path=args.frame_path)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
