"""PerceptGuard-AI: Advanced YOLO Model Configuration Architecture.

Senior Staff-level, production-grade configuration system for YOLO object detection
in autonomous-driving predictive perception pipelines.

Key Architectural Guarantees:
- Strongly typed, immutable-friendly dataclass hierarchy (Model, Runtime, Inference, Performance, Validation).
- Safe YAML parsing with strict schema and boundary verification.
- Hardware-aware parameter validation (CPU-first, FP32 enforcement, PyTorch CPU constraints).
- Full compatibility with existing Task 1 YOLOModelLoader and legacy configuration formats.
- Complete COCO class filtering validation (bounds, types, duplicate rejection).
- Stride-conforming image size checks (multiples of 32).
- Documented perception presets (balanced, high_precision, high_recall, cpu_lightweight).
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import yaml

logger = logging.getLogger("PerceptGuard.YOLOConfig")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class YOLOEnvironmentError(Exception):
    """Base exception for all YOLO environment and model loading failures."""
    pass


class YOLOConfigError(YOLOEnvironmentError):
    """Raised when YOLO configuration parameters are missing, malformed, or out of bounds."""
    pass


class YOLOModelNotFoundError(YOLOEnvironmentError):
    """Raised when model weights file does not exist at the specified path."""
    pass


# ==============================================================================
# Configuration Sections (Dataclasses)
# ==============================================================================

@dataclass
class ModelConfig:
    """Configuration governing model architecture, weights location, and task."""
    name: str = "yolo11n"
    architecture: str = "yolo11"
    weights_path: Union[str, Path] = "models/yolo/yolo11n.pt"
    task: str = "detect"
    pretrained: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "weights_path": str(self.weights_path),
            "task": self.task,
            "pretrained": self.pretrained,
        }


@dataclass
class RuntimeConfig:
    """Configuration governing compute device, numerical precision, and reproducibility."""
    device: str = "cpu"
    half_precision: bool = False
    deterministic: bool = True
    seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "half_precision": self.half_precision,
            "deterministic": self.deterministic,
            "seed": self.seed,
        }


@dataclass
class InferenceConfig:
    """Configuration governing YOLO inference hyperparameters and class filtering."""
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    agnostic_nms: bool = False
    classes: Optional[List[int]] = None
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "agnostic_nms": self.agnostic_nms,
            "classes": list(self.classes) if self.classes is not None else None,
            "verbose": self.verbose,
        }


@dataclass
class PerformanceConfig:
    """Configuration governing latency timeouts, batching, and CPU resource allocation."""
    warmup_iterations: int = 1
    max_allowed_latency_ms: float = 500.0
    batch_size: int = 1
    cpu_threads: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "max_allowed_latency_ms": self.max_allowed_latency_ms,
            "batch_size": self.batch_size,
            "cpu_threads": self.cpu_threads,
        }


@dataclass
class ValidationConfig:
    """Configuration governing validation constraints, ranges, and supported backends."""
    strict_mode: bool = True
    validate_weights: bool = False
    validate_parameters: bool = True
    min_confidence: float = 0.01
    max_confidence: float = 1.0
    min_iou: float = 0.01
    max_iou: float = 1.0
    min_image_size: int = 32
    max_image_size: int = 1920
    supported_tasks: List[str] = field(default_factory=lambda: ["detect"])
    supported_devices: List[str] = field(default_factory=lambda: ["cpu"])
    supported_models: List[str] = field(
        default_factory=lambda: ["yolo11n", "yolo11s", "yolo11m", "yolov8n", "yolov8s"]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strict_mode": self.strict_mode,
            "validate_weights": self.validate_weights,
            "validate_parameters": self.validate_parameters,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "min_iou": self.min_iou,
            "max_iou": self.max_iou,
            "min_image_size": self.min_image_size,
            "max_image_size": self.max_image_size,
            "supported_tasks": list(self.supported_tasks),
            "supported_devices": list(self.supported_devices),
            "supported_models": list(self.supported_models),
        }


# ==============================================================================
# Presets Catalog
# ==============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "image_size": 640,
        "max_detections": 100,
    },
    "high_precision": {
        "confidence_threshold": 0.50,
        "iou_threshold": 0.50,
        "image_size": 640,
        "max_detections": 50,
    },
    "high_recall": {
        "confidence_threshold": 0.10,
        "iou_threshold": 0.40,
        "image_size": 640,
        "max_detections": 200,
    },
    "cpu_lightweight": {
        "confidence_threshold": 0.30,
        "iou_threshold": 0.50,
        "image_size": 320,
        "max_detections": 50,
    },
}


# ==============================================================================
# Master YOLOConfig
# ==============================================================================

@dataclass
class YOLOConfig:
    """Master configuration container for PerceptGuard AI YOLO detection pipelines."""
    version: str = "1.0.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # --------------------------------------------------------------------------
    # Backward-Compatibility Convenience Accessors
    # --------------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def weights_path(self) -> Union[str, Path]:
        return self.model.weights_path

    @property
    def task(self) -> str:
        return self.model.task

    @property
    def device(self) -> str:
        return self.runtime.device

    @property
    def half_precision(self) -> bool:
        return self.runtime.half_precision

    @property
    def image_size(self) -> int:
        return self.inference.image_size

    @property
    def confidence_threshold(self) -> float:
        return self.inference.confidence_threshold

    @property
    def iou_threshold(self) -> float:
        return self.inference.iou_threshold

    @property
    def max_detections(self) -> int:
        return self.inference.max_detections

    @property
    def agnostic_nms(self) -> bool:
        return self.inference.agnostic_nms

    @property
    def classes(self) -> Optional[List[int]]:
        return self.inference.classes

    @property
    def verbose(self) -> bool:
        return self.inference.verbose

    # --------------------------------------------------------------------------
    # Serialization & Path Resolution
    # --------------------------------------------------------------------------
    def resolve_weights_path(self, project_root: Optional[Union[str, Path]] = None) -> Path:
        """Resolve model weights path relative to project root."""
        root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )
        p = Path(self.model.weights_path)
        return p if p.is_absolute() else (root / p).resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to a nested dictionary representation."""
        return {
            "version": self.version,
            "model": self.model.to_dict(),
            "runtime": self.runtime.to_dict(),
            "inference": self.inference.to_dict(),
            "performance": self.performance.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_yaml(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to a clean YAML formatted string or file."""
        data = self.to_dict()
        data["presets"] = PRESETS
        yaml_str = yaml.dump(data, sort_keys=False, indent=2)
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml_str, encoding="utf-8")
        return yaml_str

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export serialized configuration metadata for downstream telemetry and reproducibility."""
        data = self.to_dict()
        data["metadata"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "python_version": sys.version.split()[0],
        }
        return data

    # --------------------------------------------------------------------------
    # Preset Application
    # --------------------------------------------------------------------------
    def with_preset(self, preset_name: str) -> "YOLOConfig":
        """Return a new deep-copied YOLOConfig with specified preset parameters applied."""
        preset_key = preset_name.lower().strip()
        if preset_key not in PRESETS:
            raise YOLOConfigError(
                f"Unknown preset '{preset_name}'. Supported presets: {list(PRESETS.keys())}"
            )

        new_config = deepcopy(self)
        preset_vals = PRESETS[preset_key]
        if "confidence_threshold" in preset_vals:
            new_config.inference.confidence_threshold = preset_vals["confidence_threshold"]
        if "iou_threshold" in preset_vals:
            new_config.inference.iou_threshold = preset_vals["iou_threshold"]
        if "image_size" in preset_vals:
            new_config.inference.image_size = preset_vals["image_size"]
        if "max_detections" in preset_vals:
            new_config.inference.max_detections = preset_vals["max_detections"]

        new_config.validate()
        return new_config

    # --------------------------------------------------------------------------
    # Validation Logic
    # --------------------------------------------------------------------------
    def validate(
        self,
        project_root: Optional[Union[str, Path]] = None,
        validate_weights: Optional[bool] = None,
    ) -> None:
        """Execute exhaustive validation on all parameters, ranges, and resource constraints."""
        root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )

        val_cfg = self.validation
        if not val_cfg.validate_parameters:
            return

        # 1. Model Section Validation
        if not self.model.name or not isinstance(self.model.name, str):
            raise YOLOConfigError(f"model.name must be a non-empty string, got: {self.model.name!r}")

        if val_cfg.supported_models and self.model.name not in val_cfg.supported_models:
            logger.warning(
                "Model name '%s' not in supported_models list (%s). Proceeding under strict check.",
                self.model.name,
                val_cfg.supported_models,
            )

        if not self.model.task or not isinstance(self.model.task, str):
            raise YOLOConfigError(f"model.task must be a string, got: {self.model.task!r}")

        if self.model.task not in val_cfg.supported_tasks:
            raise YOLOConfigError(
                f"Unsupported task '{self.model.task}'. Supported tasks: {val_cfg.supported_tasks}"
            )

        # 2. Runtime Section Validation
        if not self.runtime.device or not isinstance(self.runtime.device, str):
            raise YOLOConfigError(f"runtime.device must be a string, got: {self.runtime.device!r}")

        device_lower = self.runtime.device.lower()
        if device_lower not in val_cfg.supported_devices and device_lower != "cpu":
            if device_lower.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise YOLOConfigError(
                        f"Requested device '{self.runtime.device}', but CUDA is unavailable on this host. "
                        "PerceptGuard AI operates on CPU mode."
                    )
            else:
                raise YOLOConfigError(
                    f"Unsupported device '{self.runtime.device}'. Supported devices: {val_cfg.supported_devices}"
                )

        if isinstance(self.runtime.half_precision, bool):
            if device_lower == "cpu" and self.runtime.half_precision:
                raise YOLOConfigError(
                    "half_precision=True (FP16) is not supported for CPU inference in PyTorch. "
                    "Set half_precision=False for CPU inference."
                )
        else:
            raise YOLOConfigError(
                f"runtime.half_precision must be a boolean, got: {type(self.runtime.half_precision).__name__}"
            )

        if self.runtime.seed is not None:
            if isinstance(self.runtime.seed, bool) or not isinstance(self.runtime.seed, int):
                raise YOLOConfigError(f"runtime.seed must be an integer, got: {type(self.runtime.seed).__name__}")
            if not (0 <= self.runtime.seed <= 2**32 - 1):
                raise YOLOConfigError(f"runtime.seed must be in [0, 2^32 - 1], got: {self.runtime.seed}")

        # 3. Inference Section Validation
        # Confidence threshold
        conf = self.inference.confidence_threshold
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            raise YOLOConfigError(f"confidence_threshold must be a numeric float, got: {type(conf).__name__}")
        if math.isnan(conf) or math.isinf(conf):
            raise YOLOConfigError("confidence_threshold must be a finite number")
        if not (0.0 < float(conf) <= 1.0):
            raise YOLOConfigError(
                f"confidence_threshold must be in (0.0, 1.0], got: {conf}"
            )

        # IoU threshold
        iou = self.inference.iou_threshold
        if isinstance(iou, bool) or not isinstance(iou, (int, float)):
            raise YOLOConfigError(f"iou_threshold must be a numeric float, got: {type(iou).__name__}")
        if math.isnan(iou) or math.isinf(iou):
            raise YOLOConfigError("iou_threshold must be a finite number")
        if not (0.0 < float(iou) <= 1.0):
            raise YOLOConfigError(
                f"iou_threshold must be in (0.0, 1.0], got: {iou}"
            )

        # Image size
        imgsz = self.inference.image_size
        if isinstance(imgsz, bool) or not isinstance(imgsz, int):
            raise YOLOConfigError(f"image_size must be an integer, got: {type(imgsz).__name__}")
        if not (val_cfg.min_image_size <= imgsz <= val_cfg.max_image_size):
            raise YOLOConfigError(
                f"image_size must be between {val_cfg.min_image_size} and {val_cfg.max_image_size}, got: {imgsz}"
            )
        # YOLO backbone stride validation (stride of 32 requirement)
        if imgsz % 32 != 0:
            raise YOLOConfigError(
                f"image_size must be a multiple of 32 (YOLO backbone downsampling stride constraint), got: {imgsz}"
            )

        # Max detections
        max_det = self.inference.max_detections
        if isinstance(max_det, bool) or not isinstance(max_det, int):
            raise YOLOConfigError(f"max_detections must be an integer, got: {type(max_det).__name__}")
        if max_det <= 0:
            raise YOLOConfigError(f"max_detections must be a positive integer, got: {max_det}")

        # Agnostic NMS
        if not isinstance(self.inference.agnostic_nms, bool):
            raise YOLOConfigError(
                f"agnostic_nms must be a boolean, got: {type(self.inference.agnostic_nms).__name__}"
            )

        # Classes filtering
        classes = self.inference.classes
        if classes is not None:
            if not isinstance(classes, (list, tuple)):
                raise YOLOConfigError(f"inference.classes must be null or a list of integers, got: {type(classes).__name__}")
            seen_classes: Set[int] = set()
            for c in classes:
                if isinstance(c, bool) or not isinstance(c, int):
                    raise YOLOConfigError(f"Class IDs in inference.classes must be integers, got: {c!r}")
                if c < 0:
                    raise YOLOConfigError(f"Class IDs cannot be negative, got: {c}")
                if c > 79:  # Standard COCO-80 maximum index is 79
                    raise YOLOConfigError(
                        f"Class ID {c} is out of COCO bounds [0, 79] for standard pretrained detection."
                    )
                if c in seen_classes:
                    raise YOLOConfigError(f"Duplicate class ID detected in inference.classes: {c}")
                seen_classes.add(c)

        # 4. Performance Section Validation
        if isinstance(self.performance.batch_size, bool) or not isinstance(self.performance.batch_size, int):
            raise YOLOConfigError("performance.batch_size must be an integer")
        if self.performance.batch_size <= 0:
            raise YOLOConfigError(f"performance.batch_size must be >= 1, got: {self.performance.batch_size}")

        if isinstance(self.performance.warmup_iterations, bool) or not isinstance(self.performance.warmup_iterations, int):
            raise YOLOConfigError("performance.warmup_iterations must be an integer")
        if self.performance.warmup_iterations < 0:
            raise YOLOConfigError(f"performance.warmup_iterations must be >= 0, got: {self.performance.warmup_iterations}")

        lat = self.performance.max_allowed_latency_ms
        if isinstance(lat, bool) or not isinstance(lat, (int, float)):
            raise YOLOConfigError("performance.max_allowed_latency_ms must be numeric")
        if lat <= 0.0:
            raise YOLOConfigError(f"performance.max_allowed_latency_ms must be > 0.0, got: {lat}")

        cpu_th = self.performance.cpu_threads
        if cpu_th is not None:
            if isinstance(cpu_th, bool) or not isinstance(cpu_th, int):
                raise YOLOConfigError("performance.cpu_threads must be null or an integer")
            if cpu_th <= 0:
                raise YOLOConfigError(f"performance.cpu_threads must be >= 1, got: {cpu_th}")

        # 5. Model Weights Verification (if requested)
        should_validate_weights = (
            validate_weights if validate_weights is not None else val_cfg.validate_weights
        )
        if should_validate_weights:
            resolved_p = self.resolve_weights_path(root)
            if not resolved_p.exists():
                raise YOLOModelNotFoundError(
                    f"Model weights file does not exist at: '{resolved_p}'"
                )
            if not resolved_p.is_file():
                raise YOLOModelNotFoundError(
                    f"Specified weights path is a directory, not a file: '{resolved_p}'"
                )
            if resolved_p.stat().st_size == 0:
                raise YOLOConfigError(
                    f"Model weights file is empty (0 bytes): '{resolved_p}'"
                )

    def apply_runtime_optimizations(self) -> None:
        """Apply configured runtime optimizations (CPU thread count, deterministic seed)."""
        if self.runtime.seed is not None and self.runtime.deterministic:
            torch.manual_seed(self.runtime.seed)
            np.random.seed(self.runtime.seed)
            logger.debug("Applied deterministic seed: %d", self.runtime.seed)

        if self.runtime.device == "cpu" and self.performance.cpu_threads is not None:
            try:
                torch.set_num_threads(self.performance.cpu_threads)
                logger.debug("Configured PyTorch CPU threads: %d", self.performance.cpu_threads)
            except Exception as e:
                logger.warning("Failed to set PyTorch CPU threads to %d: %s", self.performance.cpu_threads, e)

    # --------------------------------------------------------------------------
    # Deserialization Factory Methods
    # --------------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        project_root: Optional[Union[str, Path]] = None,
        validate: bool = True,
        validate_weights: Optional[bool] = None,
    ) -> "YOLOConfig":
        """Construct and optionally validate YOLOConfig from a dictionary.

        Handles both modular sections (model, runtime, inference, etc.) and legacy
        flat 'yolo:' configuration blocks for 100% backward compatibility.
        """
        if not isinstance(data, dict):
            raise YOLOConfigError(f"Configuration data must be a dictionary, got: {type(data).__name__}")

        cfg = cls()

        # Check for legacy 'yolo' block
        legacy_yolo = data.get("yolo", {})
        if legacy_yolo and isinstance(legacy_yolo, dict):
            if "model_name" in legacy_yolo:
                cfg.model.name = str(legacy_yolo["model_name"])
            if "model_weights_path" in legacy_yolo:
                cfg.model.weights_path = str(legacy_yolo["model_weights_path"])
            if "task" in legacy_yolo:
                cfg.model.task = str(legacy_yolo["task"])
            if "device" in legacy_yolo:
                cfg.runtime.device = str(legacy_yolo["device"])
            if "half_precision" in legacy_yolo:
                cfg.runtime.half_precision = bool(legacy_yolo["half_precision"])
            if "image_size" in legacy_yolo:
                cfg.inference.image_size = legacy_yolo["image_size"]
            if "confidence_threshold" in legacy_yolo:
                cfg.inference.confidence_threshold = legacy_yolo["confidence_threshold"]
            if "iou_threshold" in legacy_yolo:
                cfg.inference.iou_threshold = legacy_yolo["iou_threshold"]
            if "max_detections" in legacy_yolo:
                cfg.inference.max_detections = legacy_yolo["max_detections"]
            if "agnostic_nms" in legacy_yolo:
                cfg.inference.agnostic_nms = bool(legacy_yolo["agnostic_nms"])
            if "classes" in legacy_yolo:
                cfg.inference.classes = legacy_yolo["classes"]
            if "verbose" in legacy_yolo:
                cfg.inference.verbose = bool(legacy_yolo["verbose"])

        # Check for structured modular sections
        if "version" in data:
            cfg.version = str(data["version"])

        if "model" in data and isinstance(data["model"], dict):
            m = data["model"]
            if "name" in m:
                cfg.model.name = m["name"]
            if "architecture" in m:
                cfg.model.architecture = m["architecture"]
            if "weights_path" in m:
                cfg.model.weights_path = m["weights_path"]
            if "task" in m:
                cfg.model.task = m["task"]
            if "pretrained" in m:
                cfg.model.pretrained = bool(m["pretrained"])

        if "runtime" in data and isinstance(data["runtime"], dict):
            r = data["runtime"]
            if "device" in r:
                cfg.runtime.device = r["device"]
            if "half_precision" in r:
                cfg.runtime.half_precision = r["half_precision"]
            if "deterministic" in r:
                cfg.runtime.deterministic = bool(r["deterministic"])
            if "seed" in r:
                cfg.runtime.seed = r["seed"]

        if "inference" in data and isinstance(data["inference"], dict):
            inf = data["inference"]
            if "image_size" in inf:
                cfg.inference.image_size = inf["image_size"]
            if "confidence_threshold" in inf:
                cfg.inference.confidence_threshold = inf["confidence_threshold"]
            if "iou_threshold" in inf:
                cfg.inference.iou_threshold = inf["iou_threshold"]
            if "max_detections" in inf:
                cfg.inference.max_detections = inf["max_detections"]
            if "agnostic_nms" in inf:
                cfg.inference.agnostic_nms = inf["agnostic_nms"]
            if "classes" in inf:
                cfg.inference.classes = inf["classes"]
            if "verbose" in inf:
                cfg.inference.verbose = inf["verbose"]

        if "performance" in data and isinstance(data["performance"], dict):
            p = data["performance"]
            if "warmup_iterations" in p:
                cfg.performance.warmup_iterations = p["warmup_iterations"]
            if "max_allowed_latency_ms" in p:
                cfg.performance.max_allowed_latency_ms = p["max_allowed_latency_ms"]
            if "batch_size" in p:
                cfg.performance.batch_size = p["batch_size"]
            if "cpu_threads" in p:
                cfg.performance.cpu_threads = p["cpu_threads"]

        if "validation" in data and isinstance(data["validation"], dict):
            v = data["validation"]
            if "strict_mode" in v:
                cfg.validation.strict_mode = bool(v["strict_mode"])
            if "validate_weights" in v:
                cfg.validation.validate_weights = bool(v["validate_weights"])
            if "validate_parameters" in v:
                cfg.validation.validate_parameters = bool(v["validate_parameters"])
            if "min_confidence" in v:
                cfg.validation.min_confidence = float(v["min_confidence"])
            if "max_confidence" in v:
                cfg.validation.max_confidence = float(v["max_confidence"])
            if "min_iou" in v:
                cfg.validation.min_iou = float(v["min_iou"])
            if "max_iou" in v:
                cfg.validation.max_iou = float(v["max_iou"])
            if "min_image_size" in v:
                cfg.validation.min_image_size = int(v["min_image_size"])
            if "max_image_size" in v:
                cfg.validation.max_image_size = int(v["max_image_size"])
            if "supported_tasks" in v and isinstance(v["supported_tasks"], list):
                cfg.validation.supported_tasks = [str(x) for x in v["supported_tasks"]]
            if "supported_devices" in v and isinstance(v["supported_devices"], list):
                cfg.validation.supported_devices = [str(x) for x in v["supported_devices"]]
            if "supported_models" in v and isinstance(v["supported_models"], list):
                cfg.validation.supported_models = [str(x) for x in v["supported_models"]]

        if validate:
            cfg.validate(project_root=project_root, validate_weights=validate_weights)

        return cfg

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        project_root: Optional[Union[str, Path]] = None,
        validate: bool = True,
        validate_weights: Optional[bool] = None,
    ) -> "YOLOConfig":
        """Load, parse, and validate YOLOConfig from a YAML file.

        Args:
            config_path: Relative or absolute path to the YAML file.
            project_root: Base repository directory for path resolution.
            validate: Whether to immediately run schema and parameter validation.
            validate_weights: Whether to also verify weights file on disk.

        Returns:
            YOLOConfig: Instantiated and verified configuration object.

        Raises:
            YOLOConfigError: On missing file, invalid YAML syntax, or failed validation.
        """
        root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )
        p = Path(config_path)
        resolved_path = p if p.is_absolute() else (root / p).resolve()

        if not resolved_path.exists():
            raise YOLOConfigError(f"Configuration file not found at: '{resolved_path}'")

        if not resolved_path.is_file():
            raise YOLOConfigError(f"Configuration path is a directory, not a file: '{resolved_path}'")

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise YOLOConfigError(f"Failed to parse YAML file at '{resolved_path}': {e}") from e
        except Exception as e:
            raise YOLOConfigError(f"Error reading configuration file at '{resolved_path}': {e}") from e

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise YOLOConfigError(
                f"YAML configuration root must be a mapping/dict, got: {type(data).__name__}"
            )

        return cls.from_dict(
            data,
            project_root=root,
            validate=validate,
            validate_weights=validate_weights,
        )


# ==============================================================================
# Helper Function
# ==============================================================================

def load_yolo_config(
    config_path: Optional[Union[str, Path]] = None,
    preset: Optional[str] = None,
    project_root: Optional[Union[str, Path]] = None,
    validate: bool = True,
    validate_weights: Optional[bool] = None,
) -> YOLOConfig:
    """Convenience factory function for loading and configuring YOLOConfig.

    Args:
        config_path: Path to YAML configuration (defaults to config/yolo_config.yaml).
        preset: Optional preset name ('balanced', 'high_precision', 'high_recall', 'cpu_lightweight').
        project_root: Repository root for relative path resolution.
        validate: Whether to execute strict validation upon loading.
        validate_weights: Whether to enforce model weights file verification.

    Returns:
        YOLOConfig: Verified configuration instance.
    """
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[2]
    )
    target_path = Path(config_path) if config_path else root / "config" / "yolo_config.yaml"
    cfg = YOLOConfig.from_yaml(
        target_path,
        project_root=root,
        validate=validate,
        validate_weights=validate_weights,
    )

    if preset is not None:
        cfg = cfg.with_preset(preset)

    return cfg
