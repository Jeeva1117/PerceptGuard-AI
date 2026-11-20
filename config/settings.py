"""PerceptGuard-AI: System Configuration (Backward Compatibility Layer).

Maintains 100% backward compatibility with legacy imports while delegating
to the centralized, strongly typed SystemSettings configuration engine.
"""

from __future__ import annotations

import os
from pathlib import Path
import torch

from src.config.system_settings import get_settings, reload_settings

# Initialize system configuration
_settings = get_settings()

# Base Directory (Project Root)
BASE_DIR: Path = _settings.base_dir

# Storage Directories
DATASETS_DIR: Path = _settings.datasets_dir
RAW_DATA_DIR: Path = _settings.raw_data_dir
PROCESSED_DATA_DIR: Path = _settings.processed_data_dir
MODELS_DIR: Path = _settings.models_dir
OUTPUTS_DIR: Path = _settings.outputs_dir
CONFIG_DIR: Path = _settings.config_dir

# Set Ultralytics config directory within workspace to avoid global sandbox permission warnings
YOLO_CONFIG_DIR: Path = _settings.yolo_config_dir
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

# Compute Device Selection
DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE: str = _settings.model.device

# Detection & Tracking Settings
DEFAULT_YOLO_MODEL: str = _settings.model.yolo_model
DEFAULT_CONF_THRESHOLD: float = _settings.model.confidence_threshold
DEFAULT_IOU_THRESHOLD: float = _settings.model.iou_threshold
TRACKER_TYPE: str = _settings.video_processing.tracker_type

# Temporal Sequence & Forecasting Configuration
SEQUENCE_LENGTH: int = _settings.video_processing.historical_sequence_length
FORECAST_HORIZONS: list[int] = _settings.video_processing.forecast_horizons

# Evaluation & Ground-Truth IoU Thresholds for Reliability Labeling
RELIABILITY_IOU_CORRECT_THRESHOLD: float = _settings.video_processing.reliability_iou_correct_threshold
RELIABILITY_IOU_HIGH_THRESHOLD: float = _settings.video_processing.reliability_iou_high_threshold

__all__ = [
    "BASE_DIR",
    "DATASETS_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "CONFIG_DIR",
    "YOLO_CONFIG_DIR",
    "DEFAULT_DEVICE",
    "DEVICE",
    "DEFAULT_YOLO_MODEL",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "TRACKER_TYPE",
    "SEQUENCE_LENGTH",
    "FORECAST_HORIZONS",
    "RELIABILITY_IOU_CORRECT_THRESHOLD",
    "RELIABILITY_IOU_HIGH_THRESHOLD",
    "get_settings",
    "reload_settings",
]
