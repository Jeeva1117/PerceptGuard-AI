"""PerceptGuard-AI: System Configuration.

Centralizes paths, device selection, thresholds, and hyperparameters.
Follows 12-factor application design, reading overrides from environment variables.
"""

from pathlib import Path
import os
import torch

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIG_DIR = BASE_DIR / "config"

# Set Ultralytics config directory within workspace to avoid global sandbox permission warnings
YOLO_CONFIG_DIR = BASE_DIR / ".yolo"
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

# Compute Device Selection
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

# Detection & Tracking Settings
DEFAULT_YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
DEFAULT_CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
DEFAULT_IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "bytetrack")

# Temporal Sequence & Forecasting Configuration
# Number of past frames of features fed to LSTM/GRU
SEQUENCE_LENGTH = int(os.getenv("HISTORICAL_SEQUENCE_LENGTH", "10"))
# Future frames ahead for reliability forecasting
FORECAST_HORIZONS = [int(h.strip()) for h in os.getenv("FORECAST_HORIZONS", "1,3,5").split(",")]

# Evaluation & Ground-Truth IoU Thresholds for Reliability Labeling
RELIABILITY_IOU_CORRECT_THRESHOLD = 0.50
RELIABILITY_IOU_HIGH_THRESHOLD = 0.70
