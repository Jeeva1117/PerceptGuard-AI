"""Global pytest configuration and environment setup for PerceptGuard AI."""

import os
from pathlib import Path

# Ensure Ultralytics writes settings and assets within workspace
_workspace_root = Path(__file__).resolve().parent.parent
_ultralytics_dir = _workspace_root / ".ultralytics"
_ultralytics_dir.mkdir(parents=True, exist_ok=True)

if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str(_ultralytics_dir)
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_VERBOSE"] = "False"
