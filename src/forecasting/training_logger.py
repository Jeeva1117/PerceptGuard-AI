"""Training Logger Module for PerceptGuard AI Forecasting.

Provides structured experiment logging, metrics history tracking, system metadata capture,
and JSON telemetry persistence for LSTM training workflows.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml


class TrainingLogger:
    """Records epoch telemetry, experiment summary, and system metadata."""

    def __init__(
        self,
        experiment_dir: Union[str, Path] = "experiments/forecasting/lstm_training",
        run_id: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.run_id = run_id or f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path(experiment_dir) / self.run_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

        # Save initial experiment metadata & config
        self._save_metadata()
        if config is not None:
            self.save_config(config)

    def _save_metadata(self) -> None:
        """Capture runtime and hardware environment metadata."""
        metadata = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        }
        meta_file = self.experiment_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def save_config(self, config: Any) -> None:
        """Serialize configuration object or dictionary to experiment directory."""
        cfg_file = self.experiment_dir / "experiment_config.yaml"
        if hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            cfg_dict = {"raw_config": str(config)}

        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        epoch_duration_seconds: float,
        batch_count: int,
        gradient_norm: Optional[float] = None,
        early_stopped: bool = False,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record telemetry for a single completed training epoch."""
        epoch_record: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(learning_rate),
            "epoch_duration_seconds": float(epoch_duration_seconds),
            "batch_count": int(batch_count),
            "gradient_norm": float(gradient_norm) if gradient_norm is not None else None,
            "early_stopped": bool(early_stopped),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if extra_metrics:
            epoch_record["extra_metrics"] = extra_metrics

        self.history.append(epoch_record)
        self.save_history()
        return epoch_record

    def save_history(self) -> Path:
        """Write current training history to JSON."""
        history_file = self.experiment_dir / "training_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return history_file

    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error or anomaly during training execution."""
        err_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": str(message),
            "context": context or {},
        }
        self.errors.append(err_entry)
        err_file = self.experiment_dir / "errors.json"
        with open(err_file, "w", encoding="utf-8") as f:
            json.dump(self.errors, f, indent=2)

    def save_summary(
        self,
        total_epochs: int,
        best_epoch: int,
        best_val_loss: float,
        total_duration_seconds: float,
        status: str = "COMPLETED",
        termination_reason: Optional[str] = None,
    ) -> Path:
        """Write final experiment summary to JSON."""
        self.summary = {
            "run_id": self.run_id,
            "status": status,
            "total_epochs": int(total_epochs),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "total_duration_seconds": float(total_duration_seconds),
            "termination_reason": termination_reason,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        summary_file = self.experiment_dir / "training_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
        return summary_file

    def get_history(self) -> List[Dict[str, Any]]:
        """Return in-memory training history records."""
        return list(self.history)
