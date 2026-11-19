"""Checkpoint Management Module for PerceptGuard AI Forecasting.

Handles atomic checkpoint saving, loading, version verification, and training resumption
with strict architectural compatibility validation.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn


class CheckpointError(ValueError):
    """Raised when checkpoint saving, loading, or compatibility validation fails."""
    pass


class CheckpointManager:
    """Manages saving and loading model checkpoints and training state."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "models/forecasting",
        experiment_id: str = "default_experiment",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_id = experiment_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        is_best: bool,
        scheduler: Optional[Any] = None,
        best_epoch: Optional[int] = None,
        best_val_loss: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Save training state checkpoint atomically.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer instance.
            epoch: Current epoch (1-based).
            val_loss: Current validation loss.
            is_best: Whether this checkpoint achieves the best validation loss so far.
            scheduler: Optional learning rate scheduler.
            best_epoch: Best epoch index.
            best_val_loss: Best validation loss recorded.
            metadata: Additional experiment and environment metadata.
            filename: Optional custom filename (defaults to checkpoint_last.pt).
            
        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_dict: Dict[str, Any] = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "best_epoch": best_epoch if best_epoch is not None else epoch,
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else float(val_loss),
            "is_best": bool(is_best),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "experiment_id": self.experiment_id,
            "metadata": metadata or {},
        }

        # Extract model configuration attributes if available
        if hasattr(model, "input_size"):
            checkpoint_dict["input_size"] = getattr(model, "input_size")
        if hasattr(model, "hidden_size"):
            checkpoint_dict["hidden_size"] = getattr(model, "hidden_size")
        if hasattr(model, "output_horizons"):
            checkpoint_dict["output_horizons"] = list(getattr(model, "output_horizons"))

        target_filename = filename or "checkpoint_last.pt"
        target_path = self.checkpoint_dir / target_filename

        # Write to temporary file first for atomic persistence
        temp_file = target_path.with_suffix(".tmp")
        try:
            torch.save(checkpoint_dict, temp_file)
            shutil.move(str(temp_file), str(target_path))
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise CheckpointError(f"Failed to persist checkpoint to {target_path}: {e}") from e

        # If this is the best model, also atomically duplicate to checkpoint_best.pt
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copyfile(str(target_path), str(best_path))

        return target_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        strict_architecture: bool = True,
    ) -> Dict[str, Any]:
        """Load and validate training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint .pt file.
            model: Optional model instance to load weights into.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional scheduler to restore state into.
            device: Target device (defaults to 'cpu').
            strict_architecture: Validate feature dimensions and horizons match model.
            
        Returns:
            Dictionary containing loaded checkpoint metadata and state.
        """
        chk_path = Path(checkpoint_path)
        if not chk_path.exists():
            raise CheckpointError(f"Checkpoint file not found: {chk_path}")

        try:
            checkpoint_dict = torch.load(chk_path, map_location=device, weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint from {chk_path}: {e}") from e

        if not isinstance(checkpoint_dict, dict):
            raise CheckpointError(f"Invalid checkpoint format in {chk_path}: expected dict")

        # Architectural validation
        if model is not None and strict_architecture:
            chk_input_size = checkpoint_dict.get("input_size")
            model_input_size = getattr(model, "input_size", None)
            if chk_input_size is not None and model_input_size is not None:
                if chk_input_size != model_input_size:
                    raise CheckpointError(
                        f"Architecture mismatch: checkpoint input_size ({chk_input_size}) != "
                        f"model input_size ({model_input_size})"
                    )

            chk_horizons = checkpoint_dict.get("output_horizons")
            model_horizons = getattr(model, "output_horizons", None)
            if chk_horizons is not None and model_horizons is not None:
                if list(chk_horizons) != list(model_horizons):
                    raise CheckpointError(
                        f"Horizon mismatch: checkpoint output_horizons ({chk_horizons}) != "
                        f"model output_horizons ({list(model_horizons)})"
                    )

            if "model_state_dict" in checkpoint_dict:
                model.load_state_dict(checkpoint_dict["model_state_dict"])
            else:
                raise CheckpointError(f"Missing 'model_state_dict' in checkpoint {chk_path}")

        if optimizer is not None:
            if checkpoint_dict.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

        if scheduler is not None:
            if checkpoint_dict.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

        return checkpoint_dict

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find checkpoint_last.pt in the managed directory."""
        candidate = self.checkpoint_dir / "checkpoint_last.pt"
        if candidate.exists():
            return candidate
        return None

    def find_best_checkpoint(self) -> Optional[Path]:
        """Find checkpoint_best.pt in the managed directory."""
        candidate = self.checkpoint_dir / "checkpoint_best.pt"
        if candidate.exists():
            return candidate
        return None
