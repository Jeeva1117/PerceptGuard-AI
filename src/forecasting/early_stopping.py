"""Early Stopping Module for PerceptGuard AI Forecasting.

Provides robust, configurable early stopping mechanics to prevent overfitting
and automatically restore the best model checkpoint weights during training.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Monitors validation metrics to halt training when improvement plateaus.
    
    Attributes:
        patience: Number of epochs to wait for improvement before halting.
        min_delta: Minimum metric improvement required to qualify as progress.
        mode: Metric optimization direction ('min' for loss, 'max' for score).
        restore_best_weights: Whether to store and restore best model weights upon halt.
    """

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        if not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be an integer >= 1, got: {patience}")
        if not isinstance(min_delta, (int, float)) or min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0.0, got: {min_delta}")
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got: '{mode}'")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.early_stop: bool = False
        self.stop_reason: Optional[str] = None
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None

    def step(
        self,
        metric: float,
        epoch: int,
        model: Optional[nn.Module] = None,
    ) -> bool:
        """Evaluate a new epoch metric.
        
        Args:
            metric: Current validation metric (e.g. validation loss).
            epoch: Current epoch index (1-based).
            model: Optional PyTorch model whose state_dict is saved when improved.
            
        Returns:
            Boolean flag indicating whether training should stop (early_stop).
        """
        if not np.isfinite(metric):
            self.early_stop = True
            self.stop_reason = f"Non-finite validation metric encountered: {metric}"
            return True

        if self.best_score is None:
            # First epoch initialization
            self.best_score = float(metric)
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self._save_best_weights(model)
            return False

        improved = False
        if self.mode == "min":
            if metric < (self.best_score - self.min_delta):
                improved = True
        else:  # mode == "max"
            if metric > (self.best_score + self.min_delta):
                improved = True

        if improved:
            self.best_score = float(metric)
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self._save_best_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_reason = (
                    f"No improvement of at least {self.min_delta} for {self.patience} consecutive epochs. "
                    f"Best metric: {self.best_score:.6f} at epoch {self.best_epoch}."
                )

        return self.early_stop

    def _save_best_weights(self, model: nn.Module) -> None:
        """Deep copy model state dictionary to CPU tensors."""
        self.best_weights = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    def restore_model(self, model: nn.Module) -> bool:
        """Restore model weights to the best recorded epoch.
        
        Returns:
            True if weights were restored, False if no weights were saved.
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Serialize early stopping state for checkpointing."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "restore_best_weights": self.restore_best_weights,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "stop_reason": self.stop_reason,
            "has_best_weights": self.best_weights is not None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore early stopping state from serialized state dict."""
        self.patience = state_dict.get("patience", self.patience)
        self.min_delta = state_dict.get("min_delta", self.min_delta)
        self.mode = state_dict.get("mode", self.mode)
        self.restore_best_weights = state_dict.get("restore_best_weights", self.restore_best_weights)
        self.best_score = state_dict.get("best_score", self.best_score)
        self.best_epoch = state_dict.get("best_epoch", self.best_epoch)
        self.counter = state_dict.get("counter", self.counter)
        self.early_stop = state_dict.get("early_stop", self.early_stop)
        self.stop_reason = state_dict.get("stop_reason", self.stop_reason)
