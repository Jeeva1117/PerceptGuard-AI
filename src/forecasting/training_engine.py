"""LSTM Training Engine Module for PerceptGuard AI Forecasting.

Implements the complete, CPU-optimized training loop: forward pass, loss calculation,
gradient backpropagation, gradient clipping, AdamW optimization, ReduceLROnPlateau scheduling,
early stopping, checkpoint management, and telemetry logging.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.forecasting.checkpoint_manager import CheckpointManager
from src.forecasting.early_stopping import EarlyStopping
from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.lstm_forecaster import LSTMForecaster, create_forecasting_loss
from src.forecasting.training_data import ForecastingDataModule, ForecastingDataset
from src.forecasting.training_logger import TrainingLogger


class TrainingEngineError(ValueError):
    """Raised when training engine configuration, data readiness, or execution fails."""
    pass


class LSTMTrainingEngine:
    """Production training engine for LSTM reliability forecasting models."""

    def __init__(
        self,
        model: LSTMForecaster,
        config: Optional[ForecastingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[TrainingLogger] = None,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
    ) -> None:
        self.config = config or ForecastingConfig()
        self.device = torch.device(device)
        self.max_grad_norm = float(max_grad_norm)

        # 1. Hardware & Threading Configuration
        if self.device.type == "cpu":
            torch.set_num_threads(self.config.runtime.num_threads)

        # 2. Model
        if not isinstance(model, nn.Module):
            raise TrainingEngineError(f"model must be an instance of nn.Module, got: {type(model).__name__}")
        self.model = model.to(self.device)

        # 3. Loss Function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = create_forecasting_loss("mse")

        # 4. Optimizer (AdamW)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=float(self.config.training.learning_rate),
                weight_decay=float(self.config.training.weight_decay),
            )

        # 5. Learning Rate Scheduler (ReduceLROnPlateau)
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )

        # 6. Early Stopping
        if early_stopping is not None:
            self.early_stopping = early_stopping
        else:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                min_delta=1e-4,
                mode="min",
                restore_best_weights=True,
            )

        # 7. Checkpoint Manager & Logger
        run_id = f"lstm_{time.strftime('%Y%m%d_%H%M%S')}"
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            chk_dir = Path(self.config.artifacts.checkpoint_directory) / run_id
            self.checkpoint_manager = CheckpointManager(checkpoint_dir=chk_dir, experiment_id=run_id)

        if logger is not None:
            self.logger = logger
        else:
            exp_dir = Path(self.config.artifacts.experiment_directory) / "lstm_training"
            self.logger = TrainingLogger(experiment_dir=exp_dir, run_id=run_id, config=self.config)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one training epoch with backpropagation and gradient clipping.
        
        Args:
            dataloader: Training split DataLoader.
            
        Returns:
            Dictionary containing epoch metrics ('loss', 'grad_norm', 'batch_count').
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        batch_count = 0

        for x_batch, y_batch in dataloader:
            if y_batch.numel() == 0:
                raise TrainingEngineError(
                    "Supervised training cannot proceed: target batch is empty. "
                    "Ground truth labels are required for backpropagation."
                )

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)

            loss = self.loss_fn(y_pred, y_batch)

            if not torch.isfinite(loss):
                raise TrainingEngineError(
                    f"Non-finite loss encountered during training: {loss.item()}. "
                    "Halting training to prevent weight corruption."
                )

            loss.backward()

            # Apply gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise TrainingEngineError(f"Non-finite gradient norm encountered: {grad_norm}")

            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += float(grad_norm)
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        avg_grad_norm = total_grad_norm / max(batch_count, 1)

        return {
            "loss": avg_loss,
            "grad_norm": avg_grad_norm,
            "batch_count": batch_count,
        }

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute validation pass without gradient tracking.
        
        Args:
            dataloader: Validation split DataLoader.
            
        Returns:
            Dictionary containing validation metrics ('val_loss', 'batch_count').
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                if y_batch.numel() == 0:
                    raise TrainingEngineError(
                        "Validation pass cannot proceed: target batch is empty."
                    )

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)

                if not torch.isfinite(loss):
                    raise TrainingEngineError(
                        f"Non-finite loss encountered during validation: {loss.item()}"
                    )

                total_loss += loss.item()
                batch_count += 1

        avg_val_loss = total_loss / max(batch_count, 1)
        return {
            "val_loss": avg_val_loss,
            "batch_count": batch_count,
        }

    def train(
        self,
        data_module: ForecastingDataModule,
        epochs: Optional[int] = None,
        resume_checkpoint_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute full training lifecycle with early stopping and checkpointing.
        
        Args:
            data_module: Validated ForecastingDataModule.
            epochs: Optional max epoch override.
            resume_checkpoint_path: Optional checkpoint path to resume from.
            verbose: Whether to print per-epoch progress.
            
        Returns:
            Dictionary summarizing training history, best epoch, and status.
        """
        # 1. Supervised Target Readiness Gate
        if not data_module.has_targets:
            raise TrainingEngineError(
                "Supervised target readiness check failed: Ground truth target labels "
                "are NOT AVAILABLE in the provided dataset. Cannot initiate supervised training."
            )

        max_epochs = epochs or self.config.training.max_epochs
        start_epoch = 1
        best_val_loss = float("inf")
        best_epoch = -1

        # 2. Resume Checkpoint Handling
        if resume_checkpoint_path is not None:
            chk = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=str(self.device),
            )
            start_epoch = chk.get("epoch", 0) + 1
            best_epoch = chk.get("best_epoch", -1)
            best_val_loss = chk.get("best_val_loss", float("inf"))
            if verbose:
                print(f"[RESUME] Resuming from epoch {start_epoch} (Best val loss: {best_val_loss:.6f} at epoch {best_epoch})")

        train_loader = data_module.train_dataloader(shuffle=True)
        val_loader = data_module.val_dataloader(shuffle=False)

        total_start_time = time.time()
        early_stopped = False
        termination_reason = "Completed all scheduled epochs"

        for epoch in range(start_epoch, max_epochs + 1):
            epoch_start_time = time.time()

            # Train pass
            train_metrics = self.train_epoch(train_loader)

            # Validation pass
            val_metrics = self.validate_epoch(val_loader)

            epoch_duration = time.time() - epoch_start_time
            val_loss = val_metrics["val_loss"]
            train_loss = train_metrics["loss"]
            grad_norm = train_metrics["grad_norm"]

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check if this epoch is best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch

            # Early stopping check
            early_stopped = self.early_stopping.step(val_loss, epoch, model=self.model)

            # Log epoch
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_duration_seconds=epoch_duration,
                batch_count=train_metrics["batch_count"],
                gradient_norm=grad_norm,
                early_stopped=early_stopped,
            )

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_loss,
                is_best=is_best,
                scheduler=self.scheduler,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                metadata={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch_duration_seconds": epoch_duration,
                },
            )

            if verbose:
                best_marker = " [BEST]" if is_best else ""
                print(
                    f"Epoch {epoch:03d}/{max_epochs:03d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}{best_marker} | "
                    f"LR: {current_lr:.6f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"Time: {epoch_duration:.2f}s"
                )

            if early_stopped:
                termination_reason = self.early_stopping.stop_reason or "Early stopping triggered"
                if verbose:
                    print(f"[EARLY STOP] {termination_reason}")
                break

        total_duration = time.time() - total_start_time

        # Restore best weights if configured
        if self.early_stopping.restore_best_weights:
            restored = self.early_stopping.restore_model(self.model)
            if verbose and restored:
                print(f"[MODEL] Restored best model weights from epoch {best_epoch} (Val Loss: {best_val_loss:.6f})")

        # Save summary
        status = "EARLY_STOPPED" if early_stopped else "COMPLETED"
        summary_path = self.logger.save_summary(
            total_epochs=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_duration_seconds=total_duration,
            status=status,
            termination_reason=termination_reason,
        )

        return {
            "status": status,
            "total_epochs": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "total_duration_seconds": total_duration,
            "termination_reason": termination_reason,
            "summary_path": str(summary_path),
            "history": self.logger.get_history(),
        }
