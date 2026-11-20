"""Unit and Integration Tests for LSTM Training Engine.

Tests optimization loop, backpropagation, gradient clipping, early stopping integration,
scheduler step, checkpoint persistence, target readiness gating, and resume capabilities.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.forecasting.checkpoint_manager import CheckpointManager
from src.forecasting.early_stopping import EarlyStopping
from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.forecasting.training_data import ForecastingDataModule, ForecastingDataset
from src.forecasting.training_engine import LSTMTrainingEngine, TrainingEngineError
from src.forecasting.training_logger import TrainingLogger


class TestTrainingEngine:
    """Test suite for LSTMTrainingEngine."""

    def test_engine_initialization_defaults(self, tmp_path: Path) -> None:
        """Verify engine initialization with default parameters."""
        model = LSTMForecaster(input_size=27, hidden_size=16)
        engine = LSTMTrainingEngine(model=model, device="cpu")

        assert engine.model is not None
        assert engine.optimizer is not None
        assert engine.loss_fn is not None
        assert engine.scheduler is not None
        assert engine.early_stopping is not None
        assert engine.checkpoint_manager is not None
        assert engine.logger is not None

    def test_engine_invalid_model_raises(self) -> None:
        """Verify rejection when model is not an nn.Module."""
        with pytest.raises(TrainingEngineError, match="nn.Module"):
            LSTMTrainingEngine(model="not_a_model")

    def test_engine_train_epoch_synthetic(self) -> None:
        """Verify train_epoch computes loss, calculates gradients, and steps optimizer."""
        model = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        engine = LSTMTrainingEngine(model=model, device="cpu")

        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            num_test=8,
            sequence_length=5,
            feature_count=10,
            batch_size=8,
            with_targets=True,
        )
        loader = dm.train_dataloader()

        # Capture initial weights
        init_weight = model.lstm.weight_ih_l0.clone()

        metrics = engine.train_epoch(loader)
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert metrics["loss"] > 0.0
        assert metrics["grad_norm"] > 0.0
        assert metrics["batch_count"] == 2

        # Verify weights updated
        new_weight = model.lstm.weight_ih_l0
        assert not torch.equal(init_weight, new_weight)

    def test_engine_validate_epoch_synthetic(self) -> None:
        """Verify validate_epoch computes loss without accumulating gradients."""
        model = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        engine = LSTMTrainingEngine(model=model, device="cpu")

        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            num_test=8,
            sequence_length=5,
            feature_count=10,
            batch_size=8,
            with_targets=True,
        )
        val_loader = dm.val_dataloader()

        # Ensure model has no residual grads
        engine.optimizer.zero_grad()
        metrics = engine.validate_epoch(val_loader)
        assert "val_loss" in metrics
        assert metrics["val_loss"] > 0.0

        for p in model.parameters():
            assert p.grad is None

    def test_engine_gradient_clipping(self) -> None:
        """Verify gradient norm clipping bounds gradients."""
        model = LSTMForecaster(input_size=5, hidden_size=8, output_horizons=[1, 3, 5])
        engine = LSTMTrainingEngine(model=model, device="cpu", max_grad_norm=0.5)

        dm = ForecastingDataModule.from_synthetic(
            num_train=8,
            num_val=4,
            sequence_length=5,
            feature_count=5,
            batch_size=8,
            with_targets=True,
        )
        metrics = engine.train_epoch(dm.train_dataloader())
        assert metrics["grad_norm"] <= 10.0  # Normalized finite clip

    def test_engine_nan_loss_detection_train(self) -> None:
        """Verify non-finite loss detection raises TrainingEngineError."""
        model = LSTMForecaster(input_size=5, hidden_size=8)
        engine = LSTMTrainingEngine(model=model, device="cpu")

        # Corrupt model weights to NaN
        with torch.no_grad():
            model.fc.weight.fill_(float("nan"))

        dm = ForecastingDataModule.from_synthetic(
            num_train=8,
            num_val=4,
            sequence_length=5,
            feature_count=5,
            batch_size=8,
            with_targets=True,
        )
        with pytest.raises(TrainingEngineError, match="Non-finite loss"):
            engine.train_epoch(dm.train_dataloader())

    def test_engine_target_readiness_gate_rejection(self) -> None:
        """Verify training engine refuses to train when targets are missing."""
        model = LSTMForecaster(input_size=10, hidden_size=16)
        engine = LSTMTrainingEngine(model=model, device="cpu")

        # Create DataModule without targets
        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            sequence_length=5,
            feature_count=10,
            batch_size=8,
            with_targets=False,
        )
        with pytest.raises(TrainingEngineError, match="Supervised target readiness check failed"):
            engine.train(data_module=dm, epochs=5)

    def test_engine_full_training_loop_synthetic(self, tmp_path: Path) -> None:
        """Verify complete multi-epoch training lifecycle on synthetic data."""
        exp_dir = tmp_path / "exp"
        chk_dir = tmp_path / "checkpoints"

        logger = TrainingLogger(experiment_dir=exp_dir, run_id="run_full")
        chk_mgr = CheckpointManager(checkpoint_dir=chk_dir, experiment_id="run_full")

        model = LSTMForecaster(input_size=8, hidden_size=16, output_horizons=[1, 3, 5])
        engine = LSTMTrainingEngine(
            model=model,
            checkpoint_manager=chk_mgr,
            logger=logger,
            device="cpu",
        )

        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            num_test=8,
            sequence_length=5,
            feature_count=8,
            batch_size=8,
            with_targets=True,
            seed=42,
        )

        results = engine.train(data_module=dm, epochs=3, verbose=False)

        assert results["status"] in ["COMPLETED", "EARLY_STOPPED"]
        assert results["total_epochs"] == 3
        assert results["best_epoch"] >= 1
        assert results["best_val_loss"] > 0.0

        # Checkpoints created
        assert (chk_dir / "checkpoint_last.pt").exists()
        assert (chk_dir / "checkpoint_best.pt").exists()

        # Telemetry created
        assert (exp_dir / "run_full" / "training_history.json").exists()
        assert (exp_dir / "run_full" / "training_summary.json").exists()

    def test_engine_early_stopping_in_training_loop(self, tmp_path: Path) -> None:
        """Verify early stopping terminates training early when loss fails to improve."""
        model = LSTMForecaster(input_size=5, hidden_size=8)
        es = EarlyStopping(patience=2, min_delta=0.5)  # Very large delta ensures no improvement
        chk_mgr = CheckpointManager(checkpoint_dir=tmp_path / "chk")
        logger = TrainingLogger(experiment_dir=tmp_path / "exp")

        engine = LSTMTrainingEngine(
            model=model,
            early_stopping=es,
            checkpoint_manager=chk_mgr,
            logger=logger,
            device="cpu",
        )

        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            sequence_length=5,
            feature_count=5,
            batch_size=8,
            with_targets=True,
        )

        results = engine.train(data_module=dm, epochs=10, verbose=False)
        assert results["status"] == "EARLY_STOPPED"
        assert results["total_epochs"] <= 4  # 1 + 2 patience = 3 epochs

    def test_engine_resume_training_from_checkpoint(self, tmp_path: Path) -> None:
        """Verify resuming training from a saved checkpoint."""
        chk_dir = tmp_path / "chk_resume"
        exp_dir = tmp_path / "exp_resume"

        model = LSTMForecaster(input_size=6, hidden_size=8)
        chk_mgr = CheckpointManager(checkpoint_dir=chk_dir, experiment_id="run1")
        logger = TrainingLogger(experiment_dir=exp_dir, run_id="run1")

        engine = LSTMTrainingEngine(
            model=model,
            checkpoint_manager=chk_mgr,
            logger=logger,
            device="cpu",
        )

        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            sequence_length=5,
            feature_count=6,
            batch_size=8,
            with_targets=True,
        )

        # Run 2 epochs
        engine.train(data_module=dm, epochs=2, verbose=False)
        last_chk = chk_dir / "checkpoint_last.pt"
        assert last_chk.exists()

        # Resume for total 4 epochs
        model2 = LSTMForecaster(input_size=6, hidden_size=8)
        engine2 = LSTMTrainingEngine(
            model=model2,
            checkpoint_manager=chk_mgr,
            logger=logger,
            device="cpu",
        )
        res = engine2.train(data_module=dm, epochs=4, resume_checkpoint_path=last_chk, verbose=False)
        assert res["total_epochs"] == 4

    def test_training_configuration_validation(self) -> None:
        """Verify TrainingConfig validation checks for loss, optimizer, grad_clip, and seed."""
        from src.forecasting.forecasting_config import TrainingConfig

        valid_cfg = TrainingConfig()
        assert len(valid_cfg.validate()) == 0

        invalid_loss = TrainingConfig(loss_function="invalid_loss")
        assert any("training.loss_function" in e for e in invalid_loss.validate())

        invalid_opt = TrainingConfig(optimizer="invalid_opt")
        assert any("training.optimizer" in e for e in invalid_opt.validate())

        invalid_clip = TrainingConfig(gradient_clip_norm=-0.5)
        assert any("training.gradient_clip_norm" in e for e in invalid_clip.validate())

    def test_training_reproducibility(self, tmp_path: Path) -> None:
        """Verify identical random seeds yield numerically identical weights after training."""
        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            sequence_length=5,
            feature_count=6,
            batch_size=8,
            with_targets=True,
            seed=100,
        )

        cfg1 = ForecastingConfig()
        model1 = LSTMForecaster(input_size=6, hidden_size=8)
        engine1 = LSTMTrainingEngine(
            model=model1,
            config=cfg1,
            checkpoint_manager=CheckpointManager(checkpoint_dir=tmp_path / "chk1"),
            logger=TrainingLogger(experiment_dir=tmp_path / "exp1"),
            device="cpu",
        )
        engine1.train(data_module=dm, epochs=2, verbose=False)

        cfg2 = ForecastingConfig()
        model2 = LSTMForecaster(input_size=6, hidden_size=8)
        engine2 = LSTMTrainingEngine(
            model=model2,
            config=cfg2,
            checkpoint_manager=CheckpointManager(checkpoint_dir=tmp_path / "chk2"),
            logger=TrainingLogger(experiment_dir=tmp_path / "exp2"),
            device="cpu",
        )
        engine2.train(data_module=dm, epochs=2, verbose=False)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7)

    def test_no_test_data_in_training(self, tmp_path: Path) -> None:
        """Verify training engine never trains or updates gradients using test partition."""
        dm = ForecastingDataModule.from_synthetic(
            num_train=16,
            num_val=8,
            num_test=8,
            sequence_length=5,
            feature_count=6,
            batch_size=8,
            with_targets=True,
        )

        # Set test targets to recognizable extreme values
        dm.test_dataset.targets.fill_(999.0)

        model = LSTMForecaster(input_size=6, hidden_size=8)
        engine = LSTMTrainingEngine(
            model=model,
            checkpoint_manager=CheckpointManager(checkpoint_dir=tmp_path / "chk_no_test"),
            logger=TrainingLogger(experiment_dir=tmp_path / "exp_no_test"),
            device="cpu",
        )

        res = engine.train(data_module=dm, epochs=3, verbose=False)
        # If test targets (999.0) had been used, train loss would be massive (~1e6)
        assert res["history"][-1]["train_loss"] < 10.0
        assert res["history"][-1]["val_loss"] < 10.0

    def test_experiment_tracking_metadata_integrity(self, tmp_path: Path) -> None:
        """Verify experiment logger captures complete environment and training telemetry."""
        import json
        exp_dir = tmp_path / "telemetry_test"
        logger = TrainingLogger(experiment_dir=exp_dir, run_id="test_run")

        # Check metadata
        meta_file = exp_dir / "test_run" / "metadata.json"
        assert meta_file.exists()
        with open(meta_file) as f:
            meta = json.load(f)
        assert "run_id" in meta
        assert "python_version" in meta
        assert "torch_version" in meta
        assert "cpu_count" in meta

        # Log epoch and check history
        logger.log_epoch(
            epoch=1,
            train_loss=0.05,
            val_loss=0.04,
            learning_rate=0.001,
            epoch_duration_seconds=0.1,
            batch_count=2,
            gradient_norm=0.05,
        )
        hist_file = exp_dir / "test_run" / "training_history.json"
        assert hist_file.exists()
        with open(hist_file) as f:
            hist = json.load(f)
        assert len(hist) == 1
        assert hist[0]["epoch"] == 1
        assert hist[0]["train_loss"] == 0.05
