"""Unit Tests for Checkpoint Manager Module.

Tests saving, loading, atomic persistence, best checkpoint replication,
architecture validation, and training resumption integrity.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.forecasting.checkpoint_manager import CheckpointError, CheckpointManager
from src.forecasting.lstm_forecaster import LSTMForecaster


class TestCheckpointManager:
    """Test suite for CheckpointManager."""

    def test_checkpoint_manager_init(self, tmp_path: Path) -> None:
        """Verify initialization and directory creation."""
        chk_dir = tmp_path / "test_checkpoints"
        mgr = CheckpointManager(checkpoint_dir=chk_dir, experiment_id="exp_01")
        assert chk_dir.exists()
        assert mgr.experiment_id == "exp_01"

    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        """Verify standard save and restore roundtrip."""
        chk_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(checkpoint_dir=chk_dir, experiment_id="exp_roundtrip")

        model = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        optimizer = AdamW(model.parameters(), lr=0.005)

        saved_path = mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            val_loss=0.045,
            is_best=False,
            best_epoch=3,
            best_val_loss=0.038,
            metadata={"test_tag": "v1"},
        )
        assert saved_path.exists()
        assert saved_path.name == "checkpoint_last.pt"

        # Restore into new model and optimizer
        model2 = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        optimizer2 = AdamW(model2.parameters(), lr=0.001)

        loaded = mgr.load_checkpoint(
            checkpoint_path=saved_path,
            model=model2,
            optimizer=optimizer2,
        )

        assert loaded["epoch"] == 5
        assert loaded["val_loss"] == pytest.approx(0.045)
        assert loaded["best_epoch"] == 3
        assert loaded["best_val_loss"] == pytest.approx(0.038)
        assert loaded["metadata"]["test_tag"] == "v1"

        # Verify weights are identical
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

        # Verify optimizer lr was restored
        assert optimizer2.param_groups[0]["lr"] == pytest.approx(0.005)

    def test_save_is_best_duplicates_best_checkpoint(self, tmp_path: Path) -> None:
        """Verify saving with is_best=True creates checkpoint_best.pt."""
        chk_dir = tmp_path / "checkpoints_best"
        mgr = CheckpointManager(checkpoint_dir=chk_dir)
        model = LSTMForecaster(input_size=8, hidden_size=16)
        optimizer = AdamW(model.parameters(), lr=0.001)

        mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=2,
            val_loss=0.02,
            is_best=True,
        )

        last_pt = chk_dir / "checkpoint_last.pt"
        best_pt = chk_dir / "checkpoint_best.pt"
        assert last_pt.exists()
        assert best_pt.exists()

    def test_load_non_existent_checkpoint_raises(self, tmp_path: Path) -> None:
        """Verify error when checkpoint file does not exist."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(CheckpointError, match="not found"):
            mgr.load_checkpoint(tmp_path / "missing.pt")

    def test_load_invalid_checkpoint_format_raises(self, tmp_path: Path) -> None:
        """Verify error when checkpoint file contains invalid data."""
        bad_file = tmp_path / "corrupted.pt"
        torch.save("not_a_dictionary", bad_file)

        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(CheckpointError, match="Invalid checkpoint format"):
            mgr.load_checkpoint(bad_file)

    def test_load_checkpoint_input_size_mismatch_raises(self, tmp_path: Path) -> None:
        """Verify rejection when checkpoint input_size differs from model."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        model1 = LSTMForecaster(input_size=27, hidden_size=16)
        opt1 = AdamW(model1.parameters())
        saved_path = mgr.save_checkpoint(model=model1, optimizer=opt1, epoch=1, val_loss=0.1, is_best=True)

        # Attempt to load into model with input_size=10
        model2 = LSTMForecaster(input_size=10, hidden_size=16)
        with pytest.raises(CheckpointError, match="Architecture mismatch"):
            mgr.load_checkpoint(checkpoint_path=saved_path, model=model2)

    def test_load_checkpoint_horizons_mismatch_raises(self, tmp_path: Path) -> None:
        """Verify rejection when checkpoint horizons differ from model."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        model1 = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        opt1 = AdamW(model1.parameters())
        saved_path = mgr.save_checkpoint(model=model1, optimizer=opt1, epoch=1, val_loss=0.1, is_best=True)

        # Model with different horizons [1, 2]
        model2 = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 2])
        with pytest.raises(CheckpointError, match="Horizon mismatch"):
            mgr.load_checkpoint(checkpoint_path=saved_path, model=model2)

    def test_find_latest_and_best_checkpoint(self, tmp_path: Path) -> None:
        """Verify helper methods find_latest_checkpoint and find_best_checkpoint."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        assert mgr.find_latest_checkpoint() is None
        assert mgr.find_best_checkpoint() is None

        model = LSTMForecaster(input_size=5, hidden_size=8)
        opt = AdamW(model.parameters())
        mgr.save_checkpoint(model=model, optimizer=opt, epoch=1, val_loss=0.5, is_best=False)
        assert mgr.find_latest_checkpoint() is not None
        assert mgr.find_best_checkpoint() is None

        mgr.save_checkpoint(model=model, optimizer=opt, epoch=2, val_loss=0.3, is_best=True)
        assert mgr.find_latest_checkpoint() is not None
        assert mgr.find_best_checkpoint() is not None
