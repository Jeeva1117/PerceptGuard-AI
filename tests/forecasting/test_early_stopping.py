"""Unit Tests for Early Stopping Module.

Tests patience tracking, minimum delta thresholding, best score tracking,
model weights restoration, non-finite metric handling, and state serialization.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.forecasting.early_stopping import EarlyStopping


class SimpleTestModel(nn.Module):
    """Simple linear model for weight preservation testing."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 1)


class TestEarlyStopping:
    """Test suite for EarlyStopping class."""

    def test_early_stopping_init_defaults(self) -> None:
        """Verify default parameters."""
        es = EarlyStopping()
        assert es.patience == 8
        assert es.min_delta == 1e-4
        assert es.mode == "min"
        assert es.restore_best_weights is True
        assert es.early_stop is False
        assert es.best_score is None

    def test_early_stopping_invalid_patience(self) -> None:
        """Verify invalid patience raises ValueError."""
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=-3)

    def test_early_stopping_invalid_mode(self) -> None:
        """Verify invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(mode="invalid")

    def test_early_stopping_min_mode_improvement(self) -> None:
        """Verify continuous improvement resets counter."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        assert es.step(1.0, epoch=1) is False
        assert es.counter == 0
        assert es.best_score == 1.0

        assert es.step(0.95, epoch=2) is False
        assert es.counter == 0
        assert es.best_score == 0.95

        assert es.step(0.90, epoch=3) is False
        assert es.counter == 0
        assert es.best_score == 0.90
        assert es.early_stop is False

    def test_early_stopping_triggers_after_patience(self) -> None:
        """Verify early stopping triggers after patience consecutive stagnant epochs."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        es.step(0.50, epoch=1)  # best: 0.50

        # 3 consecutive non-improving steps
        assert es.step(0.51, epoch=2) is False
        assert es.counter == 1
        assert es.step(0.52, epoch=3) is False
        assert es.counter == 2
        assert es.step(0.505, epoch=4) is True  # Early stop triggered!
        assert es.counter == 3
        assert es.early_stop is True
        assert es.stop_reason is not None
        assert "patience" in es.stop_reason.lower() or "best metric" in es.stop_reason.lower()

    def test_early_stopping_min_delta_threshold(self) -> None:
        """Verify improvement smaller than min_delta is not counted."""
        es = EarlyStopping(patience=2, min_delta=0.05, mode="min")
        es.step(1.00, epoch=1)
        # Improvement is 0.01 < 0.05 min_delta -> counted as non-improving
        assert es.step(0.99, epoch=2) is False
        assert es.counter == 1
        assert es.step(0.98, epoch=3) is True
        assert es.counter == 2

    def test_early_stopping_max_mode(self) -> None:
        """Verify behavior in mode='max'."""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="max")
        es.step(0.5, epoch=1)
        assert es.best_score == 0.5
        es.step(0.7, epoch=2)  # improved
        assert es.best_score == 0.7
        assert es.counter == 0
        es.step(0.65, epoch=3)  # worse
        assert es.counter == 1
        assert es.step(0.60, epoch=4) is True
        assert es.counter == 2

    def test_early_stopping_non_finite_metric(self) -> None:
        """Verify NaN or Inf metric triggers immediate halt."""
        es = EarlyStopping(patience=5)
        es.step(0.5, epoch=1)
        assert es.step(float("nan"), epoch=2) is True
        assert es.early_stop is True
        assert "non-finite" in es.stop_reason.lower()

    def test_early_stopping_restore_best_model_weights(self) -> None:
        """Verify best model weights are preserved and accurately restored."""
        model = SimpleTestModel()
        es = EarlyStopping(patience=2, restore_best_weights=True)

        # Epoch 1: set weights to 1.0
        with torch.no_grad():
            model.fc.weight.fill_(1.0)
            model.fc.bias.fill_(1.0)
        es.step(0.10, epoch=1, model=model)  # Best epoch

        # Epoch 2: degrade loss, change weights to 5.0
        with torch.no_grad():
            model.fc.weight.fill_(5.0)
            model.fc.bias.fill_(5.0)
        es.step(0.50, epoch=2, model=model)

        # Epoch 3: degrade loss, change weights to 9.0
        with torch.no_grad():
            model.fc.weight.fill_(9.0)
            model.fc.bias.fill_(9.0)
        stopped = es.step(0.60, epoch=3, model=model)
        assert stopped is True

        # Now restore model to best weights
        assert model.fc.weight[0, 0].item() == pytest.approx(9.0)
        restored = es.restore_model(model)
        assert restored is True
        assert model.fc.weight[0, 0].item() == pytest.approx(1.0)
        assert model.fc.bias[0].item() == pytest.approx(1.0)

    def test_early_stopping_state_dict_serialization(self) -> None:
        """Verify serialization and deserialization via state_dict."""
        es1 = EarlyStopping(patience=4, min_delta=0.02)
        es1.step(0.8, epoch=1)
        es1.step(0.85, epoch=2)
        state = es1.state_dict()

        es2 = EarlyStopping()
        es2.load_state_dict(state)
        assert es2.patience == 4
        assert es2.min_delta == 0.02
        assert es2.counter == 1
        assert es2.best_score == 0.8
        assert es2.best_epoch == 1
