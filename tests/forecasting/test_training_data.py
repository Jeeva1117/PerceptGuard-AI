"""Unit and Integration Tests for Forecasting Training Data Module.

Tests PyTorch Dataset construction, dimension validation, non-finite checks,
DataLoader batching, split protection, and target readiness detection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.forecasting.training_data import (
    DatasetSplitError,
    ForecastingDataModule,
    ForecastingDataset,
    TargetContractError,
)


class TestTrainingData:
    """Test suite for ForecastingDataset and ForecastingDataModule."""

    def test_dataset_initialization_numpy(self) -> None:
        """Verify initialization with NumPy arrays."""
        seqs = np.random.randn(20, 5, 27).astype(np.float32)
        targets = np.random.uniform(0.1, 0.9, size=(20, 3)).astype(np.float32)
        ds = ForecastingDataset(sequences=seqs, targets=targets, horizons=[1, 3, 5])
        assert len(ds) == 20
        assert ds.sequence_length == 5
        assert ds.feature_count == 27
        assert ds.targets is not None
        assert ds.targets.shape == (20, 3)

    def test_dataset_initialization_torch(self) -> None:
        """Verify initialization with PyTorch tensors."""
        seqs = torch.randn(15, 10, 27)
        targets = torch.rand(15, 3)
        ds = ForecastingDataset(sequences=seqs, targets=targets)
        assert len(ds) == 15
        assert ds.sequence_length == 10
        assert ds.feature_count == 27

    def test_dataset_invalid_dimensions(self) -> None:
        """Verify rejection of 2D or 4D sequence tensors."""
        with pytest.raises(ValueError, match="3-dimensional"):
            ForecastingDataset(sequences=np.zeros((10, 27)))
        with pytest.raises(ValueError, match="3-dimensional"):
            ForecastingDataset(sequences=np.zeros((10, 5, 27, 2)))

    def test_dataset_empty_sequences(self) -> None:
        """Verify rejection of empty sequence tensors."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ForecastingDataset(sequences=np.empty((0, 5, 27)))

    def test_dataset_nan_sequences(self) -> None:
        """Verify rejection of sequences containing NaN."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        seqs[2, 1, 3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            ForecastingDataset(sequences=seqs)

    def test_dataset_inf_sequences(self) -> None:
        """Verify rejection of sequences containing Inf."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        seqs[3, 2, 4] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            ForecastingDataset(sequences=seqs)

    def test_dataset_targets_dimension_mismatch(self) -> None:
        """Verify rejection when target columns do not match horizons."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        targets = np.zeros((10, 4), dtype=np.float32)  # 4 columns vs 3 horizons
        with pytest.raises(TargetContractError, match="Target dimension mismatch"):
            ForecastingDataset(sequences=seqs, targets=targets, horizons=[1, 3, 5])

    def test_dataset_targets_sample_count_mismatch(self) -> None:
        """Verify rejection when target rows != sequence rows."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        targets = np.zeros((8, 3), dtype=np.float32)
        with pytest.raises(TargetContractError, match="Sample count mismatch"):
            ForecastingDataset(sequences=seqs, targets=targets)

    def test_dataset_targets_nan_detection(self) -> None:
        """Verify rejection when targets contain NaN."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        targets = np.zeros((10, 3), dtype=np.float32)
        targets[1, 1] = np.nan
        with pytest.raises(TargetContractError, match="non-finite"):
            ForecastingDataset(sequences=seqs, targets=targets)

    def test_dataset_sequence_ids_length_mismatch(self) -> None:
        """Verify rejection when sequence_ids length != sample count."""
        seqs = np.zeros((5, 5, 27), dtype=np.float32)
        with pytest.raises(ValueError, match="sequence_ids length"):
            ForecastingDataset(sequences=seqs, sequence_ids=["id1", "id2"])

    def test_dataset_getitem_with_and_without_targets(self) -> None:
        """Verify __getitem__ returns (seq, target) tuples correctly."""
        seqs = np.ones((5, 5, 27), dtype=np.float32)
        targets = np.ones((5, 3), dtype=np.float32) * 0.75
        ds_with = ForecastingDataset(sequences=seqs, targets=targets)
        s, t = ds_with[0]
        assert s.shape == (5, 27)
        assert t.shape == (3,)
        assert t[0].item() == pytest.approx(0.75)

        ds_without = ForecastingDataset(sequences=seqs, targets=None)
        s2, t2 = ds_without[0]
        assert s2.shape == (5, 27)
        assert t2.numel() == 0

    def test_datamodule_synthetic_construction(self) -> None:
        """Verify synthetic data module creation and DataLoader properties."""
        dm = ForecastingDataModule.from_synthetic(
            num_train=32,
            num_val=8,
            num_test=8,
            sequence_length=5,
            feature_count=27,
            batch_size=8,
            with_targets=True,
        )
        assert dm.has_targets is True
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert len(train_loader) == 4  # 32 / 8
        assert len(val_loader) == 1   # 8 / 8
        assert test_loader is not None
        assert len(test_loader) == 1  # 8 / 8

        # Iterate one batch
        x_b, y_b = next(iter(train_loader))
        assert x_b.shape == (8, 5, 27)
        assert y_b.shape == (8, 3)

    def test_datamodule_split_integrity_leakage_detection(self) -> None:
        """Verify detection of overlapping IDs between train and val splits."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        targets = np.zeros((10, 3), dtype=np.float32)

        # Overlapping ID 'seq_shared'
        ds_train = ForecastingDataset(
            sequences=seqs, targets=targets, sequence_ids=[f"seq_{i}" for i in range(10)]
        )
        ds_val = ForecastingDataset(
            sequences=seqs[:3], targets=targets[:3], sequence_ids=["seq_0", "val_1", "val_2"]
        )

        with pytest.raises(DatasetSplitError, match="Data leakage detected"):
            ForecastingDataModule(train_dataset=ds_train, val_dataset=ds_val)

    def test_datamodule_train_test_split_leakage_detection(self) -> None:
        """Verify detection of overlapping IDs between train and test splits."""
        seqs = np.zeros((10, 5, 27), dtype=np.float32)
        targets = np.zeros((10, 3), dtype=np.float32)

        ds_train = ForecastingDataset(
            sequences=seqs, targets=targets, sequence_ids=[f"train_{i}" for i in range(10)]
        )
        ds_val = ForecastingDataset(
            sequences=seqs[:3], targets=targets[:3], sequence_ids=[f"val_{i}" for i in range(3)]
        )
        ds_test = ForecastingDataset(
            sequences=seqs[:2], targets=targets[:2], sequence_ids=["train_0", "test_1"]
        )

        with pytest.raises(DatasetSplitError, match="Data leakage detected"):
            ForecastingDataModule(train_dataset=ds_train, val_dataset=ds_val, test_dataset=ds_test)

    def test_datamodule_from_real_day5_artifacts(self) -> None:
        """Verify loading from real Day 5 driving_sample artifacts and target honesty."""
        artifacts_dir = "datasets/processed/driving_sample/features/sequences/run_official"
        dm = ForecastingDataModule.from_artifacts(
            artifacts_dir=artifacts_dir,
            sequence_length=5,
            batch_size=8,
        )
        # Verify split isolation & record counts match Day 5 metadata
        assert len(dm.train_dataset) == 101
        assert len(dm.val_dataset) == 18
        assert dm.test_dataset is not None
        assert len(dm.test_dataset) == 19
        assert dm.has_targets is False  # Honest reporting: Day 5 has no target labels

    def test_datamodule_from_real_day8_labeled_artifacts(self) -> None:
        """Verify loading from real Day 8 labeled metadata populates ground-truth targets."""
        artifacts_dir = "datasets/processed/driving_sample/features/sequences/run_official"
        labeled_meta = "datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json"
        
        dm = ForecastingDataModule.from_artifacts(
            artifacts_dir=artifacts_dir,
            sequence_length=5,
            batch_size=8,
            metadata_file=labeled_meta,
        )
        assert len(dm.train_dataset) == 101
        assert len(dm.val_dataset) == 18
        assert dm.has_targets is True
        
        # Verify batch extraction with aligned targets
        train_loader = dm.train_dataloader()
        seq_batch, target_batch = next(iter(train_loader))
        assert seq_batch.shape == (8, 5, 27)
        assert target_batch.shape == (8, 3)
        assert torch.isfinite(seq_batch).all()
        assert torch.isfinite(target_batch).all()
