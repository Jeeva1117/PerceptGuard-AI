"""Forecasting Training Data Module for PerceptGuard AI.

Provides PyTorch Dataset and DataModule abstractions for temporal sequences,
strict dataset split protection, dimension verification, and multi-horizon target alignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.forecasting.baseline_data_loader import BaselineDataLoader, BaselineDataset
from src.forecasting.forecasting_config import ForecastingConfig


class TargetContractError(ValueError):
    """Raised when ground truth target labels fail contract or dimension validation."""
    pass


class DatasetSplitError(ValueError):
    """Raised when dataset split allocation or partition integrity fails."""
    pass


class ForecastingDataset(Dataset):
    """PyTorch Dataset container for temporal feature sequences and aligned target labels.
    
    Attributes:
        sequences: Float tensor of shape (N, sequence_length, feature_count).
        targets: Optional float tensor of shape (N, num_horizons).
        sequence_ids: List of string sequence identifiers.
        horizons: List of integer forecast horizons (e.g. [1, 3, 5]).
    """

    def __init__(
        self,
        sequences: Union[np.ndarray, torch.Tensor],
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sequence_ids: Optional[Sequence[str]] = None,
        horizons: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()

        # 1. Validate & Convert Sequences
        if isinstance(sequences, np.ndarray):
            seq_tensor = torch.from_numpy(sequences).float()
        elif isinstance(sequences, torch.Tensor):
            seq_tensor = sequences.float()
        else:
            raise TypeError(f"sequences must be numpy.ndarray or torch.Tensor, got {type(sequences).__name__}")

        if seq_tensor.ndim != 3:
            raise ValueError(f"sequences must be 3-dimensional (N, L, F), got ndim={seq_tensor.ndim}")
        if seq_tensor.shape[0] == 0:
            raise ValueError("sequences tensor cannot be empty (0 samples)")
        if not torch.isfinite(seq_tensor).all():
            raise ValueError("sequences tensor contains non-finite numbers (NaN or Inf)")

        self.sequences = seq_tensor
        self.num_samples = seq_tensor.shape[0]
        self.sequence_length = seq_tensor.shape[1]
        self.feature_count = seq_tensor.shape[2]
        self.horizons = list(horizons)

        # 2. Validate & Convert Targets
        if targets is not None:
            if isinstance(targets, np.ndarray):
                target_tensor = torch.from_numpy(targets).float()
            elif isinstance(targets, torch.Tensor):
                target_tensor = targets.float()
            else:
                raise TypeError(f"targets must be numpy.ndarray or torch.Tensor, got {type(targets).__name__}")

            if target_tensor.ndim != 2:
                raise TargetContractError(
                    f"targets must be 2-dimensional (N, num_horizons), got ndim={target_tensor.ndim}"
                )
            if target_tensor.shape[0] != self.num_samples:
                raise TargetContractError(
                    f"Sample count mismatch: sequences has {self.num_samples} samples, "
                    f"targets has {target_tensor.shape[0]} samples"
                )
            if target_tensor.shape[1] != len(self.horizons):
                raise TargetContractError(
                    f"Target dimension mismatch: targets has {target_tensor.shape[1]} columns, "
                    f"expected {len(self.horizons)} for horizons {self.horizons}"
                )
            if not torch.isfinite(target_tensor).all():
                raise TargetContractError("targets tensor contains non-finite numbers (NaN or Inf)")

            self.targets: Optional[torch.Tensor] = target_tensor
        else:
            self.targets = None

        # 3. Sequence IDs
        if sequence_ids is not None:
            if len(sequence_ids) != self.num_samples:
                raise ValueError(
                    f"sequence_ids length ({len(sequence_ids)}) != sample count ({self.num_samples})"
                )
            self.sequence_ids = list(sequence_ids)
        else:
            self.sequence_ids = [f"seq_{i:06d}" for i in range(self.num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (sequence, target) tuple for batch generation.
        
        If targets are not available, returns an empty tensor for the target.
        """
        seq = self.sequences[idx]
        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = torch.empty(0, dtype=torch.float32)
        return seq, target

    def get_sequence_id(self, idx: int) -> str:
        """Retrieve sequence identifier for the given index."""
        return self.sequence_ids[idx]


class ForecastingDataModule:
    """Manages dataset partitioning, split integrity, and DataLoader construction.
    
    Enforces strict temporal split boundaries:
    - Train split is used strictly for training.
    - Val split is used strictly for validation and early stopping.
    - Test split is preserved untouched for final evaluation.
    """

    def __init__(
        self,
        train_dataset: ForecastingDataset,
        val_dataset: ForecastingDataset,
        test_dataset: Optional[ForecastingDataset] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        horizons: Sequence[int] = (1, 3, 5),
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.horizons = list(horizons)

        # Split integrity check: verify no overlapping sequence IDs
        train_ids = set(train_dataset.sequence_ids)
        val_ids = set(val_dataset.sequence_ids)
        overlap = train_ids.intersection(val_ids)
        if overlap:
            raise DatasetSplitError(
                f"Data leakage detected! {len(overlap)} overlapping sequence IDs found between train and val splits."
            )

        if test_dataset is not None:
            test_ids = set(test_dataset.sequence_ids)
            train_test_overlap = train_ids.intersection(test_ids)
            if train_test_overlap:
                raise DatasetSplitError(
                    f"Data leakage detected! {len(train_test_overlap)} overlapping sequence IDs found between train and test splits."
                )
            val_test_overlap = val_ids.intersection(test_ids)
            if val_test_overlap:
                raise DatasetSplitError(
                    f"Data leakage detected! {len(val_test_overlap)} overlapping sequence IDs found between val and test splits."
                )

    @property
    def has_targets(self) -> bool:
        """Check if both train and validation datasets contain ground-truth targets."""
        return (self.train_dataset.targets is not None) and (self.val_dataset.targets is not None)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Construct DataLoader for training split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct DataLoader for validation split."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(self, shuffle: bool = False) -> Optional[DataLoader]:
        """Construct DataLoader for test split."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Union[str, Path],
        sequence_length: int = 5,
        horizons: Sequence[int] = (1, 3, 5),
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> ForecastingDataModule:
        """Construct ForecastingDataModule from Day 5 sequence directory artifacts.
        
        Args:
            artifacts_dir: Path to directory containing sequences_L*.npy, feature_manifest.json, etc.
            sequence_length: Temporal sequence length L.
            horizons: Forecast horizons.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.
            
        Returns:
            Configured ForecastingDataModule.
        """
        loader = BaselineDataLoader(artifacts_dir=artifacts_dir)
        dataset = loader.load_sequence_dataset(
            sequence_length=sequence_length,
            target_horizons=horizons,
        )

        train_ds = dataset.filter_by_split("train")
        val_ds = dataset.filter_by_split("val")
        test_ds = dataset.filter_by_split("test")

        if len(train_ds) == 0:
            raise DatasetSplitError("Training split contains 0 sequence records.")
        if len(val_ds) == 0:
            raise DatasetSplitError("Validation split contains 0 sequence records.")

        # Extract targets if available
        train_targets = None
        val_targets = None
        test_targets = None

        if dataset.target_labels_available:
            # Build target matrix (N, num_horizons)
            def _extract_target_matrix(ds_split: BaselineDataset) -> Optional[np.ndarray]:
                rows = []
                for rec in ds_split.records:
                    row = [rec.target_labels.get(h) for h in horizons]
                    if any(v is None or not np.isfinite(v) for v in row):
                        return None
                    rows.append(row)
                return np.array(rows, dtype=np.float32)

            train_targets = _extract_target_matrix(train_ds)
            val_targets = _extract_target_matrix(val_ds)
            test_targets = _extract_target_matrix(test_ds) if len(test_ds) > 0 else None

        train_dataset = ForecastingDataset(
            sequences=train_ds.tensors,
            targets=train_targets,
            sequence_ids=[r.sequence_id for r in train_ds.records],
            horizons=horizons,
        )

        val_dataset = ForecastingDataset(
            sequences=val_ds.tensors,
            targets=val_targets,
            sequence_ids=[r.sequence_id for r in val_ds.records],
            horizons=horizons,
        )

        test_dataset = None
        if len(test_ds) > 0:
            test_dataset = ForecastingDataset(
                sequences=test_ds.tensors,
                targets=test_targets,
                sequence_ids=[r.sequence_id for r in test_ds.records],
                horizons=horizons,
            )

        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            horizons=horizons,
        )

    @classmethod
    def from_synthetic(
        cls,
        num_train: int = 64,
        num_val: int = 16,
        num_test: int = 16,
        sequence_length: int = 5,
        feature_count: int = 27,
        horizons: Sequence[int] = (1, 3, 5),
        batch_size: int = 8,
        with_targets: bool = True,
        seed: int = 42,
    ) -> ForecastingDataModule:
        """Create a synthetic ForecastingDataModule for testing and validation."""
        rng = np.random.RandomState(seed)

        # Generate sequence tensors
        train_x = rng.randn(num_train, sequence_length, feature_count).astype(np.float32)
        val_x = rng.randn(num_val, sequence_length, feature_count).astype(np.float32)
        test_x = rng.randn(num_test, sequence_length, feature_count).astype(np.float32)

        train_y = None
        val_y = None
        test_y = None

        if with_targets:
            # Generate bounded regression targets in [0.0, 1.0]
            train_y = rng.uniform(0.1, 0.9, size=(num_train, len(horizons))).astype(np.float32)
            val_y = rng.uniform(0.1, 0.9, size=(num_val, len(horizons))).astype(np.float32)
            test_y = rng.uniform(0.1, 0.9, size=(num_test, len(horizons))).astype(np.float32)

        train_ds = ForecastingDataset(
            sequences=train_x,
            targets=train_y,
            sequence_ids=[f"train_seq_{i:04d}" for i in range(num_train)],
            horizons=horizons,
        )
        val_ds = ForecastingDataset(
            sequences=val_x,
            targets=val_y,
            sequence_ids=[f"val_seq_{i:04d}" for i in range(num_val)],
            horizons=horizons,
        )
        test_ds = ForecastingDataset(
            sequences=test_x,
            targets=test_y,
            sequence_ids=[f"test_seq_{i:04d}" for i in range(num_test)],
            horizons=horizons,
        )

        return cls(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            batch_size=batch_size,
            num_workers=0,
            horizons=horizons,
        )
