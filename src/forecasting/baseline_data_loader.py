"""Baseline Data Loader Module for PerceptGuard AI.

Loads and validates Day 5 sequence tensors, metadata, manifests, and split allocations
for baseline forecasting models, preserving sequence identifiers and preventing data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


class DataLoaderError(ValueError):
    """Raised when data loading or contract validation fails."""
    pass


@dataclass
class SequenceDatasetRecord:
    """Individual sequence record with metadata, features, and target horizon mappings."""
    sequence_id: str
    dataset_id: str
    sequence_length: int
    feature_count: int
    features: np.ndarray  # shape: (L, F)
    start_frame_id: str
    end_frame_id: str
    start_source_frame_index: int
    end_source_frame_index: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    split: str  # 'train', 'val', 'test'
    horizons: Dict[str, Dict[str, Any]]  # horizon metadata
    target_labels: Dict[int, Optional[float]]  # horizon (int) -> target value or None
    target_labels_available: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert record metadata to dictionary (excluding raw feature array)."""
        return {
            "sequence_id": self.sequence_id,
            "dataset_id": self.dataset_id,
            "sequence_length": self.sequence_length,
            "feature_count": self.feature_count,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "start_source_frame_index": self.start_source_frame_index,
            "end_source_frame_index": self.end_source_frame_index,
            "start_timestamp_seconds": self.start_timestamp_seconds,
            "end_timestamp_seconds": self.end_timestamp_seconds,
            "split": self.split,
            "horizons": self.horizons,
            "target_labels": self.target_labels,
            "target_labels_available": self.target_labels_available,
        }


@dataclass
class BaselineDataset:
    """Container for loaded and split-partitioned sequence records."""
    dataset_id: str
    sequence_length: int
    feature_count: int
    ordered_feature_names: List[str]
    records: List[SequenceDatasetRecord]
    tensors: np.ndarray  # (N, L, F)
    target_labels_available: bool

    def __len__(self) -> int:
        return len(self.records)

    def filter_by_split(self, split: str) -> BaselineDataset:
        """Filter dataset records by split partition ('train', 'val', 'test')."""
        target_split = split.lower().strip()
        filtered_indices = [
            i for i, r in enumerate(self.records) if r.split.lower() == target_split
        ]
        filtered_records = [self.records[i] for i in filtered_indices]
        filtered_tensors = (
            self.tensors[filtered_indices]
            if len(filtered_indices) > 0
            else np.empty((0, self.sequence_length, self.feature_count), dtype=self.tensors.dtype)
        )
        return BaselineDataset(
            dataset_id=self.dataset_id,
            sequence_length=self.sequence_length,
            feature_count=self.feature_count,
            ordered_feature_names=self.ordered_feature_names,
            records=filtered_records,
            tensors=filtered_tensors,
            target_labels_available=self.target_labels_available,
        )

    def get_targets_for_horizon(
        self,
        horizon: int,
        split: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract targets for a specific forecast horizon.
        
        Returns:
            Tuple of (valid_mask, target_values_array)
        """
        ds = self if split is None else self.filter_by_split(split)
        targets: List[float] = []
        valid_mask: List[bool] = []

        for r in ds.records:
            val = r.target_labels.get(horizon)
            if val is not None and np.isfinite(val):
                targets.append(float(val))
                valid_mask.append(True)
            else:
                targets.append(np.nan)
                valid_mask.append(False)

        return np.array(valid_mask, dtype=bool), np.array(targets, dtype=np.float64)


class BaselineDataLoader:
    """Loader and validator for Day 5 sequence artifacts."""

    def __init__(
        self,
        artifacts_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.metadata_file = Path(metadata_file) if metadata_file is not None else None
        self.manifest_data: Optional[Dict[str, Any]] = None
        self.metadata_data: Optional[Dict[str, Any]] = None
        self.splits_data: Optional[Dict[str, Any]] = None
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load manifest, metadata, and split JSON artifacts."""
        if not self.artifacts_dir.exists() or not self.artifacts_dir.is_dir():
            raise DataLoaderError(f"Artifacts directory does not exist: {self.artifacts_dir}")

        manifest_file = self.artifacts_dir / "feature_manifest.json"
        if not manifest_file.exists():
            raise DataLoaderError(f"Required artifact missing: {manifest_file.name}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            self.manifest_data = json.load(f)

        if self.metadata_file is not None:
            meta_file = self.metadata_file
        else:
            meta_file = self.artifacts_dir / "sequence_metadata.json"

        if not meta_file.exists():
            raise DataLoaderError(f"Required artifact missing: {meta_file.name}")
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata_data = json.load(f)

        splits_file = self.artifacts_dir / "dataset_splits.json"
        if splits_file.exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                self.splits_data = json.load(f)

    def load_sequence_dataset(
        self,
        sequence_length: int = 5,
        target_horizons: Sequence[int] = (1, 3, 5),
    ) -> BaselineDataset:
        """Load and validate sequence dataset for a specific sequence length.
        
        Args:
            sequence_length: Length L of the sequence (e.g., 5, 10, 15, 20).
            target_horizons: Forecast horizons to extract (default: 1, 3, 5).
            
        Returns:
            Validated BaselineDataset object.
        """
        tensor_file = self.artifacts_dir / f"sequences_L{sequence_length}.npy"
        if not tensor_file.exists():
            raise DataLoaderError(f"Sequence tensor not found: {tensor_file.name}")

        try:
            tensor = np.load(tensor_file)
        except Exception as e:
            raise DataLoaderError(f"Failed to read tensor file {tensor_file.name}: {e}") from e

        if tensor.ndim != 3:
            raise DataLoaderError(
                f"Expected 3D tensor (N, L, F), got ndim={tensor.ndim} for {tensor_file.name}"
            )

        n_seq, seq_len, n_feat = tensor.shape
        if seq_len != sequence_length:
            raise DataLoaderError(
                f"Tensor sequence length {seq_len} != requested length {sequence_length}"
            )

        expected_feat = self.manifest_data.get("total_features") or len(
            self.manifest_data.get("ordered_feature_names", [])
        )
        if n_feat != expected_feat:
            raise DataLoaderError(
                f"Tensor feature count {n_feat} does not match manifest count {expected_feat}"
            )

        if not np.isfinite(tensor).all():
            raise DataLoaderError(f"Tensor {tensor_file.name} contains non-finite numbers (NaN or Inf)")

        # Retrieve sequence metadata records for this length
        seq_records_meta: List[Dict[str, Any]] = []
        by_length = self.metadata_data.get("sequences_by_length", {})
        length_key = f"L{sequence_length}"

        if length_key in by_length:
            seq_records_meta = by_length[length_key]
        else:
            # Fallback to filtering all sequences
            all_seqs = self.metadata_data.get("sequences", [])
            seq_records_meta = [
                s for s in all_seqs if s.get("input_sequence_length") == sequence_length
            ]

        if len(seq_records_meta) != n_seq:
            raise DataLoaderError(
                f"Metadata record count ({len(seq_records_meta)}) does not match "
                f"tensor row count ({n_seq}) for length {sequence_length}"
            )

        # Check chronological ordering and build records
        dataset_id = self.manifest_data.get("dataset_id", "unknown")
        feature_names = self.manifest_data.get("ordered_feature_names", [])
        dataset_records: List[SequenceDatasetRecord] = []
        any_targets_available = False

        for i, meta in enumerate(seq_records_meta):
            seq_id = meta.get("sequence_id", f"seq_{length_key}_{i:06d}")
            split = meta.get("split", "train")
            horizons_meta = meta.get("horizons", {})
            target_labels: Dict[int, Optional[float]] = {}
            rec_target_avail = False

            for h in target_horizons:
                h_key = f"h{h}"
                h_info = horizons_meta.get(h_key, {})
                t_avail = h_info.get("target_label_available", False)
                t_val = h_info.get("target_label")

                if t_avail and t_val is not None:
                    try:
                        target_labels[h] = float(t_val)
                        rec_target_avail = True
                        any_targets_available = True
                    except (ValueError, TypeError):
                        target_labels[h] = None
                else:
                    target_labels[h] = None

            record = SequenceDatasetRecord(
                sequence_id=seq_id,
                dataset_id=dataset_id,
                sequence_length=sequence_length,
                feature_count=n_feat,
                features=tensor[i],
                start_frame_id=meta.get("start_frame_id", ""),
                end_frame_id=meta.get("end_frame_id", ""),
                start_source_frame_index=meta.get("start_source_frame_index", -1),
                end_source_frame_index=meta.get("end_source_frame_index", -1),
                start_timestamp_seconds=float(meta.get("start_timestamp_seconds", 0.0)),
                end_timestamp_seconds=float(meta.get("end_timestamp_seconds", 0.0)),
                split=split,
                horizons=horizons_meta,
                target_labels=target_labels,
                target_labels_available=rec_target_avail,
            )
            dataset_records.append(record)

        return BaselineDataset(
            dataset_id=dataset_id,
            sequence_length=sequence_length,
            feature_count=n_feat,
            ordered_feature_names=feature_names,
            records=dataset_records,
            tensors=tensor,
            target_labels_available=any_targets_available,
        )
