"""Unit and integration tests for BaselineDataLoader and BaselineDataset."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.forecasting.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    DataLoaderError,
    SequenceDatasetRecord,
)


class TestBaselineDataLoader:
    """Tests for data loading, split separation, and sequence validation."""

    @pytest.fixture
    def real_day5_dir(self) -> Path:
        """Path to real Day 5 sequence outputs."""
        p = Path("datasets/processed/driving_sample/features/sequences/run_official")
        if not p.exists():
            pytest.skip("Real Day 5 sequence directory not found")
        return p

    def test_missing_artifacts_directory(self, tmp_path):
        """Missing artifacts directory must raise DataLoaderError."""
        with pytest.raises(DataLoaderError, match="does not exist"):
            BaselineDataLoader(tmp_path / "nonexistent")

    def test_missing_manifest_file(self, tmp_path):
        """Missing manifest file must raise DataLoaderError."""
        d = tmp_path / "seq_dir"
        d.mkdir()
        with pytest.raises(DataLoaderError, match="Required artifact missing"):
            BaselineDataLoader(d)

    def test_missing_metadata_file(self, tmp_path):
        """Missing metadata file must raise DataLoaderError."""
        d = tmp_path / "seq_dir"
        d.mkdir()
        with open(d / "feature_manifest.json", "w") as f:
            json.dump({"total_features": 10}, f)
        with pytest.raises(DataLoaderError, match="Required artifact missing"):
            BaselineDataLoader(d)

    def test_load_real_day5_dataset_l5(self, real_day5_dir):
        """Real Day 5 driving sample L5 must load and pass all validations."""
        loader = BaselineDataLoader(real_day5_dir)
        dataset = loader.load_sequence_dataset(sequence_length=5)

        assert isinstance(dataset, BaselineDataset)
        assert dataset.dataset_id == "driving_sample"
        assert len(dataset) == 138
        assert dataset.sequence_length == 5
        assert dataset.feature_count == 27
        assert dataset.tensors.shape == (138, 5, 27)
        assert dataset.target_labels_available is False

        # First record check
        rec0 = dataset.records[0]
        assert isinstance(rec0, SequenceDatasetRecord)
        assert rec0.sequence_id.startswith("seq_driving_sample_L5_")
        assert rec0.sequence_length == 5
        assert rec0.features.shape == (5, 27)
        assert rec0.split == "train"
        assert 1 in rec0.target_labels
        assert rec0.target_labels[1] is None

    def test_load_real_day5_all_configured_lengths(self, real_day5_dir):
        """Must correctly load L10, L15, L20 from real Day 5 directory."""
        loader = BaselineDataLoader(real_day5_dir)
        ds10 = loader.load_sequence_dataset(sequence_length=10)
        assert len(ds10) == 123
        assert ds10.tensors.shape == (123, 10, 27)

        ds15 = loader.load_sequence_dataset(sequence_length=15)
        assert len(ds15) == 108
        assert ds15.tensors.shape == (108, 15, 27)

        ds20 = loader.load_sequence_dataset(sequence_length=20)
        assert len(ds20) == 93
        assert ds20.tensors.shape == (93, 20, 27)

    def test_dataset_split_filtering(self, real_day5_dir):
        """Splits must partition records disjointly with exact counts."""
        loader = BaselineDataLoader(real_day5_dir)
        dataset = loader.load_sequence_dataset(sequence_length=5)

        train_ds = dataset.filter_by_split("train")
        val_ds = dataset.filter_by_split("val")
        test_ds = dataset.filter_by_split("test")

        assert len(train_ds) == 101
        assert len(val_ds) == 18
        assert len(test_ds) == 19
        assert len(train_ds) + len(val_ds) + len(test_ds) == len(dataset)

        # Verify tensors are sliced accordingly
        assert train_ds.tensors.shape == (101, 5, 27)
        assert val_ds.tensors.shape == (18, 5, 27)
        assert test_ds.tensors.shape == (19, 5, 27)

    def test_get_targets_when_missing(self, real_day5_dir):
        """When targets are absent, get_targets_for_horizon returns all False mask."""
        loader = BaselineDataLoader(real_day5_dir)
        dataset = loader.load_sequence_dataset(sequence_length=5)

        valid_mask, targets = dataset.get_targets_for_horizon(1)
        assert not valid_mask.any()
        assert np.isnan(targets).all()

    @pytest.fixture
    def synthetic_dataset_dir(self, tmp_path) -> Path:
        """Create valid synthetic sequence directory with targets."""
        d = tmp_path / "synth_seq"
        d.mkdir()

        manifest = {
            "dataset_id": "test_ds",
            "total_features": 4,
            "ordered_feature_names": ["f0", "f1", "f2", "f3"],
        }
        with open(d / "feature_manifest.json", "w") as f:
            json.dump(manifest, f)

        # 10 sequences of length 5
        tensor = np.ones((10, 5, 4), dtype=np.float32)
        np.save(d / "sequences_L5.npy", tensor)

        seqs = []
        for i in range(10):
            seqs.append({
                "sequence_id": f"seq_{i}",
                "input_sequence_length": 5,
                "split": "train" if i < 7 else "test",
                "horizons": {
                    "h1": {
                        "horizon": 1,
                        "target_frame_id": f"f_{i+5}",
                        "target_label": 0.80 + i * 0.01,
                        "target_label_available": True,
                    }
                },
            })

        metadata = {
            "total_sequences": 10,
            "sequences": seqs,
            "sequences_by_length": {"L5": seqs},
        }
        with open(d / "sequence_metadata.json", "w") as f:
            json.dump(metadata, f)

        return d

    def test_load_synthetic_dataset_with_targets(self, synthetic_dataset_dir):
        """Synthetic dataset with ground truth must report target_labels_available=True."""
        loader = BaselineDataLoader(synthetic_dataset_dir)
        dataset = loader.load_sequence_dataset(sequence_length=5)

        assert len(dataset) == 10
        assert dataset.target_labels_available is True

        valid_mask, targets = dataset.get_targets_for_horizon(1)
        assert valid_mask.all()
        assert len(targets) == 10
        assert targets[0] == pytest.approx(0.80)

        # Check train split target extraction
        tr_mask, tr_targets = dataset.get_targets_for_horizon(1, split="train")
        assert tr_mask.all()
        assert len(tr_targets) == 7

    def test_tensor_dimension_mismatch(self, synthetic_dataset_dir):
        """2D tensor must be rejected with DataLoaderError."""
        bad_tensor = np.zeros((10, 5), dtype=np.float32)
        np.save(synthetic_dataset_dir / "sequences_L5.npy", bad_tensor)

        loader = BaselineDataLoader(synthetic_dataset_dir)
        with pytest.raises(DataLoaderError, match="Expected 3D tensor"):
            loader.load_sequence_dataset(sequence_length=5)

    def test_tensor_length_mismatch(self, synthetic_dataset_dir):
        """Tensor sequence dimension != requested length must raise DataLoaderError."""
        mismatched = np.zeros((10, 7, 4), dtype=np.float32)
        np.save(synthetic_dataset_dir / "sequences_L5.npy", mismatched)

        loader = BaselineDataLoader(synthetic_dataset_dir)
        with pytest.raises(DataLoaderError, match="sequence length 7 != requested length 5"):
            loader.load_sequence_dataset(sequence_length=5)

    def test_tensor_feature_count_mismatch(self, synthetic_dataset_dir):
        """Tensor feature dimension != manifest must raise DataLoaderError."""
        mismatched = np.zeros((10, 5, 8), dtype=np.float32)
        np.save(synthetic_dataset_dir / "sequences_L5.npy", mismatched)

        loader = BaselineDataLoader(synthetic_dataset_dir)
        with pytest.raises(DataLoaderError, match="feature count 8 does not match manifest count 4"):
            loader.load_sequence_dataset(sequence_length=5)

    def test_tensor_non_finite_values(self, synthetic_dataset_dir):
        """NaN values in tensor must raise DataLoaderError."""
        nan_tensor = np.zeros((10, 5, 4), dtype=np.float32)
        nan_tensor[0, 0, 0] = np.nan
        np.save(synthetic_dataset_dir / "sequences_L5.npy", nan_tensor)

        loader = BaselineDataLoader(synthetic_dataset_dir)
        with pytest.raises(DataLoaderError, match="contains non-finite numbers"):
            loader.load_sequence_dataset(sequence_length=5)
