"""Unit and integration tests for ForecastingDataContract and Model Specs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.forecasting_data_contract import (
    DataContractError,
    DataContractValidationResult,
    ForecastingDataContract,
    ModelInputSpec,
    ModelOutputSpec,
)


class TestForecastingDataContract:
    """Tests for Day 5 sequence tensor and metadata contract validation."""

    @pytest.fixture
    def real_day5_dir(self) -> Path:
        """Path to real Day 5 sequence outputs."""
        p = Path("datasets/processed/driving_sample/features/sequences/run_official")
        if not p.exists():
            pytest.skip("Real Day 5 artifacts not found on disk")
        return p

    def test_model_specs(self):
        """ModelInputSpec and ModelOutputSpec must define correct shapes and types."""
        contract = ForecastingDataContract()
        inp = contract.input_spec
        out = contract.output_spec

        assert inp.expected_ndim == 3
        assert inp.dtype == "torch.float32"
        assert inp.supported_sequence_lengths == [5, 10, 15, 20]

        assert out.expected_ndim == 2
        assert out.dtype == "torch.float32"
        assert out.target_type == "continuous_reliability_score"
        assert out.target_range == (0.0, 1.0)
        assert out.horizons == [1, 3, 5]

    def test_validate_real_day5_artifacts(self, real_day5_dir):
        """Real Day 5 artifacts must pass all structural data contract checks."""
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(real_day5_dir, raise_on_error=True)

        assert res.passed is True
        assert res.dataset_id == "driving_sample"
        assert res.feature_count == 27
        assert len(res.ordered_feature_names) == 27
        assert res.sequence_lengths_validated == [5, 10, 15, 20]

        # Verify tensor shapes
        assert res.tensor_shapes["L5"] == [138, 5, 27]
        assert res.tensor_shapes["L10"] == [123, 10, 27]
        assert res.tensor_shapes["L15"] == [108, 15, 27]
        assert res.tensor_shapes["L20"] == [93, 20, 27]

        # Verify health checks
        assert res.finite_check_passed is True
        assert res.splits_validated is True
        assert res.leakage_check_passed is True
        assert res.chronological_check_passed is True

        # Verify ground truth honesty
        assert res.ground_truth_labels_available is False
        assert res.supervised_readiness == "NOT AVAILABLE"
        assert len(res.warnings) > 0
        assert len(res.errors) == 0

    def test_missing_artifacts_dir(self, tmp_path):
        """Nonexistent artifacts directory must report failure without unhandled crash."""
        missing = tmp_path / "does_not_exist"
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(missing, raise_on_error=False)
        assert res.passed is False
        assert any("does not exist" in e for e in res.errors)

    def test_raise_on_error_flag(self, tmp_path):
        """Setting raise_on_error=True must raise DataContractError on failure."""
        missing = tmp_path / "does_not_exist"
        contract = ForecastingDataContract()
        with pytest.raises(DataContractError):
            contract.validate_artifacts(missing, raise_on_error=True)

    @pytest.fixture
    def synthetic_artifacts_dir(self, tmp_path) -> Path:
        """Helper creating valid synthetic sequence directory."""
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)

        feature_names = [f"f_{i}" for i in range(10)]
        manifest = {
            "schema_version": "1.0",
            "dataset_id": "synth_ds",
            "total_features": 10,
            "ordered_feature_names": feature_names,
        }
        with open(seq_dir / "feature_manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create tensors for lengths 5, 10, 15, 20
        lengths = [5, 10, 15, 20]
        n_seqs = {5: 20, 10: 18, 15: 16, 20: 14}
        split_counts = {}
        for l in lengths:
            n = n_seqs[l]
            arr = np.random.randn(n, l, 10).astype(np.float32)
            np.save(seq_dir / f"sequences_L{l}.npy", arr)
            split_counts[f"L{l}"] = {
                "train": n - 4,
                "val": 2,
                "test": 2,
                "total": n,
            }

        splits = {
            "schema_version": "1.0",
            "dataset_id": "synth_ds",
            "leakage_check_passed": True,
            "sequence_counts_per_split": split_counts,
        }
        with open(seq_dir / "dataset_splits.json", "w") as f:
            json.dump(splits, f)

        metadata = {
            "schema_version": "1.0",
            "dataset_id": "synth_ds",
            "total_sequences": sum(n_seqs.values()),
            "sequences": [
                {
                    "sequence_id": f"seq_{i}",
                    "start_timestamp_seconds": float(i) * 0.1,
                    "start_source_frame_index": i,
                    "horizons": {
                        "h1": {"horizon": 1, "target_label": None, "target_label_available": False}
                    },
                }
                for i in range(20)
            ],
        }
        with open(seq_dir / "sequence_metadata.json", "w") as f:
            json.dump(metadata, f)

        return seq_dir

    def test_synthetic_artifacts_pass(self, synthetic_artifacts_dir):
        """Synthetic directory meeting all contract invariants must pass."""
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=True)
        assert res.passed is True
        assert res.feature_count == 10
        assert res.finite_check_passed is True
        assert res.supervised_readiness == "NOT AVAILABLE"

    def test_tensor_dimension_invalid(self, synthetic_artifacts_dir):
        """Tensors that are 2D or 4D must fail validation."""
        bad_tensor = np.zeros((10, 20), dtype=np.float32)
        np.save(synthetic_artifacts_dir / "sequences_L5.npy", bad_tensor)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("invalid dimensionality" in e for e in res.errors)

    def test_tensor_sequence_length_mismatch(self, synthetic_artifacts_dir):
        """Tensor sequence dimension not matching filename length must fail."""
        mismatched = np.zeros((10, 7, 10), dtype=np.float32)
        np.save(synthetic_artifacts_dir / "sequences_L5.npy", mismatched)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("sequence length dimension 7 != expected 5" in e for e in res.errors)

    def test_tensor_feature_count_mismatch(self, synthetic_artifacts_dir):
        """Tensor feature dimension not matching manifest must fail."""
        mismatched = np.zeros((10, 5, 12), dtype=np.float32)
        np.save(synthetic_artifacts_dir / "sequences_L5.npy", mismatched)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("feature count 12 does not match manifest 10" in e for e in res.errors)

    def test_tensor_non_floating_dtype(self, synthetic_artifacts_dir):
        """Non-floating point tensors (e.g. int32) must fail."""
        int_tensor = np.ones((10, 5, 10), dtype=np.int32)
        np.save(synthetic_artifacts_dir / "sequences_L5.npy", int_tensor)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("non-floating dtype" in e for e in res.errors)

    def test_tensor_nan_detection(self, synthetic_artifacts_dir):
        """Tensors containing NaNs must fail finite validation."""
        nan_tensor = np.zeros((10, 5, 10), dtype=np.float32)
        nan_tensor[2, 1, 3] = np.nan
        np.save(synthetic_artifacts_dir / "sequences_L5.npy", nan_tensor)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert res.finite_check_passed is False
        assert any("contains non-finite values" in e for e in res.errors)

    def test_tensor_inf_detection(self, synthetic_artifacts_dir):
        """Tensors containing Infs must fail finite validation."""
        inf_tensor = np.zeros((10, 5, 10), dtype=np.float32)
        inf_tensor[0, 0, 0] = np.inf
        np.save(synthetic_artifacts_dir / "sequences_L5.npy", inf_tensor)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert res.finite_check_passed is False
        assert any("contains non-finite values" in e for e in res.errors)

    def test_missing_manifest(self, synthetic_artifacts_dir):
        """Missing feature manifest must fail contract."""
        (synthetic_artifacts_dir / "feature_manifest.json").unlink()
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("Missing required artifact: feature_manifest.json" in e for e in res.errors)

    def test_manifest_feature_count_mismatch(self, synthetic_artifacts_dir):
        """total_features != len(ordered_feature_names) in manifest must fail."""
        with open(synthetic_artifacts_dir / "feature_manifest.json", "w") as f:
            json.dump({
                "schema_version": "1.0",
                "dataset_id": "synth_ds",
                "total_features": 99,
                "ordered_feature_names": ["f1", "f2"],
            }, f)
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("does not match ordered_feature_names length" in e for e in res.errors)

    def test_missing_sequence_metadata(self, synthetic_artifacts_dir):
        """Missing sequence_metadata.json must fail contract."""
        (synthetic_artifacts_dir / "sequence_metadata.json").unlink()
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("Missing required artifact: sequence_metadata.json" in e for e in res.errors)

    def test_missing_dataset_splits(self, synthetic_artifacts_dir):
        """Missing dataset_splits.json must fail contract."""
        (synthetic_artifacts_dir / "dataset_splits.json").unlink()
        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("Missing required artifact: dataset_splits.json" in e for e in res.errors)

    def test_splits_count_mismatch(self, synthetic_artifacts_dir):
        """Split total count not matching tensor count must fail."""
        splits_path = synthetic_artifacts_dir / "dataset_splits.json"
        with open(splits_path, "r") as f:
            splits_data = json.load(f)
        splits_data["sequence_counts_per_split"]["L5"]["total"] = 999
        with open(splits_path, "w") as f:
            json.dump(splits_data, f)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=False)
        assert res.passed is False
        assert any("does not match tensor sequence count" in e for e in res.errors)

    def test_ground_truth_available_simulation(self, synthetic_artifacts_dir):
        """When valid target labels are supplied, supervised_readiness must report PASS."""
        meta_path = synthetic_artifacts_dir / "sequence_metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["sequences"][0]["horizons"]["h1"]["target_label_available"] = True
        meta["sequences"][0]["horizons"]["h1"]["target_label"] = 0.85
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        contract = ForecastingDataContract()
        res = contract.validate_artifacts(synthetic_artifacts_dir, raise_on_error=True)
        assert res.passed is True
        assert res.ground_truth_labels_available is True
        assert res.supervised_readiness == "PASS"

    def test_cli_execution_valid(self, real_day5_dir):
        """CLI execution on real Day 5 directory returns 0 and prints contract."""
        cmd = [
            sys.executable,
            "-m",
            "src.forecasting.forecasting_data_contract",
            "--artifacts-dir",
            str(real_day5_dir),
            "--validate",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "OVERALL DATA CONTRACT: [PASS]" in res.stdout
        assert "Supervised Training Ready  : NOT AVAILABLE" in res.stdout

    def test_cli_execution_missing_dir(self, tmp_path):
        """CLI execution on missing directory returns 1."""
        missing = tmp_path / "does_not_exist"
        cmd = [
            sys.executable,
            "-m",
            "src.forecasting.forecasting_data_contract",
            "--artifacts-dir",
            str(missing),
            "--validate",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "[FAIL]" in res.stdout
