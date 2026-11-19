"""Unit and integration tests for BaselineExperiment orchestrator and artifact generator."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml

from src.forecasting.baseline_experiment import BaselineExperiment


class TestBaselineExperiment:
    """Tests for baseline experiment execution, split isolation, and artifact generation."""

    @pytest.fixture
    def real_day5_dir(self) -> Path:
        """Path to real Day 5 sequence directory."""
        p = Path("datasets/processed/driving_sample/features/sequences/run_official")
        if not p.exists():
            pytest.skip("Real Day 5 sequence directory not found")
        return p

    def test_experiment_init(self, tmp_path):
        """Experiment must initialize with valid defaults and directories."""
        exp = BaselineExperiment(
            artifacts_dir="datasets/processed/driving_sample/features/sequences/run_official",
            output_dir=tmp_path / "experiments",
            run_id="test_run_01",
        )
        assert exp.run_id == "test_run_01"
        assert exp.run_dir == tmp_path / "experiments" / "test_run_01"
        assert exp.strategies == ["persistence", "mean", "trend"]
        assert exp.horizons == [1, 3, 5]

    def test_run_on_real_day5_artifacts(self, real_day5_dir, tmp_path):
        """Running on real Day 5 data must produce full artifact suite with honest readiness."""
        exp = BaselineExperiment(
            artifacts_dir=real_day5_dir,
            output_dir=tmp_path / "experiments",
            run_id="real_test_run",
        )
        res = exp.run()

        assert res["status"] == "PASS"
        assert res["supervised_readiness"] == "NOT AVAILABLE"
        assert res["targets_available"] is False
        assert res["total_predictions"] == 138 * 3 * 3  # 1242
        assert len(res["warnings"]) > 0

        # Verify all 6 artifacts exist on disk
        run_dir = tmp_path / "experiments" / "real_test_run"
        assert (run_dir / "experiment_config.yaml").exists()
        assert (run_dir / "predictions.json").exists()
        assert (run_dir / "baseline_metrics.json").exists()
        assert (run_dir / "experiment_summary.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "errors.json").exists()

        # Check experiment_config.yaml content
        with open(run_dir / "experiment_config.yaml") as f:
            cfg = yaml.safe_load(f)
            assert cfg["run_id"] == "real_test_run"
            assert cfg["dataset_id"] == "driving_sample"
            assert cfg["forecast_horizons"] == [1, 3, 5]

        # Check experiment_summary.json
        with open(run_dir / "experiment_summary.json") as f:
            summary = json.load(f)
            assert summary["status"] == "PASS"
            assert summary["supervised_readiness"] == "NOT AVAILABLE"
            assert summary["total_sequences_evaluated"] == 138

        # Check metadata.json
        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)
            assert meta["split_counts"]["train"] == 101
            assert meta["split_counts"]["val"] == 18
            assert meta["split_counts"]["test"] == 19

    @pytest.fixture
    def synthetic_dataset_dir(self, tmp_path) -> Path:
        """Create valid synthetic sequence directory with ground truth targets."""
        d = tmp_path / "synth_seq"
        d.mkdir()

        manifest = {
            "dataset_id": "synthetic_driving",
            "total_features": 3,
            "ordered_feature_names": ["f0", "f1", "f2"],
        }
        with open(d / "feature_manifest.json", "w") as f:
            json.dump(manifest, f)

        # 9 sequences of length 5
        tensor = np.ones((9, 5, 3), dtype=np.float32)
        np.save(d / "sequences_L5.npy", tensor)

        seqs = []
        for i in range(9):
            # splits: 0-4 train (5), 5-6 val (2), 7-8 test (2)
            sp = "train" if i < 5 else ("val" if i < 7 else "test")
            seqs.append({
                "sequence_id": f"seq_{i}",
                "input_sequence_length": 5,
                "split": sp,
                "horizons": {
                    "h1": {
                        "horizon": 1,
                        "target_frame_id": f"frame_{i+5}",
                        "target_label": 0.70 + 0.02 * i,
                        "target_label_available": True,
                    },
                    "h3": {
                        "horizon": 3,
                        "target_frame_id": f"frame_{i+7}",
                        "target_label": 0.65 + 0.02 * i,
                        "target_label_available": True,
                    },
                    "h5": {
                        "horizon": 5,
                        "target_frame_id": f"frame_{i+9}",
                        "target_label": 0.60 + 0.02 * i,
                        "target_label_available": True,
                    },
                },
            })

        metadata = {
            "total_sequences": 9,
            "sequences": seqs,
            "sequences_by_length": {"L5": seqs},
        }
        with open(d / "sequence_metadata.json", "w") as f:
            json.dump(metadata, f)

        return d

    def test_run_on_synthetic_data_with_ground_truth(self, synthetic_dataset_dir, tmp_path):
        """Running on dataset with ground truth must compute valid metrics."""
        hist_map = {f"seq_{i}": [0.70, 0.72, 0.74, 0.76, 0.78] for i in range(9)}

        exp = BaselineExperiment(
            artifacts_dir=synthetic_dataset_dir,
            output_dir=tmp_path / "experiments",
            run_id="synth_run_01",
            strategies=["persistence", "mean", "trend"],
            horizons=[1, 3, 5],
        )
        res = exp.run(historical_targets_map=hist_map)

        assert res["status"] == "PASS"
        assert res["supervised_readiness"] == "PASS"
        assert res["targets_available"] is True
        assert res["total_predictions"] == 9 * 3 * 3  # 81

        # Check metrics file
        metrics_file = tmp_path / "experiments" / "synth_run_01" / "baseline_metrics.json"
        assert metrics_file.exists()
        with open(metrics_file) as f:
            metrics_data = json.load(f)

        overall = metrics_data["overall"]["strategies"]
        # Persistence h1
        pers_h1 = overall["persistence"]["h1"]
        assert pers_h1["status"] == "PASS"
        assert pers_h1["mae"] is not None
        assert pers_h1["rmse"] is not None
        assert pers_h1["valid_pairs"] == 9

        # Mean h1
        mean_h1 = overall["mean"]["h1"]
        assert mean_h1["status"] == "PASS"
        assert mean_h1["mae"] is not None

        # Trend h1
        trend_h1 = overall["trend"]["h1"]
        assert trend_h1["status"] == "PASS"
        assert trend_h1["mae"] is not None

    def test_cli_execution_experiment(self, real_day5_dir, tmp_path):
        """CLI invocation of baseline_experiment must exit 0 and write outputs."""
        out_dir = tmp_path / "cli_exp"
        cmd = [
            sys.executable,
            "-m",
            "src.forecasting.baseline_experiment",
            "--input",
            str(real_day5_dir),
            "--output",
            str(out_dir),
            "--sequence-length",
            "5",
            "--horizons",
            "1",
            "3",
            "5",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Baseline Forecasting Experiment Complete" in res.stdout
        assert "Status                 : PASS" in res.stdout
        assert "Supervised Readiness   : NOT AVAILABLE" in res.stdout
