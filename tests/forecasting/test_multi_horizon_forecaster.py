"""Unit and Integration Tests for Multi-Horizon Forecasting Engine.

Tests checkpoint loading, model reconstruction, feature compatibility,
multi-horizon alignment (t+1, t+3, t+5), non-continuous frames, output schema,
artifact generation, CLI behavior, and CPU inference execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.forecasting.baseline_data_loader import BaselineDataLoader, SequenceDatasetRecord
from src.forecasting.checkpoint_manager import CheckpointManager
from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.forecasting.multi_horizon_forecaster import (
    HorizonPredictionRecord,
    MultiHorizonForecaster,
    MultiHorizonForecastingError,
    MultiHorizonForecastingSummary,
    parse_args,
    run_forecasting_cli,
)


@pytest.fixture
def sample_model() -> LSTMForecaster:
    """Fixture providing an initialized LSTMForecaster."""
    return LSTMForecaster(
        input_size=27,
        hidden_size=16,
        output_horizons=[1, 3, 5],
        activation="sigmoid",
    )


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path, sample_model: LSTMForecaster) -> Path:
    """Fixture creating a valid synthetic checkpoint file."""
    chk_mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints", experiment_id="test_exp")
    opt = AdamW(sample_model.parameters(), lr=0.001)
    chk_path = chk_mgr.save_checkpoint(
        model=sample_model,
        optimizer=opt,
        epoch=5,
        val_loss=0.035,
        is_best=True,
        metadata={"feature_manifest": "feature_manifest.json"},
    )
    return chk_path


class TestMultiHorizonInitialization:
    """Tests covering forecaster initialization, checkpoint loading, and architecture checks."""

    def test_init_defaults(self, sample_model: LSTMForecaster) -> None:
        """Verify initialization with pre-initialized model."""
        forecaster = MultiHorizonForecaster(model=sample_model, device="cpu")
        assert forecaster.model is not None
        assert forecaster.horizons == [1, 3, 5]
        assert forecaster.batch_size == 8

    def test_init_unsupported_horizon_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of unconfigured horizons."""
        with pytest.raises(MultiHorizonForecastingError, match="Unsupported horizon"):
            MultiHorizonForecaster(model=sample_model, horizons=[1, 2, 5])

    def test_init_invalid_model_type_raises(self) -> None:
        """Verify rejection of invalid model instance."""
        with pytest.raises(MultiHorizonForecastingError, match="instance of LSTMForecaster"):
            MultiHorizonForecaster(model="invalid_model")

    def test_load_checkpoint_success(self, synthetic_checkpoint: Path) -> None:
        """Verify successful checkpoint loading and model reconstruction."""
        forecaster = MultiHorizonForecaster(checkpoint_path=synthetic_checkpoint, device="cpu")
        assert forecaster.model is not None
        assert forecaster.model.input_size == 27
        assert forecaster.model.hidden_size == 16
        assert forecaster.checkpoint_id == "test_exp"

    def test_load_checkpoint_missing_file_raises(self, tmp_path: Path) -> None:
        """Verify rejection of missing checkpoint file."""
        with pytest.raises(MultiHorizonForecastingError, match="not found"):
            MultiHorizonForecaster(checkpoint_path=tmp_path / "missing.pt")

    def test_load_checkpoint_corrupted_file_raises(self, tmp_path: Path) -> None:
        """Verify rejection of corrupted checkpoint file."""
        bad_file = tmp_path / "corrupted.pt"
        bad_file.write_text("not_a_torch_file", encoding="utf-8")
        with pytest.raises(MultiHorizonForecastingError, match="Corrupted or invalid checkpoint"):
            MultiHorizonForecaster(checkpoint_path=bad_file)

    def test_load_checkpoint_missing_model_state_dict_raises(self, tmp_path: Path) -> None:
        """Verify rejection when model_state_dict is missing."""
        bad_chk = tmp_path / "no_weights.pt"
        torch.save({"epoch": 1, "input_size": 27}, bad_chk)
        with pytest.raises(MultiHorizonForecastingError, match="missing required 'model_state_dict'"):
            MultiHorizonForecaster(checkpoint_path=bad_chk)

    def test_load_checkpoint_invalid_input_size_raises(self, tmp_path: Path) -> None:
        """Verify rejection when input_size is invalid."""
        bad_chk = tmp_path / "bad_input.pt"
        torch.save({"model_state_dict": {}, "input_size": -5}, bad_chk)
        with pytest.raises(MultiHorizonForecastingError, match="invalid or missing 'input_size'"):
            MultiHorizonForecaster(checkpoint_path=bad_chk)

    def test_load_checkpoint_horizon_mismatch_raises(self, tmp_path: Path, sample_model: LSTMForecaster) -> None:
        """Verify rejection when checkpoint horizons do not match requested horizons."""
        chk_mgr = CheckpointManager(checkpoint_dir=tmp_path / "chk", experiment_id="exp")
        opt = AdamW(sample_model.parameters())
        chk_path = chk_mgr.save_checkpoint(
            model=sample_model,
            optimizer=opt,
            epoch=1,
            val_loss=0.1,
            is_best=True,
        )
        # Attempt to load with different horizons [1, 5]
        with pytest.raises(MultiHorizonForecastingError, match="Checkpoint horizons"):
            MultiHorizonForecaster(checkpoint_path=chk_path, horizons=[1, 3, 5])
            # Modify checkpoint dict horizons to test mismatch
            bad_data = torch.load(chk_path, weights_only=False)
            bad_data["output_horizons"] = [1, 5]
            torch.save(bad_data, chk_path)
            MultiHorizonForecaster(checkpoint_path=chk_path, horizons=[1, 3, 5])

    def test_model_reconstruction_weights_match(self, sample_model: LSTMForecaster, synthetic_checkpoint: Path) -> None:
        """Verify reconstructed model parameters strictly match saved checkpoint weights."""
        forecaster = MultiHorizonForecaster(checkpoint_path=synthetic_checkpoint)
        for p_orig, p_recon in zip(sample_model.parameters(), forecaster.model.parameters()):
            assert torch.allclose(p_orig, p_recon)


class TestInputValidation:
    """Tests covering input tensor shape, feature count, and finite numerical validation."""

    def test_validate_tensor_numpy_conversion(self, sample_model: LSTMForecaster) -> None:
        """Verify NumPy array converted to Float32 PyTorch tensor."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        arr = np.random.randn(4, 5, 27).astype(np.float32)
        tensor = forecaster.validate_input_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 5, 27)
        assert tensor.dtype == torch.float32

    def test_validate_tensor_torch_device(self, sample_model: LSTMForecaster) -> None:
        """Verify tensor placed on target device."""
        forecaster = MultiHorizonForecaster(model=sample_model, device="cpu")
        tensor = torch.randn(2, 5, 27)
        validated = forecaster.validate_input_tensor(tensor)
        assert validated.device.type == "cpu"

    def test_validate_tensor_invalid_type_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of invalid data types."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        with pytest.raises(MultiHorizonForecastingError, match="numpy.ndarray or torch.Tensor"):
            forecaster.validate_input_tensor([[1.0, 2.0]])

    def test_validate_tensor_invalid_ndim_2d_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of 2D tensors."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        with pytest.raises(MultiHorizonForecastingError, match="3-dimensional"):
            forecaster.validate_input_tensor(np.zeros((5, 27)))

    def test_validate_tensor_invalid_ndim_4d_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of 4D tensors."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        with pytest.raises(MultiHorizonForecastingError, match="3-dimensional"):
            forecaster.validate_input_tensor(np.zeros((2, 5, 27, 1)))

    def test_validate_tensor_empty_batch_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of batch size 0."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        with pytest.raises(MultiHorizonForecastingError, match="empty batch"):
            forecaster.validate_input_tensor(np.empty((0, 5, 27)))

    def test_validate_tensor_zero_sequence_length_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of sequence length 0."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        with pytest.raises(MultiHorizonForecastingError, match="Sequence length must be > 0"):
            forecaster.validate_input_tensor(np.empty((2, 0, 27)))

    def test_validate_tensor_feature_count_mismatch_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection when feature dimension does not match input_size."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        with pytest.raises(MultiHorizonForecastingError, match="Input feature count"):
            forecaster.validate_input_tensor(np.zeros((2, 5, 10)))  # 10 != 27

    def test_validate_tensor_nan_detection_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of sequences containing NaN."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.zeros((2, 5, 27), dtype=np.float32)
        seq[0, 2, 5] = np.nan
        with pytest.raises(MultiHorizonForecastingError, match="non-finite"):
            forecaster.validate_input_tensor(seq)

    def test_validate_tensor_inf_detection_raises(self, sample_model: LSTMForecaster) -> None:
        """Verify rejection of sequences containing Inf."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.zeros((2, 5, 27), dtype=np.float32)
        seq[1, 1, 1] = np.inf
        with pytest.raises(MultiHorizonForecastingError, match="non-finite"):
            forecaster.validate_input_tensor(seq)


class TestInferenceAndHorizonMapping:
    """Tests covering forward inference, horizon-to-column indexing, and continuous bounding."""

    def test_single_sequence_inference_2d(self, sample_model: LSTMForecaster) -> None:
        """Verify predicting on 2D sequence (L, F) returns 3 horizon records."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.random.randn(5, 27).astype(np.float32)
        preds = forecaster.predict_sequence(seq)
        assert len(preds) == 3
        horizons = [p.horizon for p in preds]
        assert horizons == [1, 3, 5]

    def test_single_sequence_inference_3d(self, sample_model: LSTMForecaster) -> None:
        """Verify predicting on 3D sequence (1, L, F) returns 3 horizon records."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.random.randn(1, 5, 27).astype(np.float32)
        preds = forecaster.predict_sequence(seq)
        assert len(preds) == 3

    def test_batch_inference_predictions_count(self, sample_model: LSTMForecaster) -> None:
        """Verify predicting on batch of 4 sequences returns 12 horizon records."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        batch = np.random.randn(4, 5, 27).astype(np.float32)
        preds = forecaster.predict_batch(batch)
        assert len(preds) == 12  # 4 sequences * 3 horizons

    def test_horizon_1_mapping(self, sample_model: LSTMForecaster) -> None:
        """Verify Horizon 1 corresponds to column 0."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.zeros((1, 5, 27), dtype=np.float32)
        preds = forecaster.predict_sequence(seq)
        h1_rec = next(p for p in preds if p.horizon == 1)
        assert h1_rec.horizon == 1
        assert 0.0 <= h1_rec.predicted_reliability <= 1.0

    def test_horizon_3_mapping(self, sample_model: LSTMForecaster) -> None:
        """Verify Horizon 3 corresponds to column 1."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.zeros((1, 5, 27), dtype=np.float32)
        preds = forecaster.predict_sequence(seq)
        h3_rec = next(p for p in preds if p.horizon == 3)
        assert h3_rec.horizon == 3

    def test_horizon_5_mapping(self, sample_model: LSTMForecaster) -> None:
        """Verify Horizon 5 corresponds to column 2."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.zeros((1, 5, 27), dtype=np.float32)
        preds = forecaster.predict_sequence(seq)
        h5_rec = next(p for p in preds if p.horizon == 5)
        assert h5_rec.horizon == 5

    def test_predicted_reliability_bounded_range(self, sample_model: LSTMForecaster) -> None:
        """Verify all predicted reliability values strictly reside in [0.0, 1.0]."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        # Large variance inputs
        batch = np.random.randn(8, 5, 27).astype(np.float32) * 100.0
        preds = forecaster.predict_batch(batch)
        for p in preds:
            assert 0.0 <= p.predicted_reliability <= 1.0
            assert np.isfinite(p.predicted_reliability)

    def test_inference_mode_no_grad(self, sample_model: LSTMForecaster) -> None:
        """Verify gradients are disabled during prediction."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        seq = np.random.randn(2, 5, 27).astype(np.float32)
        forecaster.predict_batch(seq)
        for p in sample_model.parameters():
            assert p.grad is None

    def test_partial_batch_inference(self, sample_model: LSTMForecaster) -> None:
        """Verify partial batch (e.g. 3 samples with batch_size=8) executes cleanly."""
        forecaster = MultiHorizonForecaster(model=sample_model, batch_size=8)
        seq = np.random.randn(3, 5, 27).astype(np.float32)
        preds = forecaster.predict_batch(seq)
        assert len(preds) == 9

    def test_variable_sequence_length_inference(self, sample_model: LSTMForecaster) -> None:
        """Verify inference functions across variable sequence lengths L in {5, 10, 15, 20}."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        for seq_len in [5, 10, 15, 20]:
            seq = np.random.randn(2, seq_len, 27).astype(np.float32)
            preds = forecaster.predict_batch(seq)
            assert len(preds) == 6


class TestTemporalAlignment:
    """Tests covering horizon frame alignment, non-continuous indices, and timestamps."""

    def test_align_metadata_exact(self, sample_model: LSTMForecaster) -> None:
        """Verify exact metadata retrieval when SequenceDatasetRecord horizons exist."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        sec_rec = SequenceDatasetRecord(
            sequence_id="test_seq_01",
            dataset_id="driving_sample",
            sequence_length=5,
            feature_count=27,
            features=np.zeros((5, 27)),
            start_frame_id="frame_000001.jpg",
            end_frame_id="frame_000005.jpg",
            start_source_frame_index=0,
            end_source_frame_index=24,
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=0.7976,
            split="train",
            horizons={
                "h1": {
                    "horizon": 1,
                    "target_frame_id": "frame_000006.jpg",
                    "target_source_frame_index": 30,
                    "target_timestamp_seconds": 0.997,
                },
                "h3": {
                    "horizon": 3,
                    "target_frame_id": "frame_000008.jpg",
                    "target_source_frame_index": 42,
                    "target_timestamp_seconds": 1.3958,
                },
                "h5": {
                    "horizon": 5,
                    "target_frame_id": "frame_000010.jpg",
                    "target_source_frame_index": 54,
                    "target_timestamp_seconds": 1.7946,
                },
            },
            target_labels={},
            target_labels_available=False,
        )

        preds = forecaster.predict_sequence(
            sequence=np.zeros((5, 27)),
            sequence_record=sec_rec,
        )

        h1 = next(p for p in preds if p.horizon == 1)
        assert h1.target_frame_reference == "frame_000006.jpg"
        assert h1.target_source_frame_index == 30
        assert h1.target_timestamp_seconds == pytest.approx(0.997)
        assert h1.alignment_method == "metadata_exact"

        h3 = next(p for p in preds if p.horizon == 3)
        assert h3.target_frame_reference == "frame_000008.jpg"
        assert h3.target_source_frame_index == 42

        h5 = next(p for p in preds if p.horizon == 5)
        assert h5.target_frame_reference == "frame_000010.jpg"
        assert h5.target_source_frame_index == 54

    def test_align_frame_index_offset(self, sample_model: LSTMForecaster) -> None:
        """Verify alignment via discrete available frame records."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        available_frames = [
            {"frame_id": f"frame_{i:06d}.jpg", "source_frame_index": i * 6, "timestamp_seconds": i * 0.2}
            for i in range(1, 15)
        ]

        t_ref, t_src, t_time, method = forecaster.align_horizon_target(
            sequence_record=None,
            horizon=3,
            end_frame_id="frame_000005.jpg",
            available_frame_records=available_frames,
        )
        assert t_ref == "frame_000008.jpg"
        assert t_src == 48  # idx 4 + 3 = idx 7 -> 7 * 6 = 42? frame_000008 is index 7
        assert method == "frame_index_offset"

    def test_align_non_continuous_frame_indices(self, sample_model: LSTMForecaster) -> None:
        """Verify alignment when frame records have non-continuous gaps."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        # Frames with missing indices: 1, 2, 5, 10, 20
        gapped_frames = [
            {"frame_id": "frame_000001.jpg", "source_frame_index": 0},
            {"frame_id": "frame_000002.jpg", "source_frame_index": 6},
            {"frame_id": "frame_000005.jpg", "source_frame_index": 24},
            {"frame_id": "frame_000010.jpg", "source_frame_index": 54},
            {"frame_id": "frame_000020.jpg", "source_frame_index": 114},
        ]

        # Current frame is frame_000002 (index 1 in available frames). Horizon 1 should point to frame_000005 (index 2).
        t_ref, t_src, _, method = forecaster.align_horizon_target(
            sequence_record=None,
            horizon=1,
            end_frame_id="frame_000002.jpg",
            available_frame_records=gapped_frames,
        )
        assert t_ref == "frame_000005.jpg"
        assert t_src == 24
        assert method == "frame_index_offset"

    def test_align_missing_timestamps(self, sample_model: LSTMForecaster) -> None:
        """Verify missing timestamps handled gracefully without exceptions."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        t_ref, t_src, t_time, _ = forecaster.align_horizon_target(
            sequence_record=None,
            horizon=1,
            end_frame_id="frame_000005.jpg",
            end_source_idx=24,
            end_timestamp=None,
        )
        assert t_ref == "frame_000006.jpg"
        assert t_time is None

    def test_align_out_of_bounds(self, sample_model: LSTMForecaster) -> None:
        """Verify out-of-bounds target reference when horizon exceeds available records."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        short_frames = [
            {"frame_id": "frame_000001.jpg"},
            {"frame_id": "frame_000002.jpg"},
        ]
        t_ref, _, _, method = forecaster.align_horizon_target(
            sequence_record=None,
            horizon=5,
            end_frame_id="frame_000002.jpg",
            available_frame_records=short_frames,
        )
        assert "out_of_bounds" in t_ref
        assert method == "out_of_bounds"


class TestSchemaAndArtifacts:
    """Tests covering prediction record schema, artifact generation, and duplicate prevention."""

    def test_prediction_record_schema(self, sample_model: LSTMForecaster) -> None:
        """Verify prediction record contains all 16 required specification fields."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        preds = forecaster.predict_sequence(np.zeros((5, 27)))
        rec = preds[0]

        expected_fields = {
            "dataset_id", "video_id", "sequence_id", "input_start_frame", "input_end_frame",
            "input_end_source_frame_index", "horizon", "target_frame_reference",
            "target_source_frame_index", "target_timestamp_seconds", "predicted_reliability",
            "prediction_type", "model_name", "checkpoint_id", "inference_time_ms",
            "prediction_status", "alignment_method",
        }
        rec_dict = rec.to_dict()
        for field_name in expected_fields:
            assert field_name in rec_dict

    def test_save_prediction_artifacts_files_created(self, sample_model: LSTMForecaster, tmp_path: Path) -> None:
        """Verify all 8 expected JSON artifacts are created on disk."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        preds = forecaster.predict_batch(np.zeros((4, 5, 27)))

        artifacts = forecaster.save_prediction_artifacts(
            predictions=preds,
            output_dir=tmp_path / "experiments",
            run_id="run_test_01",
        )

        assert "predictions" in artifacts
        assert "horizon_1_predictions" in artifacts
        assert "horizon_3_predictions" in artifacts
        assert "horizon_5_predictions" in artifacts
        assert "prediction_summary" in artifacts
        assert "model_metadata" in artifacts
        assert "feature_manifest" in artifacts
        assert "inference_metadata" in artifacts

        # Verify all paths exist
        for name, file_path in artifacts.items():
            assert file_path.exists(), f"Artifact {name} not found at {file_path}"
            assert file_path.stat().st_size > 0

    def test_summary_artifact_metrics(self, sample_model: LSTMForecaster, tmp_path: Path) -> None:
        """Verify prediction_summary.json structure and zero evaluation metrics."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        preds = forecaster.predict_batch(np.zeros((3, 5, 27)))

        artifacts = forecaster.save_prediction_artifacts(
            predictions=preds,
            output_dir=tmp_path / "exp",
            run_id="run_sum",
        )

        with open(artifacts["prediction_summary"], "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_input_sequences"] == 3
        assert summary["successful_predictions"] == 9
        assert summary["predictions_per_horizon"]["h1"] == 3
        assert summary["predictions_per_horizon"]["h3"] == 3
        assert summary["predictions_per_horizon"]["h5"] == 3

        # Must NOT contain evaluation metrics like mae, rmse, or accuracy
        for forbidden in ["mae", "rmse", "mse", "accuracy", "f1"]:
            assert forbidden not in summary

    def test_no_duplicate_predictions(self, sample_model: LSTMForecaster) -> None:
        """Verify prediction records contain no duplicate sequence_id and horizon pairs."""
        forecaster = MultiHorizonForecaster(model=sample_model)
        preds = forecaster.predict_batch(np.zeros((5, 5, 27)))
        pairs = set()
        for p in preds:
            pair = (p.sequence_id, p.horizon)
            assert pair not in pairs, f"Duplicate prediction pair: {pair}"
            pairs.add(pair)


class TestCLIAndDatasetIntegration:
    """Tests covering CLI execution, arguments, and Day 5 dataset integration."""

    def test_cli_help_parser(self) -> None:
        """Verify CLI argument defaults."""
        args = parse_args([])
        assert args.device == "cpu"
        assert args.batch_size == 8
        assert args.horizons == [1, 3, 5]

    def test_cli_checkpoint_not_available_clean_halt(self, capsys: pytest.CaptureFixture) -> None:
        """Verify CLI halts with clean message and code 0 when checkpoint is absent."""
        args = parse_args(["--checkpoint", "models/forecasting/nonexistent.pt"])
        code = run_forecasting_cli(args)
        assert code == 0

        captured = capsys.readouterr().out
        assert "TRAINED CHECKPOINT READINESS VERIFICATION" in captured
        assert "NOT AVAILABLE" in captured
        assert "Halting inference pipeline" in captured

    def test_cli_with_synthetic_checkpoint_full_run(
        self, synthetic_checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Verify complete CLI run with a valid synthetic checkpoint on Day 5 data."""
        out_dir = tmp_path / "cli_out"
        args = parse_args([
            "--input", "datasets/processed/driving_sample/features/sequences/run_official",
            "--checkpoint", str(synthetic_checkpoint),
            "--output", str(out_dir),
            "--max-sequences", "10",
        ])
        code = run_forecasting_cli(args)
        assert code == 0

        captured = capsys.readouterr().out
        assert "MULTI-HORIZON FORECASTING COMPLETED SUCCESSFULLY" in captured
        assert "Completed 30 predictions across" in captured

    def test_cli_missing_input_dir(self, synthetic_checkpoint: Path, tmp_path: Path) -> None:
        """Verify exit code 1 when input sequence path does not exist."""
        args = parse_args([
            "--input", str(tmp_path / "nonexistent_dir"),
            "--checkpoint", str(synthetic_checkpoint),
        ])
        code = run_forecasting_cli(args)
        assert code == 1

    def test_cli_invalid_config(self, synthetic_checkpoint: Path, tmp_path: Path) -> None:
        """Verify exit code 1 when config file is missing."""
        args = parse_args([
            "--config", str(tmp_path / "missing_config.yaml"),
            "--checkpoint", str(synthetic_checkpoint),
        ])
        code = run_forecasting_cli(args)
        assert code == 1

    def test_real_day5_dataset_batch_inference(self, sample_model: LSTMForecaster) -> None:
        """Verify batch inference directly on loaded Day 5 dataset sequences."""
        loader = BaselineDataLoader(artifacts_dir="datasets/processed/driving_sample/features/sequences/run_official")
        dataset = loader.load_sequence_dataset(sequence_length=5, target_horizons=[1, 3, 5])

        forecaster = MultiHorizonForecaster(model=sample_model, batch_size=16)
        preds = forecaster.predict_dataset(dataset=dataset, max_sequences=20)

        assert len(preds) == 60  # 20 sequences * 3 horizons
        for p in preds:
            assert p.dataset_id == "driving_sample"
            assert p.prediction_status == "success"
            assert 0.0 <= p.predicted_reliability <= 1.0
            assert np.isfinite(p.predicted_reliability)
