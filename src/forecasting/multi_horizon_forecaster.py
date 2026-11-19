"""Multi-Horizon Forecasting Engine for PerceptGuard AI.

Provides production-grade inference infrastructure for forecasting future perception reliability
across multiple forward horizons (t+1, t+3, t+5) using trained LSTMForecaster checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.forecasting.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    SequenceDatasetRecord,
)
from src.forecasting.checkpoint_manager import CheckpointError, CheckpointManager
from src.forecasting.forecasting_config import ForecastingConfig, ForecastingConfigError
from src.forecasting.lstm_forecaster import LSTMForecaster, ModelArchitectureError
from src.forecasting.model_factory import build_forecasting_model


class MultiHorizonForecastingError(ValueError):
    """Raised when multi-horizon forecasting input, alignment, or execution fails."""
    pass


@dataclass
class HorizonPredictionRecord:
    """Individual prediction record for a single sequence and specific forecast horizon."""
    dataset_id: str
    video_id: Optional[str]
    sequence_id: str
    input_start_frame: str
    input_end_frame: str
    input_end_source_frame_index: Optional[int]
    horizon: int
    target_frame_reference: str
    target_source_frame_index: Optional[int]
    target_timestamp_seconds: Optional[float]
    predicted_reliability: float
    prediction_type: str = "continuous_reliability"
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "unknown"
    inference_time_ms: float = 0.0
    prediction_status: str = "success"
    alignment_method: str = "metadata_exact"

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction record to dictionary."""
        return asdict(self)


@dataclass
class MultiHorizonForecastingSummary:
    """Summary of multi-horizon forecasting execution."""
    run_id: str
    total_input_sequences: int
    successful_predictions: int
    failed_predictions: int
    predictions_per_horizon: Dict[str, int]
    model_name: str
    checkpoint_id: str
    sequence_length: int
    feature_count: int
    inference_device: str
    average_inference_time_ms: float
    total_inference_duration_seconds: float
    target_definition: str = "continuous_reliability_in_range_0_1"
    prediction_type: str = "continuous_reliability"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


class MultiHorizonForecaster:
    """Production inference engine for multi-horizon perception reliability forecasting.
    
    Coordinates checkpoint loading, architectural verification, batch inference,
    temporal frame alignment, and multi-file artifact generation.
    """

    SUPPORTED_HORIZONS: Tuple[int, ...] = (1, 3, 5)

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[LSTMForecaster] = None,
        config: Optional[ForecastingConfig] = None,
        horizons: Sequence[int] = (1, 3, 5),
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.horizons = list(horizons)

        # Validate horizons
        for h in self.horizons:
            if h not in self.SUPPORTED_HORIZONS:
                raise MultiHorizonForecastingError(
                    f"Unsupported horizon {h}. Supported horizons are {self.SUPPORTED_HORIZONS}"
                )

        self.config = config or ForecastingConfig()
        self.checkpoint_metadata: Dict[str, Any] = {}
        self.checkpoint_id: str = "in_memory_or_synthetic"
        self.model: Optional[LSTMForecaster] = None

        if model is not None:
            if not isinstance(model, LSTMForecaster):
                raise MultiHorizonForecastingError(
                    f"Provided model must be an instance of LSTMForecaster, got: {type(model).__name__}"
                )
            self.model = model.to(self.device)
            self.model.eval()
            self._verify_model_horizons()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _verify_model_horizons(self) -> None:
        """Verify model's configured horizons match requested horizons."""
        if self.model is None:
            return
        model_horizons = list(getattr(self.model, "output_horizons", []))
        if model_horizons != self.horizons:
            raise MultiHorizonForecastingError(
                f"Model output horizons {model_horizons} do not match requested horizons {self.horizons}"
            )

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load, validate, and instantiate model from checkpoint file.
        
        Args:
            checkpoint_path: Path to PyTorch .pt checkpoint.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise MultiHorizonForecastingError(f"Checkpoint file not found: {path}")

        try:
            checkpoint_dict = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise MultiHorizonForecastingError(f"Corrupted or invalid checkpoint file at {path}: {e}") from e

        if not isinstance(checkpoint_dict, dict):
            raise MultiHorizonForecastingError(
                f"Invalid checkpoint format in {path}: expected dict, got {type(checkpoint_dict).__name__}"
            )

        if "model_state_dict" not in checkpoint_dict:
            raise MultiHorizonForecastingError(
                f"Checkpoint {path} missing required 'model_state_dict' key"
            )

        input_size = checkpoint_dict.get("input_size")
        hidden_size = checkpoint_dict.get("hidden_size", 32)
        chk_horizons = checkpoint_dict.get("output_horizons", self.horizons)

        if input_size is None or not isinstance(input_size, int) or input_size <= 0:
            raise MultiHorizonForecastingError(
                f"Checkpoint {path} contains invalid or missing 'input_size': {input_size}"
            )

        if list(chk_horizons) != self.horizons:
            raise MultiHorizonForecastingError(
                f"Checkpoint horizons {chk_horizons} do not match requested horizons {self.horizons}"
            )

        # Reconstruct model
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=checkpoint_dict.get("num_layers", 1),
            dropout=checkpoint_dict.get("dropout", 0.0),
            output_horizons=self.horizons,
            activation="sigmoid",
            batch_first=True,
        )

        try:
            model.load_state_dict(checkpoint_dict["model_state_dict"])
        except Exception as e:
            raise MultiHorizonForecastingError(f"Failed to load state_dict into model: {e}") from e

        model.to(self.device)
        model.eval()

        self.model = model
        self.checkpoint_metadata = checkpoint_dict
        self.checkpoint_id = checkpoint_dict.get("experiment_id", path.stem)

    def validate_input_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Validate input tensor shape, feature count, and finite numbers.
        
        Args:
            x: Input array or tensor of shape (batch_size, sequence_length, feature_count).
            
        Returns:
            Validated float32 PyTorch tensor on target device.
        """
        if self.model is None:
            raise MultiHorizonForecastingError(
                "Model is not loaded. Load a valid checkpoint before running inference."
            )

        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            tensor = x.float()
        else:
            raise MultiHorizonForecastingError(
                f"Input must be numpy.ndarray or torch.Tensor, got: {type(x).__name__}"
            )

        if tensor.ndim != 3:
            raise MultiHorizonForecastingError(
                f"Input must be 3-dimensional (batch_size, sequence_length, feature_count), "
                f"got ndim={tensor.ndim} with shape {list(tensor.shape)}"
            )

        batch_size, seq_len, feature_count = tensor.shape

        if batch_size == 0:
            raise MultiHorizonForecastingError("Cannot perform inference on empty batch (batch_size=0)")

        if seq_len == 0:
            raise MultiHorizonForecastingError("Sequence length must be > 0")

        if feature_count != self.model.input_size:
            raise MultiHorizonForecastingError(
                f"Input feature count ({feature_count}) does not match model input_size ({self.model.input_size})"
            )

        if not torch.isfinite(tensor).all():
            raise MultiHorizonForecastingError("Input sequence tensor contains non-finite numbers (NaN or Inf)")

        return tensor.to(self.device)

    def align_horizon_target(
        self,
        sequence_record: Optional[SequenceDatasetRecord],
        horizon: int,
        end_frame_id: str,
        end_source_idx: Optional[int] = None,
        end_timestamp: Optional[float] = None,
        available_frame_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[int], Optional[float], str]:
        """Align future target frame reference, source index, and timestamp.
        
        Args:
            sequence_record: Optional SequenceDatasetRecord with pre-computed horizon metadata.
            horizon: Forecast horizon (1, 3, or 5).
            end_frame_id: Final observed historical frame ID (e.g. 'frame_000005.jpg').
            end_source_idx: Source frame index of final historical frame.
            end_timestamp: Timestamp of final historical frame.
            available_frame_records: Optional list of chronological frame records for index lookup.
            
        Returns:
            Tuple of (target_frame_reference, target_source_idx, target_timestamp, alignment_method).
        """
        # 1. If sequence_record has pre-computed horizons metadata from Day 5
        if sequence_record is not None and sequence_record.horizons:
            h_key = f"h{horizon}"
            if h_key in sequence_record.horizons:
                h_info = sequence_record.horizons[h_key]
                t_ref = h_info.get("target_frame_id") or f"frame_t+{horizon}"
                t_src = h_info.get("target_source_frame_index")
                t_time = h_info.get("target_timestamp_seconds")
                return str(t_ref), t_src, t_time, "metadata_exact"

        # 2. If available_frame_records is provided, perform precise discrete lookup
        if available_frame_records is not None and end_frame_id:
            # Locate current frame in available records
            curr_idx = -1
            for idx, f_meta in enumerate(available_frame_records):
                if f_meta.get("frame_id") == end_frame_id:
                    curr_idx = idx
                    break

            if curr_idx != -1:
                target_idx = curr_idx + horizon
                if target_idx < len(available_frame_records):
                    target_rec = available_frame_records[target_idx]
                    t_ref = target_rec.get("frame_id", f"frame_idx_{target_idx:06d}")
                    t_src = target_rec.get("source_frame_index")
                    t_time = target_rec.get("timestamp_seconds")
                    return str(t_ref), t_src, t_time, "frame_index_offset"
                else:
                    return f"out_of_bounds_t+{horizon}", None, None, "out_of_bounds"

        # 3. Fallback: Parse frame number if pattern is standard frame_XXXXXX.ext
        if end_frame_id and "frame_" in end_frame_id:
            try:
                base_name = os.path.basename(end_frame_id)
                num_str = base_name.replace("frame_", "").split(".")[0]
                frame_num = int(num_str)
                target_num = frame_num + horizon
                t_ref = f"frame_{target_num:06d}.jpg"
                t_src = end_source_idx + (horizon * 6) if end_source_idx is not None else None
                t_time = end_timestamp + (horizon * 0.1994) if end_timestamp is not None else None
                return t_ref, t_src, t_time, "sequential_extrapolation"
            except (ValueError, IndexError):
                pass

        return f"target_t+{horizon}", None, None, "unaligned_fallback"

    def predict_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        sequence_record: Optional[SequenceDatasetRecord] = None,
        video_id: Optional[str] = None,
    ) -> List[HorizonPredictionRecord]:
        """Generate multi-horizon predictions for a single input sequence.
        
        Args:
            sequence: Sequence array or tensor of shape (sequence_length, feature_count).
            sequence_record: Optional metadata record.
            video_id: Optional video identifier.
            
        Returns:
            List of HorizonPredictionRecord objects for each configured horizon.
        """
        if isinstance(sequence, (np.ndarray, torch.Tensor)):
            if sequence.ndim == 2:
                # Add batch dimension (1, L, F)
                if isinstance(sequence, np.ndarray):
                    batched = np.expand_dims(sequence, axis=0)
                else:
                    batched = sequence.unsqueeze(0)
            elif sequence.ndim == 3:
                batched = sequence
            else:
                raise MultiHorizonForecastingError(
                    f"Expected 2D or 3D sequence array, got ndim={sequence.ndim}"
                )
        else:
            raise MultiHorizonForecastingError(f"Invalid sequence type: {type(sequence).__name__}")

        records = [sequence_record] if sequence_record is not None else None
        return self.predict_batch(sequences=batched, sequence_records=records, video_id=video_id)

    def predict_batch(
        self,
        sequences: Union[np.ndarray, torch.Tensor],
        sequence_records: Optional[List[Optional[SequenceDatasetRecord]]] = None,
        video_id: Optional[str] = None,
        available_frame_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[HorizonPredictionRecord]:
        """Execute multi-horizon inference over a batch of sequences.
        
        Args:
            sequences: Sequence tensor of shape (batch_size, sequence_length, feature_count).
            sequence_records: Optional list of matching SequenceDatasetRecord objects.
            video_id: Optional video identifier.
            available_frame_records: Optional list of chronological frame records for alignment.
            
        Returns:
            List of HorizonPredictionRecord objects.
        """
        tensor = self.validate_input_tensor(sequences)
        batch_size = tensor.shape[0]

        if sequence_records is not None and len(sequence_records) != batch_size:
            raise MultiHorizonForecastingError(
                f"sequence_records length ({len(sequence_records)}) != batch size ({batch_size})"
            )

        start_time = time.perf_counter()

        with torch.inference_mode():
            # predictions shape: (batch_size, num_horizons)
            raw_predictions = self.model(tensor)

        total_inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        avg_sample_time_ms = total_inference_time_ms / max(batch_size, 1)

        # Move to CPU numpy array
        pred_array = raw_predictions.detach().cpu().numpy()

        results: List[HorizonPredictionRecord] = []

        for b in range(batch_size):
            sec_rec = sequence_records[b] if sequence_records is not None else None

            # Extract sequence identity & boundaries
            if sec_rec is not None:
                seq_id = sec_rec.sequence_id
                dataset_id = sec_rec.dataset_id
                start_frame = sec_rec.start_frame_id
                end_frame = sec_rec.end_frame_id
                end_src_idx = sec_rec.end_source_frame_index
                end_timestamp = sec_rec.end_timestamp_seconds
            else:
                seq_id = f"seq_{b:06d}"
                dataset_id = "unknown_dataset"
                start_frame = "frame_000001"
                end_frame = "frame_000005"
                end_src_idx = None
                end_timestamp = None

            for col_idx, horizon in enumerate(self.horizons):
                pred_val = float(pred_array[b, col_idx])

                # Constrain bounded continuous score in [0.0, 1.0]
                bounded_val = max(0.0, min(1.0, pred_val))

                t_ref, t_src, t_time, align_method = self.align_horizon_target(
                    sequence_record=sec_rec,
                    horizon=horizon,
                    end_frame_id=end_frame,
                    end_source_idx=end_src_idx,
                    end_timestamp=end_timestamp,
                    available_frame_records=available_frame_records,
                )

                rec = HorizonPredictionRecord(
                    dataset_id=dataset_id,
                    video_id=video_id,
                    sequence_id=seq_id,
                    input_start_frame=start_frame,
                    input_end_frame=end_frame,
                    input_end_source_frame_index=end_src_idx,
                    horizon=horizon,
                    target_frame_reference=t_ref,
                    target_source_frame_index=t_src,
                    target_timestamp_seconds=t_time,
                    predicted_reliability=bounded_val,
                    prediction_type="continuous_reliability",
                    model_name="LSTMForecaster",
                    checkpoint_id=self.checkpoint_id,
                    inference_time_ms=avg_sample_time_ms,
                    prediction_status="success",
                    alignment_method=align_method,
                )
                results.append(rec)

        return results

    def predict_dataset(
        self,
        dataset: BaselineDataset,
        max_sequences: Optional[int] = None,
        video_id: Optional[str] = None,
    ) -> List[HorizonPredictionRecord]:
        """Execute chunked batch inference over a BaselineDataset.
        
        Args:
            dataset: BaselineDataset containing tensors and records.
            max_sequences: Optional maximum sequence limit.
            video_id: Optional video identifier.
            
        Returns:
            List of HorizonPredictionRecord objects.
        """
        all_tensors = dataset.tensors
        all_records = dataset.records

        total = len(all_records)
        if max_sequences is not None:
            total = min(total, int(max_sequences))

        predictions: List[HorizonPredictionRecord] = []

        # Iterate in memory-bounded batches
        for start_idx in range(0, total, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total)
            batch_tensors = all_tensors[start_idx:end_idx]
            batch_records = all_records[start_idx:end_idx]

            batch_preds = self.predict_batch(
                sequences=batch_tensors,
                sequence_records=batch_records,
                video_id=video_id or dataset.dataset_id,
            )
            predictions.extend(batch_preds)

        return predictions

    def save_prediction_artifacts(
        self,
        predictions: List[HorizonPredictionRecord],
        output_dir: Union[str, Path],
        manifest_data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Persist structured multi-horizon forecasting artifacts.
        
        Generates:
        - predictions.json
        - horizon_1_predictions.json
        - horizon_3_predictions.json
        - horizon_5_predictions.json
        - prediction_summary.json
        - model_metadata.json
        - feature_manifest.json
        - inference_metadata.json
        
        Returns:
            Dictionary of artifact names to Path objects.
        """
        r_id = run_id or f"forecasting_{time.strftime('%Y%m%d_%H%M%S')}"
        target_dir = Path(output_dir) / r_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Split predictions by horizon
        h1_preds = [p.to_dict() for p in predictions if p.horizon == 1]
        h3_preds = [p.to_dict() for p in predictions if p.horizon == 3]
        h5_preds = [p.to_dict() for p in predictions if p.horizon == 5]
        all_preds = [p.to_dict() for p in predictions]

        # 2. Write predictions.json and horizon-specific files
        artifacts: Dict[str, Path] = {}

        p_all = target_dir / "predictions.json"
        with open(p_all, "w", encoding="utf-8") as f:
            json.dump(all_preds, f, indent=2)
        artifacts["predictions"] = p_all

        p_h1 = target_dir / "horizon_1_predictions.json"
        with open(p_h1, "w", encoding="utf-8") as f:
            json.dump(h1_preds, f, indent=2)
        artifacts["horizon_1_predictions"] = p_h1

        p_h3 = target_dir / "horizon_3_predictions.json"
        with open(p_h3, "w", encoding="utf-8") as f:
            json.dump(h3_preds, f, indent=2)
        artifacts["horizon_3_predictions"] = p_h3

        p_h5 = target_dir / "horizon_5_predictions.json"
        with open(p_h5, "w", encoding="utf-8") as f:
            json.dump(h5_preds, f, indent=2)
        artifacts["horizon_5_predictions"] = p_h5

        # 3. Model metadata
        model_meta = {}
        if self.model is not None:
            model_meta = self.model.get_model_summary()
        m_file = target_dir / "model_metadata.json"
        with open(m_file, "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)
        artifacts["model_metadata"] = m_file

        # 4. Feature manifest
        feat_meta = manifest_data or {}
        f_file = target_dir / "feature_manifest.json"
        with open(f_file, "w", encoding="utf-8") as f:
            json.dump(feat_meta, f, indent=2)
        artifacts["feature_manifest"] = f_file

        # 5. Inference metadata
        inf_meta = {
            "run_id": r_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cpu_count": os.cpu_count(),
            "device": str(self.device),
            "batch_size": self.batch_size,
            "checkpoint_id": self.checkpoint_id,
        }
        i_file = target_dir / "inference_metadata.json"
        with open(i_file, "w", encoding="utf-8") as f:
            json.dump(inf_meta, f, indent=2)
        artifacts["inference_metadata"] = i_file

        # 6. Prediction summary
        unique_seqs = len(set(p.sequence_id for p in predictions))
        total_time_ms = sum(p.inference_time_ms for p in predictions)
        avg_time_ms = total_time_ms / max(len(predictions), 1)

        summary = MultiHorizonForecastingSummary(
            run_id=r_id,
            total_input_sequences=unique_seqs,
            successful_predictions=len(predictions),
            failed_predictions=0,
            predictions_per_horizon={
                "h1": len(h1_preds),
                "h3": len(h3_preds),
                "h5": len(h5_preds),
            },
            model_name="LSTMForecaster",
            checkpoint_id=self.checkpoint_id,
            sequence_length=self.config.data.sequence_lengths[0] if self.config.data.sequence_lengths else 5,
            feature_count=self.model.input_size if self.model else 27,
            inference_device=str(self.device),
            average_inference_time_ms=avg_time_ms,
            total_inference_duration_seconds=total_time_ms / 1000.0,
        )
        s_file = target_dir / "prediction_summary.json"
        with open(s_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        artifacts["prediction_summary"] = s_file

        return artifacts


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for multi-horizon forecasting."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI - Multi-Horizon Perception Reliability Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/processed/driving_sample/features/sequences/run_official",
        help="Path to Day 5 sequence directory or sequence tensor artifact.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained LSTMForecaster checkpoint file (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration YAML file.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Forecast horizons to predict (default: 1 3 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference execution device.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Sequence length L to load (e.g. 5, 10, 15, 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/forecasting/multi_horizon",
        help="Destination directory for forecasting artifacts.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limit number of sequences to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed progress logging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    return parser.parse_args(args)


def run_forecasting_cli(args: argparse.Namespace) -> int:
    """Execute multi-horizon forecasting CLI pipeline."""
    print("=" * 78)
    print("  PERCEPTGUARD AI - MULTI-HORIZON RELIABILITY FORECASTING ENGINE")
    print("=" * 78)

    # 1. Validate Config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        return 1

    try:
        config = ForecastingConfig.from_yaml(config_path, validate=True)
        print(f"[CONFIG] Loaded valid configuration from: {config_path}")
    except ForecastingConfigError as e:
        print(f"[ERROR] Configuration validation failed:\n{e}")
        return 1

    # 2. Checkpoint Availability Verification
    checkpoint_path = args.checkpoint
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("-" * 78)
        print("  TRAINED CHECKPOINT READINESS VERIFICATION")
        print("-" * 78)
        print("[STATUS] Trained Checkpoint: NOT AVAILABLE")
        print("[REASON] No trained LSTM checkpoint provided or found at specified path.")
        print("         Under Day 6 Task 4, real-data model training was halted before Epoch 1")
        print("         because verified ground-truth reliability labels were not available.")
        print("[POLICY] PerceptGuard AI strictly enforces scientific label & model integrity:")
        print("         - Inference cannot run on real data without a genuine trained checkpoint.")
        print("         - Fabricated checkpoints or dummy predictions are strictly forbidden.")
        print("[ACTION] Halting inference pipeline. Multi-horizon forecasting infrastructure")
        print("         is verified and ready for deployment once a valid checkpoint exists.")
        print("=" * 78)
        return 0

    # 3. Load Sequence Artifacts
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input sequence artifact directory not found: {input_path}")
        return 1

    print(f"[DATA] Loading sequence dataset (L={args.sequence_length}) from: {input_path}")
    try:
        loader = BaselineDataLoader(artifacts_dir=input_path)
        dataset = loader.load_sequence_dataset(
            sequence_length=args.sequence_length,
            target_horizons=args.horizons,
        )
        print(f"[DATA] Successfully loaded {len(dataset)} sequences (Features: {dataset.feature_count})")
    except Exception as e:
        print(f"[ERROR] Failed to load sequence dataset: {e}")
        return 1

    # 4. Instantiate MultiHorizonForecaster
    print(f"[MODEL] Loading LSTM checkpoint from: {checkpoint_path}")
    try:
        forecaster = MultiHorizonForecaster(
            checkpoint_path=checkpoint_path,
            config=config,
            horizons=args.horizons,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"[MODEL] Model successfully loaded on {args.device}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize forecaster: {e}")
        return 1

    # 5. Run Multi-Horizon Inference
    print(f"[INFERENCE] Generating multi-horizon predictions for horizons: {args.horizons}...")
    start_t = time.time()
    try:
        predictions = forecaster.predict_dataset(
            dataset=dataset,
            max_sequences=args.max_sequences,
        )
    except Exception as e:
        print(f"[ERROR] Inference execution failed: {e}")
        return 1

    elapsed = time.time() - start_t
    print(f"[INFERENCE] Completed {len(predictions)} predictions across {len(dataset)} sequences in {elapsed:.3f}s")

    # 6. Save Artifacts
    print(f"[ARTIFACTS] Writing output artifacts to: {args.output}")
    try:
        artifacts = forecaster.save_prediction_artifacts(
            predictions=predictions,
            output_dir=args.output,
            manifest_data=loader.manifest_data,
        )
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to write prediction artifacts: {e}")
        return 1

    print("=" * 78)
    print("  MULTI-HORIZON FORECASTING COMPLETED SUCCESSFULLY")
    print("=" * 78)
    return 0


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    code = run_forecasting_cli(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
