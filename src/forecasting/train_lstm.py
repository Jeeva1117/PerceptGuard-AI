"""LSTM Training CLI for PerceptGuard AI.

Provides the command-line interface for LSTM model training, hyperparameter configuration,
supervised target readiness verification, and reproducible experiment execution.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

import torch

from src.forecasting.checkpoint_manager import CheckpointManager
from src.forecasting.early_stopping import EarlyStopping
from src.forecasting.forecasting_config import ForecastingConfig, ForecastingConfigError
from src.forecasting.model_factory import build_forecasting_model
from src.forecasting.training_data import ForecastingDataModule
from src.forecasting.training_engine import LSTMTrainingEngine, TrainingEngineError
from src.forecasting.training_logger import TrainingLogger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for LSTM training CLI."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI - Advanced LSTM Model Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration YAML file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for training and validation loaders.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override AdamW learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target execution device (CPU recommended).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override directory for experiment telemetry and checkpoints.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Sequence length L to load (e.g. 5, 10, 15, 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed progress logging.",
    )
    return parser.parse_args(args)


def run_training_pipeline(args: argparse.Namespace) -> int:
    """Execute training CLI pipeline with strict target readiness gating."""
    print("=" * 78)
    print("  PERCEPTGUARD AI - ADVANCED LSTM MODEL TRAINING PIPELINE")
    print("=" * 78)

    # 1. Load Configuration
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

    # Overrides
    batch_size = args.batch_size or config.data.batch_size
    max_epochs = args.epochs or config.training.max_epochs
    learning_rate = args.learning_rate or config.training.learning_rate
    device = args.device or config.runtime.device
    seq_length = args.sequence_length

    # 2. Locate Day 5 Sequence Artifacts
    candidate_paths = [
        Path(config.data.sequence_directory) / "driving_sample" / "features" / "sequences" / "run_official",
        Path("datasets/processed/driving_sample/features/sequences/run_official"),
    ]
    artifacts_dir: Optional[Path] = None
    for p in candidate_paths:
        if p.exists() and (p / "feature_manifest.json").exists():
            artifacts_dir = p
            break

    if artifacts_dir is None:
        print(f"[ERROR] Day 5 sequence artifacts directory not found in candidate paths.")
        return 1

    print(f"[DATA] Located sequence artifacts at: {artifacts_dir}")

    # 3. Load DataModule and Verify Dataset Splits
    try:
        data_module = ForecastingDataModule.from_artifacts(
            artifacts_dir=artifacts_dir,
            sequence_length=seq_length,
            horizons=config.model.output_horizons,
            batch_size=batch_size,
            num_workers=config.data.num_workers,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load sequence data module: {e}")
        return 1

    train_count = len(data_module.train_dataset)
    val_count = len(data_module.val_dataset)
    test_count = len(data_module.test_dataset) if data_module.test_dataset else 0
    total_count = train_count + val_count + test_count

    print(f"[DATA] Sequence Length: L = {seq_length}")
    print(f"[DATA] Split Counts: Train={train_count}, Val={val_count}, Test={test_count} (Total={total_count})")
    print(f"[DATA] Batch Size: {batch_size}")

    # 4. CRITICAL: Supervised Target Readiness Gate
    print("-" * 78)
    print("  SUPERVISED TARGET READINESS GATE VERIFICATION")
    print("-" * 78)

    if not data_module.has_targets:
        print("[STATUS] Supervised Target Readiness: NOT AVAILABLE")
        print("[REASON] The Day 5 driving_sample sequence artifacts do not contain verified")
        print("         ground-truth future detection reliability target labels.")
        print("[POLICY] PerceptGuard AI strictly enforces scientific label integrity:")
        print("         - YOLO confidence scores are detection probabilities, NOT future reliability labels.")
        print("         - Synthetic labels are never fabricated or injected into real driving datasets.")
        print("         - Model weights are not updated via backpropagation without verified labels.")
        print("[ACTION] Halting training loop before Epoch 1. Training pipeline is verified & ready.")
        print("[NEXT STEP] Once annotated ground-truth reliability targets (e.g. future IoU or TP status)")
        print("            are generated, re-run this CLI to initiate supervised training.")
        print("=" * 78)
        return 0

    # 5. Build Model
    print(f"[MODEL] Building LSTM forecasting model (Device: {device})...")
    try:
        model = build_forecasting_model(
            config=config,
            artifacts_dir=artifacts_dir,
            device=device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to build model: {e}")
        return 1

    summary = model.get_model_summary()
    print(f"[MODEL] Trainable Parameters: {summary['trainable_parameters']}")
    print(f"[MODEL] Horizons: {summary['output_horizons']}")

    # 6. Initialize Engine
    run_id = f"lstm_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(args.output) if args.output else Path(config.artifacts.experiment_directory) / "lstm_training" / run_id
    chk_dir = exp_dir / "checkpoints"

    logger = TrainingLogger(experiment_dir=exp_dir, run_id=run_id, config=config)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=chk_dir, experiment_id=run_id)

    engine = LSTMTrainingEngine(
        model=model,
        config=config,
        checkpoint_manager=checkpoint_mgr,
        logger=logger,
        device=device,
    )

    # 7. Execute Training
    print("-" * 78)
    print(f"  STARTING SUPERVISED TRAINING ({max_epochs} EPOCHS)")
    print("-" * 78)

    try:
        results = engine.train(
            data_module=data_module,
            epochs=max_epochs,
            resume_checkpoint_path=args.resume,
            verbose=args.verbose,
        )
        print("=" * 78)
        print(f"[SUCCESS] Training completed with status: {results['status']}")
        print(f"[RESULTS] Best Epoch: {results['best_epoch']} | Best Val Loss: {results['best_val_loss']:.6f}")
        print(f"[RESULTS] Total Duration: {results['total_duration_seconds']:.2f}s")
        print(f"[OUTPUT] Summary: {results['summary_path']}")
        print("=" * 78)
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user.")
        logger.save_summary(
            total_epochs=0,
            best_epoch=-1,
            best_val_loss=float("inf"),
            total_duration_seconds=0.0,
            status="INTERRUPTED",
            termination_reason="Interrupted by user (KeyboardInterrupt)",
        )
        return 130
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        logger.log_error(str(e))
        return 1


def main() -> None:
    """CLI Entry point."""
    args = parse_args()
    code = run_training_pipeline(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
