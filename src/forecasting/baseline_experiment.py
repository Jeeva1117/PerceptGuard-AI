"""Baseline Experiment Orchestrator for PerceptGuard AI.

Runs baseline forecasting experiments (Persistence, Historical Mean, Linear Trend),
enforcing strict train/val/test split separation, zero data leakage, and standardized
artifact persistence under experiments/forecasting/baseline/<run_id>/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

import numpy as np
import yaml

from src.forecasting.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    DataLoaderError,
)
from src.forecasting.baseline_forecaster import (
    BaselineForecaster,
    PredictionRecord,
)
from src.forecasting.baseline_metrics import evaluate_predictions_by_horizon
from src.forecasting.forecasting_config import ForecastingConfig


class BaselineExperiment:
    """Manages baseline forecasting runs, evaluation, and artifact generation."""

    def __init__(
        self,
        artifacts_dir: Union[str, Path] = "datasets/processed/driving_sample/features/sequences/run_official",
        output_dir: Union[str, Path] = "experiments/forecasting/baseline",
        config_path: Union[str, Path] = "config/forecasting_config.yaml",
        strategies: Sequence[str] = ("persistence", "mean", "trend"),
        horizons: Sequence[int] = (1, 3, 5),
        sequence_length: int = 5,
        run_id: Optional[str] = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.output_root = Path(output_dir)
        self.config_path = Path(config_path)
        self.strategies = list(strategies)
        self.horizons = list(horizons)
        self.sequence_length = sequence_length

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:6]
        self.run_id = run_id or f"baseline_{now_str}_{short_uuid}"
        self.run_dir = self.output_root / self.run_id

    def run(
        self,
        historical_targets_map: Optional[Dict[str, Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """Execute the complete baseline forecasting experiment.
        
        Args:
            historical_targets_map: Optional mapping of sequence_id to historical targets.
            
        Returns:
            Dictionary summarizing experiment execution and results.
        """
        start_time = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Load Data
        loader = BaselineDataLoader(self.artifacts_dir)
        dataset = loader.load_sequence_dataset(
            sequence_length=self.sequence_length,
            target_horizons=self.horizons,
        )

        train_ds = dataset.filter_by_split("train")
        val_ds = dataset.filter_by_split("val")
        test_ds = dataset.filter_by_split("test")

        # 2. Check Ground-Truth Target Availability
        targets_available = dataset.target_labels_available
        if not targets_available:
            warnings.append(
                "Ground-truth reliability targets are NOT AVAILABLE in Day 5 sequence dataset. "
                "Supervised evaluation metrics cannot be calculated on real data without annotations. "
                "YOLO detection confidence alone is NOT ground-truth reliability."
            )

        # 3. Instantiate Forecaster & Fit on Train Split ONLY
        forecaster = BaselineForecaster(
            strategy="all",
            horizons=self.horizons,
        )
        forecaster.fit(train_ds)

        # 4. Generate Predictions Across Splits
        all_predictions: List[PredictionRecord] = []
        split_prediction_map: Dict[str, List[PredictionRecord]] = {}

        for split_name, split_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            preds = forecaster.predict_dataset(
                dataset=split_ds,
                strategy="all",
                horizons=self.horizons,
                historical_targets_map=historical_targets_map,
            )
            split_prediction_map[split_name] = preds
            all_predictions.extend(preds)

        # 5. Evaluate Metrics (by split and overall)
        eval_metrics = evaluate_predictions_by_horizon(
            predictions=[p.to_dict() for p in all_predictions],
            horizons=self.horizons,
            strategies=self.strategies,
        )

        eval_by_split: Dict[str, Any] = {}
        for split_name, preds in split_prediction_map.items():
            eval_by_split[split_name] = evaluate_predictions_by_horizon(
                predictions=[p.to_dict() for p in preds],
                horizons=self.horizons,
                strategies=self.strategies,
            )

        duration = round(time.perf_counter() - start_time, 4)

        # 6. Build Artifact Dictionaries
        experiment_config = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts_dir": str(self.artifacts_dir.resolve()),
            "dataset_id": dataset.dataset_id,
            "sequence_length": self.sequence_length,
            "forecast_horizons": self.horizons,
            "strategies": self.strategies,
            "training_mean": forecaster.training_mean,
            "training_samples_fitted": forecaster.training_sample_count,
        }

        metadata = {
            "run_id": self.run_id,
            "dataset_id": dataset.dataset_id,
            "feature_manifest_file": "feature_manifest.json",
            "dataset_splits_file": "dataset_splits.json",
            "sequence_metadata_file": "sequence_metadata.json",
            "feature_count": dataset.feature_count,
            "ordered_features": dataset.ordered_feature_names,
            "split_counts": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
                "total": len(dataset),
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
            },
        }

        summary = {
            "run_id": self.run_id,
            "status": "PASS" if not errors else "FAIL",
            "supervised_readiness": "PASS" if targets_available else "NOT AVAILABLE",
            "targets_available": targets_available,
            "execution_duration_seconds": duration,
            "total_sequences_evaluated": len(dataset),
            "total_predictions_generated": len(all_predictions),
            "valid_predictions_count": sum(1 for p in all_predictions if p.predicted_reliability is not None),
            "training_samples_fitted": forecaster.training_sample_count,
            "strategies_evaluated": self.strategies,
            "horizons_evaluated": self.horizons,
            "overall_metrics_status": eval_metrics.get("status"),
        }

        # 7. Persist Artifacts to Output Directory
        self._save_artifacts(
            config=experiment_config,
            predictions=[p.to_dict() for p in all_predictions],
            metrics={"overall": eval_metrics, "by_split": eval_by_split},
            summary=summary,
            metadata=metadata,
            errors={"errors": errors, "warnings": warnings},
        )

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": summary["status"],
            "supervised_readiness": summary["supervised_readiness"],
            "targets_available": targets_available,
            "execution_duration_seconds": duration,
            "total_predictions": len(all_predictions),
            "metrics": eval_metrics,
            "warnings": warnings,
            "errors": errors,
        }

    def _save_artifacts(
        self,
        config: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        summary: Dict[str, Any],
        metadata: Dict[str, Any],
        errors: Dict[str, Any],
    ) -> None:
        """Save standard experiment artifact suite."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "experiment_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        with open(self.run_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

        with open(self.run_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(self.run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(self.run_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)


def main() -> int:
    """CLI runner for baseline experiment."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI — Baseline Forecasting Experiment Runner"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/processed/driving_sample/features/sequences/run_official",
        help="Path to Day 5 sequence artifacts directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/forecasting/baseline",
        help="Directory to save experiment run outputs",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Sequence length (L) to evaluate",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Target horizons (e.g., 1 3 5)",
    )

    args = parser.parse_args()
    experiment = BaselineExperiment(
        artifacts_dir=args.input,
        output_dir=args.output,
        sequence_length=args.sequence_length,
        horizons=args.horizons,
    )
    result = experiment.run()

    print("=" * 70)
    print(" PerceptGuard AI — Baseline Forecasting Experiment Complete")
    print("=" * 70)
    print(f"Run ID                 : {result['run_id']}")
    print(f"Run Directory          : {result['run_dir']}")
    print(f"Status                 : {result['status']}")
    print(f"Supervised Readiness   : {result['supervised_readiness']}")
    print(f"Targets Available      : {result['targets_available']}")
    print(f"Total Predictions      : {result['total_predictions']}")
    print(f"Duration               : {result['execution_duration_seconds']}s")
    print("-" * 70)

    if result["warnings"]:
        print("WARNINGS:")
        for w in result["warnings"]:
            print(f"  ! {w}")

    if result["errors"]:
        print("ERRORS:")
        for e in result["errors"]:
            print(f"  * {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
