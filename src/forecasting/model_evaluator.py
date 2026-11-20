"""Model Evaluation Module for PerceptGuard AI Forecasting.

Provides scientific evaluation of multi-horizon perception reliability forecasts (horizons 1, 3, 5),
strict ground-truth validation, baseline model comparison, comprehensive error analysis,
and structured artifact generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.forecasting.baseline_metrics import calculate_mae, calculate_rmse
from src.forecasting.forecasting_config import ForecastingConfig, ForecastingConfigError
from src.forecasting.multi_horizon_forecaster import HorizonPredictionRecord

logger = logging.getLogger(__name__)


class ModelEvaluationError(ValueError):
    """Raised when evaluation inputs, ground-truth alignment, or calculation fails."""
    pass


@dataclass
class GroundTruthObservation:
    """Ground-truth reliability observation record for a sequence and horizon."""
    observation_id: str
    dataset_id: str
    video_id: str
    target_frame_index: int
    horizon: int
    ground_truth_reliability: float
    split: str = "test"
    sequence_id: Optional[str] = None
    target_frame_reference: Optional[str] = None
    target_source_frame_index: Optional[int] = None
    target_timestamp_seconds: Optional[float] = None
    target_definition: str = "future_object_detection_reliability"
    annotation_source: str = "annotated_ground_truth"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return asdict(self)


@dataclass
class AlignedEvaluationPair:
    """Aligned pair consisting of a model prediction and matched ground-truth observation."""
    prediction_id: str
    observation_id: str
    dataset_id: str
    video_id: str
    source_frame_index: Optional[int]
    target_frame_index: int
    horizon: int
    predicted_reliability: float
    ground_truth_reliability: float
    split: str = "test"
    sequence_id: Optional[str] = None
    target_frame_reference: Optional[str] = None
    target_source_frame_index: Optional[int] = None
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "best_model.pt"
    error: float = 0.0
    absolute_error: float = 0.0
    percentage_error: Optional[float] = None
    features_snapshot: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Calculate error metrics automatically if default."""
        self.error = float(self.predicted_reliability - self.ground_truth_reliability)
        self.absolute_error = float(abs(self.error))
        if abs(self.ground_truth_reliability) > 1e-6:
            self.percentage_error = float((self.absolute_error / self.ground_truth_reliability) * 100.0)
        else:
            self.percentage_error = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert aligned pair to dictionary."""
        return asdict(self)


@dataclass
class HorizonEvaluationMetrics:
    """Comprehensive regression evaluation metrics for a single forecast horizon."""
    horizon: int
    sample_count: int
    mae: float
    rmse: float
    bias: float
    median_absolute_error: float
    error_std: float
    mape: Optional[float]
    underprediction_count: int
    overprediction_count: int
    exact_prediction_count: int
    min_error: float
    max_error: float
    exact_match_count: int = 0
    mse: float = 0.0
    r2: Optional[float] = None
    mean_actual: float = 0.0
    mean_predicted: float = 0.0
    actual_std: float = 0.0
    pred_std: float = 0.0
    min_predicted: float = 0.0
    max_predicted: float = 0.0
    min_actual: float = 0.0
    max_actual: float = 0.0

    def __post_init__(self) -> None:
        if self.exact_match_count == 0 and self.exact_prediction_count > 0:
            self.exact_match_count = self.exact_prediction_count
        elif self.exact_prediction_count == 0 and self.exact_match_count > 0:
            self.exact_prediction_count = self.exact_match_count
        if self.mse == 0.0 and self.rmse > 0.0:
            self.mse = float(self.rmse ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class ForecastingModelEvaluator:
    """Scientific evaluation engine for multi-horizon forecasting models."""

    SUPPORTED_HORIZONS: Tuple[int, ...] = (1, 3, 5)

    def __init__(
        self,
        config: Optional[ForecastingConfig] = None,
        horizons: Sequence[int] = (1, 3, 5),
    ) -> None:
        self.config = config or ForecastingConfig()
        self.horizons = list(horizons)

        for h in self.horizons:
            if h not in self.SUPPORTED_HORIZONS:
                raise ModelEvaluationError(
                    f"Unsupported horizon {h}. Supported horizons are {self.SUPPORTED_HORIZONS}"
                )

    def load_predictions(
        self,
        source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Load and parse prediction records from JSON file, dictionary, or raw list."""
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise ModelEvaluationError(f"Predictions source not found: {src_path}")

            if src_path.is_dir():
                cand = src_path / "predictions.json"
                if not cand.exists():
                    raise ModelEvaluationError(f"Missing predictions.json in directory: {src_path}")
                src_path = cand

            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(source, (dict, list)):
            data = source
        else:
            raise ModelEvaluationError(f"Invalid predictions source type: {type(source).__name__}")

        if isinstance(data, dict):
            if "predictions" in data:
                data = data["predictions"]
            elif "records" in data:
                data = data["records"]
            else:
                raise ModelEvaluationError("Predictions data dict must contain 'predictions' or 'records' key")

        if not isinstance(data, list):
            raise ModelEvaluationError("Predictions data must be a JSON array of records")

        validated: List[Dict[str, Any]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ModelEvaluationError(f"Record at index {idx} is not a dictionary")

            # Validate prediction fields
            p_val = item.get("predicted_reliability")
            if p_val is None:
                raise ModelEvaluationError(f"Record {idx} missing 'predicted_reliability'")

            horizon = int(item.get("horizon", 1))
            validated.append(item)

        return validated

    def load_ground_truth(
        self,
        source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[GroundTruthObservation]:
        """Load and validate ground-truth observations catalog."""
        observations: List[GroundTruthObservation] = []

        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise ModelEvaluationError(f"Ground-truth source not found: {src_path}")

            if src_path.is_dir():
                cand = src_path / "ground_truth.json"
                if not cand.exists():
                    raise ModelEvaluationError(f"Missing ground_truth.json in directory: {src_path}")
                src_path = cand

            with open(src_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif isinstance(source, (dict, list)):
            raw_data = source
        else:
            raise ModelEvaluationError(f"Invalid ground-truth source type: {type(source).__name__}")

        # Check for target integrity violations in metadata if dict
        if isinstance(raw_data, dict):
            meta = raw_data.get("metadata", {})
            target_def = str(meta.get("target_definition", "")).lower()
            if "yolo" in target_def or "confidence" in target_def or "heuristic" in target_def:
                raise ModelEvaluationError(
                    f"CRITICAL TARGET INTEGRITY VIOLATION: Target definition '{target_def}' uses "
                    "YOLO confidence or heuristic proxies. Ground truth must represent true object "
                    "detection reliability (e.g., IoU >= 0.5 against human-annotated labels)."
                )

            obs_list = raw_data.get("observations")
            if obs_list is None:
                obs_list = raw_data.get("ground_truth")
            if obs_list is None and "sequences" in raw_data:
                # Extract observations from sequence_metadata_labeled.json format
                obs_list = []
                for s in raw_data.get("sequences", []):
                    if s.get("input_sequence_length", 5) == 5:
                        sid = s.get("sequence_id")
                        d_id = str(s.get("dataset_id", "driving_sample"))
                        v_id = str(s.get("video_id", d_id))
                        split = str(s.get("split", "test"))
                        for h_key, h_info in s.get("horizons", {}).items():
                            if h_info.get("target_label") is not None and h_info.get("target_status") == "valid":
                                obs_list.append({
                                    "observation_id": f"gt_{sid}_{h_key}",
                                    "dataset_id": d_id,
                                    "video_id": v_id,
                                    "sequence_id": sid,
                                    "target_frame_index": int(h_info.get("target_source_frame_index", 0)),
                                    "target_source_frame_index": h_info.get("target_source_frame_index"),
                                    "target_frame_reference": h_info.get("target_frame_id"),
                                    "target_timestamp_seconds": h_info.get("target_timestamp_seconds"),
                                    "horizon": int(h_info.get("horizon", 1)),
                                    "ground_truth_reliability": float(h_info["target_label"]),
                                    "split": split,
                                    "target_definition": "annotated_ground_truth_iou",
                                    "annotation_source": "human_verified_annotation",
                                })
            if obs_list is None:
                raise ModelEvaluationError("Ground-truth dict missing 'observations' list")
            data = obs_list
        elif isinstance(raw_data, list):
            data = raw_data
        else:
            raise ModelEvaluationError("Ground-truth data must be a JSON array or dict with 'observations'")

        if not isinstance(data, list):
            raise ModelEvaluationError("Ground-truth observations must be a list")

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ModelEvaluationError(f"Ground-truth observation at index {idx} is not a dictionary")

            # Check target integrity inside item if present
            item_target_def = str(item.get("target_definition", "")).lower()
            if "yolo" in item_target_def or "confidence" in item_target_def or "heuristic" in item_target_def:
                raise ModelEvaluationError(
                    f"CRITICAL TARGET INTEGRITY VIOLATION: Item {idx} uses invalid target '{item_target_def}'"
                )

            val = item.get("ground_truth_reliability")
            if val is None or not np.isfinite(val):
                raise ModelEvaluationError(f"Ground-truth observation {idx} contains non-finite reliability: {val}")

            obs = GroundTruthObservation(
                observation_id=str(item.get("observation_id", f"gt_{idx}")),
                dataset_id=str(item.get("dataset_id", "default_dataset")),
                video_id=str(item.get("video_id", "default_video")),
                target_frame_index=int(item.get("target_frame_index", item.get("target_source_frame_index", 0))),
                horizon=int(item.get("horizon", 1)),
                ground_truth_reliability=float(val),
                split=str(item.get("split", "test")),
                sequence_id=str(item.get("sequence_id", "")) if item.get("sequence_id") else None,
                target_frame_reference=str(item.get("target_frame_reference", "")) if item.get("target_frame_reference") else None,
                target_source_frame_index=item.get("target_source_frame_index"),
                target_timestamp_seconds=item.get("target_timestamp_seconds"),
                target_definition=str(item.get("target_definition", "future_object_detection_reliability")),
                annotation_source=str(item.get("annotation_source", "annotated_ground_truth")),
                metadata=item.get("metadata", {}),
            )
            observations.append(obs)

        return observations

    def validate_ground_truth_contract(
        self,
        source: Union[Dict[str, Any], List[Dict[str, Any]], str, Path],
    ) -> Dict[str, Any]:
        """Validate ground-truth schema, metadata, target definitions, and values."""
        try:
            if isinstance(source, (str, Path)):
                p = Path(source)
                if not p.exists():
                    return {"valid": False, "status": "NOT_AVAILABLE", "message": f"Source path not found: {p}"}
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = source

            if isinstance(data, dict):
                meta = data.get("metadata", {})
                target_labels_available = meta.get("target_labels_available", True)
                if not target_labels_available:
                    return {
                        "valid": False,
                        "status": "NOT_AVAILABLE",
                        "message": "Ground-truth target labels are marked as not available in metadata.",
                    }

                target_def = str(meta.get("target_definition", "")).lower()
                if "yolo" in target_def or "confidence" in target_def or "heuristic" in target_def:
                    return {
                        "valid": False,
                        "status": "FAIL",
                        "message": f"Invalid target definition '{target_def}'. YOLO confidence is not ground truth.",
                    }

                obs_raw = data.get("observations", data.get("ground_truth"))
                if obs_raw is None and "sequences" in data:
                    obs_raw = []
                    for s in data.get("sequences", []):
                        if s.get("input_sequence_length", 5) == 5:
                            for h_key, h_info in s.get("horizons", {}).items():
                                if h_info.get("target_label") is not None and h_info.get("target_status") == "valid":
                                    obs_raw.append({
                                        "observation_id": f"gt_{s.get('sequence_id')}_{h_key}",
                                        "target_frame_index": int(h_info.get("target_source_frame_index", 0)),
                                        "ground_truth_reliability": float(h_info["target_label"]),
                                        "horizon": int(h_info.get("horizon", 1)),
                                    })
            elif isinstance(data, list):
                obs_raw = data
            else:
                return {"valid": False, "status": "FAIL", "message": "Ground truth must be dict or list"}

            if not obs_raw:
                return {"valid": False, "status": "NOT_AVAILABLE", "message": "Ground-truth observations list is empty."}

            # Check individual observations
            for idx, obs in enumerate(obs_raw):
                if not isinstance(obs, dict):
                    return {"valid": False, "status": "FAIL", "message": f"Observation at index {idx} is not a dict"}

                # Required keys
                if "ground_truth_reliability" not in obs or "target_frame_index" not in obs:
                    return {"valid": False, "status": "FAIL", "message": f"Observation {idx} missing required keys."}

                val = obs["ground_truth_reliability"]
                if not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0.0 or val > 1.0:
                    return {
                        "valid": False,
                        "status": "FAIL",
                        "message": f"Observation {idx} reliability {val} is outside valid bounded range [0.0, 1.0].",
                    }

            return {
                "valid": True,
                "status": "PASS",
                "message": f"Successfully validated {len(obs_raw)} ground-truth observations.",
                "total_observations": len(obs_raw),
            }

        except Exception as e:
            return {"valid": False, "status": "FAIL", "message": f"Validation failed with exception: {e}"}

    def validate_ground_truth(self, observations: List[GroundTruthObservation]) -> Dict[str, Any]:
        """Validate an already loaded list of GroundTruthObservation objects."""
        if not observations:
            return {
                "valid": False,
                "status": "NOT AVAILABLE",
                "reason": "Ground-truth observation catalog is empty.",
                "total_observations": 0,
            }

        invalid_ranges = [o for o in observations if not (0.0 <= o.ground_truth_reliability <= 1.0)]
        if invalid_ranges:
            return {
                "valid": False,
                "status": "FAIL",
                "reason": f"{len(invalid_ranges)} observations have reliability outside [0.0, 1.0]",
                "total_observations": len(observations),
            }

        return {
            "valid": True,
            "status": "PASS",
            "total_observations": len(observations),
            "splits": list(set(o.split for o in observations)),
            "horizons": sorted(list(set(o.horizon for o in observations))),
        }

    def align_predictions_and_ground_truth(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[GroundTruthObservation],
        split: Optional[str] = "test",
        feature_metadata_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[List[AlignedEvaluationPair], List[Dict[str, Any]]]:
        """Match predictions to ground-truth observations with strict temporal and horizon alignment."""
        aligned_pairs: List[AlignedEvaluationPair] = []
        excluded_records: List[Dict[str, Any]] = []

        # Build ground-truth lookup maps
        # Primary key: (dataset_id, video_id, target_frame_index, horizon)
        # Fallback key: (dataset_id, sequence_id, horizon)
        gt_frame_map: Dict[Tuple[str, str, int, int], GroundTruthObservation] = {}
        gt_seq_map: Dict[Tuple[str, str, int], GroundTruthObservation] = {}
        duplicate_gt_keys: set = set()

        for gt in ground_truth:
            if split is not None and gt.split.lower() != split.lower():
                continue

            frame_key = (gt.dataset_id, gt.video_id, gt.target_frame_index, gt.horizon)
            if frame_key in gt_frame_map:
                duplicate_gt_keys.add(frame_key)
                excluded_records.append({
                    "reason": "DUPLICATE_GROUND_TRUTH",
                    "observation_id": gt.observation_id,
                    "frame_key": frame_key,
                })
            else:
                gt_frame_map[frame_key] = gt

            if gt.sequence_id:
                seq_key = (gt.dataset_id, gt.sequence_id, gt.horizon)
                gt_seq_map[seq_key] = gt

        # Global sequence split lookup
        gt_seq_split_map = {gt.sequence_id: gt.split.lower() for gt in ground_truth if gt.sequence_id}

        matched_gt_ids: set = set()
        seen_pred_keys: set = set()

        for pred in predictions:
            sid = pred.get("sequence_id")
            if split is not None:
                if "split" in pred and str(pred["split"]).lower() != split.lower():
                    continue
                if sid and sid in gt_seq_split_map and gt_seq_split_map[sid] != split.lower():
                    continue

            d_id = str(pred.get("dataset_id", "default_dataset"))
            v_id = str(pred.get("video_id", d_id))
            t_frame = pred.get("target_frame_index")
            if t_frame is None:
                t_frame = pred.get("target_source_frame_index")
            if t_frame is None:
                t_frame = 0
            target_frame = int(t_frame)
            horizon = int(pred.get("horizon", 1))

            pred_key = (d_id, v_id, target_frame, horizon)

            if pred_key in seen_pred_keys:
                excluded_records.append({
                    "reason": "DUPLICATE_PREDICTION",
                    "prediction_id": pred.get("prediction_id"),
                    "key": pred_key,
                })
                continue
            seen_pred_keys.add(pred_key)

            # Match against ground-truth
            matched_gt = gt_frame_map.get(pred_key)
            if matched_gt is None and sid:
                matched_gt = gt_seq_map.get((d_id, str(sid), horizon))

            if matched_gt is None:
                excluded_records.append({
                    "reason": "UNMATCHED_PREDICTION",
                    "prediction_id": pred.get("prediction_id"),
                    "key": pred_key,
                })
                continue

            matched_gt_ids.add(matched_gt.observation_id)

            p_val = float(pred["predicted_reliability"])
            g_val = float(matched_gt.ground_truth_reliability)

            if not np.isfinite(p_val) or not np.isfinite(g_val):
                excluded_records.append({
                    "reason": "NON_FINITE_VALUE",
                    "prediction_id": pred.get("prediction_id"),
                    "pred_val": p_val,
                    "gt_val": g_val,
                })
                continue

            pair = AlignedEvaluationPair(
                prediction_id=str(pred.get("prediction_id", f"pred_{len(aligned_pairs)}")),
                observation_id=matched_gt.observation_id,
                dataset_id=d_id,
                video_id=v_id,
                source_frame_index=pred.get("source_frame_index", pred.get("input_end_source_frame_index")),
                target_frame_index=target_frame,
                horizon=horizon,
                predicted_reliability=p_val,
                ground_truth_reliability=g_val,
                split=matched_gt.split,
                sequence_id=pred.get("sequence_id"),
                target_frame_reference=pred.get("target_frame_reference"),
                target_source_frame_index=pred.get("target_source_frame_index"),
                model_name=str(pred.get("model_name", "LSTMForecaster")),
                checkpoint_id=str(pred.get("checkpoint_id", "best_model.pt")),
                features_snapshot=pred.get("features_snapshot"),
            )
            aligned_pairs.append(pair)

        # Record unmatched ground truth
        for gt in ground_truth:
            if split is not None and gt.split.lower() != split.lower():
                continue
            if gt.observation_id not in matched_gt_ids:
                excluded_records.append({
                    "reason": "UNMATCHED_GROUND_TRUTH",
                    "observation_id": gt.observation_id,
                    "target_frame_index": gt.target_frame_index,
                    "horizon": gt.horizon,
                })

        return aligned_pairs, excluded_records

    # Alias for API compatibility
    align_predictions_and_targets = align_predictions_and_ground_truth

    def compute_horizon_metrics(
        self,
        pairs: List[AlignedEvaluationPair],
        horizon: int,
    ) -> HorizonEvaluationMetrics:
        """Compute standard regression metrics for a single forecast horizon."""
        filtered = [p for p in pairs if p.horizon == horizon]
        n = len(filtered)
        if n == 0:
            raise ModelEvaluationError(f"No pairs available for horizon {horizon}")

        errors = np.array([p.error for p in filtered], dtype=np.float64)
        abs_errors = np.array([p.absolute_error for p in filtered], dtype=np.float64)
        y_true = np.array([p.ground_truth_reliability for p in filtered], dtype=np.float64)
        y_pred = np.array([p.predicted_reliability for p in filtered], dtype=np.float64)

        mae = float(np.mean(abs_errors))
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        bias = float(np.mean(errors))
        median_ae = float(np.median(abs_errors))
        error_std = float(np.std(errors))

        # Safe R2 calculation
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum(errors ** 2))
        if ss_tot > 1e-12:
            r2 = float(1.0 - (ss_res / ss_tot))
        else:
            r2 = None

        mean_act = float(np.mean(y_true))
        mean_pred = float(np.mean(y_pred))
        act_std = float(np.std(y_true))
        pred_std = float(np.std(y_pred))
        min_p = float(np.min(y_pred))
        max_p = float(np.max(y_pred))
        min_a = float(np.min(y_true))
        max_a = float(np.max(y_true))

        # MAPE calculation where ground-truth > 1e-4
        valid_mapes = [p.percentage_error for p in filtered if p.ground_truth_reliability > 1e-4]
        mape = float(np.mean(valid_mapes)) if valid_mapes else 0.0

        under_count = int(np.sum(errors < -1e-6))
        over_count = int(np.sum(errors > 1e-6))
        exact_count = int(np.sum(abs_errors <= 1e-6))

        return HorizonEvaluationMetrics(
            horizon=horizon,
            sample_count=n,
            mae=mae,
            rmse=rmse,
            bias=bias,
            median_absolute_error=median_ae,
            error_std=error_std,
            mape=mape,
            underprediction_count=under_count,
            overprediction_count=over_count,
            exact_prediction_count=exact_count,
            exact_match_count=exact_count,
            min_error=float(np.min(errors)),
            max_error=float(np.max(errors)),
            mse=mse,
            r2=r2,
            mean_actual=mean_act,
            mean_predicted=mean_pred,
            actual_std=act_std,
            pred_std=pred_std,
            min_predicted=min_p,
            max_predicted=max_p,
            min_actual=min_a,
            max_actual=max_a,
        )

    def compute_all_horizon_metrics(
        self,
        pairs: List[AlignedEvaluationPair],
    ) -> Dict[str, HorizonEvaluationMetrics]:
        """Compute metrics for all configured horizons."""
        metrics: Dict[str, HorizonEvaluationMetrics] = {}
        for h in self.horizons:
            h_pairs = [p for p in pairs if p.horizon == h]
            if h_pairs:
                metrics[f"h{h}"] = self.compute_horizon_metrics(pairs, horizon=h)
        return metrics

    def compare_against_baselines(
        self,
        lstm_metrics: Dict[str, HorizonEvaluationMetrics],
        baseline_results_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
        split: Optional[str] = "test",
    ) -> List[Dict[str, Any]]:
        """Compare LSTM multi-horizon metrics against baseline models (Persistence, Mean, Trend)."""
        comparisons: List[Dict[str, Any]] = []

        baseline_data: Dict[str, Any] = {}
        if baseline_results_path is not None:
            if isinstance(baseline_results_path, (str, Path)):
                p = Path(baseline_results_path)
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        baseline_data = json.load(f)
            elif isinstance(baseline_results_path, dict):
                baseline_data = baseline_results_path

        strategies = baseline_data.get("strategies", {})
        if not strategies and "by_split" in baseline_data:
            split_key = split.lower() if split else "test"
            strategies = baseline_data.get("by_split", {}).get(split_key, {}).get("strategies", {})
        if not strategies and "overall" in baseline_data:
            strategies = baseline_data.get("overall", {}).get("strategies", {})

        for h_key, lstm_m in lstm_metrics.items():
            h_int = lstm_m.horizon

            # Add LSTM record
            comparisons.append({
                "model_name": "LSTMForecaster",
                "horizon": h_int,
                "sample_count": lstm_m.sample_count,
                "mae": round(lstm_m.mae, 6),
                "rmse": round(lstm_m.rmse, 6),
                "mse": round(lstm_m.mse, 8),
                "r2": round(lstm_m.r2, 6) if lstm_m.r2 is not None else None,
                "bias": round(lstm_m.bias, 6),
                "mean_predicted": round(lstm_m.mean_predicted, 6),
                "mean_actual": round(lstm_m.mean_actual, 6),
                "improvement_over_baseline_mae_pct": None,
                "interpretation": "Evaluated model",
                "status": "EVALUATED",
            })

            # Check each baseline strategy
            for strat_name, strat_horizons in strategies.items():
                if h_key in strat_horizons:
                    b_metrics = strat_horizons[h_key]
                    b_mae = b_metrics.get("mae")
                    b_rmse = b_metrics.get("rmse")
                    b_mse = round(b_rmse ** 2, 8) if b_rmse is not None else None

                    improvement_pct = None
                    interpretation = "Insufficient evidence"
                    if b_mae is not None:
                        if abs(b_mae) < 1e-6:
                            if lstm_m.mae <= 1e-6:
                                interpretation = "LSTM comparable to baseline"
                            else:
                                interpretation = "LSTM worse than baseline"
                        else:
                            improvement_pct = float(((b_mae - lstm_m.mae) / b_mae) * 100.0)
                            if improvement_pct > 5.0:
                                interpretation = "LSTM better than baseline"
                            elif improvement_pct < -5.0:
                                interpretation = "LSTM worse than baseline"
                            else:
                                interpretation = "LSTM comparable to baseline"

                    comparisons.append({
                        "model_name": f"Baseline_{strat_name.capitalize()}",
                        "horizon": h_int,
                        "sample_count": b_metrics.get("sample_count", lstm_m.sample_count),
                        "mae": b_mae,
                        "rmse": b_rmse,
                        "mse": b_mse,
                        "r2": b_metrics.get("r2"),
                        "bias": b_metrics.get("bias", 0.0),
                        "mean_predicted": b_metrics.get("mean_predicted"),
                        "mean_actual": b_metrics.get("mean_actual"),
                        "improvement_over_baseline_mae_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
                        "improvement_over_persistence_mae_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
                        "interpretation": interpretation,
                        "status": "EVALUATED",
                    })

        return comparisons

    def perform_error_analysis(
        self,
        pairs: List[AlignedEvaluationPair],
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical error distribution and breakdown analysis."""
        if not pairs:
            return {"status": "EMPTY", "message": "No aligned pairs available for error analysis"}

        all_errors = np.array([p.error for p in pairs])
        all_abs_errors = np.array([p.absolute_error for p in pairs])

        # 1. Overall Error Distribution
        dist = {
            "sample_count": len(pairs),
            "mean": float(np.mean(all_errors)),
            "std": float(np.std(all_errors)),
            "median": float(np.median(all_errors)),
            "min": float(np.min(all_errors)),
            "max": float(np.max(all_errors)),
            "percentiles": {
                "p5": float(np.percentile(all_errors, 5)),
                "p25": float(np.percentile(all_errors, 25)),
                "p50": float(np.percentile(all_errors, 50)),
                "p75": float(np.percentile(all_errors, 75)),
                "p95": float(np.percentile(all_errors, 95)),
            },
        }

        # 2. Per-Horizon Breakdown
        breakdown: Dict[str, Any] = {}
        for h in self.horizons:
            h_pairs = [p for p in pairs if p.horizon == h]
            if h_pairs:
                h_errs = np.array([p.error for p in h_pairs])
                breakdown[f"h{h}"] = {
                    "sample_count": len(h_pairs),
                    "mean_error": float(np.mean(h_errs)),
                    "std_error": float(np.std(h_errs)),
                    "mae": float(np.mean(np.abs(h_errs))),
                    "rmse": float(np.sqrt(np.mean(h_errs ** 2))),
                }

        # 3. High Error Cases
        mean_mae = float(np.mean(all_abs_errors))
        worst = sorted(pairs, key=lambda p: p.absolute_error, reverse=True)[:5]
        worst_cases = [
            {
                "prediction_id": p.prediction_id,
                "video_id": p.video_id,
                "target_frame_index": p.target_frame_index,
                "horizon": p.horizon,
                "predicted": p.predicted_reliability,
                "ground_truth": p.ground_truth_reliability,
                "error": p.error,
                "absolute_error": p.absolute_error,
            }
            for p in worst
        ]

        return {
            "overall_error_distribution": dist,
            "per_horizon_breakdown": breakdown,
            "mean_mae": mean_mae,
            "worst_error_cases": worst_cases,
        }

    def save_evaluation_artifacts(
        self,
        output_dir: Union[str, Path],
        gt_validation: Dict[str, Any],
        metrics: Dict[str, HorizonEvaluationMetrics],
        comparisons: List[Dict[str, Any]],
        error_analysis: Dict[str, Any],
        excluded_records: List[Dict[str, Any]],
        aligned_pairs: List[AlignedEvaluationPair],
        split: Optional[str] = "test",
    ) -> Dict[str, Path]:
        """Save comprehensive evaluation artifacts in structured JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Path] = {}

        # 1. Horizon-specific metrics
        for h_key, h_metric in metrics.items():
            f_name = f"horizon_{h_metric.horizon}_metrics.json"
            f_path = out_path / f_name
            with open(f_path, "w", encoding="utf-8") as f:
                json.dump(h_metric.to_dict(), f, indent=2)
            artifacts[f"horizon_metrics_{h_key}"] = f_path

        # 2. Evaluation results (aligned pairs)
        p_eval_results = out_path / "evaluation_results.json"
        with open(p_eval_results, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in aligned_pairs], f, indent=2)
        artifacts["evaluation_results"] = p_eval_results

        # 3. Baseline comparison
        p_baseline_comp = out_path / "baseline_comparison.json"
        with open(p_baseline_comp, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2)
        artifacts["baseline_comparison"] = p_baseline_comp

        # 4. Error analysis
        p_err_analysis = out_path / "error_analysis.json"
        with open(p_err_analysis, "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2)
        artifacts["error_analysis"] = p_err_analysis

        # 5. Excluded records
        p_excluded = out_path / "excluded_records.json"
        with open(p_excluded, "w", encoding="utf-8") as f:
            json.dump(excluded_records, f, indent=2)
        artifacts["excluded_records"] = p_excluded

        # 6. Evaluation summary
        summary = {
            "metadata": {
                "evaluation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "evaluator_version": "1.0",
                "split": split,
                "horizons": self.horizons,
                "python_version": platform.python_version(),
            },
            "evaluation_status": "COMPLETED",
            "ground_truth_validation": gt_validation,
            "total_aligned_pairs": len(aligned_pairs),
            "total_excluded_records": len(excluded_records),
            "metrics": {k: m.to_dict() for k, m in metrics.items()},
        }
        p_summary = out_path / "evaluation_summary.json"
        with open(p_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        artifacts["evaluation_summary"] = p_summary

        return artifacts


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for model evaluator."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI - Forecasting Model Evaluation & Analysis"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="artifacts/predictions/multi_horizon_predictions.json",
        help="Path to prediction records JSON.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="artifacts/ground_truth/ground_truth.json",
        help="Path to ground-truth observations JSON.",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default="artifacts/baseline_forecasting/baseline_experiment_results.json",
        help="Path to baseline experiment results JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate ('test', 'val', 'train', or 'all').",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Forecasting horizons to evaluate (default: 1 3 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/evaluation",
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate Matplotlib evaluation plots if ground-truth is available.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Optional custom directory to save generated evaluation plots (defaults to <output>/plots).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args(args)


def run_evaluation_cli(args: argparse.Namespace) -> int:
    """Execute model evaluation CLI pipeline."""
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    print("=" * 78)
    print("  PERCEPTGUARD AI - FORECASTING MODEL EVALUATION & ANALYSIS")
    print("=" * 78)

    # 1. Load configuration
    cfg = None
    if Path(args.config).exists():
        try:
            cfg = ForecastingConfig.from_yaml(args.config)
            print(f"[CONFIG] Loaded valid configuration from: {args.config}")
        except Exception as e:
            print(f"[WARN] Failed to load config from {args.config}: {e}. Using defaults.")

    evaluator = ForecastingModelEvaluator(config=cfg, horizons=args.horizons)

    # 2. Check Ground Truth availability
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"[GATE CHECK] Ground-truth file not found: {gt_path}")
        print("[STATUS] Ground truth for future detection reliability is NOT AVAILABLE.")
        print("[ACTION] Halting evaluation cleanly. No fabricated metrics generated.")
        return 2

    # Validate ground truth contract
    gt_val = evaluator.validate_ground_truth_contract(gt_path)
    if not gt_val.get("valid", False):
        print(f"[GATE CHECK] Ground truth validation status: {gt_val.get('status')}")
        print(f"[REASON] {gt_val.get('message')}")
        if gt_val.get("status") == "NOT_AVAILABLE":
            print("[STATUS] Ground truth is marked NOT AVAILABLE. Clean exit.")
            return 2
        return 1

    # 3. Load predictions & ground truth
    try:
        preds = evaluator.load_predictions(args.predictions)
        gt_obs = evaluator.load_ground_truth(gt_path)
    except Exception as e:
        print(f"[ERROR] Failed to load evaluation datasets: {e}")
        return 1

    # 4. Alignment
    split_filter = None if args.split.lower() == "all" else args.split.lower()
    aligned_pairs, excluded = evaluator.align_predictions_and_ground_truth(
        predictions=preds,
        ground_truth=gt_obs,
        split=split_filter,
    )
    print(f"[ALIGNMENT] Successfully aligned {len(aligned_pairs)} pairs. Excluded {len(excluded)} records.")

    if not aligned_pairs:
        print("[WARN] Zero pairs matched between predictions and ground-truth observations.")
        return 1

    # 5. Compute metrics
    metrics = evaluator.compute_all_horizon_metrics(aligned_pairs)
    comparisons = evaluator.compare_against_baselines(metrics, args.baseline_results, split=args.split)
    error_analysis = evaluator.perform_error_analysis(aligned_pairs)

    # 6. Save artifacts
    artifacts = evaluator.save_evaluation_artifacts(
        output_dir=args.output,
        gt_validation=gt_val,
        metrics=metrics,
        comparisons=comparisons,
        error_analysis=error_analysis,
        excluded_records=excluded,
        aligned_pairs=aligned_pairs,
        split=args.split,
    )

    print("-" * 78)
    print("  EVALUATION RESULTS SUMMARY")
    print("-" * 78)
    for h_key, h_metric in metrics.items():
        print(
            f"Horizon {h_metric.horizon} (N={h_metric.sample_count}) | "
            f"MAE: {h_metric.mae:.4f} | "
            f"RMSE: {h_metric.rmse:.4f} | "
            f"MSE: {h_metric.mse:.6f} | "
            f"Bias: {h_metric.bias:.4f} | "
            f"MedAE: {h_metric.median_absolute_error:.4f}"
        )

    # 7. Generate plots if requested
    if args.generate_plots:
        try:
            from src.forecasting.evaluation_visualizer import ForecastingEvaluationVisualizer
            visualizer = ForecastingEvaluationVisualizer()
            plot_dir = Path(args.plot_dir) if args.plot_dir else Path(artifacts["evaluation_summary"]).parent / "plots"
            plots = visualizer.generate_all_plots(
                pairs=aligned_pairs,
                metrics=metrics,
                comparison_records=comparisons,
                output_dir=plot_dir,
            )
            t6_plots = visualizer.generate_task6_visualizations(
                pairs=aligned_pairs,
                metrics=metrics,
                comparison_records=comparisons,
                output_dir=plot_dir,
            )
            plots.update(t6_plots)
            print(f"[VISUALIZATION] Generated {len(plots)} evaluation charts under: {plot_dir}")
        except Exception as e:
            print(f"[WARN] Failed to generate evaluation plots: {e}")

    print("=" * 78)
    print("  EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 78)
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """CLI entrypoint returning status code."""
    parsed_args = parse_args(args)
    return run_evaluation_cli(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
