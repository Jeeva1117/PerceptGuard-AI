"""Baseline Evaluation Metrics Module for PerceptGuard AI.

Implements regression metrics (MAE, RMSE) for multi-horizon perception reliability
forecasting, with strict data validation, array dimension checks, and missing-target handling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


class MetricCalculationError(ValueError):
    """Raised when metric inputs fail validation checks."""
    pass


def _validate_metric_inputs(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to 1D finite float64 numpy arrays."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_t.size == 0 or y_p.size == 0:
        raise MetricCalculationError("Cannot calculate metrics on empty input arrays")

    if y_t.size != y_p.size:
        raise MetricCalculationError(
            f"Array length mismatch: y_true has {y_t.size} elements, y_pred has {y_p.size} elements"
        )

    if not np.isfinite(y_t).all():
        n_nans = int(np.isnan(y_t).sum())
        n_infs = int(np.isinf(y_t).sum())
        raise MetricCalculationError(
            f"y_true contains non-finite values (NaNs: {n_nans}, Infs: {n_infs})"
        )

    if not np.isfinite(y_p).all():
        n_nans = int(np.isnan(y_p).sum())
        n_infs = int(np.isinf(y_p).sum())
        raise MetricCalculationError(
            f"y_pred contains non-finite values (NaNs: {n_nans}, Infs: {n_infs})"
        )

    return y_t, y_p


def calculate_mae(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
) -> float:
    """Calculate Mean Absolute Error (MAE).
    
    MAE = (1 / N) * sum(|y_true - y_pred|)
    
    Args:
        y_true: Ground-truth target values.
        y_pred: Model predicted values.
        
    Returns:
        MAE as a float.
    """
    y_t, y_p = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_rmse(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
) -> float:
    """Calculate Root Mean Squared Error (RMSE).
    
    RMSE = sqrt((1 / N) * sum((y_true - y_pred)^2))
    
    Args:
        y_true: Ground-truth target values.
        y_pred: Model predicted values.
        
    Returns:
        RMSE as a float.
    """
    y_t, y_p = _validate_metric_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def calculate_metrics_summary(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    horizon: Optional[int] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate standard evaluation metrics summary dictionary."""
    y_t, y_p = _validate_metric_inputs(y_true, y_pred)
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    res: Dict[str, Any] = {
        "sample_count": int(y_t.size),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mean_actual": round(float(np.mean(y_t)), 6),
        "mean_predicted": round(float(np.mean(y_p)), 6),
        "min_error": round(float(np.min(np.abs(y_t - y_p))), 6),
        "max_error": round(float(np.max(np.abs(y_t - y_p))), 6),
    }
    if horizon is not None:
        res["horizon"] = int(horizon)
    if strategy is not None:
        res["strategy"] = str(strategy)
    return res


def evaluate_predictions_by_horizon(
    predictions: List[Dict[str, Any]],
    horizons: Sequence[int] = (1, 3, 5),
    strategies: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Aggregate and evaluate multi-horizon predictions.
    
    Groups prediction records by strategy and horizon, filters valid pairs,
    and calculates metrics. If actual targets are absent, reports status honestly.
    
    Args:
        predictions: List of prediction dictionaries conforming to prediction contract.
        horizons: Target horizons to evaluate.
        strategies: Optional list of strategies to filter by.
        
    Returns:
        Hierarchical metrics dictionary.
    """
    if not predictions:
        return {
            "status": "EMPTY",
            "message": "No predictions provided for evaluation",
            "horizons": {},
        }

    # Discover strategies if not provided
    if strategies is None:
        detected_strats = sorted(list({p.get("baseline_strategy", "unknown") for p in predictions}))
    else:
        detected_strats = list(strategies)

    results: Dict[str, Any] = {
        "status": "PASS",
        "total_predictions": len(predictions),
        "strategies": {},
    }

    for strat in detected_strats:
        strat_preds = [p for p in predictions if p.get("baseline_strategy") == strat]
        results["strategies"][strat] = {}

        for h in horizons:
            h_preds = [p for p in strat_preds if p.get("target_horizon") == h]
            valid_pairs = [
                (p["actual_reliability"], p["predicted_reliability"])
                for p in h_preds
                if p.get("actual_reliability") is not None
                and p.get("predicted_reliability") is not None
            ]

            if not valid_pairs:
                results["strategies"][strat][f"h{h}"] = {
                    "status": "NOT AVAILABLE",
                    "reason": "No ground-truth actual_reliability values present in prediction records",
                    "total_records": len(h_preds),
                    "valid_pairs": 0,
                    "mae": None,
                    "rmse": None,
                }
            else:
                y_t = [pair[0] for pair in valid_pairs]
                y_p = [pair[1] for pair in valid_pairs]
                try:
                    summary = calculate_metrics_summary(y_t, y_p, horizon=h, strategy=strat)
                    summary["status"] = "PASS"
                    summary["total_records"] = len(h_preds)
                    summary["valid_pairs"] = len(valid_pairs)
                    results["strategies"][strat][f"h{h}"] = summary
                except MetricCalculationError as exc:
                    results["strategies"][strat][f"h{h}"] = {
                        "status": "ERROR",
                        "reason": str(exc),
                        "total_records": len(h_preds),
                        "valid_pairs": len(valid_pairs),
                        "mae": None,
                        "rmse": None,
                    }

    return results
