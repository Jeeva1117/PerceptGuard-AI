"""Reliability Risk Scoring Engine for PerceptGuard AI.

Implements the core transformation from multi-horizon perception reliability
forecasts into mathematically bounded, traceable perception failure risk scores.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.early_warning.interfaces import IRiskAssessmentEngine, RiskLevel, RiskScore
from src.risk.risk_config import RiskConfig
from src.risk.risk_features import TemporalRiskFeatureExtractor
from src.risk.risk_models import (
    AggregatedRiskRecord,
    HorizonRiskRecord,
    TemporalRiskFeatures,
)
from src.risk.risk_validator import (
    RiskValidationError,
    validate_aggregation_weights,
    validate_horizon,
    validate_reliability_value,
    validate_risk_value,
)


class RiskScoringError(ValueError):
    """Raised when risk scoring execution or aggregation fails."""
    pass


class ReliabilityRiskScorer(IRiskAssessmentEngine):
    """Production-grade scoring engine mapping reliability forecasts to failure risk.
    
    Transforms continuous reliability predictions in [0.0, 1.0] to derived risk
    scores in [0.0, 1.0], computes temporal trend features, and executes configurable
    multi-horizon aggregation strategies.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.feature_extractor = TemporalRiskFeatureExtractor(self.config.temporal)

    def transform_reliability_to_risk(self, reliability: float) -> float:
        """Map predicted reliability to a derived risk score using configured method.
        
        Parameters
        ----------
        reliability : float
            Predicted detection reliability score in [0.0, 1.0].
            
        Returns
        -------
        float
            Derived failure risk score in [0.0, 1.0], where 1.0 indicates maximum risk.
        """
        rel = validate_reliability_value(reliability)

        if self.config.transformation.method == "linear_inverse":
            risk = 1.0 - rel
        else:
            raise RiskScoringError(f"Unsupported transformation method: {self.config.transformation.method}")

        if self.config.transformation.clip_output:
            risk = max(
                self.config.transformation.min_risk,
                min(self.config.transformation.max_risk, risk),
            )

        rounded_risk = round(risk, self.config.precision_decimals)
        return validate_risk_value(
            rounded_risk,
            min_val=self.config.transformation.min_risk,
            max_val=self.config.transformation.max_risk,
        )

    def score_single_horizon(
        self, record: Union[EarlyWarningInputRecord, Dict[str, Any]]
    ) -> HorizonRiskRecord:
        """Score an individual single-horizon prediction record."""
        if isinstance(record, dict):
            rec = EarlyWarningInputRecord.from_dict(record)
        elif isinstance(record, EarlyWarningInputRecord):
            rec = record
        else:
            raise RiskScoringError(f"Unsupported input record type: {type(record).__name__}")

        h = validate_horizon(rec.horizon, self.config.supported_horizons)
        risk = self.transform_reliability_to_risk(rec.predicted_reliability)

        return HorizonRiskRecord(
            prediction_id=rec.prediction_id,
            sequence_id=rec.sequence_id,
            source_frame_id=rec.source_frame_id,
            target_frame_id=rec.target_frame_id,
            horizon=h,
            predicted_reliability=rec.predicted_reliability,
            risk_score=risk,
            source_frame_index=rec.source_frame_index,
            target_frame_index=rec.target_frame_index,
            target_timestamp_seconds=rec.target_timestamp_seconds,
            track_id=rec.track_id,
            model_name=rec.model_name,
            checkpoint_id=rec.checkpoint_id,
            experiment_id=rec.experiment_id,
            transformation_method=self.config.transformation.method,
        )

    def score_multi_horizon(
        self,
        records: Sequence[Union[EarlyWarningInputRecord, Dict[str, Any]]],
        history: Optional[Sequence[Any]] = None,
    ) -> AggregatedRiskRecord:
        """Score and aggregate multi-horizon predictions for a single sequence.
        
        Parameters
        ----------
        records : Sequence[Union[EarlyWarningInputRecord, Dict[str, Any]]]
            Predictions across horizons (e.g. H1, H3, H5) for the same sequence.
        history : Optional[Sequence[Any]]
            Retrospective prediction history prior to current sequence.
        """
        if not records:
            raise RiskScoringError("Cannot score empty multi-horizon records sequence.")

        horizon_records: Dict[int, HorizonRiskRecord] = {}
        horizon_reliabilities: Dict[int, float] = {}
        horizon_risks: Dict[int, float] = {}

        sequence_id: Optional[str] = None
        source_frame_id: Optional[str] = None
        target_timestamp: Optional[float] = None
        source_frame_index: Optional[int] = None
        track_id: Optional[int] = None
        model_name: str = "LSTMForecaster"
        checkpoint_id: str = "unknown"

        for raw_item in records:
            scored_item = self.score_single_horizon(raw_item)
            h = scored_item.horizon
            horizon_records[h] = scored_item
            horizon_reliabilities[h] = scored_item.predicted_reliability
            horizon_risks[h] = scored_item.risk_score

            if sequence_id is None:
                sequence_id = scored_item.sequence_id
                source_frame_id = scored_item.source_frame_id
                target_timestamp = scored_item.target_timestamp_seconds
                source_frame_index = scored_item.source_frame_index
                track_id = scored_item.track_id
                model_name = scored_item.model_name
                checkpoint_id = scored_item.checkpoint_id
            elif sequence_id != scored_item.sequence_id:
                raise RiskScoringError(
                    f"Multi-horizon records contain mismatched sequence IDs: {sequence_id} vs {scored_item.sequence_id}"
                )

        available_horizons = sorted(list(horizon_risks.keys()))
        if len(available_horizons) < self.config.aggregation.min_required_horizons:
            raise RiskScoringError(
                f"Available horizons ({len(available_horizons)}) is less than min_required_horizons "
                f"({self.config.aggregation.min_required_horizons})"
            )

        # Execute aggregation strategy
        strategy = self.config.aggregation.strategy
        weights_applied: Dict[int, float] = {}

        if strategy == "weighted_linear":
            weights = validate_aggregation_weights(
                self.config.aggregation.horizon_weights, available_horizons
            )
            total_weight = sum(weights.values())
            weights_applied = {h: w / total_weight for h, w in weights.items()}
            raw_agg_risk = sum(weights_applied[h] * horizon_risks[h] for h in available_horizons)
        elif strategy == "worst_case":
            raw_agg_risk = max(horizon_risks.values())
            weights_applied = {h: 1.0 if horizon_risks[h] == raw_agg_risk else 0.0 for h in available_horizons}
        elif strategy == "primary_horizon":
            target_h = self.config.aggregation.primary_horizon
            if target_h in horizon_risks:
                raw_agg_risk = horizon_risks[target_h]
                weights_applied = {h: 1.0 if h == target_h else 0.0 for h in available_horizons}
            else:
                # Fallback to nearest available horizon
                nearest_h = min(available_horizons, key=lambda x: abs(x - target_h))
                raw_agg_risk = horizon_risks[nearest_h]
                weights_applied = {h: 1.0 if h == nearest_h else 0.0 for h in available_horizons}
        elif strategy == "none":
            # Primary horizon without weighting
            target_h = self.config.aggregation.primary_horizon
            primary_h = target_h if target_h in horizon_risks else available_horizons[0]
            raw_agg_risk = horizon_risks[primary_h]
            weights_applied = {primary_h: 1.0}
        else:
            raise RiskScoringError(f"Unsupported aggregation strategy: {strategy}")

        agg_risk = round(raw_agg_risk, self.config.precision_decimals)
        agg_risk = validate_risk_value(agg_risk)

        # Extract temporal features
        temporal_features = self.feature_extractor.extract_features(
            sequence_id=sequence_id or "unknown",
            horizon_reliabilities=horizon_reliabilities,
            history_records=history,
        )

        return AggregatedRiskRecord(
            sequence_id=sequence_id or "unknown",
            source_frame_id=source_frame_id or "unknown",
            aggregated_risk_score=agg_risk,
            aggregation_strategy=strategy,
            primary_horizon=self.config.aggregation.primary_horizon,
            horizon_risk_scores=horizon_risks,
            horizon_reliabilities=horizon_reliabilities,
            weights_applied=weights_applied,
            target_timestamp_seconds=target_timestamp,
            source_frame_index=source_frame_index,
            track_id=track_id,
            temporal_features=temporal_features,
            is_calibrated_probability=False,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
        )

    # Implementation of IRiskAssessmentEngine abstract method
    def evaluate_risk(
        self,
        predictions: Sequence[EarlyWarningInputRecord],
        history: Optional[Sequence[EarlyWarningInputRecord]] = None,
    ) -> RiskScore:
        """Interface method producing standard RiskScore for the Early Warning pipeline."""
        agg = self.score_multi_horizon(predictions, history=history)

        # In Task 2, risk level is provisional UNKNOWN until Task 3 calibration
        return RiskScore(
            sequence_id=agg.sequence_id,
            overall_risk_score=agg.aggregated_risk_score,
            primary_horizon=agg.primary_horizon,
            risk_level=RiskLevel.UNKNOWN,
            horizon_risk_scores={f"H{h}": score for h, score in agg.horizon_risk_scores.items()},
            predicted_reliabilities={f"H{h}": rel for h, rel in agg.horizon_reliabilities.items()},
            degradation_slope=agg.temporal_features.historical_trend_slope if agg.temporal_features else None,
            target_timestamp_seconds=agg.target_timestamp_seconds,
            is_missing_data=False,
            details={
                "aggregation_strategy": agg.aggregation_strategy,
                "weights_applied": agg.weights_applied,
                "is_calibrated_probability": agg.is_calibrated_probability,
            },
        )
