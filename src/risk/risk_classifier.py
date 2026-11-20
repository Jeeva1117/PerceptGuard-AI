"""Risk Classification Engine for PerceptGuard AI.

Maps continuous multi-horizon failure risk scores into discrete, deterministic
operational risk tiers (LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN) adhering to
configurable threshold profiles and boundary policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from src.early_warning.interfaces import IRiskClassifier, RiskLevel
from src.risk.risk_config import ClassificationConfig, RiskConfig, ThresholdProfile
from src.risk.risk_models import AggregatedRiskRecord, HorizonRiskRecord
from src.risk.risk_validator import RiskValidationError, validate_risk_value


@dataclass
class ClassificationResult:
    """Structured result of operational risk tier categorization."""
    sequence_id: str
    risk_score: float
    risk_level: RiskLevel
    horizon: Optional[int] = None
    prediction_id: Optional[str] = None
    source_frame_id: Optional[str] = None
    target_frame_id: Optional[str] = None
    target_timestamp_seconds: Optional[float] = None
    thresholds_applied: Dict[str, float] = field(default_factory=dict)
    calibration_status: str = "PROVISIONAL"
    boundary_policy: str = "closed_right"
    is_stale: bool = False
    is_missing: bool = False
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert classification result to dictionary."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d


class RiskClassifier(IRiskClassifier):
    """Deterministic perception failure risk tier classifier.

    Evaluates individual horizon risk records or aggregated multi-horizon sequence
    scores against configured threshold boundaries under strict edge-case safety.
    """

    def __init__(
        self,
        config: Optional[Union[ClassificationConfig, RiskConfig]] = None,
        boundary_policy: Optional[str] = None,
    ):
        if isinstance(config, RiskConfig):
            base_cfg = config.classification
        elif isinstance(config, ClassificationConfig):
            base_cfg = config
        else:
            base_cfg = ClassificationConfig()

        if boundary_policy is not None:
            self.config = ClassificationConfig(
                boundary_policy=boundary_policy,
                default_thresholds=base_cfg.default_thresholds,
                horizon_thresholds=base_cfg.horizon_thresholds,
                missing_data_level=base_cfg.missing_data_level,
                stale_data_level=base_cfg.stale_data_level,
            )
        else:
            self.config = base_cfg

    def get_threshold_profile(self, horizon: Optional[int] = None) -> ThresholdProfile:
        """Retrieve threshold profile for specified horizon, falling back to default."""
        if horizon is not None and horizon in self.config.horizon_thresholds:
            return self.config.horizon_thresholds[horizon]
        return self.config.default_thresholds

    def classify_score(
        self,
        risk_score: Any,
        horizon: Optional[int] = None,
        profile: Optional[ThresholdProfile] = None,
        is_stale: bool = False,
        is_missing: bool = False,
    ) -> RiskLevel:
        """Map raw risk score to categorical RiskLevel under strict boundary conventions.

        Parameters
        ----------
        risk_score : Any
            Continuous risk score in [0.0, 1.0].
        horizon : Optional[int]
            Forecast horizon (1, 3, 5) or None for aggregated.
        profile : Optional[ThresholdProfile]
            Direct threshold profile override if provided.
        is_stale : bool
            Flag indicating prediction exceeded freshness TTL.
        is_missing : bool
            Flag indicating prediction was unavailable/missing.
        """
        # Critical Safety Rule: Never classify missing or stale data as LOW!
        if is_missing or is_stale or risk_score is None:
            return RiskLevel.UNKNOWN

        # Validate finite range bounds in [0.0, 1.0]
        score = validate_risk_value(risk_score)

        active_profile = profile if profile is not None else self.get_threshold_profile(horizon)
        low = active_profile.low_upper
        med = active_profile.medium_upper
        high = active_profile.high_upper

        policy = self.config.boundary_policy

        if policy == "closed_right":
            # (0.0 <= s <= low): LOW
            # (low < s <= med): MEDIUM
            # (med < s <= high): HIGH
            # (s > high): CRITICAL
            if score <= low:
                return RiskLevel.LOW
            elif score <= med:
                return RiskLevel.MEDIUM
            elif score <= high:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
        elif policy == "closed_left":
            # (s < low): LOW
            # (low <= s < med): MEDIUM
            # (med <= s < high): HIGH
            # (s >= high): CRITICAL
            if score < low:
                return RiskLevel.LOW
            elif score < med:
                return RiskLevel.MEDIUM
            elif score < high:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
        else:
            raise RiskValidationError(f"Unsupported boundary policy: {policy}")

    # IRiskClassifier interface method
    def classify(self, risk_score: float) -> RiskLevel:
        """Interface method classifying a scalar risk score using default thresholds."""
        return self.classify_score(risk_score)

    def classify_missing(
        self,
        sequence_id: str = "",
        horizon: Optional[int] = None,
        reason: str = "Missing prediction data",
    ) -> ClassificationResult:
        """Explicitly classify missing prediction as UNKNOWN (never LOW)."""
        profile = self.get_threshold_profile(horizon)
        return ClassificationResult(
            sequence_id=sequence_id,
            horizon=horizon,
            risk_score=float("nan"),
            risk_level=RiskLevel.UNKNOWN,
            thresholds_applied={
                "low_upper": profile.low_upper,
                "medium_upper": profile.medium_upper,
                "high_upper": profile.high_upper,
            },
            calibration_status=profile.calibration_status,
            boundary_policy=self.config.boundary_policy,
            is_stale=False,
            is_missing=True,
            notes=f"Missing data: {reason}",
        )

    def classify_stale(
        self,
        sequence_id: str = "",
        horizon: Optional[int] = None,
        stale_score: float = 0.0,
        staleness_seconds: float = 0.0,
    ) -> ClassificationResult:
        """Explicitly classify stale prediction as UNKNOWN (never LOW)."""
        profile = self.get_threshold_profile(horizon)
        return ClassificationResult(
            sequence_id=sequence_id,
            horizon=horizon,
            risk_score=stale_score,
            risk_level=RiskLevel.UNKNOWN,
            thresholds_applied={
                "low_upper": profile.low_upper,
                "medium_upper": profile.medium_upper,
                "high_upper": profile.high_upper,
            },
            calibration_status=profile.calibration_status,
            boundary_policy=self.config.boundary_policy,
            is_stale=True,
            is_missing=False,
            notes=f"Stale data (age: {staleness_seconds:.3f}s)",
        )

    def classify_record(
        self,
        record: HorizonRiskRecord,
        is_stale: bool = False,
        is_missing: bool = False,
    ) -> ClassificationResult:
        """Classify an individual HorizonRiskRecord."""
        is_valid = getattr(record, "is_valid", True)
        validation_notes = getattr(record, "validation_notes", "")
        effective_missing = is_missing or (not is_valid)
        level = self.classify_score(
            record.risk_score if not effective_missing else None,
            horizon=record.horizon,
            is_stale=is_stale,
            is_missing=effective_missing,
        )
        profile = self.get_threshold_profile(record.horizon)

        notes = profile.notes
        if not is_valid and validation_notes:
            notes = f"Invalid record: {validation_notes}"

        return ClassificationResult(
            sequence_id=record.sequence_id,
            horizon=record.horizon,
            prediction_id=record.prediction_id,
            risk_score=record.risk_score if not effective_missing else float("nan"),
            risk_level=level,
            source_frame_id=record.source_frame_id,
            target_frame_id=record.target_frame_id,
            target_timestamp_seconds=record.target_timestamp_seconds,
            thresholds_applied={
                "low_upper": profile.low_upper,
                "medium_upper": profile.medium_upper,
                "high_upper": profile.high_upper,
            },
            calibration_status=profile.calibration_status,
            boundary_policy=self.config.boundary_policy,
            is_stale=is_stale,
            is_missing=effective_missing,
            notes=notes,
        )

    # Alias for convenience
    classify_horizon_record = classify_record

    def classify_aggregated(
        self,
        record: AggregatedRiskRecord,
        is_stale: bool = False,
        is_missing: bool = False,
    ) -> ClassificationResult:
        """Classify a synthesized multi-horizon AggregatedRiskRecord."""
        is_valid = getattr(record, "is_valid", True)
        validation_notes = getattr(record, "validation_notes", "")
        effective_missing = is_missing or (not is_valid)
        level = self.classify_score(
            record.aggregated_risk_score if not effective_missing else None,
            horizon=record.primary_horizon,
            is_stale=is_stale,
            is_missing=effective_missing,
        )
        profile = self.get_threshold_profile(record.primary_horizon)

        notes = f"Aggregated score ({record.aggregation_strategy}). {profile.notes}"
        if not is_valid and validation_notes:
            notes = f"Invalid record: {validation_notes}"

        return ClassificationResult(
            sequence_id=record.sequence_id,
            horizon=record.primary_horizon,
            risk_score=record.aggregated_risk_score if not effective_missing else float("nan"),
            risk_level=level,
            source_frame_id=record.source_frame_id,
            target_timestamp_seconds=record.target_timestamp_seconds,
            thresholds_applied={
                "low_upper": profile.low_upper,
                "medium_upper": profile.medium_upper,
                "high_upper": profile.high_upper,
            },
            calibration_status=profile.calibration_status,
            boundary_policy=self.config.boundary_policy,
            is_stale=is_stale,
            is_missing=effective_missing,
            notes=notes,
        )

    # Alias for convenience
    classify_aggregated_record = classify_aggregated

    def classify_batch(
        self,
        records: Sequence[HorizonRiskRecord],
        is_stale: bool = False,
    ) -> List[ClassificationResult]:
        """Classify a batch of HorizonRiskRecords."""
        return [self.classify_record(r, is_stale=is_stale) for r in records]
