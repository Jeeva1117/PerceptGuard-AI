"""Temporal Risk Monitoring Engine for PerceptGuard AI.

Provides chronological risk tracking, trend analysis, rolling window statistics,
state-transition detection, and multi-horizon isolation under strict zero-leakage,
anti-lookahead, and CPU-efficient invariants.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from src.early_warning.interfaces import (
    ITemporalRiskMonitor,
    RiskLevel,
    RiskScore,
)
if TYPE_CHECKING:
    from src.risk.risk_classifier import ClassificationResult, RiskClassifier
from src.risk.risk_config import RiskConfig, TemporalMonitoringConfig
from src.risk.risk_models import AggregatedRiskRecord, HorizonRiskRecord
from src.risk.risk_validator import RiskValidationError, validate_risk_value


class TemporalMonitoringError(ValueError):
    """Raised when monitoring data violates chronological or contract integrity."""
    pass


class RiskTrendDirection(str, Enum):
    """Directional trend classification for perception risk change over time."""
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str) -> RiskTrendDirection:
        val = str(value).strip().upper()
        try:
            return cls(val)
        except ValueError:
            return cls.UNKNOWN


@dataclass
class MonitoredRiskRecord:
    """Individual timestamped risk observation tracked in sequence history."""
    sequence_id: str
    risk_score: float  # In [0.0, 1.0], or NaN if missing
    risk_level: RiskLevel
    timestamp_seconds: float  # Event time (video/sensor time)
    horizon: Optional[int] = None  # 1, 3, 5, or None for sequence-aggregated
    source_frame_id: str = ""
    source_frame_index: Optional[int] = None
    target_frame_id: Optional[str] = None
    target_frame_index: Optional[int] = None
    forecast_generation_timestamp: Optional[str] = None  # Wall-clock prediction time
    prediction_id: Optional[str] = None
    calibration_status: str = "PROVISIONAL"
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "unknown"
    is_valid: bool = True
    is_stale: bool = False
    is_out_of_order: bool = False
    is_duplicate: bool = False
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d


@dataclass
class StateTransitionEvent:
    """Discrete operational risk level transition event."""
    sequence_id: str
    horizon: Optional[int]
    previous_level: RiskLevel
    current_level: RiskLevel
    timestamp_seconds: float
    risk_score: float
    previous_risk_score: Optional[float] = None
    is_genuine_transition: bool = True  # False if triggered by missing/unknown status
    calibration_status: str = "PROVISIONAL"
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition event to dictionary representation."""
        d = asdict(self)
        d["previous_level"] = self.previous_level.value
        d["current_level"] = self.current_level.value
        return d


@dataclass
class TemporalRiskSummary:
    """Instantaneous snapshot of rolling metrics, trends, and stability indicators."""
    sequence_id: str
    horizon: Optional[int]
    current_score: float
    previous_score: Optional[float]
    absolute_change: Optional[float]
    relative_change: Optional[float]
    rolling_mean: float
    rolling_max: float
    rolling_min: float
    volatility: Optional[float]  # Rolling standard deviation over window
    trend_direction: RiskTrendDirection
    consecutive_increasing_count: int
    current_risk_level: RiskLevel
    level_history: List[RiskLevel]
    observation_count: int
    window_size: int
    is_stale: bool = False
    has_timestamp_gap: bool = False
    last_timestamp_seconds: float = 0.0
    checkpoint_id: str = "unknown"
    calibration_status: str = "PROVISIONAL"
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary representation."""
        d = asdict(self)
        d["trend_direction"] = self.trend_direction.value
        d["current_risk_level"] = self.current_risk_level.value
        d["level_history"] = [lvl.value for lvl in self.level_history]
        return d


class SequenceHorizonTracker:
    """Internal chronological state tracker for a specific (sequence_id, horizon) pair.

    Encapsulates rolling deques, incremental statistical accumulators, and transition
    event logs with zero lookahead and zero cross-sequence contamination.
    """

    def __init__(self, sequence_id: str, horizon: Optional[int], config: TemporalMonitoringConfig):
        self.sequence_id = sequence_id
        self.horizon = horizon
        self.config = config
        self.records: deque[MonitoredRiskRecord] = deque(maxlen=config.history_window_size)
        self.transitions: List[StateTransitionEvent] = []
        self.consecutive_increasing: int = 0
        self.last_timestamp: Optional[float] = None
        self.last_frame_index: Optional[int] = None
        self.last_checkpoint_id: Optional[str] = None
        self.checkpoint_change_detected: bool = False

    def record_observation(self, record: MonitoredRiskRecord) -> TemporalRiskSummary:
        """Process an incremental observation and update running chronological metrics."""
        t_current = record.timestamp_seconds

        # 1. Duplicate Detection
        if self.last_timestamp is not None and math.isclose(t_current, self.last_timestamp, abs_tol=1e-7):
            if self.config.deduplication_policy == "reject":
                raise TemporalMonitoringError(
                    f"Duplicate timestamp {t_current:.6f}s rejected for sequence '{self.sequence_id}' (horizon={self.horizon})."
                )
            elif self.config.deduplication_policy == "ignore":
                # Return current state without modifying history
                return self.compute_summary()
            elif self.config.deduplication_policy == "replace":
                if self.records:
                    self.records.pop()

        # 2. Out-of-Order Detection
        if self.last_timestamp is not None and t_current < self.last_timestamp:
            if not self.config.allow_out_of_order:
                raise TemporalMonitoringError(
                    f"Out-of-order timestamp {t_current:.6f}s < {self.last_timestamp:.6f}s for sequence '{self.sequence_id}'."
                )
            record.is_out_of_order = True

        # 3. Timestamp Gap Detection
        has_gap = False
        if self.last_timestamp is not None and not record.is_out_of_order:
            gap = t_current - self.last_timestamp
            if gap > self.config.max_timestamp_gap_seconds:
                has_gap = True
                # Reset consecutive increasing count on gap
                self.consecutive_increasing = 0

        # 4. Model Version Tracking
        if self.last_checkpoint_id is not None and record.checkpoint_id != self.last_checkpoint_id:
            self.checkpoint_change_detected = True

        # 5. State Transition Detection
        if self.config.track_state_transitions and self.records:
            prev_record = self.records[-1]
            prev_level = prev_record.risk_level
            curr_level = record.risk_level

            if prev_level != curr_level:
                is_genuine = (
                    prev_level != RiskLevel.UNKNOWN
                    and curr_level != RiskLevel.UNKNOWN
                    and not record.is_stale
                )
                prev_score_val = prev_record.risk_score if not math.isnan(prev_record.risk_score) else None

                transition = StateTransitionEvent(
                    sequence_id=self.sequence_id,
                    horizon=self.horizon,
                    previous_level=prev_level,
                    current_level=curr_level,
                    timestamp_seconds=t_current,
                    risk_score=record.risk_score,
                    previous_risk_score=prev_score_val,
                    is_genuine_transition=is_genuine,
                    calibration_status=record.calibration_status,
                    notes=f"Transition from {prev_level.value} to {curr_level.value}",
                )
                self.transitions.append(transition)

        # 6. Append to rolling window
        self.records.append(record)
        if not record.is_out_of_order:
            self.last_timestamp = t_current
            if record.source_frame_index is not None:
                self.last_frame_index = record.source_frame_index
        self.last_checkpoint_id = record.checkpoint_id

        # 7. Update Consecutive Increasing & Summary
        return self.compute_summary(has_gap=has_gap)

    def compute_summary(self, has_gap: bool = False) -> TemporalRiskSummary:
        """Derive instantaneous rolling statistics over the current window."""
        if not self.records:
            return TemporalRiskSummary(
                sequence_id=self.sequence_id,
                horizon=self.horizon,
                current_score=float("nan"),
                previous_score=None,
                absolute_change=None,
                relative_change=None,
                rolling_mean=float("nan"),
                rolling_max=float("nan"),
                rolling_min=float("nan"),
                volatility=None,
                trend_direction=RiskTrendDirection.UNKNOWN,
                consecutive_increasing_count=0,
                current_risk_level=RiskLevel.UNKNOWN,
                level_history=[],
                observation_count=0,
                window_size=self.config.history_window_size,
            )

        latest = self.records[-1]
        valid_scores = [r.risk_score for r in self.records if not math.isnan(r.risk_score) and r.is_valid]
        level_history = [r.risk_level for r in self.records]

        # Prior observation
        prev_score: Optional[float] = None
        if len(self.records) >= 2:
            prev_record = self.records[-2]
            if not math.isnan(prev_record.risk_score) and prev_record.is_valid:
                prev_score = prev_record.risk_score

        current_score = latest.risk_score
        abs_change: Optional[float] = None
        rel_change: Optional[float] = None
        trend = RiskTrendDirection.UNKNOWN

        if prev_score is not None and not math.isnan(current_score) and latest.is_valid:
            abs_change = current_score - prev_score
            if abs(prev_score) > 1e-4:
                rel_change = abs_change / prev_score
            else:
                rel_change = 0.0

            if has_gap:
                trend = RiskTrendDirection.UNKNOWN
                self.consecutive_increasing = 0
            elif abs_change > self.config.trend_stability_threshold:
                trend = RiskTrendDirection.INCREASING
                self.consecutive_increasing += 1
            elif abs_change < -self.config.trend_stability_threshold:
                trend = RiskTrendDirection.DECREASING
                self.consecutive_increasing = 0
            else:
                trend = RiskTrendDirection.STABLE
                self.consecutive_increasing = 0
        else:
            self.consecutive_increasing = 0

        # Rolling Statistics
        if valid_scores:
            rolling_mean = sum(valid_scores) / len(valid_scores)
            rolling_max = max(valid_scores)
            rolling_min = min(valid_scores)
            if len(valid_scores) >= 2:
                variance = sum((s - rolling_mean) ** 2 for s in valid_scores) / (len(valid_scores) - 1)
                volatility = math.sqrt(max(0.0, variance))
            else:
                volatility = 0.0
        else:
            rolling_mean = float("nan")
            rolling_max = float("nan")
            rolling_min = float("nan")
            volatility = None

        return TemporalRiskSummary(
            sequence_id=self.sequence_id,
            horizon=self.horizon,
            current_score=current_score,
            previous_score=prev_score,
            absolute_change=abs_change,
            relative_change=rel_change,
            rolling_mean=rolling_mean,
            rolling_max=rolling_max,
            rolling_min=rolling_min,
            volatility=volatility,
            trend_direction=trend,
            consecutive_increasing_count=self.consecutive_increasing,
            current_risk_level=latest.risk_level,
            level_history=level_history,
            observation_count=len(self.records),
            window_size=self.config.history_window_size,
            is_stale=latest.is_stale,
            has_timestamp_gap=has_gap,
            last_timestamp_seconds=latest.timestamp_seconds,
            checkpoint_id=latest.checkpoint_id,
            calibration_status=latest.calibration_status,
            notes=latest.notes,
        )


class TemporalRiskMonitor(ITemporalRiskMonitor):
    """High-performance chronological perception risk monitoring engine.

    Implements ITemporalRiskMonitor, maintaining isolated temporal histories for
    distinct video sequences and forecast horizons (H1, H3, H5, Aggregated).
    """

    def __init__(
        self,
        config: Optional[Union[TemporalMonitoringConfig, RiskConfig]] = None,
        classifier: Optional[RiskClassifier] = None,
    ):
        if isinstance(config, RiskConfig):
            self.config = config.monitoring
        elif isinstance(config, TemporalMonitoringConfig):
            self.config = config
        else:
            self.config = TemporalMonitoringConfig()

        if classifier is not None:
            self.classifier = classifier
        else:
            from src.risk.risk_classifier import RiskClassifier
            self.classifier = RiskClassifier()
        self.trackers: Dict[Tuple[str, Optional[int]], SequenceHorizonTracker] = {}

    def _get_tracker(self, sequence_id: str, horizon: Optional[int]) -> SequenceHorizonTracker:
        """Retrieve or lazily initialize tracker for (sequence_id, horizon)."""
        key = (sequence_id, horizon)
        if key not in self.trackers:
            self.trackers[key] = SequenceHorizonTracker(
                sequence_id=sequence_id,
                horizon=horizon,
                config=self.config,
            )
        return self.trackers[key]

    def update(
        self,
        observation: Union[
            HorizonRiskRecord,
            AggregatedRiskRecord,
            ClassificationResult,
            RiskScore,
            Dict[str, Any],
        ],
        timestamp_seconds: Optional[float] = None,
        is_stale: bool = False,
        is_valid: bool = True,
    ) -> TemporalRiskSummary:
        """Ingest a risk observation and return updated chronological metrics.

        Accepts polymorphic inputs from Task 1 (RiskScore), Task 2 (RiskRecord),
        Task 3 (ClassificationResult), or raw parameter dictionaries.
        """
        # 1. Parse input to MonitoredRiskRecord
        from src.risk.risk_classifier import ClassificationResult
        if isinstance(observation, ClassificationResult):
            score = observation.risk_score
            t_sec = timestamp_seconds if timestamp_seconds is not None else (observation.target_timestamp_seconds or 0.0)
            rec = MonitoredRiskRecord(
                sequence_id=observation.sequence_id,
                risk_score=score,
                risk_level=observation.risk_level,
                timestamp_seconds=float(t_sec),
                horizon=observation.horizon,
                source_frame_id=observation.source_frame_id or "",
                target_frame_id=observation.target_frame_id,
                prediction_id=observation.prediction_id,
                calibration_status=observation.calibration_status,
                is_valid=is_valid and (not observation.is_missing),
                is_stale=is_stale or observation.is_stale,
                notes=observation.notes,
            )
        elif isinstance(observation, HorizonRiskRecord):
            score = validate_risk_value(observation.risk_score)
            level = self.classifier.classify_record(observation, is_stale=is_stale).risk_level
            t_sec = timestamp_seconds if timestamp_seconds is not None else (observation.target_timestamp_seconds or 0.0)
            rec = MonitoredRiskRecord(
                sequence_id=observation.sequence_id,
                risk_score=score,
                risk_level=level,
                timestamp_seconds=float(t_sec),
                horizon=observation.horizon,
                source_frame_id=observation.source_frame_id,
                source_frame_index=observation.source_frame_index,
                target_frame_id=observation.target_frame_id,
                target_frame_index=observation.target_frame_index,
                prediction_id=observation.prediction_id,
                calibration_status="PROVISIONAL",
                model_name=observation.model_name,
                checkpoint_id=observation.checkpoint_id,
                is_valid=is_valid,
                is_stale=is_stale,
                notes=f"Horizon {observation.horizon} forecast",
            )
        elif isinstance(observation, AggregatedRiskRecord):
            score = validate_risk_value(observation.aggregated_risk_score)
            level = self.classifier.classify_aggregated(observation, is_stale=is_stale).risk_level
            t_sec = timestamp_seconds if timestamp_seconds is not None else (observation.target_timestamp_seconds or 0.0)
            rec = MonitoredRiskRecord(
                sequence_id=observation.sequence_id,
                risk_score=score,
                risk_level=level,
                timestamp_seconds=float(t_sec),
                horizon=observation.primary_horizon,
                source_frame_id=observation.source_frame_id,
                source_frame_index=observation.source_frame_index,
                calibration_status="PROVISIONAL",
                model_name=observation.model_name,
                checkpoint_id=observation.checkpoint_id,
                is_valid=is_valid,
                is_stale=is_stale,
                notes=f"Aggregated risk ({observation.aggregation_strategy})",
            )
        elif isinstance(observation, RiskScore):
            score = validate_risk_value(observation.overall_risk_score) if not observation.is_missing_data else float("nan")
            level = observation.risk_level
            t_sec = timestamp_seconds if timestamp_seconds is not None else (observation.target_timestamp_seconds or 0.0)
            rec = MonitoredRiskRecord(
                sequence_id=observation.sequence_id,
                risk_score=score,
                risk_level=level,
                timestamp_seconds=float(t_sec),
                horizon=observation.primary_horizon,
                calibration_status="PROVISIONAL",
                is_valid=not observation.is_missing_data,
                is_stale=is_stale,
            )
        elif isinstance(observation, dict):
            raw_score = observation.get("risk_score", observation.get("overall_risk_score"))
            is_miss = observation.get("is_missing", observation.get("is_missing_data", False))

            if is_miss or raw_score is None:
                score = float("nan")
                level = RiskLevel.UNKNOWN
            else:
                score = validate_risk_value(raw_score)
                raw_level = observation.get("risk_level")
                if raw_level:
                    level = RiskLevel.from_string(raw_level) if isinstance(raw_level, str) else raw_level
                else:
                    level = self.classifier.classify_score(score, horizon=observation.get("horizon"))

            t_sec = timestamp_seconds if timestamp_seconds is not None else float(observation.get("timestamp_seconds", observation.get("target_timestamp_seconds", 0.0)))
            rec = MonitoredRiskRecord(
                sequence_id=observation["sequence_id"],
                risk_score=score,
                risk_level=level,
                timestamp_seconds=float(t_sec),
                horizon=observation.get("horizon", observation.get("primary_horizon")),
                source_frame_id=observation.get("source_frame_id", ""),
                source_frame_index=observation.get("source_frame_index"),
                target_frame_id=observation.get("target_frame_id"),
                target_frame_index=observation.get("target_frame_index"),
                prediction_id=observation.get("prediction_id"),
                calibration_status=observation.get("calibration_status", "PROVISIONAL"),
                model_name=observation.get("model_name", "LSTMForecaster"),
                checkpoint_id=observation.get("checkpoint_id", "unknown"),
                is_valid=is_valid and (not is_miss),
                is_stale=is_stale or observation.get("is_stale", False),
                notes=observation.get("notes", ""),
            )
        else:
            raise TemporalMonitoringError(f"Unsupported observation type: {type(observation).__name__}")

        # Check staleness based on timestamp age if TTL exceeded
        # (Event time checks)
        tracker = self._get_tracker(rec.sequence_id, rec.horizon)
        return tracker.record_observation(rec)

    # ITemporalRiskMonitor Interface Implementation
    def record_and_check(self, risk_score: RiskScore) -> bool:
        """Record risk score and determine if persistent alert condition holds.

        Condition holds if current tier is HIGH or CRITICAL and has persisted
        for at least alert_persistence_threshold consecutive frames.
        """
        summary = self.update(risk_score)
        if summary.current_risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            # Check recent level history for persistence
            recent_levels = summary.level_history[-self.config.alert_persistence_threshold:]
            if len(recent_levels) >= self.config.alert_persistence_threshold:
                return all(lvl in (RiskLevel.HIGH, RiskLevel.CRITICAL) for lvl in recent_levels)
        return False

    def get_recent_scores(self, sequence_id: str, window_size: int) -> List[RiskScore]:
        """Retrieve recent risk scores for sequence, conforming to ITemporalRiskMonitor."""
        # Check all trackers for sequence_id
        matching_records: List[MonitoredRiskRecord] = []
        for (s_id, _), tracker in self.trackers.items():
            if s_id == sequence_id:
                matching_records.extend(list(tracker.records))

        # Sort chronologically by timestamp_seconds
        matching_records.sort(key=lambda r: r.timestamp_seconds)
        selected = matching_records[-window_size:] if len(matching_records) > window_size else matching_records

        return [
            RiskScore(
                sequence_id=r.sequence_id,
                overall_risk_score=r.risk_score if not math.isnan(r.risk_score) else 0.0,
                primary_horizon=r.horizon or 1,
                risk_level=r.risk_level,
                target_timestamp_seconds=r.timestamp_seconds,
                is_missing_data=not r.is_valid or math.isnan(r.risk_score),
            )
            for r in selected
        ]

    def get_summary(self, sequence_id: str, horizon: Optional[int] = None) -> Optional[TemporalRiskSummary]:
        """Retrieve current temporal risk summary for a sequence and horizon."""
        key = (sequence_id, horizon)
        if key in self.trackers:
            return self.trackers[key].compute_summary()
        return None

    def get_history(
        self,
        sequence_id: str,
        horizon: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[MonitoredRiskRecord]:
        """Retrieve ordered historical observations for a sequence and horizon."""
        key = (sequence_id, horizon)
        if key not in self.trackers:
            return []
        records = list(self.trackers[key].records)
        if limit is not None:
            return records[-limit:]
        return records

    def get_state_transitions(
        self,
        sequence_id: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> List[StateTransitionEvent]:
        """Retrieve recorded state transitions, optionally filtered by sequence and horizon."""
        transitions: List[StateTransitionEvent] = []
        for (s_id, h), tracker in self.trackers.items():
            if sequence_id is not None and s_id != sequence_id:
                continue
            if horizon is not None and h != horizon:
                continue
            transitions.extend(tracker.transitions)

        transitions.sort(key=lambda t: t.timestamp_seconds)
        return transitions

    def reset_sequence(self, sequence_id: str) -> None:
        """Clear all tracking state for a specific sequence."""
        keys_to_delete = [k for k in self.trackers if k[0] == sequence_id]
        for k in keys_to_delete:
            del self.trackers[k]

    def clear_all(self) -> None:
        """Clear all tracking states across all sequences and horizons."""
        self.trackers.clear()
