"""Predictive Early Warning & Alert Generation Engine for PerceptGuard AI.

Converts validated temporal risk information from TemporalRiskMonitor into
actionable, evidence-based, deduplicated early-warning alerts adhering to strict
lifecycle management, horizon isolation, and zero-leakage safety invariants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.early_warning.early_warning_config import AlertGenerationConfig, EarlyWarningConfig
from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.early_warning.interfaces import (
    AlertLifecycleState,
    AlertSeverity,
    AlertType,
    EarlyWarningAlert,
    IEarlyWarningGenerator,
    RiskLevel,
    RiskScore,
)
from src.risk.temporal_risk_monitor import (
    MonitoredRiskRecord,
    RiskTrendDirection,
    StateTransitionEvent,
    TemporalRiskSummary,
)


@dataclass
class AlertLifecycleEvent:
    """Audit record capturing an alert lifecycle state transition."""
    alert_id: str
    sequence_id: str
    from_state: AlertLifecycleState
    to_state: AlertLifecycleState
    timestamp_seconds: float
    reason: str
    actor: str = "system"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert lifecycle event to dictionary."""
        d = asdict(self)
        d["from_state"] = self.from_state.value
        d["to_state"] = self.to_state.value
        return d


class EarlyWarningEngine(IEarlyWarningGenerator):
    """Predictive perception failure early warning generation and lifecycle engine.

    Evaluates rule criteria over temporal risk summaries across H1, H3, and H5,
    enforcing deduplication, cooldown, auto-resolution, and evidence-based explanations.
    """

    def __init__(
        self,
        config: Optional[Union[AlertGenerationConfig, EarlyWarningConfig]] = None,
    ):
        if isinstance(config, EarlyWarningConfig):
            self.config = config.alert_generation
        elif isinstance(config, AlertGenerationConfig):
            self.config = config
        else:
            self.config = AlertGenerationConfig()

        # Active alerts map: (sequence_id, horizon, rule_id) -> EarlyWarningAlert
        self.active_alerts: Dict[Tuple[str, int, str], EarlyWarningAlert] = {}
        # Chronological audit history of all generated alerts
        self.alert_history: List[EarlyWarningAlert] = []
        # Audit log of alert lifecycle transitions
        self.lifecycle_events: List[AlertLifecycleEvent] = []
        # Last alert generation timestamp for cooldown: (sequence_id, horizon, rule_id) -> timestamp_seconds
        self.last_alert_timestamps: Dict[Tuple[str, int, str], float] = {}

    def _generate_alert_id(
        self,
        sequence_id: str,
        horizon: int,
        rule_id: str,
        timestamp_seconds: float,
    ) -> str:
        """Construct deterministic, unique alert identifier."""
        ts_ms = int(timestamp_seconds * 1000)
        return f"alert_{sequence_id}_h{horizon}_{rule_id}_{ts_ms}"

    def evaluate(
        self,
        summary: TemporalRiskSummary,
        transition: Optional[StateTransitionEvent] = None,
        source_record: Optional[MonitoredRiskRecord] = None,
    ) -> List[EarlyWarningAlert]:
        """Evaluate early warning rules against a temporal risk summary.

        Parameters
        ----------
        summary : TemporalRiskSummary
            Instantaneous chronological summary from TemporalRiskMonitor.
        transition : Optional[StateTransitionEvent]
            Recent state transition event if one occurred on this frame.
        source_record : Optional[MonitoredRiskRecord]
            Raw source monitored record for frame IDs and model provenance.
        """
        horizon = summary.horizon or 1
        sequence_id = summary.sequence_id
        t_current = summary.last_timestamp_seconds
        source_frame_id = source_record.source_frame_id if source_record else ""
        target_frame_id = source_record.target_frame_id if source_record else f"frame_t+{horizon}"
        model_version = source_record.model_name if source_record else "LSTMForecaster"

        generated_alerts: List[EarlyWarningAlert] = []

        # 1. Check for Expired Active Alerts
        self._expire_alerts(t_current)

        # 2. Check for Auto-Resolution on Nominal Risk
        if summary.current_risk_level == RiskLevel.LOW and not summary.is_stale:
            if self.config.auto_resolve_on_low_risk:
                self._auto_resolve_sequence_horizon(sequence_id, horizon, t_current)

        # 3. Rule Evaluation: Data Quality Warning
        if summary.is_stale or summary.current_risk_level == RiskLevel.UNKNOWN or summary.has_timestamp_gap:
            if self.config.enable_data_quality_warnings:
                reason_parts = []
                if summary.is_stale:
                    reason_parts.append("Telemetry is stale (freshness TTL exceeded)")
                if summary.current_risk_level == RiskLevel.UNKNOWN:
                    reason_parts.append("Prediction unavailable or missing")
                if summary.has_timestamp_gap:
                    reason_parts.append("Large timestamp gap detected in sensor stream")

                alert = self._create_or_update_alert(
                    sequence_id=sequence_id,
                    horizon=horizon,
                    rule_id="RULE_DATA_QUALITY",
                    severity=AlertSeverity.DATA_QUALITY_WARNING,
                    alert_type=AlertType.DATA_QUALITY,
                    risk_score=summary.current_score if not math.isnan(summary.current_score) else 0.0,
                    risk_level=RiskLevel.UNKNOWN,
                    timestamp_seconds=t_current,
                    source_frame_id=source_frame_id,
                    target_frame_id=target_frame_id,
                    warning_message=f"Data Quality Warning: {'; '.join(reason_parts)}.",
                    recommended_action="Inspect sensor stream and inference pipeline; engage fail-safe perception mode.",
                    trigger_reason=f"Telemetry irregularity on horizon H{horizon}: {'; '.join(reason_parts)}.",
                    risk_trend=summary.trend_direction.value,
                    model_version=model_version,
                    calibration_status=summary.calibration_status,
                )
                if alert:
                    generated_alerts.append(alert)

            # Safety Rule: If data is missing or stale, NEVER generate failure/nominal risk alerts!
            return generated_alerts

        # 4. Rule Evaluation: CRITICAL Threshold Breach
        if summary.current_risk_level == RiskLevel.CRITICAL:
            alert = self._create_or_update_alert(
                sequence_id=sequence_id,
                horizon=horizon,
                rule_id="RULE_CRITICAL_BREACH",
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.THRESHOLD_BREACH,
                risk_score=summary.current_score,
                risk_level=RiskLevel.CRITICAL,
                timestamp_seconds=t_current,
                source_frame_id=source_frame_id,
                target_frame_id=target_frame_id,
                warning_message=f"Critical perception failure predicted: risk score {summary.current_score:.4f} breached CRITICAL threshold.",
                recommended_action="Issue immediate acoustic/haptic driver handover alert; initiate fail-safe deceleration.",
                trigger_reason=f"Imminent failure predicted: risk score ({summary.current_score:.4f}) breached CRITICAL threshold (0.85) on horizon H{horizon} at t={t_current:.3f}s.",
                risk_trend=summary.trend_direction.value,
                model_version=model_version,
                calibration_status=summary.calibration_status,
            )
            if alert:
                generated_alerts.append(alert)

        # 5. Rule Evaluation: HIGH Threshold Breach
        elif summary.current_risk_level == RiskLevel.HIGH:
            alert = self._create_or_update_alert(
                sequence_id=sequence_id,
                horizon=horizon,
                rule_id="RULE_HIGH_BREACH",
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.THRESHOLD_BREACH,
                risk_score=summary.current_score,
                risk_level=RiskLevel.HIGH,
                timestamp_seconds=t_current,
                source_frame_id=source_frame_id,
                target_frame_id=target_frame_id,
                warning_message=f"Elevated perception failure risk: risk score {summary.current_score:.4f} breached HIGH threshold.",
                recommended_action="Display Stage-1 visual warning; dampen throttle response and verify sensor field of view.",
                trigger_reason=f"Elevated risk predicted: risk score ({summary.current_score:.4f}) breached HIGH threshold (0.60) on horizon H{horizon} at t={t_current:.3f}s.",
                risk_trend=summary.trend_direction.value,
                model_version=model_version,
                calibration_status=summary.calibration_status,
            )
            if alert:
                generated_alerts.append(alert)

        # 6. Rule Evaluation: MEDIUM Advisory Warning
        elif summary.current_risk_level == RiskLevel.MEDIUM:
            # Requires minimum consecutive observations to prevent minor score jitter from spamming
            if summary.observation_count >= self.config.min_consecutive_for_medium_alert:
                alert = self._create_or_update_alert(
                    sequence_id=sequence_id,
                    horizon=horizon,
                    rule_id="RULE_MEDIUM_ADVISORY",
                    severity=AlertSeverity.INFO,
                    alert_type=AlertType.THRESHOLD_BREACH,
                    risk_score=summary.current_score,
                    risk_level=RiskLevel.MEDIUM,
                    timestamp_seconds=t_current,
                    source_frame_id=source_frame_id,
                    target_frame_id=target_frame_id,
                    warning_message=f"Advisory: mild perception reliability degradation detected ({summary.current_score:.4f}).",
                    recommended_action="Log telemetry; prepare secondary sensor redundancy.",
                    trigger_reason=f"Advisory condition: risk score ({summary.current_score:.4f}) in MEDIUM band (0.30-0.60] on horizon H{horizon}.",
                    risk_trend=summary.trend_direction.value,
                    model_version=model_version,
                    calibration_status=summary.calibration_status,
                )
                if alert:
                    generated_alerts.append(alert)

        # 7. Rule Evaluation: Rapid Degradation Trend
        if (
            summary.trend_direction == RiskTrendDirection.INCREASING
            and summary.consecutive_increasing_count >= self.config.min_consecutive_increasing_for_trend_alert
        ):
            delta_val = summary.absolute_change or 0.0
            alert = self._create_or_update_alert(
                sequence_id=sequence_id,
                horizon=horizon,
                rule_id="RULE_RAPID_DEGRADATION",
                severity=AlertSeverity.TREND_WARNING,
                alert_type=AlertType.RAPID_DEGRADATION,
                risk_score=summary.current_score,
                risk_level=summary.current_risk_level,
                timestamp_seconds=t_current,
                source_frame_id=source_frame_id,
                target_frame_id=target_frame_id,
                warning_message=f"Rapid perception degradation detected: risk increased across {summary.consecutive_increasing_count} consecutive frames.",
                recommended_action="Pre-charge braking system and alert driver to potential upcoming tracking drop.",
                trigger_reason=f"Sustained degradation trend: risk score increased across {summary.consecutive_increasing_count} frames (ΔS={delta_val:+.4f}) on horizon H{horizon}.",
                risk_trend=summary.trend_direction.value,
                model_version=model_version,
                calibration_status=summary.calibration_status,
            )
            if alert:
                generated_alerts.append(alert)

        # 8. Rule Evaluation: Operational State Escalation
        if transition is not None and transition.is_genuine_transition:
            if transition.current_level.severity_rank > transition.previous_level.severity_rank:
                sev = AlertSeverity.CRITICAL if transition.current_level == RiskLevel.CRITICAL else AlertSeverity.WARNING
                alert = self._create_or_update_alert(
                    sequence_id=sequence_id,
                    horizon=horizon,
                    rule_id="RULE_STATE_ESCALATION",
                    severity=sev,
                    alert_type=AlertType.STATE_TRANSITION,
                    risk_score=transition.risk_score,
                    risk_level=transition.current_level,
                    timestamp_seconds=t_current,
                    source_frame_id=source_frame_id,
                    target_frame_id=target_frame_id,
                    warning_message=f"Operational risk tier escalated: {transition.previous_level.value} -> {transition.current_level.value}.",
                    recommended_action="Escalate driver warning state and active monitoring.",
                    trigger_reason=f"Risk tier escalated from {transition.previous_level.value} to {transition.current_level.value} at t={t_current:.3f}s (score={transition.risk_score:.4f}).",
                    risk_trend=summary.trend_direction.value,
                    model_version=model_version,
                    previous_category=transition.previous_level,
                    current_category=transition.current_level,
                    calibration_status=summary.calibration_status,
                )
                if alert:
                    generated_alerts.append(alert)

        return generated_alerts

    def _create_or_update_alert(
        self,
        sequence_id: str,
        horizon: int,
        rule_id: str,
        severity: AlertSeverity,
        alert_type: AlertType,
        risk_score: float,
        risk_level: RiskLevel,
        timestamp_seconds: float,
        source_frame_id: str,
        target_frame_id: str,
        warning_message: str,
        recommended_action: str,
        trigger_reason: str,
        risk_trend: str,
        model_version: str,
        calibration_status: str,
        previous_category: Optional[RiskLevel] = None,
        current_category: Optional[RiskLevel] = None,
    ) -> Optional[EarlyWarningAlert]:
        """Process an alert candidate under deduplication, cooldown, and escalation policies."""
        key = (sequence_id, horizon, rule_id)
        last_t = self.last_alert_timestamps.get(key)

        # Cooldown check
        if last_t is not None and (timestamp_seconds - last_t) < self.config.cooldown_seconds:
            # Check for critical severity escalation
            if key in self.active_alerts and self.config.enable_critical_escalation:
                active = self.active_alerts[key]
                if severity.priority_rank > active.severity.priority_rank:
                    # Escalate active alert
                    self._transition_lifecycle(active, AlertLifecycleState.ACTIVE, "Severity escalated under cooldown")
                    active.severity = severity
                    active.risk_score = risk_score
                    active.timestamp = timestamp_seconds
                    active.warning_message = warning_message
                    active.trigger_reason = trigger_reason
                    self.last_alert_timestamps[key] = timestamp_seconds
                    return active
            # Cooldown active; suppress duplicate alert
            return None

        # Build new alert
        alert_id = self._generate_alert_id(sequence_id, horizon, rule_id, timestamp_seconds)
        alert = EarlyWarningAlert(
            alert_id=alert_id,
            sequence_id=sequence_id,
            risk_level=risk_level,
            risk_score=risk_score,
            trigger_horizon=horizon,
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            target_timestamp_seconds=timestamp_seconds,
            warning_message=warning_message,
            recommended_action=recommended_action,
            is_suppressed=False,
            severity=severity,
            alert_type=alert_type,
            lifecycle_state=AlertLifecycleState.ACTIVE,
            timestamp=timestamp_seconds,
            forecast_horizon=horizon,
            risk_category=risk_level,
            trigger_reason=trigger_reason,
            risk_trend=risk_trend,
            previous_category=previous_category,
            current_category=current_category or risk_level,
            calibration_status=calibration_status,
            model_version=model_version,
            rule_id=rule_id,
            metadata={
                "cooldown_applied_seconds": self.config.cooldown_seconds,
                "is_escalated": False,
            },
        )

        # Register active alert and update history
        self.active_alerts[key] = alert
        self.alert_history.append(alert)
        self.last_alert_timestamps[key] = timestamp_seconds

        # Record initial lifecycle transition: NEW -> ACTIVE
        self.lifecycle_events.append(
            AlertLifecycleEvent(
                alert_id=alert_id,
                sequence_id=sequence_id,
                from_state=AlertLifecycleState.NEW,
                to_state=AlertLifecycleState.ACTIVE,
                timestamp_seconds=timestamp_seconds,
                reason=f"Alert triggered by {rule_id}",
            )
        )

        return alert

    def _auto_resolve_sequence_horizon(self, sequence_id: str, horizon: int, timestamp_seconds: float) -> None:
        """Automatically resolve active threshold/trend alerts when risk returns to nominal LOW."""
        keys_to_resolve = [
            k for k in self.active_alerts
            if k[0] == sequence_id and k[1] == horizon and k[2] in ("RULE_CRITICAL_BREACH", "RULE_HIGH_BREACH", "RULE_RAPID_DEGRADATION")
        ]
        for key in keys_to_resolve:
            alert = self.active_alerts[key]
            self._transition_lifecycle(alert, AlertLifecycleState.RESOLVED, "Perception reliability recovered to nominal LOW", timestamp_seconds)
            del self.active_alerts[key]

    def _expire_alerts(self, current_timestamp: float) -> None:
        """Transition active alerts exceeding TTL to EXPIRED state."""
        expired_keys = []
        for key, alert in self.active_alerts.items():
            if (current_timestamp - alert.timestamp) > self.config.alert_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            alert = self.active_alerts[key]
            self._transition_lifecycle(alert, AlertLifecycleState.EXPIRED, f"Alert exceeded TTL ({self.config.alert_ttl_seconds}s)", current_timestamp)
            del self.active_alerts[key]

    def _transition_lifecycle(
        self,
        alert: EarlyWarningAlert,
        to_state: AlertLifecycleState,
        reason: str,
        timestamp_seconds: Optional[float] = None,
        actor: str = "system",
    ) -> None:
        """Record lifecycle state change for an alert."""
        from_state = alert.lifecycle_state
        if from_state == to_state:
            return

        alert.lifecycle_state = to_state
        t_sec = timestamp_seconds if timestamp_seconds is not None else alert.timestamp

        event = AlertLifecycleEvent(
            alert_id=alert.alert_id,
            sequence_id=alert.sequence_id,
            from_state=from_state,
            to_state=to_state,
            timestamp_seconds=t_sec,
            reason=reason,
            actor=actor,
        )
        self.lifecycle_events.append(event)

    # IEarlyWarningGenerator interface implementation
    def generate_alert(
        self,
        risk_score: RiskScore,
        input_record: EarlyWarningInputRecord,
    ) -> Optional[EarlyWarningAlert]:
        """Interface method generating alert from scalar RiskScore and InputRecord."""
        # Wrap into mock summary for backward interface compatibility
        summary = TemporalRiskSummary(
            sequence_id=risk_score.sequence_id,
            horizon=risk_score.primary_horizon,
            current_score=risk_score.overall_risk_score,
            previous_score=None,
            absolute_change=None,
            relative_change=None,
            rolling_mean=risk_score.overall_risk_score,
            rolling_max=risk_score.overall_risk_score,
            rolling_min=risk_score.overall_risk_score,
            volatility=0.0,
            trend_direction=RiskTrendDirection.UNKNOWN,
            consecutive_increasing_count=0,
            current_risk_level=risk_score.risk_level,
            level_history=[risk_score.risk_level],
            observation_count=1,
            window_size=10,
            is_stale=False,
            has_timestamp_gap=False,
            last_timestamp_seconds=risk_score.target_timestamp_seconds or 0.0,
        )
        alerts = self.evaluate(summary)
        return alerts[0] if alerts else None

    # Lifecycle Management API
    def acknowledge_alert(
        self,
        alert_id: str,
        operator_id: str = "operator",
        timestamp_seconds: Optional[float] = None,
    ) -> bool:
        """Operator or downstream controller acknowledges an active alert."""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                self._transition_lifecycle(alert, AlertLifecycleState.ACKNOWLEDGED, "Operator acknowledged alert", timestamp_seconds, actor=operator_id)
                alert.acknowledgement_status = "ACKNOWLEDGED"
                return True
        return False

    def resolve_alert(
        self,
        alert_id: str,
        reason: str = "Manual resolution",
        timestamp_seconds: Optional[float] = None,
        operator_id: str = "operator",
    ) -> bool:
        """Operator or system manually resolves an alert."""
        for key, alert in list(self.active_alerts.items()):
            if alert.alert_id == alert_id:
                self._transition_lifecycle(alert, AlertLifecycleState.RESOLVED, reason, timestamp_seconds, actor=operator_id)
                del self.active_alerts[key]
                return True
        return False

    def get_active_alerts(
        self,
        sequence_id: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> List[EarlyWarningAlert]:
        """Retrieve active alerts, optionally filtered by sequence and horizon."""
        alerts = list(self.active_alerts.values())
        if sequence_id is not None:
            alerts = [a for a in alerts if a.sequence_id == sequence_id]
        if horizon is not None:
            alerts = [a for a in alerts if a.trigger_horizon == horizon]
        return alerts

    def get_alert_history(
        self,
        sequence_id: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> List[EarlyWarningAlert]:
        """Retrieve chronological history of all created alerts."""
        alerts = list(self.alert_history)
        if sequence_id is not None:
            alerts = [a for a in alerts if a.sequence_id == sequence_id]
        if horizon is not None:
            alerts = [a for a in alerts if a.trigger_horizon == horizon]
        return alerts

    def get_lifecycle_events(
        self,
        alert_id: Optional[str] = None,
    ) -> List[AlertLifecycleEvent]:
        """Retrieve audit log of lifecycle events, optionally filtered by alert ID."""
        if alert_id is not None:
            return [e for e in self.lifecycle_events if e.alert_id == alert_id]
        return list(self.lifecycle_events)

    def clear(self) -> None:
        """Clear all active alerts and history."""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.lifecycle_events.clear()
        self.last_alert_timestamps.clear()
