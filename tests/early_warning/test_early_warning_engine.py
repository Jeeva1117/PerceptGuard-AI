"""Comprehensive unit and integration tests for EarlyWarningEngine (Day 10 Task 5).

Verifies multi-level alerts (MEDIUM, HIGH, CRITICAL), rapid degradation trends,
H1/H3/H5 horizon independence, sequence isolation, cooldown & deduplication,
full alert lifecycle (NEW, ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED),
data quality warnings for missing/stale data, evidence-based traceability,
and end-to-end integration with Task 2, Task 3, and Task 4.
"""

import math
from pathlib import Path
import pytest

from src.early_warning.early_warning_config import AlertGenerationConfig, EarlyWarningConfig
from src.early_warning.early_warning_engine import AlertLifecycleEvent, EarlyWarningEngine
from src.early_warning.interfaces import (
    AlertLifecycleState,
    AlertSeverity,
    AlertType,
    EarlyWarningAlert,
    RiskLevel,
    RiskScore,
)
from src.risk.risk_classifier import ClassificationResult, RiskClassifier
from src.risk.risk_models import AggregatedRiskRecord, HorizonRiskRecord
from src.risk.temporal_risk_monitor import (
    MonitoredRiskRecord,
    RiskTrendDirection,
    StateTransitionEvent,
    TemporalRiskMonitor,
    TemporalRiskSummary,
)


def make_summary(
    sequence_id: str = "seq_test",
    horizon: int = 1,
    current_score: float = 0.88,
    current_risk_level: RiskLevel = RiskLevel.HIGH,
    trend_direction: RiskTrendDirection = RiskTrendDirection.STABLE,
    consecutive_increasing: int = 0,
    timestamp_seconds: float = 0.500,
    is_stale: bool = False,
    observation_count: int = 5,
    calibration_status: str = "PROVISIONAL",
) -> TemporalRiskSummary:
    """Helper to synthesize deterministic TemporalRiskSummary fixtures."""
    return TemporalRiskSummary(
        sequence_id=sequence_id,
        horizon=horizon,
        current_score=current_score,
        previous_score=current_score - 0.05 if current_score >= 0.05 else None,
        absolute_change=0.05,
        relative_change=0.06,
        rolling_mean=current_score,
        rolling_max=current_score,
        rolling_min=current_score,
        volatility=0.01,
        trend_direction=trend_direction,
        consecutive_increasing_count=consecutive_increasing,
        current_risk_level=current_risk_level,
        level_history=[current_risk_level] * observation_count,
        observation_count=observation_count,
        window_size=30,
        is_stale=is_stale,
        has_timestamp_gap=False,
        last_timestamp_seconds=timestamp_seconds,
        calibration_status=calibration_status,
    )


class TestAlertGenerationRules:
    """Tests 1, 2, 3, 4, 5: HIGH, CRITICAL, MEDIUM, Trend, and Sustained Alerts."""

    @pytest.fixture
    def engine(self):
        config = AlertGenerationConfig(cooldown_seconds=0.0)  # zero cooldown for direct unit tests
        return EarlyWarningEngine(config=config)

    def test_high_risk_alert_generation(self, engine):
        summary = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH)
        alerts = engine.evaluate(summary)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.severity == AlertSeverity.WARNING
        assert a.alert_type == AlertType.THRESHOLD_BREACH
        assert a.risk_level == RiskLevel.HIGH
        assert a.rule_id == "RULE_HIGH_BREACH"
        assert "breached HIGH threshold" in a.trigger_reason
        assert a.calibration_status == "PROVISIONAL"

    def test_critical_risk_alert_generation(self, engine):
        summary = make_summary(current_score=0.92, current_risk_level=RiskLevel.CRITICAL)
        alerts = engine.evaluate(summary)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.severity == AlertSeverity.CRITICAL
        assert a.alert_type == AlertType.THRESHOLD_BREACH
        assert a.risk_level == RiskLevel.CRITICAL
        assert a.rule_id == "RULE_CRITICAL_BREACH"
        assert "breached CRITICAL threshold" in a.trigger_reason

    def test_medium_risk_informational_warning(self, engine):
        # min_consecutive_for_medium_alert is 2
        summary = make_summary(current_score=0.45, current_risk_level=RiskLevel.MEDIUM, observation_count=3)
        alerts = engine.evaluate(summary)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.severity == AlertSeverity.INFO
        assert a.rule_id == "RULE_MEDIUM_ADVISORY"
        assert a.risk_level == RiskLevel.MEDIUM

    def test_rapid_degradation_trend_warning(self, engine):
        # min_consecutive_increasing_for_trend_alert is 3
        summary = make_summary(
            current_score=0.55,
            current_risk_level=RiskLevel.MEDIUM,
            trend_direction=RiskTrendDirection.INCREASING,
            consecutive_increasing=4,
        )
        alerts = engine.evaluate(summary)
        # Should generate advisory (if consecutive >= 2) and rapid degradation
        rule_ids = [a.rule_id for a in alerts]
        assert "RULE_RAPID_DEGRADATION" in rule_ids
        trend_alert = next(a for a in alerts if a.rule_id == "RULE_RAPID_DEGRADATION")
        assert trend_alert.severity == AlertSeverity.TREND_WARNING
        assert trend_alert.alert_type == AlertType.RAPID_DEGRADATION
        assert "Rapid perception degradation" in trend_alert.warning_message

    def test_state_escalation_alert(self, engine):
        summary = make_summary(current_score=0.72, current_risk_level=RiskLevel.HIGH)
        transition = StateTransitionEvent(
            sequence_id="seq_test",
            horizon=1,
            previous_level=RiskLevel.MEDIUM,
            current_level=RiskLevel.HIGH,
            timestamp_seconds=0.500,
            risk_score=0.72,
            is_genuine_transition=True,
        )
        alerts = engine.evaluate(summary, transition=transition)
        rule_ids = [a.rule_id for a in alerts]
        assert "RULE_STATE_ESCALATION" in rule_ids
        esc_alert = next(a for a in alerts if a.rule_id == "RULE_STATE_ESCALATION")
        assert esc_alert.alert_type == AlertType.STATE_TRANSITION
        assert esc_alert.severity == AlertSeverity.WARNING
        assert "escalated: MEDIUM -> HIGH" in esc_alert.warning_message


class TestMultiHorizonAndIsolation:
    """Tests 6, 7, 8: Independent H1/H3/H5 alerts, horizon isolation, sequence isolation."""

    @pytest.fixture
    def engine(self):
        return EarlyWarningEngine(config=AlertGenerationConfig(cooldown_seconds=0.0))

    def test_horizon_isolation(self, engine):
        # H1 is nominal LOW
        s_h1 = make_summary(sequence_id="seq_iso", horizon=1, current_score=0.15, current_risk_level=RiskLevel.LOW)
        # H5 is CRITICAL
        s_h5 = make_summary(sequence_id="seq_iso", horizon=5, current_score=0.90, current_risk_level=RiskLevel.CRITICAL)

        alerts_h1 = engine.evaluate(s_h1)
        alerts_h5 = engine.evaluate(s_h5)

        assert len(alerts_h1) == 0
        assert len(alerts_h5) == 1
        assert alerts_h5[0].trigger_horizon == 5
        assert alerts_h5[0].severity == AlertSeverity.CRITICAL

    def test_sequence_isolation(self, engine):
        # Sequence A is HIGH
        s_a = make_summary(sequence_id="seq_A", horizon=3, current_score=0.75, current_risk_level=RiskLevel.HIGH)
        # Sequence B is LOW
        s_b = make_summary(sequence_id="seq_B", horizon=3, current_score=0.20, current_risk_level=RiskLevel.LOW)

        alerts_a = engine.evaluate(s_a)
        alerts_b = engine.evaluate(s_b)

        assert len(alerts_a) == 1
        assert alerts_a[0].sequence_id == "seq_A"
        assert len(alerts_b) == 0

        active_a = engine.get_active_alerts(sequence_id="seq_A")
        active_b = engine.get_active_alerts(sequence_id="seq_B")
        assert len(active_a) == 1
        assert len(active_b) == 0


class TestAlertDeduplicationAndLifecycle:
    """Tests 10 to 14: Deduplication, cooldown, acknowledgement, resolution, expiration."""

    @pytest.fixture
    def engine(self):
        config = AlertGenerationConfig(
            cooldown_seconds=1.0,
            enable_critical_escalation=True,
            alert_ttl_seconds=5.0,
        )
        return EarlyWarningEngine(config=config)

    def test_cooldown_suppresses_duplicate_alerts(self, engine):
        # Frame 1 at t=0.0 -> Generates HIGH breach alert
        s1 = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.0)
        alerts_1 = engine.evaluate(s1)
        assert len(alerts_1) == 1

        # Frame 2 at t=0.2 (within 1.0s cooldown) -> Suppressed!
        s2 = make_summary(current_score=0.78, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.2)
        alerts_2 = engine.evaluate(s2)
        assert len(alerts_2) == 0  # Deduplicated under cooldown

        # Frame 3 at t=1.2 (outside 1.0s cooldown) -> Generates new alert!
        s3 = make_summary(current_score=0.80, current_risk_level=RiskLevel.HIGH, timestamp_seconds=1.2)
        alerts_3 = engine.evaluate(s3)
        assert len(alerts_3) == 1

    def test_critical_escalation_bypasses_cooldown(self, engine):
        # Frame 1 at t=0.0 -> HIGH alert (WARNING severity)
        s1 = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.0)
        alerts_1 = engine.evaluate(s1)
        assert alerts_1[0].severity == AlertSeverity.WARNING

        # Frame 2 at t=0.3 -> Escalates to CRITICAL (CRITICAL severity) within cooldown!
        s2 = make_summary(current_score=0.92, current_risk_level=RiskLevel.CRITICAL, timestamp_seconds=0.3)
        alerts_2 = engine.evaluate(s2)
        assert len(alerts_2) == 1
        assert alerts_2[0].severity == AlertSeverity.CRITICAL

    def test_alert_acknowledgement_lifecycle(self, engine):
        s1 = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.0)
        alerts = engine.evaluate(s1)
        alert_id = alerts[0].alert_id

        assert alerts[0].lifecycle_state == AlertLifecycleState.ACTIVE
        assert alerts[0].acknowledgement_status == "UNACKNOWLEDGED"

        # Operator acknowledges
        success = engine.acknowledge_alert(alert_id, operator_id="SafetyOperator_1")
        assert success is True
        assert alerts[0].lifecycle_state == AlertLifecycleState.ACKNOWLEDGED
        assert alerts[0].acknowledgement_status == "ACKNOWLEDGED"

    def test_auto_resolution_on_nominal_low_risk(self, engine):
        # Ingest HIGH alert
        s_high = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.0)
        alerts = engine.evaluate(s_high)
        assert len(engine.get_active_alerts()) == 1

        # Ingest LOW risk observation -> Auto-resolves!
        s_low = make_summary(current_score=0.15, current_risk_level=RiskLevel.LOW, timestamp_seconds=0.5)
        engine.evaluate(s_low)

        assert len(engine.get_active_alerts()) == 0
        events = engine.get_lifecycle_events()
        assert any(e.to_state == AlertLifecycleState.RESOLVED for e in events)

    def test_alert_expiration_after_ttl(self, engine):
        # Ingest alert at t=0.0
        s1 = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.0)
        engine.evaluate(s1)
        assert len(engine.get_active_alerts()) == 1

        # Evaluate at t=6.0 (> 5.0s TTL) -> Alert expires
        s_far = make_summary(current_score=0.10, current_risk_level=RiskLevel.LOW, timestamp_seconds=6.0)
        engine.evaluate(s_far)

        assert len(engine.get_active_alerts()) == 0
        events = engine.get_lifecycle_events()
        assert any(e.to_state == AlertLifecycleState.EXPIRED for e in events)


class TestMissingStaleAndEdgeCases:
    """Tests 15 to 22: Stale telemetry, missing data safety, uncalibrated/provisional status."""

    @pytest.fixture
    def engine(self):
        return EarlyWarningEngine(config=AlertGenerationConfig(cooldown_seconds=0.0))

    def test_stale_prediction_warning(self, engine):
        s_stale = make_summary(current_score=0.20, current_risk_level=RiskLevel.LOW, is_stale=True)
        alerts = engine.evaluate(s_stale)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.severity == AlertSeverity.DATA_QUALITY_WARNING
        assert a.alert_type == AlertType.DATA_QUALITY
        assert a.rule_id == "RULE_DATA_QUALITY"
        assert "stale" in a.trigger_reason.lower()

    def test_missing_data_never_generates_low_risk_alert(self, engine):
        """CRITICAL: missing data must NOT generate normal LOW risk or resolve alerts."""
        s_miss = make_summary(current_score=float("nan"), current_risk_level=RiskLevel.UNKNOWN)
        alerts = engine.evaluate(s_miss)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.DATA_QUALITY_WARNING
        assert alerts[0].risk_level == RiskLevel.UNKNOWN
        assert alerts[0].risk_level != RiskLevel.LOW

    def test_missing_data_does_not_auto_resolve_active_alerts(self, engine):
        """If data drops, active high-risk alerts must remain ACTIVE (not auto-resolved)."""
        # First: HIGH alert active
        s_high = make_summary(current_score=0.75, current_risk_level=RiskLevel.HIGH, timestamp_seconds=0.0)
        engine.evaluate(s_high)
        assert len(engine.get_active_alerts()) == 1

        # Second: telemetry drops (UNKNOWN)
        s_miss = make_summary(current_score=float("nan"), current_risk_level=RiskLevel.UNKNOWN, timestamp_seconds=0.5)
        engine.evaluate(s_miss)

        # Active HIGH alert must STILL be present
        active = engine.get_active_alerts()
        assert any(a.rule_id == "RULE_HIGH_BREACH" for a in active)

    def test_uncalibrated_provisional_status_preserved(self, engine):
        summary = make_summary(calibration_status="PROVISIONAL")
        alerts = engine.evaluate(summary)
        assert alerts[0].calibration_status == "PROVISIONAL"

    def test_explanation_and_traceability(self, engine):
        summary = make_summary(sequence_id="seq_trace_01", horizon=3, current_score=0.89, current_risk_level=RiskLevel.CRITICAL)
        alerts = engine.evaluate(summary)
        a = alerts[0]
        assert a.sequence_id == "seq_trace_01"
        assert a.forecast_horizon == 3
        assert "CRITICAL threshold" in a.trigger_reason
        assert "0.89" in a.trigger_reason
        assert a.rule_id == "RULE_CRITICAL_BREACH"


class TestEndToEndIntegrations:
    """Tests 23 to 26: Integration with Task 2 (RiskScorer), Task 3 (RiskClassifier), and Task 4 (Monitor)."""

    def test_end_to_end_pipeline_integration(self):
        """Stream Task 2 HorizonRiskRecord through Task 3, Task 4 Monitor, and Task 5 EarlyWarningEngine."""
        monitor = TemporalRiskMonitor()
        engine = EarlyWarningEngine(config=AlertGenerationConfig(cooldown_seconds=0.0))

        # 1. Nominal Frame at t=0.0
        rec_nominal = HorizonRiskRecord(
            prediction_id="p1",
            sequence_id="seq_pipe_001",
            source_frame_id="frame_0001",
            target_frame_id="frame_0004",
            horizon=3,
            predicted_reliability=0.85,
            risk_score=0.15,
            target_timestamp_seconds=0.100,
        )
        sum_1 = monitor.update(rec_nominal)
        alerts_1 = engine.evaluate(sum_1)
        assert len(alerts_1) == 0  # LOW risk -> no alert

        # 2. Elevated Risk Frame at t=0.2 (breaches HIGH)
        rec_elevated = HorizonRiskRecord(
            prediction_id="p2",
            sequence_id="seq_pipe_001",
            source_frame_id="frame_0002",
            target_frame_id="frame_0005",
            horizon=3,
            predicted_reliability=0.25,
            risk_score=0.75,
            target_timestamp_seconds=0.200,
        )
        sum_2 = monitor.update(rec_elevated)
        transitions = monitor.get_state_transitions(sequence_id="seq_pipe_001", horizon=3)
        recent_trans = transitions[-1] if transitions else None

        alerts_2 = engine.evaluate(sum_2, transition=recent_trans, source_record=monitor.get_history("seq_pipe_001", 3)[-1])
        assert len(alerts_2) >= 1
        severities = [a.severity for a in alerts_2]
        assert AlertSeverity.WARNING in severities

        # 3. Critical Failure Frame at t=0.3 (breaches CRITICAL)
        rec_critical = HorizonRiskRecord(
            prediction_id="p3",
            sequence_id="seq_pipe_001",
            source_frame_id="frame_0003",
            target_frame_id="frame_0006",
            horizon=3,
            predicted_reliability=0.05,
            risk_score=0.95,
            target_timestamp_seconds=0.300,
        )
        sum_3 = monitor.update(rec_critical)
        transitions_3 = monitor.get_state_transitions(sequence_id="seq_pipe_001", horizon=3)
        recent_trans_3 = transitions_3[-1] if transitions_3 else None

        alerts_3 = engine.evaluate(sum_3, transition=recent_trans_3, source_record=monitor.get_history("seq_pipe_001", 3)[-1])
        severities_3 = [a.severity for a in alerts_3]
        assert AlertSeverity.CRITICAL in severities_3

    def test_stream_real_predictions_to_alerts(self):
        """Integration with authentic project predictions from predictions.json."""
        import json
        pred_path = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")
        if not pred_path.exists():
            pytest.skip("Predictions file not found")

        with open(pred_path, "r", encoding="utf-8") as f:
            raw_preds = json.load(f)

        monitor = TemporalRiskMonitor()
        engine = EarlyWarningEngine(config=AlertGenerationConfig(cooldown_seconds=1.0))

        total_alerts = 0
        for p in raw_preds:
            risk_score = 1.0 - p["predicted_reliability"]
            t_raw = p.get("target_timestamp_seconds")
            if t_raw is not None:
                t_sec = float(t_raw)
            else:
                idx = p.get("target_source_frame_index") or p.get("input_end_source_frame_index") or 0
                t_sec = float(idx) * 0.0333333

            summary = monitor.update({
                "sequence_id": p["sequence_id"],
                "horizon": p["horizon"],
                "risk_score": risk_score,
                "timestamp_seconds": t_sec,
                "source_frame_id": p.get("input_end_frame", ""),
                "target_frame_id": p.get("target_frame_reference", ""),
                "prediction_id": f"{p['sequence_id']}_h{p['horizon']}",
                "checkpoint_id": p.get("checkpoint_id", "official_lstm_training"),
            })
            alerts = engine.evaluate(summary)
            total_alerts += len(alerts)

        # On zero-inflated validation driving clip, forecaster predicted high risks,
        # so early warning engine should appropriately produce critical alerts under cooldown.
        assert total_alerts > 0
        active_alerts = engine.get_active_alerts()
        assert len(active_alerts) > 0
        assert all(a.calibration_status == "PROVISIONAL" for a in active_alerts)
