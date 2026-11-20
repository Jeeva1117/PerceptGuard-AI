"""Unit and integration tests for TemporalRiskMonitor (Day 10 Task 4).

Verifies chronological tracking, duplicate and out-of-order detection,
independent sequence & horizon isolation, rolling statistics, trend direction,
consecutive increasing counts, state transition events, missing/stale data safety,
and integration with Task 2 (RiskScorer) and Task 3 (RiskClassifier).
"""

import math
from pathlib import Path
import pytest

from src.early_warning.interfaces import RiskLevel, RiskScore
from src.risk.risk_classifier import ClassificationResult, RiskClassifier
from src.risk.risk_config import RiskConfig, TemporalMonitoringConfig, ThresholdProfile
from src.risk.risk_models import AggregatedRiskRecord, HorizonRiskRecord
from src.risk.risk_validator import RiskValidationError
from src.risk.temporal_risk_monitor import (
    MonitoredRiskRecord,
    RiskTrendDirection,
    StateTransitionEvent,
    TemporalMonitoringError,
    TemporalRiskMonitor,
    TemporalRiskSummary,
)


class TestChronologicalTrackingAndIntegrity:
    """Tests 1, 2, 3: Chronological insertion, out-of-order rejection, duplicate handling."""

    @pytest.fixture
    def monitor(self):
        config = TemporalMonitoringConfig(
            history_window_size=10,
            deduplication_policy="reject",
            allow_out_of_order=False,
        )
        return TemporalRiskMonitor(config=config)

    def test_chronological_insertion(self, monitor):
        # Insert 3 sequential observations at t=0.0, 0.1, 0.2
        s1 = monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.0})
        s2 = monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.25, "timestamp_seconds": 0.1})
        s3 = monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.30, "timestamp_seconds": 0.2})

        history = monitor.get_history("seq_1", horizon=1)
        assert len(history) == 3
        assert [r.timestamp_seconds for r in history] == [0.0, 0.1, 0.2]
        assert [r.risk_score for r in history] == [0.20, 0.25, 0.30]
        assert s3.current_score == 0.30
        assert s3.observation_count == 3

    def test_out_of_order_rejection(self, monitor):
        monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.5})
        with pytest.raises(TemporalMonitoringError, match="Out-of-order timestamp"):
            monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.25, "timestamp_seconds": 0.4})

    def test_duplicate_timestamp_rejection(self, monitor):
        monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.5})
        with pytest.raises(TemporalMonitoringError, match="Duplicate timestamp"):
            monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.25, "timestamp_seconds": 0.5})

    def test_duplicate_timestamp_ignore_policy(self):
        config = TemporalMonitoringConfig(deduplication_policy="ignore")
        mon = TemporalRiskMonitor(config=config)
        mon.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.5})
        s2 = mon.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.80, "timestamp_seconds": 0.5})
        # Score remains 0.20 because duplicate was ignored
        assert s2.current_score == 0.20
        history = mon.get_history("seq_1", horizon=1)
        assert len(history) == 1


class TestHistoryIsolation:
    """Tests 4, 5: Independent sequence histories and independent horizon histories."""

    @pytest.fixture
    def monitor(self):
        return TemporalRiskMonitor()

    def test_independent_sequence_histories(self, monitor):
        # Sequence A
        monitor.update({"sequence_id": "seq_A", "horizon": 1, "risk_score": 0.10, "timestamp_seconds": 0.0})
        monitor.update({"sequence_id": "seq_A", "horizon": 1, "risk_score": 0.15, "timestamp_seconds": 0.1})

        # Sequence B
        monitor.update({"sequence_id": "seq_B", "horizon": 1, "risk_score": 0.90, "timestamp_seconds": 0.0})

        sum_A = monitor.get_summary("seq_A", horizon=1)
        sum_B = monitor.get_summary("seq_B", horizon=1)

        assert sum_A.observation_count == 2
        assert sum_A.current_score == 0.15
        assert sum_A.current_risk_level == RiskLevel.LOW

        assert sum_B.observation_count == 1
        assert sum_B.current_score == 0.90
        assert sum_B.current_risk_level == RiskLevel.CRITICAL

    def test_independent_horizon_histories(self, monitor):
        # Horizon 1
        monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.0})
        monitor.update({"sequence_id": "seq_1", "horizon": 1, "risk_score": 0.22, "timestamp_seconds": 0.1})

        # Horizon 5
        monitor.update({"sequence_id": "seq_1", "horizon": 5, "risk_score": 0.75, "timestamp_seconds": 0.0})

        hist_h1 = monitor.get_history("seq_1", horizon=1)
        hist_h5 = monitor.get_history("seq_1", horizon=5)

        assert len(hist_h1) == 2
        assert len(hist_h5) == 1
        assert hist_h1[-1].risk_score == 0.22
        assert hist_h5[-1].risk_score == 0.75


class TestRollingStatisticsAndTrends:
    """Tests 6, 7, 8: Rolling mean, min, max, volatility, trend direction, consecutive increasing."""

    @pytest.fixture
    def monitor(self):
        config = TemporalMonitoringConfig(history_window_size=5, trend_stability_threshold=0.01)
        return TemporalRiskMonitor(config=config)

    def test_rolling_mean_min_max_volatility(self, monitor):
        # Ingest: 0.2, 0.4, 0.6
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 1.0})
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.40, "timestamp_seconds": 1.1})
        s3 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.60, "timestamp_seconds": 1.2})

        assert math.isclose(s3.rolling_mean, 0.40, abs_tol=1e-5)
        assert math.isclose(s3.rolling_min, 0.20, abs_tol=1e-5)
        assert math.isclose(s3.rolling_max, 0.60, abs_tol=1e-5)
        assert s3.volatility is not None and s3.volatility > 0.0

    def test_trend_direction_and_consecutive_increasing(self, monitor):
        # 1. First observation: trend UNKNOWN
        s1 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.30, "timestamp_seconds": 0.0})
        assert s1.trend_direction == RiskTrendDirection.UNKNOWN
        assert s1.consecutive_increasing_count == 0

        # 2. Increase: 0.30 -> 0.35 (> 0.01 threshold)
        s2 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.35, "timestamp_seconds": 0.1})
        assert s2.trend_direction == RiskTrendDirection.INCREASING
        assert s2.consecutive_increasing_count == 1
        assert math.isclose(s2.absolute_change, 0.05, abs_tol=1e-5)

        # 3. Increase again: 0.35 -> 0.42
        s3 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.42, "timestamp_seconds": 0.2})
        assert s3.trend_direction == RiskTrendDirection.INCREASING
        assert s3.consecutive_increasing_count == 2

        # 4. Stable change: 0.42 -> 0.425 (diff = 0.005 < 0.01)
        s4 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.425, "timestamp_seconds": 0.3})
        assert s4.trend_direction == RiskTrendDirection.STABLE
        assert s4.consecutive_increasing_count == 0

        # 5. Decrease: 0.425 -> 0.38
        s5 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.38, "timestamp_seconds": 0.4})
        assert s5.trend_direction == RiskTrendDirection.DECREASING
        assert s5.consecutive_increasing_count == 0

    def test_window_overflow_eviction(self, monitor):
        # Window size is 5; insert 6 items
        for i in range(6):
            monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.10 * (i + 1), "timestamp_seconds": float(i)})

        history = monitor.get_history("s1", horizon=1)
        assert len(history) == 5
        # Oldest item (0.10) should have been evicted
        assert math.isclose(history[0].risk_score, 0.20, abs_tol=1e-5)
        assert math.isclose(history[-1].risk_score, 0.60, abs_tol=1e-5)


class TestStateTransitions:
    """Tests 9, 10: State transition detection, unrepeated transitions."""

    @pytest.fixture
    def monitor(self):
        return TemporalRiskMonitor()

    def test_state_transitions_recorded(self, monitor):
        # LOW (0.15)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.15, "timestamp_seconds": 0.0})
        # Transitions to MEDIUM (0.45)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.45, "timestamp_seconds": 0.1})
        # Transitions to HIGH (0.75)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.75, "timestamp_seconds": 0.2})
        # Transitions to CRITICAL (0.90)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.90, "timestamp_seconds": 0.3})

        transitions = monitor.get_state_transitions(sequence_id="s1", horizon=1)
        assert len(transitions) == 3

        assert transitions[0].previous_level == RiskLevel.LOW
        assert transitions[0].current_level == RiskLevel.MEDIUM

        assert transitions[1].previous_level == RiskLevel.MEDIUM
        assert transitions[1].current_level == RiskLevel.HIGH

        assert transitions[2].previous_level == RiskLevel.HIGH
        assert transitions[2].current_level == RiskLevel.CRITICAL

    def test_no_repeated_transitions_when_level_unchanged(self, monitor):
        # Ingest 3 consecutive HIGH scores (0.70, 0.72, 0.74)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.0})  # LOW
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.70, "timestamp_seconds": 0.1})  # HIGH (1 transition)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.72, "timestamp_seconds": 0.2})  # HIGH (no transition)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.74, "timestamp_seconds": 0.3})  # HIGH (no transition)

        transitions = monitor.get_state_transitions(sequence_id="s1", horizon=1)
        assert len(transitions) == 1
        assert transitions[0].previous_level == RiskLevel.LOW
        assert transitions[0].current_level == RiskLevel.HIGH


class TestMissingStaleAndEdgeCases:
    """Tests 11 to 18: Missing data, stale data, gaps, non-finite rejection, version changes, empty history."""

    @pytest.fixture
    def monitor(self):
        config = TemporalMonitoringConfig(max_timestamp_gap_seconds=1.0)
        return TemporalRiskMonitor(config=config)

    def test_missing_data_never_becomes_zero_risk(self, monitor):
        # Missing data marked via is_missing=True
        summary = monitor.update(
            {"sequence_id": "s_miss", "horizon": 1, "is_missing": True, "timestamp_seconds": 0.0}
        )
        assert summary.current_risk_level == RiskLevel.UNKNOWN
        assert summary.current_risk_level != RiskLevel.LOW
        assert math.isnan(summary.current_score)

    def test_stale_data_flagging(self, monitor):
        summary = monitor.update(
            {"sequence_id": "s_stale", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 0.0},
            is_stale=True,
        )
        assert summary.is_stale is True

    def test_large_timestamp_gap_handling(self, monitor):
        # t=0.0 -> t=0.1 -> t=5.0 (gap = 4.9s > 1.0s max gap)
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.30, "timestamp_seconds": 0.0})
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.35, "timestamp_seconds": 0.1})
        s3 = monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.40, "timestamp_seconds": 5.0})

        assert s3.has_timestamp_gap is True
        # Consecutive increasing counter was reset due to temporal discontinuity
        assert s3.consecutive_increasing_count == 0

    def test_reject_non_finite_risk_scores(self, monitor):
        with pytest.raises(RiskValidationError):
            monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": float("nan"), "timestamp_seconds": 0.0})

        with pytest.raises(RiskValidationError):
            monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": float("inf"), "timestamp_seconds": 0.0})

        with pytest.raises(RiskValidationError):
            monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": -0.1, "timestamp_seconds": 0.0})

    def test_checkpoint_version_change_detected(self, monitor):
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.30, "timestamp_seconds": 0.0, "checkpoint_id": "ckpt_v1"})
        monitor.update({"sequence_id": "s1", "horizon": 1, "risk_score": 0.32, "timestamp_seconds": 0.1, "checkpoint_id": "ckpt_v2"})

        tracker = monitor._get_tracker("s1", 1)
        assert tracker.checkpoint_change_detected is True

    def test_empty_history_query(self, monitor):
        summary = monitor.get_summary("non_existent_seq", horizon=1)
        assert summary is None
        history = monitor.get_history("non_existent_seq", horizon=1)
        assert history == []

    def test_insufficient_observations_handled(self, monitor):
        # 1 observation: insufficient for delta or volatility
        s = monitor.update({"sequence_id": "s_single", "horizon": 1, "risk_score": 0.50, "timestamp_seconds": 0.0})
        assert s.observation_count == 1
        assert s.previous_score is None
        assert s.absolute_change is None
        assert s.relative_change is None
        assert s.volatility == 0.0
        assert s.trend_direction == RiskTrendDirection.UNKNOWN

    def test_no_future_data_leakage(self, monitor):
        """Calculations at time T must strictly depend on observations at or before T."""
        s1 = monitor.update({"sequence_id": "s_leak", "horizon": 1, "risk_score": 0.20, "timestamp_seconds": 1.0})
        assert s1.rolling_mean == 0.20

        s2 = monitor.update({"sequence_id": "s_leak", "horizon": 1, "risk_score": 0.80, "timestamp_seconds": 2.0})
        assert math.isclose(s2.rolling_mean, 0.50, abs_tol=1e-5)

        # Re-querying at earlier sequence history state is isolated
        hist = monitor.get_history("s_leak", horizon=1)
        assert hist[0].risk_score == 0.20
        assert hist[1].risk_score == 0.80


class TestTask2And3Integration:
    """Tests 19, 20: Ingestion of Task 2 records and Task 3 classification results."""

    @pytest.fixture
    def monitor(self):
        return TemporalRiskMonitor()

    def test_ingest_horizon_risk_record(self, monitor):
        record = HorizonRiskRecord(
            prediction_id="p_1",
            sequence_id="seq_integ_001",
            source_frame_id="frame_0001",
            target_frame_id="frame_0004",
            horizon=3,
            predicted_reliability=0.15,
            risk_score=0.85,
            target_timestamp_seconds=0.1333,
            checkpoint_id="lstm_best_v1",
        )
        summary = monitor.update(record)
        assert summary.sequence_id == "seq_integ_001"
        assert summary.horizon == 3
        assert summary.current_score == 0.85
        assert summary.current_risk_level == RiskLevel.HIGH
        assert summary.checkpoint_id == "lstm_best_v1"

    def test_ingest_aggregated_risk_record(self, monitor):
        agg_record = AggregatedRiskRecord(
            sequence_id="seq_integ_002",
            source_frame_id="frame_0001",
            aggregated_risk_score=0.92,
            aggregation_strategy="weighted_linear",
            primary_horizon=1,
            horizon_risk_scores={1: 0.90, 3: 0.92, 5: 0.94},
            horizon_reliabilities={1: 0.10, 3: 0.08, 5: 0.06},
            weights_applied={1: 0.50, 3: 0.35, 5: 0.15},
            target_timestamp_seconds=0.2000,
        )
        summary = monitor.update(agg_record)
        assert summary.sequence_id == "seq_integ_002"
        assert summary.horizon == 1
        assert summary.current_score == 0.92
        assert summary.current_risk_level == RiskLevel.CRITICAL

    def test_ingest_classification_result(self, monitor):
        cls_result = ClassificationResult(
            sequence_id="seq_integ_003",
            horizon=5,
            risk_score=0.45,
            risk_level=RiskLevel.MEDIUM,
            target_timestamp_seconds=0.3333,
            calibration_status="PROVISIONAL",
        )
        summary = monitor.update(cls_result)
        assert summary.sequence_id == "seq_integ_003"
        assert summary.horizon == 5
        assert summary.current_score == 0.45
        assert summary.current_risk_level == RiskLevel.MEDIUM
        assert summary.calibration_status == "PROVISIONAL"

    def test_record_and_check_interface(self, monitor):
        """Test ITemporalRiskMonitor interface method record_and_check with persistence."""
        # Config has alert_persistence_threshold = 3
        r1 = RiskScore("seq_alert", 0.90, 1, RiskLevel.HIGH, target_timestamp_seconds=0.1)
        r2 = RiskScore("seq_alert", 0.92, 1, RiskLevel.HIGH, target_timestamp_seconds=0.2)
        r3 = RiskScore("seq_alert", 0.94, 1, RiskLevel.CRITICAL, target_timestamp_seconds=0.3)

        assert monitor.record_and_check(r1) is False  # Frame 1: not yet persisted
        assert monitor.record_and_check(r2) is False  # Frame 2: not yet persisted
        assert monitor.record_and_check(r3) is True   # Frame 3: 3 consecutive HIGH/CRITICAL frames!


class TestRealDatasetMonitoringIntegration:
    """Integration test: streaming real project predictions into TemporalRiskMonitor."""

    def test_stream_real_predictions(self):
        pred_path = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")
        if not pred_path.exists():
            pytest.skip("Predictions file not found")

        import json
        with open(pred_path, "r", encoding="utf-8") as f:
            raw_preds = json.load(f)

        monitor = TemporalRiskMonitor()
        ingested_count = 0
        for p in raw_preds:
            risk_score = 1.0 - p["predicted_reliability"]
            t_raw = p.get("target_timestamp_seconds")
            if t_raw is not None:
                t_sec = float(t_raw)
            else:
                idx = p.get("target_source_frame_index")
                if idx is None:
                    idx = p.get("input_end_source_frame_index") or 0
                t_sec = float(idx) * 0.0333333
            monitor.update({
                "sequence_id": p["sequence_id"],
                "horizon": p["horizon"],
                "risk_score": risk_score,
                "timestamp_seconds": t_sec,
                "source_frame_id": p.get("input_end_frame", ""),
                "target_frame_id": p.get("target_frame_reference", ""),
                "prediction_id": f"{p['sequence_id']}_h{p['horizon']}",
                "checkpoint_id": p.get("checkpoint_id", "lstm_best_official"),
            })
            ingested_count += 1

        assert ingested_count == len(raw_preds)
        seq_0 = raw_preds[0]["sequence_id"]
        summary_h1 = monitor.get_summary(seq_0, horizon=1)
        assert summary_h1 is not None
        assert summary_h1.current_risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
