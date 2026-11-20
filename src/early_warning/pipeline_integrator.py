"""End-to-End Pipeline Integrator for PerceptGuard AI.

Connects the verified perception reliability components into a unified execution flow:
Video / Tracking Context
  -> Multi-Horizon Forecasts (H1, H3, H5)
  -> Reliability Risk Scorer (Task 2)
  -> Risk Classifier & Calibrator (Task 3)
  -> Temporal Risk Monitor (Task 4)
  -> Early Warning Engine (Task 5)
  -> Dashboard Service (Task 6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.early_warning.dashboard_service import DashboardService
from src.early_warning.early_warning_config import EarlyWarningConfig
from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.early_warning.early_warning_engine import EarlyWarningEngine
from src.early_warning.interfaces import (
    AlertLifecycleState,
    AlertSeverity,
    AlertType,
    EarlyWarningAlert,
    RiskLevel,
    RiskScore,
)
from src.risk.risk_classifier import RiskClassifier
from src.risk.risk_config import RiskConfig
from src.risk.risk_models import HorizonRiskRecord
from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.temporal_risk_monitor import (
    MonitoredRiskRecord,
    StateTransitionEvent,
    TemporalRiskMonitor,
    TemporalRiskSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineProcessingResult:
    """Output metrics from processing a single prediction record through the pipeline."""
    prediction_id: str
    sequence_id: str
    horizon: int
    risk_score: float
    risk_level: RiskLevel
    trend_direction: str
    alerts_generated: List[EarlyWarningAlert]
    latency_ms: float
    timestamp_seconds: float


@dataclass
class PipelineBatchResult:
    """Summary of batch ingestion of multi-horizon predictions."""
    total_processed: int
    total_alerts_generated: int
    active_alerts_count: int
    per_horizon_processed: Dict[int, int]
    per_severity_alerts: Dict[str, int]
    mean_latency_ms: float
    total_latency_ms: float
    calibration_status: str = "PROVISIONAL"
    execution_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class PipelineIntegrator:
    """End-to-end integration orchestrator connecting perception forecasting to dashboard."""

    def __init__(
        self,
        config: Optional[EarlyWarningConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        scorer: Optional[ReliabilityRiskScorer] = None,
        classifier: Optional[RiskClassifier] = None,
        monitor: Optional[TemporalRiskMonitor] = None,
        engine: Optional[EarlyWarningEngine] = None,
        dashboard_service: Optional[DashboardService] = None,
        root_dir: Optional[Path] = None,
    ):
        self.config = config or EarlyWarningConfig()
        self.risk_config = risk_config or RiskConfig()
        self.scorer = scorer or ReliabilityRiskScorer(config=self.risk_config)
        self.classifier = classifier or RiskClassifier(config=self.risk_config)
        self.monitor = monitor or TemporalRiskMonitor(config=self.risk_config, classifier=self.classifier)
        self.engine = engine or EarlyWarningEngine(config=self.config)
        self.root_dir = root_dir or Path.cwd()
        self.dashboard_service = dashboard_service or DashboardService(
            config=self.config,
            risk_config=self.risk_config,
            monitor=self.monitor,
            engine=self.engine,
            root_dir=self.root_dir,
        )

        self._processed_count = 0
        self._total_alerts_count = 0

    def process_prediction(
        self,
        record: Union[Dict[str, Any], EarlyWarningInputRecord],
    ) -> PipelineProcessingResult:
        """Process a single forecast prediction through the complete pipeline.

        1. Transform predicted reliability to risk score (ReliabilityRiskScorer).
        2. Classify risk level against calibrated thresholds (RiskClassifier).
        3. Ingest into chronological temporal monitor (TemporalRiskMonitor).
        4. Evaluate early-warning rules and trigger alerts (EarlyWarningEngine).
        """
        t_start = time.perf_counter()

        # Parse dict if necessary
        if isinstance(record, dict):
            # Map predictions.json schema if input comes from raw file
            pred_id = record.get("prediction_id", f"pred_{self._processed_count:06d}")
            seq_id = record.get("sequence_id", "driving_sample")
            src_frame = record.get("input_end_frame", record.get("source_frame_id", "frame_000001.jpg"))
            tgt_frame = record.get("target_frame_reference", record.get("target_frame_id", "frame_000006.jpg"))
            horizon = int(record.get("horizon", 1))
            rel = float(record.get("predicted_reliability", 1.0))
            src_idx = record.get("input_end_source_frame_index") or record.get("source_frame_index") or 0
            tgt_idx = record.get("target_source_frame_index") or record.get("target_frame_index") or horizon
            raw_t = record.get("target_timestamp_seconds")
            if raw_t is None:
                raw_t = record.get("timestamp_seconds")
            if raw_t is None:
                raw_t = float(tgt_idx) * (1.0 / 30.0) if tgt_idx is not None else 0.0
            t_sec = float(raw_t)
            model_name = record.get("model_name", "LSTMForecaster")
            ckpt_id = record.get("checkpoint_id", "official_lstm_training")

            input_dict = {
                "prediction_id": pred_id,
                "sequence_id": seq_id,
                "source_frame_id": src_frame,
                "target_frame_id": tgt_frame,
                "horizon": horizon,
                "predicted_reliability": rel,
                "source_frame_index": src_idx,
                "target_frame_index": tgt_idx,
                "target_timestamp_seconds": t_sec,
                "model_name": model_name,
                "checkpoint_id": ckpt_id,
            }
        else:
            input_dict = record.to_dict()
            seq_id = record.sequence_id
            horizon = record.horizon
            pred_id = record.prediction_id
            t_sec = record.target_timestamp_seconds or 0.0

        # 1. Risk Scoring (Task 2)
        risk_record: HorizonRiskRecord = self.scorer.score_single_horizon(input_dict)

        # 2 & 3. Temporal Risk Monitoring & Classification (Task 3 & 4)
        summary: TemporalRiskSummary = self.monitor.update(risk_record)

        # Retrieve latest transition and record for engine evaluation
        tracker = self.monitor._get_tracker(risk_record.sequence_id, risk_record.horizon)
        latest_rec: Optional[MonitoredRiskRecord] = tracker.records[-1] if tracker.records else None
        latest_trans: Optional[StateTransitionEvent] = tracker.transitions[-1] if tracker.transitions else None

        # 4. Early Warning Alert Evaluation (Task 5)
        alerts: List[EarlyWarningAlert] = self.engine.evaluate(
            summary=summary,
            transition=latest_trans,
            source_record=latest_rec,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._processed_count += 1
        self._total_alerts_count += len(alerts)

        return PipelineProcessingResult(
            prediction_id=pred_id,
            sequence_id=seq_id,
            horizon=horizon,
            risk_score=risk_record.risk_score,
            risk_level=summary.current_risk_level,
            trend_direction=summary.trend_direction.value,
            alerts_generated=alerts,
            latency_ms=latency_ms,
            timestamp_seconds=t_sec,
        )

    def ingest_predictions_file(
        self,
        filepath: Optional[Path] = None,
        max_records: Optional[int] = None,
    ) -> PipelineBatchResult:
        """Ingest predictions from disk artifact through the end-to-end pipeline.

        Parameters
        ----------
        filepath : Optional[Path]
            Path to predictions.json. If None, defaults to official multi-horizon run.
        max_records : Optional[int]
            Maximum records to process (for bounded smoke tests or pagination).
        """
        path = filepath or (
            self.root_dir
            / "experiments"
            / "forecasting"
            / "multi_horizon"
            / "official_multi_horizon_run"
            / "predictions.json"
        )

        if not path.exists():
            raise FileNotFoundError(f"Predictions artifact not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            predictions: List[Dict[str, Any]] = json.load(f)

        if max_records is not None:
            predictions = predictions[:max_records]

        per_h = {1: 0, 3: 0, 5: 0}
        per_sev: Dict[str, int] = {}
        total_latency = 0.0
        alerts_total = 0

        for pred in predictions:
            res = self.process_prediction(pred)
            per_h[res.horizon] = per_h.get(res.horizon, 0) + 1
            total_latency += res.latency_ms
            alerts_total += len(res.alerts_generated)
            for alert in res.alerts_generated:
                sev_name = alert.severity.value
                per_sev[sev_name] = per_sev.get(sev_name, 0) + 1

        total_processed = len(predictions)
        mean_latency = total_latency / total_processed if total_processed > 0 else 0.0

        return PipelineBatchResult(
            total_processed=total_processed,
            total_alerts_generated=alerts_total,
            active_alerts_count=len(self.engine.get_active_alerts()),
            per_horizon_processed=per_h,
            per_severity_alerts=per_sev,
            mean_latency_ms=round(mean_latency, 4),
            total_latency_ms=round(total_latency, 2),
            calibration_status="PROVISIONAL",
        )

    def run_smoke_test(self, num_records: int = 15) -> Dict[str, Any]:
        """Execute a verified bounded smoke test across all integrated stages."""
        t0 = time.perf_counter()
        batch_result = self.ingest_predictions_file(max_records=num_records)
        health = self.dashboard_service.get_system_health()
        risk_overview = self.dashboard_service.get_reliability_risk_overview()
        temporal_data = self.dashboard_service.get_temporal_monitoring_data(limit=10)
        alerts_data = self.dashboard_service.get_alerts(page=1, page_size=10)
        tracking_context = self.dashboard_service.get_detection_tracking_context()

        smoke_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": "PASS" if batch_result.total_processed == num_records else "FAIL",
            "records_processed": batch_result.total_processed,
            "mean_pipeline_latency_ms": batch_result.mean_latency_ms,
            "smoke_total_latency_ms": round(smoke_latency_ms, 2),
            "system_health": health["pipeline_status"],
            "overall_health": health["overall_health"],
            "tracking_available": tracking_context["status"] == "AVAILABLE",
            "horizons_scored": list(risk_overview["horizons"].keys()),
            "alerts_in_history": alerts_data["total_count"],
            "calibration_status": "PROVISIONAL",
        }
