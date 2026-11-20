"""Early Warning Data Contract Module for PerceptGuard AI.

Defines the standardized input data contract for the Predictive Early Warning
and Perception Risk Assessment system. Ingests, validates, and standardizes
multi-horizon forecasting outputs generated during Day 9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.early_warning.early_warning_config import EarlyWarningConfig


class EarlyWarningDataContractError(ValueError):
    """Raised when early warning input data contract validation fails."""
    pass


@dataclass
class EarlyWarningInputRecord:
    """Standardized input contract for early warning risk assessment.
    
    Represents a single prediction record consumable by the risk assessment engine,
    fully compatible with Day 9 HorizonPredictionRecord schema.
    """
    prediction_id: str
    sequence_id: str
    source_frame_id: str
    target_frame_id: str
    horizon: int
    predicted_reliability: float
    source_frame_index: Optional[int] = None
    target_frame_index: Optional[int] = None
    target_timestamp_seconds: Optional[float] = None
    track_id: Optional[int] = None
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "unknown"
    experiment_id: Optional[str] = None
    prediction_status: str = "success"
    inference_time_ms: float = 0.0
    alignment_method: str = "metadata_exact"
    dataset_id: Optional[str] = None
    video_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EarlyWarningInputRecord:
        """Construct record from raw dictionary, handling Day 9 field mappings."""
        if not isinstance(data, dict):
            raise EarlyWarningDataContractError(f"Expected dictionary input, got: {type(data)}")

        sequence_id = str(data.get("sequence_id", "")).strip()
        horizon_val = data.get("horizon")
        if horizon_val is None:
            raise EarlyWarningDataContractError("Missing mandatory field: 'horizon'")
        try:
            horizon = int(horizon_val)
        except (ValueError, TypeError):
            raise EarlyWarningDataContractError(f"Invalid horizon type: {horizon_val}")

        # Derive prediction_id if not present
        prediction_id = data.get("prediction_id")
        if not prediction_id:
            prediction_id = f"{sequence_id}_H{horizon}" if sequence_id else f"pred_H{horizon}"

        # Frame identifiers with fallback to Day 9 naming conventions
        source_frame_id = (
            data.get("source_frame_id")
            or data.get("input_end_frame")
            or data.get("source_frame_reference")
            or ""
        )
        target_frame_id = (
            data.get("target_frame_id")
            or data.get("target_frame_reference")
            or ""
        )

        source_frame_index = data.get("source_frame_index") or data.get("input_end_source_frame_index")
        if source_frame_index is not None:
            source_frame_index = int(source_frame_index)

        target_source_frame_index = data.get("target_frame_index") or data.get("target_source_frame_index")
        if target_source_frame_index is not None:
            target_source_frame_index = int(target_source_frame_index)

        target_timestamp = data.get("target_timestamp_seconds")
        if target_timestamp is not None:
            try:
                target_timestamp = float(target_timestamp)
            except (ValueError, TypeError):
                target_timestamp = None

        reliability_val = data.get("predicted_reliability")
        if reliability_val is None:
            raise EarlyWarningDataContractError("Missing mandatory field: 'predicted_reliability'")
        try:
            predicted_reliability = float(reliability_val)
        except (ValueError, TypeError):
            raise EarlyWarningDataContractError(f"Invalid predicted_reliability type: {reliability_val}")

        track_id_val = data.get("track_id")
        track_id = int(track_id_val) if track_id_val is not None else None

        inference_time_ms = float(data.get("inference_time_ms", 0.0) or 0.0)

        record = cls(
            prediction_id=str(prediction_id),
            sequence_id=str(sequence_id),
            source_frame_id=str(source_frame_id),
            target_frame_id=str(target_frame_id),
            horizon=horizon,
            predicted_reliability=predicted_reliability,
            source_frame_index=source_frame_index,
            target_frame_index=target_source_frame_index,
            target_timestamp_seconds=target_timestamp,
            track_id=track_id,
            model_name=str(data.get("model_name", "LSTMForecaster")),
            checkpoint_id=str(data.get("checkpoint_id", "unknown")),
            experiment_id=data.get("experiment_id"),
            prediction_status=str(data.get("prediction_status", "success")),
            inference_time_ms=inference_time_ms,
            alignment_method=str(data.get("alignment_method", "metadata_exact")),
            dataset_id=data.get("dataset_id"),
            video_id=data.get("video_id"),
        )
        return record

    @classmethod
    def from_day9_record(cls, record: Any) -> EarlyWarningInputRecord:
        """Construct directly from Day 9 HorizonPredictionRecord or dict."""
        if hasattr(record, "to_dict"):
            return cls.from_dict(record.to_dict())
        elif isinstance(record, dict):
            return cls.from_dict(record)
        raise EarlyWarningDataContractError(f"Unsupported record format: {type(record)}")

    def validate(self, allowed_horizons: Sequence[int] = (1, 3, 5)) -> List[str]:
        """Validate this record against the early warning data contract specification."""
        errors: List[str] = []

        if not self.prediction_id or not isinstance(self.prediction_id, str):
            errors.append("prediction_id must be a non-empty string")

        if not self.sequence_id or not isinstance(self.sequence_id, str):
            errors.append("sequence_id must be a non-empty string")

        if not self.source_frame_id or not isinstance(self.source_frame_id, str):
            errors.append("source_frame_id must be a non-empty string")

        if not self.target_frame_id or not isinstance(self.target_frame_id, str):
            errors.append("target_frame_id must be a non-empty string")

        if self.horizon not in allowed_horizons:
            errors.append(f"horizon {self.horizon} not in allowed horizons: {allowed_horizons}")

        # Reliability value bounds check
        if math.isnan(self.predicted_reliability) or math.isinf(self.predicted_reliability):
            errors.append("predicted_reliability cannot be NaN or Inf")
        elif not (0.0 <= self.predicted_reliability <= 1.0):
            errors.append(f"predicted_reliability must be in [0.0, 1.0], got: {self.predicted_reliability}")

        # Frame index chronological ordering check
        if self.source_frame_index is not None and self.target_frame_index is not None:
            if self.target_frame_index <= self.source_frame_index:
                errors.append(
                    f"target_frame_index ({self.target_frame_index}) must be strictly greater than "
                    f"source_frame_index ({self.source_frame_index}) for horizon {self.horizon}"
                )

        # Status validation
        valid_statuses = {"success", "failed", "unavailable", "degraded"}
        if self.prediction_status.lower() not in valid_statuses:
            errors.append(f"prediction_status '{self.prediction_status}' not in valid set: {valid_statuses}")

        return errors


@dataclass
class EarlyWarningDataContractValidationSummary:
    """Summary of early warning dataset/batch validation."""
    total_records: int
    valid_records: int
    invalid_records: int
    horizons_present: List[int]
    sequences_count: int
    validation_passed: bool
    errors_by_record: Dict[str, List[str]]
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EarlyWarningDataContract:
    """Validator and manager for early warning input contracts."""

    def __init__(self, config: Optional[EarlyWarningConfig] = None):
        self.config = config or EarlyWarningConfig()
        self.allowed_horizons = tuple(self.config.forecast_horizons.supported)

    def validate_record(self, record: EarlyWarningInputRecord) -> List[str]:
        """Validate an individual input record."""
        return record.validate(allowed_horizons=self.allowed_horizons)

    def validate_batch(
        self, records: Sequence[EarlyWarningInputRecord]
    ) -> EarlyWarningDataContractValidationSummary:
        """Validate a batch of input records and produce a structured summary."""
        total = len(records)
        valid = 0
        invalid = 0
        errors_by_record: Dict[str, List[str]] = {}
        horizons_seen: Set[int] = set()
        sequences_seen: Set[str] = set()

        for rec in records:
            sequences_seen.add(rec.sequence_id)
            horizons_seen.add(rec.horizon)
            errs = self.validate_record(rec)
            if errs:
                invalid += 1
                errors_by_record[rec.prediction_id] = errs
            else:
                valid += 1

        passed = (invalid == 0 and total > 0)
        return EarlyWarningDataContractValidationSummary(
            total_records=total,
            valid_records=valid,
            invalid_records=invalid,
            horizons_present=sorted(list(horizons_seen)),
            sequences_count=len(sequences_seen),
            validation_passed=passed,
            errors_by_record=errors_by_record,
        )

    def load_and_validate_file(
        self, path: Union[str, Path]
    ) -> Tuple[List[EarlyWarningInputRecord], EarlyWarningDataContractValidationSummary]:
        """Load records from a JSON file, parse, and validate."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise EarlyWarningDataContractError(f"Expected list of predictions in {p}, got: {type(raw_data)}")

        records: List[EarlyWarningInputRecord] = []
        for item in raw_data:
            records.append(EarlyWarningInputRecord.from_day9_record(item))

        summary = self.validate_batch(records)
        return records, summary

    @staticmethod
    def group_by_sequence(
        records: Sequence[EarlyWarningInputRecord]
    ) -> Dict[str, Dict[int, EarlyWarningInputRecord]]:
        """Group records by sequence_id and forecast horizon."""
        grouped: Dict[str, Dict[int, EarlyWarningInputRecord]] = {}
        for r in records:
            if r.sequence_id not in grouped:
                grouped[r.sequence_id] = {}
            grouped[r.sequence_id][r.horizon] = r
        return grouped
