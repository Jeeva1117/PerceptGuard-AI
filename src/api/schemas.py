"""Strongly typed Pydantic V2 schemas for API requests and responses.

Enforces strict input validation, consistent output contracts, numeric bounds,
and reliable error payloads across all endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------------------
# Standard Error Envelope Schemas
# ------------------------------------------------------------------------------

class StandardErrorDetail(BaseModel):
    """Detailed error payload contained within the API error response."""
    code: str = Field(..., description="Machine-readable error classification code.")
    message: str = Field(..., description="Human-readable summary of the error.")
    details: Optional[Any] = Field(None, description="Optional granular diagnostics or field errors.")


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by all error handlers."""
    error: StandardErrorDetail
    request_id: str = Field(..., description="Unique request tracing ID.")
    timestamp: str = Field(..., description="ISO 8601 UTC error timestamp.")


# ------------------------------------------------------------------------------
# System Health & Readiness Schemas
# ------------------------------------------------------------------------------

class LivenessResponse(BaseModel):
    """Response schema for container orchestrator liveness probe."""
    status: str = Field("alive", description="Process health flag.")
    timestamp: str = Field(..., description="Current UTC timestamp.")


class ReadinessResponse(BaseModel):
    """Response schema for readiness probe validating required dependencies."""
    status: str = Field(..., description="Overall readiness: 'ready', 'degraded', or 'unready'.")
    subsystems: Dict[str, str] = Field(..., description="Component availability map.")
    details: str = Field(..., description="Operational summary.")
    timestamp: str = Field(..., description="Current UTC timestamp.")


class OperationalMetricsResponse(BaseModel):
    """Response schema for operational process telemetry and SRE monitoring."""
    service: str = Field("perceptguard-backend", description="Service identifier.")
    uptime_seconds: float = Field(..., description="Process uptime in seconds.")
    memory_rss_mb: float = Field(..., description="Resident set memory consumption in megabytes.")
    memory_vms_mb: float = Field(..., description="Virtual memory consumption in megabytes.")
    cpu_percent: float = Field(..., description="Process CPU utilization percentage.")
    num_threads: int = Field(..., description="Active thread count.")
    status: str = Field("HEALTHY", description="Operational health status.")
    timestamp: str = Field(..., description="Current UTC timestamp.")


class HealthStatusResponse(BaseModel):
    """Comprehensive system operational health schema."""
    project_name: str
    pipeline_status: str
    overall_health: str
    health_score: float = Field(..., ge=0.0, le=100.0)
    current_sequence_id: str
    current_video_id: str
    last_prediction_timestamp_seconds: float
    data_freshness: str
    model_availability: Dict[str, str]
    calibration_gate_status: str
    timestamp: str


class ComponentStatusDetail(BaseModel):
    """Individual perception pipeline stage status."""
    status: str
    details: str


class PipelineStatusResponse(BaseModel):
    """8-stage perception pipeline status response."""
    components: Dict[str, ComponentStatusDetail]
    overall_readiness: str


# ------------------------------------------------------------------------------
# Reliability Risk Overview Schemas
# ------------------------------------------------------------------------------

class HorizonRiskDetail(BaseModel):
    """Individual horizon risk metrics."""
    horizon: int = Field(..., ge=1, le=10)
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    risk_level: str = Field("UNKNOWN", description="Risk classification level.")
    trend_direction: str = Field("UNKNOWN", description="Risk trend direction.")
    rolling_mean: Optional[float] = None
    rolling_max: Optional[float] = None
    rolling_min: Optional[float] = None
    volatility: Optional[float] = None
    observation_count: Optional[int] = 0
    timestamp_seconds: Optional[float] = None
    is_stale: Optional[bool] = False
    calibration_status: Optional[str] = "PROVISIONAL"
    model_config = ConfigDict(extra="allow")


class RiskOverviewResponse(BaseModel):
    """Multi-horizon reliability risk assessment summary."""
    sequence_id: str
    calibration_status: str
    calibration_notice: str
    horizons: Dict[str, HorizonRiskDetail]
    timestamp: str


# ------------------------------------------------------------------------------
# Temporal Monitoring Schemas
# ------------------------------------------------------------------------------

class TemporalRiskRecordSchema(BaseModel):
    """Historical risk point along a temporal sequence."""
    timestamp_seconds: float
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    is_valid: bool
    is_stale: bool
    source_frame_id: str


class StateTransitionRecordSchema(BaseModel):
    """Discrete operational risk level transition event."""
    sequence_id: str
    horizon: int
    previous_level: str
    current_level: str
    timestamp_seconds: float
    risk_score: float
    is_genuine_transition: bool
    notes: str


class TemporalMonitoringResponse(BaseModel):
    """Time-series telemetry response for charts and drift analysis."""
    sequence_id: str
    limit: int
    horizons: Dict[str, List[TemporalRiskRecordSchema]]
    state_transitions: List[StateTransitionRecordSchema]
    timestamp: str


# ------------------------------------------------------------------------------
# Alert Management Schemas
# ------------------------------------------------------------------------------

class AlertSummarySchema(BaseModel):
    """Summary of an early-warning alert incident."""
    alert_id: str
    sequence_id: str
    severity: str
    alert_type: str
    trigger_horizon: int
    risk_score: float
    lifecycle_state: str
    warning_message: str
    timestamp_seconds: Optional[float] = None
    target_timestamp_seconds: Optional[float] = None
    timestamp: Optional[Union[float, str]] = None
    created_at: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class AlertListResponse(BaseModel):
    """Paginated list of early-warning alerts."""
    total_count: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1)
    total_pages: int = Field(..., ge=0)
    alerts: List[AlertSummarySchema]
    active_alerts_count: int = Field(..., ge=0)


class AlertDetailResponse(AlertSummarySchema):
    """Complete alert record with audit history and context."""
    trigger_frame_index: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None
    lifecycle_events: Optional[List[Dict[str, Any]]] = None
    model_config = ConfigDict(extra="allow")


class AcknowledgeAlertRequest(BaseModel):
    """Payload to acknowledge an active alert."""
    operator_id: str = Field("operator", min_length=1, max_length=64, description="ID of acknowledging operator.")


class ResolveAlertRequest(BaseModel):
    """Payload to resolve an alert with mandatory reason."""
    reason: str = Field(..., min_length=3, max_length=256, description="Resolution justification.")
    operator_id: str = Field("operator", min_length=1, max_length=64, description="ID of resolving operator.")


class AlertActionResponse(BaseModel):
    """Confirmation of alert lifecycle state change."""
    success: bool
    alert_id: str
    new_state: str
    updated_at: str


# ------------------------------------------------------------------------------
# Detection & Tracking Schemas
# ------------------------------------------------------------------------------

class DetectionTrackingContextResponse(BaseModel):
    """Authentic detection and multi-object tracking operational telemetry."""
    status: str
    model_name: str
    tracker_type: str
    total_frames_processed: int
    total_detections: int
    total_unique_tracks: int
    max_simultaneous_active_tracks: int
    mean_confidence_overall: float
    mean_processing_latency_ms: float
    per_class_track_counts: Dict[str, int]
    is_synthetic: bool


# ------------------------------------------------------------------------------
# Video Upload & Processing Schemas
# ------------------------------------------------------------------------------

class VideoMetadataSchema(BaseModel):
    """Structural properties of an uploaded or ingested video file."""
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    size_bytes: int


class VideoValidationResponse(BaseModel):
    """Result of video file integrity and format verification."""
    is_valid: bool
    message: str
    metadata: Optional[VideoMetadataSchema] = None


class VideoProcessResponse(BaseModel):
    """Execution telemetry from processing a video sequence."""
    task_id: str
    status: str
    frames_processed: int
    latency_ms: float
    detections_count: int
    timestamp: str


class JobStatusResponse(BaseModel):
    """Real-time processing job lifecycle and progress record."""
    job_id: str
    filename: str
    status: str
    stage: str
    progress_percentage: float
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    max_frames_requested: int = 30
    frames_processed: int = 0
    total_detections: int = 0
    unique_tracks: int = 0
    alerts_count: int = 0
    error_message: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    video_path: Optional[str] = None
    has_source_video: bool = False
    rotation: int = 0
    analysis_mode: Optional[str] = "FULL_BALANCED"
    sample_fps: Optional[float] = 2.0
    timeline_telemetry: Optional[Dict[str, Any]] = None


class JobSubmissionResponse(BaseModel):
    """Immediate acknowledgement returned upon video upload and job dispatch."""
    job_id: str
    filename: str
    status: str
    stage: str
    message: str
    max_frames: int
    created_at: str
    analysis_mode: Optional[str] = "FULL_BALANCED"
    sample_fps: Optional[float] = None


class VideoUploadResponse(BaseModel):
    """Acknowledgement returned upon successful standalone video file upload."""
    video_id: str
    filename: str
    size_bytes: int
    format: str
    metadata: VideoMetadataSchema
    stream_url: str
    message: str = "Video successfully uploaded, validated, and ready for processing."


class StartProcessingRequest(BaseModel):
    """Payload to initiate analysis on an uploaded video."""
    video_id: str
    analysis_mode: str = "FULL_BALANCED"
    rotation: int = 0
    sample_fps: Optional[float] = None
    max_frames: Optional[int] = None


