"""Telemetry and operational state query routes."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_dashboard_service
from src.api.exceptions import InvalidInputError
from src.api.schemas import (
    DetectionTrackingContextResponse,
    HealthStatusResponse,
    PipelineStatusResponse,
    RiskOverviewResponse,
    TemporalMonitoringResponse,
)
from src.early_warning.dashboard_service import DashboardService

router = APIRouter(prefix="/telemetry", tags=["Telemetry & Monitoring"])


@router.get("/health", response_model=HealthStatusResponse)
def get_telemetry_health(
    service: DashboardService = Depends(get_dashboard_service),
) -> HealthStatusResponse:
    """Query current telemetry health status."""
    return HealthStatusResponse(**service.get_system_health())


@router.get("/pipeline-status", response_model=PipelineStatusResponse)
def get_pipeline_status(
    service: DashboardService = Depends(get_dashboard_service),
) -> PipelineStatusResponse:
    """Query 8-stage perception pipeline execution status."""
    return PipelineStatusResponse(**service.get_pipeline_status())


@router.get("/risk-overview", response_model=RiskOverviewResponse)
def get_risk_overview(
    sequence_id: Optional[str] = Query(None, description="Optional target sequence identifier."),
    service: DashboardService = Depends(get_dashboard_service),
) -> RiskOverviewResponse:
    """Retrieve multi-horizon failure risk scores (H1, H3, H5) and trends."""
    raw = service.get_reliability_risk_overview(sequence_id=sequence_id)
    return RiskOverviewResponse(**raw)


@router.get("/temporal-monitoring", response_model=TemporalMonitoringResponse)
def get_temporal_monitoring(
    sequence_id: Optional[str] = Query(None, description="Filter records by sequence."),
    horizon: Optional[int] = Query(None, description="Filter records by horizon (1, 3, or 5)."),
    limit: int = Query(50, ge=1, le=500, description="Maximum historical records per horizon."),
    service: DashboardService = Depends(get_dashboard_service),
) -> TemporalMonitoringResponse:
    """Retrieve historical risk trajectories and state transitions for drift analysis."""
    if horizon is not None and horizon not in (1, 3, 5):
        raise InvalidInputError(
            f"Invalid horizon {horizon}. PerceptGuard AI supports horizons: [1, 3, 5]."
        )
    raw = service.get_temporal_monitoring_data(
        sequence_id=sequence_id,
        horizon=horizon,
        limit=limit,
    )
    return TemporalMonitoringResponse(**raw)


@router.get("/detection-tracking", response_model=DetectionTrackingContextResponse)
def get_detection_tracking(
    sequence_id: Optional[str] = Query(None, description="Optional sequence filter."),
    service: DashboardService = Depends(get_dashboard_service),
) -> DetectionTrackingContextResponse:
    """Retrieve authentic YOLO detection and ByteTrack tracking context."""
    raw = service.get_detection_tracking_context(sequence_id=sequence_id)
    return DetectionTrackingContextResponse(**raw)
