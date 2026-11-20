"""Alert query and lifecycle action routes."""

from __future__ import annotations

import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query, status

from src.api.dependencies import get_dashboard_service
from src.api.exceptions import InvalidInputError, ResourceNotFoundError
from src.api.schemas import (
    AcknowledgeAlertRequest,
    AlertActionResponse,
    AlertDetailResponse,
    AlertListResponse,
    ResolveAlertRequest,
)
from src.early_warning.dashboard_service import DashboardService

router = APIRouter(prefix="/alerts", tags=["Alerts & Incidents"])

ALLOWED_SEVERITIES = {"CRITICAL", "WARNING", "CAUTION", "INFO", "DATA_QUALITY_WARNING"}
ALLOWED_STATUSES = {"ACTIVE", "ACKNOWLEDGED", "RESOLVED", "EXPIRED"}


@router.get("", response_model=AlertListResponse)
def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity level."),
    horizon: Optional[int] = Query(None, description="Filter by prediction horizon (1, 3, 5)."),
    alert_type: Optional[str] = Query(None, description="Filter by alert type classification."),
    status: Optional[str] = Query(None, description="Filter by lifecycle status."),
    page: int = Query(1, ge=1, description="Page index (1-based)."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page."),
    service: DashboardService = Depends(get_dashboard_service),
) -> AlertListResponse:
    """Query paginated, filtered early-warning alert notifications."""
    if severity is not None and severity.upper() not in ALLOWED_SEVERITIES:
        raise InvalidInputError(
            f"Invalid severity '{severity}'. Must be one of: {sorted(ALLOWED_SEVERITIES)}"
        )
    if horizon is not None and horizon not in (1, 3, 5):
        raise InvalidInputError(f"Invalid horizon {horizon}. Must be one of: [1, 3, 5]")
    if status is not None and status.upper() not in ALLOWED_STATUSES:
        raise InvalidInputError(
            f"Invalid status '{status}'. Must be one of: {sorted(ALLOWED_STATUSES)}"
        )

    raw = service.get_alerts(
        severity=severity.upper() if severity else None,
        horizon=horizon,
        alert_type=alert_type,
        status=status.upper() if status else None,
        page=page,
        page_size=page_size,
    )
    return AlertListResponse(**raw)


@router.get("/{alert_id}", response_model=AlertDetailResponse)
def get_alert_detail(
    alert_id: str,
    service: DashboardService = Depends(get_dashboard_service),
) -> AlertDetailResponse:
    """Retrieve detailed metadata, evidence explanation, and audit trail for an alert."""
    detail = service.get_alert_detail(alert_id=alert_id)
    if not detail:
        raise ResourceNotFoundError(f"Alert with ID '{alert_id}' was not found.")
    return AlertDetailResponse(**detail)


@router.post("/{alert_id}/acknowledge", response_model=AlertActionResponse)
def acknowledge_alert(
    alert_id: str,
    payload: AcknowledgeAlertRequest,
    service: DashboardService = Depends(get_dashboard_service),
) -> AlertActionResponse:
    """Acknowledge an active alert notification."""
    result = service.acknowledge_alert(alert_id=alert_id, operator_id=payload.operator_id)
    if not result.get("acknowledged", False):
        raise ResourceNotFoundError(
            f"Alert '{alert_id}' was not found or is not in an acknowledgeable state."
        )
    return AlertActionResponse(
        success=True,
        alert_id=alert_id,
        new_state="ACKNOWLEDGED",
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.post("/{alert_id}/resolve", response_model=AlertActionResponse)
def resolve_alert(
    alert_id: str,
    payload: ResolveAlertRequest,
    service: DashboardService = Depends(get_dashboard_service),
) -> AlertActionResponse:
    """Resolve an active or acknowledged alert with required audit justification."""
    result = service.resolve_alert(
        alert_id=alert_id, reason=payload.reason, operator_id=payload.operator_id
    )
    if not result.get("resolved", False):
        raise ResourceNotFoundError(
            f"Alert '{alert_id}' was not found or could not be transitioned to RESOLVED."
        )
    return AlertActionResponse(
        success=True,
        alert_id=alert_id,
        new_state="RESOLVED",
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
