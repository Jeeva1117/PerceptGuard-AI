"""Health, liveness, and readiness monitoring routes."""

from __future__ import annotations

import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, Response, status

from src.api.dependencies import get_current_settings, get_dashboard_service
from src.api.schemas import HealthStatusResponse, LivenessResponse, ReadinessResponse
from src.config.system_settings import SystemSettings
from src.early_warning.dashboard_service import DashboardService

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health", response_model=HealthStatusResponse)
def get_health(
    service: DashboardService = Depends(get_dashboard_service),
) -> HealthStatusResponse:
    """Return comprehensive system health and model availability."""
    raw = service.get_system_health()
    return HealthStatusResponse(**raw)


@router.get("/health/live", response_model=LivenessResponse)
def get_liveness() -> LivenessResponse:
    """Kubernetes / orchestrator liveness probe confirming event loop responsiveness."""
    return LivenessResponse(
        status="alive",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
def get_readiness(
    response: Response,
    service: DashboardService = Depends(get_dashboard_service),
    settings: SystemSettings = Depends(get_current_settings),
) -> ReadinessResponse:
    """Readiness probe evaluating operational dependencies and checkpoint readiness."""
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    subsystems = {}

    # Check dashboard service initialization
    subsystems["dashboard_service"] = "READY" if service is not None else "UNAVAILABLE"

    # Check model checkpoints on disk
    yolo_path = settings.models_dir / "yolo" / settings.model.yolo_model
    yolo_ready = yolo_path.exists() or Path(settings.model.yolo_model).exists()
    subsystems["yolo_checkpoint"] = "READY" if yolo_ready else "AVAILABLE_DEFAULT"

    lstm_path = settings.model.lstm_model_path
    subsystems["lstm_checkpoint"] = "READY" if lstm_path.exists() else "UNAVAILABLE"

    # Check predictions artifact availability
    preds_ready = service.predictions_path.exists()
    subsystems["predictions_artifact"] = "READY" if preds_ready else "UNAVAILABLE"

    # Overall readiness assessment
    if not service:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="unready",
            subsystems=subsystems,
            details="Core dashboard service uninitialized.",
            timestamp=now_str,
        )

    if not preds_ready and not lstm_path.exists():
        response.status_code = status.HTTP_200_OK
        return ReadinessResponse(
            status="degraded",
            subsystems=subsystems,
            details="Service operational in fallback/zero-state mode. ML checkpoints pending volume mount.",
            timestamp=now_str,
        )

    return ReadinessResponse(
        status="ready",
        subsystems=subsystems,
        details="PerceptGuard AI backend ready for operational inference and telemetry queries.",
        timestamp=now_str,
    )
