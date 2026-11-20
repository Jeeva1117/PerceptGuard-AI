"""FastAPI dependency injection providers.

Supplies singleton configuration, pipeline integrator, and dashboard service
to route handlers in a testable, decoupled manner.
"""

from __future__ import annotations

from typing import Optional
from fastapi import Request

from src.config.system_settings import SystemSettings, get_settings
from src.early_warning.dashboard_service import DashboardService
from src.early_warning.pipeline_integrator import PipelineIntegrator


def get_current_settings(request: Request) -> SystemSettings:
    """Provide the application configuration singleton, preferring app.state.settings."""
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        return settings
    return get_settings()


def get_pipeline_integrator(request: Request) -> PipelineIntegrator:
    """Retrieve singleton PipelineIntegrator attached to application state."""
    integrator: Optional[PipelineIntegrator] = getattr(request.app.state, "integrator", None)
    if integrator is None:
        # Fallback for lightweight testing or isolated router invocations
        integrator = PipelineIntegrator()
        request.app.state.integrator = integrator
    return integrator


def get_dashboard_service(request: Request) -> DashboardService:
    """Retrieve singleton DashboardService from application state or integrator."""
    service: Optional[DashboardService] = getattr(request.app.state, "dashboard_service", None)
    if service is None:
        integrator = get_pipeline_integrator(request)
        service = integrator.dashboard_service
        request.app.state.dashboard_service = service
    return service
