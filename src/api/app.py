"""FastAPI application factory, lifecycle management, and route registration.

Integrates centralized settings, lifespan context management, defensive middleware,
and uniform exception handling for PerceptGuard AI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api.exceptions import register_exception_handlers
from src.api.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from src.api.routers import alerts_router, health_router, telemetry_router, video_router
from src.config.environment import EnvironmentType
from src.config.system_settings import SystemSettings, get_settings
from src.early_warning.pipeline_integrator import PipelineIntegrator

logger = logging.getLogger("perceptguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager controlling application initialization and teardown."""
    settings: SystemSettings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    app.state.startup_time = time.time()

    logger.info(
        f"Initializing PerceptGuard AI Backend: "
        f"env={settings.app.environment.value.upper()}, "
        f"version={settings.app.version}, "
        f"device={settings.model.device}"
    )

    # Initialize core singleton pipeline integrator
    try:
        integrator = PipelineIntegrator(root_dir=settings.base_dir)
        preds_file = (
            settings.base_dir
            / "experiments"
            / "forecasting"
            / "multi_horizon"
            / "official_multi_horizon_run"
            / "predictions.json"
        )
        if preds_file.exists():
            logger.info(f"Ingesting pre-verified multi-horizon predictions: {preds_file.name}")
            integrator.ingest_predictions_file(preds_file)
        else:
            logger.warning(
                f"Multi-horizon predictions artifact not found at {preds_file}. "
                f"Operating in replay/zero-state telemetry mode."
            )

        app.state.integrator = integrator
        app.state.dashboard_service = integrator.dashboard_service
        logger.info("Pipeline integrator and dashboard service initialized successfully.")

    except Exception as exc:
        logger.error(f"Critical error during application startup initialization: {exc}")
        app.state.integrator = None
        app.state.dashboard_service = None

    yield

    # Teardown & Resource Cleanup
    logger.info("Shutting down PerceptGuard AI Backend services...")
    app.state.integrator = None
    app.state.dashboard_service = None
    logger.info("Backend resources released cleanly.")


def create_app(settings: Optional[SystemSettings] = None) -> FastAPI:
    """Construct and configure a production-hardened FastAPI application instance."""
    resolved_settings = settings or get_settings()
    is_prod = resolved_settings.app.environment == EnvironmentType.PRODUCTION

    app = FastAPI(
        title=resolved_settings.app.app_name,
        version=resolved_settings.app.version,
        description=(
            "Autonomous Driving Perception Reliability Forecasting & Early Warning Backend API. "
            "Predicts detection failures (IoU degradation) across H1, H3, and H5 horizons."
        ),
        lifespan=lifespan,
        docs_url="/docs" if resolved_settings.app.debug else None,
        redoc_url="/redoc" if resolved_settings.app.debug else None,
        openapi_url="/openapi.json" if resolved_settings.app.debug else None,
    )

    # Attach settings to application state
    app.state.settings = resolved_settings

    # 1. Register security headers middleware
    app.add_middleware(SecurityHeadersMiddleware, is_production=is_prod)

    # 2. Register request tracking and access logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # 3. Register CORS middleware
    cors_origins = resolved_settings.api.cors_origins or ["http://localhost:8501"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 4. Register TrustedHostMiddleware if configured with explicit hosts
    allowed_hosts = resolved_settings.security.allowed_hosts
    if allowed_hosts and allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # 5. Register standardized exception handlers
    register_exception_handlers(app)

    # 6. Mount root health probe for container orchestrator conventions
    app.include_router(health_router)

    # 7. Mount all subsystem routers under configured API prefix (e.g. /api/v1)
    api_prefix = resolved_settings.api.prefix
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(telemetry_router, prefix=api_prefix)
    app.include_router(alerts_router, prefix=api_prefix)
    app.include_router(video_router, prefix=api_prefix)

    # 8. Root service metadata endpoint
    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": resolved_settings.app.app_name,
            "version": resolved_settings.app.version,
            "environment": resolved_settings.app.environment.value,
            "api_prefix": api_prefix,
            "status": "ONLINE",
            "calibration_gate": resolved_settings.dashboard.calibration_status_label,
        }

    return app


# Default ASGI application instance for Uvicorn / Gunicorn production runners
app = create_app()

