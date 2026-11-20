"""API router declarations for PerceptGuard REST service."""

from src.api.routers.alerts import router as alerts_router
from src.api.routers.health import router as health_router
from src.api.routers.telemetry import router as telemetry_router
from src.api.routers.video import router as video_router

__all__ = [
    "health_router",
    "telemetry_router",
    "alerts_router",
    "video_router",
]
