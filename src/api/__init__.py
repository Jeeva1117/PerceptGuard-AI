"""PerceptGuard AI — FastAPI REST API Package.

Provides production-hardened REST services for perception reliability forecasting,
early-warning alert management, system telemetry, and video ingestion.
"""

from src.api.app import create_app

__all__ = ["create_app"]
