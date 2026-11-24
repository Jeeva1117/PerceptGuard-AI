"""Production build compatibility and verification tests for the React frontend."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class TestReactFrontendProductionBuild:
    """Validate that React + TypeScript frontend builds cleanly and contains all five core features."""

    def test_frontend_dist_artifacts_exist(self):
        """Verify frontend dist build artifacts exist and are non-empty."""
        dist_dir = FRONTEND_DIR / "dist"
        index_html = dist_dir / "index.html"
        assert index_html.exists(), "frontend/dist/index.html does not exist. Run: npm run build --prefix frontend"
        html_content = index_html.read_text(encoding="utf-8")
        assert '<div id="root">' in html_content, "index.html missing root container"

        assets_dir = dist_dir / "assets"
        assert assets_dir.exists(), "frontend/dist/assets does not exist"
        js_files = list(assets_dir.glob("*.js"))
        css_files = list(assets_dir.glob("*.css"))
        assert len(js_files) >= 1, "At least one JavaScript bundle should exist in dist/assets"
        assert len(css_files) >= 1, "At least one CSS bundle should exist in dist/assets"

    def test_five_primary_feature_pages_exist(self):
        """Verify all five primary ADAS sidebar feature pages exist in frontend/src/pages/."""
        pages_dir = FRONTEND_DIR / "src" / "pages"
        expected_pages = [
            "CameraWeatherPage.tsx",
            "ObjectDetectionPage.tsx",
            "TrajectoryPredictionPage.tsx",
            "CollisionRiskPage.tsx",
            "SafetyAlertsPage.tsx",
        ]
        for page_name in expected_pages:
            page_file = pages_dir / page_name
            assert page_file.exists(), f"Feature page {page_name} missing in frontend/src/pages/"

    def test_package_json_dependencies(self):
        """Verify package.json contains required production packages."""
        pkg_json_file = FRONTEND_DIR / "package.json"
        assert pkg_json_file.exists(), "frontend/package.json missing"
        with open(pkg_json_file, "r", encoding="utf-8") as f:
            pkg = json.load(f)

        deps = pkg.get("dependencies", {})
        assert "react" in deps, "react missing in dependencies"
        assert "react-dom" in deps, "react-dom missing in dependencies"
        assert "react-router-dom" in deps, "react-router-dom missing in dependencies"
        assert "lucide-react" in deps, "lucide-react missing in dependencies"
        assert "recharts" in deps, "recharts missing in dependencies"

    def test_backend_api_five_feature_routes_registered(self):
        """Verify FastAPI router registers all dedicated endpoints for the 5 product features."""
        from src.api.app import create_app
        app = create_app()
        routes = list(app.openapi()["paths"].keys())

        expected_endpoints = [
            "/api/v1/video/jobs",
            "/api/v1/video/jobs/{job_id}",
            "/api/v1/video/jobs/{job_id}/results",
            "/api/v1/video/jobs/{job_id}/weather",
            "/api/v1/video/jobs/{job_id}/trajectories",
            "/api/v1/video/jobs/{job_id}/collision-risk",
            "/api/v1/video/jobs/{job_id}/safety-recommendations",
            "/api/v1/video/jobs/{job_id}/frames/{frame_name}",
        ]
        for ep in expected_endpoints:
            assert ep in routes, f"Endpoint {ep} not registered in FastAPI routes: {routes}"
