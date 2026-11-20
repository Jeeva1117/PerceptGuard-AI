"""Production build compatibility and static compilation tests for frontend."""

from __future__ import annotations

import importlib
from pathlib import Path
import py_compile
import pytest


DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"


class TestFrontendProductionBuild:
    """Validate that all frontend files compile without syntax or bytecode errors."""

    def test_py_compile_all_dashboard_modules(self):
        """Verify py_compile succeeds on all dashboard Python files."""
        dashboard_files = list(DASHBOARD_DIR.glob("**/*.py"))
        assert len(dashboard_files) >= 5, f"Expected at least 5 dashboard files, found {len(dashboard_files)}"

        for file_path in dashboard_files:
            # doraise=True ensures an exception is raised on compilation failure
            compiled = py_compile.compile(str(file_path), doraise=True)
            assert compiled is not None, f"Failed to compile {file_path}"

    def test_required_frontend_dependencies_available(self):
        """Verify essential frontend packages are importable and functional."""
        required_modules = [
            "streamlit",
            "httpx",
            "pandas",
            "pydantic",
        ]
        for mod_name in required_modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Module {mod_name} could not be imported"

    def test_dashboard_components_public_api(self):
        """Verify all advertised components in dashboard.components are importable."""
        from dashboard import components

        expected_symbols = [
            "render_empty_state",
            "render_error_boundary",
            "render_stale_warning",
            "render_header",
            "render_system_overview",
            "render_risk_overview",
            "render_temporal_monitoring",
            "render_early_warning_alerts",
            "render_detection_tracking",
            "render_pipeline_status_accordion",
        ]
        for symbol in expected_symbols:
            assert hasattr(components, symbol), f"Symbol {symbol} missing in dashboard.components"
