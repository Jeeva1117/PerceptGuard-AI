"""Tests for the Master Project Auditor (scripts/audit_project.py).

Validates individual audit checks, integration checks, data integrity checks,
CLI flags, and report generation across Days 1 through 6.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.audit_project import ProjectAuditor, _project_root


@pytest.fixture
def auditor() -> ProjectAuditor:
    """Fixture providing a ProjectAuditor instance rooted at the workspace."""
    return ProjectAuditor(root_dir=_project_root)


def test_audit_project_file_inventory(auditor: ProjectAuditor):
    """Verify that all Day 1-6 files are present and valid."""
    result = auditor.check_files()
    assert result is True, f"File inventory check failed with missing files: {auditor.missing_files}"
    assert len(auditor.missing_files) == 0


def test_audit_project_imports(auditor: ProjectAuditor):
    """Verify that all core modules and packages import without error."""
    result = auditor.check_imports()
    assert result is True, "Import check failed for one or more core packages"


def test_audit_project_configs(auditor: ProjectAuditor):
    """Verify that all 5 YAML configuration files parse with required schema keys."""
    result = auditor.check_configs()
    assert result is True, "Configuration check failed for one or more YAML files"


def test_audit_project_artifacts(auditor: ProjectAuditor):
    """Verify that processed video and Day 5 sequence artifacts exist and are finite."""
    result = auditor.check_artifacts()
    assert result is True, "Artifact check failed"


def test_audit_project_integration(auditor: ProjectAuditor):
    """Verify cross-module integrations from features to LSTM forecaster."""
    result = auditor.check_integration()
    assert result is True, "Cross-module integration check failed"
    assert len(auditor.integration_results) >= 3


def test_audit_project_data_integrity(auditor: ProjectAuditor):
    """Verify scientific ground-truth honesty and temporal embargo invariants."""
    result = auditor.check_data_integrity()
    assert result is True, "Data integrity check failed"
    assert auditor.data_integrity_results.get("ground_truth_policy") == "STRICT_HONESTY_ENFORCED"
    assert auditor.data_integrity_results.get("yolo_confidence_treated_as_ground_truth") is False
    assert auditor.data_integrity_results.get("leakage_check_passed") is True


def test_audit_project_full_run_and_reports(auditor: ProjectAuditor, tmp_path: Path):
    """Verify that full audit execution produces all 5 audit reports."""
    report_dir = tmp_path / "final_audit"
    result = auditor.run_full_audit(output_dir=report_dir)
    assert result.overall_status == "PASS"

    expected_files = [
        "project_audit.json",
        "project_audit_summary.md",
        "missing_files_report.json",
        "integration_audit.json",
        "data_integrity_report.json",
    ]
    for filename in expected_files:
        out_file = report_dir / filename
        assert out_file.exists(), f"Expected report {filename} was not created"
        assert out_file.stat().st_size > 0, f"Report {filename} is empty"

    # Verify JSON parsability
    with open(report_dir / "project_audit.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["overall_status"] == "PASS"
        assert len(data["checks"]) >= 170


def test_audit_project_cli():
    """Verify CLI execution via python scripts/audit_project.py --check-imports."""
    cmd = [sys.executable, str(_project_root / "scripts" / "audit_project.py"), "--check-imports"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_project_root))
    assert proc.returncode == 0
    assert "STAGE 2: PYTHON PACKAGE IMPORT" in proc.stdout
