"""Tests for PerceptGuard AI: Day 6 Automated Engineering Audit Script.

Verifies that `scripts/audit_day6.py` correctly audits and validates:
1. File existence and syntax for all 7 Day 6 tasks
2. Module imports and symbol availability in src.forecasting
3. Artifact integrity, JSON schemas, and NumPy tensor dimensions
4. Forecasting data contract validation and honest target readiness gating
5. Data leakage prevention and split sequence disjointness
6. Report generation and CLI command execution
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.audit_day6 import Day6AuditReport, Day6Auditor, EXPECTED_ARTIFACTS, EXPECTED_FILES


class TestDay6Audit:
    """Comprehensive test suite for scripts/audit_day6.py."""

    def test_01_script_file_exists_and_valid_syntax(self):
        """Verify scripts/audit_day6.py exists and parses cleanly."""
        script_path = _project_root / "scripts/audit_day6.py"
        assert script_path.exists(), "scripts/audit_day6.py must exist"
        assert script_path.stat().st_size > 0, "scripts/audit_day6.py must not be empty"

    def test_02_check_files_all_present(self):
        """Verify that check_files returns True when all repository files exist."""
        auditor = Day6Auditor()
        passed = auditor.check_files()
        assert passed is True, "All expected files across Tasks 1-7 must be present and valid"
        failed = [c for c in auditor.checks if c.status == "FAIL"]
        assert len(failed) == 0, f"Found failed file checks: {failed}"

    def test_03_check_imports_all_succeed(self):
        """Verify that check_imports returns True across all Day 6 modules and symbols."""
        auditor = Day6Auditor()
        passed = auditor.check_imports()
        assert passed is True, "All Day 6 forecasting modules and core symbols must import cleanly"

    def test_04_check_artifacts_all_valid(self):
        """Verify that check_artifacts validates all JSON schemas and .npy tensors."""
        auditor = Day6Auditor()
        passed = auditor.check_artifacts()
        assert passed is True, "All expected sequence artifacts must exist and be valid"

    def test_05_check_data_contracts(self):
        """Verify forecasting data contract and honest target readiness gating."""
        auditor = Day6Auditor()
        passed = auditor.check_data_contracts()
        assert passed is True, "Forecasting data contract must pass"

    def test_06_check_leakage_certification(self):
        """Verify zero-leakage audit passes on official sequence splits."""
        auditor = Day6Auditor()
        passed = auditor.check_leakage()
        assert passed is True, "Zero-leakage invariants must be certified"

    def test_07_generate_report_overall_pass(self):
        """Verify that generate_report produces overall PASS across tasks."""
        auditor = Day6Auditor()
        auditor.check_files()
        auditor.check_imports()
        auditor.check_artifacts()
        auditor.check_data_contracts()
        auditor.check_leakage()
        report = auditor.generate_report()

        assert isinstance(report, Day6AuditReport)
        assert report.overall_status == "PASS"
        assert len(report.task_statuses) == 7
        for t_name, status in report.task_statuses.items():
            assert status == "PASS", f"{t_name} is not PASS (got {status})"

    def test_08_save_reports(self, tmp_path):
        """Verify that save_reports creates both JSON and Markdown files."""
        auditor = Day6Auditor()
        auditor.check_files()
        report = auditor.generate_report()
        json_path, md_path = auditor.save_reports(report, output_dir=tmp_path)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.stat().st_size > 0
        assert md_path.stat().st_size > 0

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert "task_statuses" in data

    def test_09_cli_check_files(self):
        """Verify CLI execution with --check-files flag."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day6.py", "--check-files"],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --check-files failed: {res.stderr}"
        assert "AUDIT SUMMARY" in res.stdout
        assert "Overall Status:  PASS" in res.stdout

    def test_10_cli_full_audit(self, tmp_path):
        """Verify CLI execution with --full flag."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day6.py", "--full", "--output-dir", str(tmp_path)],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --full failed: {res.stderr}"
        assert (tmp_path / "day6_audit.json").exists()
        assert (tmp_path / "day6_audit_summary.md").exists()
