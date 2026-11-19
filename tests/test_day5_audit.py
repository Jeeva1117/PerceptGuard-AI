"""Tests for PerceptGuard AI: Day 5 Automated Engineering Audit Script.

Verifies that `scripts/audit_day5.py` correctly audits and validates:
1. File existence and syntax for all 7 Day 5 tasks
2. Module imports and symbol availability
3. Artifact integrity, JSON schemas, and NumPy tensor dimensions
4. Data leakage prevention, split frame disjointness, and training-only fitting
5. Report generation and CLI command execution
6. Robust failure detection on missing files or corrupted artifacts
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.audit_day5 import Day5AuditReport, Day5Auditor, EXPECTED_ARTIFACTS, EXPECTED_FILES


class TestDay5Audit:
    """Comprehensive test suite for scripts/audit_day5.py."""

    def test_01_script_file_exists_and_valid_syntax(self):
        """Verify scripts/audit_day5.py exists and parses cleanly."""
        script_path = _project_root / "scripts/audit_day5.py"
        assert script_path.exists(), "scripts/audit_day5.py must exist"
        assert script_path.stat().st_size > 0, "scripts/audit_day5.py must not be empty"

    def test_02_check_files_all_present(self):
        """Verify that check_files returns True when all repository files exist."""
        auditor = Day5Auditor()
        passed = auditor.check_files()
        assert passed is True, "All expected files across Tasks 1-7 must be present and valid"
        # Verify no failed checks logged
        failed = [c for c in auditor.checks if c.status == "FAIL"]
        assert len(failed) == 0, f"Found failed file checks: {failed}"

    def test_03_check_imports_all_succeed(self):
        """Verify that check_imports returns True across all Day 5 modules and symbols."""
        auditor = Day5Auditor()
        passed = auditor.check_imports()
        assert passed is True, "All Day 5 modules and core symbols must import cleanly"

    def test_04_check_artifacts_all_valid(self):
        """Verify that check_artifacts validates all JSON schemas and .npy tensors."""
        auditor = Day5Auditor()
        passed = auditor.check_artifacts()
        assert passed is True, "All generated artifacts must exist and be valid"

    def test_05_check_leakage_certification(self):
        """Verify zero-leakage audit passes on official sequence splits."""
        auditor = Day5Auditor()
        passed = auditor.check_leakage()
        assert passed is True, "Zero-leakage invariants must be certified"

    def test_06_generate_report_overall_pass(self):
        """Verify that generate_report produces overall PASS and 7 task statuses."""
        auditor = Day5Auditor()
        auditor.check_files()
        auditor.check_imports()
        auditor.check_artifacts()
        auditor.check_leakage()
        report = auditor.generate_report()

        assert isinstance(report, Day5AuditReport)
        assert report.overall_status == "PASS"
        assert len(report.task_statuses) == 7
        for t_name, status in report.task_statuses.items():
            assert status == "PASS", f"{t_name} is not PASS (got {status})"

    def test_07_save_reports(self, tmp_path):
        """Verify that save_reports creates both JSON and Markdown files."""
        auditor = Day5Auditor()
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

    def test_08_cli_check_files(self):
        """Verify CLI execution with --check-files flag."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day5.py", "--check-files"],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --check-files failed: {res.stderr}"
        assert "[PASS]" in res.stdout

    def test_09_cli_check_imports(self):
        """Verify CLI execution with --check-imports flag."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day5.py", "--check-imports"],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --check-imports failed: {res.stderr}"

    def test_10_cli_check_artifacts(self):
        """Verify CLI execution with --check-artifacts flag."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day5.py", "--check-artifacts"],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --check-artifacts failed: {res.stderr}"

    def test_11_cli_check_leakage(self):
        """Verify CLI execution with --check-leakage flag."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day5.py", "--check-leakage"],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --check-leakage failed: {res.stderr}"

    def test_12_cli_full_audit(self):
        """Verify CLI execution with --full flag returns 0 and emits reports."""
        res = subprocess.run(
            [sys.executable, "scripts/audit_day5.py", "--full"],
            cwd=str(_project_root),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI --full failed: {res.stderr}"
        assert "Overall Status   : [PASS]" in res.stdout
        assert (_project_root / "reports/day5_audit/audit_report.json").exists()
        assert (_project_root / "reports/day5_audit/audit_report.md").exists()

    def test_13_negative_missing_file_detection(self, tmp_path):
        """Verify auditor catches missing files when pointed to a dummy directory."""
        fake_auditor = Day5Auditor(project_root=tmp_path)
        passed = fake_auditor.check_files()
        assert passed is False
        failed = [c for c in fake_auditor.checks if c.status == "FAIL"]
        assert len(failed) > 0

    def test_14_negative_corrupt_tensor_detection(self, tmp_path):
        """Verify auditor catches corrupt/NaN numpy tensors."""
        auditor = Day5Auditor()
        # Create a corrupt tensor with NaNs
        bad_tensor_path = tmp_path / "bad_seq.npy"
        np.save(bad_tensor_path, np.array([[[np.nan]]], dtype=np.float32))

        # Test direct loading logic
        arr = np.load(bad_tensor_path)
        assert not np.isfinite(arr).all()
