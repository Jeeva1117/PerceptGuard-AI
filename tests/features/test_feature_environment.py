"""Unit and integration tests for Feature Environment Verification (Day 5 Task 1).

Tests runtime diagnostic checks, hardware CPU/RAM inspection, library integrity,
configuration auditing, upstream artifact detection, formatting, and CLI interfaces.
"""

from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.features.feature_config import FeatureConfig
from src.features.feature_environment import (
    FeatureEnvironmentReport,
    FeatureEnvironmentVerifier,
    main,
)


class TestFeatureEnvironmentReport:
    """Tests for FeatureEnvironmentReport dataclass."""

    def test_report_instantiation_and_dict_conversion(self) -> None:
        report = FeatureEnvironmentReport(
            status="PASS",
            python_version="3.13.3",
            operating_system="Windows 11",
            cpu_cores_logical=8,
            cpu_cores_physical=4,
            ram_total_gb=7.8,
            ram_available_gb=3.2,
            pytorch_version="2.15.0+cpu",
            pytorch_cpu_only=True,
            cuda_available=False,
            numpy_version="2.5.4",
            opencv_version="5.0.0",
            scipy_version="1.18.1",
            pyyaml_version="6.0.3",
            pandas_available=True,
            config_valid=True,
            config_path="config/feature_config.yaml",
            detection_artifact_found=True,
            tracking_artifact_found=True,
            metadata_artifact_found=True,
            preprocessed_frames_found=True,
            artifact_validation_valid=True,
            artifact_validation_errors=[],
            notes=[],
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "PASS"
        assert d["python_version"] == "3.13.3"
        assert d["pytorch_cpu_only"] is True
        assert d["config_valid"] is True
        assert d["artifact_validation_valid"] is True
        assert len(d["artifact_validation_errors"]) == 0


class TestFeatureEnvironmentVerifier:
    """Tests for FeatureEnvironmentVerifier."""

    def test_default_initialization(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        assert verifier.config_path is not None
        assert "feature_config.yaml" in str(verifier.config_path)

    def test_custom_config_initialization(self, tmp_path: Path) -> None:
        custom_cfg = tmp_path / "custom_features.yaml"
        custom_cfg.write_text("schema_version: '1.0'\n", encoding="utf-8")
        verifier = FeatureEnvironmentVerifier(config_path=custom_cfg)
        assert verifier.config_path == custom_cfg

    def test_live_audit_execution(self) -> None:
        """Run audit against current active repository environment."""
        verifier = FeatureEnvironmentVerifier()
        report = verifier.run_audit()

        assert report.status == "PASS"
        assert report.config_valid is True
        assert report.cpu_cores_logical >= 1
        assert report.ram_total_gb > 0.0
        assert report.pytorch_version != "NOT_INSTALLED"
        assert report.numpy_version != "NOT_INSTALLED"
        assert report.opencv_version != "NOT_INSTALLED"
        assert report.pyyaml_version != "NOT_INSTALLED"
        assert report.pytorch_cpu_only is True
        assert report.cuda_available is False
        assert report.detection_artifact_found is True
        assert report.tracking_artifact_found is True
        assert report.metadata_artifact_found is True
        assert report.preprocessed_frames_found is True
        assert report.artifact_validation_valid is True

    def test_human_readable_formatting(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        report = verifier.run_audit()
        formatted = verifier.format_report_human_readable(report)

        assert "PERCEPTGUARD AI — FEATURE ENGINEERING ENVIRONMENT REPORT" in formatted
        assert "Environment Status         : [PASS]" in formatted
        assert "Runtime & Host Platform :" in formatted
        assert "Core Scientific & Vision Libraries :" in formatted
        assert "Feature Engineering Configuration :" in formatted
        assert "Upstream Artifact Verification :" in formatted
        assert "PyTorch" in formatted
        assert "NumPy" in formatted

    def test_formatting_with_diagnostic_notes(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        report = FeatureEnvironmentReport(
            status="FAIL",
            python_version="3.8.0",
            operating_system="TestOS",
            cpu_cores_logical=2,
            cpu_cores_physical=2,
            ram_total_gb=4.0,
            ram_available_gb=1.0,
            pytorch_version="NOT_INSTALLED",
            pytorch_cpu_only=False,
            cuda_available=False,
            numpy_version="2.0.0",
            opencv_version="4.9.0",
            scipy_version="NOT_INSTALLED",
            pyyaml_version="6.0",
            pandas_available=False,
            config_valid=False,
            config_path="bad_path.yaml",
            detection_artifact_found=False,
            tracking_artifact_found=False,
            metadata_artifact_found=False,
            preprocessed_frames_found=False,
            artifact_validation_valid=False,
            artifact_validation_errors=["Sample error"],
            notes=["Python older than 3.9", "PyTorch missing", "Config missing"],
        )
        formatted = verifier.format_report_human_readable(report)
        assert "[FAIL]" in formatted
        assert "Notes & Diagnostics :" in formatted
        assert "* Python older than 3.9" in formatted
        assert "* PyTorch missing" in formatted

    def test_audit_fails_on_old_python_version(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        with patch("sys.version_info", (3, 8, 5)):
            report = verifier.run_audit()
            assert report.status == "FAIL"
            assert any("older than recommended 3.9+" in note for note in report.notes)

    def test_audit_fails_on_missing_config(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist.yaml"
        verifier = FeatureEnvironmentVerifier(config_path=nonexistent)
        report = verifier.run_audit()

        assert report.status == "FAIL"
        assert report.config_valid is False
        assert any("Feature configuration invalid or missing" in note for note in report.notes)

    def test_audit_handles_missing_torch(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        orig_import = __import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("Mocked missing torch")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            report = verifier.run_audit()
            assert report.status == "FAIL"
            assert report.pytorch_version == "NOT_INSTALLED"
            assert any("PyTorch not available" in note for note in report.notes)

    def test_audit_handles_missing_opencv(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        orig_import = __import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "cv2":
                raise ImportError("Mocked missing cv2")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            report = verifier.run_audit()
            assert report.status == "FAIL"
            assert report.opencv_version == "NOT_INSTALLED"
            assert any("OpenCV not available" in note for note in report.notes)

    def test_audit_handles_psutil_import_error(self) -> None:
        verifier = FeatureEnvironmentVerifier()
        orig_import = __import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "psutil":
                raise ImportError("Mocked missing psutil")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            report = verifier.run_audit()
            # psutil missing should produce a note, but not fail the audit
            assert any("psutil not installed" in note for note in report.notes)


class TestFeatureEnvironmentCLI:
    """Tests for Feature Environment CLI main() entry point."""

    def test_cli_human_readable_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main([])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "PERCEPTGUARD AI — FEATURE ENGINEERING ENVIRONMENT REPORT" in captured.out
        assert "[PASS]" in captured.out

    def test_cli_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--json"])
        captured = capsys.readouterr()

        assert exit_code == 0
        data = json.loads(captured.out)
        assert data["status"] == "PASS"
        assert data["config_valid"] is True
        assert data["pytorch_cpu_only"] is True

    def test_cli_custom_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--config", "config/feature_config.yaml", "--json"])
        captured = capsys.readouterr()

        assert exit_code == 0
        data = json.loads(captured.out)
        assert data["status"] == "PASS"

    def test_cli_missing_config_returns_exit_code_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--config", "nonexistent_file_path.yaml"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "[FAIL]" in captured.out
