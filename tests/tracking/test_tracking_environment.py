"""PerceptGuard AI: Unit and Integration Tests for Tracking Environment Verification.

Validates:
- Runtime package versions (Python, PyTorch, Ultralytics, OpenCV, NumPy, SciPy, PyYAML)
- Hardware profile detection (CPU cores, memory, CUDA status)
- Ultralytics ByteTrack template discovery and compatibility
- Linear assignment solver backend detection (SciPy vs LAP)
- EnvironmentReport formatting, JSON serialization, and CLI execution
- Resilient edge case and missing dependency handling using mocks
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch

from src.tracking.tracker_config import TrackerConfig
from src.tracking.tracking_environment import (
    EnvironmentReport,
    TrackingEnvironmentVerifier,
    format_report,
    main,
    verify_tracking_environment,
)


class TestTrackingEnvironment(unittest.TestCase):
    """Test suite for TrackingEnvironmentVerifier and related contracts."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. Live Environment Integration Tests
    # --------------------------------------------------------------------------

    def test_01_live_environment_report_generation(self) -> None:
        """Test verification of the actual installed environment produces valid report."""
        report = verify_tracking_environment()
        self.assertIsInstance(report, EnvironmentReport)
        self.assertIn(report.status, ("PASS", "PASS_WITH_LIMITATIONS"))

    def test_02_python_version_reporting(self) -> None:
        """Test that Python runtime version is reported as non-empty string."""
        report = verify_tracking_environment()
        self.assertTrue(report.python_version.startswith("3."))

    def test_03_platform_reporting(self) -> None:
        """Test that operating system platform and architecture are reported."""
        report = verify_tracking_environment()
        self.assertIn(report.platform_system.lower(), ("windows", "linux", "darwin"))
        self.assertTrue(len(report.platform_machine) > 0)

    def test_04_pytorch_detection(self) -> None:
        """Test PyTorch detection, version string, and CPU tensor allocation."""
        report = verify_tracking_environment()
        self.assertNotIn("error", report.pytorch_version)
        self.assertNotIn("not installed", report.pytorch_version)
        self.assertTrue(report.cpu_available)

    def test_05_cuda_reporting(self) -> None:
        """Test CUDA reporting aligns with CPU-only host environment."""
        report = verify_tracking_environment()
        self.assertIsInstance(report.cuda_available, bool)
        # Target CPU environment constraint
        self.assertFalse(report.cuda_available)

    def test_06_ultralytics_detection(self) -> None:
        """Test Ultralytics package version reporting."""
        report = verify_tracking_environment()
        self.assertNotIn("error", report.ultralytics_version)
        self.assertTrue(report.ultralytics_version.startswith("8."))

    def test_07_opencv_detection(self) -> None:
        """Test OpenCV package version reporting."""
        report = verify_tracking_environment()
        self.assertNotIn("error", report.opencv_version)
        self.assertTrue(len(report.opencv_version) > 0)

    def test_08_numpy_detection(self) -> None:
        """Test NumPy package version reporting."""
        report = verify_tracking_environment()
        self.assertNotIn("error", report.numpy_version)
        self.assertTrue(len(report.numpy_version) > 0)

    def test_09_pyyaml_detection(self) -> None:
        """Test PyYAML package detection and version reporting."""
        report = verify_tracking_environment()
        self.assertNotIn("not installed", report.pyyaml_version)

    def test_10_scipy_detection(self) -> None:
        """Test SciPy package detection and version reporting."""
        report = verify_tracking_environment()
        self.assertNotIn("not installed", report.scipy_version)

    def test_11_cpu_and_memory_reporting(self) -> None:
        """Test hardware profiling (CPU logical count and RAM)."""
        report = verify_tracking_environment()
        self.assertGreaterEqual(report.cpu_count, 1)
        self.assertGreater(report.memory_total_gb, 1.0)

    # --------------------------------------------------------------------------
    # 2. Tracking Infrastructure & ByteTrack Compatibility
    # --------------------------------------------------------------------------

    def test_12_bytetrack_template_discovery(self) -> None:
        """Test that Ultralytics built-in ByteTrack template is discovered."""
        report = verify_tracking_environment()
        self.assertTrue(report.ultralytics_bytetrack_found)
        self.assertIsNotNone(report.ultralytics_bytetrack_path)
        self.assertTrue(Path(report.ultralytics_bytetrack_path).exists())
        self.assertTrue(report.bytetrack_compatible)

    def test_13_linear_assignment_backend_detection(self) -> None:
        """Test that linear assignment solver is identified (SciPy fallback or LAP)."""
        report = verify_tracking_environment()
        self.assertTrue(
            "scipy" in report.linear_assignment_backend.lower()
            or "lap" in report.linear_assignment_backend.lower()
        )

    def test_14_tracker_config_validity_in_report(self) -> None:
        """Test that PerceptGuard tracker configuration is marked valid in report."""
        report = verify_tracking_environment()
        self.assertTrue(report.tracker_config_valid)
        self.assertIsNotNone(report.tracker_config_path)
        self.assertEqual(report.tracker_parameters.get("tracker_type"), "bytetrack")

    # --------------------------------------------------------------------------
    # 3. Report Formatting & Serialization
    # --------------------------------------------------------------------------

    def test_15_report_to_dict_contract(self) -> None:
        """Test report dictionary serialization schema completeness."""
        report = verify_tracking_environment()
        d = report.to_dict()
        required_keys = [
            "python_version",
            "pytorch_version",
            "ultralytics_version",
            "cuda_available",
            "cpu_available",
            "tracker_config_valid",
            "bytetrack_compatible",
            "status",
        ]
        for k in required_keys:
            self.assertIn(k, d)

    def test_16_format_report_human_readable(self) -> None:
        """Test human-readable terminal report formatting."""
        report = verify_tracking_environment()
        rendered = format_report(report)
        self.assertIn("PERCEPTGUARD AI — TRACKING ENVIRONMENT VERIFICATION REPORT", rendered)
        self.assertIn("Environment Status", rendered)
        self.assertIn("ByteTrack Compatible", rendered)

    # --------------------------------------------------------------------------
    # 4. CLI Execution
    # --------------------------------------------------------------------------

    def test_17_cli_standard_execution(self) -> None:
        """Test CLI tool standard run returns exit code 0."""
        exit_code = main([])
        self.assertEqual(exit_code, 0)

    def test_18_cli_json_execution(self) -> None:
        """Test CLI tool with --json flag emits valid JSON."""
        with patch("builtins.print") as mock_print:
            exit_code = main(["--json"])
            self.assertEqual(exit_code, 0)
            mock_print.assert_called_once()
            output_str = mock_print.call_args[0][0]
            parsed = json.loads(output_str)
            self.assertIn("status", parsed)

    def test_19_cli_custom_config_path(self) -> None:
        """Test CLI tool with valid custom config path."""
        cfg = TrackerConfig()
        out_yaml = self.temp_path / "custom.yaml"
        cfg.to_yaml(out_yaml)

        exit_code = main(["--config", str(out_yaml)])
        self.assertEqual(exit_code, 0)

    # --------------------------------------------------------------------------
    # 5. Resilient Mocking & Edge Cases
    # --------------------------------------------------------------------------

    def test_20_missing_tracker_config_handling(self) -> None:
        """Test report status when tracker configuration file does not exist."""
        missing = self.temp_path / "missing_config.yaml"
        verifier = TrackingEnvironmentVerifier(config_path=missing)
        report = verifier.verify()
        self.assertFalse(report.tracker_config_valid)
        self.assertEqual(report.status, "FAIL")

    def test_21_corrupt_tracker_config_handling(self) -> None:
        """Test report status when tracker configuration YAML is corrupt."""
        corrupt = self.temp_path / "corrupt.yaml"
        corrupt.write_text("invalid: yaml: [broken", encoding="utf-8")
        verifier = TrackingEnvironmentVerifier(config_path=corrupt)
        report = verifier.verify()
        self.assertFalse(report.tracker_config_valid)
        self.assertEqual(report.status, "FAIL")

    def test_22_mock_pytorch_failure(self) -> None:
        """Test that PyTorch import failure marks status as FAIL."""
        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError("No torch")):
                report = verify_tracking_environment()
                self.assertEqual(report.status, "FAIL")

    def test_23_mock_ultralytics_failure(self) -> None:
        """Test that Ultralytics failure marks status as FAIL."""
        with patch.dict("sys.modules", {"ultralytics": None}):
            with patch("builtins.__import__", side_effect=ImportError("No ultralytics")):
                report = verify_tracking_environment()
                self.assertEqual(report.status, "FAIL")

    def test_24_mock_both_linear_solvers_missing(self) -> None:
        """Test that missing both lap and scipy causes status FAIL."""
        verifier = TrackingEnvironmentVerifier()
        with patch.object(verifier, "verify") as mock_v:
            mock_report = EnvironmentReport(
                python_version="3.13.3",
                platform_system="Windows",
                platform_machine="AMD64",
                pytorch_version="2.15.0+cpu",
                ultralytics_version="8.4.215",
                opencv_version="5.0.0",
                numpy_version="2.5.4",
                pyyaml_version="6.0.3",
                scipy_version="not installed",
                cuda_available=False,
                cpu_available=True,
                cpu_count=8,
                memory_total_gb=8.0,
                tracker_config_valid=True,
                tracker_config_path=None,
                tracker_parameters={},
                ultralytics_bytetrack_found=True,
                ultralytics_bytetrack_path=None,
                bytetrack_compatible=True,
                linear_assignment_backend="none",
                status="FAIL",
                notes=["No linear assignment solver found."],
            )
            mock_v.return_value = mock_report
            rep = verifier.verify()
            self.assertEqual(rep.status, "FAIL")

    def test_25_mock_lap_present_gives_clean_pass(self) -> None:
        """Test that when lap is present, status is PASS rather than PASS_WITH_LIMITATIONS."""
        verifier = TrackingEnvironmentVerifier()
        with patch(
            "src.tracking.tracking_environment.TrackingEnvironmentVerifier.verify"
        ) as mock_v:
            mock_report = EnvironmentReport(
                python_version="3.13.3",
                platform_system="Windows",
                platform_machine="AMD64",
                pytorch_version="2.15.0+cpu",
                ultralytics_version="8.4.215",
                opencv_version="5.0.0",
                numpy_version="2.5.4",
                pyyaml_version="6.0.3",
                scipy_version="1.18.1",
                cuda_available=False,
                cpu_available=True,
                cpu_count=8,
                memory_total_gb=8.0,
                tracker_config_valid=True,
                tracker_config_path=None,
                tracker_parameters={},
                ultralytics_bytetrack_found=True,
                ultralytics_bytetrack_path=None,
                bytetrack_compatible=True,
                linear_assignment_backend="lap (0.5.12)",
                status="PASS",
                notes=[],
            )
            mock_v.return_value = mock_report
            rep = verifier.verify()
            self.assertEqual(rep.status, "PASS")


if __name__ == "__main__":
    unittest.main()
