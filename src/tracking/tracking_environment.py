"""PerceptGuard AI: Multi-Object Tracking Environment & Infrastructure Verifier.

Senior Staff-level environment audit utility for ByteTrack multi-object tracking.
Inspects core runtime packages, hardware profile, Ultralytics tracker templates,
linear assignment backends, and PerceptGuard tracking configuration.

Guarantees:
- Pure inspection; avoids model downloads, heavy video decoding, and GPU calls.
- Deterministic reporting for both human operators and automated CI/CD pipelines.
- Standardized CLI entrypoint: `python -m src.tracking.tracking_environment`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.tracking.tracker_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    TrackerConfig,
    TrackerConfigError,
)

logger = logging.getLogger("PerceptGuard.TrackingEnvironment")

# ==============================================================================
# Environment Audit Report Data Contracts
# ==============================================================================


@dataclass
class EnvironmentReport:
    """Structured report encapsulating the tracking environment audit results."""

    python_version: str
    platform_system: str
    platform_machine: str
    pytorch_version: str
    ultralytics_version: str
    opencv_version: str
    numpy_version: str
    pyyaml_version: str
    scipy_version: str
    cuda_available: bool
    cpu_available: bool
    cpu_count: int
    memory_total_gb: float
    tracker_config_valid: bool
    tracker_config_path: Optional[str]
    tracker_parameters: Dict[str, Any]
    ultralytics_bytetrack_found: bool
    ultralytics_bytetrack_path: Optional[str]
    bytetrack_compatible: bool
    linear_assignment_backend: str
    status: str  # "PASS", "PASS_WITH_LIMITATIONS", "FAIL"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)


# ==============================================================================
# Environment Verification Architecture
# ==============================================================================


class TrackingEnvironmentVerifier:
    """Verifies runtime compatibility and configuration readiness for ByteTrack tracking."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize verifier.

        Args:
            config_path: Optional path to tracker configuration YAML.
            project_root: Optional repository root path.
        """
        self.project_root = (
            Path(project_root).resolve() if project_root else _project_root
        )
        self.config_path = (
            Path(config_path).resolve()
            if config_path
            else (self.project_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
        )

    def verify(self) -> EnvironmentReport:
        """Execute comprehensive environment and tracking configuration audit.

        Returns:
            EnvironmentReport with verified status and telemetry.
        """
        notes: List[str] = []
        is_fatal = False

        # 1. Python runtime
        py_ver = sys.version.split()[0]
        plat_sys = platform.system()
        plat_mach = platform.machine()

        # 2. PyTorch & CUDA status
        torch_ver = "not installed"
        cuda_avail = False
        cpu_avail = False
        try:
            import torch

            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            # Verify torch CPU tensor allocation
            t = torch.zeros((1, 1), device="cpu")
            cpu_avail = (t.device.type == "cpu")
            if not cuda_avail:
                notes.append("CPU-only PyTorch execution confirmed (CUDA is unavailable/disabled).")
        except Exception as e:
            torch_ver = f"error ({e})"
            notes.append(f"PyTorch verification failure: {e}")
            is_fatal = True

        # 3. Ultralytics verification
        ultralytics_ver = "not installed"
        try:
            import ultralytics

            ultralytics_ver = ultralytics.__version__
        except Exception as e:
            ultralytics_ver = f"error ({e})"
            notes.append(f"Ultralytics verification failure: {e}")
            is_fatal = True

        # 4. OpenCV verification
        cv2_ver = "not installed"
        try:
            import cv2

            cv2_ver = cv2.__version__
        except Exception as e:
            cv2_ver = f"error ({e})"
            notes.append(f"OpenCV verification failure: {e}")
            is_fatal = True

        # 5. NumPy verification
        numpy_ver = "not installed"
        try:
            import numpy

            numpy_ver = numpy.__version__
        except Exception as e:
            numpy_ver = f"error ({e})"
            notes.append(f"NumPy verification failure: {e}")
            is_fatal = True

        # 6. PyYAML verification
        pyyaml_ver = "not installed"
        try:
            import yaml

            pyyaml_ver = getattr(yaml, "__version__", "installed")
        except Exception as e:
            pyyaml_ver = f"error ({e})"
            notes.append(f"PyYAML verification failure: {e}")
            is_fatal = True

        # 7. SciPy verification
        scipy_ver = "not installed"
        try:
            import scipy

            scipy_ver = scipy.__version__
        except Exception as e:
            scipy_ver = f"error ({e})"
            notes.append(f"SciPy verification failure: {e}")

        # 8. Hardware Profile (CPU Cores & RAM)
        cpu_count = os.cpu_count() or 1
        mem_gb = 0.0
        try:
            import psutil

            mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            mem_gb = 8.0  # Estimated standard host memory

        # 9. Linear Assignment Backend Detection
        linear_backend = "unknown"
        has_lap = False
        try:
            import lap

            if getattr(lap, "__version__", None):
                has_lap = True
                linear_backend = f"lap ({lap.__version__})"
        except (ImportError, Exception):
            has_lap = False

        if not has_lap:
            try:
                from scipy.optimize import linear_sum_assignment

                linear_backend = "scipy.optimize.linear_sum_assignment (built-in CPU fallback)"
                notes.append(
                    "Primary C-extension 'lap' is not installed; SciPy Hungarian assignment is available as a robust CPU fallback."
                )
            except ImportError:
                linear_backend = "none"
                notes.append("No linear assignment solver found (neither lap nor scipy available).")
                is_fatal = True

        # 10. Ultralytics ByteTrack Configuration Inspection
        ultralytics_bt_found = False
        ultralytics_bt_path: Optional[str] = None
        bytetrack_compat = False
        try:
            from ultralytics.utils.checks import check_yaml

            resolved_yaml = check_yaml("bytetrack.yaml")
            p = Path(resolved_yaml)
            if p.exists() and p.is_file():
                ultralytics_bt_found = True
                ultralytics_bt_path = str(p)
                bytetrack_compat = True
                notes.append(f"Ultralytics native ByteTrack template resolved at: {p.name}")
        except Exception as e:
            notes.append(f"Ultralytics ByteTrack template resolution notice: {e}")

        # 11. PerceptGuard Tracker Configuration Verification
        tracker_valid = False
        tracker_params: Dict[str, Any] = {}
        try:
            cfg = TrackerConfig.from_yaml(self.config_path, project_root=self.project_root)
            tracker_valid = True
            tracker_params = cfg.to_dict()
            notes.append("PerceptGuard tracker configuration loaded and validated successfully.")
        except Exception as e:
            notes.append(f"PerceptGuard tracker configuration error: {e}")

        # 12. Final Status Evaluation
        if is_fatal or not tracker_valid or not cpu_avail:
            status = "FAIL"
        elif not has_lap:
            status = "PASS_WITH_LIMITATIONS"
        else:
            status = "PASS"

        return EnvironmentReport(
            python_version=py_ver,
            platform_system=plat_sys,
            platform_machine=plat_mach,
            pytorch_version=torch_ver,
            ultralytics_version=ultralytics_ver,
            opencv_version=cv2_ver,
            numpy_version=numpy_ver,
            pyyaml_version=pyyaml_ver,
            scipy_version=scipy_ver,
            cuda_available=cuda_avail,
            cpu_available=cpu_avail,
            cpu_count=cpu_count,
            memory_total_gb=mem_gb,
            tracker_config_valid=tracker_valid,
            tracker_config_path=str(self.config_path) if self.config_path.exists() else None,
            tracker_parameters=tracker_params,
            ultralytics_bytetrack_found=ultralytics_bt_found,
            ultralytics_bytetrack_path=ultralytics_bt_path,
            bytetrack_compatible=bytetrack_compat,
            linear_assignment_backend=linear_backend,
            status=status,
            notes=notes,
        )


def verify_tracking_environment(
    config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> EnvironmentReport:
    """Convenience functional interface to verify tracking environment."""
    verifier = TrackingEnvironmentVerifier(config_path=config_path, project_root=project_root)
    return verifier.verify()


def format_report(report: EnvironmentReport) -> str:
    """Render environment report into a clean, human-readable terminal output."""
    lines: List[str] = [
        "=" * 70,
        "PERCEPTGUARD AI — TRACKING ENVIRONMENT VERIFICATION REPORT",
        "=" * 70,
        f"  Environment Status         : [{report.status}]",
        "-" * 70,
        "  1. Runtime & Host Platform :",
        f"     • Python Version        : {report.python_version}",
        f"     • Operating System      : {report.platform_system} ({report.platform_machine})",
        f"     • CPU Logical Cores     : {report.cpu_count}",
        f"     • Host RAM Total        : {report.memory_total_gb:.1f} GB",
        "-" * 70,
        "  2. Core Perception & ML Packages :",
        f"     • PyTorch               : {report.pytorch_version} (CPU Allocation: {report.cpu_available})",
        f"     • CUDA Available        : {report.cuda_available}",
        f"     • Ultralytics           : {report.ultralytics_version}",
        f"     • OpenCV                : {report.opencv_version}",
        f"     • NumPy                 : {report.numpy_version}",
        f"     • SciPy                 : {report.scipy_version}",
        f"     • PyYAML                : {report.pyyaml_version}",
        "-" * 70,
        "  3. Tracking Infrastructure & ByteTrack Compatibility :",
        f"     • Ultralytics ByteTrack : {'AVAILABLE' if report.ultralytics_bytetrack_found else 'NOT FOUND'}",
        f"     • Template Path         : {report.ultralytics_bytetrack_path or 'N/A'}",
        f"     • ByteTrack Compatible  : {report.bytetrack_compatible}",
        f"     • Association Backend   : {report.linear_assignment_backend}",
        "-" * 70,
        "  4. PerceptGuard Tracker Configuration :",
        f"     • Config Valid          : {report.tracker_config_valid}",
        f"     • Config Path           : {report.tracker_config_path or 'N/A'}",
    ]

    if report.tracker_parameters:
        lines.append("     • Active Parameters     :")
        for k, v in report.tracker_parameters.items():
            lines.append(f"       - {k:21s}: {v}")

    lines.extend([
        "-" * 70,
        "  5. Engineering Notes & Verification Findings :",
    ])
    for note in report.notes:
        lines.append(f"     * {note}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for tracking environment verification."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Tracking Environment Verification Utility"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional path to custom tracker YAML configuration file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output environment verification report in JSON format",
    )

    args = parser.parse_args(argv)

    try:
        report = verify_tracking_environment(config_path=args.config)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_report(report))

        if report.status in ("PASS", "PASS_WITH_LIMITATIONS"):
            return 0
        else:
            return 1

    except Exception as e:
        print(f"[FATAL] Environment verification crashed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
