"""PerceptGuard AI: Feature Engineering Environment Verifier.

Performs static and live runtime environmental audits for the Day 5 Feature
Engineering pipeline, verifying:
- Python version compatibility (>= 3.9).
- Hardware architecture: CPU core allocation and RAM budget verification.
- Core scientific and perception libraries: PyTorch (CPU-only), NumPy, SciPy, OpenCV, PyYAML, Pandas.
- Feature configuration validity and schema adherence.
- Upstream artifact availability: detection results, tracking outputs, metadata, preprocessed frames.
- CLI introspection for automated CI/CD and deployment checks.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.feature_config import FeatureConfig, FeatureConfigError, DEFAULT_CONFIG_RELATIVE_PATH
from src.features.feature_schema import validate_input_artifacts

logger = logging.getLogger("PerceptGuard.FeatureEnvironment")


@dataclass
class FeatureEnvironmentReport:
    """Strongly typed report capturing feature engineering environment diagnostics."""
    status: str
    python_version: str
    operating_system: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    ram_total_gb: float
    ram_available_gb: float
    pytorch_version: str
    pytorch_cpu_only: bool
    cuda_available: bool
    numpy_version: str
    opencv_version: str
    scipy_version: str
    pyyaml_version: str
    pandas_available: bool
    config_valid: bool
    config_path: str
    detection_artifact_found: bool
    tracking_artifact_found: bool
    metadata_artifact_found: bool
    preprocessed_frames_found: bool
    artifact_validation_valid: bool
    artifact_validation_errors: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureEnvironmentVerifier:
    """Production verifier for Feature Engineering runtime infrastructure."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config_path = config_path or (_project_root / DEFAULT_CONFIG_RELATIVE_PATH)

    def run_audit(self) -> FeatureEnvironmentReport:
        """Execute complete diagnostic evaluation of environment, libraries, and artifacts."""
        notes: List[str] = []
        is_pass = True

        # 1. Python Version
        py_ver = platform.python_version()
        if sys.version_info < (3, 9):
            notes.append(f"Python version {py_ver} is older than recommended 3.9+")
            is_pass = False

        # 2. Operating System & Platform
        os_desc = f"{platform.system()} {platform.release()} ({platform.machine()})"

        # 3. CPU and RAM Audit
        cpu_logical = os.cpu_count() or 1
        cpu_physical = cpu_logical
        ram_total_gb = 0.0
        ram_avail_gb = 0.0

        try:
            import psutil
            cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
            vmem = psutil.virtual_memory()
            ram_total_gb = round(vmem.total / (1024 ** 3), 2)
            ram_avail_gb = round(vmem.available / (1024 ** 3), 2)
        except ImportError:
            notes.append("psutil not installed; hardware memory metrics approximate.")

        # 4. Core Libraries
        # PyTorch
        try:
            import torch
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            pytorch_cpu = not cuda_avail or ("+cpu" in torch_ver)
            if not pytorch_cpu:
                notes.append("PyTorch with CUDA detected; CPU-only execution is standard for this profile.")
        except ImportError as ex:
            torch_ver = "NOT_INSTALLED"
            cuda_avail = False
            pytorch_cpu = False
            notes.append(f"PyTorch not available: {ex}")
            is_pass = False

        # NumPy
        try:
            import numpy as np
            numpy_ver = np.__version__
        except ImportError as ex:
            numpy_ver = "NOT_INSTALLED"
            notes.append(f"NumPy not available: {ex}")
            is_pass = False

        # OpenCV
        try:
            import cv2
            opencv_ver = cv2.__version__
        except ImportError as ex:
            opencv_ver = "NOT_INSTALLED"
            notes.append(f"OpenCV not available: {ex}")
            is_pass = False

        # SciPy
        try:
            import scipy
            scipy_ver = scipy.__version__
        except ImportError:
            scipy_ver = "NOT_INSTALLED"
            notes.append("SciPy not installed (recommended for advanced motion metrics).")

        # PyYAML
        try:
            import yaml
            yaml_ver = yaml.__version__
        except ImportError as ex:
            yaml_ver = "NOT_INSTALLED"
            notes.append(f"PyYAML not available: {ex}")
            is_pass = False

        # Pandas
        try:
            import pandas
            pandas_avail = True
        except ImportError:
            pandas_avail = False

        # 5. Configuration Validation
        cfg: Optional[FeatureConfig] = None
        cfg_valid = False
        try:
            cfg = FeatureConfig.from_yaml(self.config_path)
            cfg_valid = True
        except Exception as ex:
            notes.append(f"Feature configuration invalid or missing at '{self.config_path}': {ex}")
            is_pass = False

        # 6. Artifact Resolution
        det_found = False
        trk_found = False
        meta_found = False
        frames_found = False

        artifact_val_valid = True
        artifact_val_errors: List[str] = []

        if cfg is not None:
            # Check detection file
            if cfg.detection_results_path:
                det_p = _project_root / Path(cfg.detection_results_path)
                det_found = det_p.exists() and det_p.stat().st_size > 0
            else:
                det_p = None

            # Check tracking file
            if cfg.tracking_results_path:
                trk_p = _project_root / Path(cfg.tracking_results_path)
                trk_found = trk_p.exists() and trk_p.stat().st_size > 0
            else:
                trk_p = None

            # Check metadata file
            if cfg.dataset_metadata_path:
                meta_p = _project_root / Path(cfg.dataset_metadata_path)
                meta_found = meta_p.exists() and meta_p.stat().st_size > 0
            else:
                meta_p = None

            # Check preprocessed frames
            if cfg.preprocessed_frames_dir:
                frm_p = _project_root / Path(cfg.preprocessed_frames_dir)
                frames_found = frm_p.exists() and any(frm_p.glob("*.jpg"))
            else:
                frm_p = None

            # Validate integration if artifacts exist
            if det_found or trk_found:
                val_res = validate_input_artifacts(
                    detection_file=det_p if det_found else None,
                    tracking_file=trk_p if trk_found else None,
                    metadata_file=meta_p if meta_found else None,
                    strict=False,
                )
                artifact_val_valid = val_res.is_valid
                artifact_val_errors = val_res.errors
                if not val_res.is_valid:
                    notes.append("Upstream artifacts contain schema or ordering warnings.")

        final_status = "PASS" if is_pass else "FAIL"

        return FeatureEnvironmentReport(
            status=final_status,
            python_version=py_ver,
            operating_system=os_desc,
            cpu_cores_logical=cpu_logical,
            cpu_cores_physical=cpu_physical,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_avail_gb,
            pytorch_version=torch_ver,
            pytorch_cpu_only=pytorch_cpu,
            cuda_available=cuda_avail,
            numpy_version=numpy_ver,
            opencv_version=opencv_ver,
            scipy_version=scipy_ver,
            pyyaml_version=yaml_ver,
            pandas_available=pandas_avail,
            config_valid=cfg_valid,
            config_path=str(self.config_path),
            detection_artifact_found=det_found,
            tracking_artifact_found=trk_found,
            metadata_artifact_found=meta_found,
            preprocessed_frames_found=frames_found,
            artifact_validation_valid=artifact_val_valid,
            artifact_validation_errors=artifact_val_errors,
            notes=notes,
        )

    def format_report_human_readable(self, report: FeatureEnvironmentReport) -> str:
        """Format the environment report for human terminal review."""
        lines = [
            "=" * 70,
            "PERCEPTGUARD AI — FEATURE ENGINEERING ENVIRONMENT REPORT",
            "=" * 70,
            f"  Environment Status         : [{report.status}]",
            "-" * 70,
            "  1. Runtime & Host Platform :",
            f"     • Python Version        : {report.python_version}",
            f"     • Operating System      : {report.operating_system}",
            f"     • CPU Cores (Log/Phys)  : {report.cpu_cores_logical} / {report.cpu_cores_physical}",
            f"     • Host RAM (Avail/Total): {report.ram_available_gb} GB / {report.ram_total_gb} GB",
            "-" * 70,
            "  2. Core Scientific & Vision Libraries :",
            f"     • PyTorch               : {report.pytorch_version} (CPU Allocation: {report.pytorch_cpu_only})",
            f"     • CUDA Available        : {report.cuda_available}",
            f"     • NumPy                 : {report.numpy_version}",
            f"     • OpenCV                : {report.opencv_version}",
            f"     • SciPy                 : {report.scipy_version}",
            f"     • PyYAML                : {report.pyyaml_version}",
            f"     • Pandas Available      : {report.pandas_available}",
            "-" * 70,
            "  3. Feature Engineering Configuration :",
            f"     • Config Valid          : {report.config_valid}",
            f"     • Config Path           : {report.config_path}",
            "-" * 70,
            "  4. Upstream Artifact Verification :",
            f"     • Detection Artifact    : {'FOUND' if report.detection_artifact_found else 'NOT FOUND'}",
            f"     • Tracking Artifact     : {'FOUND' if report.tracking_artifact_found else 'NOT FOUND'}",
            f"     • Dataset Metadata      : {'FOUND' if report.metadata_artifact_found else 'NOT FOUND'}",
            f"     • Preprocessed Frames   : {'FOUND' if report.preprocessed_frames_found else 'NOT FOUND'}",
            f"     • Artifact Integrity   : {'VALID' if report.artifact_validation_valid else 'WARNING'}",
        ]

        if report.notes:
            lines.append("-" * 70)
            lines.append("  5. Notes & Diagnostics :")
            for n in report.notes:
                lines.append(f"     * {n}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI for running Feature Environment Verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Feature Engineering Environment Verifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to feature configuration YAML file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON telemetry.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    verifier = FeatureEnvironmentVerifier(config_path=args.config)
    report = verifier.run_audit()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(verifier.format_report_human_readable(report))

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
