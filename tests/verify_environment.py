"""PerceptGuard-AI: Environment and Dependency Validation Script.

Verifies the integrity of the Python runtime and required Day 1 dependencies:
- Python version & platform
- PyTorch & CPU/CUDA compute capability
- OpenCV (cv2)
- NumPy
- Pandas
- Ultralytics YOLO
- Scikit-learn
- Streamlit
- Matplotlib

Usage:
    python tests/verify_environment.py
"""

import importlib
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Tuple

# Direct Ultralytics to workspace or temp config directory to avoid global AppData permission warnings
_workspace_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(_workspace_root / ".yolo"))


def check_python_version() -> Tuple[bool, str]:
    """Verify Python runtime version."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    # Target Python 3.10+
    is_valid = v.major == 3 and v.minor >= 10
    return is_valid, ver_str


def verify_package(module_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Attempt to import a package and extract its version string."""
    name_to_import = import_name or module_name
    try:
        mod = importlib.import_module(name_to_import)
        version = getattr(mod, "__version__", "Installed (version unassigned)")
        return True, str(version)
    except ImportError as e:
        return False, f"Missing: {e.name if hasattr(e, 'name') else str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_torch_compute() -> Dict[str, str]:
    """Inspect PyTorch hardware acceleration and device availability."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU Mode Active)"
        return {
            "torch_version": torch.__version__,
            "cuda_available": str(cuda_available),
            "device": "CUDA" if cuda_available else "CPU",
            "device_name": gpu_name,
        }
    except ImportError:
        return {
            "torch_version": "Not Installed",
            "cuda_available": "False",
            "device": "Unavailable",
            "device_name": "Unavailable",
        }


def run_validation() -> bool:
    """Execute full environment verification test suite."""
    print("=" * 70)
    print("  PerceptGuard-AI: System Environment & Dependency Verification")
    print("=" * 70)

    # 1. System Runtime Information
    py_ok, py_ver = check_python_version()
    print(f"Platform            : {platform.platform()}")
    print(f"Operating System    : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Executable   : {sys.executable}")
    py_status = "[ PASS ]" if py_ok else "[ WARN ]"
    print(f"Python Version      : {py_ver:<15} {py_status}")
    print("-" * 70)

    # 2. Package Dependency Checks
    packages = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("PyTorch", "torch"),
        ("Ultralytics (YOLO)", "ultralytics"),
        ("Scikit-Learn", "sklearn"),
        ("Streamlit", "streamlit"),
        ("Matplotlib", "matplotlib"),
    ]

    all_passed = py_ok
    missing_packages: List[str] = []

    print(f"{'Package':<24} | {'Status':<10} | {'Detected Version'}")
    print("-" * 70)

    for display_name, mod_name in packages:
        ok, ver_info = verify_package(mod_name)
        status_tag = "[ PASS ]" if ok else "[ FAIL ]"
        print(f"{display_name:<24} | {status_tag:<10} | {ver_info}")
        if not ok:
            all_passed = False
            missing_packages.append(display_name)

    print("-" * 70)

    # 3. Hardware Compute Verification
    torch_info = check_torch_compute()
    print(f"PyTorch Version     : {torch_info['torch_version']}")
    print(f"CUDA Available      : {torch_info['cuda_available']}")
    print(f"Target Compute Unit : {torch_info['device']} ({torch_info['device_name']})")
    print("=" * 70)

    # 4. Final Verdict
    if all_passed:
        print("  RESULT: Environment validation PASSED. All core dependencies ready.")
        print("=" * 70)
        return True
    else:
        print(f"  RESULT: Validation FAILED. Missing or broken: {', '.join(missing_packages)}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
