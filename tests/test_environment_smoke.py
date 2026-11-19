"""Environment and dependency smoke test suite for PerceptGuard-AI.

Validates:
1. Python interpreter and version.
2. Core dependency imports.
3. PyTorch CPU tensor operations.
4. Ultralytics YOLO model loading and CPU inference.
5. OpenCV image read, write, and color transformations.
6. NumPy and PyTorch zero-copy interoperability.
7. LSTM CPU model construction and forward pass.
8. YAML configuration loading and schema validation.
9. CLI module importability.
10. Built-in environment verification command execution.
"""

import os
from pathlib import Path
import platform
import sys
import tempfile
import pytest

import numpy as np
import torch
import cv2
import yaml

from src.detection.single_image_detector import SingleImageDetector
from src.detection.yolo_config import YOLOConfig
from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.model_factory import build_forecasting_model
from tests.verify_environment import run_validation, check_python_version, verify_package


def test_01_python_interpreter():
    """Check 1: Python runtime and version compatibility."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 10
    is_valid, ver_str = check_python_version()
    assert is_valid is True
    assert "64" in platform.architecture()[0]


def test_02_core_dependency_imports():
    """Check 2: Core runtime and perception imports."""
    required = ["torch", "ultralytics", "cv2", "numpy", "pandas", "yaml", "sklearn", "matplotlib", "scipy", "psutil"]
    for pkg in required:
        ok, ver = verify_package(pkg)
        assert ok is True, f"Failed importing {pkg}: {ver}"


def test_03_pytorch_cpu_tensors():
    """Check 3: PyTorch CPU tensor allocation and mathematical operations."""
    assert torch.cuda.is_available() is False
    x = torch.randn(3, 4, device="cpu")
    w = torch.randn(4, 2, device="cpu")
    y = torch.matmul(x, w)
    assert y.shape == (3, 2)
    assert y.device == torch.device("cpu")


def test_04_yolo_model_loading_and_inference():
    """Check 4: Ultralytics YOLO model loading and CPU inference."""
    model_path = Path("models/yolo/yolo11n.pt")
    if not model_path.exists():
        pytest.skip(f"YOLO weights not present at {model_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        img_path = Path(tmpdir) / "smoke.jpg"
        cv2.imwrite(str(img_path), dummy_img)

        cfg = YOLOConfig.from_yaml("config/yolo_config.yaml")
        detector = SingleImageDetector(config=cfg)
        result = detector.detect(img_path)
        assert result.detection_count >= 0
        assert result.inference_time_ms >= 0.0


def test_05_opencv_io_and_processing():
    """Check 5: OpenCV image read/write and processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_img = np.full((64, 64, 3), 127, dtype=np.uint8)
        img_path = Path(tmpdir) / "test_io.png"
        success = cv2.imwrite(str(img_path), test_img)
        assert success is True

        loaded = cv2.imread(str(img_path))
        assert loaded is not None
        assert loaded.shape == (64, 64, 3)

        gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (64, 64)


def test_06_numpy_pytorch_interop():
    """Check 6: NumPy and PyTorch zero-copy bridge interoperability."""
    np_arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    torch_tensor = torch.from_numpy(np_arr)
    assert torch_tensor.dtype == torch.float32
    np_back = torch_tensor.numpy()
    assert np.allclose(np_arr, np_back)


def test_07_lstm_cpu_forward_pass():
    """Check 7: LSTM forecasting model initialization and CPU forward pass."""
    cfg = ForecastingConfig.from_yaml("config/forecasting_config.yaml")
    model = build_forecasting_model(cfg)
    assert model is not None

    batch_size = 2
    seq_len = 10
    input_dim = model.input_size
    dummy_input = torch.randn(batch_size, seq_len, input_dim, device="cpu")

    model.eval()
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (batch_size, model.output_size)
    assert out.device == torch.device("cpu")


def test_08_yaml_configuration_loading():
    """Check 8: All YAML configurations load and parse cleanly."""
    config_dir = Path("config")
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    assert len(yaml_files) >= 4
    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Failed parsing {yf.name}"


def test_09_cli_imports():
    """Check 9: Core entrypoint importability without side-effects."""
    import main
    assert hasattr(main, "build_parser")
    assert hasattr(main, "display_system_info")


def test_10_project_environment_validation_command():
    """Check 10: Existing tests/verify_environment.py validation runner."""
    result = run_validation()
    assert result is True
