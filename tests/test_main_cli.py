"""Unit tests for the main CLI entrypoint (main.py)."""

import subprocess
import sys
from pathlib import Path
import pytest

from main import build_parser, display_system_info, main


def test_main_parser_flags():
    """Verify CLI parser configuration and arguments."""
    parser = build_parser()
    args = parser.parse_args(["--info"])
    assert args.info is True

    args_device = parser.parse_args(["--device", "cpu"])
    assert args_device.device == "cpu"

    args_audit = parser.parse_args(["--audit"])
    assert args_audit.audit is True

    args_env = parser.parse_args(["--verify-env"])
    assert args_env.verify_env is True


def test_main_display_system_info(caplog):
    """Verify system info displays without error."""
    with caplog.at_level("INFO"):
        display_system_info(target_device="cpu")
    assert "PerceptGuard-AI: Predictive Object Detection Reliability" in caplog.text
    assert "Subsystem Pipeline Status:" in caplog.text


def test_main_entrypoint_info(monkeypatch):
    """Verify calling main() with --info returns 0."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--info"])
    assert main() == 0


def test_main_cli_subprocess():
    """Verify subprocess execution of python main.py --info."""
    cmd = [sys.executable, "main.py", "--info"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Predictive Object Detection Reliability" in res.stderr or "Predictive Object Detection Reliability" in res.stdout
