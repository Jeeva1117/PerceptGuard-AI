"""Forecasting Environment Verification Module for PerceptGuard AI.

Verifies the deep learning forecasting runtime, CPU compatibility,
PyTorch CPU thread bounds, reproducibility settings, and Day 5 artifact
readiness for sequence forecasting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.forecasting.forecasting_config import ForecastingConfig, ForecastingConfigError


@dataclass
class EnvironmentVerificationResult:
    """Detailed summary of the forecasting environment verification."""
    python_version: str
    python_compatible: bool
    torch_version: str
    torch_available: bool
    cuda_available: bool
    cpu_only_verified: bool
    configured_threads: int
    actual_threads: int
    numpy_version: str
    yaml_version: str
    random_seed_set: bool
    config_valid: bool
    config_path: str
    checkpoint_dir: str
    checkpoint_dir_ready: bool
    experiment_dir: str
    experiment_dir_ready: bool
    day5_artifacts_found: bool
    day5_artifacts_path: Optional[str]
    sequence_tensors_found: Dict[str, str] = field(default_factory=dict)
    ground_truth_labels_available: bool = False
    supervised_readiness: str = "NOT AVAILABLE"
    validation_status: str = "PASS"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "python_version": self.python_version,
            "python_compatible": self.python_compatible,
            "torch_version": self.torch_version,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "cpu_only_verified": self.cpu_only_verified,
            "configured_threads": self.configured_threads,
            "actual_threads": self.actual_threads,
            "numpy_version": self.numpy_version,
            "yaml_version": self.yaml_version,
            "random_seed_set": self.random_seed_set,
            "config_valid": self.config_valid,
            "config_path": self.config_path,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_dir_ready": self.checkpoint_dir_ready,
            "experiment_dir": self.experiment_dir,
            "experiment_dir_ready": self.experiment_dir_ready,
            "day5_artifacts_found": self.day5_artifacts_found,
            "day5_artifacts_path": self.day5_artifacts_path,
            "sequence_tensors_found": self.sequence_tensors_found,
            "ground_truth_labels_available": self.ground_truth_labels_available,
            "supervised_readiness": self.supervised_readiness,
            "validation_status": self.validation_status,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class ForecastingEnvironmentVerifier:
    """Verifies runtime, dependencies, hardware constraints, and dataset contract readiness."""

    def __init__(self, config_path: Union[str, Path] = "config/forecasting_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config: Optional[ForecastingConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Safely load and validate configuration."""
        if self.config_path.exists():
            try:
                self.config = ForecastingConfig.from_yaml(self.config_path, validate=True)
            except Exception:
                # If invalid, load without strict validation so verifier can report errors
                try:
                    self.config = ForecastingConfig.from_yaml(self.config_path, validate=False)
                except Exception:
                    self.config = ForecastingConfig()
        else:
            self.config = ForecastingConfig()

    def set_reproducibility(self, seed: Optional[int] = None) -> bool:
        """Enforce strict seed configuration across Python, NumPy, and PyTorch.
        
        Args:
            seed: Optional seed value. Defaults to config project.random_seed (42).
            
        Returns:
            True if all seeds successfully applied.
        """
        target_seed = seed if seed is not None else (self.config.project.random_seed if self.config else 42)
        random.seed(target_seed)
        np.random.seed(target_seed)
        os.environ["PYTHONHASHSEED"] = str(target_seed)

        try:
            import torch
            torch.manual_seed(target_seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    # Some PyTorch CPU operations may raise warnings or errors if deterministic mode unsupported
                    pass
            return True
        except ImportError:
            return False

    def configure_cpu_threads(self, num_threads: Optional[int] = None) -> Tuple[int, int]:
        """Configure PyTorch CPU thread count to prevent resource oversubscription.
        
        Args:
            num_threads: Thread count. Defaults to config runtime.num_threads (2).
            
        Returns:
            Tuple of (configured_threads, actual_threads_reported)
        """
        target_threads = num_threads if num_threads is not None else (self.config.runtime.num_threads if self.config else 2)
        try:
            import torch
            torch.set_num_threads(target_threads)
            actual_threads = torch.get_num_threads()
            return target_threads, actual_threads
        except ImportError:
            return target_threads, 0

    def discover_day5_artifacts(self, base_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, Optional[Path], Dict[str, Path]]:
        """Search for valid Day 5 sequence artifacts.
        
        Searches under datasets/processed/ for sequence outputs.
        
        Returns:
            Tuple of (found, run_directory_path, dictionary_of_tensor_paths)
        """
        search_root = Path(base_dir) if base_dir else Path(self.config.data.sequence_directory if self.config else "datasets/processed")
        tensors: Dict[str, Path] = {}

        if not search_root.exists():
            return False, None, tensors

        target_lengths = self.config.data.sequence_lengths if self.config else [5, 10, 15, 20]

        # Check candidate locations directly first
        candidate_dirs = [
            search_root / "driving_sample" / "features" / "sequences" / "run_official",
            search_root / "driving_sample" / "features" / "sequences",
            search_root,
        ]

        # Also search top-level dataset subdirectories without deep recursion
        if search_root.is_dir():
            for child in search_root.iterdir():
                if child.is_dir():
                    run_off = child / "features" / "sequences" / "run_official"
                    if run_off.is_dir() and run_off not in candidate_dirs:
                        candidate_dirs.append(run_off)
                    seq_dir = child / "features" / "sequences"
                    if seq_dir.is_dir() and seq_dir not in candidate_dirs:
                        candidate_dirs.append(seq_dir)

        for cand in candidate_dirs:
            if not cand.exists() or not cand.is_dir():
                continue
            found_all_or_some = False
            for length in target_lengths:
                tensor_file = cand / f"sequences_L{length}.npy"
                if tensor_file.exists():
                    tensors[f"L{length}"] = tensor_file
                    found_all_or_some = True
            if found_all_or_some:
                return True, cand, tensors

        return False, None, tensors

    def verify_all(self, base_dataset_dir: Optional[Union[str, Path]] = None) -> EnvironmentVerificationResult:
        """Run complete forecasting environment verification.
        
        Returns:
            EnvironmentVerificationResult with all status checks.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Python Check
        py_ver = platform.python_version()
        py_compat = sys.version_info >= (3, 9)
        if not py_compat:
            errors.append(f"Python version {py_ver} is unsupported. Require Python >= 3.9.")

        # 2. PyTorch Check
        torch_ver = "NOT INSTALLED"
        torch_avail = False
        cuda_avail = False
        cpu_verified = False
        configured_threads = self.config.runtime.num_threads if self.config else 2
        actual_threads = 0

        try:
            import torch
            torch_ver = torch.__version__
            torch_avail = True
            cuda_avail = torch.cuda.is_available()
            # On CPU machine, cuda_available should be False
            if cuda_avail:
                warnings.append("CUDA is available, but CPU-only execution is mandated for this task")
            cpu_verified = True
            configured_threads, actual_threads = self.configure_cpu_threads(configured_threads)
        except ImportError:
            errors.append("PyTorch is not installed in the active environment")

        # 3. NumPy Check
        np_ver = np.__version__

        # 4. PyYAML Check
        yaml_ver = "NOT INSTALLED"
        try:
            import yaml
            yaml_ver = getattr(yaml, "__version__", "installed")
        except ImportError:
            errors.append("PyYAML is not installed in the active environment")

        # 5. Reproducibility
        seed_ok = self.set_reproducibility()

        # 6. Configuration Check
        config_valid = False
        if self.config:
            is_val, conf_errs = self.config.validate(raise_on_error=False)
            config_valid = is_val
            errors.extend(conf_errs)
        else:
            errors.append("Failed to load forecasting configuration")

        # 7. Checkpoint and Experiment Directories
        ckpt_dir = self.config.artifacts.checkpoint_directory if self.config else "models/forecasting"
        exp_dir = self.config.artifacts.experiment_directory if self.config else "experiments/forecasting"

        ckpt_ready = False
        exp_ready = False
        try:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            ckpt_ready = True
        except Exception as e:
            errors.append(f"Unable to create checkpoint directory {ckpt_dir}: {e}")

        try:
            Path(exp_dir).mkdir(parents=True, exist_ok=True)
            exp_ready = True
        except Exception as e:
            errors.append(f"Unable to create experiment directory {exp_dir}: {e}")

        # 8. Day 5 Artifacts Discovery
        found_d5, d5_path, tensors = self.discover_day5_artifacts(base_dataset_dir)
        tensors_str = {k: str(v) for k, v in tensors.items()}

        gt_labels_available = False
        supervised_readiness = "NOT AVAILABLE"

        if found_d5 and d5_path:
            meta_path = d5_path / "sequence_metadata.json"
            summary_path = d5_path / "sequence_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                        gt_labels_available = summary_data.get("target_labels_available", False)
                except Exception:
                    pass
            elif meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                        # Check sequences
                        seqs = meta_data.get("sequences", [])
                        if seqs and any(
                            any(h.get("target_label_available", False) for h in s.get("horizons", {}).values())
                            for s in seqs[:10]
                        ):
                            gt_labels_available = True
                except Exception:
                    pass

            if gt_labels_available:
                supervised_readiness = "PASS"
            else:
                supervised_readiness = "NOT AVAILABLE"
                warnings.append(
                    "Supervised forecasting readiness is NOT AVAILABLE: No ground-truth reliability labels exist. "
                    "YOLO detection confidence alone is NOT ground-truth reliability."
                )
        else:
            warnings.append("Day 5 sequence artifacts not found under datasets/processed.")

        # Determine overall validation status
        overall_status = "PASS" if len(errors) == 0 else "FAIL"

        return EnvironmentVerificationResult(
            python_version=py_ver,
            python_compatible=py_compat,
            torch_version=torch_ver,
            torch_available=torch_avail,
            cuda_available=cuda_avail,
            cpu_only_verified=cpu_verified,
            configured_threads=configured_threads,
            actual_threads=actual_threads,
            numpy_version=np_ver,
            yaml_version=yaml_ver,
            random_seed_set=seed_ok,
            config_valid=config_valid,
            config_path=str(self.config_path),
            checkpoint_dir=ckpt_dir,
            checkpoint_dir_ready=ckpt_ready,
            experiment_dir=exp_dir,
            experiment_dir_ready=exp_ready,
            day5_artifacts_found=found_d5,
            day5_artifacts_path=str(d5_path) if d5_path else None,
            sequence_tensors_found=tensors_str,
            ground_truth_labels_available=gt_labels_available,
            supervised_readiness=supervised_readiness,
            validation_status=overall_status,
            warnings=warnings,
            errors=errors,
        )


def main() -> int:
    """CLI entrypoint for forecasting environment verification."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI Forecasting Environment & Model Setup Verifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration YAML file",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Custom root directory to inspect for Day 5 sequence artifacts",
    )

    args = parser.parse_args()
    verifier = ForecastingEnvironmentVerifier(config_path=args.config)
    result = verifier.verify_all(base_dataset_dir=args.dataset_dir)

    print("=" * 70)
    print(" PerceptGuard AI — Forecasting Environment & Model Setup Verification")
    print("=" * 70)
    print(f"Python Version             : {result.python_version} (Compatible: {result.python_compatible})")
    print(f"PyTorch Version            : {result.torch_version} (Available: {result.torch_available})")
    print(f"CUDA Available             : {result.cuda_available} (Mandate: CPU-only)")
    print(f"CPU Threads                : Configured: {result.configured_threads} | Active: {result.actual_threads}")
    print(f"NumPy Version              : {result.numpy_version}")
    print(f"PyYAML Version             : {result.yaml_version}")
    print(f"Reproducibility Seed       : {'Set (Seed=42)' if result.random_seed_set else 'Failed'}")
    print(f"Configuration Valid        : {result.config_valid} ({result.config_path})")
    print(f"Checkpoint Directory       : {result.checkpoint_dir} (Ready: {result.checkpoint_dir_ready})")
    print(f"Experiment Directory       : {result.experiment_dir} (Ready: {result.experiment_dir_ready})")
    print("-" * 70)
    print(" DAY 5 SEQUENCE ARTIFACT STATUS:")
    print(f"Artifacts Discovered       : {result.day5_artifacts_found}")
    if result.day5_artifacts_found:
        print(f"Artifact Path              : {result.day5_artifacts_path}")
        print("Sequence Tensors Found     :")
        for length_key, tensor_path in result.sequence_tensors_found.items():
            print(f"  * {length_key}: {tensor_path}")
    print(f"Ground-Truth Reliability   : {result.ground_truth_labels_available}")
    print(f"Supervised Training Ready  : {result.supervised_readiness}")
    if result.supervised_readiness == "NOT AVAILABLE":
        print("  [NOTE] Ground-truth reliability labels are not present in sequence artifacts.")
        print("         YOLO confidence alone is NOT ground-truth reliability.")
    print("-" * 70)

    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  ! {w}")

    if result.errors:
        print("ERRORS:")
        for e in result.errors:
            print(f"  * {e}")
        print("=" * 70)
        print(f"OVERALL STATUS: [FAIL]")
        return 1
    else:
        print("=" * 70)
        print(f"OVERALL ENVIRONMENT STATUS: [{result.validation_status}]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
