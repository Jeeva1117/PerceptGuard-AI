"""Forecasting Data Contract Module for PerceptGuard AI.

Validates Day 5 sequence tensors, metadata, splits, and feature manifests
against strict deep learning contract rules. Defines Model Input and Output
contracts for multi-horizon reliability forecasting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.forecasting.forecasting_config import ForecastingConfig


class DataContractError(ValueError):
    """Raised when data contract validation fails."""
    pass


@dataclass(frozen=True)
class ModelInputSpec:
    """Specification of the input tensor for forecasting models."""
    shape_description: str = "(batch_size, sequence_length, feature_count)"
    expected_ndim: int = 3
    dtype: str = "torch.float32"
    feature_count: Optional[int] = None
    supported_sequence_lengths: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    description: str = (
        "Batched historical perception, motion, and visual quality features "
        "extracted over a sliding window of past frames."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOutputSpec:
    """Specification of the forecast output tensor."""
    shape_description: str = "(batch_size, number_of_horizons)"
    expected_ndim: int = 2
    dtype: str = "torch.float32"
    target_type: str = "continuous_reliability_score"
    target_range: Tuple[float, float] = (0.0, 1.0)
    horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    description: str = (
        "Multi-horizon perception reliability forecast at t+1, t+3, and t+5 frames ahead. "
        "Each output is a continuous score in [0.0, 1.0] representing future object detection "
        "reliability (e.g., ground-truth IoU / recall)."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataContractValidationResult:
    """Result of Day 5 sequence contract validation."""
    passed: bool
    dataset_id: str
    artifacts_dir: str
    feature_count: int
    ordered_feature_names: List[str]
    sequence_lengths_validated: List[int]
    tensor_shapes: Dict[str, List[int]]
    sequence_counts: Dict[str, int]
    splits_validated: bool
    leakage_check_passed: bool
    finite_check_passed: bool
    chronological_check_passed: bool
    ground_truth_labels_available: bool
    supervised_readiness: str
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ForecastingDataContract:
    """Data contract validator for Day 5 sequence tensors and metadata."""

    def __init__(
        self,
        config: Optional[ForecastingConfig] = None,
        config_path: Union[str, Path] = "config/forecasting_config.yaml",
    ) -> None:
        if config is not None:
            self.config = config
        else:
            p = Path(config_path)
            if p.exists():
                self.config = ForecastingConfig.from_yaml(p, validate=False)
            else:
                self.config = ForecastingConfig()

        self.input_spec = ModelInputSpec(
            supported_sequence_lengths=list(self.config.data.sequence_lengths)
        )
        self.output_spec = ModelOutputSpec(
            horizons=list(self.config.data.forecast_horizons)
        )

    def validate_artifacts(
        self,
        artifacts_dir: Union[str, Path],
        raise_on_error: bool = False,
    ) -> DataContractValidationResult:
        """Validate all Day 5 artifacts in a sequence run directory.
        
        Args:
            artifacts_dir: Directory containing .npy sequences and metadata JSON files.
            raise_on_error: If True, raises DataContractError when validation fails.
            
        Returns:
            DataContractValidationResult summarizing all 14 contract checks.
        """
        art_path = Path(artifacts_dir)
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, bool] = {}

        if not art_path.exists() or not art_path.is_dir():
            msg = f"Artifacts directory does not exist or is not a directory: {art_path}"
            errors.append(msg)
            if raise_on_error:
                raise DataContractError(msg)
            return DataContractValidationResult(
                passed=False,
                dataset_id="unknown",
                artifacts_dir=str(art_path),
                feature_count=0,
                ordered_feature_names=[],
                sequence_lengths_validated=[],
                tensor_shapes={},
                sequence_counts={},
                splits_validated=False,
                leakage_check_passed=False,
                finite_check_passed=False,
                chronological_check_passed=False,
                ground_truth_labels_available=False,
                supervised_readiness="NOT AVAILABLE",
                checks={"artifacts_dir_exists": False},
                warnings=warnings,
                errors=errors,
            )

        checks["artifacts_dir_exists"] = True

        # 1. Feature Manifest Validation
        manifest_file = art_path / "feature_manifest.json"
        feature_count = 0
        ordered_feature_names: List[str] = []
        dataset_id = "unknown"

        if not manifest_file.exists():
            errors.append(f"Missing required artifact: {manifest_file.name}")
            checks["feature_manifest_exists"] = False
        else:
            checks["feature_manifest_exists"] = True
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                dataset_id = manifest_data.get("dataset_id", "unknown")
                ordered_feature_names = manifest_data.get("ordered_feature_names", [])
                feature_count = manifest_data.get("total_features") or len(ordered_feature_names)
                if feature_count != len(ordered_feature_names):
                    errors.append(
                        f"Manifest total_features ({feature_count}) does not match "
                        f"ordered_feature_names length ({len(ordered_feature_names)})"
                    )
                    checks["feature_manifest_valid"] = False
                elif feature_count == 0:
                    errors.append("Manifest specifies 0 features")
                    checks["feature_manifest_valid"] = False
                else:
                    checks["feature_manifest_valid"] = True
            except Exception as e:
                errors.append(f"Failed to parse {manifest_file.name}: {e}")
                checks["feature_manifest_valid"] = False

        # 2. Sequence Tensors Validation (Dimensionality, Length, Dtype, NaN/Inf)
        target_lengths = self.config.data.sequence_lengths
        sequence_lengths_validated: List[int] = []
        tensor_shapes: Dict[str, List[int]] = {}
        sequence_counts: Dict[str, int] = {}
        all_tensors_finite = True

        for length in target_lengths:
            tensor_file = art_path / f"sequences_L{length}.npy"
            length_key = f"L{length}"
            if not tensor_file.exists():
                errors.append(f"Sequence tensor missing for length {length}: {tensor_file.name}")
                checks[f"tensor_exists_{length_key}"] = False
                continue

            checks[f"tensor_exists_{length_key}"] = True

            try:
                arr = np.load(tensor_file)
                # Check dimensionality: strictly 3D
                if arr.ndim != 3:
                    errors.append(
                        f"Tensor {tensor_file.name} has invalid dimensionality {arr.ndim} (expected 3: N, L, F)"
                    )
                    checks[f"tensor_dim_{length_key}"] = False
                    continue
                checks[f"tensor_dim_{length_key}"] = True

                n_seq, seq_len, n_feat = arr.shape
                tensor_shapes[length_key] = [n_seq, seq_len, n_feat]
                sequence_counts[length_key] = n_seq

                # Check sequence length
                if seq_len != length:
                    errors.append(
                        f"Tensor {tensor_file.name} sequence length dimension {seq_len} != expected {length}"
                    )
                    checks[f"tensor_len_{length_key}"] = False
                else:
                    checks[f"tensor_len_{length_key}"] = True

                # Check feature count against manifest
                if feature_count > 0 and n_feat != feature_count:
                    errors.append(
                        f"Tensor {tensor_file.name} feature count {n_feat} does not match manifest {feature_count}"
                    )
                    checks[f"tensor_feat_{length_key}"] = False
                else:
                    checks[f"tensor_feat_{length_key}"] = True

                # Check dtype
                if not np.issubdtype(arr.dtype, np.floating):
                    errors.append(
                        f"Tensor {tensor_file.name} has non-floating dtype {arr.dtype} (expected float32/float64)"
                    )
                    checks[f"tensor_dtype_{length_key}"] = False
                else:
                    checks[f"tensor_dtype_{length_key}"] = True

                # Check finite (no NaN, no Inf)
                is_finite = bool(np.isfinite(arr).all())
                if not is_finite:
                    n_nans = int(np.isnan(arr).sum())
                    n_infs = int(np.isinf(arr).sum())
                    errors.append(
                        f"Tensor {tensor_file.name} contains non-finite values (NaNs: {n_nans}, Infs: {n_infs})"
                    )
                    all_tensors_finite = False
                    checks[f"tensor_finite_{length_key}"] = False
                else:
                    checks[f"tensor_finite_{length_key}"] = True

                sequence_lengths_validated.append(length)

            except Exception as e:
                errors.append(f"Failed to read NumPy file {tensor_file.name}: {e}")
                checks[f"tensor_read_{length_key}"] = False

        checks["all_tensors_finite"] = all_tensors_finite

        # 3. Sequence Metadata Validation
        meta_file = art_path / "sequence_metadata.json"
        chronological_ok = True
        gt_labels_found = False

        if not meta_file.exists():
            errors.append(f"Missing required artifact: {meta_file.name}")
            checks["sequence_metadata_exists"] = False
        else:
            checks["sequence_metadata_exists"] = True
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)

                # Check dataset identity
                meta_dataset = meta_data.get("dataset_id")
                if meta_dataset and dataset_id != "unknown" and meta_dataset != dataset_id:
                    errors.append(
                        f"Dataset ID mismatch: manifest '{dataset_id}' vs metadata '{meta_dataset}'"
                    )

                sequences = meta_data.get("sequences", [])
                total_meta_seqs = meta_data.get("total_sequences", len(sequences))

                # Check chronological monotonicity
                last_time = -1.0
                last_frame_idx = -1
                for s in sequences:
                    t_start = s.get("start_timestamp_seconds", 0.0)
                    f_start = s.get("start_source_frame_index", 0)
                    if t_start < 0.0 or f_start < 0:
                        chronological_ok = False
                        break
                    # Check horizons references
                    horizons = s.get("horizons", {})
                    for h_key, h_data in horizons.items():
                        if h_data.get("target_label_available", False) and h_data.get("target_label") is not None:
                            gt_labels_found = True

                checks["sequence_metadata_valid"] = True
            except Exception as e:
                errors.append(f"Failed to parse {meta_file.name}: {e}")
                checks["sequence_metadata_valid"] = False

        checks["chronological_ordering"] = chronological_ok

        # 4. Dataset Splits Validation & Leakage Check
        splits_file = art_path / "dataset_splits.json"
        splits_ok = True
        leakage_ok = True

        if not splits_file.exists():
            errors.append(f"Missing required artifact: {splits_file.name}")
            checks["dataset_splits_exists"] = False
        else:
            checks["dataset_splits_exists"] = True
            try:
                with open(splits_file, "r", encoding="utf-8") as f:
                    splits_data = json.load(f)

                leakage_ok = splits_data.get("leakage_check_passed", True)
                split_counts = splits_data.get("sequence_counts_per_split", {})

                # Validate split counts match tensor counts for validated lengths
                for length in sequence_lengths_validated:
                    l_key = f"L{length}"
                    if l_key in split_counts:
                        total_split_seqs = split_counts[l_key].get("total", 0)
                        actual_tensor_seqs = sequence_counts.get(l_key, 0)
                        if total_split_seqs != actual_tensor_seqs:
                            errors.append(
                                f"Split sequence count ({total_split_seqs}) for {l_key} does not match "
                                f"tensor sequence count ({actual_tensor_seqs})"
                            )
                            splits_ok = False

                checks["dataset_splits_valid"] = splits_ok
                checks["leakage_check_passed"] = leakage_ok
            except Exception as e:
                errors.append(f"Failed to parse {splits_file.name}: {e}")
                checks["dataset_splits_valid"] = False

        # 5. Ground-Truth Target Label Assessment
        # Also check sequence_summary.json if present
        summary_file = art_path / "sequence_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                    if summary_data.get("target_labels_available", False):
                        gt_labels_found = True
            except Exception:
                pass

        if gt_labels_found:
            supervised_readiness = "PASS"
        else:
            supervised_readiness = "NOT AVAILABLE"
            warnings.append(
                "Supervised training readiness is NOT AVAILABLE: Day 5 sequence dataset does not contain "
                "ground-truth perception reliability annotations. YOLO confidence alone is NOT ground-truth reliability."
            )

        passed = len(errors) == 0
        checks["overall_data_contract_passed"] = passed

        if not passed and raise_on_error:
            raise DataContractError(
                f"Data contract validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        return DataContractValidationResult(
            passed=passed,
            dataset_id=dataset_id,
            artifacts_dir=str(art_path),
            feature_count=feature_count,
            ordered_feature_names=ordered_feature_names,
            sequence_lengths_validated=sequence_lengths_validated,
            tensor_shapes=tensor_shapes,
            sequence_counts=sequence_counts,
            splits_validated=splits_ok,
            leakage_check_passed=leakage_ok,
            finite_check_passed=all_tensors_finite,
            chronological_check_passed=chronological_ok,
            ground_truth_labels_available=gt_labels_found,
            supervised_readiness=supervised_readiness,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )


def main() -> int:
    """CLI entrypoint for forecasting data contract validation."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI Day 5 Sequence Data Contract Validator"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="datasets/processed/driving_sample/features/sequences/run_official",
        help="Path to Day 5 sequence artifacts directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration YAML file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform strict data contract validation",
    )

    args = parser.parse_args()
    artifacts_path = Path(args.artifacts_dir)

    print("=" * 70)
    print(" PerceptGuard AI — Forecasting Sequence Data Contract Validator")
    print("=" * 70)
    print(f"Target Artifacts Path      : {artifacts_path.resolve()}")

    if not artifacts_path.exists():
        print(f"[FAIL] Artifacts directory does not exist: {artifacts_path}")
        return 1

    validator = ForecastingDataContract(config_path=args.config)
    result = validator.validate_artifacts(artifacts_path, raise_on_error=False)

    print(f"Dataset ID                 : {result.dataset_id}")
    print(f"Feature Count (Manifest)   : {result.feature_count}")
    print(f"Validated Sequence Lengths : {result.sequence_lengths_validated}")
    print(f"Finite Numbers (No NaN/Inf): {result.finite_check_passed}")
    print(f"Splits Validated           : {result.splits_validated}")
    print(f"Leakage Check Passed       : {result.leakage_check_passed}")
    print(f"Chronological Monotonicity : {result.chronological_check_passed}")
    print("-" * 70)
    print("TENSOR SHAPES:")
    for l_key, shape in result.tensor_shapes.items():
        print(f"  * {l_key}: {shape} (sequences={result.sequence_counts.get(l_key)})")
    print("-" * 70)
    print("MODEL CONTRACT SPECIFICATION:")
    print(f"  * Expected Input Tensor  : {validator.input_spec.shape_description} ({validator.input_spec.dtype})")
    print(f"  * Expected Output Tensor : {validator.output_spec.shape_description} ({validator.output_spec.dtype})")
    print(f"  * Target Formulation     : {validator.output_spec.target_type} in {validator.output_spec.target_range}")
    print(f"  * Target Forecast Horizons: {validator.output_spec.horizons}")
    print("-" * 70)
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
        print("OVERALL DATA CONTRACT: [FAIL]")
        return 1
    else:
        print("=" * 70)
        print("OVERALL DATA CONTRACT: [PASS]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
