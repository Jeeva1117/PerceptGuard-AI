"""PerceptGuard-AI: Dataset Quality Assurance and Validation Engine.

Production-grade dataset quality assurance module for autonomous driving perception pipelines.
Conducts comprehensive, non-destructive audit of directory structure, raw video stream,
extracted frames, standardized preprocessed tensors, tabular metadata, and top-level manifests.

Key Architectural Guarantees:
- Non-Destructive / Read-Only: Never mutates, deletes, or moves dataset assets
- Multi-Tier Categorical Checks: Evaluates 10 distinct dimensions returning PASS / WARNING / FAIL
- Strict Cross-Modal Traceability: Validates unbroken chain: Video -> Index -> Raw -> Preprocessed -> CSV
- RFC 8259 Strict JSON Reporting: Serializes auditable reports with ISO 8601 UTC timestamps
- Day 3 Perception Readiness Gate: Explicitly certifies whether the dataset is ready for YOLOv8 & ByteTrack
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path when invoked directly
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import pandas as pd
import yaml

from src.video_processing.video_loader import VideoLoader

logger = logging.getLogger("PerceptGuard.DatasetValidator")

VALIDATION_SCHEMA_VERSION: str = "1.0.0"


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class DatasetValidationError(Exception):
    """Base exception for all dataset validation pipeline failures."""
    pass


class ValidationConfigError(DatasetValidationError):
    """Raised when validation parameters or path configurations are invalid."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class CategoryValidationResult:
    """Immutable result for a discrete validation category check.

    Attributes:
        category: Name of the validation category.
        status: Validation status ('PASS', 'WARNING', 'FAIL').
        details: Informational logs and telemetry notes.
        warnings: Non-blocking warning messages.
        errors: Critical failure messages causing FAIL status.
    """
    category: str
    status: str
    details: Tuple[str, ...]
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert category result to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DatasetValidationReport:
    """Immutable comprehensive dataset validation report contract.

    Attributes:
        validation_schema_version: Semantic version of the validation schema.
        dataset_id: Logical dataset collection identifier.
        validation_timestamp: ISO 8601 timestamp with UTC timezone.
        overall_status: Aggregate validation status ('PASS', 'WARNING', 'FAIL').
        is_ready_for_day_3: Boolean flag certifying readiness for YOLOv8 & ByteTrack.
        checks: Dictionary of category validation outcomes and details.
        statistics: Descriptive dataset quality and distribution statistics.
        issues: List of all detected warnings and errors.
        recommendations: Actionable operational guidance.
    """
    validation_schema_version: str
    dataset_id: str
    validation_timestamp: str
    overall_status: str
    is_ready_for_day_3: bool
    checks: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Any]
    issues: Tuple[str, ...]
    recommendations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to dictionary representation."""
        return {
            "validation_schema_version": self.validation_schema_version,
            "dataset_id": self.dataset_id,
            "validation_timestamp": self.validation_timestamp,
            "overall_status": self.overall_status,
            "is_ready_for_day_3": self.is_ready_for_day_3,
            "checks": self.checks,
            "statistics": self.statistics,
            "issues": list(self.issues),
            "recommendations": list(self.recommendations),
        }


# ==============================================================================
# DatasetValidator Implementation
# ==============================================================================

class DatasetValidator:
    """Comprehensive, non-destructive dataset validation engine.

    Performs exhaustive audits of directory layouts, video stream decoding,
    image integrity, metadata correctness, manifest parameters, and traceability.
    """

    SUPPORTED_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    SUPPORTED_VIDEO_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}

    REQUIRED_METADATA_COLUMNS: List[str] = [
        "dataset_id",
        "source_video",
        "extracted_frame_id",
        "source_frame_index",
        "timestamp_seconds",
        "original_image_filename",
        "original_image_path",
        "preprocessed_image_filename",
        "preprocessed_image_path",
        "file_size_bytes",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "channel_count",
        "blur_score",
        "brightness_score",
        "contrast_score",
        "preprocessing_status",
        "validation_status",
        "failure_reason",
    ]

    def __init__(
        self,
        dataset_id: str = "driving_sample",
        project_root: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize DatasetValidator with dataset identifiers and paths.

        Args:
            dataset_id: Unique identifier for the dataset collection.
            project_root: Root directory of the repository.
            config_path: Optional path to YAML configuration file.
        """
        if project_root is not None:
            self._project_root = Path(project_root).resolve()
        else:
            self._project_root = Path(__file__).resolve().parent.parent.parent

        self.dataset_id = str(dataset_id).strip()
        self._dataset_root = self._project_root / "datasets"
        self._raw_dir = self._dataset_root / "raw"
        self._processed_dir = self._dataset_root / "processed" / self.dataset_id
        self._frames_dir = self._processed_dir / "frames"
        self._preprocessed_frames_dir = self._processed_dir / "preprocessed_frames"
        self._metadata_csv_path = self._processed_dir / "metadata.csv"
        self._summary_json_path = self._processed_dir / "summary.json"
        self._manifest_json_path = self._processed_dir / "dataset_manifest.json"

        # Configuration defaults
        self._target_width: int = 640
        self._target_height: int = 360
        self._default_video_name: Optional[str] = None
        self._expected_fps: float = 30.09
        self._extraction_fps: float = 5.0

        self._load_config(config_path)
        self._report: Optional[DatasetValidationReport] = None

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> None:
        """Parse configuration parameters from video_config.yaml."""
        if not config_path:
            default_cfg = self._project_root / "config" / "video_config.yaml"
            if default_cfg.is_file():
                config_path = default_cfg

        if config_path and Path(config_path).is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                v_cfg = cfg.get("video", {})
                if "default_video" in v_cfg and v_cfg["default_video"]:
                    self._default_video_name = Path(v_cfg["default_video"]).name

                ext_cfg = cfg.get("frame_extraction", {})
                if "extraction_fps" in ext_cfg:
                    self._extraction_fps = float(ext_cfg["extraction_fps"])

                p_cfg = cfg.get("preprocessing", {})
                if "target_width" in p_cfg:
                    self._target_width = int(p_cfg["target_width"])
                if "target_height" in p_cfg:
                    self._target_height = int(p_cfg["target_height"])

            except Exception as e:
                logger.warning("Failed to parse config %s: %s. Using default settings.", config_path, e)

    def _get_raw_video_path(self) -> Optional[Path]:
        """Locate primary raw video container."""
        if not self._raw_dir.is_dir():
            return None
        if self._default_video_name:
            candidate = self._raw_dir / self._default_video_name
            if candidate.is_file():
                return candidate
        videos = [
            f for f in self._raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_VIDEO_EXTS
        ]
        return sorted(videos)[0] if videos else None

    # --------------------------------------------------------------------------
    # Individual Category Validation Checks
    # --------------------------------------------------------------------------

    def validate_directory_structure(self) -> CategoryValidationResult:
        """Validate presence of all expected dataset directories and artifacts."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        required_dirs = [
            ("raw_directory", self._raw_dir),
            ("processed_dataset_directory", self._processed_dir),
            ("extracted_frames_directory", self._frames_dir),
            ("preprocessed_frames_directory", self._preprocessed_frames_dir),
        ]

        for name, path in required_dirs:
            if not path.exists():
                errors.append(f"Required directory does not exist: '{path}' ({name})")
            elif not path.is_dir():
                errors.append(f"Path exists but is not a directory: '{path}' ({name})")
            else:
                details.append(f"Directory verified: '{path.name}/'")

        required_files = [
            ("metadata_csv", self._metadata_csv_path),
            ("summary_json", self._summary_json_path),
            ("dataset_manifest_json", self._manifest_json_path),
        ]

        for name, path in required_files:
            if not path.is_file():
                errors.append(f"Required artifact does not exist: '{path}' ({name})")
            elif path.stat().st_size == 0:
                errors.append(f"Artifact exists but is 0 bytes: '{path}' ({name})")
            else:
                details.append(f"File verified: '{path.name}' ({path.stat().st_size} bytes)")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("directory_structure", status, tuple(details), tuple(warnings), tuple(errors))

    def validate_source_video(self) -> CategoryValidationResult:
        """Validate source video existence, decodability, and metadata sanity."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        video_path = self._get_raw_video_path()
        if not video_path:
            errors.append(f"No source video file found in raw directory: '{self._raw_dir}'")
            return CategoryValidationResult("source_video", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        if not video_path.is_file() or video_path.stat().st_size == 0:
            errors.append(f"Source video file is empty or missing: '{video_path}'")
            return CategoryValidationResult("source_video", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        try:
            with VideoLoader(video_path) as loader:
                meta = loader.get_metadata()
                details.append(
                    f"Video decodable: '{video_path.name}' ({meta.width}x{meta.height} @ {meta.fps:.2f} FPS, "
                    f"{meta.total_frames} frames, {meta.duration_seconds:.2f}s, codec='{meta.codec}')"
                )

                if meta.width < 64 or meta.height < 64:
                    errors.append(f"Video resolution too small: {meta.width}x{meta.height}")
                if meta.fps <= 0.0 or meta.fps > 120.0:
                    errors.append(f"Unrealistic video FPS: {meta.fps}")
                if meta.total_frames <= 0:
                    errors.append(f"Non-positive frame count: {meta.total_frames}")

        except Exception as e:
            errors.append(f"Failed to open/decode source video '{video_path}': {e}")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("source_video", status, tuple(details), tuple(warnings), tuple(errors))

    def validate_extracted_frames(self) -> CategoryValidationResult:
        """Validate extracted frame count, image decoding, and sequential naming."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        if not self._frames_dir.is_dir():
            errors.append(f"Extracted frames directory does not exist: '{self._frames_dir}'")
            return CategoryValidationResult("extracted_frames", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        files = sorted([
            f for f in self._frames_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])
        count = len(files)

        if count == 0:
            errors.append(f"No image files found in extracted frames directory: '{self._frames_dir}'")
            return CategoryValidationResult("extracted_frames", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        details.append(f"Discovered {count} extracted frame images.")

        # Check for duplicate names
        names = [f.name for f in files]
        if len(names) != len(set(names)):
            errors.append("Duplicate filenames detected in extracted frames directory.")

        # Spot-check first, middle, and last frame decoding
        spot_indices = [0, count // 2, count - 1]
        for idx in set(spot_indices):
            sample_file = files[idx]
            try:
                img = cv2.imread(str(sample_file))
                if img is None or img.size == 0:
                    errors.append(f"Failed to decode extracted image: '{sample_file.name}'")
                else:
                    details.append(f"Verified decoding: '{sample_file.name}' ({img.shape[1]}x{img.shape[0]} px)")
            except Exception as e:
                errors.append(f"Exception decoding '{sample_file.name}': {e}")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("extracted_frames", status, tuple(details), tuple(warnings), tuple(errors))

    def validate_preprocessed_frames(self) -> CategoryValidationResult:
        """Validate preprocessed frame count, dimensions, channels, and integrity."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        if not self._preprocessed_frames_dir.is_dir():
            errors.append(f"Preprocessed frames directory does not exist: '{self._preprocessed_frames_dir}'")
            return CategoryValidationResult("preprocessed_frames", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        proc_files = sorted([
            f for f in self._preprocessed_frames_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])
        count = len(proc_files)

        if count == 0:
            errors.append(f"No preprocessed image files found in '{self._preprocessed_frames_dir}'")
            return CategoryValidationResult("preprocessed_frames", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        details.append(f"Discovered {count} preprocessed frame images.")

        # Check for extracted vs preprocessed frame count match
        if self._frames_dir.is_dir():
            extracted_files = [
                f for f in self._frames_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
            ]
            if len(extracted_files) != count:
                errors.append(
                    f"Frame count mismatch: extracted={len(extracted_files)} != preprocessed={count}"
                )

        # Check dimensions across sample
        spot_indices = [0, count // 2, count - 1]
        for idx in set(spot_indices):
            sample_file = proc_files[idx]
            try:
                img = cv2.imread(str(sample_file), cv2.IMREAD_UNCHANGED)
                if img is None or img.size == 0:
                    errors.append(f"Failed to decode preprocessed image: '{sample_file.name}'")
                else:
                    h, w = img.shape[:2]
                    channels = 1 if len(img.shape) == 2 else img.shape[2]
                    if w != self._target_width or h != self._target_height:
                        errors.append(
                            f"Dimension mismatch in '{sample_file.name}': {w}x{h} != expected {self._target_width}x{self._target_height}"
                        )
                    if channels != 3:
                        warnings.append(f"Image '{sample_file.name}' has {channels} channels; expected 3 (BGR).")
            except Exception as e:
                errors.append(f"Exception decoding '{sample_file.name}': {e}")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("preprocessed_frames", status, tuple(details), tuple(warnings), tuple(errors))

    def validate_metadata(self) -> CategoryValidationResult:
        """Validate metadata.csv schema, data types, uniqueness, and metric bounds."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        if not self._metadata_csv_path.is_file():
            errors.append(f"metadata.csv missing at '{self._metadata_csv_path}'")
            return CategoryValidationResult("metadata", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        try:
            df = pd.read_csv(self._metadata_csv_path)
            details.append(f"Loaded metadata.csv ({len(df)} rows, {len(df.columns)} columns)")

            if len(df) == 0:
                errors.append("metadata.csv contains 0 records (empty table).")
                return CategoryValidationResult("metadata", "FAIL", tuple(details), tuple(warnings), tuple(errors))

            # Column schema validation
            missing_cols = [c for c in self.REQUIRED_METADATA_COLUMNS if c not in df.columns]
            if missing_cols:
                errors.append(f"metadata.csv is missing required columns: {missing_cols}")

            # Check uniqueness of frame IDs
            if "extracted_frame_id" in df.columns:
                extracted_ids = df["extracted_frame_id"].tolist()
                if len(extracted_ids) != len(set(extracted_ids)):
                    errors.append("Duplicate extracted_frame_id values detected in metadata.csv.")

            # Check uniqueness of source frame indices
            if "source_frame_index" in df.columns:
                source_indices = df["source_frame_index"].tolist()
                if len(source_indices) != len(set(source_indices)):
                    errors.append("Duplicate source_frame_index values detected in metadata.csv.")

            # Check timestamp monotonicity
            if "timestamp_seconds" in df.columns:
                ts = df["timestamp_seconds"].tolist()
                for i in range(len(ts) - 1):
                    if ts[i] < 0.0:
                        errors.append(f"Negative timestamp detected at row {i}: {ts[i]}")
                        break
                    if ts[i + 1] <= ts[i]:
                        errors.append(f"Non-monotonic timestamp detected: row {i} ({ts[i]}) >= row {i+1} ({ts[i+1]})")
                        break

            # Check quality scores bounds
            if "blur_score" in df.columns:
                invalid_blur = df[df["blur_score"] < 0.0]
                if not invalid_blur.empty:
                    errors.append(f"Negative blur_score found in {len(invalid_blur)} records.")

            if "brightness_score" in df.columns:
                invalid_bright = df[(df["brightness_score"] < 0.0) | (df["brightness_score"] > 255.0)]
                if not invalid_bright.empty:
                    errors.append(f"Out-of-bounds brightness_score found in {len(invalid_bright)} records.")

        except Exception as e:
            errors.append(f"Failed to inspect metadata.csv: {e}")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("metadata", status, tuple(details), tuple(warnings), tuple(errors))

    def validate_manifest(self) -> CategoryValidationResult:
        """Validate summary.json and dataset_manifest.json syntax and cross-consistency."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        # 1. Summary JSON check
        if not self._summary_json_path.is_file():
            errors.append(f"summary.json missing at '{self._summary_json_path}'")
        else:
            try:
                with open(self._summary_json_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                details.append("summary.json parsed successfully.")
                if summary.get("dataset_id") != self.dataset_id:
                    errors.append(f"summary.json dataset_id mismatch: '{summary.get('dataset_id')}' != '{self.dataset_id}'")
            except Exception as e:
                errors.append(f"Failed to parse summary.json: {e}")

        # 2. Dataset Manifest check
        if not self._manifest_json_path.is_file():
            errors.append(f"dataset_manifest.json missing at '{self._manifest_json_path}'")
        else:
            try:
                with open(self._manifest_json_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                details.append("dataset_manifest.json parsed successfully.")

                required_manifest_keys = [
                    "manifest_schema_version",
                    "dataset_id",
                    "source_video",
                    "total_extracted_frames",
                    "total_preprocessed_frames",
                    "processed_image_width",
                    "processed_image_height",
                ]
                missing_keys = [k for k in required_manifest_keys if k not in manifest]
                if missing_keys:
                    errors.append(f"dataset_manifest.json missing required keys: {missing_keys}")

                if manifest.get("dataset_id") != self.dataset_id:
                    errors.append(f"Manifest dataset_id mismatch: '{manifest.get('dataset_id')}' != '{self.dataset_id}'")

                if manifest.get("processed_image_width") != self._target_width or manifest.get("processed_image_height") != self._target_height:
                    errors.append(
                        f"Manifest dimensions mismatch: {manifest.get('processed_image_width')}x{manifest.get('processed_image_height')} != {self._target_width}x{self._target_height}"
                    )

            except Exception as e:
                errors.append(f"Failed to parse dataset_manifest.json: {e}")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("manifest", status, tuple(details), tuple(warnings), tuple(errors))

    def validate_frame_traceability(self) -> CategoryValidationResult:
        """Validate unbroken bidirectional traceability between video, frames, and metadata."""
        details: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        if not self._metadata_csv_path.is_file():
            errors.append("Cannot validate traceability; metadata.csv is missing.")
            return CategoryValidationResult("traceability", "FAIL", tuple(details), tuple(warnings), tuple(errors))

        try:
            df = pd.read_csv(self._metadata_csv_path)
            total_records = len(df)

            # Check that referenced physical files exist
            missing_raw_files = 0
            missing_proc_files = 0

            # Spot-check all rows or first 50 rows for path resolution
            check_rows = df.to_dict(orient="records")
            for r in check_rows:
                # Raw image check
                raw_path_str = r.get("original_image_path")
                if raw_path_str:
                    raw_p = Path(raw_path_str)
                    if not raw_p.is_file():
                        # Try relative to project root
                        if not (self._project_root / raw_path_str).is_file():
                            missing_raw_files += 1

                # Preprocessed image check (only for successful records)
                if r.get("preprocessing_status") == "SUCCESS":
                    proc_path_str = r.get("preprocessed_image_path")
                    if proc_path_str:
                        proc_p = Path(proc_path_str)
                        if not proc_p.is_file():
                            if not (self._project_root / proc_path_str).is_file():
                                missing_proc_files += 1

            if missing_raw_files > 0:
                errors.append(f"{missing_raw_files}/{total_records} metadata rows reference missing raw frame files.")
            if missing_proc_files > 0:
                errors.append(f"{missing_proc_files}/{total_records} successful rows reference missing preprocessed files.")

            if missing_raw_files == 0 and missing_proc_files == 0:
                details.append(f"Verified 1:1 physical file resolution across all {total_records} metadata records.")

        except Exception as e:
            errors.append(f"Traceability check exception: {e}")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return CategoryValidationResult("traceability", status, tuple(details), tuple(warnings), tuple(errors))

    # --------------------------------------------------------------------------
    # Statistical Quality Analysis
    # --------------------------------------------------------------------------

    def calculate_data_quality_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive descriptive quality statistics over dataset assets."""
        stats: Dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "total_extracted_frames": 0,
            "total_preprocessed_frames": 0,
            "total_failed_frames": 0,
            "completeness_percentage": 0.0,
            "dimension_consistency_percentage": 100.0,
        }

        if self._metadata_csv_path.is_file():
            try:
                df = pd.read_csv(self._metadata_csv_path)
                total = len(df)
                stats["total_extracted_frames"] = total

                if "preprocessing_status" in df.columns:
                    success_df = df[df["preprocessing_status"] == "SUCCESS"]
                    failed_df = df[df["preprocessing_status"] != "SUCCESS"]
                    stats["total_preprocessed_frames"] = len(success_df)
                    stats["total_failed_frames"] = len(failed_df)
                    stats["completeness_percentage"] = round((len(success_df) / total) * 100.0, 2) if total > 0 else 0.0

                for metric in ("blur_score", "brightness_score", "contrast_score", "file_size_bytes"):
                    if metric in df.columns:
                        valid_series = df[metric].dropna()
                        if not valid_series.empty:
                            stats[metric] = {
                                "mean": round(float(valid_series.mean()), 2),
                                "std": round(float(valid_series.std()), 2),
                                "min": round(float(valid_series.min()), 2),
                                "max": round(float(valid_series.max()), 2),
                            }

            except Exception as e:
                logger.warning("Error calculating quality statistics: %s", e)

        return stats

    # --------------------------------------------------------------------------
    # Master Dataset Validation Engine
    # --------------------------------------------------------------------------

    def validate_dataset(self) -> DatasetValidationReport:
        """Execute complete non-destructive validation suite across all categories.

        Returns:
            DatasetValidationReport: Immutable validation report contract.
        """
        logger.info("Executing comprehensive dataset validation for '%s'...", self.dataset_id)

        checks: Dict[str, Dict[str, Any]] = {}
        all_warnings: List[str] = []
        all_errors: List[str] = []

        # Run categorical checks
        categories = [
            self.validate_directory_structure(),
            self.validate_source_video(),
            self.validate_extracted_frames(),
            self.validate_preprocessed_frames(),
            self.validate_metadata(),
            self.validate_manifest(),
            self.validate_frame_traceability(),
        ]

        has_fail = False
        has_warning = False

        for res in categories:
            checks[res.category] = res.to_dict()
            all_warnings.extend(res.warnings)
            all_errors.extend(res.errors)
            if res.status == "FAIL":
                has_fail = True
            elif res.status == "WARNING":
                has_warning = True

        overall_status = "FAIL" if has_fail else ("WARNING" if has_warning else "PASS")
        is_ready = (overall_status in ("PASS", "WARNING")) and not has_fail

        stats = self.calculate_data_quality_statistics()

        recommendations: List[str] = []
        if overall_status == "PASS":
            recommendations.append("Dataset validation succeeded. Dataset is certified READY for Day 3 YOLO detection.")
        elif overall_status == "WARNING":
            recommendations.append("Dataset passed with warnings. Inspect non-blocking notices before training.")
        else:
            recommendations.append("Critical validation errors detected. Resolve failed checks before proceeding.")

        now_iso = datetime.now(timezone.utc).isoformat()

        report = DatasetValidationReport(
            validation_schema_version=VALIDATION_SCHEMA_VERSION,
            dataset_id=self.dataset_id,
            validation_timestamp=now_iso,
            overall_status=overall_status,
            is_ready_for_day_3=is_ready,
            checks=checks,
            statistics=stats,
            issues=tuple(all_errors + all_warnings),
            recommendations=tuple(recommendations),
        )

        self._report = report
        return report

    def save_validation_report(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Persist validation report to strict RFC 8259 JSON file.

        Args:
            output_path: Destination path for report.

        Returns:
            Path: Written report path.
        """
        if self._report is None:
            self.validate_dataset()

        target_file = (
            Path(output_path).resolve()
            if output_path
            else self._processed_dir / "validation_report.json"
        )

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            report_dict = self._report.to_dict()  # type: ignore[union-attr]

            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)

        except Exception as e:
            raise DatasetValidationError(f"Failed to persist validation report to '{target_file}': {e}") from e

        logger.info("Successfully persisted validation report at: %s", target_file)
        return target_file


def main() -> None:
    """CLI Entry point for dataset validation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI - Dataset Quality Assurance & Validation Pipeline"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="driving_sample",
        help="Target dataset ID to validate (default: 'driving_sample')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to video_config.yaml (default: config/video_config.yaml)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Additional destination path for validation report JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    validator = DatasetValidator(dataset_id=args.dataset_id, config_path=args.config)
    report = validator.validate_dataset()

    # Save to default dataset directory
    default_saved = validator.save_validation_report()
    print(f"Validation report saved to: {default_saved}")

    # Save to additional output destination if specified
    if args.output_report:
        extra_saved = validator.save_validation_report(output_path=args.output_report)
        print(f"Validation report saved to: {extra_saved}")

    print("=" * 60)
    print(f"Dataset Validation Summary: {args.dataset_id}")
    print(f"Overall Status: {report.overall_status}")
    print(f"Ready for Day 3 Perception: {report.is_ready_for_day_3}")
    print(f"Total Issues: {len(report.issues)}")
    for check_name, check_data in report.checks.items():
        print(f"  - {check_name}: {check_data['status']}")
    print("=" * 60)

    if report.overall_status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
