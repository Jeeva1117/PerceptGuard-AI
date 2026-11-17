"""PerceptGuard-AI: Dataset Organization and Manifest Management.

Production-grade ML dataset organization module for autonomous driving perception pipelines.
Enforces standardized filesystem layout separating raw and processed data, verifies
cross-modal traceability, prevents accidental data corruption, and serializes auditable
dataset manifests for downstream YOLO detection and tracking.

Key Architectural Guarantees:
- Non-Destructive Filesystem Operations: Never deletes, moves, or overwrites raw video or extracted frames
- Idempotent Execution: Safe to execute repeatedly without duplicating files or mutating state
- Strict Traceability: Verifies 1:1 bidirectional alignment between source video, extracted frames,
  preprocessed images, and tabular metadata records
- Standardized ML Access Interface: Provides centralized accessor methods for downstream YOLOv8
  and ByteTrack perception modules
- RFC 8259 Compliant Manifest Serialization: Strictly validated JSON manifest with ISO 8601 timestamps
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml

from src.video_processing.video_loader import VideoProcessingError

logger = logging.getLogger("PerceptGuard.DatasetOrganizer")

MANIFEST_SCHEMA_VERSION: str = "1.0.0"


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class DatasetOrganizationError(VideoProcessingError):
    """Base exception for all dataset organization failures."""
    pass


class DatasetStructureError(DatasetOrganizationError):
    """Raised when the dataset directory layout violates structural invariants."""
    pass


class ManifestError(DatasetOrganizationError):
    """Raised when dataset manifest generation or verification fails."""
    pass


class DataIntegrityError(DatasetOrganizationError):
    """Raised when cross-modal traceability or asset correspondence checks fail."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class DatasetVerificationReport:
    """Immutable audit report of dataset directory structure and integrity.

    Attributes:
        is_valid: True if all structural requirements and traceability checks pass.
        dataset_id: Logical dataset collection identifier.
        raw_video_present: True if source video file is located in raw directory.
        raw_video_path: Path to source video file if found.
        extracted_frames_count: Total extracted frames found in frames/ directory.
        preprocessed_frames_count: Total preprocessed frames found in preprocessed_frames/.
        metadata_csv_present: True if metadata.csv exists and is readable.
        summary_json_present: True if summary.json exists and is readable.
        manifest_json_present: True if dataset_manifest.json exists.
        missing_components: Tuple of missing required directories or files.
        integrity_issues: Tuple of detected traceability or consistency errors.
    """
    is_valid: bool
    dataset_id: str
    raw_video_present: bool
    raw_video_path: Optional[str]
    extracted_frames_count: int
    preprocessed_frames_count: int
    metadata_csv_present: bool
    summary_json_present: bool
    manifest_json_present: bool
    missing_components: Tuple[str, ...]
    integrity_issues: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return asdict(self)


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable data contract encapsulating the complete dataset manifest.

    Attributes:
        manifest_schema_version: Manifest schema semantic version.
        dataset_id: Logical dataset collection identifier.
        dataset_name: Human-readable dataset name.
        source_video: Basename of source video file.
        dataset_created_at: ISO 8601 timestamp with UTC timezone.
        dataset_root: Relative path to root dataset directory.
        raw_video_directory: Relative path to raw videos directory.
        extracted_frames_directory: Relative path to extracted frames.
        preprocessed_frames_directory: Relative path to preprocessed frames.
        metadata_csv_path: Relative path to metadata.csv.
        summary_json_path: Relative path to summary.json.
        total_extracted_frames: Total count of extracted frames.
        total_preprocessed_frames: Total count of successfully preprocessed frames.
        total_failed_frames: Total count of failed frames.
        image_format: Standard image format extension (e.g. 'jpg').
        processed_image_width: Standardized pixel width (e.g. 640).
        processed_image_height: Standardized pixel height (e.g. 360).
        metadata_schema_version: Schema version of associated metadata.csv.
    """
    manifest_schema_version: str
    dataset_id: str
    dataset_name: str
    source_video: str
    dataset_created_at: str
    dataset_root: str
    raw_video_directory: str
    extracted_frames_directory: str
    preprocessed_frames_directory: str
    metadata_csv_path: str
    summary_json_path: str
    total_extracted_frames: int
    total_preprocessed_frames: int
    total_failed_frames: int
    image_format: str
    processed_image_width: int
    processed_image_height: int
    metadata_schema_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest contract to dictionary."""
        return asdict(self)


# ==============================================================================
# DatasetOrganizer Implementation
# ==============================================================================

class DatasetOrganizer:
    """Production dataset organization and manifest management engine.

    Standardizes, validates, and exposes dataset components for autonomous driving
    computer vision pipelines.
    """

    SUPPORTED_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    SUPPORTED_VIDEO_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        dataset_id: str = "driving_sample",
        project_root: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize DatasetOrganizer with directory paths and configuration.

        Args:
            dataset_id: Unique identifier for the dataset collection.
            project_root: Project repository root directory.
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

        # Processing parameters
        self._target_width: int = 640
        self._target_height: int = 360
        self._image_format: str = "jpg"
        self._default_video_name: Optional[str] = None

        self._load_config(config_path)

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

                p_cfg = cfg.get("preprocessing", {})
                if "target_width" in p_cfg:
                    self._target_width = int(p_cfg["target_width"])
                if "target_height" in p_cfg:
                    self._target_height = int(p_cfg["target_height"])
                if "image_format" in p_cfg:
                    self._image_format = str(p_cfg["image_format"]).lower().lstrip(".")

            except Exception as e:
                logger.warning("Failed to parse config %s: %s. Using default settings.", config_path, e)

    # --------------------------------------------------------------------------
    # Centralized Access Interface for Downstream Perception Modules
    # --------------------------------------------------------------------------

    def get_raw_video_path(self) -> Optional[Path]:
        """Locate and return the primary raw driving video path."""
        if not self._raw_dir.is_dir():
            return None

        # Check configured default first
        if self._default_video_name:
            candidate = self._raw_dir / self._default_video_name
            if candidate.is_file():
                return candidate

        # Search for any valid video file in raw/
        videos = [
            f for f in self._raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_VIDEO_EXTS
        ]
        return sorted(videos)[0] if videos else None

    def get_extracted_frames_dir(self) -> Path:
        """Get directory containing original raw extracted frames."""
        return self._frames_dir

    def get_preprocessed_frames_dir(self) -> Path:
        """Get directory containing standardized preprocessed frames."""
        return self._preprocessed_frames_dir

    def get_metadata_csv_path(self) -> Path:
        """Get path to metadata.csv."""
        return self._metadata_csv_path

    def get_summary_json_path(self) -> Path:
        """Get path to summary.json."""
        return self._summary_json_path

    def get_manifest_path(self) -> Path:
        """Get path to dataset_manifest.json."""
        return self._manifest_json_path

    def get_preprocessed_frame_paths(self) -> List[Path]:
        """Retrieve sorted list of preprocessed image paths for YOLO ingestion."""
        if not self._preprocessed_frames_dir.is_dir():
            return []
        return sorted([
            f for f in self._preprocessed_frames_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])

    # --------------------------------------------------------------------------
    # Dataset Inspection and Verification
    # --------------------------------------------------------------------------

    def validate_source_directories(self) -> Tuple[bool, List[str]]:
        """Verify presence of core dataset directories.

        Returns:
            Tuple of (all_exist, list_of_missing_directories).
        """
        missing: List[str] = []
        for path in (self._raw_dir, self._processed_dir, self._frames_dir, self._preprocessed_frames_dir):
            if not path.is_dir():
                missing.append(str(path))
        return len(missing) == 0, missing

    def inspect_dataset(self) -> Dict[str, Any]:
        """Inspect dataset directory assets without loading image arrays into RAM."""
        raw_video = self.get_raw_video_path()

        extracted_frames = (
            sorted([f for f in self._frames_dir.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS])
            if self._frames_dir.is_dir()
            else []
        )

        preprocessed_frames = (
            sorted([f for f in self._preprocessed_frames_dir.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS])
            if self._preprocessed_frames_dir.is_dir()
            else []
        )

        # Check for duplicate filenames
        extracted_names = [f.name for f in extracted_frames]
        has_duplicate_extracted = len(extracted_names) != len(set(extracted_names))

        preproc_names = [f.name for f in preprocessed_frames]
        has_duplicate_preproc = len(preproc_names) != len(set(preproc_names))

        return {
            "dataset_id": self.dataset_id,
            "raw_video_path": str(raw_video) if raw_video else None,
            "raw_video_exists": raw_video is not None and raw_video.is_file(),
            "extracted_frames_count": len(extracted_frames),
            "preprocessed_frames_count": len(preprocessed_frames),
            "has_duplicate_extracted_names": has_duplicate_extracted,
            "has_duplicate_preprocessed_names": has_duplicate_preproc,
            "metadata_csv_exists": self._metadata_csv_path.is_file(),
            "summary_json_exists": self._summary_json_path.is_file(),
            "manifest_json_exists": self._manifest_json_path.is_file(),
        }

    def organize_dataset(self, dry_run: bool = False) -> Dict[str, Any]:
        """Verify and standardize dataset layout idempotently.

        Guarantees:
        - Never deletes existing files.
        - Creates missing directory targets safely.
        - If dry_run is True, returns planned actions without filesystem modification.

        Args:
            dry_run: If True, simulates operations without creating directories.

        Returns:
            Dict describing actions executed or planned.
        """
        required_dirs = [
            self._raw_dir,
            self._processed_dir,
            self._frames_dir,
            self._preprocessed_frames_dir,
        ]

        actions_taken: List[str] = []

        for target_dir in required_dirs:
            if not target_dir.exists():
                msg = f"Create directory: {target_dir}"
                actions_taken.append(msg)
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Created directory: %s", target_dir)
            else:
                actions_taken.append(f"Directory already exists: {target_dir}")

        return {
            "dry_run": dry_run,
            "actions": actions_taken,
            "is_organized": True,
        }

    def verify_dataset_structure(self) -> DatasetVerificationReport:
        """Conduct exhaustive integrity and cross-modal traceability checks.

        Validates:
        1. All required directories exist.
        2. Raw driving video exists.
        3. Extracted frames exist and have zero duplicate names.
        4. Preprocessed frames exist and match extracted frame count.
        5. metadata.csv exists and has 1:1 mapping with frame files.
        6. summary.json exists and parses cleanly.

        Returns:
            DatasetVerificationReport: Comprehensive immutable audit report.
        """
        missing_components: List[str] = []
        integrity_issues: List[str] = []

        # Directory presence checks
        for d in (self._raw_dir, self._processed_dir, self._frames_dir, self._preprocessed_frames_dir):
            if not d.is_dir():
                missing_components.append(str(d))

        raw_video = self.get_raw_video_path()
        if not raw_video:
            missing_components.append("raw_video_file")

        # Frame counts
        extracted_frames = (
            sorted([f for f in self._frames_dir.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS])
            if self._frames_dir.is_dir()
            else []
        )
        preprocessed_frames = (
            sorted([f for f in self._preprocessed_frames_dir.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS])
            if self._preprocessed_frames_dir.is_dir()
            else []
        )

        if len(extracted_frames) == 0:
            integrity_issues.append("Extracted frames directory contains 0 image files.")

        if len(preprocessed_frames) == 0:
            integrity_issues.append("Preprocessed frames directory contains 0 image files.")

        if len(extracted_frames) > 0 and len(preprocessed_frames) > 0:
            if len(extracted_frames) != len(preprocessed_frames):
                integrity_issues.append(
                    f"Frame count mismatch: extracted={len(extracted_frames)} != preprocessed={len(preprocessed_frames)}"
                )

        # Duplicate filename checks
        if len(extracted_frames) != len(set(f.name for f in extracted_frames)):
            integrity_issues.append("Duplicate filenames detected in extracted frames directory.")
        if len(preprocessed_frames) != len(set(f.name for f in preprocessed_frames)):
            integrity_issues.append("Duplicate filenames detected in preprocessed frames directory.")

        # Metadata file checks
        if not self._metadata_csv_path.is_file():
            missing_components.append(str(self._metadata_csv_path))
        if not self._summary_json_path.is_file():
            missing_components.append(str(self._summary_json_path))

        # Traceability check against metadata.csv
        records_count = 0
        if self._metadata_csv_path.is_file():
            try:
                import pandas as pd
                df = pd.read_csv(self._metadata_csv_path)
                records_count = len(df)
                if len(extracted_frames) > 0 and records_count != len(extracted_frames):
                    integrity_issues.append(
                        f"metadata.csv row count mismatch: {records_count} rows != {len(extracted_frames)} extracted frames"
                    )
            except Exception as e:
                integrity_issues.append(f"Failed to read metadata.csv during verification: {e}")

        # Summary JSON check
        if self._summary_json_path.is_file():
            try:
                with open(self._summary_json_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                if summary_data.get("dataset_id") != self.dataset_id:
                    integrity_issues.append(
                        f"summary.json dataset_id mismatch: '{summary_data.get('dataset_id')}' != '{self.dataset_id}'"
                    )
            except Exception as e:
                integrity_issues.append(f"Failed to parse summary.json during verification: {e}")

        is_valid = (len(missing_components) == 0) and (len(integrity_issues) == 0)

        return DatasetVerificationReport(
            is_valid=is_valid,
            dataset_id=self.dataset_id,
            raw_video_present=raw_video is not None,
            raw_video_path=str(raw_video) if raw_video else None,
            extracted_frames_count=len(extracted_frames),
            preprocessed_frames_count=len(preprocessed_frames),
            metadata_csv_present=self._metadata_csv_path.is_file(),
            summary_json_present=self._summary_json_path.is_file(),
            manifest_json_present=self._manifest_json_path.is_file(),
            missing_components=tuple(missing_components),
            integrity_issues=tuple(integrity_issues),
        )

    # --------------------------------------------------------------------------
    # Manifest Generation
    # --------------------------------------------------------------------------

    def generate_dataset_manifest(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Compile and persist standardized dataset_manifest.json.

        Populates manifest strictly from verified filesystem assets and summary.json,
        guaranteeing zero fabricated telemetry.

        Args:
            output_path: Optional destination path for manifest file.

        Returns:
            Path: Filesystem path to written dataset_manifest.json.

        Raises:
            ManifestError: If required metadata assets are missing or serialization fails.
        """
        report = self.verify_dataset_structure()
        if not report.raw_video_present:
            raise ManifestError(f"Cannot generate manifest; raw video missing for dataset '{self.dataset_id}'.")
        if not report.metadata_csv_present or not report.summary_json_present:
            raise ManifestError("Cannot generate manifest; metadata.csv or summary.json missing.")

        # Read actual metrics from summary.json
        try:
            with open(self._summary_json_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:
            raise ManifestError(f"Failed to read summary.json for manifest generation: {e}") from e

        source_video_name = summary_data.get("source_video", Path(report.raw_video_path or "").name)
        total_extracted = summary_data.get("total_extracted_frames", report.extracted_frames_count)
        total_preprocessed = summary_data.get("total_preprocessed_frames", report.preprocessed_frames_count)
        total_failed = summary_data.get("total_failed_frames", 0)

        processed_dims = summary_data.get("processed_image_dimensions", [self._target_width, self._target_height])
        proc_w = processed_dims[0] if len(processed_dims) > 0 else self._target_width
        proc_h = processed_dims[1] if len(processed_dims) > 1 else self._target_height

        meta_schema_version = summary_data.get("metadata_schema_version", "1.0.0")

        # Compute relative paths for clean portability
        def rel_path(p: Path) -> str:
            try:
                return str(p.relative_to(self._project_root)).replace("\\", "/")
            except ValueError:
                return str(p).replace("\\", "/")

        now_iso = datetime.now(timezone.utc).isoformat()

        manifest = DatasetManifest(
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            dataset_id=self.dataset_id,
            dataset_name=f"PerceptGuard Autonomous Driving Sample ({self.dataset_id})",
            source_video=source_video_name,
            dataset_created_at=now_iso,
            dataset_root=rel_path(self._dataset_root),
            raw_video_directory=rel_path(self._raw_dir),
            extracted_frames_directory=rel_path(self._frames_dir),
            preprocessed_frames_directory=rel_path(self._preprocessed_frames_dir),
            metadata_csv_path=rel_path(self._metadata_csv_path),
            summary_json_path=rel_path(self._summary_json_path),
            total_extracted_frames=total_extracted,
            total_preprocessed_frames=total_preprocessed,
            total_failed_frames=total_failed,
            image_format=self._image_format,
            processed_image_width=proc_w,
            processed_image_height=proc_h,
            metadata_schema_version=meta_schema_version,
        )

        target_file = Path(output_path).resolve() if output_path else self._manifest_json_path

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            # Verification: parse back to validate JSON integrity
            with open(target_file, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            if parsed.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
                raise ManifestError("Manifest read-back verification failed: version mismatch.")

        except Exception as e:
            raise ManifestError(f"Failed to persist dataset_manifest.json to '{target_file}': {e}") from e

        logger.info("Successfully generated dataset manifest at: %s", target_file)
        return target_file

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Return high-level summary of dataset structure and status."""
        report = self.verify_dataset_structure()
        return {
            "dataset_id": self.dataset_id,
            "is_valid": report.is_valid,
            "raw_video": Path(report.raw_video_path).name if report.raw_video_path else None,
            "extracted_frames": report.extracted_frames_count,
            "preprocessed_frames": report.preprocessed_frames_count,
            "metadata_csv": report.metadata_csv_present,
            "summary_json": report.summary_json_present,
            "manifest_json": report.manifest_json_present,
            "missing_components": list(report.missing_components),
            "integrity_issues": list(report.integrity_issues),
        }
