"""PerceptGuard AI: Feature Schema & Validation Specification.

Defines the strongly typed, versioned schema contract for multidimensional feature vectors
combining object detection, motion kinematics, visual quality, and tracking continuity.

Schema Version: 1.0

Feature Groups:
1. Detection Features:
   - confidence: Detection confidence score [0.0, 1.0].
   - class_id: Integer semantic class identifier (COCO).
   - class_name: String label of semantic category.
   - bbox_x1, bbox_y1, bbox_x2, bbox_y2: Spatial pixel boundaries.
   - bbox_width, bbox_height, bbox_area: Geometric dimensions.
   - frame_detection_count: Total detections in host frame.
2. Motion Features:
   - center_x, center_y: Centroid coordinates.
   - displacement_pixels: Euclidean distance from previous observation.
   - velocity_pixels_per_second: Calibrated pixel speed.
   - movement_direction_radians: Direction angle [-pi, pi].
   - movement_direction_degrees: Direction angle [0, 360).
3. Visual Quality Features:
   - blur_score: Laplacian variance blur metric.
   - brightness: Mean intensity in [0, 255].
   - contrast: RMS intensity standard deviation.
4. Tracking Features:
   - track_id: Persistent identity assigned by tracker.
   - track_observation_count: Total observed frames for tracklet.
   - is_visible: Boolean visibility flag in host frame.
   - observation_ratio: Ratio of observed frames to track lifespan span.
   - consecutive_lost_frames: Number of consecutive missing frames.
   - reacquisition_count: Count of times reacquired after gap.
   - tracking_state: State machine status (NEW, TRACKING, TEMPORARILY_LOST, REACQUIRED, TERMINATED).
5. Metadata & Alignment:
   - frame_id: Unique string frame identifier (e.g., frame_000001.jpg).
   - source_frame_index: Chronological frame sequence integer.
   - timestamp_seconds: Frame capture timestamp if available.
   - dataset_id: Origin dataset identifier.
"""

from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger("PerceptGuard.FeatureSchema")

FEATURE_SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1", 1, 1.0}

VALID_TRACKING_STATES = {
    "NEW",
    "INITIALIZED",
    "TRACKING",
    "TEMPORARILY_LOST",
    "REACQUIRED",
    "TERMINATED",
}


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class FeatureSchemaError(Exception):
    """Base exception for all feature schema and validation errors."""
    pass


class InvalidFeatureRecordError(FeatureSchemaError):
    """Raised when an individual feature record violates type or range constraints."""
    pass


class MissingRequiredFieldError(FeatureSchemaError):
    """Raised when a required schema field is missing or null."""
    pass


class IncompatibleArtifactError(FeatureSchemaError):
    """Raised when input artifacts (detection, tracking, metadata) fail integration checks."""
    pass


# ==============================================================================
# Field Descriptor & Schema Registry
# ==============================================================================

@dataclass(frozen=True)
class FeatureFieldDescriptor:
    """Metadata specification for a single feature field in the schema."""
    name: str
    data_type: str
    description: str
    feature_group: str
    is_required: bool
    is_nullable: bool
    default_value: Optional[Any] = None


FEATURE_FIELD_REGISTRY: List[FeatureFieldDescriptor] = [
    # Metadata & Identifiers
    FeatureFieldDescriptor("frame_id", "str", "Unique frame identifier filename", "metadata", True, False),
    FeatureFieldDescriptor("source_frame_index", "int", "Chronological frame index integer", "metadata", True, False),
    FeatureFieldDescriptor("timestamp_seconds", "float", "Timestamp of frame in seconds", "metadata", False, True, None),
    FeatureFieldDescriptor("dataset_id", "str", "Origin dataset name", "metadata", True, False),

    # Detection Features
    FeatureFieldDescriptor("confidence", "float", "Detection confidence score in [0.0, 1.0]", "detection", True, False),
    FeatureFieldDescriptor("class_id", "int", "COCO integer class identifier", "detection", True, False),
    FeatureFieldDescriptor("class_name", "str", "Human readable category label", "detection", True, False),
    FeatureFieldDescriptor("bbox_x1", "float", "Bounding box minimum x coordinate", "detection", True, False),
    FeatureFieldDescriptor("bbox_y1", "float", "Bounding box minimum y coordinate", "detection", True, False),
    FeatureFieldDescriptor("bbox_x2", "float", "Bounding box maximum x coordinate", "detection", True, False),
    FeatureFieldDescriptor("bbox_y2", "float", "Bounding box maximum y coordinate", "detection", True, False),
    FeatureFieldDescriptor("bbox_width", "float", "Bounding box width in pixels", "detection", True, False),
    FeatureFieldDescriptor("bbox_height", "float", "Bounding box height in pixels", "detection", True, False),
    FeatureFieldDescriptor("bbox_area", "float", "Bounding box area in square pixels", "detection", True, False),
    FeatureFieldDescriptor("frame_detection_count", "int", "Total object count detected in host frame", "detection", True, False),

    # Motion Features
    FeatureFieldDescriptor("center_x", "float", "Object centroid x coordinate", "motion", True, False),
    FeatureFieldDescriptor("center_y", "float", "Object centroid y coordinate", "motion", True, False),
    FeatureFieldDescriptor("displacement_pixels", "float", "Inter-frame Euclidean centroid shift", "motion", False, True, None),
    FeatureFieldDescriptor("velocity_pixels_per_second", "float", "Speed in pixels per second", "motion", False, True, None),
    FeatureFieldDescriptor("movement_direction_radians", "float", "Direction angle in radians [-pi, pi]", "motion", False, True, None),
    FeatureFieldDescriptor("movement_direction_degrees", "float", "Direction angle in degrees [0, 360)", "motion", False, True, None),

    # Visual Quality Features
    FeatureFieldDescriptor("blur_score", "float", "Laplacian variance blur metric", "visual_quality", False, True, None),
    FeatureFieldDescriptor("brightness", "float", "Mean pixel intensity in [0, 255]", "visual_quality", False, True, None),
    FeatureFieldDescriptor("contrast", "float", "RMS intensity standard deviation", "visual_quality", False, True, None),

    # Tracking Features
    FeatureFieldDescriptor("track_id", "int", "Persistent unique identity assigned by tracker", "tracking", True, False),
    FeatureFieldDescriptor("track_observation_count", "int", "Cumulative frames object was observed", "tracking", True, False),
    FeatureFieldDescriptor("is_visible", "bool", "True if detected/observed in host frame", "tracking", True, False, True),
    FeatureFieldDescriptor("observation_ratio", "float", "Ratio of observations over temporal lifespan span", "tracking", True, False, 1.0),
    FeatureFieldDescriptor("consecutive_lost_frames", "int", "Count of consecutive frames missing", "tracking", True, False, 0),
    FeatureFieldDescriptor("reacquisition_count", "int", "Count of reacquisitions after loss", "tracking", True, False, 0),
    FeatureFieldDescriptor("tracking_state", "str", "Lifecycle state machine status", "tracking", True, False, "NEW"),
]


class FeatureSchemaDefinition:
    """Registry and query interface for PerceptGuard AI Feature Schema v1.0."""

    SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

    @classmethod
    def get_all_fields(cls) -> List[FeatureFieldDescriptor]:
        return list(FEATURE_FIELD_REGISTRY)

    @classmethod
    def get_field_names(cls) -> List[str]:
        return [f.name for f in FEATURE_FIELD_REGISTRY]

    @classmethod
    def get_required_fields(cls) -> List[FeatureFieldDescriptor]:
        return [f for f in FEATURE_FIELD_REGISTRY if f.is_required]

    @classmethod
    def get_required_field_names(cls) -> List[str]:
        return [f.name for f in FEATURE_FIELD_REGISTRY if f.is_required]

    @classmethod
    def get_optional_field_names(cls) -> List[str]:
        return [f.name for f in FEATURE_FIELD_REGISTRY if not f.is_required]

    @classmethod
    def get_fields_by_group(cls, group_name: str) -> List[FeatureFieldDescriptor]:
        return [f for f in FEATURE_FIELD_REGISTRY if f.feature_group == group_name]

    @classmethod
    def get_descriptor(cls, field_name: str) -> Optional[FeatureFieldDescriptor]:
        for f in FEATURE_FIELD_REGISTRY:
            if f.name == field_name:
                return f
        return None


# ==============================================================================
# Feature Record Contract
# ==============================================================================

@dataclass
class FeatureRecord:
    """Strongly typed feature representation for an individual object observation."""

    # Metadata & Identifiers
    frame_id: str
    source_frame_index: int
    dataset_id: str
    timestamp_seconds: Optional[float] = None

    # Detection Features
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "object"
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    bbox_area: float = 0.0
    frame_detection_count: int = 1

    # Motion Features
    center_x: float = 0.0
    center_y: float = 0.0
    displacement_pixels: Optional[float] = None
    velocity_pixels_per_second: Optional[float] = None
    movement_direction_radians: Optional[float] = None
    movement_direction_degrees: Optional[float] = None

    # Visual Quality Features
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None

    # Tracking Features
    track_id: int = 1
    track_observation_count: int = 1
    is_visible: bool = True
    observation_ratio: float = 1.0
    consecutive_lost_frames: int = 0
    reacquisition_count: int = 0
    tracking_state: str = "NEW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to Python dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Container representing outcome of schema or artifact validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise an appropriate domain error if validation failed."""
        if not self.is_valid:
            msg = "Validation failed with errors:\n  - " + "\n  - ".join(self.errors)
            raise InvalidFeatureRecordError(msg)


# ==============================================================================
# Validation Engines
# ==============================================================================

def validate_feature_record(
    record: Union[Dict[str, Any], FeatureRecord],
    strict: bool = True,
) -> ValidationResult:
    """Validate a single feature record against Feature Schema v1.0 constraints.

    Args:
        record: Dictionary or FeatureRecord instance to evaluate.
        strict: If True, treats missing optional fields or slight deviations as warnings.

    Returns:
        ValidationResult indicating validity, error details, and warnings.
    """
    if isinstance(record, FeatureRecord):
        rec_dict = record.to_dict()
    elif isinstance(record, dict):
        rec_dict = record
    else:
        return ValidationResult(
            is_valid=False,
            errors=[f"Expected dict or FeatureRecord, got {type(record).__name__}"]
        )

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Required Fields Check
    for descriptor in FeatureSchemaDefinition.get_required_fields():
        field_name = descriptor.name
        if field_name not in rec_dict:
            errors.append(f"Missing required field: '{field_name}'")
        elif rec_dict[field_name] is None:
            errors.append(f"Required field '{field_name}' cannot be null/None")

    # If critical required fields are missing, return immediately
    if errors:
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 2. Confidence Validation
    conf = rec_dict.get("confidence")
    if conf is not None:
        if not isinstance(conf, (int, float)) or math.isnan(conf) or math.isinf(conf):
            errors.append(f"confidence must be a finite float, got {conf}")
        elif not (0.0 <= conf <= 1.0):
            errors.append(f"confidence must be in [0.0, 1.0], got {conf}")

    # 3. Bounding Box Geometry Validation
    try:
        x1 = float(rec_dict["bbox_x1"])
        y1 = float(rec_dict["bbox_y1"])
        x2 = float(rec_dict["bbox_x2"])
        y2 = float(rec_dict["bbox_y2"])

        if any(math.isnan(v) or math.isinf(v) for v in [x1, y1, x2, y2]):
            errors.append(f"Bounding box coordinates contain non-finite values: [{x1}, {y1}, {x2}, {y2}]")
        elif x2 <= x1:
            errors.append(f"Invalid bounding box width (x2 <= x1): x1={x1}, x2={x2}")
        elif y2 <= y1:
            errors.append(f"Invalid bounding box height (y2 <= y1): y1={y1}, y2={y2}")
    except (KeyError, ValueError, TypeError) as ex:
        errors.append(f"Failed to parse bounding box coordinates: {ex}")

    # 4. Dimension & Area Consistency
    w = rec_dict.get("bbox_width")
    h = rec_dict.get("bbox_height")
    area = rec_dict.get("bbox_area")
    if w is not None and w <= 0.0:
        errors.append(f"bbox_width must be positive, got {w}")
    if h is not None and h <= 0.0:
        errors.append(f"bbox_height must be positive, got {h}")
    if area is not None and area <= 0.0:
        errors.append(f"bbox_area must be positive, got {area}")

    # 5. Tracking State & Identification Validation
    tid = rec_dict.get("track_id")
    if tid is not None:
        if not isinstance(tid, int) or tid < 1:
            errors.append(f"track_id must be an integer >= 1, got {tid}")

    t_state = rec_dict.get("tracking_state")
    if t_state is not None and t_state not in VALID_TRACKING_STATES:
        errors.append(f"Invalid tracking_state '{t_state}'. Must be one of {VALID_TRACKING_STATES}")

    obs_ratio = rec_dict.get("observation_ratio")
    if obs_ratio is not None:
        if not isinstance(obs_ratio, (int, float)) or not (0.0 <= obs_ratio <= 1.0):
            errors.append(f"observation_ratio must be in [0.0, 1.0], got {obs_ratio}")

    # 6. Motion Kinematics Range Checks
    disp = rec_dict.get("displacement_pixels")
    if disp is not None:
        if not isinstance(disp, (int, float)) or disp < 0.0 or math.isnan(disp):
            errors.append(f"displacement_pixels must be a non-negative float, got {disp}")

    vel = rec_dict.get("velocity_pixels_per_second")
    if vel is not None:
        if not isinstance(vel, (int, float)) or vel < 0.0 or math.isnan(vel):
            errors.append(f"velocity_pixels_per_second must be a non-negative float, got {vel}")

    # 7. Visual Quality Sanity Checks
    blur = rec_dict.get("blur_score")
    if blur is not None:
        if not isinstance(blur, (int, float)) or blur < 0.0 or math.isnan(blur):
            errors.append(f"blur_score must be a non-negative float, got {blur}")

    bright = rec_dict.get("brightness")
    if bright is not None:
        if not isinstance(bright, (int, float)) or not (0.0 <= bright <= 255.0):
            errors.append(f"brightness must be in [0.0, 255.0], got {bright}")

    contrast = rec_dict.get("contrast")
    if contrast is not None:
        if not isinstance(contrast, (int, float)) or contrast < 0.0 or math.isnan(contrast):
            errors.append(f"contrast must be non-negative, got {contrast}")

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )


def validate_input_artifacts(
    detection_file: Optional[Union[str, Path]] = None,
    tracking_file: Optional[Union[str, Path]] = None,
    metadata_file: Optional[Union[str, Path]] = None,
    strict: bool = True,
) -> ValidationResult:
    """Validate integration integrity of upstream detection, tracking, and metadata artifacts.

    Checks:
    - File existence and non-empty status.
    - JSON parsing and schema version adherence.
    - Frame sequencing and chronological monotonic ordering.
    - Detection bounding box coordinate sanity.
    - Timestamp sanity and monotonicity.
    - Cross-artifact dataset identifier alignment.

    Args:
        detection_file: Path to detection results JSON.
        tracking_file: Path to tracking results JSON.
        metadata_file: Path to dataset metadata CSV or JSON.
        strict: If True, treats missing optional keys or non-consecutive frames as errors.

    Returns:
        ValidationResult indicating validity and catalog of findings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    detection_data: Optional[Dict[str, Any]] = None
    tracking_data: Optional[Dict[str, Any]] = None
    metadata_rows: List[Dict[str, Any]] = []

    dataset_ids: Set[str] = set()

    # --------------------------------------------------------------------------
    # 1. Detection Artifact Validation
    # --------------------------------------------------------------------------
    if detection_file is not None:
        d_path = Path(detection_file)
        if not d_path.exists():
            errors.append(f"Detection results file does not exist: '{d_path}'")
        elif d_path.stat().st_size == 0:
            errors.append(f"Detection results file is empty: '{d_path}'")
        else:
            try:
                with open(d_path, "r", encoding="utf-8") as f:
                    detection_data = json.load(f)
            except Exception as ex:
                errors.append(f"Detection file '{d_path}' contains invalid JSON: {ex}")

        if detection_data is not None:
            d_schema = str(detection_data.get("schema_version"))
            if d_schema not in SUPPORTED_SCHEMA_VERSIONS:
                errors.append(f"Detection artifact has unsupported schema_version '{d_schema}'")

            d_set_id = detection_data.get("dataset_id")
            if d_set_id:
                dataset_ids.add(str(d_set_id))

            d_frames = detection_data.get("frame_results") or detection_data.get("frames")
            if not isinstance(d_frames, list):
                errors.append("Detection artifact missing 'frame_results' list")
            else:
                seen_frame_ids: Set[str] = set()
                last_idx = -1
                for f_entry in d_frames:
                    fid = f_entry.get("frame_id") or f_entry.get("image_filename") or f_entry.get("extracted_frame_id")
                    if not fid:
                        errors.append("Detection frame record missing frame identifier (frame_id or image_filename)")
                    elif str(fid) in seen_frame_ids:
                        errors.append(f"Duplicate frame identifier in detection results: '{fid}'")
                    if fid:
                        seen_frame_ids.add(str(fid))

                    idx = f_entry.get("source_frame_index", f_entry.get("frame_index"))
                    if idx is not None:
                        if idx < last_idx:
                            errors.append(f"Non-monotonic frame ordering in detections: index {idx} after {last_idx}")
                        last_idx = idx

                    # Check bounding boxes in frame
                    detections = f_entry.get("detections", [])
                    for det in detections:
                        bbox = det.get("bbox_xyxy", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            if any(math.isnan(v) or math.isinf(v) for v in [x1, y1, x2, y2]):
                                errors.append(f"Non-finite bbox coordinates in frame '{fid}': {bbox}")
                            elif x2 <= x1 or y2 <= y1:
                                errors.append(f"Inverted or zero area bbox in frame '{fid}': {bbox}")

    # --------------------------------------------------------------------------
    # 2. Tracking Artifact Validation
    # --------------------------------------------------------------------------
    if tracking_file is not None:
        t_path = Path(tracking_file)
        if not t_path.exists():
            errors.append(f"Tracking results file does not exist: '{t_path}'")
        elif t_path.stat().st_size == 0:
            errors.append(f"Tracking results file is empty: '{t_path}'")
        else:
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    tracking_data = json.load(f)
            except Exception as ex:
                errors.append(f"Tracking file '{t_path}' contains invalid JSON: {ex}")

        if tracking_data is not None:
            t_schema = str(tracking_data.get("schema_version"))
            if t_schema not in SUPPORTED_SCHEMA_VERSIONS:
                errors.append(f"Tracking artifact has unsupported schema_version '{t_schema}'")

            t_set_id = tracking_data.get("dataset_id")
            if t_set_id:
                dataset_ids.add(str(t_set_id))

            t_frames = tracking_data.get("frame_results") or tracking_data.get("frames") or tracking_data.get("observations")
            if not isinstance(t_frames, list):
                errors.append("Tracking artifact missing frame results or observations list")
            else:
                last_idx = -1
                for f_entry in t_frames:
                    idx = f_entry.get("source_frame_index", f_entry.get("frame_index"))
                    if idx is not None:
                        if idx < last_idx:
                            errors.append(f"Non-monotonic frame ordering in tracking: index {idx} after {last_idx}")
                        last_idx = idx

                    # Check track coordinates
                    tracks = f_entry.get("tracks", [])
                    if isinstance(f_entry.get("bbox_xyxy"), list) and "track_id" in f_entry:
                        # Single-object tracking observation format
                        tracks = [f_entry]

                    for trk in tracks:
                        bbox = trk.get("bbox_xyxy", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            if any(math.isnan(v) or math.isinf(v) for v in [x1, y1, x2, y2]):
                                errors.append(f"Non-finite bbox in tracking record: {bbox}")
                            elif x2 <= x1 or y2 <= y1:
                                errors.append(f"Inverted bbox in tracking record: {bbox}")

    # --------------------------------------------------------------------------
    # 3. Dataset Metadata Validation (CSV or JSON)
    # --------------------------------------------------------------------------
    if metadata_file is not None:
        m_path = Path(metadata_file)
        if not m_path.exists():
            errors.append(f"Dataset metadata file does not exist: '{m_path}'")
        elif m_path.stat().st_size == 0:
            errors.append(f"Dataset metadata file is empty: '{m_path}'")
        else:
            if m_path.suffix.lower() == ".csv":
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        metadata_rows = list(reader)
                        if not metadata_rows:
                            warnings.append(f"Metadata CSV '{m_path}' contains zero data rows")
                except Exception as ex:
                    errors.append(f"Failed to parse metadata CSV '{m_path}': {ex}")
            else:
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        if isinstance(m_data, list):
                            metadata_rows = m_data
                        elif isinstance(m_data, dict):
                            metadata_rows = m_data.get("frames", [])
                except Exception as ex:
                    errors.append(f"Failed to parse metadata JSON '{m_path}': {ex}")

    # --------------------------------------------------------------------------
    # 4. Cross-Artifact Dataset ID Alignment
    # --------------------------------------------------------------------------
    if len(dataset_ids) > 1:
        errors.append(f"Incompatible dataset IDs found across artifacts: {list(dataset_ids)}")

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )
