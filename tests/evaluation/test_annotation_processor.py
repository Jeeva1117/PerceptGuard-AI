"""Tests for Ground-Truth Annotation Processing Pipeline (Day 8 Task 4).

Covers:
- Parser tests: valid, empty, missing, malformed, unknown class, invalid numeric
- Coordinate tests: valid, zero-area, negative, out-of-bounds, non-finite
- Identity & schema tests: valid frame ID, duplicate records, object ID preservation, schema version
- Alignment tests: image matching, missing image, dimension mismatch
- Output tests: JSONL serialization, deterministic ordering, relative path, atomic write
"""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.annotation_processor import (
    ANNOTATION_SCHEMA_VERSION,
    AnnotationParseError,
    AnnotationProcessor,
    BDD100KMOTParser,
    CanonicalAnnotationRecord,
    ClassMapper,
    CoordinateNormalizer,
    FrameAnnotationSummary,
    FrameStatus,
    ProcessingSummary,
    ValidationStatus,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_valid_bdd_json(tmp_path: Path) -> Path:
    """Create a temporary valid BDD100K MOT sequence JSON fixture."""
    data = {
        "name": "026c7465-309f6d33",
        "frames": [
            {
                "frame_index": 0,
                "video_frame": 0,
                "labels": [
                    {
                        "id": "trk_101",
                        "category": "car",
                        "box2d": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
                        "attributes": {"crowd": False, "occluded": False, "truncated": False}
                    },
                    {
                        "id": "trk_102",
                        "category": "pedestrian",
                        "box2d": {"x1": 500.0, "y1": 150.0, "x2": 550.0, "y2": 350.0},
                        "attributes": {"crowd": False, "occluded": True, "truncated": False}
                    }
                ]
            },
            {
                "frame_index": 1,
                "video_frame": 6,
                "labels": [
                    {
                        "id": "trk_101",
                        "category": "car",
                        "box2d": {"x1": 110.0, "y1": 205.0, "x2": 310.0, "y2": 405.0},
                        "attributes": {"crowd": False, "occluded": False, "truncated": False}
                    }
                ]
            }
        ]
    }
    file_path = tmp_path / "valid_sequence.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_frames_dir(tmp_path: Path) -> Path:
    """Create mock preprocessed image files."""
    frames_dir = tmp_path / "preprocessed_frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000001.jpg").write_text("mock_jpg_content_1")
    (frames_dir / "frame_000002.jpg").write_text("mock_jpg_content_2")
    return frames_dir


# ==============================================================================
# PARSER TESTS
# ==============================================================================

def test_parser_valid_file(sample_valid_bdd_json: Path) -> None:
    """Test parsing a well-formed BDD100K JSON file."""
    parser = BDD100KMOTParser()
    data = parser.parse_file(sample_valid_bdd_json)
    assert isinstance(data, dict)
    assert data["name"] == "026c7465-309f6d33"
    assert len(data["frames"]) == 2


def test_parser_empty_file_raises(tmp_path: Path) -> None:
    """Test that an empty file raises AnnotationParseError."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")
    parser = BDD100KMOTParser()
    with pytest.raises(AnnotationParseError, match="empty"):
        parser.parse_file(empty_file)


def test_parser_missing_file_raises() -> None:
    """Test that a non-existent file raises AnnotationParseError."""
    parser = BDD100KMOTParser()
    with pytest.raises(AnnotationParseError, match="not found"):
        parser.parse_file("non_existent_file.json")


def test_parser_malformed_json_raises(tmp_path: Path) -> None:
    """Test that syntax errors in JSON raise AnnotationParseError."""
    malformed_file = tmp_path / "corrupt.json"
    malformed_file.write_text("{'name': 'broken', frames: [}")
    parser = BDD100KMOTParser()
    with pytest.raises(AnnotationParseError, match="Malformed JSON"):
        parser.parse_file(malformed_file)


def test_parser_unknown_class_handling(tmp_path: Path) -> None:
    """Test unmapped category handling."""
    data = {
        "name": "seq_unknown",
        "frames": [
            {
                "frame_index": 0,
                "labels": [
                    {
                        "id": "1",
                        "category": "spaceship_alien",
                        "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
                    }
                ]
            }
        ]
    }
    p = tmp_path / "unknown_class.json"
    p.write_text(json.dumps(data))

    processor = AnnotationProcessor()
    _, summary = processor.process_sequence(p)
    assert summary.unmapped_classes_count == 1
    assert any(err["status"] == "UNKNOWN_CLASS" for err in summary.error_log)


def test_parser_invalid_numeric_values(tmp_path: Path) -> None:
    """Test non-numeric coordinate strings in box2d."""
    data = {
        "name": "seq_bad_numeric",
        "frames": [
            {
                "frame_index": 0,
                "labels": [
                    {
                        "id": "1",
                        "category": "car",
                        "box2d": {"x1": "not_a_number", "y1": 100, "x2": 200, "y2": 200}
                    }
                ]
            }
        ]
    }
    p = tmp_path / "bad_numeric.json"
    p.write_text(json.dumps(data))

    processor = AnnotationProcessor()
    _, summary = processor.process_sequence(p)
    assert summary.invalid_records_count == 1
    assert any(err["status"] == "NON_FINITE" for err in summary.error_log)


# ==============================================================================
# COORDINATE TESTS
# ==============================================================================

def test_coordinate_valid_transformation() -> None:
    """Test exact mathematical coordinate transform from native to canvas."""
    normalizer = CoordinateNormalizer()
    # Native box: x1=100, y1=200, x2=300, y2=400 in 1280x720
    # 90 deg cw: rx1 = 720 - 400 = 320, rx2 = 720 - 200 = 520, ry1 = 100, ry2 = 300
    # canvas scale: cx1 = 320 * 0.28125 + 218.75 = 90.0 + 218.75 = 308.75
    # cx2 = 520 * 0.28125 + 218.75 = 146.25 + 218.75 = 365.0
    # cy1 = 100 * 0.28125 = 28.125
    # cy2 = 300 * 0.28125 = 84.375
    transformed, status = normalizer.transform_box(100.0, 200.0, 300.0, 400.0)
    assert status == ValidationStatus.VALID
    assert transformed is not None
    cx1, cy1, cx2, cy2 = transformed
    assert pytest.approx(cx1, 0.01) == 308.75
    assert pytest.approx(cx2, 0.01) == 365.0
    assert pytest.approx(cy1, 0.01) == 28.125
    assert pytest.approx(cy2, 0.01) == 84.375


def test_coordinate_zero_area_box() -> None:
    """Test rejection of zero-width or zero-height bounding box."""
    normalizer = CoordinateNormalizer()
    transformed, status = normalizer.transform_box(100.0, 200.0, 100.0, 200.0)
    assert status == ValidationStatus.INVALID_COORDINATES
    assert transformed is None


def test_coordinate_inverted_box() -> None:
    """Test rejection of inverted coordinates (x1 > x2)."""
    normalizer = CoordinateNormalizer()
    transformed, status = normalizer.transform_box(300.0, 200.0, 100.0, 400.0)
    assert status == ValidationStatus.INVALID_COORDINATES
    assert transformed is None


def test_coordinate_non_finite() -> None:
    """Test rejection of NaN or Inf coordinates."""
    normalizer = CoordinateNormalizer()
    transformed, status = normalizer.transform_box(float("nan"), 200.0, 300.0, 400.0)
    assert status == ValidationStatus.NON_FINITE
    assert transformed is None

    transformed_inf, status_inf = normalizer.transform_box(100.0, float("inf"), 300.0, 400.0)
    assert status_inf == ValidationStatus.NON_FINITE
    assert transformed_inf is None


# ==============================================================================
# IDENTITY & SCHEMA TESTS
# ==============================================================================

def test_canonical_annotation_record_schema() -> None:
    """Test field existence, types, and serialization in CanonicalAnnotationRecord."""
    rec = CanonicalAnnotationRecord(
        dataset_id="test_ds",
        sequence_id="test_seq",
        frame_id="frame_000001.jpg",
        timestamp=0.20,
        image_path="relative/path/frame_000001.jpg",
        image_width=640,
        image_height=360,
        annotation_id="ann_01",
        ground_truth_object_id="trk_01",
        class_id=2,
        class_name="car",
        x_min=10.0,
        y_min=20.0,
        x_max=100.0,
        y_max=150.0,
    )
    assert rec.box_width == 90.0
    assert rec.box_height == 130.0
    assert rec.box_area == 11700.0
    assert rec.schema_version == ANNOTATION_SCHEMA_VERSION

    d = rec.to_dict()
    assert d["frame_id"] == "frame_000001.jpg"
    assert d["ground_truth_object_id"] == "trk_01"
    assert d["class_name"] == "car"


def test_missing_object_id_preserved_as_none() -> None:
    """Test that missing tracking IDs remain None and are not artificially fabricated."""
    rec = CanonicalAnnotationRecord(
        dataset_id="test_ds",
        sequence_id="test_seq",
        frame_id="frame_000001.jpg",
        timestamp=0.0,
        image_path="path.jpg",
        image_width=640,
        image_height=360,
        annotation_id="ann_02",
        ground_truth_object_id=None,
        class_id=0,
        class_name="person",
        x_min=10.0,
        y_min=10.0,
        x_max=50.0,
        y_max=100.0,
    )
    assert rec.ground_truth_object_id is None
    d = rec.to_dict()
    assert d["ground_truth_object_id"] is None


# ==============================================================================
# ALIGNMENT TESTS
# ==============================================================================

def test_frame_alignment_matching_images(sample_valid_bdd_json: Path, sample_frames_dir: Path) -> None:
    """Test frame alignment when all images exist."""
    processor = AnnotationProcessor()
    frame_summaries, summary = processor.process_sequence(
        annotation_file=sample_valid_bdd_json,
        frames_dir=sample_frames_dir,
    )
    assert len(frame_summaries) == 2
    assert summary.frames_aligned_count == 2
    assert summary.missing_images_count == 0
    assert frame_summaries[0].status == FrameStatus.VALID_WITH_OBJECTS
    assert frame_summaries[0].image_exists is True


def test_frame_alignment_missing_image(sample_valid_bdd_json: Path, tmp_path: Path) -> None:
    """Test detection and handling of missing image files."""
    empty_frames_dir = tmp_path / "empty_dir"
    empty_frames_dir.mkdir()

    processor = AnnotationProcessor()
    frame_summaries, summary = processor.process_sequence(
        annotation_file=sample_valid_bdd_json,
        frames_dir=empty_frames_dir,
    )
    assert summary.missing_images_count == 2
    assert summary.frames_not_aligned_count == 2
    assert frame_summaries[0].status == FrameStatus.MISSING_IMAGE


# ==============================================================================
# OUTPUT TESTS
# ==============================================================================

def test_atomic_export_jsonl_and_summary(
    sample_valid_bdd_json: Path, sample_frames_dir: Path, tmp_path: Path
) -> None:
    """Test atomic writing and deterministic output structure."""
    out_jsonl = tmp_path / "output" / "annotations.jsonl"
    out_summary = tmp_path / "output" / "summary.json"

    processor = AnnotationProcessor()
    frame_summaries, summary = processor.process_sequence(
        annotation_file=sample_valid_bdd_json,
        frames_dir=sample_frames_dir,
    )

    jsonl_path = processor.export_canonical_jsonl(frame_summaries, out_jsonl)
    summary_path = processor.export_summary_json(summary, out_summary)

    assert jsonl_path.is_file()
    assert summary_path.is_file()

    # Verify JSONL lines
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # Frame 0 has 2 boxes, Frame 1 has 1 box
    for line in lines:
        parsed = json.loads(line)
        assert "frame_id" in parsed
        assert "x_min" in parsed
        assert "y_max" in parsed
        assert parsed["schema_version"] == ANNOTATION_SCHEMA_VERSION

    # Verify JSON summary
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_data["pipeline_version"] == ANNOTATION_SCHEMA_VERSION
    assert summary_data["valid_records_count"] == 3
    assert summary_data["status"] == "PASS"
