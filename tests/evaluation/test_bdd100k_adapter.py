"""Unit and Integration Tests for BDD100K Annotation Adapter & Ingestion Pipeline.

Verifies:
- BoundingBox geometric calculations, validity checks, and boundary clamping.
- Coordinate transformations across native, raw video, and canvas letterbox spaces.
- Category taxonomy and class ID mapping (BDD100K to YOLO COCO).
- Sequence parsing, frame index alignment, and persistent tracking ID preservation.
- Manifest generation with portable relative paths.
- Bounded data integrity auditing (corrupt boxes, missing images, duplicate frames).
- Error handling for malformed JSON, missing files, and invalid schemas.
- Real dataset integration against the acquired BDD100K sequence 026c7465-309f6d33.
"""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bdd100k_adapter import (
    BDD100KAdapter,
    BDD100KAdapterError,
    AnnotationFileNotFoundError,
    AnnotationSchemaError,
    InvalidBoundingBoxError,
    CoordinateConversionError,
    FrameAlignmentError,
    BoundingBox,
    GroundTruthObject,
    FrameGroundTruth,
    SequenceGroundTruth,
    CoordinateTransformer,
    DEFAULT_BDD_TO_COCO_MAPPING,
)

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_raw_sequence_dict():
    """Provides a controlled synthetic BDD100K MOT sequence dict."""
    return {
        "name": "synthetic_seq_001",
        "dataset": "bdd100k",
        "task": "mot",
        "frames": [
            {
                "name": "synthetic_seq_001-0000001.jpg",
                "frame_index": 0,
                "video_frame": 0,
                "labels": [
                    {
                        "id": "1001",
                        "category": "car",
                        "box2d": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
                        "attributes": {"crowd": False, "occluded": False, "truncated": False}
                    },
                    {
                        "id": "1002",
                        "category": "pedestrian",
                        "box2d": {"x1": 500.0, "y1": 250.0, "x2": 550.0, "y2": 350.0},
                        "attributes": {"crowd": False, "occluded": True, "truncated": False}
                    }
                ]
            },
            {
                "name": "synthetic_seq_001-0000002.jpg",
                "frame_index": 1,
                "video_frame": 6,
                "labels": [
                    {
                        "id": "1001",
                        "category": "car",
                        "box2d": {"x1": 105.0, "y1": 202.0, "x2": 305.0, "y2": 402.0},
                        "attributes": {"crowd": False, "occluded": False, "truncated": False}
                    }
                ]
            }
        ]
    }

# ==============================================================================
# 1. BOUNDING BOX GEOMETRY & CLAMPING TESTS
# ==============================================================================

def test_bounding_box_properties():
    box = BoundingBox(x1=50.0, y1=100.0, x2=250.0, y2=300.0)
    assert box.width == 200.0
    assert box.height == 200.0
    assert box.area == 40000.0
    assert box.is_finite is True
    assert box.to_list() == [50.0, 100.0, 250.0, 300.0]
    assert box.to_dict() == {"x1": 50.0, "y1": 100.0, "x2": 250.0, "y2": 300.0}

def test_bounding_box_validity():
    # Valid box
    box_valid = BoundingBox(10.0, 20.0, 100.0, 150.0)
    assert box_valid.is_valid(canvas_width=640, canvas_height=360) is True

    # Inverted box (x2 <= x1)
    box_inverted_x = BoundingBox(100.0, 20.0, 50.0, 150.0)
    assert box_inverted_x.is_valid(canvas_width=640, canvas_height=360) is False

    # Inverted box (y2 <= y1)
    box_inverted_y = BoundingBox(10.0, 150.0, 100.0, 20.0)
    assert box_inverted_y.is_valid(canvas_width=640, canvas_height=360) is False

    # Out of bounds (negative)
    box_neg = BoundingBox(-5.0, 20.0, 100.0, 150.0)
    assert box_neg.is_valid(canvas_width=640, canvas_height=360) is False

    # Out of bounds (exceeding canvas)
    box_oob = BoundingBox(10.0, 20.0, 700.0, 150.0)
    assert box_oob.is_valid(canvas_width=640, canvas_height=360) is False

    # Non-finite (NaN / Inf)
    box_nan = BoundingBox(float("nan"), 20.0, 100.0, 150.0)
    assert box_nan.is_valid(canvas_width=640, canvas_height=360) is False

def test_bounding_box_clamping():
    box = BoundingBox(x1=-20.0, y1=10.0, x2=700.0, y2=400.0)
    clamped = box.clamp(canvas_width=640.0, canvas_height=360.0)
    assert clamped.x1 == 0.0
    assert clamped.y1 == 10.0
    assert clamped.x2 == 640.0
    assert clamped.y2 == 360.0

# ==============================================================================
# 2. COORDINATE TRANSFORMATION TESTS
# ==============================================================================

def test_coordinate_transformer_native_to_raw_video():
    transformer = CoordinateTransformer()
    # Box in 1280x720: x in [100, 300], y in [200, 400]
    # Clockwise 90 deg rotation into 720x1280:
    # x_raw = 720 - y_native -> [720 - 400, 720 - 200] = [320, 520]
    # y_raw = x_native -> [100, 300]
    native_box = BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=400.0)
    raw_box = transformer.native_to_raw_video(native_box)
    assert pytest.approx(raw_box.x1, 0.1) == 320.0
    assert pytest.approx(raw_box.x2, 0.1) == 520.0
    assert pytest.approx(raw_box.y1, 0.1) == 100.0
    assert pytest.approx(raw_box.y2, 0.1) == 300.0

def test_coordinate_transformer_raw_video_to_canvas():
    transformer = CoordinateTransformer()
    # Raw box in 720x1280: x in [0, 720], y in [0, 1280]
    # scale = 0.28125, offset_x = 218.75
    # x_canvas = x_raw * 0.28125 + 218.75 -> [218.75, 720 * 0.28125 + 218.75] = [218.75, 421.25]
    # y_canvas = y_raw * 0.28125 -> [0, 1280 * 0.28125] = [0, 360.0]
    raw_box = BoundingBox(x1=0.0, y1=0.0, x2=720.0, y2=1280.0)
    canvas_box = transformer.raw_video_to_canvas(raw_box)
    assert pytest.approx(canvas_box.x1, 0.1) == 218.75
    assert pytest.approx(canvas_box.x2, 0.1) == 421.25
    assert pytest.approx(canvas_box.y1, 0.1) == 0.0
    assert pytest.approx(canvas_box.y2, 0.1) == 360.0

def test_coordinate_transformer_native_to_canvas():
    transformer = CoordinateTransformer()
    native_box = BoundingBox(x1=0.0, y1=0.0, x2=1280.0, y2=720.0)
    canvas_box = transformer.native_to_canvas(native_box)
    # The entire native image maps inside the canvas letterbox active region:
    assert pytest.approx(canvas_box.x1, 0.1) == 218.75
    assert pytest.approx(canvas_box.x2, 0.1) == 421.25
    assert pytest.approx(canvas_box.y1, 0.1) == 0.0
    assert pytest.approx(canvas_box.y2, 0.1) == 360.0

def test_coordinate_transformer_unsupported_space():
    transformer = CoordinateTransformer()
    box = BoundingBox(10, 10, 50, 50)
    with pytest.raises(CoordinateConversionError):
        transformer.transform(box, target_space="non_existent_space")

# ==============================================================================
# 3. CLASS MAPPING TESTS
# ==============================================================================

def test_class_mapping_taxonomy():
    adapter = BDD100KAdapter()
    assert adapter.class_mapping["pedestrian"] == ("person", 0)
    assert adapter.class_mapping["rider"] == ("person", 0)
    assert adapter.class_mapping["car"] == ("car", 2)
    assert adapter.class_mapping["truck"] == ("truck", 7)
    assert adapter.class_mapping["bus"] == ("bus", 5)
    assert adapter.class_mapping["traffic light"] == ("traffic light", 9)
    assert adapter.class_mapping["bicycle"] == ("bicycle", 1)
    assert adapter.class_mapping["motorcycle"] == ("motorcycle", 3)
    assert adapter.class_mapping["train"] == ("train", 6)

# ==============================================================================
# 4. SEQUENCE PARSER TESTS
# ==============================================================================

def test_parse_sequence_synthetic(sample_raw_sequence_dict):
    adapter = BDD100KAdapter()
    seq = adapter.parse_sequence(sample_raw_sequence_dict, target_space="canvas")

    assert seq.sequence_id == "synthetic_seq_001"
    assert seq.total_frames == 2
    assert seq.total_annotations == 3
    assert seq.unique_track_count == 2
    assert "car" in seq.class_distribution
    assert "person" in seq.class_distribution

    # Frame 1 assertions
    f1 = seq.frames[0]
    assert f1.extracted_frame_id == 1
    assert f1.frame_filename == "frame_000001.jpg"
    assert f1.object_count == 2
    assert f1.image_width == 640
    assert f1.image_height == 360

    # Verify tracking ID preservation
    track_ids = {a.track_id for a in f1.annotations}
    assert track_ids == {"1001", "1002"}

    # Frame 2 assertions
    f2 = seq.frames[1]
    assert f2.extracted_frame_id == 2
    assert f2.frame_filename == "frame_000002.jpg"
    assert f2.object_count == 1
    assert f2.annotations[0].track_id == "1001"

def test_parse_sequence_invalid_schema():
    adapter = BDD100KAdapter()
    # Missing frames
    with pytest.raises(AnnotationSchemaError):
        adapter.parse_sequence({"name": "bad_seq"})

    # Non-list frames
    with pytest.raises(AnnotationSchemaError):
        adapter.parse_sequence({"name": "bad_seq", "frames": "not_a_list"})

    # Empty frames list
    with pytest.raises(AnnotationSchemaError):
        adapter.parse_sequence({"name": "bad_seq", "frames": []})

# ==============================================================================
# 5. INTEGRITY VALIDATION & MANIFEST TESTS
# ==============================================================================

def test_validate_integrity_pass(sample_raw_sequence_dict):
    adapter = BDD100KAdapter()
    seq = adapter.parse_sequence(sample_raw_sequence_dict, target_space="canvas")
    report = adapter.validate_integrity(seq)
    assert report["status"] == "PASS"
    assert report["invalid_boxes_count"] == 0
    assert report["corrupt_frames_count"] == 0

def test_validate_integrity_missing_image(sample_raw_sequence_dict, tmp_path):
    adapter = BDD100KAdapter()
    seq = adapter.parse_sequence(sample_raw_sequence_dict, target_space="canvas")
    # Empty frames dir -> images missing
    report = adapter.validate_integrity(seq, frames_dir=tmp_path)
    assert report["missing_images_count"] == 2
    assert report["status"] == "PARTIAL"

def test_generate_manifest_and_export(sample_raw_sequence_dict, tmp_path):
    adapter = BDD100KAdapter()
    seq = adapter.parse_sequence(sample_raw_sequence_dict, target_space="canvas")

    ann_json_path = tmp_path / "annotations.json"
    manifest_path = tmp_path / "manifest.json"

    adapter.export_standardized_json(seq, ann_json_path)
    assert ann_json_path.is_file()

    manifest = adapter.generate_manifest(
        seq,
        annotation_file_rel_path="annotations.json",
        frames_dir_rel_path="frames",
        output_manifest_path=manifest_path
    )
    assert manifest_path.is_file()
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["total_frames"] == 2
    assert manifest["relative_paths"]["annotations_file"] == "annotations.json"

# ==============================================================================
# 6. ERROR HANDLING & FILE LOADING TESTS
# ==============================================================================

def test_load_raw_json_non_existent():
    adapter = BDD100KAdapter()
    with pytest.raises(AnnotationFileNotFoundError):
        adapter.load_raw_json("non_existent_annotation_path_12345.json")

def test_load_raw_json_invalid_format(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ unclosed json: ", encoding="utf-8")
    adapter = BDD100KAdapter()
    with pytest.raises(AnnotationSchemaError):
        adapter.load_raw_json(bad_json)

# ==============================================================================
# 7. REAL DATASET INTEGRATION TEST
# ==============================================================================

def test_real_bdd100k_dataset_integration():
    """Validates the official BDD100K sequence 026c7465-309f6d33 annotations."""
    raw_ann_path = Path("datasets/annotations/bdd100k/bdd100k_mot_026c7465_309f6d33.json")
    if not raw_ann_path.is_file():
        pytest.skip("Official BDD100K raw annotation file not available on disk.")

    adapter = BDD100KAdapter()
    raw_data = adapter.load_raw_json(raw_ann_path)
    seq = adapter.parse_sequence(raw_data, target_space="canvas", max_frames=200)

    assert seq.sequence_id == "026c7465-309f6d33"
    assert seq.total_frames == 200
    assert seq.total_annotations > 5000
    assert seq.unique_track_count > 100

    # Verify preprocessed frames alignment
    frames_dir = Path("datasets/processed/driving_sample/preprocessed_frames")
    report = adapter.validate_integrity(seq, frames_dir=frames_dir)
    assert report["status"] == "PASS"
    assert report["missing_images_count"] == 0
    assert report["invalid_boxes_count"] == 0
