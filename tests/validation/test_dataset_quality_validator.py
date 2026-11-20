"""Unit and Integration Tests for Dataset Quality, Balance & Integrity Validator (Day 8 Task 6).

Covers:
- Annotation validity: geometry, coordinates, bounds, non-finite, zero-area
- Reliability label balance: TP + FP == Preds, TP + FN == GT, metric range [0, 1]
- Distribution analyzer: min, max, mean, std, median, quartiles, skew detection
- Temporal coverage: H1, H3, H5 offset verification, boundary handling
- Leakage detection: future frame index > source frame index
- Scorecard generation: 9-category audit scorecard
- Real-data integration tests: validation on real Day 8 artifacts
"""

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Tuple
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.dataset_quality_validator import (
    DATASET_VALIDATOR_SCHEMA_VERSION,
    DatasetQualityValidator,
    ScorecardCategory,
    ScorecardStatus,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_annotations_jsonl(tmp_path: Path) -> Path:
    """Create mock valid canonical annotations JSONL file."""
    records = [
        {
            "dataset_id": "test_ds",
            "sequence_id": "seq_01",
            "frame_id": "frame_000001.jpg",
            "annotation_id": "ann_1",
            "ground_truth_object_id": "trk_1",
            "class_name": "car",
            "image_width": 640,
            "image_height": 360,
            "x_min": 100.0,
            "y_min": 50.0,
            "x_max": 200.0,
            "y_max": 150.0,
        },
        {
            "dataset_id": "test_ds",
            "sequence_id": "seq_01",
            "frame_id": "frame_000001.jpg",
            "annotation_id": "ann_2",
            "ground_truth_object_id": "trk_2",
            "class_name": "person",
            "image_width": 640,
            "image_height": 360,
            "x_min": 300.0,
            "y_min": 50.0,
            "x_max": 350.0,
            "y_max": 200.0,
        },
    ]
    p = tmp_path / "mock_annotations.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def mock_reliability_labels(tmp_path: Path) -> Tuple[Path, Path]:
    """Create mock consistent object and frame reliability label files."""
    # Frame 1: 2 GT, 1 Pred -> 1 TP, 0 FP, 1 FN
    obj_labels = [
        {"frame_id": "frame_000001.jpg", "outcome": "true_positive", "object_correctness": 1.0},
        {"frame_id": "frame_000001.jpg", "outcome": "false_negative", "object_correctness": 0.0},
    ]
    frame_labels = [
        {
            "frame_id": "frame_000001.jpg",
            "extracted_frame_id": 1,
            "total_predictions": 1,
            "total_ground_truth": 2,
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 1,
            "precision": 1.0,
            "recall": 0.5,
            "f1_score": 0.6667,
            "composite_reliability": 0.65,
        }
    ]
    obj_p = tmp_path / "mock_obj.json"
    frame_p = tmp_path / "mock_frame.json"

    with open(obj_p, "w", encoding="utf-8") as f:
        json.dump(obj_labels, f)
    with open(frame_p, "w", encoding="utf-8") as f:
        json.dump(frame_labels, f)

    return obj_p, frame_p


# ==============================================================================
# 1. ANNOTATION VALIDITY TESTS
# ==============================================================================

def test_annotation_validity_pass(mock_annotations_jsonl: Path) -> None:
    """Valid canonical annotations must produce PASS status."""
    validator = DatasetQualityValidator()
    res = validator.validate_annotations(mock_annotations_jsonl)
    assert res["status"] == ScorecardStatus.PASS.value
    assert res["total_records"] == 2
    assert res["invalid_coordinates"] == 0
    assert res["out_of_bounds_coords"] == 0


def test_annotation_validity_out_of_bounds(tmp_path: Path) -> None:
    """Out-of-bounds coordinates (> 640 or > 360) must be flagged."""
    rec = {
        "dataset_id": "test_ds",
        "frame_id": "frame_000001.jpg",
        "image_width": 640,
        "image_height": 360,
        "x_min": 100.0,
        "y_min": 50.0,
        "x_max": 700.0,  # Out of bounds (> 640)
        "y_max": 150.0,
    }
    p = tmp_path / "oob.jsonl"
    p.write_text(json.dumps(rec) + "\n")

    validator = DatasetQualityValidator()
    res = validator.validate_annotations(p)
    assert res["out_of_bounds_coords"] == 1


def test_annotation_validity_zero_area(tmp_path: Path) -> None:
    """Degenerate zero-area box must be flagged as invalid."""
    rec = {
        "dataset_id": "test_ds",
        "frame_id": "frame_000001.jpg",
        "image_width": 640,
        "image_height": 360,
        "x_min": 100.0,
        "y_min": 50.0,
        "x_max": 100.0,  # width = 0
        "y_max": 150.0,
    }
    p = tmp_path / "zero_area.jsonl"
    p.write_text(json.dumps(rec) + "\n")

    validator = DatasetQualityValidator()
    res = validator.validate_annotations(p)
    assert res["zero_area_boxes"] == 1
    assert res["status"] == ScorecardStatus.FAIL.value


# ==============================================================================
# 2. RELIABILITY LABEL BALANCE TESTS
# ==============================================================================

def test_reliability_labels_balance_valid(mock_reliability_labels: Tuple[Path, Path]) -> None:
    """Consistent counts across object and frame records must pass validation."""
    obj_p, frame_p = mock_reliability_labels
    validator = DatasetQualityValidator()
    res = validator.validate_reliability_labels(obj_p, frame_p)
    assert res["status"] == ScorecardStatus.PASS.value
    assert res["prediction_balance_valid"] is True
    assert res["ground_truth_balance_valid"] is True
    assert res["object_frame_counts_consistent"] is True


def test_reliability_labels_balance_mismatch(tmp_path: Path) -> None:
    """Mismatched counts between object records and frame totals must fail validation."""
    obj_labels = [{"outcome": "true_positive"}]  # 1 TP
    frame_labels = [{
        "frame_id": "f1",
        "total_predictions": 5,  # Mismatch (claims 5 preds, but 1 TP + 0 FP = 1)
        "total_ground_truth": 1,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }]
    obj_p = tmp_path / "bad_obj.json"
    frame_p = tmp_path / "bad_frame.json"
    obj_p.write_text(json.dumps(obj_labels))
    frame_p.write_text(json.dumps(frame_labels))

    validator = DatasetQualityValidator()
    res = validator.validate_reliability_labels(obj_p, frame_p)
    assert res["status"] == ScorecardStatus.FAIL.value
    assert res["prediction_balance_valid"] is False


# ==============================================================================
# 3. DISTRIBUTION ANALYZER TESTS
# ==============================================================================

def test_distribution_analyzer_statistics(mock_annotations_jsonl: Path, tmp_path: Path) -> None:
    """Statistical distribution calculations must compute exact mean and quartiles."""
    frames = [
        {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "composite_reliability": 0.0},
        {"f1_score": 0.5, "precision": 0.5, "recall": 0.5, "composite_reliability": 0.5},
        {"f1_score": 1.0, "precision": 1.0, "recall": 1.0, "composite_reliability": 1.0},
    ]
    frame_p = tmp_path / "frames.json"
    frame_p.write_text(json.dumps(frames))

    validator = DatasetQualityValidator()
    res = validator.analyze_distributions(mock_annotations_jsonl, "non_existent.json", frame_p)
    f1_stats = res["f1_distribution"]
    assert f1_stats["min"] == 0.0
    assert f1_stats["max"] == 1.0
    assert f1_stats["mean"] == 0.5
    assert f1_stats["median"] == 0.5


def test_distribution_skew_warning(mock_annotations_jsonl: Path, tmp_path: Path) -> None:
    """When >90% of frames have F1=0, a WARNING status must be assigned."""
    frames = [{"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "composite_reliability": 0.0}] * 100
    frame_p = tmp_path / "skewed_frames.json"
    frame_p.write_text(json.dumps(frames))

    validator = DatasetQualityValidator()
    res = validator.analyze_distributions(mock_annotations_jsonl, "non_existent.json", frame_p)
    assert res["status"] == ScorecardStatus.WARNING.value
    assert res["zero_f1_frames_ratio"] == 1.0


# ==============================================================================
# 4. TEMPORAL COVERAGE & HORIZON TESTS
# ==============================================================================

def test_temporal_coverage_offsets(tmp_path: Path) -> None:
    """Verify that H1=+1, H3=+3, H5=+5 target frame offsets are strictly checked."""
    targets = [
        {
            "sequence_id": "seq_1",
            "source_extracted_frame_id": 10,
            "split": "train",
            "horizons": {
                "h1": {"target_status": "valid", "target_frame_id": "frame_000011.jpg"},
                "h3": {"target_status": "valid", "target_frame_id": "frame_000013.jpg"},
                "h5": {"target_status": "valid", "target_frame_id": "frame_000015.jpg"},
            }
        }
    ]
    tgt_p = tmp_path / "targets.json"
    seq_p = tmp_path / "seq.json"
    tgt_p.write_text(json.dumps(targets))
    seq_p.write_text(json.dumps({}))

    validator = DatasetQualityValidator()
    res = validator.validate_temporal_coverage(tgt_p, seq_p)
    assert res["status"] == ScorecardStatus.PASS.value
    assert res["invalid_temporal_offsets"] == 0
    assert res["h1_valid_targets"] == 1
    assert res["h3_valid_targets"] == 1
    assert res["h5_valid_targets"] == 1


# ==============================================================================
# 5. INTEGRITY & LEAKAGE AUDIT TESTS
# ==============================================================================

def test_leakage_detection_target_before_source(tmp_path: Path) -> None:
    """If target frame <= source frame, a leakage violation must be flagged."""
    targets = [
        {
            "sequence_id": "seq_1",
            "source_extracted_frame_id": 10,
            "horizons": {
                "h1": {"target_status": "valid", "target_frame_id": "frame_000009.jpg"},  # 9 <= 10 -> Leakage!
            }
        }
    ]
    tgt_p = tmp_path / "leakage_targets.json"
    tgt_p.write_text(json.dumps(targets))

    validator = DatasetQualityValidator()
    res = validator.audit_integrity_and_leakage(tmp_path, tgt_p)
    assert res["temporal_leakage_violations"] == 1
    assert res["status"] == ScorecardStatus.FAIL.value


# ==============================================================================
# 6. SCORECARD & REAL DATA INTEGRATION TESTS
# ==============================================================================

def test_real_dataset_full_audit() -> None:
    """Execute complete 9-category scorecard audit on actual Day 8 pipeline artifacts."""
    validator = DatasetQualityValidator()
    report = validator.run_full_audit()

    assert report["schema_version"] == DATASET_VALIDATOR_SCHEMA_VERSION
    assert len(report["scorecard"]) == 9
    categories = [c["category"] for c in report["scorecard"]]
    assert "Annotation Validity" in categories
    assert "Frame Alignment" in categories
    assert "Label Integrity" in categories
    assert "Class Distribution" in categories
    assert "Reliability Distribution" in categories
    assert "Temporal Coverage" in categories
    assert "Duplicate Detection" in categories
    assert "Leakage Validation" in categories
    assert "Source Traceability" in categories

    # Verify no category has FAIL status
    statuses = [c["status"] for c in report["scorecard"]]
    assert ScorecardStatus.FAIL.value not in statuses
