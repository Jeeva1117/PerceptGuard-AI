"""Unit Tests for Canonical Annotation Schema & Reliability Labeling Specifications.

Verifies:
- CanonicalBoundingBox geometric calculation, boundary checks, and rejection rules.
- calculate_iou mathematical correctness (identical, disjoint, partial, containment).
- compute_frame_f1 edge cases (empty scenes, pure false alarms, total detection blackouts).
- compute_composite_reliability weighting and bounds.
- ReliabilityDataContract validation across GT, detections, frame records, and temporal targets.
- Serialization integrity across all data model structures.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reliability_schema import (
    CanonicalBoundingBox,
    MatchingOutcome,
    TargetStatus,
    ObjectReliabilityRecord,
    FrameReliabilityRecord,
    HorizonTargetSpec,
    TemporalTargetRecord,
    ReliabilityDataContract,
    calculate_iou,
    compute_frame_f1,
    compute_composite_reliability,
    RELIABILITY_SCHEMA_VERSION,
)

# ==============================================================================
# 1. CANONICAL BOUNDING BOX TESTS
# ==============================================================================

def test_canonical_bounding_box_geometry():
    box = CanonicalBoundingBox(10.0, 20.0, 110.0, 220.0)
    assert box.width == 100.0
    assert box.height == 200.0
    assert box.area == 20000.0
    assert box.is_finite is True
    assert box.is_valid(640, 360) is True
    assert box.to_list() == [10.0, 20.0, 110.0, 220.0]

def test_canonical_bounding_box_rejection():
    # Inverted box
    inv_box = CanonicalBoundingBox(100.0, 20.0, 50.0, 200.0)
    assert inv_box.is_valid(640, 360) is False

    # Negative coordinates
    neg_box = CanonicalBoundingBox(-10.0, 20.0, 50.0, 200.0)
    assert neg_box.is_valid(640, 360) is False

    # Out of bounds
    oob_box = CanonicalBoundingBox(10.0, 20.0, 700.0, 200.0)
    assert oob_box.is_valid(640, 360) is False

    # Non-finite (NaN)
    nan_box = CanonicalBoundingBox(float("nan"), 20.0, 50.0, 200.0)
    assert nan_box.is_valid(640, 360) is False

# ==============================================================================
# 2. IOU CALCULATION TESTS
# ==============================================================================

def test_calculate_iou_identical():
    box1 = CanonicalBoundingBox(0.0, 0.0, 100.0, 100.0)
    box2 = CanonicalBoundingBox(0.0, 0.0, 100.0, 100.0)
    assert calculate_iou(box1, box2) == 1.0

def test_calculate_iou_disjoint():
    box1 = CanonicalBoundingBox(0.0, 0.0, 50.0, 50.0)
    box2 = CanonicalBoundingBox(60.0, 60.0, 100.0, 100.0)
    assert calculate_iou(box1, box2) == 0.0

def test_calculate_iou_partial():
    # Box 1: [0, 0, 10, 10] (Area = 100)
    # Box 2: [5, 0, 15, 10] (Area = 100)
    # Intersection: [5, 0, 10, 10] (Area = 50)
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    assert pytest.approx(calculate_iou(box1, box2), 0.001) == 1.0 / 3.0

def test_calculate_iou_containment():
    # Outer: [0, 0, 100, 100] (Area = 10,000)
    # Inner: [0, 0, 50, 50] (Area = 2,500)
    # Intersection: 2,500
    # Union: 10,000
    # IoU: 2,500 / 10,000 = 0.25
    outer = CanonicalBoundingBox(0.0, 0.0, 100.0, 100.0)
    inner = CanonicalBoundingBox(0.0, 0.0, 50.0, 50.0)
    assert pytest.approx(calculate_iou(outer, inner), 0.001) == 0.25

# ==============================================================================
# 3. FRAME-LEVEL METRICS & EDGE CASE TESTS
# ==============================================================================

def test_compute_frame_f1_standard():
    # TP=8, FP=2, FN=2
    # Precision = 8 / 10 = 0.80
    # Recall = 8 / 10 = 0.80
    # F1 = 0.80
    p, r, f1 = compute_frame_f1(true_positives=8, false_positives=2, false_negatives=2, total_predictions=10, total_ground_truth=10)
    assert pytest.approx(p, 0.0001) == 0.80
    assert pytest.approx(r, 0.0001) == 0.80
    assert pytest.approx(f1, 0.0001) == 0.80

def test_compute_frame_f1_both_empty():
    # No predictions and no ground truth (empty scene)
    p, r, f1 = compute_frame_f1(0, 0, 0, total_predictions=0, total_ground_truth=0)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0

def test_compute_frame_f1_predictions_only():
    # Pure false alarms (FP=5, GT=0)
    p, r, f1 = compute_frame_f1(0, 5, 0, total_predictions=5, total_ground_truth=0)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

def test_compute_frame_f1_ground_truth_only():
    # Complete detection failure (FN=5, Pred=0)
    p, r, f1 = compute_frame_f1(0, 0, 5, total_predictions=0, total_ground_truth=5)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

def test_compute_frame_f1_zero_matches():
    # Both non-empty but zero true positives
    p, r, f1 = compute_frame_f1(0, 3, 3, total_predictions=3, total_ground_truth=3)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

def test_compute_composite_reliability():
    # F1 = 0.80, mean_iou = 0.70, alpha = 0.70
    # Composite = 0.70 * 0.80 + 0.30 * 0.70 = 0.56 + 0.21 = 0.77
    score = compute_composite_reliability(f1_score=0.80, mean_matched_iou=0.70, alpha=0.70)
    assert pytest.approx(score, 0.001) == 0.77

# ==============================================================================
# 4. DATA CONTRACT VALIDATION TESTS
# ==============================================================================

def test_reliability_data_contract_gt():
    valid_gt = {
        "schema_version": "1.0.0",
        "sequence_id": "026c7465-309f6d33",
        "dataset_id": "driving_sample",
        "frames": [
            {
                "frame_index": 0,
                "extracted_frame_id": 1,
                "frame_filename": "frame_000001.jpg",
                "annotations": []
            }
        ]
    }
    assert ReliabilityDataContract.validate_ground_truth_schema(valid_gt) is True

    # Missing frames key
    assert ReliabilityDataContract.validate_ground_truth_schema({"schema_version": "1.0.0"}) is False

def test_reliability_data_contract_detection():
    valid_det = {
        "schema_version": "1.0",
        "dataset_id": "driving_sample",
        "total_frames": 1,
        "frame_results": [
            {
                "image_filename": "frame_000001.jpg",
                "extracted_frame_id": 1,
                "detections": []
            }
        ]
    }
    assert ReliabilityDataContract.validate_detection_schema(valid_det) is True

    # Empty frame results
    assert ReliabilityDataContract.validate_detection_schema({"schema_version": "1.0", "dataset_id": "ds", "total_frames": 0, "frame_results": []}) is False

def test_reliability_data_contract_frame_reliability():
    valid_frame = {
        "frame_id": "frame_000001.jpg",
        "extracted_frame_id": 1,
        "timestamp_seconds": 0.0,
        "true_positives": 2,
        "false_positives": 0,
        "false_negatives": 1,
        "precision": 1.0,
        "recall": 0.6667,
        "f1_score": 0.80,
        "primary_reliability": 0.80
    }
    assert ReliabilityDataContract.validate_frame_reliability_schema(valid_frame) is True

    # Out of bounds F1
    invalid_frame = dict(valid_frame)
    invalid_frame["f1_score"] = 1.5
    assert ReliabilityDataContract.validate_frame_reliability_schema(invalid_frame) is False

def test_reliability_data_contract_temporal_targets():
    valid_target = {
        "sequence_id": "seq_001",
        "source_frame_id": "frame_000005.jpg",
        "split": "train",
        "horizons": {
            "h1": {"horizon_step": 1, "target_status": "valid"},
            "h3": {"horizon_step": 3, "target_status": "valid"},
            "h5": {"horizon_step": 5, "target_status": "valid"}
        }
    }
    assert ReliabilityDataContract.validate_temporal_target_schema(valid_target) is True

# ==============================================================================
# 5. DATA MODEL SERIALIZATION TESTS
# ==============================================================================

def test_data_model_serialization():
    obj_rec = ObjectReliabilityRecord(
        frame_id="frame_000001.jpg",
        extracted_frame_id=1,
        source_frame_index=0,
        timestamp_seconds=0.0,
        detection_index=0,
        ground_truth_id="gt_1001",
        track_id="1001",
        predicted_class="car",
        ground_truth_class="car",
        confidence=0.88,
        iou=0.75,
        outcome=MatchingOutcome.TRUE_POSITIVE,
        object_correctness=1.0
    )
    d = obj_rec.to_dict()
    assert d["outcome"] == "true_positive"
    assert d["object_correctness"] == 1.0

    frame_rec = FrameReliabilityRecord(
        frame_id="frame_000001.jpg",
        extracted_frame_id=1,
        source_frame_index=0,
        timestamp_seconds=0.0,
        total_predictions=2,
        total_ground_truth=3,
        true_positives=2,
        false_positives=0,
        false_negatives=1,
        precision=1.0,
        recall=0.6667,
        f1_score=0.80,
        mean_matched_iou=0.72,
        composite_reliability=0.776,
        primary_reliability=0.80
    )
    fd = frame_rec.to_dict()
    assert fd["f1_score"] == 0.80
    assert fd["status"] == "valid"

    h_spec = HorizonTargetSpec(
        horizon_step=1,
        target_frame_id="frame_000006.jpg",
        target_source_frame_index=30,
        target_timestamp_seconds=1.0,
        target_reliability=0.85,
        target_status=TargetStatus.VALID
    )
    hd = h_spec.to_dict()
    assert hd["target_status"] == "valid"
    assert hd["target_reliability"] == 0.85
