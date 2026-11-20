"""PerceptGuard-AI: Ground-Truth Reliability Label Generation Pipeline (Day 8 Task 5).

Generates scientifically defensible, deterministic, reproducible object-level correctness
labels, frame-level reliability scores (F1-score & composite), and multi-horizon temporal
forecasting targets (H1, H3, H5) via class-aware Hungarian bipartite matching between YOLO
detections and authoritative BDD100K ground-truth annotations.

Key Guarantees:
- Bipartite one-to-one Hungarian matching (cost = 1.0 - IoU, threshold = 0.50).
- Pure mathematical IoU calculation with strict coordinate validation.
- Complete traceability: prediction ID, GT annotation ID, track ID, bounding boxes, IoU.
- Frame-level F1-score with codified edge cases (empty scene = 1.0, hallucination = 0.0).
- Multi-horizon temporal target population across sequences without future information leakage.
- Atomic, reproducible file exports with strictly relative paths.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from src.evaluation.reliability_schema import (
    CanonicalBoundingBox,
    FrameReliabilityRecord,
    HorizonTargetSpec,
    MatchingOutcome,
    ObjectReliabilityRecord,
    ReliabilityDataContract,
    TargetStatus,
    TemporalTargetRecord,
    calculate_iou,
    compute_composite_reliability,
    compute_frame_f1,
    SCHEMA_VERSION as RELIABILITY_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION LOADER
# ==============================================================================

@dataclass
class LabelingConfig:
    """Configuration parameters governing reliability label generation."""
    schema_version: str = "1.0.0"
    iou_threshold: float = 0.50
    class_aware: bool = True
    min_detection_confidence: float = 0.25
    min_box_area_pixels: float = 4.0
    primary_metric: str = "f1_score"
    f1_weight: float = 0.70
    mean_iou_weight: float = 0.30
    horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    fps: float = 5.0
    input_ground_truth: str = "datasets/processed/driving_sample/annotations/annotations_canonical.jsonl"
    input_detections: str = "datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json"
    sequence_metadata_path: str = "datasets/processed/driving_sample/features/sequences/run_official/sequence_metadata.json"
    output_directory: str = "datasets/processed/driving_sample/reliability_labels"
    output_object_labels: str = "object_detection_labels.json"
    output_frame_labels: str = "frame_reliability_labels.json"
    output_temporal_targets: str = "temporal_reliability_targets.json"
    output_manifest: str = "reliability_manifest.json"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "LabelingConfig":
        """Load configuration from YAML file with fallbacks."""
        p = Path(yaml_path)
        if not p.is_file():
            logger.warning(f"Config file not found at '{p}'; using default LabelingConfig.")
            return cls()

        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        matching = raw.get("matching", {})
        frame_rel = raw.get("frame_level_reliability", {})
        comp = frame_rel.get("composite_formulation", {})
        temp = raw.get("temporal_horizons", {})
        contracts = raw.get("data_contracts", {})
        out_files = contracts.get("output_files", {})

        return cls(
            schema_version=raw.get("schema_version", "1.0.0"),
            iou_threshold=float(matching.get("iou_threshold", 0.50)),
            class_aware=bool(matching.get("class_aware", True)),
            min_detection_confidence=float(matching.get("min_detection_confidence", 0.25)),
            min_box_area_pixels=float(matching.get("min_box_area_pixels", 4.0)),
            primary_metric=str(frame_rel.get("primary_metric", "f1_score")),
            f1_weight=float(comp.get("f1_weight", 0.70)),
            mean_iou_weight=float(comp.get("mean_iou_weight", 0.30)),
            horizons=list(temp.get("horizons", [1, 3, 5])),
            fps=float(temp.get("frame_rate_fps", 5.0)),
            input_ground_truth=str(contracts.get("input_ground_truth", "datasets/processed/driving_sample/annotations/annotations_canonical.jsonl")),
            input_detections=str(contracts.get("input_detections", "datasets/processed/driving_sample/detections/batch_runs/run_20261117_115348_e1789b/detection_results.json")),
            output_directory=str(contracts.get("output_directory", "datasets/processed/driving_sample/reliability_labels")),
            output_object_labels=str(out_files.get("object_labels", "object_detection_labels.json")),
            output_frame_labels=str(out_files.get("frame_labels", "frame_reliability_labels.json")),
            output_temporal_targets=str(out_files.get("temporal_targets", "temporal_reliability_targets.json")),
            output_manifest=str(out_files.get("matching_manifest", "reliability_manifest.json")),
        )


# ==============================================================================
# SPATIAL HUNGARIAN MATCHER
# ==============================================================================

class SpatialHungarianMatcher:
    """Class-aware Hungarian (Munkres) bipartite assignment matcher."""

    def __init__(
        self,
        iou_threshold: float = 0.50,
        class_aware: bool = True,
        min_box_area: float = 4.0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.class_aware = class_aware
        self.min_box_area = min_box_area

    def match_frame(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform class-aware one-to-one matching between predictions and ground truths.
        
        Returns:
            matched_pairs: List of (pred_dict, gt_dict, iou_score)
            unmatched_predictions: List of pred_dict (False Positives)
            unmatched_ground_truth: List of gt_dict (False Negatives)
        """
        if len(predictions) == 0 and len(ground_truths) == 0:
            return [], [], []

        if len(predictions) == 0:
            return [], [], list(ground_truths)

        if len(ground_truths) == 0:
            return [], list(predictions), []

        # If class-aware, group objects by class name
        if self.class_aware:
            pred_classes = {p.get("class_name", "").lower().strip() for p in predictions}
            gt_classes = {g.get("class_name", g.get("mapped_category", "")).lower().strip() for g in ground_truths}
            all_classes = pred_classes.union(gt_classes)

            all_matched: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
            all_unmatched_preds: List[Dict[str, Any]] = []
            all_unmatched_gts: List[Dict[str, Any]] = []

            for cls_name in sorted(all_classes):
                cls_preds = [p for p in predictions if p.get("class_name", "").lower().strip() == cls_name]
                cls_gts = [g for g in ground_truths if g.get("class_name", g.get("mapped_category", "")).lower().strip() == cls_name]

                matched, un_preds, un_gts = self._bipartite_match_subgroup(cls_preds, cls_gts)
                all_matched.extend(matched)
                all_unmatched_preds.extend(un_preds)
                all_unmatched_gts.extend(un_gts)

            return all_matched, all_unmatched_preds, all_unmatched_gts
        else:
            return self._bipartite_match_subgroup(predictions, ground_truths)

    def _bipartite_match_subgroup(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform Hungarian matching within a single class group."""
        n_preds = len(predictions)
        n_gts = len(ground_truths)

        if n_preds == 0:
            return [], [], list(ground_truths)
        if n_gts == 0:
            return [], list(predictions), []

        # Compute pairwise IoU matrix [n_preds, n_gts]
        iou_matrix = np.zeros((n_preds, n_gts), dtype=np.float64)
        for i, p in enumerate(predictions):
            p_box = p.get("bbox_xyxy", [p.get("x_min", 0), p.get("y_min", 0), p.get("x_max", 0), p.get("y_max", 0)])
            for j, g in enumerate(ground_truths):
                g_box = g.get("bbox_xyxy", [g.get("x_min", 0), g.get("y_min", 0), g.get("x_max", 0), g.get("y_max", 0)])
                iou_matrix[i, j] = calculate_iou(p_box, g_box)

        # Cost matrix: 1.0 - IoU, penalize sub-threshold pairs with large constant
        cost_matrix = 1.0 - iou_matrix
        # Enforce threshold gate in cost matrix
        cost_matrix[iou_matrix < self.iou_threshold] = 1000.0

        # Hungarian assignment
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
        matched_pred_idx: Set[int] = set()
        matched_gt_idx: Set[int] = set()

        for pi, gi in zip(pred_indices, gt_indices):
            iou_val = float(iou_matrix[pi, gi])
            if iou_val >= self.iou_threshold:
                matched_pairs.append((predictions[pi], ground_truths[gi], iou_val))
                matched_pred_idx.add(pi)
                matched_gt_idx.add(gi)

        unmatched_preds = [p for i, p in enumerate(predictions) if i not in matched_pred_idx]
        unmatched_gts = [g for j, g in enumerate(ground_truths) if j not in matched_gt_idx]

        return matched_pairs, unmatched_preds, unmatched_gts


# ==============================================================================
# RELIABILITY LABEL GENERATOR PIPELINE
# ==============================================================================

class ReliabilityLabelGenerator:
    """Production-grade pipeline orchestrator for IoU-based reliability label generation."""

    def __init__(
        self,
        config: Optional[LabelingConfig] = None,
        matcher: Optional[SpatialHungarianMatcher] = None,
    ) -> None:
        self.config = config or LabelingConfig()
        self.matcher = matcher or SpatialHungarianMatcher(
            iou_threshold=self.config.iou_threshold,
            class_aware=self.config.class_aware,
            min_box_area=self.config.min_box_area_pixels,
        )

    def load_detections(self, detection_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """Load and index YOLO detections by frame filename."""
        p = Path(detection_path)
        if not p.is_file():
            raise FileNotFoundError(f"Detection results file not found at: '{p}'")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        frame_results = data.get("frame_results", [])
        indexed: Dict[str, Dict[str, Any]] = {}

        for fr in frame_results:
            fname = fr.get("image_filename", "")
            if not fname and "extracted_frame_id" in fr:
                fname = f"frame_{int(fr['extracted_frame_id']):06d}.jpg"
            if fname:
                # Filter detections by confidence and min area
                raw_dets = fr.get("detections", [])
                filtered_dets: List[Dict[str, Any]] = []
                for d in raw_dets:
                    conf = float(d.get("confidence", 0.0))
                    box = d.get("bbox_xyxy", [0, 0, 0, 0])
                    w = max(0.0, float(box[2]) - float(box[0]))
                    h = max(0.0, float(box[3]) - float(box[1]))
                    if conf >= self.config.min_detection_confidence and (w * h) >= self.config.min_box_area_pixels:
                        filtered_dets.append(d)
                
                fr_copy = dict(fr)
                fr_copy["detections"] = filtered_dets
                indexed[fname] = fr_copy

        return indexed

    def load_ground_truth(self, gt_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
        """Load and index canonical ground-truth annotations by frame filename."""
        p = Path(gt_path)
        if not p.is_file():
            # Check fallback to standardized json
            fallback = p.parent / "ground_truth_annotations.json"
            if fallback.is_file():
                p = fallback
            else:
                raise FileNotFoundError(f"Ground-truth annotations file not found at: '{p}'")

        indexed: Dict[str, List[Dict[str, Any]]] = {}

        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    fid = rec.get("frame_id", "")
                    if fid:
                        indexed.setdefault(fid, []).append(rec)
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            frames = data.get("frames", [])
            for fr in frames:
                fid = fr.get("frame_filename", "")
                if fid:
                    indexed[fid] = fr.get("annotations", [])

        return indexed

    def generate_frame_labels(
        self,
        detections: Dict[str, Dict[str, Any]],
        ground_truth: Dict[str, List[Dict[str, Any]]],
        total_frames: int = 200,
    ) -> Tuple[List[ObjectReliabilityRecord], List[FrameReliabilityRecord], Dict[str, Any]]:
        """Perform frame-by-frame matching and generate object and frame reliability records."""
        object_records: List[ObjectReliabilityRecord] = []
        frame_records: List[FrameReliabilityRecord] = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_preds = 0
        total_gts = 0

        for frame_idx in range(1, total_frames + 1):
            frame_id = f"frame_{frame_idx:06d}.jpg"
            source_frame_idx = (frame_idx - 1) * 6
            timestamp_sec = (frame_idx - 1) * (1.0 / self.config.fps)

            frame_det_data = detections.get(frame_id, {})
            frame_preds = frame_det_data.get("detections", [])
            frame_gts = ground_truth.get(frame_id, [])

            total_preds += len(frame_preds)
            total_gts += len(frame_gts)

            # Match frame
            matched_pairs, unmatched_preds, unmatched_gts = self.matcher.match_frame(
                predictions=frame_preds,
                ground_truths=frame_gts,
            )

            matched_ious: List[float] = []

            # 1. Process True Positives
            for p, g, iou in matched_pairs:
                matched_ious.append(iou)
                pred_cls = str(p.get("class_name", ""))
                gt_cls = str(g.get("class_name", g.get("mapped_category", "")))
                obj_rec = ObjectReliabilityRecord(
                    frame_id=frame_id,
                    extracted_frame_id=frame_idx,
                    source_frame_index=source_frame_idx,
                    timestamp_seconds=timestamp_sec,
                    detection_index=p.get("detection_index"),
                    ground_truth_id=str(g.get("annotation_id", g.get("label_id", ""))),
                    track_id=str(g.get("ground_truth_object_id", g.get("track_id", ""))) if g.get("ground_truth_object_id") or g.get("track_id") else None,
                    predicted_class=pred_cls,
                    ground_truth_class=gt_cls,
                    confidence=float(p.get("confidence", 0.0)),
                    iou=float(iou),
                    outcome=MatchingOutcome.TRUE_POSITIVE,
                    object_correctness=1.0,
                )
                object_records.append(obj_rec)

            # 2. Process False Positives (unmatched predictions or duplicates)
            for p in unmatched_preds:
                pred_cls = str(p.get("class_name", ""))
                obj_rec = ObjectReliabilityRecord(
                    frame_id=frame_id,
                    extracted_frame_id=frame_idx,
                    source_frame_index=source_frame_idx,
                    timestamp_seconds=timestamp_sec,
                    detection_index=p.get("detection_index"),
                    ground_truth_id=None,
                    track_id=None,
                    predicted_class=pred_cls,
                    ground_truth_class=None,
                    confidence=float(p.get("confidence", 0.0)),
                    iou=0.0,
                    outcome=MatchingOutcome.FALSE_POSITIVE,
                    object_correctness=0.0,
                )
                object_records.append(obj_rec)

            # 3. Process False Negatives (unmatched ground truth objects)
            for g in unmatched_gts:
                gt_cls = str(g.get("class_name", g.get("mapped_category", "")))
                obj_rec = ObjectReliabilityRecord(
                    frame_id=frame_id,
                    extracted_frame_id=frame_idx,
                    source_frame_index=source_frame_idx,
                    timestamp_seconds=timestamp_sec,
                    detection_index=None,
                    ground_truth_id=str(g.get("annotation_id", g.get("label_id", ""))),
                    track_id=str(g.get("ground_truth_object_id", g.get("track_id", ""))) if g.get("ground_truth_object_id") or g.get("track_id") else None,
                    predicted_class=None,
                    ground_truth_class=gt_cls,
                    confidence=None,
                    iou=0.0,
                    outcome=MatchingOutcome.FALSE_NEGATIVE,
                    object_correctness=0.0,
                )
                object_records.append(obj_rec)

            tp_cnt = len(matched_pairs)
            fp_cnt = len(unmatched_preds)
            fn_cnt = len(unmatched_gts)

            total_tp += tp_cnt
            total_fp += fp_cnt
            total_fn += fn_cnt

            prec, rec, f1 = compute_frame_f1(
                true_positives=tp_cnt,
                false_positives=fp_cnt,
                false_negatives=fn_cnt,
                total_predictions=len(frame_preds),
                total_ground_truth=len(frame_gts),
            )

            mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0
            comp_rel = compute_composite_reliability(
                f1_score=f1,
                mean_matched_iou=mean_iou,
                alpha=self.config.f1_weight,
            )

            frame_rec = FrameReliabilityRecord(
                frame_id=frame_id,
                extracted_frame_id=frame_idx,
                source_frame_index=source_frame_idx,
                timestamp_seconds=timestamp_sec,
                total_predictions=len(frame_preds),
                total_ground_truth=len(frame_gts),
                true_positives=tp_cnt,
                false_positives=fp_cnt,
                false_negatives=fn_cnt,
                precision=prec,
                recall=rec,
                f1_score=f1,
                mean_matched_iou=mean_iou,
                composite_reliability=comp_rel,
                primary_reliability=f1,
                status="valid",
            )
            frame_records.append(frame_rec)

        summary_metrics = {
            "total_frames_processed": total_frames,
            "total_ground_truth_objects": total_gts,
            "total_predictions": total_preds,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "overall_precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0,
            "overall_recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0,
            "overall_f1_score": round(2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn), 4) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0,
        }

        return object_records, frame_records, summary_metrics

    def generate_temporal_targets(
        self,
        frame_records: List[FrameReliabilityRecord],
        sequence_metadata_path: Union[str, Path],
    ) -> Tuple[List[TemporalTargetRecord], Dict[str, Any]]:
        """Construct multi-horizon temporal targets aligned with sequence metadata."""
        seq_path = Path(sequence_metadata_path)
        frame_map = {fr.extracted_frame_id: fr for fr in frame_records}
        temporal_records: List[TemporalTargetRecord] = []

        h_counts = {1: 0, 3: 0, 5: 0}
        updated_seq_data: Dict[str, Any] = {}

        if seq_path.is_file():
            with open(seq_path, "r", encoding="utf-8") as f:
                updated_seq_data = json.load(f)

            sequences = updated_seq_data.get("sequences", [])
            for seq in sequences:
                seq_id = seq.get("sequence_id", "")
                ds_id = seq.get("dataset_id", "driving_sample")
                split = seq.get("split", "train")
                end_fid_str = seq.get("end_frame_id", "frame_000005.jpg")
                try:
                    end_fid_num = int(end_fid_str.replace("frame_", "").replace(".jpg", ""))
                except ValueError:
                    end_fid_num = 5

                horizons_spec: Dict[str, HorizonTargetSpec] = {}

                # Inspect and update each horizon
                for h_step in self.config.horizons:
                    h_key = f"h{h_step}"
                    target_fid_num = end_fid_num + h_step
                    target_fid_str = f"frame_{target_fid_num:06d}.jpg"

                    if target_fid_num in frame_map:
                        target_fr = frame_map[target_fid_num]
                        target_val = target_fr.primary_reliability
                        t_status = TargetStatus.VALID
                        h_counts[h_step] += 1
                    else:
                        target_val = None
                        t_status = TargetStatus.OUT_OF_BOUNDS

                    h_spec = HorizonTargetSpec(
                        horizon_step=h_step,
                        target_frame_id=target_fid_str if t_status == TargetStatus.VALID else None,
                        target_source_frame_index=(target_fid_num - 1) * 6 if t_status == TargetStatus.VALID else None,
                        target_timestamp_seconds=(target_fid_num - 1) * (1.0 / self.config.fps) if t_status == TargetStatus.VALID else None,
                        target_reliability=target_val,
                        target_status=t_status,
                    )
                    horizons_spec[h_key] = h_spec

                    # Update raw sequence metadata dict if present
                    if "horizons" in seq and h_key in seq["horizons"]:
                        seq["horizons"][h_key]["target_label"] = round(target_val, 4) if target_val is not None else None
                        seq["horizons"][h_key]["target_label_available"] = (target_val is not None)
                        if t_status != TargetStatus.VALID:
                            seq["horizons"][h_key]["target_status"] = t_status.value

                t_rec = TemporalTargetRecord(
                    sequence_id=seq_id,
                    dataset_id=ds_id,
                    source_frame_id=end_fid_str,
                    source_extracted_frame_id=end_fid_num,
                    source_timestamp_seconds=(end_fid_num - 1) * (1.0 / self.config.fps),
                    split=split,
                    horizons=horizons_spec,
                )
                temporal_records.append(t_rec)

            # Sync sequences_by_length mapping so downstream loaders obtain target labels
            if "sequences_by_length" in updated_seq_data:
                by_len: Dict[str, List[Dict[str, Any]]] = {}
                for s in sequences:
                    l_val = s.get("input_sequence_length", 5)
                    by_len.setdefault(f"L{l_val}", []).append(s)
                updated_seq_data["sequences_by_length"] = by_len

        return temporal_records, updated_seq_data

    def export_artifacts(
        self,
        object_records: List[ObjectReliabilityRecord],
        frame_records: List[FrameReliabilityRecord],
        temporal_records: List[TemporalTargetRecord],
        summary_metrics: Dict[str, Any],
        output_dir: Union[str, Path],
        updated_seq_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Atomically export all generated reliability label artifacts."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exported_paths: Dict[str, Path] = {}

        # 1. Export Object Reliability Labels (JSON and JSONL)
        obj_json_path = out_dir / self.config.output_object_labels
        self._atomic_json_write(obj_json_path, [o.to_dict() for o in object_records])
        exported_paths["object_labels_json"] = obj_json_path

        obj_jsonl_path = out_dir / self.config.output_object_labels.replace(".json", ".jsonl")
        self._atomic_jsonl_write(obj_jsonl_path, [o.to_dict() for o in object_records])
        exported_paths["object_labels_jsonl"] = obj_jsonl_path

        # 2. Export Frame Reliability Labels
        frame_json_path = out_dir / self.config.output_frame_labels
        self._atomic_json_write(frame_json_path, [f.to_dict() for f in frame_records])
        exported_paths["frame_labels"] = frame_json_path

        # 3. Export Temporal Reliability Targets
        temp_json_path = out_dir / self.config.output_temporal_targets
        self._atomic_json_write(temp_json_path, [t.to_dict() for t in temporal_records])
        exported_paths["temporal_targets"] = temp_json_path

        # 4. Export Reliability Manifest
        def get_rel_str(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
            except ValueError:
                return str(p).replace("\\", "/")

        manifest = {
            "schema_version": self.config.schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "PerceptGuard-AI Reliability Label Generation",
            "matching_configuration": {
                "strategy": "class_aware_bipartite",
                "iou_threshold": self.config.iou_threshold,
                "class_aware": self.config.class_aware,
                "min_detection_confidence": self.config.min_detection_confidence,
                "primary_metric": self.config.primary_metric,
                "composite_weights": {
                    "f1": self.config.f1_weight,
                    "mean_iou": self.config.mean_iou_weight,
                },
                "temporal_horizons": self.config.horizons,
            },
            "summary_metrics": summary_metrics,
            "total_object_labels": len(object_records),
            "total_frame_labels": len(frame_records),
            "total_temporal_targets": len(temporal_records),
            "relative_paths": {
                "object_labels_json": get_rel_str(obj_json_path),
                "frame_labels_json": get_rel_str(frame_json_path),
                "temporal_targets_json": get_rel_str(temp_json_path),
            }
        }
        manifest_path = out_dir / self.config.output_manifest
        self._atomic_json_write(manifest_path, manifest)
        exported_paths["manifest"] = manifest_path

        # 5. Export Updated Sequence Metadata if available
        if updated_seq_data:
            seq_out_path = out_dir / "sequence_metadata_labeled.json"
            self._atomic_json_write(seq_out_path, updated_seq_data)
            exported_paths["sequence_metadata_labeled"] = seq_out_path

        return exported_paths

    def _atomic_json_write(self, target_path: Path, data: Any) -> None:
        """Atomically write data to target path using a temporary file."""
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="rel_tmp_", suffix=".json", dir=str(target_path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if os.path.exists(target_path):
                os.replace(tmp_path, target_path)
            else:
                os.rename(tmp_path, target_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _atomic_jsonl_write(self, target_path: Path, records: List[Dict[str, Any]]) -> None:
        """Atomically write records to target JSONL path using a temporary file."""
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="rel_tmp_", suffix=".jsonl", dir=str(target_path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            if os.path.exists(target_path):
                os.replace(tmp_path, target_path)
            else:
                os.rename(tmp_path, target_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================

def main() -> None:
    """CLI execution entrypoint for Reliability Label Generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard-AI: Ground-Truth Reliability Label Generator (Day 8 Task 5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/reliability_label_config.yaml",
        help="Path to reliability label configuration YAML",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum frames to process (default: 200)",
    )
    parser.add_argument(
        "--bounded",
        type=int,
        default=None,
        help="Run on bounded sample of N frames for fast verification",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Perform matching and print metrics without writing output files",
    )

    args = parser.parse_args()
    limit_frames = args.bounded if args.bounded is not None else args.max_frames

    print("=" * 80)
    print("  PERCEPTGUARD-AI: RELIABILITY LABEL GENERATOR (DAY 8 TASK 5)")
    print("=" * 80)
    print(f"  Configuration:     {args.config}")
    print(f"  Max Frames:        {limit_frames}")
    print(f"  Validate Only:     {args.validate_only}")
    print("-" * 80)

    config = LabelingConfig.from_yaml(args.config)
    generator = ReliabilityLabelGenerator(config=config)

    print(f"  Loading Detections:   {config.input_detections}")
    detections = generator.load_detections(config.input_detections)
    print(f"  [OK] Loaded Detections for {len(detections)} frames.")

    print(f"  Loading Ground Truth: {config.input_ground_truth}")
    ground_truth = generator.load_ground_truth(config.input_ground_truth)
    print(f"  [OK] Loaded Ground Truth for {len(ground_truth)} frames.")

    start_time = time.time()
    obj_records, frame_records, summary_metrics = generator.generate_frame_labels(
        detections=detections,
        ground_truth=ground_truth,
        total_frames=limit_frames,
    )

    temporal_records, updated_seqs = generator.generate_temporal_targets(
        frame_records=frame_records,
        sequence_metadata_path=config.sequence_metadata_path,
    )
    duration = time.time() - start_time

    print("-" * 80)
    print(f"  Frames Evaluated:   {summary_metrics['total_frames_processed']}")
    print(f"  Total Ground Truth: {summary_metrics['total_ground_truth_objects']}")
    print(f"  Total Predictions:  {summary_metrics['total_predictions']}")
    print(f"  True Positives (TP):{summary_metrics['total_true_positives']}")
    print(f"  False Positives(FP):{summary_metrics['total_false_positives']}")
    print(f"  False Negatives(FN):{summary_metrics['total_false_negatives']}")
    print(f"  Overall Precision:  {summary_metrics['overall_precision']:.4f}")
    print(f"  Overall Recall:     {summary_metrics['overall_recall']:.4f}")
    print(f"  Overall F1-Score:   {summary_metrics['overall_f1_score']:.4f}")
    print(f"  Temporal Targets:   {len(temporal_records)} sequences labeled")
    print(f"  Execution Duration: {duration:.4f} seconds")
    print("-" * 80)

    if not args.validate_only:
        paths = generator.export_artifacts(
            object_records=obj_records,
            frame_records=frame_records,
            temporal_records=temporal_records,
            summary_metrics=summary_metrics,
            output_dir=config.output_directory,
            updated_seq_data=updated_seqs,
        )
        for k, p in paths.items():
            print(f"  [OK] Exported {k}: {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
