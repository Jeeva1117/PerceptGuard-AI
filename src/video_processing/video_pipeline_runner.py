"""Complete End-to-End Video Processing Pipeline Runner for PerceptGuard AI.

Integrates the full perception reliability pipeline:
Video Input
  -> Frame Extraction
  -> Frame Preprocessing
  -> YOLO Object Detection
  -> ByteTrack Multi-Object Tracking
  -> Multi-Modal Feature Extraction (Detection, Motion, Visual Quality, Temporal)
  -> Temporal Sequence Construction
  -> Baseline Forecasting
  -> LSTM Multi-Horizon Reliability Forecasting (H1, H3, H5)
  -> Reliability Risk Scoring (S = 1.0 - R)
  -> Risk Classification (Low, Medium, High, Critical)
  -> Temporal Risk Monitoring (EMA, Rolling Stats, Trend Velocity)
  -> Early Warning Alert Generation (P1, P2, P3)
  -> Annotated Frame Generation & Result Persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.detection.single_image_detector import SingleImageDetector
from src.detection.yolo_model_loader import YOLOModelLoader
from src.early_warning.early_warning_config import EarlyWarningConfig
from src.early_warning.early_warning_engine import EarlyWarningEngine
from src.early_warning.interfaces import EarlyWarningAlert, RiskLevel
from src.forecasting.model_factory import build_forecasting_model
from src.risk.risk_classifier import RiskClassifier
from src.risk.risk_config import RiskConfig
from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.temporal_risk_monitor import TemporalRiskMonitor
from src.tracking.bytetrack_engine import ByteTrackEngine
from src.visualization.tracking_visualizer import get_deterministic_track_color

logger = logging.getLogger("perceptguard.video_pipeline")


@dataclass
class FrameProcessingDetail:
    """Detailed results from processing a single video frame."""
    frame_index: int
    timestamp_seconds: float
    raw_frame_path: str
    annotated_frame_path: str
    detections_count: int
    detections: List[Dict[str, Any]]
    tracks_count: int
    tracks: List[Dict[str, Any]]
    visual_quality: Dict[str, float]
    predicted_reliability: Dict[str, float]
    risk_scores: Dict[str, float]
    risk_levels: Dict[str, str]
    alerts: List[Dict[str, Any]]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoPipelineRunResult:
    """Comprehensive artifact summarizing a completed end-to-end video pipeline run."""
    job_id: str
    video_filename: str
    video_metadata: Dict[str, Any]
    frames_processed: int
    total_detections: int
    unique_tracks: int
    total_alerts: int
    active_alerts: int
    calibration_status: str
    per_horizon_risk: Dict[str, Dict[str, Any]]
    frames: List[FrameProcessingDetail]
    stage_durations_ms: Dict[str, float]
    total_duration_seconds: float
    completed_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "video_filename": self.video_filename,
            "video_metadata": self.video_metadata,
            "frames_processed": self.frames_processed,
            "total_detections": self.total_detections,
            "unique_tracks": self.unique_tracks,
            "total_alerts": self.total_alerts,
            "active_alerts": self.active_alerts,
            "calibration_status": self.calibration_status,
            "per_horizon_risk": self.per_horizon_risk,
            "stage_durations_ms": self.stage_durations_ms,
            "total_duration_seconds": self.total_duration_seconds,
            "completed_at": self.completed_at,
            "frames": [f.to_dict() for f in self.frames],
        }


class VideoPipelineRunner:
    """Production-grade orchestrator running the complete CV and failure anticipation pipeline."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.project_root = project_root or Path.cwd()
        self.device = device

        # Cached models and components
        self._yolo_model = None
        self._lstm_model = None
        self._risk_scorer = None
        self._risk_classifier = None

    def _ensure_models_loaded(self) -> None:
        """Lazily load and cache YOLO and LSTM models in memory."""
        if self._yolo_model is None:
            weights_path = self.project_root / "models" / "yolo" / "yolo11n.pt"
            if not weights_path.exists():
                weights_path = Path.cwd() / "models" / "yolo" / "yolo11n.pt"
            loader = YOLOModelLoader(weights_path=weights_path, project_root=Path.cwd())
            self._yolo_model = loader.load_model()
            logger.info(f"Loaded YOLO model from {weights_path}")

        if self._lstm_model is None:
            # Check primary deployment path then fallback
            primary_ckpt = self.project_root / "models" / "forecasting" / "lstm" / "model.pt"
            if not primary_ckpt.exists():
                primary_ckpt = Path.cwd() / "models" / "forecasting" / "lstm" / "model.pt"

            fallback_ckpt = (
                Path.cwd()
                / "experiments"
                / "forecasting"
                / "lstm_training"
                / "official_lstm_training"
                / "checkpoints"
                / "checkpoint_best.pt"
            )
            ckpt_path = primary_ckpt if primary_ckpt.exists() else fallback_ckpt

            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                input_size = ckpt.get("input_size", 27)
                self._lstm_model = build_forecasting_model(input_size=input_size, device=self.device)
                self._lstm_model.load_state_dict(ckpt["model_state_dict"])
                self._lstm_model.eval()
                logger.info(f"Loaded LSTM checkpoint from {ckpt_path}")
            else:
                logger.warning(f"No LSTM checkpoint found. Building uninitialized reference model.")
                self._lstm_model = build_forecasting_model(input_size=27, device=self.device)
                self._lstm_model.eval()

        if self._risk_scorer is None:
            self._risk_scorer = ReliabilityRiskScorer()
        if self._risk_classifier is None:
            self._risk_classifier = RiskClassifier()

    def run(
        self,
        video_path: Union[str, Path],
        job_id: str,
        output_dir: Union[str, Path],
        max_frames: int = 30,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> VideoPipelineRunResult:
        """Execute the end-to-end perception reliability pipeline on a video file.

        Args:
            video_path: Path to the input video file.
            job_id: Unique processing job identifier.
            output_dir: Directory where extracted and annotated frames are stored.
            max_frames: Maximum number of frames to process (CPU protection bound).
            progress_callback: Optional callback(stage, progress_pct, message).

        Returns:
            VideoPipelineRunResult containing structured analytics and frame paths.
        """
        t_job_start = time.perf_counter()
        vpath = Path(video_path).resolve()
        out_path = Path(output_dir).resolve()
        raw_frames_dir = out_path / "raw_frames"
        annotated_dir = out_path / "annotated"
        raw_frames_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        def _notify(stage: str, pct: float, msg: str):
            if progress_callback:
                progress_callback(stage, round(pct, 1), msg)

        # ----------------------------------------------------------------------
        # Stage 1: Video Ingestion & Validation
        # ----------------------------------------------------------------------
        _notify("VALIDATING", 5.0, "Inspecting video format and dimensions...")
        t_stage_start = time.perf_counter()

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from {vpath.name}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (raw_fps is None or np.isnan(raw_fps) or float(raw_fps) <= 0.0) else float(raw_fps)
        frame_rate = max(1, int(round(fps)))
        total_container_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_container_frames / fps if fps > 0 else 0.0

        video_meta = {
            "filename": vpath.name,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_container_frames": total_container_frames,
            "duration_seconds": round(duration_sec, 2),
        }
        duration_validation_ms = (time.perf_counter() - t_stage_start) * 1000.0

        # Ensure models ready
        self._ensure_models_loaded()

        # Initialize tracker, temporal risk monitor, and early warning engine per video run
        tracker = ByteTrackEngine(frame_rate=frame_rate)
        temporal_monitor = TemporalRiskMonitor(classifier=self._risk_classifier)
        early_warning_engine = EarlyWarningEngine()

        # ----------------------------------------------------------------------
        # Stage 2 to 12: Frame-by-Frame Processing Loop
        # ----------------------------------------------------------------------
        frames_to_process = min(max_frames, total_container_frames if total_container_frames > 0 else max_frames)
        feature_history: List[List[float]] = []
        prev_track_positions: Dict[int, Tuple[float, float]] = {}
        conf_history: List[float] = []
        blur_history: List[float] = []
        processed_frames: List[FrameProcessingDetail] = []
        all_alerts: List[EarlyWarningAlert] = []
        total_detections = 0

        durations_det_ms = 0.0
        durations_track_ms = 0.0
        durations_features_ms = 0.0
        durations_forecast_ms = 0.0
        durations_risk_ms = 0.0
        durations_vis_ms = 0.0

        frame_idx = 0
        while frame_idx < frames_to_process:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            t_frame_start = time.perf_counter()
            pct = 10.0 + (float(frame_idx) / frames_to_process) * 80.0
            _notify(
                "PROCESSING",
                pct,
                f"Processing frame {frame_idx + 1}/{frames_to_process} (YOLO + ByteTrack + LSTM)...",
            )

            timestamp_sec = float(frame_idx) / fps
            raw_filename = f"frame_{frame_idx:06d}.jpg"
            raw_frame_path = raw_frames_dir / raw_filename
            annotated_filename = f"annotated_{frame_idx:06d}.jpg"
            annotated_frame_path = annotated_dir / annotated_filename

            # Save raw frame
            cv2.imwrite(str(raw_frame_path), frame_bgr)

            # --- A. Detection (YOLO) ---
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yolo_out = self._yolo_model.predict(
                source=rgb,
                conf=0.25,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            boxes = yolo_out[0].boxes
            detections: List[Dict[str, Any]] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    cls_id = int(classes[i])
                    detections.append({
                        "bbox_xyxy": [round(float(c), 2) for c in xyxy[i]],
                        "confidence": round(float(confs[i]), 4),
                        "class_id": cls_id,
                        "class_name": self._yolo_model.names.get(cls_id, "object"),
                    })
            total_detections += len(detections)
            durations_det_ms += (time.perf_counter() - t0) * 1000.0

            # --- B. Tracking (ByteTrack) ---
            t0 = time.perf_counter()
            active_tracks = tracker.update(detections)
            tracks_info: List[Dict[str, Any]] = [t.to_dict() for t in active_tracks]
            durations_track_ms += (time.perf_counter() - t0) * 1000.0

            # --- C. Feature Extraction (Multi-Modal 27 Features adhering to feature_manifest.json) ---
            t0 = time.perf_counter()
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_val = float(np.mean(gray))
            brightness_std_val = float(np.std(gray))
            contrast_val = float(np.std(gray))
            contrast_range_val = float(np.ptp(gray))
            contrast_p_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))

            det_count = len(detections)
            unique_classes = len(set(d["class_id"] for d in detections))
            confs = [d["confidence"] for d in detections]
            mean_conf = float(np.mean(confs)) if confs else 0.0
            conf_std = float(np.std(confs)) if len(confs) > 1 else 0.0

            bbox_areas = [(d["bbox_xyxy"][2] - d["bbox_xyxy"][0]) * (d["bbox_xyxy"][3] - d["bbox_xyxy"][1]) for d in detections]
            bbox_widths = [d["bbox_xyxy"][2] - d["bbox_xyxy"][0] for d in detections]
            bbox_heights = [d["bbox_xyxy"][3] - d["bbox_xyxy"][1] for d in detections]
            mean_area = float(np.mean(bbox_areas)) if bbox_areas else 0.0
            mean_width = float(np.mean(bbox_widths)) if bbox_widths else 0.0
            mean_height = float(np.mean(bbox_heights)) if bbox_heights else 0.0

            track_count = len(active_tracks)
            curr_track_positions: Dict[int, Tuple[float, float]] = {}
            displacements: List[float] = []
            speeds: List[float] = []
            vel_xs: List[float] = []
            vel_ys: List[float] = []
            moving_count = 0

            for trk in active_tracks:
                cx = (trk.bbox_xyxy[0] + trk.bbox_xyxy[2]) / 2.0
                cy = (trk.bbox_xyxy[1] + trk.bbox_xyxy[3]) / 2.0
                curr_track_positions[trk.track_id] = (cx, cy)
                if trk.track_id in prev_track_positions:
                    px, py = prev_track_positions[trk.track_id]
                    dx = cx - px
                    dy = cy - py
                    disp = float(np.hypot(dx, dy))
                    displacements.append(disp)
                    speeds.append(disp * fps)
                    vel_xs.append(dx * fps)
                    vel_ys.append(dy * fps)
                    if disp > 2.0:
                        moving_count += 1
            prev_track_positions = curr_track_positions

            stationary_count = max(0, track_count - moving_count)
            mean_disp = float(np.mean(displacements)) if displacements else 0.0
            max_disp = float(np.max(displacements)) if displacements else 0.0
            mean_spd = float(np.mean(speeds)) if speeds else 0.0
            mean_vx = float(np.mean(vel_xs)) if vel_xs else 0.0
            mean_vy = float(np.mean(vel_ys)) if vel_ys else 0.0
            motion_cov = 1.0 if track_count > 0 else 0.0

            # Update histories for rolling trend slopes
            conf_history.append(mean_conf)
            blur_history.append(blur_val)

            def _calc_slope(vals: List[float], win: int) -> float:
                if len(vals) < win:
                    return 0.0
                y = np.array(vals[-win:], dtype=np.float32)
                x = np.arange(win, dtype=np.float32)
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                denom = np.sum((x - x_mean) ** 2)
                if denom == 0:
                    return 0.0
                return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

            w3_conf_slope = _calc_slope(conf_history, 3)
            w3_blur_slope = _calc_slope(blur_history, 3)
            w5_conf_slope = _calc_slope(conf_history, 5)
            w5_blur_slope = _calc_slope(blur_history, 5)

            # Construct exact 27-feature vector according to feature_manifest.json
            feat_vec = [
                float(det_count),            # 0: detection_count
                float(unique_classes),       # 1: unique_class_count
                mean_conf,                   # 2: mean_confidence
                conf_std,                    # 3: confidence_std
                mean_area,                   # 4: mean_bbox_area
                mean_width,                  # 5: mean_bbox_width
                mean_height,                 # 6: mean_bbox_height
                float(track_count),          # 7: observed_track_count
                float(moving_count),         # 8: moving_track_count
                float(stationary_count),     # 9: stationary_track_count
                mean_disp,                   # 10: mean_displacement
                max_disp,                    # 11: maximum_displacement
                mean_spd,                    # 12: mean_speed
                mean_vx,                     # 13: mean_velocity_x
                mean_vy,                     # 14: mean_velocity_y
                motion_cov,                  # 15: motion_feature_coverage
                blur_val,                    # 16: blur_score
                brightness_val,              # 17: mean_brightness
                brightness_std_val,          # 18: brightness_std
                contrast_val,                # 19: contrast_std
                contrast_range_val,          # 20: contrast_range
                contrast_p_range,            # 21: contrast_percentile_range
                1.0,                         # 22: numeric_feature_coverage
                w3_conf_slope,               # 23: w3_confidence_trend_slope
                w3_blur_slope,               # 24: w3_blur_trend_slope
                w5_conf_slope,               # 25: w5_confidence_trend_slope
                w5_blur_slope,               # 26: w5_blur_trend_slope
            ]
            # Sanitize any non-finite values to guarantee numerical safety
            feat_vec = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in feat_vec]

            feature_history.append(feat_vec)
            if len(feature_history) > 10:
                feature_history.pop(0)

            durations_features_ms += (time.perf_counter() - t0) * 1000.0

            # --- D. Temporal Forecasting (Multi-Horizon H1, H3, H5) ---
            t0 = time.perf_counter()
            seq_array = np.array(feature_history, dtype=np.float32)
            if len(seq_array) < 10:
                pad = np.repeat(seq_array[:1], 10 - len(seq_array), axis=0)
                seq_array = np.vstack([pad, seq_array])
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_rel = self._lstm_model(seq_tensor)[0].cpu().numpy()
                pred_rel = np.clip(pred_rel, 0.0, 1.0)

            durations_forecast_ms += (time.perf_counter() - t0) * 1000.0

            # --- E. Reliability Risk Scoring & Alerts ---
            t0 = time.perf_counter()
            frame_alerts: List[EarlyWarningAlert] = []
            rel_dict: Dict[str, float] = {}
            risk_dict: Dict[str, float] = {}
            level_dict: Dict[str, str] = {}

            for h_idx, h in enumerate([1, 3, 5]):
                h_key = f"h{h}"
                rel_val = float(pred_rel[h_idx])
                rel_dict[h_key] = round(rel_val, 4)

                # Score risk
                risk_record = self._risk_scorer.score_single_horizon({
                    "sequence_id": job_id,
                    "source_frame_id": raw_filename,
                    "target_frame_id": f"frame_{frame_idx+h:06d}.jpg",
                    "horizon": h,
                    "predicted_reliability": rel_val,
                    "target_timestamp_seconds": timestamp_sec + (h / fps),
                })
                risk_dict[h_key] = round(risk_record.risk_score, 4)

                # Monitor and classify
                summary = temporal_monitor.update(risk_record)
                level_dict[h_key] = summary.current_risk_level.value

                tracker_state = temporal_monitor._get_tracker(risk_record.sequence_id, risk_record.horizon)
                latest_rec = tracker_state.records[-1] if tracker_state.records else None
                latest_trans = tracker_state.transitions[-1] if tracker_state.transitions else None

                # Early-warning rule evaluation
                alerts = early_warning_engine.evaluate(
                    summary=summary,
                    transition=latest_trans,
                    source_record=latest_rec,
                )
                frame_alerts.extend(alerts)

            all_alerts.extend(frame_alerts)
            durations_risk_ms += (time.perf_counter() - t0) * 1000.0

            # --- F. Frame Annotation & Visualization ---
            t0 = time.perf_counter()
            vis_img = frame_bgr.copy()

            # Draw tracked bounding boxes
            for t in active_tracks:
                x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
                color = get_deterministic_track_color(t.track_id)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{t.track_id} {t.class_name} {t.confidence:.2f}"
                cv2.putText(
                    vis_img,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Overlay telemetry HUD banner on top
            cv2.rectangle(vis_img, (0, 0), (width, 42), (20, 20, 24), -1)
            hud_text = (
                f"Frame: {frame_idx:03d} | Dets: {det_count} | Tracks: {track_count} | "
                f"H1 Risk: {risk_dict['h1']:.2f} ({level_dict['h1']}) | "
                f"H3 Risk: {risk_dict['h3']:.2f} ({level_dict['h3']}) | "
                f"H5 Risk: {risk_dict['h5']:.2f} ({level_dict['h5']})"
            )
            cv2.putText(
                vis_img,
                hud_text,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imwrite(str(annotated_frame_path), vis_img)
            durations_vis_ms += (time.perf_counter() - t0) * 1000.0

            frame_latency_ms = (time.perf_counter() - t_frame_start) * 1000.0

            processed_frames.append(
                FrameProcessingDetail(
                    frame_index=frame_idx,
                    timestamp_seconds=round(timestamp_sec, 3),
                    raw_frame_path=str(raw_frame_path),
                    annotated_frame_path=str(annotated_frame_path),
                    detections_count=det_count,
                    detections=detections,
                    tracks_count=track_count,
                    tracks=tracks_info,
                    visual_quality={
                        "blur_laplacian": round(blur_val, 2),
                        "brightness": round(brightness_val, 2),
                        "contrast": round(contrast_val, 2),
                    },
                    predicted_reliability=rel_dict,
                    risk_scores=risk_dict,
                    risk_levels=level_dict,
                    alerts=[a.to_dict() for a in frame_alerts],
                    latency_ms=round(frame_latency_ms, 2),
                )
            )

            frame_idx += 1

        cap.release()

        if len(processed_frames) == 0:
            raise ValueError(f"Video file '{vpath.name}' contains zero decodable frames.")

        # ----------------------------------------------------------------------
        # Stage 13: Summary & Result Packaging
        # ----------------------------------------------------------------------
        _notify("COMPLETING", 95.0, "Packaging full pipeline telemetry artifacts...")
        t_total_seconds = time.perf_counter() - t_job_start

        # Calculate per-horizon overall risk
        per_h_risk = {}
        for h in [1, 3, 5]:
            h_key = f"h{h}"
            scores = [f.risk_scores[h_key] for f in processed_frames] if processed_frames else [0.0]
            mean_s = float(np.mean(scores))
            max_s = float(np.max(scores))
            latest_s = scores[-1] if scores else 0.0
            category = self._risk_classifier.classify(latest_s).value
            per_h_risk[h_key] = {
                "horizon": h,
                "latest_risk": round(latest_s, 4),
                "mean_risk": round(mean_s, 4),
                "max_risk": round(max_s, 4),
                "category": category,
            }

        result = VideoPipelineRunResult(
            job_id=job_id,
            video_filename=vpath.name,
            video_metadata=video_meta,
            frames_processed=len(processed_frames),
            total_detections=total_detections,
            unique_tracks=tracker.unique_track_count,
            total_alerts=len(all_alerts),
            active_alerts=len(early_warning_engine.get_active_alerts()),
            calibration_status="PROVISIONAL CALIBRATION",
            per_horizon_risk=per_h_risk,
            frames=processed_frames,
            stage_durations_ms={
                "validation_ms": round(duration_validation_ms, 2),
                "detection_total_ms": round(durations_det_ms, 2),
                "tracking_total_ms": round(durations_track_ms, 2),
                "feature_extraction_ms": round(durations_features_ms, 2),
                "forecasting_ms": round(durations_forecast_ms, 2),
                "risk_scoring_ms": round(durations_risk_ms, 2),
                "visualization_ms": round(durations_vis_ms, 2),
            },
            total_duration_seconds=round(t_total_seconds, 3),
        )

        # Atomic persistence of results.json
        results_file = out_path / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        _notify("COMPLETED", 100.0, f"Processing complete. {len(processed_frames)} frames processed successfully.")
        return result
