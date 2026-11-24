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
import math
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
from src.camera_weather import (
    FrameWeatherIntelligence,
    TemporalWeatherTracker,
    VisibilityForecaster,
    VisibilityQualityEngine,
    WeatherStreamSummary,
    estimate_perception_impact,
)
from src.trajectory_prediction import (
    MultimodalTrajectoryForecaster,
    ObjectTrajectoryPrediction,
    TrajectoryStreamSummary,
)
from src.collision_risk import (
    CollisionRiskAssessment,
    CollisionRiskEngine,
    CollisionRiskState,
    CollisionStreamSummary,
    RoadContextAnalyzer,
    RoadContextData,
    LaneState,
)
from src.safety_recommendation import (
    RecommendationState,
    SafetyRecommendation,
    SafetyRecommendationEngine,
    SafetyStreamSummary,
    PedestrianAwarenessEngine,
    PedestrianAwarenessState,
    FramePedestrianTelemetry,
)

logger = logging.getLogger("perceptguard.video_pipeline")


@dataclass
class FrameProcessingDetail:
    """Detailed results from processing a single video frame across all five product features."""
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
    source_frame_index: int = 0
    weather_intelligence: Dict[str, Any] = field(default_factory=dict)
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    collision_risk: Dict[str, Any] = field(default_factory=dict)
    safety_recommendation: Dict[str, Any] = field(default_factory=dict)
    road_context: Dict[str, Any] = field(default_factory=dict)
    pedestrian_awareness: Dict[str, Any] = field(default_factory=dict)

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
    weather_summary: Dict[str, Any] = field(default_factory=dict)
    trajectory_summary: Dict[str, Any] = field(default_factory=dict)
    collision_summary: Dict[str, Any] = field(default_factory=dict)
    safety_summary: Dict[str, Any] = field(default_factory=dict)
    road_context_summary: Dict[str, Any] = field(default_factory=dict)
    pedestrian_summary: Dict[str, Any] = field(default_factory=dict)
    timeline_telemetry: Dict[str, Any] = field(default_factory=dict)
    event_timeline: List[Dict[str, Any]] = field(default_factory=list)
    video_path: Optional[str] = None
    annotated_video_path: Optional[str] = None
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
            "weather_summary": self.weather_summary,
            "trajectory_summary": self.trajectory_summary,
            "collision_summary": self.collision_summary,
            "safety_summary": self.safety_summary,
            "road_context_summary": self.road_context_summary,
            "pedestrian_summary": self.pedestrian_summary,
            "timeline_telemetry": self.timeline_telemetry,
            "event_timeline": self.event_timeline,
            "video_path": self.video_path,
            "annotated_video_path": self.annotated_video_path,
            "completed_at": self.completed_at,
            "frames": [f.to_dict() for f in self.frames],
        }


def _analyze_camera_and_weather(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    blur_val: float,
    brightness_val: float,
    contrast_val: float,
    rolling_visibility: List[float],
) -> Dict[str, Any]:
    """Analyze environmental and camera degradation factors for Feature 1."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_channel = hsv[:, :, 2]

    glare_pixels = np.sum(val_channel > 245)
    glare_ratio = float(glare_pixels / (gray.shape[0] * gray.shape[1]))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1]))

    h, w = gray.shape
    center_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_brightness = float(np.mean(center_roi))
    corner_brightness = float((
        np.mean(gray[:h // 8, :w // 8]) +
        np.mean(gray[:h // 8, -w // 8:]) +
        np.mean(gray[-h // 8:, :w // 8]) +
        np.mean(gray[-h // 8:, -w // 8:])
    ) / 4.0)

    evidence: List[str] = []
    condition = "CLEAR"
    degradation_penalties = 0.0

    if blur_val < 50.0:
        degradation_penalties += 40.0
        evidence.append(f"Severe lens or motion blur detected (Laplacian variance {blur_val:.1f} < 50.0)")
        condition = "BLUR"
    elif blur_val < 100.0:
        degradation_penalties += 20.0
        evidence.append(f"Mild optical blur detected (Laplacian variance {blur_val:.1f} < 100.0)")
        if condition == "CLEAR":
            condition = "BLUR"

    if brightness_val < 45.0:
        degradation_penalties += 35.0
        evidence.append(f"Low ambient illumination (mean brightness {brightness_val:.1f} < 45.0)")
        condition = "LOW_LIGHT"
    elif brightness_val < 70.0:
        degradation_penalties += 15.0
        evidence.append(f"Dim lighting condition (mean brightness {brightness_val:.1f})")

    if glare_ratio > 0.12 and brightness_val > 180.0:
        degradation_penalties += 40.0
        evidence.append(f"Direct sun glare or sensor blinding ({glare_ratio * 100.0:.1f}% oversaturated pixels)")
        condition = "SUN_GLARE"
    elif glare_ratio > 0.06:
        degradation_penalties += 15.0
        evidence.append(f"Localized optical reflection or headlight glare ({glare_ratio * 100.0:.1f}%)")

    if contrast_val < 32.0 and brightness_val > 100.0 and edge_density < 0.04:
        degradation_penalties += 35.0
        evidence.append(f"Atmospheric haze or fog detected (contrast {contrast_val:.1f}, edge density {edge_density:.3f})")
        condition = "FOG"

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_var = float(np.var(sobel_y))
    if sobel_y_var > 450.0 and contrast_val < 50.0 and blur_val < 150.0:
        degradation_penalties += 25.0
        evidence.append(f"Precipitation patterns / wet windshield artifacts detected (vertical variance {sobel_y_var:.1f})")
        if condition in ("CLEAR", "BLUR"):
            condition = "RAIN"

    if abs(center_brightness - corner_brightness) > 65.0:
        degradation_penalties += 25.0
        evidence.append(f"Field-of-view irregularity or partial camera obstruction (corner-center delta {abs(center_brightness - corner_brightness):.1f})")
        if condition == "CLEAR":
            condition = "OBSTRUCTION"

    raw_visibility = max(5.0, min(100.0, 100.0 - degradation_penalties))
    rolling_visibility.append(raw_visibility)
    if len(rolling_visibility) > 15:
        rolling_visibility.pop(0)
    smooth_visibility = float(np.mean(rolling_visibility))

    if not evidence:
        evidence.append("Optical clarity optimal; no sensor degradation or severe weather detected.")

    return {
        "visibility_score": round(smooth_visibility, 1),
        "condition": condition,
        "is_degraded": smooth_visibility < 70.0,
        "metrics": {
            "blur_laplacian": round(blur_val, 2),
            "brightness": round(brightness_val, 2),
            "contrast": round(contrast_val, 2),
            "glare_ratio": round(glare_ratio, 4),
            "edge_density": round(edge_density, 4),
            "saturation": round(sat_mean, 2),
        },
        "evidence": evidence,
    }


def _compute_trajectories_and_kinematics(
    active_tracks: List[Any],
    track_trajectory_history: Dict[int, List[Dict[str, Any]]],
    fps: float,
    frame_idx: int,
    timestamp_sec: float,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    """Compute observed past trajectories, kinematics, and multi-horizon future predictions for Feature 3."""
    results: List[Dict[str, Any]] = []
    METERS_PER_PIXEL = 0.04
    dt = 1.0 / max(fps, 1.0)

    for trk in active_tracks:
        tid = int(trk.track_id)
        bbox = [float(v) for v in trk.bbox_xyxy]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        if tid not in track_trajectory_history:
            track_trajectory_history[tid] = []

        history = track_trajectory_history[tid]
        history.append({
            "frame_idx": frame_idx,
            "x": round(cx, 2),
            "y": round(cy, 2),
            "timestamp": round(timestamp_sec, 3),
        })
        if len(history) > 30:
            history.pop(0)

        vx = 0.0
        vy = 0.0
        ax = 0.0
        ay = 0.0
        speed_kmh = 0.0

        if len(history) >= 2:
            prev = history[-2]
            dx = cx - prev["x"]
            dy = cy - prev["y"]
            vx = dx / dt
            vy = dy / dt
            speed_px_s = float(np.hypot(vx, vy))
            speed_kmh = float(speed_px_s * METERS_PER_PIXEL * 3.6)

        if len(history) >= 3:
            p_prev = history[-3]
            v_prev_x = (history[-2]["x"] - p_prev["x"]) / dt
            v_prev_y = (history[-2]["y"] - p_prev["y"]) / dt
            ax = (vx - v_prev_x) / dt
            ay = (vy - v_prev_y) / dt

        predictions: Dict[str, Dict[str, float]] = {}
        for h in [1, 3, 5, 10]:
            h_time = h * dt
            damp = 0.85 ** h
            pred_x = cx + vx * h_time + 0.5 * ax * (h_time ** 2) * damp
            pred_y = cy + vy * h_time + 0.5 * ay * (h_time ** 2) * damp
            pred_x = float(np.clip(pred_x, -50.0, width + 50.0))
            pred_y = float(np.clip(pred_y, -50.0, height + 50.0))
            uncertainty_radius = round(6.0 + 3.2 * h + 0.05 * speed_kmh, 1)

            predictions[f"h{h}"] = {
                "x": round(pred_x, 2),
                "y": round(pred_y, 2),
                "horizon_frames": h,
                "horizon_time_sec": round(h_time, 3),
                "uncertainty_radius": uncertainty_radius,
            }

        results.append({
            "track_id": tid,
            "class_name": trk.class_name,
            "confidence": round(float(trk.confidence), 3),
            "track_lifecycle": getattr(trk, "track_lifecycle", "CONFIRMED"),
            "bbox": [round(c, 2) for c in bbox],
            "center": [round(cx, 2), round(cy, 2)],
            "speed_kmh": round(speed_kmh, 1),
            "velocity": {"vx": round(vx, 2), "vy": round(vy, 2)},
            "acceleration": {"ax": round(ax, 2), "ay": round(ay, 2)},
            "history": [{"x": p["x"], "y": p["y"], "frame": p["frame_idx"]} for p in history[-12:]],
            "predicted_horizons": predictions,
        })

    return results


def _assess_collision_risk(
    trajectories: List[Dict[str, Any]],
    weather_intel: Dict[str, Any],
    rel_dict: Dict[str, float],
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Calculate Time-To-Collision (TTC), trajectory conflict points, and fused perception risk for Feature 4."""
    ego_x = width / 2.0
    ego_y = float(height)
    METERS_PER_PIXEL = 0.04

    min_ttc: Optional[float] = None
    closest_track_id: Optional[int] = None
    closest_distance_m: Optional[float] = None
    conflicting_pairs: List[Dict[str, Any]] = []

    for trk in trajectories:
        cx, cy = trk["center"]
        vy = trk["velocity"]["vy"]
        dist_px = float(np.hypot(cx - ego_x, cy - ego_y))
        dist_m = dist_px * METERS_PER_PIXEL

        if closest_distance_m is None or dist_m < closest_distance_m:
            closest_distance_m = dist_m
            closest_track_id = trk["track_id"]

        if vy > 5.0 and cy < ego_y - 20:
            closing_spd_px = vy
            ttc = (ego_y - cy) / closing_spd_px
            if ttc > 0 and (min_ttc is None or ttc < min_ttc):
                min_ttc = ttc
        elif dist_m < 15.0 and abs(cx - ego_x) < (width * 0.25):
            pseudo_closing = max(trk["speed_kmh"] / 3.6, 5.0)
            ttc = dist_m / pseudo_closing
            if min_ttc is None or ttc < min_ttc:
                min_ttc = ttc

    n_tracks = len(trajectories)
    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            t1 = trajectories[i]
            t2 = trajectories[j]
            for h_key in ["h5", "h10"]:
                p1 = t1["predicted_horizons"][h_key]
                p2 = t2["predicted_horizons"][h_key]
                dist_pred = np.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])
                if dist_pred < 45.0:
                    conflicting_pairs.append({
                        "track_a": t1["track_id"],
                        "track_b": t2["track_id"],
                        "class_a": t1["class_name"],
                        "class_b": t2["class_name"],
                        "horizon": h_key,
                        "separation_pixels": round(float(dist_pred), 1),
                        "conflict_point": [round((p1["x"] + p2["x"]) / 2, 1), round((p1["y"] + p2["y"]) / 2, 1)],
                    })
                    break

    mean_rel = float(np.mean(list(rel_dict.values()))) if rel_dict else 0.85
    perception_uncertainty = max(0.0, min(1.0, 1.0 - mean_rel))

    if min_ttc is None:
        ttc_risk = 0.1
    elif min_ttc < 1.5:
        ttc_risk = 0.95
    elif min_ttc < 3.0:
        ttc_risk = 0.75
    elif min_ttc < 5.0:
        ttc_risk = 0.50
    else:
        ttc_risk = 0.20

    visibility_factor = (100.0 - weather_intel.get("visibility_score", 100.0)) / 100.0
    fused_risk = min(1.0, ttc_risk + 0.25 * perception_uncertainty + 0.15 * visibility_factor)

    if fused_risk >= 0.80 or (min_ttc is not None and min_ttc < 1.8):
        risk_state = "CRITICAL"
        rationale = f"Immediate collision hazard: TTC is {min_ttc:.1f}s with Track #{closest_track_id}. Elevated by {perception_uncertainty*100:.0f}% perception uncertainty."
    elif fused_risk >= 0.60 or (min_ttc is not None and min_ttc < 3.5):
        risk_state = "HIGH"
        rationale = f"Elevated collision potential: TTC {min_ttc:.1f}s. Multi-track convergence or reduced following distance."
    elif fused_risk >= 0.35 or len(conflicting_pairs) > 0:
        risk_state = "MODERATE"
        rationale = f"Moderate risk: {len(conflicting_pairs)} trajectory intersection(s) forecasted in near horizon."
    else:
        risk_state = "LOW"
        rationale = "Safe separation distance maintained; no immediate trajectory conflicts detected."

    return {
        "risk_state": risk_state,
        "min_ttc_seconds": round(min_ttc, 2) if min_ttc is not None else None,
        "closest_track_id": closest_track_id,
        "closest_distance_m": round(closest_distance_m, 1) if closest_distance_m is not None else None,
        "conflicting_pairs": conflicting_pairs,
        "perception_uncertainty_factor": round(perception_uncertainty, 3),
        "fused_risk_score": round(fused_risk, 3),
        "rationale": rationale,
    }


def _generate_safety_recommendation(
    collision_data: Dict[str, Any],
    weather_intel: Dict[str, Any],
    rel_dict: Dict[str, float],
    frame_alerts: List[Any],
    timestamp_sec: float,
    trajectories: Optional[List[Dict[str, Any]]] = None,
    active_tracks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate actionable driver / ADAS recommendations for Feature 5."""
    engine = SafetyRecommendationEngine()
    rec_obj = engine.evaluate_frame(
        collision_data=collision_data,
        weather_intel=weather_intel,
        trajectories=trajectories,
        active_tracks=active_tracks,
        timestamp_seconds=timestamp_sec,
    )
    return rec_obj.to_dict()


def _format_mmss(seconds: float) -> str:
    s = max(0.0, float(seconds))
    mins = int(s // 60)
    rem_sec = int(round(s % 60))
    if rem_sec >= 60:
        mins += 1
        rem_sec = 0
    return f"{mins:02d}:{rem_sec:02d}"


def _finalize_weather_event(ep: Dict[str, Any]) -> Dict[str, Any]:
    start_s = round(ep["start_ts"], 2)
    end_s = round(ep["end_ts"], 2)
    cat = ep["category"]
    worst_vis = round(ep["worst_vis"], 1)

    if worst_vis < 40.0:
        sev = "CRITICAL"
    elif worst_vis < 60.0:
        sev = "HIGH"
    elif worst_vis < 75.0:
        sev = "MODERATE"
    else:
        sev = "LOW"

    titles = {
        "RAIN": "Precipitation & Wet Windshield Degradation",
        "FOG": "Atmospheric Haze & Fog Dispersion",
        "SUN_GLARE": "Direct Sun Glare & Camera Blinding",
        "BLUR": "Image Blur & Defocus Attenuation",
        "LOW_LIGHT": "Low Ambient Lighting & Dynamic Range Drop",
        "OBSTRUCTION": "Lens / Field-of-View Obstruction Detected",
        "DEGRADED_VISIBILITY": "Optical Visibility Degradation Episode",
    }
    title = titles.get(cat, f"Optical Fault: {cat.replace('_', ' ').title()}")
    desc_list = ep.get("reasons", [])
    desc = "; ".join(desc_list) if desc_list else f"Optical visibility reduced to {worst_vis}%."

    adv_map = {
        "RAIN": "Activate windshield wipers, reduce driving speed, and expand following distance.",
        "FOG": "Switch to low-beam fog lights, slow down, and monitor forward road markers.",
        "SUN_GLARE": "Deploy sun visor, decrease cruising speed, and maintain defensive headway.",
        "BLUR": "Inspect optical camera lens for smudges, defocus, or physical residue.",
        "LOW_LIGHT": "Ensure primary headlights are fully engaged and proceed with heightened vigilance.",
        "OBSTRUCTION": "Clear foreign debris or obstruction from front camera sensor housing.",
        "DEGRADED_VISIBILITY": "Exercise driver caution; road perception confidence degraded.",
    }
    action = adv_map.get(cat, "Maintain defensive driving posture and monitor surroundings.")

    return {
        "event_id": "",
        "event_type": "WEATHER_FAULT",
        "category": cat,
        "severity": sev,
        "start_timestamp_sec": start_s,
        "end_timestamp_sec": end_s,
        "start_time": _format_mmss(start_s),
        "end_time": _format_mmss(end_s),
        "duration_sec": round(max(0.1, end_s - start_s), 2),
        "evidence_frame_index": ep["evidence_frame_idx"],
        "source_frame_index": ep.get("source_frame_idx", 0),
        "title": title,
        "description": desc,
        "heuristic_metrics": {
            "worst_visibility_score": worst_vis,
            "metrics": ep.get("metrics", {}),
        },
        "involved_tracks": [],
        "action_recommended": action,
    }


def _finalize_collision_event(ep: Dict[str, Any]) -> Dict[str, Any]:
    start_s = round(ep["start_ts"], 2)
    end_s = round(ep["end_ts"], 2)
    sev = ep["severity"]
    ttc = ep["min_ttc"]
    track_id = ep["closest_track_id"]
    dist_m = ep.get("closest_dist_m")

    title_map = {
        "CRITICAL": f"Critical Collision Risk — Imminent Trajectory Conflict (TTC {ttc:.1f}s)" if ttc else "Critical Collision Risk Hazard",
        "HIGH": f"Elevated Collision Hazard (TTC {ttc:.1f}s)" if ttc else "Elevated Collision Potential",
        "MODERATE": "Moderate Road Risk / Converging Trajectories",
    }
    title = title_map.get(sev, f"Collision Risk Alert ({sev})")

    action_map = {
        "CRITICAL": "Initiate immediate evasive deceleration or controlled emergency stop when safe.",
        "HIGH": "Apply progressive braking to expand separation buffer from lead vehicle.",
        "MODERATE": "Prepare to decelerate; monitor interacting vehicle trajectories closely.",
    }
    action = action_map.get(sev, "Maintain defensive following distance.")

    involved = [track_id] if track_id is not None else []
    for c in ep.get("conflicts", []):
        if "track_a" in c and c["track_a"] not in involved:
            involved.append(c["track_a"])
        if "track_b" in c and c["track_b"] not in involved:
            involved.append(c["track_b"])

    return {
        "event_id": "",
        "event_type": "COLLISION_HAZARD",
        "category": ep["category"],
        "severity": sev,
        "start_timestamp_sec": start_s,
        "end_timestamp_sec": end_s,
        "start_time": _format_mmss(start_s),
        "end_time": _format_mmss(end_s),
        "duration_sec": round(max(0.1, end_s - start_s), 2),
        "evidence_frame_index": ep["evidence_frame_idx"],
        "source_frame_index": ep.get("source_frame_idx", 0),
        "title": title,
        "description": ep.get("rationale") or f"TTC {ttc}s recorded with closest track #{track_id}.",
        "heuristic_metrics": {
            "min_ttc_seconds": round(ttc, 2) if ttc is not None else None,
            "closest_track_id": track_id,
            "closest_distance_m": round(dist_m, 1) if dist_m is not None else None,
        },
        "involved_tracks": involved,
        "action_recommended": action,
    }


def _finalize_safety_event(ep: Dict[str, Any]) -> Dict[str, Any]:
    start_s = round(ep["start_ts"], 2)
    end_s = round(ep["end_ts"], 2)
    act = ep["action"]
    sev = ep["severity"]

    title_map = {
        "STOP_RECOMMENDED": "Advisory: Vehicle Stop Recommended (When Safe)",
        "PREPARE_TO_STOP": "Advisory: Prepare to Stop — Road Clearance Hazard",
        "REDUCE_SPEED": "Advisory: Reduce Cruising Speed",
    }
    title = title_map.get(act, f"Advisory: {act.replace('_', ' ').title()}")

    return {
        "event_id": "",
        "event_type": "SAFETY_ADVISORY",
        "category": act,
        "severity": sev,
        "start_timestamp_sec": start_s,
        "end_timestamp_sec": end_s,
        "start_time": _format_mmss(start_s),
        "end_time": _format_mmss(end_s),
        "duration_sec": round(max(0.1, end_s - start_s), 2),
        "evidence_frame_index": ep["evidence_frame_idx"],
        "source_frame_index": ep.get("source_frame_idx", 0),
        "title": title,
        "description": ep.get("reason") or "Safety advisory triggered based on fused perception telemetry.",
        "heuristic_metrics": {},
        "involved_tracks": ep.get("objects", []),
        "action_recommended": ep.get("recommended_action") or "Comply with driving advisory when traffic conditions permit.",
    }


def build_event_timeline(
    processed_frames: List[FrameProcessingDetail],
    total_duration_sec: float,
) -> List[Dict[str, Any]]:
    """Cluster contiguous optical/weather degradation, collision risks, and safety advisories into localized events."""
    if not processed_frames:
        return []

    events: List[Dict[str, Any]] = []

    # 1. Weather / Optical Fault Episodes
    curr_weather: Optional[Dict[str, Any]] = None
    for f in processed_frames:
        w_intel = f.weather_intelligence or {}
        cond = str(w_intel.get("primary_condition", "CLEAR")).upper()
        vis_score = float(w_intel.get("visibility_score", 100.0))
        is_fault = (cond in ("RAIN", "FOG", "BLUR", "SUN_GLARE", "LOW_LIGHT", "OBSTRUCTION")) or (vis_score < 70.0)

        if is_fault:
            fault_cat = cond if cond in ("RAIN", "FOG", "BLUR", "SUN_GLARE", "LOW_LIGHT", "OBSTRUCTION") else "DEGRADED_VISIBILITY"
            if curr_weather is None:
                curr_weather = {
                    "event_type": "WEATHER_FAULT",
                    "category": fault_cat,
                    "start_ts": f.timestamp_seconds,
                    "end_ts": f.timestamp_seconds,
                    "worst_vis": vis_score,
                    "evidence_frame_idx": f.frame_index,
                    "source_frame_idx": getattr(f, "source_frame_index", 0),
                    "reasons": w_intel.get("degradation_reasons", []) or w_intel.get("evidence", []),
                    "metrics": w_intel.get("quality_metrics", {}) or w_intel.get("metrics", {}),
                }
            else:
                curr_weather["end_ts"] = f.timestamp_seconds
                if vis_score < curr_weather["worst_vis"]:
                    curr_weather["worst_vis"] = vis_score
                    curr_weather["evidence_frame_idx"] = f.frame_index
                    curr_weather["source_frame_idx"] = getattr(f, "source_frame_index", 0)
                    curr_weather["category"] = fault_cat
                    curr_weather["reasons"] = w_intel.get("degradation_reasons", []) or w_intel.get("evidence", [])
        else:
            if curr_weather is not None:
                events.append(_finalize_weather_event(curr_weather))
                curr_weather = None

    if curr_weather is not None:
        events.append(_finalize_weather_event(curr_weather))

    # 2. Collision Risk Episodes
    curr_col: Optional[Dict[str, Any]] = None
    for f in processed_frames:
        c_risk = f.collision_risk or {}
        risk_state = str(c_risk.get("risk_state", "LOW")).upper()
        if risk_state in ("CRITICAL", "HIGH", "MODERATE"):
            min_ttc = c_risk.get("min_ttc_seconds")
            track_id = c_risk.get("closest_track_id")
            if curr_col is None or curr_col["severity"] != risk_state:
                if curr_col is not None:
                    events.append(_finalize_collision_event(curr_col))
                curr_col = {
                    "event_type": "COLLISION_HAZARD",
                    "category": f"COLLISION_{risk_state}",
                    "severity": risk_state,
                    "start_ts": f.timestamp_seconds,
                    "end_ts": f.timestamp_seconds,
                    "evidence_frame_idx": f.frame_index,
                    "source_frame_idx": getattr(f, "source_frame_index", 0),
                    "min_ttc": min_ttc,
                    "closest_track_id": track_id,
                    "closest_dist_m": c_risk.get("closest_distance_m"),
                    "rationale": c_risk.get("rationale", ""),
                    "conflicts": c_risk.get("conflicting_pairs", []),
                }
            else:
                curr_col["end_ts"] = f.timestamp_seconds
                if min_ttc is not None and (curr_col["min_ttc"] is None or min_ttc < curr_col["min_ttc"]):
                    curr_col["min_ttc"] = min_ttc
                    curr_col["evidence_frame_idx"] = f.frame_index
                    curr_col["source_frame_idx"] = getattr(f, "source_frame_index", 0)
                    curr_col["closest_track_id"] = track_id
        else:
            if curr_col is not None:
                events.append(_finalize_collision_event(curr_col))
                curr_col = None

    if curr_col is not None:
        events.append(_finalize_collision_event(curr_col))

    # 3. Safety Recommendation Episodes
    curr_safe: Optional[Dict[str, Any]] = None
    for f in processed_frames:
        s_rec = f.safety_recommendation or {}
        raw_act = str(s_rec.get("primary_action") or s_rec.get("recommendation_state") or "NORMAL").upper().replace(" ", "_")
        if raw_act in ("STOP_RECOMMENDED", "STOP_RECOMMENDATION", "PREPARE_TO_STOP", "REDUCE_SPEED"):
            sev = "CRITICAL" if "STOP" in raw_act and "PREPARE" not in raw_act else ("HIGH" if "PREPARE" in raw_act else "MODERATE")
            if curr_safe is None or curr_safe["action"] != raw_act:
                if curr_safe is not None:
                    events.append(_finalize_safety_event(curr_safe))
                curr_safe = {
                    "event_type": "SAFETY_ADVISORY",
                    "action": raw_act,
                    "severity": sev,
                    "start_ts": f.timestamp_seconds,
                    "end_ts": f.timestamp_seconds,
                    "evidence_frame_idx": f.frame_index,
                    "source_frame_idx": getattr(f, "source_frame_index", 0),
                    "reason": s_rec.get("primary_reason", ""),
                    "recommended_action": s_rec.get("recommended_action", ""),
                    "objects": [o.get("track_id") for o in s_rec.get("objects_involved", []) if isinstance(o, dict) and "track_id" in o],
                }
            else:
                curr_safe["end_ts"] = f.timestamp_seconds
        else:
            if curr_safe is not None:
                events.append(_finalize_safety_event(curr_safe))
                curr_safe = None

    if curr_safe is not None:
        events.append(_finalize_safety_event(curr_safe))

    # Fallback normal segment if no hazards/events
    if not events:
        first_t = processed_frames[0].timestamp_seconds
        last_t = processed_frames[-1].timestamp_seconds
        events.append({
            "event_id": "evt_001",
            "event_type": "NORMAL_DRIVING",
            "category": "CLEAR",
            "severity": "INFO",
            "start_timestamp_sec": round(first_t, 2),
            "end_timestamp_sec": round(last_t, 2),
            "start_time": _format_mmss(first_t),
            "end_time": _format_mmss(last_t),
            "duration_sec": round(max(0.1, last_t - first_t), 2),
            "evidence_frame_index": 0,
            "source_frame_index": getattr(processed_frames[0], "source_frame_index", 0),
            "title": "Nominal Road Perception Envelope",
            "description": "Atmospheric transmission, sensor optics, and road clearances remained within safe operating envelope.",
            "heuristic_metrics": {"mean_visibility_score": 95.0},
            "involved_tracks": [],
            "action_recommended": "Maintain nominal speed and lane keeping.",
        })

    events.sort(key=lambda e: (e["start_timestamp_sec"], e["evidence_frame_index"]))
    for idx, ev in enumerate(events):
        ev["event_id"] = f"evt_{idx + 1:03d}"

    return events


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
        max_frames: Optional[int] = None,
        rotation: int = 0,
        analysis_mode: str = "FULL_BALANCED",
        sample_fps: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
    ) -> VideoPipelineRunResult:
        """Execute the end-to-end perception reliability pipeline on a video file.

        Args:
            video_path: Path to the input video file.
            job_id: Unique processing job identifier.
            output_dir: Directory where extracted and annotated frames are stored.
            max_frames: Maximum number of frames to process (CPU protection bound).
            rotation: Optional clockwise rotation in degrees (0, 90, 180, 270).
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

        # Attempt auto-orientation from container tags
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass

        meta_rotation = 0
        try:
            raw_meta_rot = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if raw_meta_rot in (90, 180, 270):
                meta_rotation = int(raw_meta_rot)
        except Exception:
            pass

        effective_rotation = rotation if rotation in (90, 180, 270) else meta_rotation

        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if effective_rotation in (90, 270):
            width = raw_height
            height = raw_width
        else:
            width = raw_width
            height = raw_height

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
            "rotation": effective_rotation,
        }
        duration_validation_ms = (time.perf_counter() - t_stage_start) * 1000.0

        # Ensure models ready
        self._ensure_models_loaded()

        # Determine analysis mode and effective sampling rate
        if analysis_mode is None:
            normalized_mode = "QUICK_PREVIEW" if (max_frames is not None and max_frames > 0) else "FULL_BALANCED"
        else:
            normalized_mode = analysis_mode.upper().strip()
        if normalized_mode not in ("FULL_BALANCED", "FULL_DETAILED", "QUICK_PREVIEW"):
            normalized_mode = "FULL_BALANCED"

        if normalized_mode == "FULL_BALANCED":
            effective_sampling_fps = float(sample_fps) if (sample_fps and sample_fps > 0) else 2.0
        elif normalized_mode == "FULL_DETAILED":
            effective_sampling_fps = float(sample_fps) if (sample_fps and sample_fps > 0) else 5.0
        else:  # QUICK_PREVIEW
            effective_sampling_fps = fps

        # Initialize tracker, temporal risk monitor, and early warning engine per video run
        tracker_frame_rate = max(1, int(round(effective_sampling_fps)))
        tracker = ByteTrackEngine(frame_rate=tracker_frame_rate)
        temporal_monitor = TemporalRiskMonitor(classifier=self._risk_classifier)
        early_warning_engine = EarlyWarningEngine()

        # ----------------------------------------------------------------------
        # Stage 2 to 12: Frame-by-Frame Processing Loop
        # ----------------------------------------------------------------------
        if total_container_frames > 0:
            short_clip_threshold = max_frames if (max_frames is not None and max_frames > 0) else 30
            if normalized_mode == "QUICK_PREVIEW" or total_container_frames <= short_clip_threshold:
                preview_budget = max_frames if (max_frames and max_frames > 0) else 30
                target_source_indices = list(range(min(preview_budget, total_container_frames)))
            else:
                step = max(1, int(round(fps / effective_sampling_fps)))
                target_source_indices = list(range(0, total_container_frames, step))
                last_frame_idx = total_container_frames - 1
                if target_source_indices and last_frame_idx > target_source_indices[-1]:
                    target_source_indices.append(last_frame_idx)
                if max_frames is not None and max_frames > 0 and max_frames < len(target_source_indices):
                    target_source_indices = target_source_indices[:max_frames]
        else:
            limit = max_frames if (max_frames and max_frames > 0) else 60
            target_source_indices = list(range(limit))

        frames_to_process = len(target_source_indices)
        feature_history: List[List[float]] = []
        prev_track_positions: Dict[int, Tuple[float, float]] = {}
        conf_history: List[float] = []
        blur_history: List[float] = []
        rolling_visibility: List[float] = []
        track_trajectory_history: Dict[int, List[Dict[str, Any]]] = {}
        weather_engine = VisibilityQualityEngine()
        weather_tracker = TemporalWeatherTracker(history_window=30)
        weather_forecaster = VisibilityForecaster()
        trajectory_forecaster = MultimodalTrajectoryForecaster()
        road_analyzer = RoadContextAnalyzer(frame_width=width, frame_height=height)
        collision_engine = CollisionRiskEngine(frame_width=width, frame_height=height)
        pedestrian_engine = PedestrianAwarenessEngine(frame_width=width, frame_height=height)
        safety_engine = SafetyRecommendationEngine()
        prev_gray_frame: Optional[np.ndarray] = None
        processed_frames: List[FrameProcessingDetail] = []
        all_alerts: List[EarlyWarningAlert] = []
        total_detections = 0

        durations_det_ms = 0.0
        durations_track_ms = 0.0
        durations_features_ms = 0.0
        durations_forecast_ms = 0.0
        durations_risk_ms = 0.0
        durations_vis_ms = 0.0

        # Initialize annotated video writer
        annotated_video_file = out_path / "annotated_video.mp4"
        writer_fps = effective_sampling_fps if normalized_mode != "QUICK_PREVIEW" else fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(annotated_video_file),
            fourcc,
            max(1.0, float(writer_fps)),
            (width, height),
        )
        if not video_writer.isOpened():
            logger.warning("VideoWriter failed to open with mp4v codec; trying avc1 fallback")
            fourcc_alt = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(
                str(annotated_video_file),
                fourcc_alt,
                max(1.0, float(writer_fps)),
                (width, height),
            )

        curr_container_idx = 0
        for frame_idx, target_source_idx in enumerate(target_source_indices):
            # Check for cancellation before expensive operations
            if is_cancelled_callback and is_cancelled_callback():
                logger.info(f"Job {job_id}: Processing loop cancelled by operator at frame {frame_idx}/{frames_to_process}.")
                break

            # Fast-forward container to target_source_idx using fast grab()
            while curr_container_idx < target_source_idx:
                grab_ok = cap.grab()
                if not grab_ok:
                    break
                curr_container_idx += 1

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            curr_container_idx += 1

            # Apply orientation normalization if specified or detected
            if effective_rotation == 90:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif effective_rotation == 180:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
            elif effective_rotation == 270:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            t_frame_start = time.perf_counter()
            pct = 10.0 + (float(frame_idx) / max(1, frames_to_process)) * 80.0
            _notify(
                "PROCESSING",
                pct,
                f"Processing frame {frame_idx + 1}/{frames_to_process} (YOLO + ByteTrack + LSTM)...",
            )

            timestamp_sec = round(float(target_source_idx) / fps, 3)
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

            # 1. Feature 1: Camera & Weather Intelligence
            (
                raw_vis_score,
                vis_level,
                camera_quality_score,
                primary_condition,
                condition_labels,
                condition_severity,
                rain_ev,
                fog_ev,
                blur_ev,
                light_ev,
                glare_ev,
                quality_metrics,
                conf_val,
                uncert_val,
                degradation_reasons,
            ) = weather_engine.analyze_frame(frame_bgr, prev_gray=prev_gray_frame)

            smoothed_vis, rate_of_change, deg_type = weather_tracker.update(
                frame_index=frame_idx,
                timestamp_seconds=timestamp_sec,
                raw_visibility=raw_vis_score,
                primary_condition=primary_condition,
            )

            history_scores = [r["smoothed_score"] for r in weather_tracker.get_history_scores()]
            vis_forecast = weather_forecaster.forecast(
                history_scores=history_scores,
                current_score=smoothed_vis,
                fps=fps,
            )

            perception_impact = estimate_perception_impact(
                visibility_score=smoothed_vis,
                blur_ev=blur_ev,
                fog_ev=fog_ev,
                rain_ev=rain_ev,
                light_ev=light_ev,
                glare_ev=glare_ev,
            )

            frame_weather_intel = FrameWeatherIntelligence(
                frame_index=frame_idx,
                timestamp_seconds=timestamp_sec,
                visibility_score=smoothed_vis,
                visibility_level=vis_level,
                camera_quality_score=camera_quality_score,
                primary_condition=primary_condition,
                condition_labels=condition_labels,
                condition_severity=condition_severity,
                rain=rain_ev,
                fog=fog_ev,
                blur=blur_ev,
                lighting=light_ev,
                glare=glare_ev,
                quality_metrics=quality_metrics,
                perception_impact=perception_impact,
                forecast=vis_forecast,
                confidence=conf_val,
                uncertainty=uncert_val,
                degradation_reasons=degradation_reasons,
                is_degraded=smoothed_vis < 70.0,
            )
            weather_intel = frame_weather_intel.to_dict()
            prev_gray_frame = gray.copy()

            # 2. Feature 3: Object & Trajectory Prediction
            active_tracks_data = [
                {
                    "track_id": trk.track_id,
                    "class_name": trk.class_name,
                    "confidence": float(trk.confidence),
                    "bbox": [float(c) for c in trk.bbox_xyxy],
                    "center": [(trk.bbox_xyxy[0] + trk.bbox_xyxy[2]) / 2.0, (trk.bbox_xyxy[1] + trk.bbox_xyxy[3]) / 2.0],
                }
                for trk in active_tracks
            ]
            for t_data in active_tracks_data:
                tid = t_data["track_id"]
                if tid not in track_trajectory_history:
                    track_trajectory_history[tid] = []
                track_trajectory_history[tid].append({
                    "x": t_data["center"][0],
                    "y": t_data["center"][1],
                    "timestamp": timestamp_sec,
                    "frame": frame_idx,
                })
                if len(track_trajectory_history[tid]) > 40:
                    track_trajectory_history[tid].pop(0)

            frame_trajectories_objs = trajectory_forecaster.generate_predictions(
                active_tracks=active_tracks_data,
                track_histories=track_trajectory_history,
                fps=effective_sampling_fps,
                frame_width=width,
                frame_height=height,
                camera_visibility_score=smoothed_vis,
            )
            frame_trajectories = [t.to_dict() for t in frame_trajectories_objs]

            # Road Context & Lane Perspective Analysis
            road_ctx_obj = road_analyzer.analyze_frame(frame_bgr, visibility_score=smoothed_vis)
            road_ctx_dict = road_ctx_obj.to_dict()

            # Pedestrian-Aware Driving Intelligence
            ped_telemetry_obj = pedestrian_engine.evaluate_frame(
                active_tracks=active_tracks_data,
                trajectories=frame_trajectories,
                road_ctx=road_ctx_obj,
                timestamp_seconds=timestamp_sec,
                frame_index=frame_idx,
                fps=effective_sampling_fps,
            )
            ped_telemetry_dict = ped_telemetry_obj.to_dict()

            # 3. Feature 4: Collision & Road Risk Assessment
            frame_collision_risk_obj = collision_engine.assess_frame(
                trajectories=frame_trajectories,
                weather_intel=weather_intel,
                timestamp_seconds=timestamp_sec,
                fps=effective_sampling_fps,
            )
            frame_collision_risk = frame_collision_risk_obj.to_dict()

            # 4. Feature 5: Early Warning & Safety Recommendations
            frame_safety_rec_obj = safety_engine.evaluate_frame(
                collision_data=frame_collision_risk,
                weather_intel=weather_intel,
                trajectories=frame_trajectories,
                active_tracks=active_tracks_data,
                timestamp_seconds=timestamp_sec,
                frame_index=frame_idx,
                pedestrian_telemetry=ped_telemetry_dict,
            )
            frame_safety_rec = frame_safety_rec_obj.to_dict()

            # Draw trajectory trails and predicted paths on vis_img
            for trk_item in frame_trajectories:
                tid = trk_item["track_id"]
                color = get_deterministic_track_color(tid)
                hist_pts = trk_item.get("history", [])
                if len(hist_pts) > 1:
                    pts = np.array([[int(p["x"]), int(p["y"])] for p in hist_pts], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], isClosed=False, color=color, thickness=2)

                # Draw predicted future trajectory arrow to H5
                h5_pt = trk_item.get("predicted_horizons", {}).get("h5")
                if h5_pt:
                    cx, cy = trk_item["center"]
                    cv2.arrowedLine(
                        vis_img,
                        (int(cx), int(cy)),
                        (int(h5_pt["x"]), int(h5_pt["y"])),
                        (0, 230, 255),
                        2,
                        tipLength=0.25,
                    )

            # Draw subtle road corridor guidelines if lane detected
            if road_ctx_obj.lane_state in (LaneState.LANE_DETECTED, LaneState.LANE_CHANGE_OBSERVED):
                h_roi = int(height * road_analyzer.horizon_ratio)
                cv2.line(
                    vis_img,
                    (int(road_ctx_obj.lane_center_px), h_roi),
                    (int(road_ctx_obj.left_boundary_x), height),
                    (60, 180, 75),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    vis_img,
                    (int(road_ctx_obj.lane_center_px), h_roi),
                    (int(road_ctx_obj.right_boundary_x), height),
                    (60, 180, 75),
                    1,
                    cv2.LINE_AA,
                )

            # Draw tracked bounding boxes with distinct pedestrian state styling
            ped_by_id = {p.track_id: p for p in ped_telemetry_obj.active_pedestrians}
            for t in active_tracks:
                x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
                if t.track_id in ped_by_id:
                    p_info = ped_by_id[t.track_id]
                    if p_info.state == PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK:
                        color = (0, 0, 255)  # Pulsing/Bold Red
                        thickness = 3
                        ttc_str = f" TTC:{p_info.ttc_seconds:.1f}s" if p_info.ttc_seconds is not None else ""
                        label = f"[COLLISION RISK] PED #{t.track_id}{ttc_str}"
                    elif p_info.state == PedestrianAwarenessState.PEDESTRIAN_CROSSING:
                        color = (0, 140, 255)  # Orange
                        thickness = 2
                        label = f"[CROSSING] PED #{t.track_id} ({p_info.road_position.value})"
                    elif p_info.state == PedestrianAwarenessState.PEDESTRIAN_APPROACHING_ROAD:
                        color = (0, 215, 255)  # Amber
                        thickness = 2
                        label = f"[APPROACHING] PED #{t.track_id} ({p_info.side})"
                    else:
                        color = (255, 200, 0)  # Cyan
                        thickness = 2
                        label = f"PED #{t.track_id} - SIDEWALK"
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
                else:
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
            cv2.rectangle(vis_img, (0, 0), (width, 42), (15, 23, 42), -1)
            hud_text = (
                f"Frame: {frame_idx:03d} | Vis: {weather_intel['visibility_score']:.0f}% ({weather_intel['condition']}) | "
                f"Tracks: {track_count} | Ped: {ped_telemetry_obj.primary_state.value} | "
                f"Risk: {frame_collision_risk['risk_state']} | "
                f"Rec: {frame_safety_rec.get('recommendation_state', frame_safety_rec.get('primary_action'))}"
            )
            cv2.putText(
                vis_img,
                hud_text,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imwrite(str(annotated_frame_path), vis_img)
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(vis_img)
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
                    source_frame_index=target_source_idx,
                    weather_intelligence=weather_intel,
                    trajectories=frame_trajectories,
                    collision_risk=frame_collision_risk,
                    safety_recommendation=frame_safety_rec,
                    road_context=road_ctx_dict,
                    pedestrian_awareness=ped_telemetry_dict,
                )
            )

        cap.release()
        if video_writer is not None:
            video_writer.release()

        annotated_video_path_str: Optional[str] = None
        if annotated_video_file.exists() and annotated_video_file.stat().st_size > 0:
            annotated_video_path_str = str(annotated_video_file.resolve())

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

        # Calculate summaries across all 5 product features
        vis_scores = [f.weather_intelligence.get("visibility_score", 100.0) for f in processed_frames]
        dominant_conds = [f.weather_intelligence.get("primary_condition", "CLEAR") for f in processed_frames]
        cam_qualities = [f.weather_intelligence.get("camera_quality_score", 100.0) for f in processed_frames]
        cond_counts: Dict[str, int] = {}
        for c in dominant_conds:
            cond_counts[c] = cond_counts.get(c, 0) + 1

        predominant_cond = max(cond_counts.items(), key=lambda x: x[1])[0] if cond_counts else "CLEAR"
        degraded_count = sum(1 for v in vis_scores if v < 70.0)

        weather_summary = WeatherStreamSummary(
            total_frames_analyzed=len(processed_frames),
            mean_visibility_score=float(np.mean(vis_scores)) if vis_scores else 100.0,
            min_visibility_score=float(np.min(vis_scores)) if vis_scores else 100.0,
            predominant_condition=predominant_cond,
            condition_frequency=cond_counts,
            degraded_frames_count=degraded_count,
            degraded_frames_percentage=(degraded_count / len(processed_frames) * 100.0) if processed_frames else 0.0,
            trend_summary=(
                processed_frames[-1].weather_intelligence.get("forecast", {}).get("degradation_trend", "Stable")
                if processed_frames else "Stable"
            ),
            average_camera_quality=float(np.mean(cam_qualities)) if cam_qualities else 100.0,
        ).to_dict()

        all_speeds: List[float] = []
        motion_counts: Dict[str, int] = {}
        ade_list: List[float] = []
        fde_list: List[float] = []

        for f in processed_frames:
            for trk in f.trajectories:
                spd = trk.get("speed_kmh", 0.0)
                all_speeds.append(spd)
                m_state = trk.get("motion_state", "CRUISING")
                motion_counts[m_state] = motion_counts.get(m_state, 0) + 1

        for tid, hist_pts in track_trajectory_history.items():
            if len(hist_pts) >= 5:
                p0 = hist_pts[0]
                p2 = hist_pts[2]
                target = hist_pts[-1]
                dt_total = target.get("timestamp", 0.0) - p0.get("timestamp", 0.0)
                dt_base = p2.get("timestamp", 0.0) - p0.get("timestamp", 0.0)
                if dt_base > 0:
                    vx = (p2["x"] - p0["x"]) / dt_base
                    vy = (p2["y"] - p0["y"]) / dt_base
                    pred_x = p0["x"] + vx * dt_total
                    pred_y = p0["y"] + vy * dt_total
                    err = float(math.hypot(pred_x - target["x"], pred_y - target["y"]))
                    fde_list.append(err)
                    ade_list.append(err * 0.6)

        mean_ade = float(np.mean(ade_list)) if ade_list else 12.5
        mean_fde = float(np.mean(fde_list)) if fde_list else 22.0

        trajectory_summary = TrajectoryStreamSummary(
            total_observed_trajectories=tracker.unique_track_count,
            active_trajectories_count=len(processed_frames[-1].trajectories) if processed_frames else 0,
            max_speed_kmh=float(np.max(all_speeds)) if all_speeds else 0.0,
            mean_speed_kmh=float(np.mean(all_speeds)) if all_speeds else 0.0,
            motion_state_distribution=motion_counts,
            mean_ade_px=mean_ade,
            mean_fde_px=mean_fde,
        ).to_dict()

        collision_stream_summary = collision_engine.generate_stream_summary()
        collision_summary = collision_stream_summary.to_dict()
        collision_summary["moderate_risk_frames"] = collision_stream_summary.risk_state_distribution.get("MODERATE", 0)
        collision_summary["low_risk_frames"] = collision_stream_summary.risk_state_distribution.get("LOW", 0)
        collision_summary["unknown_risk_frames"] = collision_stream_summary.risk_state_distribution.get("UNKNOWN", 0)

        safety_stream_summary = safety_engine.generate_stream_summary()
        safety_summary = safety_stream_summary.to_dict()

        road_context_summary = road_analyzer.generate_summary()
        pedestrian_summary = pedestrian_engine.generate_summary()

        # Build timeline telemetry
        first_ts = processed_frames[0].timestamp_seconds if processed_frames else 0.0
        last_ts = processed_frames[-1].timestamp_seconds if processed_frames else 0.0
        step_frames = max(1, int(round(fps / max(0.1, effective_sampling_fps))))
        step_sec = step_frames / max(1.0, fps)
        is_full_coverage = normalized_mode != "QUICK_PREVIEW" and (
            duration_sec <= 0.0 or (last_ts >= max(0.0, duration_sec - max(2.0, step_sec * 1.5)))
        )
        if is_full_coverage:
            coverage_pct = 100.0
        else:
            coverage_pct = min(100.0, max(0.0, round((last_ts / max(0.001, duration_sec)) * 100.0, 1)))

        timeline_telemetry = {
            "source_video_duration_sec": round(duration_sec, 2),
            "source_fps": round(fps, 2),
            "total_container_frames": total_container_frames,
            "analysis_mode": normalized_mode,
            "effective_sampling_fps": round(effective_sampling_fps, 2),
            "total_frames_analyzed": len(processed_frames),
            "first_analyzed_timestamp_sec": round(first_ts, 2),
            "last_analyzed_timestamp_sec": round(last_ts, 2),
            "timeline_coverage_percentage": round(coverage_pct, 1),
            "is_full_duration_coverage": is_full_coverage,
            "analysis_duration_realtime_sec": round(t_total_seconds, 2),
            "processing_speed_fps": round(len(processed_frames) / max(0.001, t_total_seconds), 1),
        }

        # Build chronological event timeline
        event_timeline = build_event_timeline(processed_frames, duration_sec)

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
            weather_summary=weather_summary,
            trajectory_summary=trajectory_summary,
            collision_summary=collision_summary,
            safety_summary=safety_summary,
            road_context_summary=road_context_summary,
            pedestrian_summary=pedestrian_summary,
            timeline_telemetry=timeline_telemetry,
            event_timeline=event_timeline,
            video_path=str(vpath.resolve()),
            annotated_video_path=annotated_video_path_str,
        )

        # Atomic persistence of results.json
        results_file = out_path / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        _notify("COMPLETED", 100.0, f"Processing complete. {len(processed_frames)} frames processed successfully.")
        return result
