"""PerceptGuard AI - Feature 5 Safety Recommendation & Early Warning Test Suite.

Comprehensive validation covering all 22 required scenarios from Master Prompt Section 12:
 1. Normal road conditions.
 2. Gradually increasing collision risk.
 3. Moderate-to-high risk escalation.
 4. High-to-critical risk escalation.
 5. Vehicle-pedestrian trajectory conflict.
 6. Vehicle-to-vehicle conflict.
 7. Decreasing TTC.
 8. Invalid TTC.
 9. Degraded visibility.
10. Heavy rain with no detected collision threat.
11. Fog with unreliable detection.
12. Missing trajectory predictions.
13. Stale risk assessment.
14. Missing object tracks.
15. Uncertain trajectory prediction.
16. Conflicting upstream signals.
17. Duplicate warning suppression.
18. Warning escalation.
19. Warning de-escalation.
20. UNKNOWN-state behavior.
21. Recovery after reliable data returns.
22. Invalid timestamps.

Plus Video Pipeline Runner integration and FastAPI endpoint tests.
NOTE: All test data is synthetically constructed for algorithmic verification.
"""

import math
from typing import Any, Dict, List
import pytest

from src.safety_recommendation import (
    EarlyWarningEngine,
    InvolvedObject,
    RecommendationRulesEngine,
    RecommendationState,
    SafetyRecommendation,
    SafetyRecommendationEngine,
    SafetyStreamSummary,
    SupportingEvidence,
    WarningHistoryManager,
    WarningPriority,
    WarningPriorityManager,
    WarningStatus,
)


# ==============================================================================
# Helper Synthetic Fixtures
# ==============================================================================

def make_nominal_weather(vis_score: float = 95.0, cond: str = "CLEAR") -> Dict[str, Any]:
    """Synthetic clear weather intelligence fixture."""
    return {
        "visibility_score": vis_score,
        "condition": cond,
        "primary_condition": cond,
        "is_degraded": vis_score < 70.0,
        "uncertainty": 0.05,
    }


def make_nominal_collision(
    risk_state: str = "LOW",
    fused_risk: float = 0.10,
    min_ttc: Any = None,
    closest_dist: float = 45.0,
    closest_track_id: Any = None,
    timestamp: float = 1.0,
    uncertainty: float = 0.10,
    conflicts: Any = None,
) -> Dict[str, Any]:
    """Synthetic collision assessment fixture."""
    return {
        "risk_state": risk_state,
        "fused_risk_score": fused_risk,
        "min_ttc_seconds": min_ttc,
        "closest_distance_m": closest_dist,
        "closest_track_id": closest_track_id,
        "conflicting_pairs": conflicts or [],
        "uncertainty_score": uncertainty,
        "perception_uncertainty_factor": uncertainty,
        "detection_reliability": 1.0 - uncertainty,
        "timestamp_seconds": timestamp,
    }


# ==============================================================================
# Scenario 1: Normal Road Conditions
# ==============================================================================

def test_scenario_01_normal_road_conditions():
    """Verify that clear weather, safe headway, and no conflicts yield NORMAL state."""
    engine = SafetyRecommendationEngine()
    weather = make_nominal_weather(vis_score=98.0, cond="CLEAR")
    collision = make_nominal_collision(risk_state="LOW", min_ttc=None, closest_dist=50.0, timestamp=1.0)

    rec = engine.evaluate_frame(
        collision_data=collision,
        weather_intel=weather,
        trajectories=[],
        active_tracks=[],
        timestamp_seconds=1.0,
        frame_index=0,
    )

    assert rec.recommendation_state == RecommendationState.NORMAL
    assert rec.priority == WarningPriority.INFO
    assert "Nominal road clearance" in rec.primary_reason
    assert "safe to do so" in rec.disclaimer.lower()


# ==============================================================================
# Scenario 2: Gradually Increasing Collision Risk
# ==============================================================================

def test_scenario_02_gradually_increasing_collision_risk():
    """Verify system steps from NORMAL to CAUTION and REDUCE SPEED as danger increases."""
    engine = SafetyRecommendationEngine(deescalation_confirmation_frames=1)

    # Frame 1: LOW risk, clear road
    w = make_nominal_weather()
    c1 = make_nominal_collision(risk_state="LOW", timestamp=1.0)
    r1 = engine.evaluate_frame(c1, w, timestamp_seconds=1.0, frame_index=0)
    assert r1.recommendation_state == RecommendationState.NORMAL

    # Frame 2: Peripheral vehicle enters, mild uncertainty
    c2 = make_nominal_collision(risk_state="LOW", closest_dist=25.0, closest_track_id=2, uncertainty=0.30, timestamp=1.1)
    tracks = [{"track_id": 2, "class_name": "car"}]
    trajs = [{"track_id": 2, "class_name": "car", "speed_kmh": 20.0}]
    r2 = engine.evaluate_frame(c2, w, trajectories=trajs, active_tracks=tracks, timestamp_seconds=1.1, frame_index=1)
    assert r2.recommendation_state == RecommendationState.CAUTION

    # Frame 3: MODERATE risk, headway narrowing
    c3 = make_nominal_collision(risk_state="MODERATE", min_ttc=4.5, closest_dist=18.0, closest_track_id=2, timestamp=1.2)
    r3 = engine.evaluate_frame(c3, w, trajectories=trajs, active_tracks=tracks, timestamp_seconds=1.2, frame_index=2)
    assert r3.recommendation_state == RecommendationState.REDUCE_SPEED


# ==============================================================================
# Scenario 3: Moderate-to-High Risk Escalation
# ==============================================================================

def test_scenario_03_moderate_to_high_risk_escalation():
    """Verify immediate escalation from REDUCE SPEED to PREPARE TO STOP on HIGH risk."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()

    # Moderate frame
    c_mod = make_nominal_collision(risk_state="MODERATE", min_ttc=4.0, timestamp=1.0)
    r_mod = engine.evaluate_frame(c_mod, w, timestamp_seconds=1.0, frame_index=0)
    assert r_mod.recommendation_state == RecommendationState.REDUCE_SPEED

    # High frame (TTC drops to 3.0s)
    c_high = make_nominal_collision(risk_state="HIGH", min_ttc=3.0, closest_dist=12.0, timestamp=1.1)
    r_high = engine.evaluate_frame(c_high, w, timestamp_seconds=1.1, frame_index=1)
    assert r_high.recommendation_state == RecommendationState.PREPARE_TO_STOP
    assert r_high.priority == WarningPriority.WARNING


# ==============================================================================
# Scenario 4: High-to-Critical Risk Escalation
# ==============================================================================

def test_scenario_04_high_to_critical_risk_escalation():
    """Verify immediate escalation to STOP RECOMMENDED when TTC breaches critical margin."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()

    # Frame 1: HIGH risk
    c1 = make_nominal_collision(risk_state="HIGH", min_ttc=2.8, timestamp=1.0)
    r1 = engine.evaluate_frame(c1, w, timestamp_seconds=1.0, frame_index=0)
    assert r1.recommendation_state == RecommendationState.PREPARE_TO_STOP

    # Frame 2: CRITICAL risk (TTC = 1.4s)
    c2 = make_nominal_collision(risk_state="CRITICAL", min_ttc=1.4, closest_dist=5.2, timestamp=1.1)
    r2 = engine.evaluate_frame(c2, w, timestamp_seconds=1.1, frame_index=1)
    assert r2.recommendation_state == RecommendationState.STOP_RECOMMENDED
    assert r2.priority == WarningPriority.CRITICAL
    assert "Stop vehicle when safe to do so" in r2.recommended_action


# ==============================================================================
# Scenario 5: Vehicle-Pedestrian Trajectory Conflict
# ==============================================================================

def test_scenario_05_vehicle_pedestrian_trajectory_conflict():
    """Verify dedicated pedestrian warning triggers REDUCE SPEED with priority."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    trajs = [{
        "track_id": 5,
        "class_name": "person",
        "speed_kmh": 4.5,
    }]
    conflicts = [{
        "track_a": 1,
        "track_b": 5,
        "class_a": "car",
        "class_b": "person",
        "horizon": "h3",
    }]
    c = make_nominal_collision(
        risk_state="MODERATE",
        closest_dist=10.0,
        timestamp=1.0,
        conflicts=conflicts,
    )

    rec = engine.evaluate_frame(c, w, trajectories=trajs, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.REDUCE_SPEED
    assert "pedestrian" in rec.primary_reason.lower() or "pedestrian" in rec.recommended_action.lower()


# ==============================================================================
# Scenario 6: Vehicle-to-Vehicle Conflict
# ==============================================================================

def test_scenario_06_vehicle_to_vehicle_conflict():
    """Verify inter-vehicle conflict creates trajectory collision advisory."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    conflicts = [{
        "track_a": 10,
        "track_b": 11,
        "class_a": "car",
        "class_b": "truck",
        "horizon": "h5",
        "separation_pixels": 25.0,
    }]
    c = make_nominal_collision(
        risk_state="MODERATE",
        min_ttc=4.2,
        timestamp=1.0,
        conflicts=conflicts,
    )

    rec = engine.evaluate_frame(c, w, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.REDUCE_SPEED
    assert rec.supporting_evidence.trajectory_conflict is True


# ==============================================================================
# Scenario 7: Decreasing TTC
# ==============================================================================

def test_scenario_07_decreasing_ttc():
    """Verify temporal early warning detects rapidly decreasing TTC."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()

    # Feed sequence with plunging TTC
    engine.evaluate_frame(make_nominal_collision(min_ttc=4.5, timestamp=1.0), w, timestamp_seconds=1.0, frame_index=0)
    r2 = engine.evaluate_frame(make_nominal_collision(min_ttc=2.2, timestamp=1.2), w, timestamp_seconds=1.2, frame_index=1)
    r3 = engine.evaluate_frame(
        make_nominal_collision(risk_state="CRITICAL", min_ttc=1.5, closest_dist=4.5, timestamp=1.3),
        w,
        timestamp_seconds=1.3,
        frame_index=2,
    )

    assert r3.recommendation_state == RecommendationState.STOP_RECOMMENDED
    assert r3.min_ttc_seconds == 1.5


# ==============================================================================
# Scenario 8: Invalid TTC
# ==============================================================================

def test_scenario_08_invalid_ttc():
    """Verify that when TTC is None or mathematically invalid, distance & state guide the recommendation."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    c = make_nominal_collision(risk_state="MODERATE", min_ttc=None, closest_dist=12.0, timestamp=1.0)

    rec = engine.evaluate_frame(c, w, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.REDUCE_SPEED
    assert rec.min_ttc_seconds is None


# ==============================================================================
# Scenario 9: Degraded Visibility
# ==============================================================================

def test_scenario_09_degraded_visibility():
    """Verify degraded visibility triggers CAUTION when road is clear, but REDUCE SPEED when road agents exist."""
    engine = SafetyRecommendationEngine()

    # Clear road, but foggy camera (vis = 55%) -> CAUTION
    w_fog = make_nominal_weather(vis_score=55.0, cond="FOG")
    c_clear = make_nominal_collision(risk_state="LOW", timestamp=1.0)
    rec1 = engine.evaluate_frame(c_clear, w_fog, active_tracks=[], timestamp_seconds=1.0, frame_index=0)
    assert rec1.recommendation_state == RecommendationState.CAUTION

    # Same fog, but active vehicle track in roadway narrowing distance -> REDUCE SPEED
    c_agents = make_nominal_collision(risk_state="MODERATE", closest_dist=18.0, closest_track_id=7, timestamp=1.1)
    tracks = [{"track_id": 7, "class_name": "car"}]
    trajs = [{"track_id": 7, "class_name": "car", "speed_kmh": 30.0}]
    rec2 = engine.evaluate_frame(c_agents, w_fog, trajectories=trajs, active_tracks=tracks, timestamp_seconds=1.1, frame_index=1)
    assert rec2.recommendation_state == RecommendationState.REDUCE_SPEED


# ==============================================================================
# Scenario 10: Heavy Rain with No Detected Collision Threat
# ==============================================================================

def test_scenario_10_heavy_rain_with_no_collision_threat():
    """Verify heavy rain alone does NOT trigger STOP or abrupt REDUCE SPEED if no collision threat exists."""
    engine = SafetyRecommendationEngine()
    w_rain = make_nominal_weather(vis_score=60.0, cond="RAIN")
    c_nominal = make_nominal_collision(risk_state="LOW", min_ttc=None, timestamp=1.0)

    rec = engine.evaluate_frame(c_nominal, w_rain, active_tracks=[], timestamp_seconds=1.0, frame_index=0)
    # Must be CAUTION, NOT STOP or abrupt panic
    assert rec.recommendation_state == RecommendationState.CAUTION
    assert rec.priority == WarningPriority.ADVISORY
    assert "RAIN" in rec.primary_reason or "degradation" in rec.primary_reason.lower()


# ==============================================================================
# Scenario 11: Fog with Unreliable Detection
# ==============================================================================

def test_scenario_11_fog_with_unreliable_detection():
    """Verify severe fog causing high uncertainty (> 0.70) triggers UNKNOWN advisory."""
    engine = SafetyRecommendationEngine()
    w_fog = make_nominal_weather(vis_score=35.0, cond="FOG")
    c_fog = make_nominal_collision(risk_state="MODERATE", uncertainty=0.78, timestamp=1.0)

    rec = engine.evaluate_frame(c_fog, w_fog, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.UNKNOWN
    assert rec.status == WarningStatus.DEGRADED
    assert "uncertainty excessive" in rec.primary_reason.lower() or "unreliable" in rec.primary_reason.lower()


# ==============================================================================
# Scenario 12: Missing Trajectory Predictions
# ==============================================================================

def test_scenario_12_missing_trajectory_predictions():
    """Verify missing trajectory predictions when active tracks exist triggers UNKNOWN."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    c = make_nominal_collision(risk_state="LOW", timestamp=1.0)
    tracks = [{"track_id": 9, "class_name": "car"}]

    # Pass None for trajectories when tracks exist
    rec = engine.evaluate_frame(c, w, trajectories=None, active_tracks=tracks, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.UNKNOWN
    assert "trajectory prediction subsystem output is missing" in rec.primary_reason


# ==============================================================================
# Scenario 13: Stale Risk Assessment
# ==============================================================================

def test_scenario_13_stale_risk_assessment():
    """Verify stale collision assessment (age delta > 0.5s) triggers UNKNOWN."""
    engine = SafetyRecommendationEngine(stale_expiration_sec=0.50)
    w = make_nominal_weather()
    # Collision assessment has timestamp 1.0s, but current frame timestamp is 2.2s (delta 1.2s > 0.5s)
    c_stale = make_nominal_collision(risk_state="HIGH", timestamp=1.0)

    rec = engine.evaluate_frame(c_stale, w, timestamp_seconds=2.2, frame_index=0)
    assert rec.recommendation_state == RecommendationState.UNKNOWN
    assert "stale" in rec.primary_reason.lower()


# ==============================================================================
# Scenario 14: Missing Object Tracks
# ==============================================================================

def test_scenario_14_missing_object_tracks():
    """Verify empty active tracks list on clear road resolves cleanly to NORMAL."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    c = make_nominal_collision(risk_state="LOW", timestamp=1.0)

    rec = engine.evaluate_frame(c, w, trajectories=[], active_tracks=[], timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.NORMAL
    assert len(rec.objects_involved) == 0


# ==============================================================================
# Scenario 15: Uncertain Trajectory Prediction
# ==============================================================================

def test_scenario_15_uncertain_trajectory_prediction():
    """Verify elevated trajectory uncertainty reflects in supporting evidence and modulates risk."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    c = make_nominal_collision(risk_state="LOW", uncertainty=0.48, timestamp=1.0)

    rec = engine.evaluate_frame(c, w, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.CAUTION
    assert rec.supporting_evidence.uncertainty_level == "HIGH"


# ==============================================================================
# Scenario 16: Conflicting Upstream Signals
# ==============================================================================

def test_scenario_16_conflicting_upstream_signals():
    """Verify conflicting signals (e.g. CRITICAL risk state but severe uncertainty > 0.70) falls back to UNKNOWN."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()
    c_conflict = make_nominal_collision(risk_state="CRITICAL", uncertainty=0.75, timestamp=1.0)

    rec = engine.evaluate_frame(c_conflict, w, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.UNKNOWN
    # Must NEVER convert UNKNOWN to NORMAL
    assert rec.recommendation_state != RecommendationState.NORMAL


# ==============================================================================
# Scenario 17: Duplicate Warning Suppression
# ==============================================================================

def test_scenario_17_duplicate_warning_suppression():
    """Verify repeated identical recommendations within suppression window are marked SUPPRESSED."""
    engine = SafetyRecommendationEngine(suppression_window_sec=0.35)
    w = make_nominal_weather()
    c = make_nominal_collision(risk_state="MODERATE", min_ttc=4.0, timestamp=1.0)

    # Frame 1: Initial emission -> ACTIVE
    r1 = engine.evaluate_frame(c, w, timestamp_seconds=1.0, frame_index=0)
    assert r1.status == WarningStatus.ACTIVE

    # Frame 2: Identical state 0.05s later -> SUPPRESSED
    c_dup = make_nominal_collision(risk_state="MODERATE", min_ttc=4.0, timestamp=1.05)
    r2 = engine.evaluate_frame(c_dup, w, timestamp_seconds=1.05, frame_index=1)
    assert r2.status == WarningStatus.SUPPRESSED
    assert r2.warning_id == r1.warning_id


# ==============================================================================
# Scenario 18: Warning Escalation
# ==============================================================================

def test_scenario_18_warning_escalation():
    """Verify warning escalation from ADVISORY to WARNING and CRITICAL is immediate (zero delay)."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()

    # Frame 1: REDUCE SPEED (ADVISORY)
    c1 = make_nominal_collision(risk_state="MODERATE", timestamp=1.0)
    r1 = engine.evaluate_frame(c1, w, timestamp_seconds=1.0, frame_index=0)
    assert r1.priority == WarningPriority.ADVISORY

    # Frame 2: Emergency escalation to STOP RECOMMENDED (CRITICAL)
    c2 = make_nominal_collision(risk_state="CRITICAL", min_ttc=1.2, closest_dist=3.8, timestamp=1.1)
    r2 = engine.evaluate_frame(c2, w, timestamp_seconds=1.1, frame_index=1)
    # Immediate escalation with zero frame delay
    assert r2.priority == WarningPriority.CRITICAL
    assert r2.recommendation_state == RecommendationState.STOP_RECOMMENDED
    assert r2.status == WarningStatus.ACTIVE


# ==============================================================================
# Scenario 19: Warning De-escalation
# ==============================================================================

def test_scenario_19_warning_deescalation():
    """Verify de-escalation requires hysteresis confirmation frames to prevent state flicker."""
    # Configure 2-frame confirmation
    engine = SafetyRecommendationEngine(deescalation_confirmation_frames=2)
    w = make_nominal_weather()

    # Frame 1: STOP RECOMMENDED (CRITICAL)
    c_crit = make_nominal_collision(risk_state="CRITICAL", min_ttc=1.2, timestamp=1.0)
    r1 = engine.evaluate_frame(c_crit, w, timestamp_seconds=1.0, frame_index=0)
    assert r1.recommendation_state == RecommendationState.STOP_RECOMMENDED

    # Frame 2: Risk drops to LOW, but 1st frame should be held under hysteresis
    c_low = make_nominal_collision(risk_state="LOW", timestamp=1.1)
    r2 = engine.evaluate_frame(c_low, w, timestamp_seconds=1.1, frame_index=1)
    assert r2.recommendation_state == RecommendationState.STOP_RECOMMENDED  # held by hysteresis

    # Frame 3: 2nd consecutive LOW frame -> confirms de-escalation
    r3 = engine.evaluate_frame(c_low, w, timestamp_seconds=1.2, frame_index=2)
    assert r3.recommendation_state == RecommendationState.NORMAL


# ==============================================================================
# Scenario 20: UNKNOWN-State Behavior
# ==============================================================================

def test_scenario_20_unknown_state_behavior():
    """Verify UNKNOWN state behavior when upstream collision data is None."""
    engine = SafetyRecommendationEngine()
    w = make_nominal_weather()

    rec = engine.evaluate_frame(collision_data=None, weather_intel=w, timestamp_seconds=1.0, frame_index=0)
    assert rec.recommendation_state == RecommendationState.UNKNOWN
    assert rec.priority == WarningPriority.ADVISORY
    assert "missing" in rec.primary_reason.lower()


# ==============================================================================
# Scenario 21: Recovery After Reliable Data Returns
# ==============================================================================

def test_scenario_21_recovery_after_reliable_data_returns():
    """Verify recovery to NORMAL state once reliable upstream telemetry resumes."""
    engine = SafetyRecommendationEngine(deescalation_confirmation_frames=2)
    w = make_nominal_weather()

    # Frame 1: UNKNOWN due to missing data
    r1 = engine.evaluate_frame(collision_data=None, weather_intel=w, timestamp_seconds=1.0, frame_index=0)
    assert r1.recommendation_state == RecommendationState.UNKNOWN

    # Frame 2: Reliable data arrives (hysteresis frame 1)
    c_ok = make_nominal_collision(risk_state="LOW", timestamp=1.1)
    r2 = engine.evaluate_frame(collision_data=c_ok, weather_intel=w, timestamp_seconds=1.1, frame_index=1)

    # Frame 3: Reliable data confirmed (hysteresis frame 2) -> recovers to NORMAL
    r3 = engine.evaluate_frame(collision_data=c_ok, weather_intel=w, timestamp_seconds=1.2, frame_index=2)
    assert r3.recommendation_state == RecommendationState.NORMAL


# ==============================================================================
# Scenario 22: Invalid Timestamps
# ==============================================================================

def test_scenario_22_invalid_timestamps():
    """Verify non-monotonic or negative timestamp jumps are safely handled without crash."""
    engine = SafetyRecommendationEngine(stale_expiration_sec=0.50)
    w = make_nominal_weather()

    # Normal frame
    c1 = make_nominal_collision(risk_state="LOW", timestamp=5.0)
    r1 = engine.evaluate_frame(c1, w, timestamp_seconds=5.0, frame_index=0)
    assert r1.recommendation_state == RecommendationState.NORMAL

    # Erroneous timestamp jump backwards in time (timestamp=1.0 < 5.0)
    c_jump = make_nominal_collision(risk_state="LOW", timestamp=1.0)
    r2 = engine.evaluate_frame(c_jump, w, timestamp_seconds=5.1, frame_index=1)
    # Stale delta is 4.1s > 0.5s -> triggers UNKNOWN
    assert r2.recommendation_state == RecommendationState.UNKNOWN


# ==============================================================================
# Stream Summary Validation
# ==============================================================================

def test_safety_stream_summary_generation():
    """Verify aggregated safety summary counts, transitions, and prototype disclaimer."""
    engine = SafetyRecommendationEngine(deescalation_confirmation_frames=1)
    w = make_nominal_weather()

    # 3 frames: NORMAL -> PREPARE TO STOP -> STOP RECOMMENDED
    engine.evaluate_frame(make_nominal_collision(risk_state="LOW", timestamp=1.0), w, timestamp_seconds=1.0, frame_index=0)
    engine.evaluate_frame(make_nominal_collision(risk_state="HIGH", min_ttc=3.0, timestamp=1.1), w, timestamp_seconds=1.1, frame_index=1)
    engine.evaluate_frame(
        make_nominal_collision(risk_state="CRITICAL", min_ttc=1.5, closest_dist=4.0, timestamp=1.2),
        w,
        timestamp_seconds=1.2,
        frame_index=2,
    )

    summary = engine.generate_stream_summary()
    assert summary.total_recommendations == 3
    assert summary.critical_stop_count == 1
    assert "STOP RECOMMENDED" in summary.recommendation_counts
    assert "safe to do so" in summary.prototype_disclaimer.lower()
    summary_dict = summary.to_dict()
    assert summary_dict["total_recommendations"] == 3


# ==============================================================================
# Video Pipeline Runner Integration Test
# ==============================================================================

def test_video_pipeline_runner_feature5_integration(tmp_path):
    """Verify VideoPipelineRunner integrates Feature 5 into processing loop and packaging."""
    import cv2
    import numpy as np
    from src.video_processing.video_pipeline_runner import VideoPipelineRunner

    # Create synthetic test video file
    video_path = tmp_path / "test_safety_stream.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        out.write(frame)
    out.release()

    runner = VideoPipelineRunner()
    result = runner.run(
        video_path=video_path,
        job_id="test_safety_job",
        output_dir=tmp_path / "output",
        max_frames=5,
    )

    assert result.frames_processed == 5
    # Verify each frame has rich safety recommendation
    for f in result.frames:
        assert "safety_recommendation" in f.__dict__
        rec = f.safety_recommendation
        assert "recommendation_state" in rec
        assert "priority" in rec
        assert "recommended_action" in rec
        assert "primary_reason" in rec

    # Verify overall safety summary packaged
    assert result.safety_summary is not None
    assert "action_counts" in result.safety_summary
    assert "prototype_disclaimer" in result.safety_summary


# ==============================================================================
# FastAPI Safety Recommendations Endpoint Test
# ==============================================================================

def test_fastapi_safety_recommendations_endpoint(tmp_path):
    """Verify GET /api/v1/video/jobs/{job_id}/safety-recommendations is registered and handles job queries."""
    import json
    import uuid
    from src.api.app import app
    from src.api.routers.video import get_video_job_safety_recommendations
    from src.api.job_manager import JobRecord, JobStatus, get_job_manager

    # 1. Verify endpoint is in OpenAPI path specifications
    paths = app.openapi()["paths"]
    assert "/api/v1/video/jobs/{job_id}/safety-recommendations" in paths

    # 2. Test direct invocation when job results exist
    job_mgr = get_job_manager()
    job_id = f"job_test_{uuid.uuid4().hex[:8]}"
    job_out_dir = job_mgr.jobs_dir / job_id
    job_out_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "job_id": job_id,
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "safety_recommendation": {
                    "recommendation_state": "NORMAL",
                    "priority": "INFO",
                    "primary_action": "MAINTAIN_SPEED",
                    "primary_reason": "Clear travel corridor.",
                },
            }
        ],
        "safety_summary": {
            "total_recommendations": 1,
            "most_frequent_recommendation": "NORMAL",
            "most_frequent_action": "MAINTAIN_SPEED",
        },
    }
    with open(job_out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f)

    record = JobRecord(
        job_id=job_id,
        filename="synthetic_video.mp4",
        status=JobStatus.COMPLETED,
        results_dir=str(job_out_dir),
    )
    try:
        with job_mgr._jobs_lock:
            job_mgr._jobs[job_id] = record

        response = get_video_job_safety_recommendations(job_id=job_id)
        assert response["job_id"] == job_id
        assert response["frames_count"] == 1
        assert response["frames"][0]["safety_recommendation"]["recommendation_state"] == "NORMAL"
        assert response["safety_summary"]["most_frequent_recommendation"] == "NORMAL"
    finally:
        import shutil
        with job_mgr._jobs_lock:
            job_mgr._jobs.pop(job_id, None)
        shutil.rmtree(job_out_dir, ignore_errors=True)
