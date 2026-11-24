"""End-to-End Five-Feature Integration & Data Flow Verification Test.

Validates the complete PerceptGuard-AI pipeline execution chain:
Raw Video -> Ingestion/Validation -> Frame Extraction ->
Feature 1: Camera & Weather Intelligence ->
Feature 2: Object Detection & Tracking ->
Feature 3: Object & Trajectory Prediction ->
Feature 4: Collision & Road Risk Assessment ->
Feature 5: Early Warning & Vehicle Stop Recommendation ->
Results Packaging & Persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import pytest

from src.video_processing.video_pipeline_runner import VideoPipelineRunner
from src.safety_recommendation.schemas import RecommendationState


@pytest.fixture
def synthetic_traffic_video(tmp_path: Path) -> Path:
    """Generate a deterministic synthetic traffic video with motion and varying visual conditions."""
    video_path = tmp_path / "traffic_clip_synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    width, height = 320, 240
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # 12 frames with a moving bright rectangle simulating a vehicle moving left-to-right
    for i in range(12):
        frame = np.full((height, width, 3), 100 + i * 5, dtype=np.uint8)  # Gradual lighting shift
        # Simulated vehicle box moving horizontally
        x1 = 20 + i * 15
        y1 = 80
        x2 = x1 + 50
        y2 = y1 + 30
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), -1)
        writer.write(frame)

    writer.release()
    return video_path


def test_five_features_end_to_end_data_flow(tmp_path: Path, synthetic_traffic_video: Path):
    """Execute the full 13-stage pipeline and strictly verify all 5 feature data contracts."""
    job_id = "test_e2e_five_features"
    output_dir = tmp_path / "outputs" / job_id

    runner = VideoPipelineRunner()
    result = runner.run(
        video_path=synthetic_traffic_video,
        job_id=job_id,
        output_dir=output_dir,
        max_frames=12,
    )

    # 1. Pipeline Execution Health
    assert result.job_id == job_id
    assert result.frames_processed == 12
    assert result.total_duration_seconds > 0.0

    # 2. Output Artifact Integrity
    results_json = output_dir / "results.json"
    assert results_json.exists(), "results.json was not persisted"
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["job_id"] == job_id
    assert len(data["frames"]) == 12

    # 3. Feature 1 Contract: Camera & Weather Intelligence
    assert "weather_summary" in data
    assert "mean_visibility_score" in data["weather_summary"]
    assert 0.0 <= data["weather_summary"]["mean_visibility_score"] <= 100.0

    for f in data["frames"]:
        weather = f.get("weather_intelligence")
        assert weather is not None, "Frame missing weather_intelligence"
        assert "visibility_score" in weather
        assert 0.0 <= weather["visibility_score"] <= 100.0
        assert "camera_quality_score" in weather
        assert "primary_condition" in weather
        assert "forecast" in weather
        assert "perception_impact" in weather

    # 4. Feature 2 Contract: Object Detection & Tracking
    for f in data["frames"]:
        assert "detections" in f
        assert "tracks" in f
        assert isinstance(f["detections"], list)
        assert isinstance(f["tracks"], list)
        for trk in f["tracks"]:
            assert "track_id" in trk
            assert "bbox_xyxy" in trk
            assert len(trk["bbox_xyxy"]) == 4

    # 5. Feature 3 Contract: Object & Trajectory Prediction
    assert "trajectory_summary" in data
    assert "max_speed_kmh" in data["trajectory_summary"]
    assert "mean_ade_px" in data["trajectory_summary"]

    for f in data["frames"]:
        assert "trajectories" in f
        assert isinstance(f["trajectories"], list)
        for traj in f["trajectories"]:
            assert "track_id" in traj
            assert "predicted_horizons" in traj
            assert "h1" in traj["predicted_horizons"]
            assert "h3" in traj["predicted_horizons"]
            assert "h5" in traj["predicted_horizons"]
            assert "motion_state" in traj

    # 6. Feature 4 Contract: Collision & Road Risk Assessment
    assert "collision_summary" in data
    assert "critical_risk_frames" in data["collision_summary"]
    assert "high_risk_frames" in data["collision_summary"]

    for f in data["frames"]:
        col = f.get("collision_risk")
        assert col is not None, "Frame missing collision_risk"
        assert "risk_state" in col
        assert col["risk_state"] in {"LOW", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN"}
        assert "uncertainty_score" in col

    # 7. Feature 5 Contract: Early Warning & Vehicle Stop Recommendation
    assert "safety_summary" in data
    assert "total_recommendations" in data["safety_summary"]
    assert "prototype_disclaimer" in data["safety_summary"]
    assert "safe to do so" in data["safety_summary"]["prototype_disclaimer"].lower()

    for f in data["frames"]:
        safety = f.get("safety_recommendation")
        assert safety is not None, "Frame missing safety_recommendation"
        assert "recommendation_state" in safety
        assert safety["recommendation_state"] in [
            "NORMAL",
            "CAUTION",
            "REDUCE SPEED",
            "PREPARE TO STOP",
            "STOP RECOMMENDED",
            "UNKNOWN",
        ]
        assert "priority" in safety
        assert "recommended_action" in safety
        assert "primary_reason" in safety
        assert "evidence_chain" in safety
        assert isinstance(safety["evidence_chain"], list)

    # 8. Visual Asset Verification
    raw_frames = list((output_dir / "raw_frames").glob("*.jpg"))
    annotated_frames = list((output_dir / "annotated").glob("*.jpg"))
    assert len(raw_frames) == 12, "Raw frames count mismatch"
    assert len(annotated_frames) == 12, "Annotated frames count mismatch"
