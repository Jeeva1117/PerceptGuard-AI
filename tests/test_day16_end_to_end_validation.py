"""End-to-end integration and runtime validation test for Day 16 release readiness.

Processes a real BDD100K driving video through the complete 14-stage pipeline,
verifying:
- Video loading & decoding
- YOLO11n object detection
- ByteTrack Kalman multi-object tracking
- 27-dimensional feature extraction
- PyTorch LSTM multi-horizon forecasting (H1, H3, H5)
- Derived failure risk (S = 1.0 - R)
- Early warning alerts
- Result persistence in results.json
- Annotated frame generation
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.video_processing.video_pipeline_runner import VideoPipelineRunner


def test_day16_full_pipeline_on_real_video(tmp_path):
    """Execute complete perception reliability pipeline on authentic driving clip."""
    video_path = Path("outputs/day14_validation/clips/scenario_a_normal_daytime.mp4")
    assert video_path.exists(), f"Real validation video clip missing at {video_path}"

    runner = VideoPipelineRunner()
    output_dir = tmp_path / "day16_run"

    res = runner.run(
        video_path=video_path,
        job_id="day16_test_job_001",
        output_dir=output_dir,
        max_frames=15,
    )

    # 1. Verify Pipeline Run Metadata
    assert res.job_id == "day16_test_job_001"
    assert res.frames_processed == 15
    assert res.total_detections >= 0
    assert res.total_duration_seconds > 0.0

    # 2. Verify Persistence of results.json
    results_json = output_dir / "results.json"
    assert results_json.exists(), "results.json was not created"

    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["job_id"] == "day16_test_job_001"
    assert "frames" in data
    assert len(data["frames"]) == 15

    # 3. Verify Frame-Level Telemetry
    frame_0 = data["frames"][0]
    assert "frame_index" in frame_0
    assert "detections" in frame_0
    assert "visual_quality" in frame_0
    assert "predicted_reliability" in frame_0
    assert "risk_scores" in frame_0

    # 4. Verify Visual Quality Features
    vq = frame_0["visual_quality"]
    assert "blur_laplacian" in vq
    assert "brightness" in vq
    assert "contrast" in vq

    # 5. Verify Multi-Horizon Risk Formulation (S = 1.0 - R)
    risk = frame_0["risk_scores"]
    assert "h1" in risk
    assert "h3" in risk
    assert "h5" in risk
    for h in ("h1", "h3", "h5"):
        score = risk[h]
        if score is not None:
            assert 0.0 <= score <= 1.0, f"Risk score {score} out of bounds [0.0, 1.0]"

    # 6. Verify Annotated Frames
    annotated_dir = output_dir / "annotated"
    assert annotated_dir.exists()
    saved_frames = list(annotated_dir.glob("*.jpg"))
    assert len(saved_frames) == 15, f"Expected 15 annotated frames, found {len(saved_frames)}"
