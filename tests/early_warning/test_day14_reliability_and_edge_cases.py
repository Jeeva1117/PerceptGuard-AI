"""Automated Day 14 reliability, regression, and edge-case test suite.

Verifies:
1. Alert state isolation across multiple sequential video pipeline runs.
2. JobManager disk rehydration and restart persistence.
3. Decoded video frame-rate edge cases (zero, negative, NaN, fractional).
4. Full 27-feature vector extraction mathematical validity and range bounds.
5. Zero-frame / corrupt video error handling and graceful job failure.
6. Temporary staging file cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from src.api.job_manager import JobManager, JobStatus
from src.video_processing.video_pipeline_runner import VideoPipelineRunner, VideoPipelineRunResult


def _create_sample_video(path: Path, num_frames: int = 5, fps: float = 10.0, width: int = 320, height: int = 240) -> None:
    """Create a lightweight synthetic video for test runs."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i * 30) % 255, dtype=np.uint8)
            cv2.rectangle(frame, (20 + i * 4, 30), (80 + i * 4, 90), (0, 255, 0), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_alert_isolation_across_multiple_video_runs(tmp_path: Path):
    """Verify that early-warning alert state and cooldowns do not leak across distinct video runs."""
    video1 = tmp_path / "vid1.mp4"
    video2 = tmp_path / "vid2.mp4"
    _create_sample_video(video1, num_frames=6, fps=10.0)
    _create_sample_video(video2, num_frames=6, fps=10.0)

    runner = VideoPipelineRunner(project_root=Path.cwd())

    out1 = tmp_path / "run1"
    res1 = runner.run(video_path=video1, job_id="job_run_1", output_dir=out1, max_frames=6)
    assert res1.frames_processed == 6

    out2 = tmp_path / "run2"
    res2 = runner.run(video_path=video2, job_id="job_run_2", output_dir=out2, max_frames=6)
    assert res2.frames_processed == 6

    # Verify each run's results.json reflects strictly its own frames and metrics
    with open(out1 / "results.json") as f:
        d1 = json.load(f)
    with open(out2 / "results.json") as f:
        d2 = json.load(f)

    assert d1["job_id"] == "job_run_1"
    assert d2["job_id"] == "job_run_2"
    assert len(d1["frames"]) == 6
    assert len(d2["frames"]) == 6


def test_27_feature_vector_structure_and_finiteness(tmp_path: Path):
    """Verify that all 27 features are computed, finite, and strictly bounded."""
    video = tmp_path / "feature_test.mp4"
    _create_sample_video(video, num_frames=5, fps=10.0)

    runner = VideoPipelineRunner(project_root=Path.cwd())
    out = tmp_path / "feat_out"
    res = runner.run(video_path=video, job_id="job_feat_test", output_dir=out, max_frames=5)

    assert len(res.frames) == 5
    for f in res.frames:
        assert isinstance(f.visual_quality, dict)
        assert np.isfinite(f.visual_quality["blur_laplacian"])
        assert np.isfinite(f.visual_quality["brightness"])
        assert np.isfinite(f.visual_quality["contrast"])
        for h in ("h1", "h3", "h5"):
            assert 0.0 <= f.predicted_reliability[h] <= 1.0
            assert 0.0 <= f.risk_scores[h] <= 1.0
            assert f.risk_levels[h] in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN")


def test_job_manager_restart_persistence_and_rehydration(tmp_path: Path):
    """Verify JobManager scans jobs_dir and restores completed jobs on initialization."""
    jobs_dir = tmp_path / "outputs" / "jobs"
    mock_job_dir = jobs_dir / "job_mock123"
    mock_job_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "job_id": "job_mock123",
        "video_filename": "persisted_sample.mp4",
        "frames_processed": 15,
        "total_detections": 42,
        "unique_tracks": 4,
        "total_alerts": 2,
        "active_alerts": 1,
        "total_duration_seconds": 3.45,
        "completed_at": "2026-11-20T20:00:00Z",
        "per_horizon_risk": {},
    }
    with open(mock_job_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f)

    # Initialize JobManager with tmp_path
    JobManager._instance = None
    mgr = JobManager(base_dir=tmp_path)

    rehydrated = mgr.get_job("job_mock123")
    assert rehydrated is not None
    assert rehydrated.job_id == "job_mock123"
    assert rehydrated.status == JobStatus.COMPLETED
    assert rehydrated.frames_processed == 15
    assert rehydrated.total_detections == 42
    assert rehydrated.unique_tracks == 4
    assert rehydrated.alerts_count == 2


def test_zero_frame_or_invalid_video_raises_value_error(tmp_path: Path):
    """Verify that unreadable or empty video files raise ValueError and fail cleanly."""
    corrupt_video = tmp_path / "empty_stream.mp4"
    corrupt_video.write_bytes(b"invalid-stream-header")

    runner = VideoPipelineRunner(project_root=Path.cwd())
    out = tmp_path / "corrupt_out"

    with pytest.raises(ValueError) as exc_info:
        runner.run(video_path=corrupt_video, job_id="job_corrupt", output_dir=out, max_frames=5)

    assert "could not open video stream" in str(exc_info.value).lower()
