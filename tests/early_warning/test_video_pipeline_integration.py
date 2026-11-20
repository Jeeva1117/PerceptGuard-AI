"""Automated end-to-end integration tests for video pipeline processing and dashboard integration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest

from src.api.job_manager import JobManager, JobStatus
from src.video_processing.video_pipeline_runner import VideoPipelineRunner, VideoPipelineRunResult
from dashboard.api_client import DashboardApiClient


def _create_test_video(path: Path, num_frames: int = 10, fps: float = 10.0) -> None:
    """Helper creating a valid test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    try:
        for i in range(num_frames):
            frame = np.full((240, 320, 3), (i * 25) % 255, dtype=np.uint8)
            # Draw a simple box simulating an object
            cv2.rectangle(frame, (50 + i * 5, 50), (120 + i * 5, 120), (0, 255, 0), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_video_pipeline_runner_execution(tmp_path: Path):
    """Verify VideoPipelineRunner processes frames, computes features, and produces results.json."""
    test_video = tmp_path / "test_run.mp4"
    _create_test_video(test_video, num_frames=8, fps=10.0)

    runner = VideoPipelineRunner(project_root=Path.cwd())
    out_dir = tmp_path / "pipeline_output"

    stages_recorded = []
    def _track_progress(stage, pct, msg):
        stages_recorded.append((stage, pct))

    result = runner.run(
        video_path=test_video,
        job_id="test_job_001",
        output_dir=out_dir,
        max_frames=8,
        progress_callback=_track_progress,
    )

    assert isinstance(result, VideoPipelineRunResult)
    assert result.frames_processed == 8
    assert result.calibration_status == "PROVISIONAL CALIBRATION"
    assert "h1" in result.per_horizon_risk
    assert "h3" in result.per_horizon_risk
    assert "h5" in result.per_horizon_risk

    # Verify disk artifacts
    assert (out_dir / "results.json").exists()
    assert (out_dir / "raw_frames" / "frame_000000.jpg").exists()
    assert (out_dir / "annotated" / "annotated_000000.jpg").exists()
    assert any(s[0] == "COMPLETED" for s in stages_recorded)


def test_job_manager_lifecycle(tmp_path: Path):
    """Verify JobManager submits, tracks progress, and completes video jobs asynchronously."""
    test_video = tmp_path / "test_job_video.mp4"
    _create_test_video(test_video, num_frames=6, fps=10.0)

    JobManager._instance = None  # Reset singleton for testing
    mgr = JobManager(base_dir=tmp_path)
    job = mgr.submit_job(video_path=test_video, filename="test_job_video.mp4", max_frames=6)

    assert job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
    assert job.job_id.startswith("job_")

    # Wait for completion (bounded timeout)
    max_wait = 20.0
    t0 = time.time()
    while time.time() - t0 < max_wait:
        rec = mgr.get_job(job.job_id)
        if rec and rec.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.5)

    final_rec = mgr.get_job(job.job_id)
    assert final_rec is not None
    assert final_rec.status == JobStatus.COMPLETED
    assert final_rec.frames_processed == 6

    # Test result retrieval
    results = mgr.get_job_results(job.job_id)
    assert results is not None
    assert results["job_id"] == job.job_id


def test_dashboard_api_client_video_methods(tmp_path: Path):
    """Verify DashboardApiClient submit, status, and list methods in fallback mode."""
    test_video = tmp_path / "client_video.mp4"
    _create_test_video(test_video, num_frames=5, fps=10.0)

    client = DashboardApiClient(use_in_process_fallback=True)
    with open(test_video, "rb") as f:
        file_bytes = f.read()

    # Submit job
    sub_res = client.submit_video_job(file_bytes=file_bytes, filename="client_video.mp4", max_frames=5)
    assert "job_id" in sub_res
    job_id = sub_res["job_id"]

    # Status
    stat = client.get_video_job_status(job_id)
    assert stat["job_id"] == job_id

    # List
    jobs = client.list_video_jobs(limit=5)
    assert isinstance(jobs, list)
    assert any(j["job_id"] == job_id for j in jobs)
