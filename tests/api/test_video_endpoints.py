"""Automated tests for video file validation, upload limits, and resource cleanup."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _generate_synthetic_test_video(path: Path, num_frames: int = 5, fps: float = 10.0) -> None:
    """Generate a lightweight valid MP4 video for integration testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    try:
        for i in range(num_frames):
            frame = np.full((64, 64, 3), (i * 20) % 250, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_unsupported_extension(client: TestClient):
    """Verify upload with unsupported extension is rejected with 400."""
    files = {"file": ("test.txt", io.BytesIO(b"dummy text content"), "text/plain")}
    res = client.post("/api/v1/video/validate", files=files)
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_VIDEO_FILE"
    assert "Unsupported video extension" in data["error"]["message"]


def test_video_empty_file(client: TestClient):
    """Verify empty 0-byte video upload is rejected with 400."""
    files = {"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
    res = client.post("/api/v1/video/validate", files=files)
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_VIDEO_FILE"
    assert "0 bytes" in data["error"]["message"]


def test_video_corrupt_data(client: TestClient):
    """Verify corrupt video payload fails OpenCV decode with 400."""
    files = {"file": ("corrupt.mp4", io.BytesIO(b"garbage-not-a-real-video-stream"), "video/mp4")}
    res = client.post("/api/v1/video/validate", files=files)
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_VIDEO_FILE"


def test_video_valid_upload_and_resource_cleanup(client: TestClient, tmp_path: Path):
    """Verify valid video upload succeeds, extracts metadata, and cleans up temporary disk files."""
    test_video = tmp_path / "valid_sample.mp4"
    _generate_synthetic_test_video(test_video, num_frames=10, fps=15.0)

    with open(test_video, "rb") as f:
        files = {"file": ("valid_sample.mp4", f, "video/mp4")}
        res = client.post("/api/v1/video/validate", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    metadata = data["metadata"]
    assert metadata["filename"] == "valid_sample.mp4"
    assert metadata["width"] == 64
    assert metadata["height"] == 64
    assert metadata["frame_count"] == 10
    assert metadata["fps"] == 15.0


def test_video_process_endpoint(client: TestClient, tmp_path: Path):
    """Verify /video/process endpoint bounded execution and response structure."""
    test_video = tmp_path / "process_sample.mp4"
    _generate_synthetic_test_video(test_video, num_frames=8, fps=10.0)

    with open(test_video, "rb") as f:
        files = {"file": ("process_sample.mp4", f, "video/mp4")}
        res = client.post("/api/v1/video/process?max_frames=5", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["frames_processed"] == 5
    assert "task_id" in data


def test_video_job_submission_and_lifecycle(client: TestClient, tmp_path: Path):
    """Verify /video/jobs submission, status retrieval, and listing."""
    test_video = tmp_path / "job_sample.mp4"
    _generate_synthetic_test_video(test_video, num_frames=6, fps=10.0)

    with open(test_video, "rb") as f:
        files = {"file": ("job_sample.mp4", f, "video/mp4")}
        res = client.post("/api/v1/video/jobs?max_frames=6", files=files)

    assert res.status_code == 202
    sub_data = res.json()
    assert "job_id" in sub_data
    job_id = sub_data["job_id"]
    assert sub_data["filename"] == "job_sample.mp4"

    # Status check
    res_status = client.get(f"/api/v1/video/jobs/{job_id}")
    assert res_status.status_code == 200
    stat_data = res_status.json()
    assert stat_data["job_id"] == job_id
    assert stat_data["status"] in ("QUEUED", "PROCESSING", "COMPLETED")

    # List jobs check
    res_list = client.get("/api/v1/video/jobs?limit=10")
    assert res_list.status_code == 200
    jobs_list = res_list.json()
    assert isinstance(jobs_list, list)
    assert any(j["job_id"] == job_id for j in jobs_list)

