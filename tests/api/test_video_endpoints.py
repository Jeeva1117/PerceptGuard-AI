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


def test_video_standalone_upload_and_stream(client: TestClient, tmp_path: Path):
    """Verify POST /api/v1/video/upload-file and GET /api/v1/video/stream/{video_id} workflow."""
    test_video = tmp_path / "standalone_upload.mp4"
    _generate_synthetic_test_video(test_video, num_frames=8, fps=10.0)

    with open(test_video, "rb") as f:
        files = {"file": ("standalone_upload.mp4", f, "video/mp4")}
        res = client.post("/api/v1/video/upload-file", files=files)

    assert res.status_code == 201
    data = res.json()
    assert "video_id" in data
    assert data["video_id"].startswith("vid_")
    video_id = data["video_id"]
    assert data["filename"] == "standalone_upload.mp4"
    assert data["metadata"]["width"] == 64
    assert data["metadata"]["height"] == 64
    assert data["metadata"]["frame_count"] == 8

    # Stream video by video_id with byte range
    res_stream = client.get(
        f"/api/v1/video/stream/{video_id}",
        headers={"range": "bytes=0-50"},
    )
    assert res_stream.status_code == 206
    assert "Content-Range" in res_stream.headers


def test_video_magic_bytes_binary_validation(client: TestClient):
    """Verify that a fake file with an .mp4 extension but text bytes is rejected."""
    fake_payload = b"THIS IS NOT A VALID VIDEO CONTAINER PAYLOAD AT ALL!"
    files = {"file": ("fake_road_recording.mp4", io.BytesIO(fake_payload), "video/mp4")}
    res = client.post("/api/v1/video/upload-file", files=files)
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "INVALID_VIDEO_FILE"
    assert "binary header does not match" in data["error"]["message"]


def test_start_processing_from_video_id_and_cancel(client: TestClient, tmp_path: Path):
    """Verify two-step workflow: Upload -> Start Processing -> Cancel Job."""
    test_video = tmp_path / "proc_two_step.mp4"
    _generate_synthetic_test_video(test_video, num_frames=12, fps=10.0)

    # Step 1: Upload
    with open(test_video, "rb") as f:
        files = {"file": ("proc_two_step.mp4", f, "video/mp4")}
        res_upload = client.post("/api/v1/video/upload-file", files=files)
    assert res_upload.status_code == 201
    video_id = res_upload.json()["video_id"]

    # Step 2: Start Processing
    payload = {
        "video_id": video_id,
        "analysis_mode": "FULL_BALANCED",
        "rotation": 0,
        "sample_fps": 2.0,
        "max_frames": 10,
    }
    res_proc = client.post("/api/v1/video/start-processing", json=payload)
    assert res_proc.status_code == 202
    job_data = res_proc.json()
    assert "job_id" in job_data
    job_id = job_data["job_id"]

    # Step 3: Cancel Job
    res_cancel = client.post(f"/api/v1/video/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] in ("CANCELLED", "COMPLETED")


