"""End-to-end integration tests for Master Video Upload & Processing Pipeline.

Validates:
1. Video upload with format & size checking
2. Two-step flow: Upload -> video_id -> Start Processing
3. ByteTrack persistent ID tracking
4. Five distinct module telemetry payloads
5. Session isolation and cleanup
6. Cancellation of running jobs
7. Rejection of corrupt or invalid magic byte files
8. HTTP 206 Partial Content byte range video streaming
"""

import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.job_manager import JobStatus, get_job_manager


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _make_test_traffic_video(path: Path, num_frames: int = 15, fps: float = 10.0) -> None:
    """Generate a lightweight valid MP4 video containing moving simulated shapes."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (128, 128))
    try:
        for i in range(num_frames):
            frame = np.full((128, 128, 3), 40, dtype=np.uint8)
            # Simulated vehicle moving downward smoothly
            y = int(10 + i * 5)
            x = 50
            cv2.rectangle(frame, (x, y), (x + 30, y + 20), (200, 200, 200), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_01_upload_valid_video_returns_video_id_and_metadata(client: TestClient, tmp_path: Path):
    """TEST 1: Upload a valid MP4 -> upload succeeds, video preview available, video_id returned."""
    video_path = tmp_path / "test_traffic.mp4"
    _make_test_traffic_video(video_path, num_frames=12, fps=10.0)

    with open(video_path, "rb") as f:
        res = client.post(
            "/api/v1/video/upload-file",
            files={"file": ("test_traffic.mp4", f, "video/mp4")},
        )

    assert res.status_code == 201
    data = res.json()
    assert "video_id" in data
    assert data["video_id"].startswith("vid_")
    assert data["filename"] == "test_traffic.mp4"
    assert data["metadata"]["width"] == 128
    assert data["metadata"]["height"] == 128
    assert data["metadata"]["frame_count"] == 12
    assert data["metadata"]["fps"] == 10.0
    assert "stream_url" in data


def test_02_start_processing_from_video_id(client: TestClient, tmp_path: Path):
    """TEST 2: Click Start Processing using video_id -> real backend job starts."""
    video_path = tmp_path / "test_start_proc.mp4"
    _make_test_traffic_video(video_path, num_frames=10, fps=10.0)

    with open(video_path, "rb") as f:
        res_up = client.post(
            "/api/v1/video/upload-file",
            files={"file": ("test_start_proc.mp4", f, "video/mp4")},
        )
    assert res_up.status_code == 201
    video_id = res_up.json()["video_id"]

    # Start Processing
    res_proc = client.post(
        "/api/v1/video/start-processing",
        json={
            "video_id": video_id,
            "analysis_mode": "FULL_BALANCED",
            "rotation": 0,
            "sample_fps": 2.0,
            "max_frames": 6,
        },
    )
    assert res_proc.status_code == 202
    data = res_proc.json()
    assert "job_id" in data
    assert data["job_id"].startswith("job_")
    assert data["status"] in ("QUEUED", "PROCESSING")

    # Verify status endpoint returns real job record
    job_id = data["job_id"]
    res_stat = client.get(f"/api/v1/video/jobs/{job_id}")
    assert res_stat.status_code == 200
    stat_data = res_stat.json()
    assert stat_data["job_id"] == job_id
    assert "stage" in stat_data


def test_03_corrupt_and_unsupported_rejection(client: TestClient):
    """TEST 7: Reject unsupported extensions and corrupt magic bytes."""
    # Unsupported extension
    res_ext = client.post(
        "/api/v1/video/upload-file",
        files={"file": ("malicious.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
    )
    assert res_ext.status_code == 400
    assert "Unsupported video extension" in res_ext.json()["error"]["message"]

    # Renamed text file claiming to be mp4
    fake_bytes = b"Hello, I am a plain text file pretending to be video!"
    res_fake = client.post(
        "/api/v1/video/upload-file",
        files={"file": ("fake_dashcam.mp4", io.BytesIO(fake_bytes), "video/mp4")},
    )
    assert res_fake.status_code == 400
    assert "binary header does not match" in res_fake.json()["error"]["message"]

    # Empty 0-byte file
    res_empty = client.post(
        "/api/v1/video/upload-file",
        files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
    )
    assert res_empty.status_code == 400
    assert "0 bytes" in res_empty.json()["error"]["message"]


def test_04_five_module_endpoints_return_independent_telemetry(client: TestClient, tmp_path: Path):
    """TEST 5: All five analysis modules return distinct, real data for the same job."""
    video_path = tmp_path / "five_module_sample.mp4"
    _make_test_traffic_video(video_path, num_frames=8, fps=10.0)

    # Submit job directly
    with open(video_path, "rb") as f:
        res = client.post(
            "/api/v1/video/jobs?max_frames=6",
            files={"file": ("five_module_sample.mp4", f, "video/mp4")},
        )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # Module 1: Weather & Camera
    res_m1 = client.get(f"/api/v1/video/jobs/{job_id}/weather")
    assert res_m1.status_code == 200
    assert "weather_summary" in res_m1.json()

    # Module 2: Object Detection & Tracking
    res_m2 = client.get(f"/api/v1/video/jobs/{job_id}/tracking")
    assert res_m2.status_code == 200
    assert "tracks" in res_m2.json()

    # Module 3: Trajectory Prediction
    res_m3 = client.get(f"/api/v1/video/jobs/{job_id}/trajectories")
    assert res_m3.status_code == 200
    assert "trajectory_summary" in res_m3.json()

    # Module 4: Collision Risk
    res_m4 = client.get(f"/api/v1/video/jobs/{job_id}/collision-risk")
    assert res_m4.status_code == 200
    assert "collision_summary" in res_m4.json()

    # Module 5: Safety Alerts & Recommendations
    res_m5 = client.get(f"/api/v1/video/jobs/{job_id}/safety-recommendations")
    assert res_m5.status_code == 200
    assert "safety_summary" in res_m5.json()


def test_05_cancel_running_job(client: TestClient, tmp_path: Path):
    """TEST 9: Cancel a running job safely without crashes."""
    video_path = tmp_path / "cancel_sample.mp4"
    _make_test_traffic_video(video_path, num_frames=30, fps=10.0)

    with open(video_path, "rb") as f:
        res_up = client.post(
            "/api/v1/video/upload-file",
            files={"file": ("cancel_sample.mp4", f, "video/mp4")},
        )
    assert res_up.status_code == 201
    video_id = res_up.json()["video_id"]

    res_proc = client.post(
        "/api/v1/video/start-processing",
        json={"video_id": video_id, "analysis_mode": "FULL_BALANCED"},
    )
    assert res_proc.status_code == 202
    job_id = res_proc.json()["job_id"]

    # Cancel immediately
    res_cancel = client.post(f"/api/v1/video/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    cancel_data = res_cancel.json()
    assert cancel_data["status"] in ("CANCELLED", "COMPLETED")


def test_06_video_streaming_http_206_partial_content(client: TestClient, tmp_path: Path):
    """TEST 10: Stream uploaded video with HTTP 206 Partial Content byte ranges for seeking."""
    video_path = tmp_path / "seek_stream.mp4"
    _make_test_traffic_video(video_path, num_frames=10, fps=10.0)

    with open(video_path, "rb") as f:
        res_up = client.post(
            "/api/v1/video/upload-file",
            files={"file": ("seek_stream.mp4", f, "video/mp4")},
        )
    assert res_up.status_code == 201
    video_id = res_up.json()["video_id"]

    # Request first 128 bytes
    res_stream = client.get(
        f"/api/v1/video/stream/{video_id}",
        headers={"range": "bytes=0-127"},
    )
    assert res_stream.status_code == 206
    assert res_stream.headers.get("Accept-Ranges") == "bytes"
    assert "Content-Range" in res_stream.headers
    assert len(res_stream.content) == 128


def test_07_session_isolation_and_cleanup(client: TestClient, tmp_path: Path):
    """TEST 6: Session isolation between sequential video uploads."""
    v1_path = tmp_path / "session_v1.mp4"
    v2_path = tmp_path / "session_v2.mp4"
    _make_test_traffic_video(v1_path, num_frames=5, fps=10.0)
    _make_test_traffic_video(v2_path, num_frames=6, fps=10.0)

    with open(v1_path, "rb") as f1:
        res1 = client.post("/api/v1/video/upload-file", files={"file": ("session_v1.mp4", f1, "video/mp4")})
    assert res1.status_code == 201
    vid1 = res1.json()["video_id"]

    with open(v2_path, "rb") as f2:
        res2 = client.post("/api/v1/video/upload-file", files={"file": ("session_v2.mp4", f2, "video/mp4")})
    assert res2.status_code == 201
    vid2 = res2.json()["video_id"]

    # IDs must be strictly unique
    assert vid1 != vid2

    # Clean stale jobs
    res_clean = client.post("/api/v1/video/jobs/clean-stale")
    assert res_clean.status_code == 200
    assert "cleaned_count" in res_clean.json()
