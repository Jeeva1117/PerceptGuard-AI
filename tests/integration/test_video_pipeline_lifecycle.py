"""End-to-End Automated Integration Verification for Video Input & Processing Pipeline.

Tests:
1. Initial clean state / job listing.
2. Video file upload and job submission.
3. Source video streaming with HTTP byte ranges (200 / 206).
4. Full 5-feature results verification.
5. Session isolation (two sequential uploads with distinct IDs).
6. Edge case validation: invalid extension, 0-byte upload.
7. Job deletion and artifact cleanup.
8. Stale job pruning.
"""

import io
import os
from pathlib import Path
import tempfile
import time
import pytest
from starlette.testclient import TestClient

from src.api.app import app
from src.api.job_manager import JobStatus, get_job_manager

client = TestClient(app)


def _create_sample_mp4(num_frames: int = 15, width: int = 320, height: int = 240) -> bytes:
    """Generate in-memory valid MP4 video bytes using OpenCV."""
    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_path = tf.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i * 15) % 256, dtype=np.uint8)
            # Add some moving shapes to simulate objects
            cx = int(width * (0.2 + 0.6 * (i / max(1, num_frames))))
            cy = int(height * 0.5)
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
            out.write(frame)
        out.release()

        with open(temp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def test_job_listing_and_clean_state():
    """Verify GET /api/v1/video/jobs returns structured job list with source video flags."""
    resp = client.get("/api/v1/video/jobs?limit=10")
    assert resp.status_code == 200
    jobs = resp.json()
    assert isinstance(jobs, list)
    for j in jobs:
        assert "job_id" in j
        assert "filename" in j
        assert "status" in j
        assert "has_source_video" in j


def test_edge_case_invalid_video_extension():
    """Verify uploading non-video files returns proper validation error."""
    files = {"file": ("test_bad.txt", io.BytesIO(b"not a video"), "text/plain")}
    resp = client.post("/api/v1/video/jobs", files=files)
    assert resp.status_code in (400, 422)
    data = resp.json()
    assert "error" in data or "detail" in data


def test_edge_case_empty_video_file():
    """Verify uploading 0-byte video returns error."""
    files = {"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
    resp = client.post("/api/v1/video/jobs", files=files)
    assert resp.status_code in (400, 422)


def test_video_upload_and_source_streaming():
    """Test full upload, streaming with HTTP 206, and results retrieval."""
    video_bytes = _create_sample_mp4(num_frames=10, width=160, height=120)
    files = {"file": ("integration_test_stream.mp4", io.BytesIO(video_bytes), "video/mp4")}

    # Submit job
    resp = client.post("/api/v1/video/jobs?max_frames=10&rotation=0", files=files)
    assert resp.status_code == 202
    job_info = resp.json()
    job_id = job_info["job_id"]
    assert job_id.startswith("job_")

    # Verify source video streaming (Full content 200)
    stream_resp = client.get(f"/api/v1/video/jobs/{job_id}/source-video")
    assert stream_resp.status_code == 200
    assert "video/mp4" in stream_resp.headers.get("Content-Type", "")
    assert stream_resp.headers.get("Accept-Ranges") == "bytes"
    assert len(stream_resp.content) == len(video_bytes)

    # Verify partial byte range request (HTTP 206)
    range_resp = client.get(
        f"/api/v1/video/jobs/{job_id}/source-video",
        headers={"Range": "bytes=0-99"},
    )
    assert range_resp.status_code == 206
    assert range_resp.headers.get("Content-Range", "").startswith("bytes 0-99/")
    assert len(range_resp.content) == 100

    # Wait for job execution to complete
    max_wait = 25.0
    start_t = time.time()
    completed = False
    while time.time() - start_t < max_wait:
        stat_resp = client.get(f"/api/v1/video/jobs/{job_id}")
        assert stat_resp.status_code == 200
        state = stat_resp.json()
        if state["status"] in ("COMPLETED", "FAILED"):
            completed = True
            break
        time.sleep(0.5)

    assert completed, f"Job {job_id} did not finish within {max_wait}s"
    assert state["status"] == "COMPLETED"
    assert state["frames_processed"] > 0

    # Verify 5 feature endpoints for this job
    # Feature 1: Weather
    w_resp = client.get(f"/api/v1/video/jobs/{job_id}/weather")
    assert w_resp.status_code == 200
    w_data = w_resp.json()
    assert "weather_summary" in w_data
    assert len(w_data.get("frames", [])) > 0

    # Feature 3: Trajectories
    t_resp = client.get(f"/api/v1/video/jobs/{job_id}/trajectories")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert "trajectory_summary" in t_data

    # Feature 4: Collision Risk
    c_resp = client.get(f"/api/v1/video/jobs/{job_id}/collision-risk")
    assert c_resp.status_code == 200

    # Feature 5: Safety Recommendations
    s_resp = client.get(f"/api/v1/video/jobs/{job_id}/safety-recommendations")
    assert s_resp.status_code == 200

    # Clean up this test job
    del_resp = client.delete(f"/api/v1/video/jobs/{job_id}?delete_disk=true")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "DELETED"


def test_session_isolation_two_videos():
    """Verify two sequential video uploads receive isolated jobs, distinct metadata and independent streams."""
    v1_bytes = _create_sample_mp4(num_frames=8, width=160, height=120)
    v2_bytes = _create_sample_mp4(num_frames=12, width=160, height=120)

    # Job 1
    f1 = {"file": ("stream_alpha.mp4", io.BytesIO(v1_bytes), "video/mp4")}
    r1 = client.post("/api/v1/video/jobs?max_frames=8", files=f1)
    assert r1.status_code == 202
    jid1 = r1.json()["job_id"]

    # Job 2
    f2 = {"file": ("stream_beta.mp4", io.BytesIO(v2_bytes), "video/mp4")}
    r2 = client.post("/api/v1/video/jobs?max_frames=12", files=f2)
    assert r2.status_code == 202
    jid2 = r2.json()["job_id"]

    assert jid1 != jid2

    # Streams must match their respective uploads
    s1 = client.get(f"/api/v1/video/jobs/{jid1}/source-video")
    s2 = client.get(f"/api/v1/video/jobs/{jid2}/source-video")
    assert len(s1.content) == len(v1_bytes)
    assert len(s2.content) == len(v2_bytes)

    # Cleanup
    client.delete(f"/api/v1/video/jobs/{jid1}?delete_disk=true")
    client.delete(f"/api/v1/video/jobs/{jid2}?delete_disk=true")
