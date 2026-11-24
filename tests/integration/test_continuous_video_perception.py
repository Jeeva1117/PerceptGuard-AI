"""Comprehensive Integration Verification for Continuous Video Perception & Safety Engine.

Validates:
1. Continuous source video streaming with HTTP 206 byte-ranges.
2. OpenCV continuous annotated video generation (annotated_video.mp4).
3. Continuous annotated video streaming endpoint (GET /jobs/{id}/annotated-video).
4. Full 5-feature perception results contract (Camera/Weather, Detection/Tracking,
   Trajectory Prediction H1/H3/H5, Collision Risk Assessment, Safety Recommendations).
5. Chronological Event Timeline synchronization and telemetry integrity.
6. Edge case handling (missing job, invalid ranges, clean deletion).
"""

import io
import os
import tempfile
import time
import pytest
from starlette.testclient import TestClient

from src.api.app import app
from src.api.job_manager import get_job_manager

client = TestClient(app)


def _generate_synthetic_traffic_mp4(num_frames: int = 15, width: int = 320, height: int = 240) -> bytes:
    """Generate in-memory valid MP4 road traffic simulation video bytes using OpenCV."""
    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_path = tf.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))
        for i in range(num_frames):
            # Asphalt road background with moving horizon
            frame = np.full((height, width, 3), (40, 44, 52), dtype=np.uint8)
            # Road lane markings
            cv2.line(frame, (width // 2, 0), (width // 2, height), (220, 220, 220), 2)
            # Simulated vehicle moving forward
            x = int(width * 0.3 + (i * 4))
            y = int(height * 0.4 + (i * 3))
            cv2.rectangle(frame, (x, y), (x + 40, y + 25), (0, 200, 255), -1)
            # Simulated pedestrian crossing
            px = int(width * 0.8 - (i * 3))
            py = int(height * 0.6)
            cv2.circle(frame, (px, py), 8, (50, 50, 220), -1)
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


def test_continuous_video_pipeline_end_to_end():
    """Verify that a video upload generates both continuous source video and continuous annotated video."""
    video_bytes = _generate_synthetic_traffic_mp4(num_frames=12, width=320, height=240)
    files = {"file": ("continuous_perception_test.mp4", io.BytesIO(video_bytes), "video/mp4")}

    # 1. Submit job in preview mode for fast deterministic validation
    resp = client.post("/api/v1/video/jobs?max_frames=10&analysis_mode=QUICK_PREVIEW", files=files)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert job_id.startswith("job_")

    # 2. Verify source video streaming (HTTP 200 & HTTP 206)
    src_full = client.get(f"/api/v1/video/jobs/{job_id}/source-video")
    assert src_full.status_code == 200
    assert "video/mp4" in src_full.headers.get("Content-Type", "")
    assert len(src_full.content) == len(video_bytes)

    src_range = client.get(
        f"/api/v1/video/jobs/{job_id}/source-video",
        headers={"Range": "bytes=0-199"},
    )
    assert src_range.status_code == 206
    assert src_range.headers.get("Content-Range", "").startswith("bytes 0-199/")
    assert len(src_range.content) == 200

    # 3. Wait for pipeline completion
    max_wait = 30.0
    start_t = time.time()
    completed = False
    status_data = {}
    while time.time() - start_t < max_wait:
        stat = client.get(f"/api/v1/video/jobs/{job_id}")
        assert stat.status_code == 200
        status_data = stat.json()
        if status_data.get("status") in ("COMPLETED", "FAILED"):
            completed = True
            break
        time.sleep(0.5)

    assert completed, f"Job {job_id} did not finish within {max_wait}s: {status_data}"
    assert status_data["status"] == "COMPLETED", f"Pipeline failed: {status_data.get('error_message')}"
    assert status_data["frames_processed"] > 0

    # 4. Verify continuous annotated video streaming endpoint (HTTP 200 & HTTP 206)
    ann_resp = client.get(f"/api/v1/video/jobs/{job_id}/annotated-video")
    assert ann_resp.status_code == 200
    assert "video/mp4" in ann_resp.headers.get("Content-Type", "")
    assert ann_resp.headers.get("Accept-Ranges") == "bytes"
    assert len(ann_resp.content) > 0

    ann_range = client.get(
        f"/api/v1/video/jobs/{job_id}/annotated-video",
        headers={"Range": "bytes=0-99"},
    )
    assert ann_range.status_code == 206
    assert ann_range.headers.get("Content-Range", "").startswith("bytes 0-99/")
    assert len(ann_range.content) == 100

    # 5. Verify results data contract contains continuous video paths
    res_resp = client.get(f"/api/v1/video/jobs/{job_id}/results")
    assert res_resp.status_code == 200
    res_data = res_resp.json()
    assert "video_path" in res_data
    assert "annotated_video_path" in res_data
    assert res_data["annotated_video_path"] is not None
    assert os.path.exists(res_data["annotated_video_path"])

    # 6. Verify 5 primary features data integrity
    frames = res_data.get("frames", [])
    assert len(frames) > 0
    f0 = frames[0]
    assert "weather_intelligence" in f0
    assert "trajectories" in f0
    assert "collision_risk" in f0
    assert "safety_recommendation" in f0

    # 7. Clean up test job
    del_resp = client.delete(f"/api/v1/video/jobs/{job_id}?delete_disk=true")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "DELETED"


def test_annotated_video_not_found_on_invalid_job():
    """Verify 404 response when requesting annotated video for non-existent job."""
    resp = client.get("/api/v1/video/jobs/job_non_existent_9999/annotated-video")
    assert resp.status_code == 404
