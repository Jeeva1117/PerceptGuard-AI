"""Live end-to-end integration and verification script for PerceptGuard-AI video pipeline.
Tests against http://127.0.0.1:8000.
"""

import io
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request
import urllib.error
import cv2
import numpy as np

BASE_URL = "http://127.0.0.1:8000"


def generate_mp4(num_frames=6, width=160, height=120) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_name = tf.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_name, fourcc, 10.0, (width, height))
        for i in range(num_frames):
            frame = np.full((height, width, 3), 100 + (i * 20) % 155, dtype=np.uint8)
            cv2.circle(frame, (30 + i * 15, 60), 12, (0, 255, 0), -1)
            writer.write(frame)
        writer.release()
        with open(tmp_name, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def post_multipart(url, filename, file_bytes, max_frames=6, rotation=0):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: video/mp4\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    full_url = f"{url}?max_frames={max_frames}&rotation={rotation}"
    req = urllib.request.Request(
        full_url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return urllib.request.urlopen(req)


def run_checks():
    print("=== PERCEPTGUARD-AI VIDEO INPUT PIPELINE VERIFICATION ===")

    # 1. Health
    print("\n[CHECK 1] Testing backend health...")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print(f"PASS: Backend status={data.get('overall_health')}, project={data.get('project_name')}")

    # 2. List jobs
    print("\n[CHECK 2] Listing active jobs in catalog...")
    req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        jobs = json.loads(resp.read().decode())
        print(f"PASS: Retrieved {len(jobs)} jobs. Top entries:")
        for j in jobs[:3]:
            print(f"   - {j['job_id']}: {j['filename']} [{j['status']}] (has_source_video={j.get('has_source_video')})")

    # 3. Invalid file format
    print("\n[CHECK 3] Verifying format rejection on invalid extension...")
    boundary = "----Boundary123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.exe"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
        f"BAD DATA\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/video/jobs",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        print("FAIL: Expected 400/422 on .exe extension")
        assert False
    except urllib.error.HTTPError as err:
        assert err.code in (400, 422)
        print(f"PASS: Correctly rejected invalid file with HTTP {err.code}")

    # 4. Upload valid video
    print("\n[CHECK 4] Uploading valid synthetic MP4 video...")
    mp4_bytes = generate_mp4(num_frames=6, width=160, height=120)
    resp = post_multipart(f"{BASE_URL}/api/v1/video/jobs", "live_test_road_footage.mp4", mp4_bytes, max_frames=6)
    assert resp.status == 202
    job_payload = json.loads(resp.read().decode())
    job_id = job_payload["job_id"]
    print(f"PASS: Submitted job {job_id} successfully (Status: {job_payload['status']})")

    # 5. Full Source Video Stream
    print("\n[CHECK 5] Streaming full source video via GET /jobs/{job_id}/source-video...")
    req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs/{job_id}/source-video")
    with urllib.request.urlopen(req) as vresp:
        assert vresp.status == 200
        ctype = vresp.headers.get("Content-Type")
        clen = int(vresp.headers.get("Content-Length", 0))
        aranges = vresp.headers.get("Accept-Ranges")
        content = vresp.read()
        assert "video/mp4" in ctype
        assert aranges == "bytes"
        assert len(content) == len(mp4_bytes)
        print(f"PASS: Video stream valid. Content-Type={ctype}, Size={clen} bytes, Accept-Ranges={aranges}")

    # 6. HTTP 206 Partial Content (Byte Range)
    print("\n[CHECK 6] Testing HTTP 206 Partial Content byte-range streaming...")
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/video/jobs/{job_id}/source-video",
        headers={"Range": "bytes=0-255"},
    )
    with urllib.request.urlopen(req) as vresp:
        assert vresp.status == 206
        crange = vresp.headers.get("Content-Range")
        clen = int(vresp.headers.get("Content-Length", 0))
        chunk = vresp.read()
        assert len(chunk) == 256
        assert crange.startswith("bytes 0-255/")
        print(f"PASS: Byte range valid. HTTP {vresp.status}, Content-Range={crange}, Chunk={len(chunk)} bytes")

    # 7. Poll until completion
    print("\n[CHECK 7] Polling execution pipeline progress...")
    max_wait = 45.0
    start_t = time.time()
    final_job = None
    while time.time() - start_t < max_wait:
        req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs/{job_id}")
        with urllib.request.urlopen(req) as jresp:
            final_job = json.loads(jresp.read().decode())
            print(f"   Stage: {final_job.get('stage')} | Progress: {final_job.get('progress_percentage')}% | Status: {final_job.get('status')}")
            if final_job.get("status") in ("COMPLETED", "FAILED"):
                break
        time.sleep(1.0)

    assert final_job is not None
    assert final_job.get("status") == "COMPLETED", f"Pipeline failed: {final_job.get('error_message')}"
    print(f"PASS: Processing finished with status=COMPLETED ({final_job.get('frames_processed')} frames processed)")

    # 8. Verify all 5 features on this job
    print("\n[CHECK 8] Verifying all 5 primary feature endpoints on job...")
    endpoints = [
        ("Feature 1: Weather", f"{BASE_URL}/api/v1/video/jobs/{job_id}/weather"),
        ("Feature 2: Tracking", f"{BASE_URL}/api/v1/video/jobs/{job_id}/tracking"),
        ("Feature 3: Trajectories", f"{BASE_URL}/api/v1/video/jobs/{job_id}/trajectories"),
        ("Feature 4: Collision Risk", f"{BASE_URL}/api/v1/video/jobs/{job_id}/collision-risk"),
        ("Feature 5: Safety Recommendations", f"{BASE_URL}/api/v1/video/jobs/{job_id}/safety-recommendations"),
    ]
    for feat_name, url in endpoints:
        with urllib.request.urlopen(url) as r:
            assert r.status == 200
            data = json.loads(r.read().decode())
            print(f"   PASS: {feat_name} -> HTTP 200 (frames: {data.get('frames_count', len(data.get('frames', [])))})")

    # 9. Cleanup job
    print("\n[CHECK 9] Deleting job and verifying storage removal...")
    req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs/{job_id}?delete_disk=true", method="DELETE")
    with urllib.request.urlopen(req) as dresp:
        assert dresp.status == 200
        ddata = json.loads(dresp.read().decode())
        print(f"PASS: Job deletion successful: {ddata.get('status')}")

    # Confirm it is no longer returned
    try:
        urllib.request.urlopen(f"{BASE_URL}/api/v1/video/jobs/{job_id}")
        print("FAIL: Expected 404 after deletion")
        assert False
    except urllib.error.HTTPError as err:
        assert err.code == 404
        print("PASS: Verified job no longer exists (HTTP 404)")

    # 10. Clean stale jobs endpoint
    print("\n[CHECK 10] Testing POST /api/v1/video/jobs/clean-stale...")
    req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs/clean-stale", method="POST")
    with urllib.request.urlopen(req) as cresp:
        assert cresp.status == 200
        cdata = json.loads(cresp.read().decode())
        print(f"PASS: clean-stale returned: {cdata.get('message')}")

    print("\n========================================================")
    print("ALL 10 VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
    print("========================================================")


if __name__ == "__main__":
    run_checks()
