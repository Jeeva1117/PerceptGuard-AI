"""Script to verify the end-to-end live video upload and processing pipeline on localhost:8000."""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import cv2

BACKEND_URL = "http://127.0.0.1:8000"

def create_sample_video(path: Path) -> None:
    """Create a small 2-second valid MP4 road video with moving cars."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 15.0, (640, 360))
    for f in range(30):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Asphalt road
        frame[180:, :] = (50, 50, 50)
        # Lane divider
        cv2.line(frame, (320, 180), (320, 360), (255, 255, 255), 2)
        # Moving car
        x = 240 + f * 2
        y = 220 + f * 1
        cv2.rectangle(frame, (x, y), (x + 80, y + 40), (0, 150, 255), -1)
        out.write(frame)
    out.release()
    print(f"Created sample test video at {path} ({path.stat().st_size} bytes)")

def test_workflow():
    test_dir = Path("outputs/test_temp")
    test_dir.mkdir(parents=True, exist_ok=True)
    video_file = test_dir / "live_test_road.mp4"
    create_sample_video(video_file)

    # 1. Health check
    print("\n--- Step 1: Health Check ---")
    req = urllib.request.urlopen(f"{BACKEND_URL}/health")
    assert req.getcode() == 200, f"Health check failed with {req.getcode()}"
    print("Health check OK (200)")

    # 2. Upload video file via multipart
    print("\n--- Step 2: Upload Video (POST /api/v1/video/upload-file) ---")
    boundary = "----WebKitFormBoundaryPerceptGuardLiveTest"
    with open(video_file, "rb") as f:
        video_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{video_file.name}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + video_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    upload_req = urllib.request.Request(
        f"{BACKEND_URL}/api/v1/video/upload-file",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(upload_req) as resp:
        assert resp.getcode() in (200, 201), f"Upload returned {resp.getcode()}"
        upload_data = json.loads(resp.read().decode())
    
    video_id = upload_data.get("video_id")
    print(f"Upload successful: video_id={video_id}")
    print(f"Response: {json.dumps(upload_data, indent=2)}")
    assert video_id, "video_id missing from upload response!"

    # 3. Stream preview check
    print("\n--- Step 3: Video Preview Stream Check (GET /api/v1/video/stream/{video_id}) ---")
    stream_req = urllib.request.Request(
        f"{BACKEND_URL}/api/v1/video/stream/{video_id}",
        headers={"Range": "bytes=0-1023"},
    )
    with urllib.request.urlopen(stream_req) as s_resp:
        print(f"Stream HTTP code: {s_resp.getcode()} (Expected 206 Partial Content)")
        assert s_resp.getcode() in (200, 206), f"Stream failed with {s_resp.getcode()}"
        chunk = s_resp.read(512)
        assert len(chunk) > 0, "Stream returned empty chunk!"
    print("Video preview streaming verified!")

    # 4. Start Processing
    print("\n--- Step 4: Start Processing (POST /api/v1/video/start-processing) ---")
    proc_payload = json.dumps({
        "video_id": video_id,
        "max_frames": 10,
        "rotation": 0,
        "analysis_mode": "QUICK_PREVIEW",
        "sample_fps": 15.0,
    }).encode("utf-8")

    start_req = urllib.request.Request(
        f"{BACKEND_URL}/api/v1/video/start-processing",
        data=proc_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(start_req) as p_resp:
        assert p_resp.getcode() == 202, f"Start processing failed with {p_resp.getcode()}"
        job_data = json.loads(p_resp.read().decode())

    job_id = job_data.get("job_id")
    print(f"Processing started: job_id={job_id}, status={job_data.get('status')}")
    assert job_id, "job_id missing from start processing response!"

    # 5. Poll Job Status
    print("\n--- Step 5: Poll Job Progress ---")
    for attempt in range(40):
        time.sleep(1.0)
        with urllib.request.urlopen(f"{BACKEND_URL}/api/v1/video/jobs/{job_id}") as stat_resp:
            stat = json.loads(stat_resp.read().decode())
            status = stat.get("status")
            progress = stat.get("progress_percent", 0)
            stage = stat.get("stage", "")
            print(f"[{attempt+1}s] Status: {status} | Progress: {progress}% | Stage: {stage}")
            if status == "COMPLETED":
                break
            if status == "FAILED":
                raise RuntimeError(f"Job failed: {stat.get('error') or stat.get('message')}")
    else:
        raise TimeoutError("Job did not complete within 40 seconds.")

    # 6. Retrieve Results and Verify Five Modules
    print("\n--- Step 6: Verify Five Modules in Results ---")
    with urllib.request.urlopen(f"{BACKEND_URL}/api/v1/video/jobs/{job_id}/results") as res_resp:
        results = json.loads(res_resp.read().decode())

    print(f"Total analyzed frames: {len(results.get('frames', []))}")
    print(f"Unique tracks: {results.get('unique_tracks')}")
    print(f"Total detections: {results.get('total_detections')}")
    print(f"Timeline events: {len(results.get('event_timeline', []))}")
    if "final_summary" in results:
        fs = results["final_summary"]
        print(f"Final Summary OK: status={fs.get('status')}, duration={fs.get('video_duration_seconds')}s, collision_risk={fs.get('collision_risk')}, reason={fs.get('reason')}")
        print(f"Road Users: {fs.get('road_user_counts')}")

    # Verify Module 1: Camera & Weather
    first_frame = results["frames"][0]
    assert "weather_intelligence" in first_frame, "Module 1 (Weather) missing from frame!"
    w = first_frame["weather_intelligence"]
    print(f"Module 1 (Weather) OK: condition={w.get('primary_condition')}, visibility={w.get('visibility_score')}%")

    # Verify Module 2: Object Detection & Tracking
    assert "detections" in first_frame, "Module 2 (Detections) missing from frame!"
    assert "tracks" in first_frame, "Module 2 (Tracks) missing from frame!"
    print(f"Module 2 (Object Detection & Tracking) OK: detections_count={len(first_frame['detections'])}, tracks_count={len(first_frame['tracks'])}")

    # Verify Module 3: Trajectory Prediction
    assert "trajectories" in first_frame, "Module 3 (Trajectories) missing from frame!"
    print(f"Module 3 (Trajectory Prediction) OK: trajectories_count={len(first_frame['trajectories'])}")

    # Verify Module 4: Collision Risk Assessment
    assert "collision_risk" in first_frame, "Module 4 (Collision Risk) missing from frame!"
    c = first_frame["collision_risk"]
    print(f"Module 4 (Collision Risk) OK: risk_state={c.get('risk_state')}, rationale={c.get('rationale')}")

    # Verify Module 5: Safety Recommendations & Alerts
    assert "safety_recommendation" in first_frame, "Module 5 (Safety Rec) missing from frame!"
    s = first_frame["safety_recommendation"]
    print(f"Module 5 (Safety Rec & Alerts) OK: recommendation={s.get('recommendation_state') or s.get('primary_action')}, reason={s.get('primary_reason')}")

    # 7. Verify Frame Image Serving
    print("\n--- Step 7: Verify Frame Image Serving ---")
    with urllib.request.urlopen(f"{BACKEND_URL}/api/v1/video/jobs/{job_id}/frames/0") as f_resp:
        assert f_resp.getcode() == 200
        f_bytes = f_resp.read()
        assert len(f_bytes) > 0
    print("Frame serving /frames/0 OK (HTTP 200)")

    print("\n==========================================")
    print("ALL 7 END-TO-END WORKFLOW CHECKS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    test_workflow()
