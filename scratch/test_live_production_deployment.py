"""Live Production Deployment & Five-Module Integration Test for PerceptGuard-AI.

Tests:
1. Production port and settings validation (Render PORT handling)
2. Production CORS validation with onrender.com origin
3. Model weights presence and integrity
4. Real video upload to /api/v1/video/upload-file
5. Asynchronous job initiation via /api/v1/video/start-processing
6. Live polling until job COMPLETED
7. Complete validation of all 5 primary analysis modules:
   - Module 1: Weather & Visibility (/weather)
   - Module 2: Trajectory Prediction (/trajectories)
   - Module 3: Collision Risk Assessment (/collision-risk)
   - Module 4: Evidence-Driven Safety Alerts (/safety-recommendations)
   - Module 5: Event Timeline (/timeline)
8. Video streaming endpoint validation (/stream)
"""

import sys
import os
import time
import json
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

# Anchor project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "http://127.0.0.1:8000"

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def test_settings_and_models():
    log("=== TEST 1: Production Settings & Dynamic PORT Resolution ===")
    from src.config.system_settings import SystemSettings
    test_env = {
        "PORT": "10000",
        "CORS_ORIGINS": "http://localhost:3000",
        "FRONTEND_URL": "https://custom-domain.com",
        "PERCEPTGUARD_STORAGE_DIR": str(PROJECT_ROOT / "outputs")
    }
    s = SystemSettings.load(env_override=test_env)
    assert s.api.port == 10000, f"Expected port 10000 from $PORT, got {s.api.port}"
    assert "https://custom-domain.com" in s.api.cors_origins, "FRONTEND_URL not in cors_origins"
    log(f"  Dynamic $PORT resolution: {s.api.port} [PASS]")
    log(f"  Configured CORS origins: {s.api.cors_origins} [PASS]")

    log("=== TEST 2: Model Weights & Configuration Artifacts ===")
    yolo_file = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
    lstm_file = PROJECT_ROOT / "models" / "forecasting" / "lstm" / "model.pt"
    assert yolo_file.exists() and yolo_file.stat().st_size > 1_000_000, f"Invalid YOLO weights at {yolo_file}"
    assert lstm_file.exists() and lstm_file.stat().st_size > 10_000, f"Invalid LSTM weights at {lstm_file}"
    log(f"  YOLO11n model weights: {yolo_file.name} ({yolo_file.stat().st_size / 1024 / 1024:.2f} MB) [PASS]")
    log(f"  LSTM model weights: {lstm_file.name} ({lstm_file.stat().st_size / 1024:.1f} KB) [PASS]")

def test_cors_headers():
    log("=== TEST 3: Production Render CORS Preflight ===")
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/video/jobs",
        headers={
            "Origin": "https://perceptguard-client.onrender.com",
            "Access-Control-Request-Method": "GET"
        },
        method="OPTIONS"
    )
    with urllib.request.urlopen(req) as resp:
        allow_origin = resp.headers.get("access-control-allow-origin")
        log(f"  Render origin preflight status: {resp.status}, allow-origin: {allow_origin} [PASS]")
        assert allow_origin == "https://perceptguard-client.onrender.com", f"Unexpected origin header: {allow_origin}"

def test_live_pipeline():
    log("=== TEST 4: Live Video Upload & Five-Module Analysis ===")

    # 1. Probe health
    with urllib.request.urlopen(f"{BASE_URL}/health/live") as resp:
        assert resp.status == 200
        health = json.loads(resp.read().decode())
        log(f"  Backend live probe: {health} [PASS]")

    # 2. Upload video
    video_path = PROJECT_ROOT / "outputs" / "day14_validation" / "clips" / "scenario_a_normal_daytime.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4"
    assert video_path.exists(), f"Sample video not found at {video_path}"

    boundary = f"----WebKitFormBoundary{os.urandom(16).hex()}"
    filename = video_path.name
    with open(video_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    upload_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/video/upload-file",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(upload_req) as resp:
        assert resp.status == 201, f"Expected 201 Created, got {resp.status}"
        upload_data = json.loads(resp.read().decode())
        video_id = upload_data["video_id"]
        log(f"  Video uploaded: video_id={video_id}, filename={upload_data['filename']} [PASS]")

    # 3. Start processing
    start_payload = json.dumps({
        "video_id": video_id,
        "max_frames": 10,
        "analysis_mode": "FULL_BALANCED"
    }).encode("utf-8")

    start_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/video/start-processing",
        data=start_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(start_req) as resp:
        assert resp.status in (200, 202), f"Expected 200 or 202, got {resp.status}"
        start_data = json.loads(resp.read().decode())
        job_id = start_data["job_id"]
        log(f"  Job launched: job_id={job_id}, status={start_data['status']} [PASS]")

    # 4. Poll until complete
    log("  Polling job status until completion...")
    start_time = time.time()
    max_wait = 90
    job_completed = False
    last_pct = -1

    while time.time() - start_time < max_wait:
        poll_req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs/{job_id}")
        with urllib.request.urlopen(poll_req) as resp:
            info = json.loads(resp.read().decode())
            status = info.get("status")
            pct = info.get("progress_percentage", 0.0)
            stage = info.get("stage", "")

            if pct != last_pct:
                log(f"    [+{(time.time()-start_time):.1f}s] status={status}, stage={stage}, progress={pct:.1f}%")
                last_pct = pct

            if status == "COMPLETED":
                job_completed = True
                log(f"  Job {job_id} COMPLETED successfully! Frames: {info.get('frames_processed')}, Detections: {info.get('total_detections')}, Tracks: {info.get('unique_tracks')} [PASS]")
                break
            elif status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Job failed: {info.get('error_message')}")
        time.sleep(1.0)

    assert job_completed, f"Job did not finish within {max_wait}s!"

    # 5. Validate the Five Primary Modules
    log("=== TEST 5: Validating All Five Analysis Module Endpoints ===")

    # Module 1: Camera & Weather
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/video/jobs/{job_id}/weather") as resp:
        assert resp.status == 200
        m1 = json.loads(resp.read().decode())
        log(f"  [1/5] Camera & Weather: condition={m1.get('weather_condition')}, visibility_score={m1.get('visibility_score')} [PASS]")

    # Module 2: Trajectory Prediction
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/video/jobs/{job_id}/trajectories") as resp:
        assert resp.status == 200
        m2 = json.loads(resp.read().decode())
        log(f"  [2/5] Trajectory Prediction: trajectories_count={len(m2.get('trajectories', []))} [PASS]")

    # Module 3: Collision Risk Assessment
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/video/jobs/{job_id}/collision-risk") as resp:
        assert resp.status == 200
        m3 = json.loads(resp.read().decode())
        log(f"  [3/5] Collision Risk: level={m3.get('risk_level')}, score={m3.get('composite_risk_score')}, ttc={m3.get('ttc_seconds')}s [PASS]")

    # Module 4: Evidence-Driven Safety Alerts
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/video/jobs/{job_id}/safety-recommendations") as resp:
        assert resp.status == 200
        m4 = json.loads(resp.read().decode())
        log(f"  [4/5] Safety Recommendations: alert_level={m4.get('alert_level')}, recommendations={len(m4.get('recommendations', []))} [PASS]")

    # Module 5: Event Timeline
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/video/jobs/{job_id}/timeline") as resp:
        assert resp.status == 200
        m5 = json.loads(resp.read().decode())
        log(f"  [5/5] Event Timeline: events_count={len(m5.get('events', []))} [PASS]")

    # Video Stream Endpoints
    stream_req = urllib.request.Request(f"{BASE_URL}/api/v1/video/stream/{video_id}", headers={"Range": "bytes=0-1024"})
    with urllib.request.urlopen(stream_req) as resp:
        assert resp.status in (200, 206), f"Unexpected video stream status: {resp.status}"
        log(f"  Raw Video Stream (/stream/{video_id}): HTTP {resp.status}, Content-Type: {resp.headers.get('content-type')} [PASS]")

    source_stream_req = urllib.request.Request(f"{BASE_URL}/api/v1/video/jobs/{job_id}/source-video", headers={"Range": "bytes=0-1024"})
    with urllib.request.urlopen(source_stream_req) as resp:
        assert resp.status in (200, 206), f"Unexpected source video stream status: {resp.status}"
        log(f"  Job Source Video Stream (/jobs/{job_id}/source-video): HTTP {resp.status}, Content-Type: {resp.headers.get('content-type')} [PASS]")

    log("=" * 70)
    log(" ALL PRODUCTION DEPLOYMENT & PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    log("=" * 70)

def main():
    try:
        test_settings_and_models()
        test_cors_headers()
        test_live_pipeline()
    except Exception as e:
        log(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
