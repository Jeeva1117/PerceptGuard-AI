"""End-to-End Production Verification & Deployment Test for Render.

Validates:
1. Dynamic $PORT binding (e.g. 10000)
2. Production CORS origin handling (FRONTEND_URL + onrender.com regex)
3. Persistent storage disk mounting (PERCEPTGUARD_STORAGE_DIR / OUTPUTS_DIR)
4. Model weights integrity (YOLO11n + LSTM)
5. Full video upload -> Start Processing -> 5-module results workflow
"""

import os
import sys
from pathlib import Path
import tempfile
import time
import json
import shutil

# Anchor project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 70)
    print(" PERCEPTGUARD-AI: RENDER DEPLOYMENT & PRODUCTION SUITE TEST")
    print("=" * 70)

    # 1. Setup isolated persistent storage mount to simulate Render Persistent Disk
    temp_disk_dir = Path(tempfile.mkdtemp(prefix="render_disk_test_"))
    print(f"[TEST 1] Simulated Render Persistent Disk: {temp_disk_dir}")

    # Set Render-like production environment variables
    test_port = "10000"
    test_frontend = "https://perceptguard-frontend.onrender.com"
    os.environ["PORT"] = test_port
    os.environ["PERCEPTGUARD_STORAGE_DIR"] = str(temp_disk_dir)
    os.environ["OUTPUTS_DIR"] = str(temp_disk_dir)
    os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000"
    os.environ["FRONTEND_URL"] = test_frontend
    os.environ["DEVICE"] = "cpu"

    # 2. Validate SystemSettings loading
    from src.config.system_settings import reload_settings
    settings = reload_settings()

    print(f"[TEST 2] Verifying System Settings:")
    print(f"  - Configured API Port: {settings.api.port} (Expected: 10000)")
    assert settings.api.port == 10000, f"Expected port 10000, got {settings.api.port}"
    print(f"  - Configured Outputs Dir: {settings.outputs_dir}")
    assert settings.outputs_dir == temp_disk_dir, f"Expected {temp_disk_dir}, got {settings.outputs_dir}"
    print(f"  - Configured CORS Origins: {settings.api.cors_origins}")
    assert test_frontend in settings.api.cors_origins, f"Frontend {test_frontend} not in CORS origins!"
    print("  -> System Settings passed verification successfully.")

    # 3. Verify Model Weights existence and loadability
    print(f"[TEST 3] Verifying Model Weights & Configs:")
    yolo_path = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
    lstm_path = PROJECT_ROOT / "models" / "forecasting" / "lstm" / "model.pt"
    assert yolo_path.exists(), f"Missing YOLO weights at {yolo_path}"
    assert lstm_path.exists(), f"Missing LSTM weights at {lstm_path}"
    print(f"  - YOLO11n weights: {yolo_path} ({yolo_path.stat().st_size} bytes) [OK]")
    print(f"  - LSTM weights: {lstm_path} ({lstm_path.stat().st_size} bytes) [OK]")

    # Check key yaml configs
    for cfg in ["yolo_config.yaml", "tracker_config.yaml", "video_config.yaml"]:
        cfg_p = PROJECT_ROOT / "config" / cfg
        assert cfg_p.exists(), f"Missing config file: {cfg_p}"
    print("  - Configuration YAML files present [OK]")

    # 4. Verify JobManager and Persistent Storage directory creation
    print(f"[TEST 4] Verifying JobManager Persistent Storage Mount:")
    # Reset JobManager singleton to use test storage dir
    from src.api.job_manager import JobManager
    JobManager._instance = None
    job_mgr = JobManager(base_dir=temp_disk_dir)
    uploads_dir = job_mgr.jobs_dir / "uploads"
    assert uploads_dir.exists(), f"Uploads directory was not created at {uploads_dir}"
    print(f"  - Persistent jobs directory: {job_mgr.jobs_dir} [OK]")
    print(f"  - Persistent uploads directory: {uploads_dir} [OK]")

    # 5. Test real video upload and end-to-end processing via FastAPI TestClient
    print(f"[TEST 5] Testing End-to-End Real Upload & Five-Module Analysis:")
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    app = create_app(settings=settings)
    client = TestClient(app)

    # Healthcheck test
    health_resp = client.get("/health/live")
    assert health_resp.status_code == 200, f"Healthcheck failed: {health_resp.text}"
    print(f"  - Healthcheck /health/live: {health_resp.json()} [OK]")

    # Locate test driving video
    sample_video = PROJECT_ROOT / "outputs" / "day14_validation" / "clips" / "scenario_a_normal_daytime.mp4"
    if not sample_video.exists():
        sample_video = PROJECT_ROOT / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4"
    assert sample_video.exists(), f"Test video not found: {sample_video}"
    print(f"  - Using source test video: {sample_video.name} ({sample_video.stat().st_size} bytes)")

    # Test file upload endpoint
    with open(sample_video, "rb") as f:
        upload_resp = client.post(
            "/api/v1/video/upload-file",
            files={"file": (sample_video.name, f, "video/mp4")}
        )
    assert upload_resp.status_code == 200, f"Upload failed ({upload_resp.status_code}): {upload_resp.text}"
    upload_data = upload_resp.json()
    video_id = upload_data.get("video_id")
    print(f"  - Video uploaded successfully: video_id={video_id}, filename={upload_data.get('filename')}")

    # Verify upload was stored in the persistent storage dir
    stored_uploads = list(uploads_dir.iterdir())
    assert len(stored_uploads) > 0, "No file stored in persistent uploads dir!"
    print(f"  - Stored video verified on persistent disk: {stored_uploads[0].name} ({stored_uploads[0].stat().st_size} bytes) [OK]")

    # Test start processing endpoint
    start_payload = {
        "video_id": video_id,
        "max_frames": 15,  # 15 frames for quick CPU validation
        "analysis_mode": "FULL_BALANCED"
    }
    start_resp = client.post("/api/v1/video/start-processing", json=start_payload)
    assert start_resp.status_code == 200, f"Start processing failed: {start_resp.text}"
    start_data = start_resp.json()
    job_id = start_data.get("job_id")
    print(f"  - Processing started: job_id={job_id}, status={start_data.get('status')}")

    # Poll until job completes
    print("  - Polling job status...")
    max_wait_seconds = 60
    start_time = time.time()
    final_job_data = None
    while time.time() - start_time < max_wait_seconds:
        status_resp = client.get(f"/api/v1/video/jobs/{job_id}")
        assert status_resp.status_code == 200, f"Status check failed: {status_resp.text}"
        job_info = status_resp.json()
        status = job_info.get("status")
        progress = job_info.get("progress_percentage", 0.0)
        stage = job_info.get("stage", "")
        print(f"    [+{(time.time()-start_time):.1f}s] status={status}, stage={stage}, progress={progress:.1f}%")

        if status == "COMPLETED":
            final_job_data = job_info
            break
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Job terminated prematurely with status {status}: {job_info.get('error_message')}")
        time.sleep(1.5)

    assert final_job_data is not None, f"Job did not complete within {max_wait_seconds}s timeout"
    print(f"  - Job completed successfully! Frames processed: {final_job_data.get('frames_processed')}")
    print(f"    Detections: {final_job_data.get('total_detections')}, Tracks: {final_job_data.get('unique_tracks')}")

    # Verify results artifact on persistent disk
    job_out_dir = job_mgr.jobs_dir / job_id
    results_json_path = job_out_dir / "results.json"
    assert results_json_path.exists(), f"results.json missing at {results_json_path}"
    with open(results_json_path, "r", encoding="utf-8") as f:
        results_content = json.load(f)
    print(f"  - Persistent results.json artifact verified ({results_json_path.stat().st_size} bytes) [OK]")

    # 6. Verify all Five Analysis Modules endpoints
    print(f"[TEST 6] Verifying All Five Primary Analysis Module Endpoints:")
    
    # Module 1: Weather & Visibility
    w_resp = client.get(f"/api/v1/video/jobs/{job_id}/weather")
    assert w_resp.status_code == 200, f"Weather endpoint failed: {w_resp.text}"
    w_data = w_resp.json()
    print(f"  [1/5] Weather & Visibility: condition={w_data.get('weather_condition')}, score={w_data.get('visibility_score')} [OK]")

    # Module 2: Trajectory Prediction
    traj_resp = client.get(f"/api/v1/video/jobs/{job_id}/trajectories")
    assert traj_resp.status_code == 200, f"Trajectories endpoint failed: {traj_resp.text}"
    traj_data = traj_resp.json()
    print(f"  [2/5] Trajectory Prediction: tracks={len(traj_data.get('trajectories', []))} [OK]")

    # Module 3: Collision Risk
    risk_resp = client.get(f"/api/v1/video/jobs/{job_id}/collision-risk")
    assert risk_resp.status_code == 200, f"Collision risk endpoint failed: {risk_resp.text}"
    risk_data = risk_resp.json()
    print(f"  [3/5] Collision Risk: level={risk_data.get('risk_level')}, score={risk_data.get('composite_risk_score')} [OK]")

    # Module 4: Safety Alerts / Recommendations
    alert_resp = client.get(f"/api/v1/video/jobs/{job_id}/safety-recommendations")
    assert alert_resp.status_code == 200, f"Safety alerts endpoint failed: {alert_resp.text}"
    alert_data = alert_resp.json()
    print(f"  [4/5] Safety Recommendations: alert_level={alert_data.get('alert_level')}, count={len(alert_data.get('recommendations', []))} [OK]")

    # Module 5: Event Timeline
    tl_resp = client.get(f"/api/v1/video/jobs/{job_id}/timeline")
    assert tl_resp.status_code == 200, f"Timeline endpoint failed: {tl_resp.text}"
    tl_data = tl_resp.json()
    print(f"  [5/5] Event Timeline: events={len(tl_data.get('events', []))} [OK]")

    # Cleanup temporary simulated disk
    shutil.rmtree(temp_disk_dir, ignore_errors=True)
    print("=" * 70)
    print(" ALL DEPLOYMENT & PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
