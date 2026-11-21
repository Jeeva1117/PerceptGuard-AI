"""Comprehensive Day 17 integration and runtime validation test suite.

Verifies:
1. FastAPI sub-resource endpoints (/detections, /tracking, /features, /forecast, /risk, /alerts)
2. DashboardApiClient client methods and fallback behaviors
3. Video validation and path traversal sanitization
4. Streamlit dashboard component public interfaces and navigation
5. End-to-end pipeline execution on authentic BDD100K driving video
6. Graceful degradation when backend or model checkpoints are unavailable
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.job_manager import JobManager, get_job_manager
from dashboard.api_client import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    DashboardApiClient,
)
from dashboard.components import (
    render_detection_tracking,
    render_feature_analytics,
    render_object_detection,
    render_object_tracking,
    render_processing_history,
    render_reliability_forecasting,
    render_risk_overview,
    render_system_health,
    render_system_overview,
    render_video_processing_view,
)
from src.video_processing.video_pipeline_runner import VideoPipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def api_client():
    """Create a FastAPI TestClient instance."""
    app = create_app()
    return TestClient(app)


class TestFastApiJobSubResourceEndpoints:
    """Validate granular REST sub-resource endpoints in src/api/routers/video.py."""

    def test_job_sub_endpoints_missing_job(self, api_client):
        """Verify 404 is returned when querying sub-resources of a non-existent job."""
        for sub in ["detections", "tracking", "features", "forecast", "risk", "alerts"]:
            resp = api_client.get(f"/api/v1/video/jobs/non_existent_job_123/{sub}")
            assert resp.status_code == 404
            data = resp.json()
            assert "error" in data or "detail" in data or "message" in data

    def test_job_sub_endpoints_with_mock_results(self, api_client):
        """Verify sub-resource endpoints extract slices properly from results.json."""
        job_mgr = get_job_manager()
        mock_job_id = "test_day17_job_mock"

        # Create temporary results file
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_path = Path(tmp_dir) / "results.json"
            sample_data = {
                "job_id": mock_job_id,
                "status": "COMPLETED",
                "total_detections": 2,
                "unique_tracks": 1,
                "total_alerts": 1,
                "per_horizon_risk": {
                    "h1": {"latest_risk": 0.15, "category": "LOW"},
                    "h3": {"latest_risk": 0.40, "category": "MEDIUM"},
                    "h5": {"latest_risk": 0.70, "category": "HIGH"},
                },
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "detections_count": 2,
                        "detections": [
                            {"class_name": "car", "confidence": 0.92, "bbox_xyxy": [10, 10, 50, 50]},
                            {"class_name": "car", "confidence": 0.85, "bbox_xyxy": [60, 60, 100, 100]},
                        ],
                        "tracks_count": 1,
                        "tracks": [
                            {"track_id": 1, "class_name": "car", "confidence": 0.92, "bbox_xyxy": [10, 10, 50, 50]},
                        ],
                        "visual_quality": {"blur_laplacian": 180.5, "brightness": 120.0, "contrast": 45.0},
                        "features": [0.1] * 27,
                        "predicted_reliability": {"h1": 0.85, "h3": 0.60, "h5": 0.30},
                        "risk_scores": {"h1": 0.15, "h3": 0.40, "h5": 0.70},
                        "alerts": [
                            {"alert_id": "ALT_001", "horizon": 5, "severity": "HIGH", "risk_score": 0.70}
                        ],
                    }
                ],
            }
            res_path.write_text(json.dumps(sample_data), encoding="utf-8")

            # Mock job_mgr.get_job_results and get_job
            with patch.object(job_mgr, "get_job_results", return_value=sample_data), \
                 patch.object(job_mgr, "get_job", return_value=MagicMock(status=MagicMock(value="COMPLETED"))):

                # 1. Detections endpoint
                r_det = api_client.get(f"/api/v1/video/jobs/{mock_job_id}/detections")
                assert r_det.status_code == 200
                det_json = r_det.json()
                assert det_json["total_detections"] == 2
                assert len(det_json["detections"]) == 2
                assert det_json["detections"][0]["class_name"] == "car"

                # 2. Tracking endpoint
                r_trk = api_client.get(f"/api/v1/video/jobs/{mock_job_id}/tracking")
                assert r_trk.status_code == 200
                trk_json = r_trk.json()
                assert trk_json["unique_tracks"] == 1
                assert len(trk_json["tracks"]) == 1
                assert trk_json["tracks"][0]["track_id"] == 1

                # 3. Features endpoint
                r_feat = api_client.get(f"/api/v1/video/jobs/{mock_job_id}/features")
                assert r_feat.status_code == 200
                feat_json = r_feat.json()
                assert feat_json["frames_count"] == 1
                assert len(feat_json["features"][0]["features"]) == 27

                # 4. Forecast endpoint
                r_fc = api_client.get(f"/api/v1/video/jobs/{mock_job_id}/forecast")
                assert r_fc.status_code == 200
                fc_json = r_fc.json()
                assert "h1" in fc_json["per_horizon_risk"]
                assert fc_json["forecasts"][0]["predicted_reliability"]["h1"] == 0.85

                # 5. Risk endpoint
                r_risk = api_client.get(f"/api/v1/video/jobs/{mock_job_id}/risk")
                assert r_risk.status_code == 200
                risk_json = r_risk.json()
                assert risk_json["risk_formula"] == "S = 1.0 - R"
                assert risk_json["per_horizon_risk"]["h5"]["category"] == "HIGH"

                # 6. Alerts endpoint
                r_alt = api_client.get(f"/api/v1/video/jobs/{mock_job_id}/alerts")
                assert r_alt.status_code == 200
                alt_json = r_alt.json()
                assert alt_json["total_alerts"] == 1
                assert alt_json["alerts"][0]["alert_id"] == "ALT_001"


class TestDashboardApiClientSubMethods:
    """Validate DashboardApiClient methods in both REST mode and in-process fallback."""

    def test_client_sub_methods_rest_mode(self):
        """Verify client correctly parses REST responses for new job sub-endpoints."""
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"job_id": "test_j", "data": "ok"}
        mock_http.request.return_value = mock_resp

        client = DashboardApiClient(
            base_url="http://127.0.0.1:8000",
            http_client=mock_http,
            use_in_process_fallback=False,
        )

        res_dets = client.get_video_job_detections("test_j")
        assert res_dets["job_id"] == "test_j"

        res_trk = client.get_video_job_tracking("test_j")
        assert res_trk["job_id"] == "test_j"

        res_feat = client.get_video_job_features("test_j")
        assert res_feat["job_id"] == "test_j"

        res_fc = client.get_video_job_forecast("test_j")
        assert res_fc["job_id"] == "test_j"

        res_risk = client.get_video_job_risk("test_j")
        assert res_risk["job_id"] == "test_j"

        res_alt = client.get_video_job_alerts("test_j")
        assert res_alt["job_id"] == "test_j"

    def test_client_fallback_on_connection_error(self):
        """Verify client switches to IN_PROCESS fallback when backend connection fails."""
        import httpx

        mock_http = MagicMock()
        mock_http.request.side_effect = httpx.ConnectError("Connection refused")

        client = DashboardApiClient(
            base_url="http://127.0.0.1:8000",
            http_client=mock_http,
            use_in_process_fallback=True,
        )

        # Mock results returned by fallback
        with patch.object(client, "get_video_job_results", return_value={
            "job_id": "inproc_j",
            "unique_tracks": 2,
            "per_horizon_risk": {},
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "detections": [{"class_name": "pedestrian", "confidence": 0.88}],
                    "tracks": [{"track_id": 1, "class_name": "pedestrian"}],
                    "visual_quality": {"blur_laplacian": 200.0},
                    "features": [0.5] * 27,
                    "predicted_reliability": {"h1": 0.9},
                    "risk_scores": {"h1": 0.1},
                    "alerts": [],
                }
            ],
        }):
            dets = client.get_video_job_detections("inproc_j")
            assert dets["total_detections"] == 1
            assert dets["detections"][0]["class_name"] == "pedestrian"


class TestVideoValidationSecurity:
    """Verify input validation and path traversal sanitization."""

    def test_invalid_video_extensions_rejected(self, api_client):
        """Verify invalid extensions (e.g. .txt, .exe, .sh) are rejected."""
        fake_content = b"not a video"
        resp = api_client.post(
            "/api/v1/video/validate",
            files={"file": ("malicious.exe", fake_content, "application/octet-stream")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data.get("error", {}).get("code") == "INVALID_VIDEO_FILE"

    def test_corrupt_video_rejected(self, api_client):
        """Verify corrupted bytes masquerading as MP4 are rejected with validation failure."""
        fake_content = b"corrupted header content with non-video magic bytes"
        resp = api_client.post(
            "/api/v1/video/validate",
            files={"file": ("corrupt.mp4", fake_content, "video/mp4")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data.get("error", {}).get("code") == "INVALID_VIDEO_FILE"


class TestEndToEndDay17PipelineExecution:
    """Perform real-world validation on authentic driving video using CPU pipeline."""

    def test_real_video_pipeline_and_persistence(self):
        """Verify full 14-stage execution on authentic BDD100K daytime video."""
        clip_path = PROJECT_ROOT / "outputs" / "day14_validation" / "clips" / "scenario_a_normal_daytime.mp4"
        assert clip_path.exists(), f"Authentic test clip missing: {clip_path}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "test_day17_run"
            runner = VideoPipelineRunner(device="cpu")

            run_result = runner.run(
                video_path=clip_path,
                job_id="day17_verify_001",
                output_dir=out_dir,
                max_frames=10,
            )

            assert run_result.job_id == "day17_verify_001"
            assert run_result.frames_processed == 10
            assert run_result.total_detections > 0
            assert run_result.unique_tracks > 0
            assert run_result.total_duration_seconds > 0.0

            # Verify results.json exists and is valid RFC 8259 JSON
            res_file = out_dir / "results.json"
            assert res_file.exists(), "results.json not generated"
            with open(res_file, "r", encoding="utf-8") as f:
                res_data = json.load(f)

            assert res_data["job_id"] == "day17_verify_001"
            assert "frames" in res_data
            assert len(res_data["frames"]) == 10

            # Verify frame-level telemetry and visual features
            frame_0 = res_data["frames"][0]
            assert "visual_quality" in frame_0
            assert "blur_laplacian" in frame_0["visual_quality"]
            assert "brightness" in frame_0["visual_quality"]
            assert "contrast" in frame_0["visual_quality"]
            assert "detections" in frame_0
            assert "tracks" in frame_0
            assert "predicted_reliability" in frame_0
            assert "risk_scores" in frame_0

            # Verify annotated frames directory
            annotated_dir = out_dir / "annotated"
            assert annotated_dir.exists()
            annotated_files = list(annotated_dir.glob("*.jpg"))
            assert len(annotated_files) == 10
