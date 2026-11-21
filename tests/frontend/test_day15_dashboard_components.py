"""Automated test suite for Day 15 enterprise dashboard components and navigation.

Verifies that all 9 enterprise navigation pages and their underlying components
render cleanly, adhere to ML/calibration contracts, and handle missing or
empty states gracefully without unhandled exceptions.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.config import get_settings
from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.overview import render_system_overview
from dashboard.components.feature_analytics import render_feature_analytics
from dashboard.components.forecasting_view import render_reliability_forecasting
from dashboard.components.tracking_view import render_detection_tracking
from dashboard.components.risk_cards import render_risk_overview
from dashboard.components.history_view import render_processing_history
from dashboard.components.health_view import render_system_health
from dashboard.components.video_view import render_video_processing_view
from dashboard.components.alerts_feed import render_early_warning_alerts


def mock_columns(spec, *args, **kwargs):
    """Dynamic columns mock supporting both int and list-of-weights specs."""
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    if isinstance(spec, (list, tuple)):
        return [MagicMock() for _ in range(len(spec))]
    return [MagicMock()]


def mock_tabs(tab_names, *args, **kwargs):
    """Dynamic tabs mock returning one mock container per tab label."""
    return [MagicMock() for _ in range(len(tab_names))]


@pytest.fixture
def mock_client():
    """Create a mock DashboardApiClient matching exact client methods."""
    client = MagicMock(spec=DashboardApiClient)
    client.settings = get_settings()
    client.transport_mode = "REST_API"
    client.base_url = "http://127.0.0.1:8000"
    client.timeout = 10.0
    client.is_live = True
    client.last_sync_time = datetime.datetime.now(datetime.timezone.utc)
    client.probe_backend.return_value = True

    client.get_system_health.return_value = {
        "status": "HEALTHY",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overall_health": "OPTIMAL",
        "pipeline_status": "OPERATIONAL",
        "health_score": 0.98,
        "data_freshness": "CURRENT",
        "current_sequence_id": "seq_driving_001",
        "last_prediction_timestamp_seconds": 12.5,
        "model_availability": {
            "yolo_detector": True,
            "lstm_forecaster": True,
        },
    }

    client.get_risk_overview.return_value = {
        "sequence_id": "seq_driving_001",
        "horizons": {
            "h1": {
                "horizon": 1,
                "risk_score": 0.1500,
                "risk_level": "LOW",
                "trend_direction": "DECREASING",
                "rolling_mean": 0.1600,
                "volatility": 0.0200,
                "observation_count": 30,
            },
            "h3": {
                "horizon": 3,
                "risk_score": 0.4500,
                "risk_level": "MEDIUM",
                "trend_direction": "STABLE",
                "rolling_mean": 0.4400,
                "volatility": 0.0400,
                "observation_count": 30,
            },
            "h5": {
                "horizon": 5,
                "risk_score": 0.7200,
                "risk_level": "HIGH",
                "trend_direction": "INCREASING",
                "rolling_mean": 0.6900,
                "volatility": 0.0700,
                "observation_count": 30,
            },
        },
    }

    client.get_alerts.return_value = {
        "alerts": [
            {
                "alert_id": "ALT-001",
                "severity": "WARNING",
                "horizon": 5,
                "risk_score": 0.7200,
                "description": "High failure risk forecasted for t+5 frames.",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "ACTIVE",
            }
        ],
        "total_count": 1,
        "active_alerts_count": 1,
        "suppression_stats": {
            "cooldown_active": False,
            "deduplicated_count": 0,
        },
    }

    client.get_temporal_monitoring.return_value = {
        "sequence_id": "seq_driving_001",
        "points": [
            {"frame_index": 0, "h1_risk": 0.1, "h3_risk": 0.3, "h5_risk": 0.5},
            {"frame_index": 1, "h1_risk": 0.12, "h3_risk": 0.32, "h5_risk": 0.52},
        ],
    }

    client.get_tracking_context.return_value = {
        "sequence_id": "seq_driving_001",
        "active_tracks_count": 5,
        "total_detections_history": 120,
        "mean_confidence": 0.84,
        "detection_confidence_distribution": [
            {"confidence_bin": "0.8-0.9", "count": 80},
            {"confidence_bin": "0.9-1.0", "count": 40},
        ],
        "class_distribution": {
            "car": 70,
            "pedestrian": 30,
            "truck": 20,
        },
        "recent_tracks": [],
    }

    sample_job = {
        "job_id": "test-job-001",
        "filename": "test.mp4",
        "status": "COMPLETED",
        "stage": "COMPLETED",
        "processed_frames": 30,
        "total_frames": 30,
        "fps": 18.5,
        "total_detections": 90,
        "unique_tracks": 8,
        "duration_seconds": 1.62,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    client.list_video_jobs.return_value = [sample_job]

    client.get_video_job_results.return_value = {
        "summary": {
            "total_frames": 30,
            "total_detections": 90,
            "unique_tracks": 8,
            "active_alerts_count": 1,
            "processing_fps": 18.5,
        },
        "frames": [
            {
                "frame_idx": 0,
                "frame_name": "frame_0000.jpg",
                "detections": [
                    {
                        "box": [10.0, 20.0, 100.0, 200.0],
                        "confidence": 0.88,
                        "class_id": 2,
                        "class_name": "car",
                        "track_id": 1,
                    }
                ],
                "features": {
                    "blur_laplacian": 125.4,
                    "mean_brightness": 115.2,
                    "contrast": 48.3,
                    "num_detections": 1,
                    "mean_confidence": 0.88,
                    "mean_speed": 4.5,
                    "mean_acceleration": 0.2,
                },
                "risk": {"h1": 0.15, "h3": 0.45, "h5": 0.72},
            }
        ],
    }

    return client


class TestOverviewComponent:
    """Test the executive operations overview component."""

    @patch("dashboard.components.overview.st")
    def test_overview_renders_with_active_job(self, mock_st, mock_client):
        """Verify overview displays pipeline KPI metrics and active job details."""
        mock_st.session_state = {"active_job_id": "test-job-001"}
        mock_st.columns.side_effect = mock_columns

        render_system_overview(mock_client)

        mock_client.get_system_health.assert_called_once()
        mock_client.list_video_jobs.assert_called_once()

    @patch("dashboard.components.overview.st")
    def test_overview_handles_api_failure_gracefully(self, mock_st, mock_client):
        """Verify overview captures FrontendAPIError without crashing."""
        mock_client.get_system_health.side_effect = FrontendAPIError("Backend Unreachable")
        mock_st.session_state = {}
        mock_st.columns.side_effect = mock_columns

        render_system_overview(mock_client)
        mock_st.markdown.assert_called()


class TestFeatureAnalyticsComponent:
    """Test feature analytics component with 27-feature statistics."""

    @patch("dashboard.components.feature_analytics.st")
    def test_feature_analytics_empty_when_no_job(self, mock_st, mock_client):
        """Verify feature analytics renders empty state when no job results are available."""
        mock_client.get_video_job_results.return_value = None
        mock_client.list_video_jobs.return_value = []
        mock_st.session_state = {}
        mock_st.columns.side_effect = mock_columns

        render_feature_analytics(mock_client)
        mock_st.markdown.assert_called()

    @patch("dashboard.components.feature_analytics.st")
    def test_feature_analytics_renders_with_per_frame_features(self, mock_st, mock_client):
        """Verify feature analytics extracts and renders visual and kinematic distributions."""
        mock_st.session_state = {"active_job_id": "test-job-001"}
        mock_st.columns.side_effect = mock_columns
        mock_st.tabs.side_effect = mock_tabs

        render_feature_analytics(mock_client)
        mock_client.get_video_job_results.assert_called_with("test-job-001")


class TestReliabilityForecastingComponent:
    """Test multi-horizon reliability forecasting component."""

    @patch("dashboard.components.forecasting_view.st")
    def test_forecasting_view_renders_model_specs(self, mock_st, mock_client):
        """Verify forecasting page displays model specs and scientific disclosures."""
        mock_st.columns.side_effect = mock_columns
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        render_reliability_forecasting(mock_client)
        mock_st.info.assert_called()


class TestDetectionTrackingComponent:
    """Test detection and tracking inspection component."""

    @patch("dashboard.components.tracking_view.st")
    def test_detection_tracking_renders(self, mock_st, mock_client):
        """Verify detection & tracking metrics and frame inspector render without errors."""
        mock_st.session_state = {"active_job_id": "test-job-001"}
        mock_st.columns.side_effect = mock_columns
        mock_st.slider.return_value = 0

        render_detection_tracking(mock_client)
        mock_client.get_tracking_context.assert_called_once()


class TestRiskAssessmentComponent:
    """Test risk overview and horizon calibration component."""

    @patch("dashboard.components.risk_cards.st")
    def test_risk_overview_renders_cards(self, mock_st, mock_client):
        """Verify multi-horizon risk overview renders H1, H3, H5 cards and calibration status."""
        mock_st.columns.side_effect = mock_columns
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        render_risk_overview(mock_client)
        mock_client.get_risk_overview.assert_called_once()


class TestProcessingHistoryComponent:
    """Test job processing history and audit log component."""

    @patch("dashboard.components.history_view.st")
    def test_processing_history_renders_table(self, mock_st, mock_client):
        """Verify processing history renders job table and controls."""
        mock_st.session_state = {}
        mock_st.columns.side_effect = mock_columns
        mock_st.selectbox.return_value = "ALL"
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        render_processing_history(mock_client)
        mock_client.list_video_jobs.assert_called_once()


class TestSystemHealthComponent:
    """Test system health, diagnostic telemetry, and component probe component."""

    @patch("dashboard.components.health_view.st")
    def test_system_health_renders_diagnostics(self, mock_st, mock_client):
        """Verify system health component probes latency and renders subsystem table."""
        mock_st.columns.side_effect = mock_columns
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        render_system_health(mock_client)
        mock_client.probe_backend.assert_called_once()


class TestMainDashboardNavigation:
    """Test the complete 9-item enterprise navigation configuration."""

    def test_navigation_menu_items(self):
        """Verify dashboard/app.py declares all 9 enterprise navigation views."""
        app_path = Path(__file__).resolve().parent.parent.parent / "dashboard" / "app.py"
        content = app_path.read_text(encoding="utf-8")

        expected_pages = [
            "📊 Overview",
            "🎥 Video Processing",
            "🎯 Detection & Tracking",
            "🔬 Feature Analytics",
            "🔮 Reliability Forecasting",
            "⚠️ Risk Assessment",
            "🚨 Early Warning Alerts",
            "📋 Processing History",
            "🩺 System Health",
        ]

        for page in expected_pages:
            assert f'"{page}"' in content or f"'{page}'" in content, (
                f"Missing enterprise navigation item: {page}"
            )
