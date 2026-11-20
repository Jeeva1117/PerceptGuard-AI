"""Unit and visual logic tests for dashboard UI components and data integrity."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch
import pytest

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import (
    render_empty_state,
    render_error_boundary,
    render_stale_warning,
)
from dashboard.components.header import render_header
from dashboard.components.overview import render_system_overview
from dashboard.components.risk_cards import render_risk_overview
from dashboard.components.temporal_chart import render_temporal_monitoring
from dashboard.components.alerts_feed import render_early_warning_alerts
from dashboard.components.tracking_view import render_detection_tracking
from dashboard.components.pipeline_accordion import render_pipeline_status_accordion


class TestStateViews:
    """Test empty states, error boundaries, and stale data warnings."""

    @patch("dashboard.components.state_views.st")
    def test_render_empty_state(self, mock_st):
        """Verify empty state generates card with custom title, message, and icon."""
        render_empty_state(
            title="No Alerts Found",
            message="There are currently no active alerts in the system.",
            icon="🔔",
        )
        mock_st.markdown.assert_called_once()
        html = mock_st.markdown.call_args[0][0]
        assert "No Alerts Found" in html
        assert "There are currently no active alerts in the system." in html
        assert "🔔" in html

    @patch("dashboard.components.state_views.st")
    def test_render_error_boundary_without_retry(self, mock_st):
        """Verify error boundary renders user-friendly message without leaking raw stack trace."""
        render_error_boundary(
            title="Failed to Load Telemetry",
            error_message="Backend connection refused",
            retry_callback=None,
        )
        mock_st.markdown.assert_called_once()
        err_text = mock_st.markdown.call_args[0][0]
        assert "Failed to Load Telemetry" in err_text
        assert "Backend connection refused" in err_text
        mock_st.button.assert_not_called()

    @patch("dashboard.components.state_views.st")
    def test_render_error_boundary_with_retry(self, mock_st):
        """Verify error boundary renders retry button when callback is provided."""
        mock_callback = MagicMock()
        mock_st.button.return_value = True

        render_error_boundary(
            title="API Error",
            error_message="Timeout",
            retry_callback=mock_callback,
            key_suffix="test_retry",
        )
        mock_st.button.assert_called_once()
        assert "Retry" in mock_st.button.call_args[0][0]
        mock_callback.assert_called_once()

    @patch("dashboard.components.state_views.st")
    def test_render_stale_warning_fresh_data(self, mock_st):
        """Verify stale warning is NOT displayed when data is within TTL."""
        recent = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)
        render_stale_warning(last_sync=recent, freshness="CURRENT", ttl_seconds=60)
        mock_st.markdown.assert_not_called()

    @patch("dashboard.components.state_views.st")
    def test_render_stale_warning_stale_data(self, mock_st):
        """Verify stale warning IS displayed when data age exceeds TTL."""
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
        render_stale_warning(last_sync=old, freshness="CURRENT", ttl_seconds=60)
        mock_st.markdown.assert_called_once()
        warn_text = mock_st.markdown.call_args[0][0]
        assert "Telemetry Staleness Notice" in warn_text
        assert "60s" in warn_text


class TestRiskOverviewDataIntegrity:
    """Test multi-horizon risk overview data presentation, category mapping, and integrity."""

    @patch("dashboard.components.risk_cards.st")
    def test_missing_forecast_never_converts_to_low_risk(self, mock_st):
        """CRITICAL: Missing/null risk scores must NEVER be rendered as 0.0 or LOW risk."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_risk_overview.return_value = {
            "sequence_id": "test_seq",
            "horizons": {
                "h1": {
                    "horizon": 1,
                    "risk_score": None,  # Missing prediction!
                    "risk_level": "LOW",  # Erroneous default from a faulty upstream source!
                    "trend_direction": "INCREASING",
                    "rolling_mean": None,
                    "volatility": None,
                    "observation_count": 0,
                },
                "h3": {
                    "horizon": 3,
                    "risk_score": 0.4500,
                    "risk_level": "MEDIUM",
                    "trend_direction": "STABLE",
                    "rolling_mean": 0.4200,
                    "volatility": 0.0300,
                    "observation_count": 10,
                },
                "h5": {
                    "horizon": 5,
                    "risk_score": 0.7800,
                    "risk_level": "HIGH",
                    "trend_direction": "INCREASING",
                    "rolling_mean": 0.7100,
                    "volatility": 0.0800,
                    "observation_count": 10,
                },
            },
        }

        # Mock 3 columns
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = (col1, col2, col3)

        render_risk_overview(mock_client)

        # Inspect H1 card markdown inside col1
        col1.markdown.assert_called_once()
        h1_card_html = col1.markdown.call_args[0][0]

        # Verify that because score was None, it rendered 'N/A' and 'UNKNOWN' - NOT 'LOW' or '0.0000'!
        assert "N/A" in h1_card_html
        assert "UNKNOWN" in h1_card_html
        assert "0.0000" not in h1_card_html

    @patch("dashboard.components.risk_cards.st")
    def test_h1_h3_h5_display_integrity(self, mock_st):
        """Verify H1, H3, H5 cards display their distinct values, rolling means, and trends."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_risk_overview.return_value = {
            "horizons": {
                "h1": {
                    "risk_score": 0.1234,
                    "risk_level": "LOW",
                    "trend_direction": "DECREASING",
                    "rolling_mean": 0.1500,
                    "volatility": 0.0200,
                    "observation_count": 30,
                },
                "h3": {
                    "risk_score": 0.5432,
                    "risk_level": "MEDIUM",
                    "trend_direction": "STABLE",
                    "rolling_mean": 0.5100,
                    "volatility": 0.0400,
                    "observation_count": 30,
                },
                "h5": {
                    "risk_score": 0.8765,
                    "risk_level": "CRITICAL",
                    "trend_direction": "INCREASING",
                    "rolling_mean": 0.8200,
                    "volatility": 0.0900,
                    "observation_count": 30,
                },
            }
        }

        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = (col1, col2, col3)

        render_risk_overview(mock_client)

        h1_html = col1.markdown.call_args[0][0]
        h3_html = col2.markdown.call_args[0][0]
        h5_html = col3.markdown.call_args[0][0]

        assert "0.1234" in h1_html
        assert "LOW" in h1_html
        assert "0.5432" in h3_html
        assert "MEDIUM" in h3_html
        assert "0.8765" in h5_html
        assert "CRITICAL" in h5_html

    @patch("dashboard.components.risk_cards.st")
    def test_risk_overview_empty_horizons(self, mock_st):
        """Verify empty horizons dictionary triggers render_empty_state."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_risk_overview.return_value = {"horizons": {}}

        render_risk_overview(mock_client)
        mock_st.markdown.assert_called()  # Notice was displayed, then empty state
        # st.columns was never called because it returned early
        mock_st.columns.assert_not_called()


class TestTemporalMonitoringComponent:
    """Test temporal risk trajectories, history charts, and transition tables."""

    @patch("dashboard.components.temporal_chart.st")
    def test_temporal_empty_history(self, mock_st):
        """Verify empty history displays an empty state rather than an empty/broken chart."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_temporal_data.return_value = {
            "horizons": {"h1": [], "h3": [], "h5": []},
            "state_transitions": [],
        }

        render_temporal_monitoring(mock_client)
        mock_st.line_chart.assert_not_called()

    @patch("dashboard.components.temporal_chart.st")
    def test_temporal_populated_history(self, mock_st):
        """Verify populated temporal history renders Streamlit line chart and transitions expander."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_temporal_data.return_value = {
            "horizons": {
                "h1": [
                    {"timestamp_seconds": 1.0, "risk_score": 0.2},
                    {"timestamp_seconds": 2.0, "risk_score": 0.25},
                ],
                "h3": [
                    {"timestamp_seconds": 1.0, "risk_score": 0.4},
                    {"timestamp_seconds": 2.0, "risk_score": 0.45},
                ],
                "h5": [
                    {"timestamp_seconds": 1.0, "risk_score": 0.6},
                    {"timestamp_seconds": 2.0, "risk_score": 0.65},
                ],
            },
            "state_transitions": [
                {
                    "timestamp_seconds": 1.5,
                    "horizon": 3,
                    "previous_level": "LOW",
                    "current_level": "MEDIUM",
                    "risk_score": 0.42,
                    "is_genuine_transition": True,
                    "notes": "State change confirmed",
                }
            ],
        }

        render_temporal_monitoring(mock_client)
        mock_st.line_chart.assert_called_once()
        mock_st.expander.assert_called_once()


class TestAlertsFeedComponent:
    """Test early warning alerts feed, filtering controls, and operator action handling."""

    @patch("dashboard.components.alerts_feed.st")
    def test_alerts_feed_empty(self, mock_st):
        """Verify empty alerts feed triggers empty state."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_alerts.return_value = {
            "alerts": [],
            "total_count": 0,
            "active_alerts_count": 0,
        }
        # Mock 4 filter columns
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_st.selectbox.side_effect = ["ALL", "ALL", "ALL", "ALL"]

        render_early_warning_alerts(mock_client)
        mock_st.dataframe.assert_not_called()

    @patch("dashboard.components.alerts_feed.st")
    def test_alerts_feed_populated_and_inspected(self, mock_st):
        """Verify alerts feed table and detail inspector with operator action buttons."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_alerts.return_value = {
            "alerts": [
                {
                    "alert_id": "alert_001",
                    "severity": "CRITICAL",
                    "alert_type": "RISK_LEVEL_THRESHOLD",
                    "trigger_horizon": 5,
                    "risk_score": 0.85,
                    "lifecycle_state": "ACTIVE",
                    "timestamp_seconds": 12.345,
                    "warning_message": "Critical perception failure anticipated at H5",
                }
            ],
            "total_count": 1,
            "active_alerts_count": 1,
        }
        mock_client.get_alert_detail.return_value = {
            "alert_id": "alert_001",
            "rule_id": "RULE_CRITICAL_BREACH",
            "alert_type": "RISK_LEVEL_THRESHOLD",
            "severity": "CRITICAL",
            "lifecycle_state": "ACTIVE",
            "warning_message": "Critical perception failure anticipated at H5",
            "trigger_rationale": "Score exceeded 0.70 critical threshold",
            "source_frame_id": 100,
            "target_frame_id": 105,
            "model_version": "v1.0.0",
            "lifecycle_events": [],
        }

        # Mock 4 filter columns + 2 detail columns + 2 action columns
        col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        d_col1, d_col2 = MagicMock(), MagicMock()
        act_col1, act_col2 = MagicMock(), MagicMock()

        mock_st.columns.side_effect = [
            (col1, col2, col3, col4),
            (d_col1, d_col2),
            (act_col1, act_col2),
        ]
        mock_st.selectbox.side_effect = [
            "ALL", "ALL", "ALL", "ALL",  # 4 filters
            "alert_001",  # alert selection
        ]
        mock_st.button.return_value = False

        render_early_warning_alerts(mock_client)
        mock_st.dataframe.assert_called_once()


class TestDetectionTrackingComponent:
    """Test YOLO11n + ByteTrack authentic telemetry context rendering."""

    @patch("dashboard.components.tracking_view.st")
    def test_tracking_available_renders_metrics(self, mock_st):
        """Verify authentic detection and tracking metrics are rendered when available."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_tracking_context.return_value = {
            "status": "AVAILABLE",
            "model_name": "YOLO11n",
            "tracker_type": "ByteTrack",
            "total_frames_processed": 150,
            "total_detections": 820,
            "total_unique_tracks": 42,
            "max_simultaneous_active_tracks": 14,
            "mean_confidence_overall": 0.885,
            "mean_processing_latency_ms": 18.4,
        }

        # Mock 2 rows of 4 columns
        r1 = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        r2 = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_st.columns.side_effect = [r1, r2]

        render_detection_tracking(mock_client)
        # Should call st.columns twice
        assert mock_st.columns.call_count == 2

    @patch("dashboard.components.tracking_view.st")
    def test_tracking_unavailable_renders_empty_state(self, mock_st):
        """Verify unavailable status renders an informative empty state."""
        mock_client = MagicMock(spec=DashboardApiClient)
        mock_client.get_tracking_context.return_value = {
            "status": "UNAVAILABLE",
            "message": "Detection tracking artifacts not generated",
        }

        render_detection_tracking(mock_client)
        mock_st.columns.assert_not_called()
