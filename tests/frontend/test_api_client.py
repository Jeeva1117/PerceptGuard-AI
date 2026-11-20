"""Comprehensive unit and integration tests for DashboardApiClient."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import httpx
from fastapi.testclient import TestClient

from src.config import get_settings
from src.api.app import create_app
from dashboard.api_client import (
    DashboardApiClient,
    FrontendAPIError,
    APIConnectionError,
    APITimeoutError,
    APIResponseError,
)


@pytest.fixture(scope="module")
def api_test_client():
    """FastAPI TestClient fixture for REST integration testing."""
    app = create_app()
    with TestClient(app) as tc:
        yield tc


class TestDashboardApiClientInitialization:
    """Test frontend application startup, configuration, and client setup."""

    def test_client_init_defaults(self):
        """Test default client initialization uses system settings."""
        client = DashboardApiClient()
        assert client.base_url.startswith("http")
        assert client.timeout > 0
        assert client.use_in_process_fallback is True
        assert client.connection_mode == "INITIALIZING"
        assert client.transport_mode == "INITIALIZING"
        client.close()

    def test_client_custom_config(self):
        """Test client with custom base URL and timeout parameters."""
        client = DashboardApiClient(
            base_url="http://custom-api:9000/",
            timeout=15.0,
            use_in_process_fallback=False,
        )
        assert client.base_url == "http://custom-api:9000"
        assert client.timeout == 15.0
        assert client.use_in_process_fallback is False
        client.close()


class TestDashboardApiClientRESTIntegration:
    """Test frontend API integration when communicating with FastAPI REST backend."""

    def test_probe_backend_live(self, api_test_client):
        """Verify probe_backend detects live HTTP backend and sets REST_API mode."""
        client = DashboardApiClient(http_client=api_test_client)
        is_live = client.probe_backend()
        assert is_live is True
        assert client.transport_mode == "REST_API"
        client.close()

    def test_get_system_health_rest(self, api_test_client):
        """Verify get_system_health executes over REST API."""
        client = DashboardApiClient(http_client=api_test_client)
        health = client.get_system_health()
        assert "pipeline_status" in health
        assert "overall_health" in health
        assert "health_score" in health
        assert "data_freshness" in health
        assert client.transport_mode == "REST_API"
        assert client.last_sync_time is not None
        client.close()

    def test_get_pipeline_status_rest(self, api_test_client):
        """Verify get_pipeline_status retrieves 8-stage perception architecture status."""
        client = DashboardApiClient(http_client=api_test_client)
        status = client.get_pipeline_status()
        assert "overall_status" in status
        assert "components" in status
        assert len(status["components"]) >= 8
        client.close()

    def test_get_risk_overview_rest(self, api_test_client):
        """Verify get_risk_overview retrieves multi-horizon risk (H1, H3, H5) over REST."""
        client = DashboardApiClient(http_client=api_test_client)
        risk = client.get_risk_overview()
        assert "horizons" in risk
        assert "h1" in risk["horizons"]
        assert "h3" in risk["horizons"]
        assert "h5" in risk["horizons"]
        client.close()

    def test_get_temporal_monitoring_rest(self, api_test_client):
        """Verify get_temporal_monitoring and alias get_temporal_data over REST."""
        client = DashboardApiClient(http_client=api_test_client)
        temp1 = client.get_temporal_monitoring(limit=25)
        assert "horizons" in temp1
        assert "state_transitions" in temp1
        temp2 = client.get_temporal_data(limit=25)
        assert len(temp2["horizons"]) == len(temp1["horizons"])
        client.close()

    def test_get_alerts_and_detail_rest(self, api_test_client):
        """Verify get_alerts filtering and detail retrieval over REST."""
        client = DashboardApiClient(http_client=api_test_client)
        alerts_page = client.get_alerts(severity="ALL", page=1, page_size=10)
        assert "alerts" in alerts_page
        assert "total_count" in alerts_page
        if alerts_page["alerts"]:
            first_id = alerts_page["alerts"][0]["alert_id"]
            detail = client.get_alert_detail(first_id)
            assert detail is not None
            assert detail["alert_id"] == first_id

        # Nonexistent alert should return None (404)
        missing_detail = client.get_alert_detail("nonexistent_alert_xyz")
        assert missing_detail is None
        client.close()

    def test_acknowledge_and_resolve_alert_rest(self, api_test_client):
        """Verify operator lifecycle actions (acknowledge, resolve) over REST."""
        client = DashboardApiClient(http_client=api_test_client)
        alerts_page = client.get_alerts(status="ACTIVE", page=1, page_size=5)
        if alerts_page["alerts"]:
            target_id = alerts_page["alerts"][0]["alert_id"]
            ack_res = client.acknowledge_alert(target_id, operator_id="test_operator")
            assert "acknowledged" in ack_res

            res_res = client.resolve_alert(
                target_id, reason="Resolved in test", operator_id="test_operator"
            )
            assert "resolved" in res_res
        client.close()

    def test_get_detection_tracking_rest(self, api_test_client):
        """Verify get_detection_tracking and alias get_tracking_context over REST."""
        client = DashboardApiClient(http_client=api_test_client)
        tr1 = client.get_detection_tracking()
        assert "status" in tr1
        tr2 = client.get_tracking_context()
        assert tr2["status"] == tr1["status"]
        client.close()


class TestDashboardApiClientFallbackAndResilience:
    """Test in-process fallback and error handling when REST API is offline or failing."""

    def test_backend_unavailable_fallback_to_in_process(self):
        """Verify client gracefully falls back to in-process service when REST endpoint is down."""
        # Point client to a port that is guaranteed not running backend
        client = DashboardApiClient(
            base_url="http://127.0.0.1:59999",
            timeout=0.5,
            use_in_process_fallback=True,
        )
        # Probe backend should return False
        is_live = client.probe_backend()
        assert is_live is False

        # Request should transparently fall back to in-process service
        health = client.get_system_health()
        assert "pipeline_status" in health
        assert client.transport_mode == "IN_PROCESS"
        assert client.last_sync_time is not None
        client.close()

    def test_backend_unavailable_raises_when_fallback_disabled(self):
        """Verify client raises connection/timeout error when backend is unreachable and fallback is disabled."""
        client = DashboardApiClient(
            base_url="http://127.0.0.1:59999",
            timeout=0.2,
            use_in_process_fallback=False,
        )
        with pytest.raises((APIConnectionError, APITimeoutError)) as exc_info:
            client.get_system_health()
        assert any(term in str(exc_info.value) for term in ("Cannot reach backend", "timed out"))
        client.close()

    def test_api_timeout_handling(self):
        """Verify APITimeoutError or fallback when request exceeds timeout."""
        # Use an unreachable IP that causes timeout
        client = DashboardApiClient(
            base_url="http://10.255.255.1:8000",
            timeout=0.1,
            use_in_process_fallback=False,
        )
        with pytest.raises((APITimeoutError, APIConnectionError)):
            client.get_system_health()
        client.close()

    def test_api_response_error_on_http_error(self):
        """Verify APIResponseError is raised when backend returns 5xx error."""
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.is_success = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.json.return_value = {"error": {"message": "Database crashed", "code": 500}}
        mock_http.request.return_value = mock_resp

        client = DashboardApiClient(
            http_client=mock_http,
            use_in_process_fallback=False,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.get_system_health()
        assert exc_info.value.status_code == 500
        assert "Database crashed" in exc_info.value.message
        client.close()
