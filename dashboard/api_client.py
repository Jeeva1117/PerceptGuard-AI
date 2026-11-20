"""Centralized API client for PerceptGuard AI frontend dashboard.

Provides reliable HTTP transport to backend FastAPI services with automatic
timeout handling, structured error propagation, connection state tracking,
and seamless fallback to in-process DashboardService for unified deployment.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from src.config.system_settings import SystemSettings, get_settings
from src.early_warning.dashboard_service import DashboardService
from src.early_warning.pipeline_integrator import PipelineIntegrator

logger = logging.getLogger("perceptguard.frontend.client")


class FrontendAPIError(Exception):
    """Base exception for dashboard API client failures."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class APIConnectionError(FrontendAPIError):
    """Raised when backend REST service cannot be reached."""
    pass


class APITimeoutError(FrontendAPIError):
    """Raised when API request execution deadline is exceeded."""
    pass


class APIResponseError(FrontendAPIError):
    """Raised when API returns non-2xx error payload."""
    pass


class DashboardApiClient:
    """Robust, production-grade API client connecting frontend UI to backend services."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        settings: Optional[SystemSettings] = None,
        use_in_process_fallback: bool = True,
        http_client: Optional[httpx.Client] = None,
    ):
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.dashboard.api_base_url).rstrip("/")
        self.api_prefix = self.settings.api.prefix.rstrip("/")
        self.timeout = timeout
        self.use_in_process_fallback = use_in_process_fallback

        self._in_process_service: Optional[DashboardService] = None
        self._last_sync_timestamp: Optional[datetime.datetime] = None
        self._connection_mode: str = "INITIALIZING"
        self._last_error: Optional[str] = None

        # Initialize or accept HTTP client
        self._external_http_client = http_client is not None
        if http_client is not None:
            self._http_client = http_client
        else:
            self._http_client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"User-Agent": f"PerceptGuard-UI/{self.settings.app.version}"},
            )

    @property
    def connection_mode(self) -> str:
        """Current operational transport mode: 'REST_API' or 'IN_PROCESS'."""
        return self._connection_mode

    @property
    def transport_mode(self) -> str:
        """Current operational transport mode alias: 'REST_API' or 'IN_PROCESS'."""
        return self._connection_mode

    @property
    def is_in_process_ready(self) -> bool:
        """Whether in-process fallback service is instantiated and ready."""
        return self._in_process_service is not None

    def initialize_in_process_service(self, root_dir: Optional[Path] = None) -> None:
        """Explicitly initialize in-process DashboardService."""
        self._get_fallback_service()

    def probe_backend(self) -> bool:
        """Probe backend /health/live endpoint to determine if HTTP REST is reachable."""
        try:
            resp = self._http_client.get("/health/live", timeout=1.5)
            if resp.is_success:
                self._connection_mode = "REST_API"
                return True
        except Exception:
            pass
        self._connection_mode = "IN_PROCESS"
        return False

    @property
    def last_sync_time(self) -> Optional[datetime.datetime]:
        """Timestamp of most recent successful data retrieval."""
        return self._last_sync_timestamp

    @property
    def last_error(self) -> Optional[str]:
        """Description of last encountered operational exception."""
        return self._last_error

    def _get_fallback_service(self) -> DashboardService:
        """Instantiate or retrieve in-process DashboardService singleton."""
        if self._in_process_service is None:
            root_dir = self.settings.base_dir
            integrator = PipelineIntegrator(root_dir=root_dir)
            preds_file = (
                root_dir
                / "experiments"
                / "forecasting"
                / "multi_horizon"
                / "official_multi_horizon_run"
                / "predictions.json"
            )
            if preds_file.exists():
                integrator.ingest_predictions_file(preds_file)
            self._in_process_service = integrator.dashboard_service
        return self._in_process_service

    def _execute_request(
        self,
        method: str,
        path: str,
        fallback_fn: Optional[Any] = None,
        fallback_args: Optional[Tuple] = None,
        fallback_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute HTTP request with automatic error translation and in-process fallback."""
        url = f"{self.api_prefix}{path}"
        try:
            resp = self._http_client.request(method, url, **kwargs)
            if resp.is_success:
                self._connection_mode = "REST_API"
                self._last_sync_timestamp = datetime.datetime.now(datetime.timezone.utc)
                self._last_error = None
                return resp.json()

            # HTTP 4xx or 5xx
            err_data = None
            try:
                err_data = resp.json().get("error", {})
            except Exception:
                pass

            err_msg = err_data.get("message") if err_data else f"HTTP {resp.status_code}: {resp.text}"
            raise APIResponseError(
                message=err_msg,
                status_code=resp.status_code,
                details=err_data,
            )

        except (httpx.ConnectError, httpx.NetworkError) as conn_err:
            self._last_error = f"Cannot reach backend at {self.base_url}: {conn_err}"
            if self.use_in_process_fallback and fallback_fn is not None:
                logger.info(f"Backend HTTP unreachable. Falling back to in-process service for {path}.")
                self._connection_mode = "IN_PROCESS"
                self._last_sync_timestamp = datetime.datetime.now(datetime.timezone.utc)
                args = fallback_args or ()
                kws = fallback_kwargs or {}
                return fallback_fn(*args, **kws)
            raise APIConnectionError(self._last_error) from conn_err

        except httpx.TimeoutException as time_err:
            self._last_error = f"Request to {url} timed out after {self.timeout}s"
            if self.use_in_process_fallback and fallback_fn is not None:
                self._connection_mode = "IN_PROCESS"
                self._last_sync_timestamp = datetime.datetime.now(datetime.timezone.utc)
                args = fallback_args or ()
                kws = fallback_kwargs or {}
                return fallback_fn(*args, **kws)
            raise APITimeoutError(self._last_error) from time_err

    # --------------------------------------------------------------------------
    # API Methods
    # --------------------------------------------------------------------------

    def get_system_health(self) -> Dict[str, Any]:
        """Fetch system operational status, health score, and model availability."""
        fallback_fn = self._get_fallback_service().get_system_health
        return self._execute_request("GET", "/health", fallback_fn=fallback_fn)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Fetch execution status of all 8 perception pipeline stages."""
        fallback_fn = self._get_fallback_service().get_pipeline_status
        return self._execute_request("GET", "/telemetry/pipeline-status", fallback_fn=fallback_fn)

    def get_risk_overview(self, sequence_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch multi-horizon reliability risk scores (H1, H3, H5) and trends."""
        params = {"sequence_id": sequence_id} if sequence_id else {}
        fallback_fn = self._get_fallback_service().get_reliability_risk_overview
        return self._execute_request(
            "GET",
            "/telemetry/risk-overview",
            fallback_fn=fallback_fn,
            fallback_kwargs={"sequence_id": sequence_id},
            params=params,
        )

    def get_temporal_monitoring(
        self,
        sequence_id: Optional[str] = None,
        horizon: Optional[int] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch historical risk trajectories and discrete state transitions."""
        params: Dict[str, Any] = {"limit": limit}
        if sequence_id:
            params["sequence_id"] = sequence_id
        if horizon:
            params["horizon"] = horizon

        fallback_fn = self._get_fallback_service().get_temporal_monitoring_data
        return self._execute_request(
            "GET",
            "/telemetry/temporal-monitoring",
            fallback_fn=fallback_fn,
            fallback_kwargs={"sequence_id": sequence_id, "horizon": horizon, "limit": limit},
            params=params,
        )

    def get_temporal_data(
        self,
        sequence_id: Optional[str] = None,
        horizon: Optional[int] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Alias for get_temporal_monitoring."""
        return self.get_temporal_monitoring(sequence_id=sequence_id, horizon=horizon, limit=limit)

    def get_alerts(
        self,
        severity: Optional[str] = None,
        horizon: Optional[int] = None,
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Fetch paginated, filtered early-warning alert notifications."""
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if severity and severity != "ALL":
            params["severity"] = severity
        if horizon:
            params["horizon"] = horizon
        if alert_type and alert_type != "ALL":
            params["alert_type"] = alert_type
        if status and status != "ALL":
            params["status"] = status

        fallback_fn = self._get_fallback_service().get_alerts
        return self._execute_request(
            "GET",
            "/alerts",
            fallback_fn=fallback_fn,
            fallback_kwargs={
                "severity": severity if severity != "ALL" else None,
                "horizon": horizon,
                "alert_type": alert_type if alert_type != "ALL" else None,
                "status": status if status != "ALL" else None,
                "page": page,
                "page_size": page_size,
            },
            params=params,
        )

    def get_alert_detail(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full incident metadata, trigger rationale, and audit history."""
        fallback_fn = self._get_fallback_service().get_alert_detail
        try:
            return self._execute_request(
                "GET",
                f"/alerts/{alert_id}",
                fallback_fn=fallback_fn,
                fallback_kwargs={"alert_id": alert_id},
            )
        except APIResponseError as err:
            if err.status_code == 404:
                return None
            raise

    def acknowledge_alert(self, alert_id: str, operator_id: str = "dashboard_operator") -> Dict[str, Any]:
        """Acknowledge an active early-warning alert incident."""
        payload = {"operator_id": operator_id}
        fallback_fn = self._get_fallback_service().acknowledge_alert
        res = self._execute_request(
            "POST",
            f"/alerts/{alert_id}/acknowledge",
            fallback_fn=fallback_fn,
            fallback_kwargs={"alert_id": alert_id, "operator_id": operator_id},
            json=payload,
        )
        if "success" in res and "acknowledged" not in res:
            res["acknowledged"] = res["success"]
        return res

    def resolve_alert(
        self, alert_id: str, reason: str = "Resolved from dashboard", operator_id: str = "dashboard_operator"
    ) -> Dict[str, Any]:
        """Resolve an alert with mandatory operational justification."""
        payload = {"reason": reason, "operator_id": operator_id}
        fallback_fn = self._get_fallback_service().resolve_alert
        res = self._execute_request(
            "POST",
            f"/alerts/{alert_id}/resolve",
            fallback_fn=fallback_fn,
            fallback_kwargs={"alert_id": alert_id, "reason": reason, "operator_id": operator_id},
            json=payload,
        )
        if "success" in res and "resolved" not in res:
            res["resolved"] = res["success"]
        return res

    def get_detection_tracking(self, sequence_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch YOLO detection and ByteTrack tracking context."""
        params = {"sequence_id": sequence_id} if sequence_id else {}
        fallback_fn = self._get_fallback_service().get_detection_tracking_context
        return self._execute_request(
            "GET",
            "/telemetry/detection-tracking",
            fallback_fn=fallback_fn,
            fallback_kwargs={"sequence_id": sequence_id},
            params=params,
        )

    def get_tracking_context(self, sequence_id: Optional[str] = None) -> Dict[str, Any]:
        """Alias for get_detection_tracking."""
        return self.get_detection_tracking(sequence_id=sequence_id)

    def validate_video_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Upload video file to backend validation service."""
        files = {"file": (filename, file_bytes, "video/mp4")}
        return self._execute_request(
            "POST",
            "/video/validate",
            files=files,
        )

    def submit_video_job(self, file_bytes: bytes, filename: str, max_frames: int = 30) -> Dict[str, Any]:
        """Submit video for full asynchronous pipeline processing."""
        files = {"file": (filename, file_bytes, "video/mp4")}
        params = {"max_frames": max_frames}

        def _fallback_submit(f_bytes, fname, m_frames):
            from src.api.job_manager import get_job_manager
            import tempfile
            job_mgr = get_job_manager()
            temp_path = Path(tempfile.gettempdir()) / f"inproc_{fname}"
            with open(temp_path, "wb") as f_out:
                f_out.write(f_bytes)
            job = job_mgr.submit_job(video_path=temp_path, filename=fname, max_frames=m_frames)
            return {
                "job_id": job.job_id,
                "filename": job.filename,
                "status": job.status.value,
                "stage": job.stage,
                "message": job.message,
                "max_frames": m_frames,
                "created_at": job.created_at,
            }

        return self._execute_request(
            "POST",
            "/video/jobs",
            fallback_fn=_fallback_submit,
            fallback_args=(file_bytes, filename, max_frames),
            files=files,
            params=params,
        )

    def get_video_job_status(self, job_id: str) -> Dict[str, Any]:
        """Fetch live execution status and stage progress for a video job."""
        def _fallback_status(j_id):
            from src.api.job_manager import get_job_manager
            job = get_job_manager().get_job(j_id)
            if not job:
                return {"job_id": j_id, "status": "NOT_FOUND", "stage": "UNKNOWN", "progress_percentage": 0.0, "message": "Job not found"}
            return job.to_dict()

        return self._execute_request(
            "GET",
            f"/video/jobs/{job_id}",
            fallback_fn=_fallback_status,
            fallback_args=(job_id,),
        )

    def get_video_job_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve full structured results.json artifact for a completed job."""
        def _fallback_results(j_id):
            from src.api.job_manager import get_job_manager
            res = get_job_manager().get_job_results(j_id)
            return res or {}

        return self._execute_request(
            "GET",
            f"/video/jobs/{job_id}/results",
            fallback_fn=_fallback_results,
            fallback_args=(job_id,),
        )

    def list_video_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recently submitted video jobs."""
        def _fallback_list(lim):
            from src.api.job_manager import get_job_manager
            jobs = get_job_manager().list_jobs(limit=lim)
            return [j.to_dict() for j in jobs]

        res = self._execute_request(
            "GET",
            "/video/jobs",
            fallback_fn=_fallback_list,
            fallback_args=(limit,),
            params={"limit": limit},
        )
        return res if isinstance(res, list) else []

    def get_annotated_frame_url(self, job_id: str, frame_name: str) -> str:
        """Construct URL or local filesystem path for an annotated frame."""
        if self._connection_mode == "REST_API":
            return f"{self.base_url}{self.api_prefix}/video/jobs/{job_id}/frames/{frame_name}"
        else:
            from src.api.job_manager import get_job_manager
            job = get_job_manager().get_job(job_id)
            if job and job.results_dir:
                p = Path(job.results_dir) / "annotated" / frame_name
                if p.exists():
                    return str(p)
                p_raw = Path(job.results_dir) / "raw_frames" / frame_name
                if p_raw.exists():
                    return str(p_raw)
            return ""

    def close(self) -> None:
        """Release underlying HTTP client connections."""
        if not getattr(self, "_external_http_client", False):
            self._http_client.close()
