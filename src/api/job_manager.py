"""In-process asynchronous job manager for video processing tasks.

Provides bounded, thread-safe background execution for long-running computer vision
and predictive reliability pipelines without requiring external infrastructure (Celery/Redis).
Optimized for resource-constrained edge machines (Intel Core i3 / 8 GB RAM).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from src.video_processing.video_pipeline_runner import VideoPipelineRunResult, VideoPipelineRunner

logger = logging.getLogger("perceptguard.api.jobs")


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobRecord:
    """Metadata and execution state for an in-flight or completed video processing job."""
    job_id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = "QUEUED"
    progress_percentage: float = 0.0
    message: str = "Job submitted and queued for execution."
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    max_frames_requested: int = 30
    frames_processed: int = 0
    total_detections: int = 0
    unique_tracks: int = 0
    alerts_count: int = 0
    error_message: Optional[str] = None
    results_dir: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class JobManager:
    """Singleton background job orchestrator maintaining thread-bounded video processing."""

    _instance: Optional[JobManager] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> JobManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, base_dir: Optional[Path] = None, max_workers: int = 1):
        if getattr(self, "_initialized", False):
            return
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd()
        self.jobs_dir = self.base_dir / "outputs" / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, JobRecord] = {}
        self._jobs_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pg_worker")
        self._runner = VideoPipelineRunner(project_root=Path.cwd())
        self._rehydrate_existing_jobs()
        self._initialized = True
        logger.info(f"Initialized JobManager (max_workers={max_workers}, jobs_dir={self.jobs_dir}, rehydrated={len(self._jobs)})")

    def _rehydrate_existing_jobs(self) -> None:
        """Scan jobs_dir and rehydrate completed job records into memory on startup for restart persistence."""
        if not self.jobs_dir.exists():
            return
        for child in self.jobs_dir.iterdir():
            if child.is_dir() and child.name.startswith("job_"):
                res_file = child / "results.json"
                if res_file.exists():
                    try:
                        with open(res_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        job_id = data.get("job_id", child.name)
                        rec = JobRecord(
                            job_id=job_id,
                            filename=data.get("video_filename", "video.mp4"),
                            status=JobStatus.COMPLETED,
                            stage="COMPLETED",
                            progress_percentage=100.0,
                            message=f"Restored from disk storage. {data.get('frames_processed', 0)} frames.",
                            created_at=data.get("completed_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                            completed_at=data.get("completed_at"),
                            max_frames_requested=data.get("frames_processed", 30),
                            frames_processed=data.get("frames_processed", 0),
                            total_detections=data.get("total_detections", 0),
                            unique_tracks=data.get("unique_tracks", 0),
                            alerts_count=data.get("total_alerts", 0),
                            results_dir=str(child),
                            result_summary={
                                "frames_processed": data.get("frames_processed", 0),
                                "total_detections": data.get("total_detections", 0),
                                "unique_tracks": data.get("unique_tracks", 0),
                                "total_alerts": data.get("total_alerts", 0),
                                "active_alerts": data.get("active_alerts", 0),
                                "total_duration_seconds": data.get("total_duration_seconds", 0.0),
                                "per_horizon_risk": data.get("per_horizon_risk", {}),
                            },
                        )
                        self._jobs[job_id] = rec
                    except Exception as e:
                        logger.warning(f"Could not rehydrate job from {res_file}: {e}")

    def submit_job(
        self,
        video_path: Union[str, Path],
        filename: str,
        max_frames: int = 30,
    ) -> JobRecord:
        """Submit a video file for asynchronous pipeline processing."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_out_dir = self.jobs_dir / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)

        record = JobRecord(
            job_id=job_id,
            filename=filename,
            status=JobStatus.QUEUED,
            stage="QUEUED",
            progress_percentage=0.0,
            message="Queued for processing",
            max_frames_requested=max_frames,
            results_dir=str(job_out_dir),
        )

        with self._jobs_lock:
            self._jobs[job_id] = record

        # Dispatch to worker thread
        self._executor.submit(
            self._execute_job_worker,
            job_id,
            Path(video_path),
            job_out_dir,
            max_frames,
        )
        logger.info(f"Submitted job {job_id} for video '{filename}' (max_frames={max_frames})")
        return record

    def _execute_job_worker(
        self,
        job_id: str,
        video_path: Path,
        job_out_dir: Path,
        max_frames: int,
    ) -> None:
        """Worker thread entry point executing the VideoPipelineRunner."""
        record = self.get_job(job_id)
        if not record:
            return

        with self._jobs_lock:
            record.status = JobStatus.PROCESSING
            record.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            record.stage = "VALIDATING"
            record.progress_percentage = 5.0
            record.message = "Starting video pipeline execution..."

        def _on_progress(stage: str, pct: float, msg: str):
            with self._jobs_lock:
                record.stage = stage
                record.progress_percentage = pct
                record.message = msg

        try:
            result: VideoPipelineRunResult = self._runner.run(
                video_path=video_path,
                job_id=job_id,
                output_dir=job_out_dir,
                max_frames=max_frames,
                progress_callback=_on_progress,
            )

            with self._jobs_lock:
                record.status = JobStatus.COMPLETED
                record.stage = "COMPLETED"
                record.progress_percentage = 100.0
                record.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                record.frames_processed = result.frames_processed
                record.total_detections = result.total_detections
                record.unique_tracks = result.unique_tracks
                record.alerts_count = result.total_alerts
                record.message = f"Completed successfully. {result.frames_processed} frames analyzed."
                record.result_summary = {
                    "frames_processed": result.frames_processed,
                    "total_detections": result.total_detections,
                    "unique_tracks": result.unique_tracks,
                    "total_alerts": result.total_alerts,
                    "active_alerts": result.active_alerts,
                    "total_duration_seconds": result.total_duration_seconds,
                    "per_horizon_risk": result.per_horizon_risk,
                }
            logger.info(f"Job {job_id} completed successfully in {result.total_duration_seconds:.2f}s")

        except Exception as exc:
            logger.error(f"Job {job_id} failed with error: {exc}", exc_info=True)
            with self._jobs_lock:
                record.status = JobStatus.FAILED
                record.stage = "FAILED"
                record.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                record.error_message = str(exc)
                record.message = f"Pipeline execution failed: {exc}"
        finally:
            try:
                if video_path.exists() and "uploads" in str(video_path):
                    video_path.unlink()
            except OSError:
                pass

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve job record by ID."""
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[JobRecord]:
        """List recently submitted jobs sorted by creation timestamp descending."""
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full structured results.json artifact for a completed job."""
        record = self.get_job(job_id)
        if not record or not record.results_dir:
            return None
        res_file = Path(record.results_dir) / "results.json"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Mark a queued job as cancelled if not yet completed."""
        with self._jobs_lock:
            rec = self._jobs.get(job_id)
            if rec and rec.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                rec.status = JobStatus.CANCELLED
                rec.message = "Job cancelled by operator."
                return True
        return False


def get_job_manager() -> JobManager:
    """Dependency helper providing singleton JobManager."""
    return JobManager()
