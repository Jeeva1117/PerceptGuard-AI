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
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
    rotation: int = 0
    frames_processed: int = 0
    total_detections: int = 0
    unique_tracks: int = 0
    alerts_count: int = 0
    error_message: Optional[str] = None
    results_dir: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    video_path: Optional[str] = None
    has_source_video: bool = False
    analysis_mode: str = "FULL_BALANCED"
    sample_fps: Optional[float] = 2.0
    timeline_telemetry: Optional[Dict[str, Any]] = None
    event_timeline: Optional[List[Dict[str, Any]]] = None

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
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
            self.jobs_dir = self.base_dir / "outputs" / "jobs"
        else:
            # Support persistent storage mount (Render Persistent Disk or custom path)
            storage_env = os.environ.get("PERCEPTGUARD_STORAGE_DIR") or os.environ.get("OUTPUTS_DIR")
            if storage_env:
                p = Path(storage_env).resolve()
                self.jobs_dir = p / "jobs" if p.name != "jobs" else p
                self.base_dir = self.jobs_dir.parent
            else:
                try:
                    from src.config.system_settings import get_settings
                    settings = get_settings()
                    self.jobs_dir = settings.outputs_dir / "jobs"
                    self.base_dir = settings.base_dir
                except Exception:
                    self.base_dir = Path.cwd()
                    self.jobs_dir = self.base_dir / "outputs" / "jobs"

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        (self.jobs_dir / "uploads").mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, JobRecord] = {}
        self._uploaded_videos: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pg_worker")
        self._runner = VideoPipelineRunner(project_root=Path.cwd())
        self._rehydrate_existing_jobs()
        self._initialized = True
        logger.info(f"Initialized JobManager (max_workers={max_workers}, jobs_dir={self.jobs_dir}, rehydrated={len(self._jobs)})")

    def register_uploaded_video(
        self,
        video_id: str,
        file_path: Union[str, Path],
        filename: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store catalog entry for a validated user-uploaded video."""
        p = Path(file_path).resolve()
        entry = {
            "video_id": video_id,
            "filename": filename,
            "file_path": str(p),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": metadata,
        }
        with self._jobs_lock:
            self._uploaded_videos[video_id] = entry
        logger.info(f"Registered uploaded video {video_id} ('{filename}', {entry['size_bytes']} bytes)")
        return entry

    def get_uploaded_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve uploaded video catalog entry by video_id."""
        with self._jobs_lock:
            if video_id in self._uploaded_videos:
                return self._uploaded_videos[video_id]
            uploads_dir = self.jobs_dir / "uploads"
            if uploads_dir.exists():
                for f in uploads_dir.iterdir():
                    if f.is_file() and (f.name == video_id or f.name.startswith(f"{video_id}_")):
                        clean_name = f.name[len(video_id) + 1:] if f.name.startswith(f"{video_id}_") else f.name
                        entry = {
                            "video_id": video_id,
                            "filename": clean_name,
                            "file_path": str(f.resolve()),
                            "size_bytes": f.stat().st_size,
                            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "metadata": {},
                        }
                        self._uploaded_videos[video_id] = entry
                        return entry
        return None

    def _find_source_video(self, data: Dict[str, Any], child_dir: Path) -> Tuple[Optional[str], bool]:
        """Attempt to locate the authentic source video on disk for a job."""
        candidates = []
        if data.get("video_path"):
            candidates.append(Path(data["video_path"]))

        candidates.append(child_dir / "source_video.mp4")

        uploads_dir = self.jobs_dir / "uploads"
        v_filename = data.get("video_filename") or data.get("video_metadata", {}).get("filename")
        if v_filename:
            candidates.append(uploads_dir / v_filename)
            candidates.append(self.base_dir / "outputs" / "uploads" / v_filename)

        if uploads_dir.exists() and v_filename:
            clean_name = v_filename.split("_", 2)[-1] if v_filename.startswith("up_") else v_filename
            for f in uploads_dir.iterdir():
                if f.is_file() and (f.name == v_filename or clean_name in f.name):
                    candidates.append(f)

        for c in candidates:
            try:
                if c and c.exists() and c.is_file() and c.stat().st_size > 0:
                    return str(c.resolve()), True
            except OSError:
                continue

        return None, False

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
                        source_path, has_video = self._find_source_video(data, child)
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
                            video_path=source_path,
                            has_source_video=has_video,
                            analysis_mode=data.get("timeline_telemetry", {}).get("analysis_mode", "FULL_BALANCED"),
                            sample_fps=data.get("timeline_telemetry", {}).get("effective_sampling_fps", 2.0),
                            timeline_telemetry=data.get("timeline_telemetry"),
                            event_timeline=data.get("event_timeline"),
                            result_summary={
                                "frames_processed": data.get("frames_processed", 0),
                                "total_detections": data.get("total_detections", 0),
                                "unique_tracks": data.get("unique_tracks", 0),
                                "total_alerts": data.get("total_alerts", 0),
                                "active_alerts": data.get("active_alerts", 0),
                                "total_duration_seconds": data.get("total_duration_seconds", 0.0),
                                "per_horizon_risk": data.get("per_horizon_risk", {}),
                                "timeline_telemetry": data.get("timeline_telemetry"),
                            },
                        )
                        self._jobs[job_id] = rec
                    except Exception as e:
                        logger.warning(f"Could not rehydrate job from {res_file}: {e}")

    def submit_job(
        self,
        video_path: Union[str, Path],
        filename: str,
        max_frames: Optional[int] = None,
        rotation: int = 0,
        analysis_mode: str = "FULL_BALANCED",
        sample_fps: Optional[float] = None,
    ) -> JobRecord:
        """Submit a video file for asynchronous pipeline processing."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_out_dir = self.jobs_dir / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)

        vpath = Path(video_path).resolve()
        has_video = vpath.exists() and vpath.stat().st_size > 0

        record = JobRecord(
            job_id=job_id,
            filename=filename,
            status=JobStatus.QUEUED,
            stage="QUEUED",
            progress_percentage=0.0,
            message="Queued for processing",
            max_frames_requested=max_frames or 0,
            rotation=rotation,
            results_dir=str(job_out_dir),
            video_path=str(vpath),
            has_source_video=has_video,
            analysis_mode=analysis_mode,
            sample_fps=sample_fps or (2.0 if analysis_mode == "FULL_BALANCED" else (5.0 if analysis_mode == "FULL_DETAILED" else None)),
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
            rotation,
            analysis_mode,
            sample_fps,
        )
        logger.info(f"Submitted job {job_id} for video '{filename}' (mode={analysis_mode}, max_frames={max_frames}, rotation={rotation})")
        return record

    def _execute_job_worker(
        self,
        job_id: str,
        video_path: Path,
        job_out_dir: Path,
        max_frames: Optional[int],
        rotation: int = 0,
        analysis_mode: str = "FULL_BALANCED",
        sample_fps: Optional[float] = None,
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

        def _is_cancelled() -> bool:
            with self._jobs_lock:
                rec = self._jobs.get(job_id)
                return rec is not None and rec.status == JobStatus.CANCELLED

        try:
            result: VideoPipelineRunResult = self._runner.run(
                video_path=video_path,
                job_id=job_id,
                output_dir=job_out_dir,
                max_frames=max_frames,
                rotation=rotation,
                analysis_mode=analysis_mode,
                sample_fps=sample_fps,
                progress_callback=_on_progress,
                is_cancelled_callback=_is_cancelled,
            )

            with self._jobs_lock:
                if record.status == JobStatus.CANCELLED:
                    logger.info(f"Job {job_id} aborted following operator cancellation request.")
                    return
                record.status = JobStatus.COMPLETED
                record.stage = "COMPLETED"
                record.progress_percentage = 100.0
                record.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                record.frames_processed = result.frames_processed
                record.total_detections = result.total_detections
                record.unique_tracks = result.unique_tracks
                record.alerts_count = result.total_alerts
                record.timeline_telemetry = result.timeline_telemetry
                record.event_timeline = result.event_timeline
                record.message = f"Completed successfully. {result.frames_processed} frames analyzed across video timeline."
                record.result_summary = {
                    "frames_processed": result.frames_processed,
                    "total_detections": result.total_detections,
                    "unique_tracks": result.unique_tracks,
                    "total_alerts": result.total_alerts,
                    "active_alerts": result.active_alerts,
                    "total_duration_seconds": result.total_duration_seconds,
                    "per_horizon_risk": result.per_horizon_risk,
                    "timeline_telemetry": result.timeline_telemetry,
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
            # Preserve user uploaded video for session traceability and audit retention
            pass

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve job record by ID."""
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: Any = 20) -> List[JobRecord]:
        """List recently submitted jobs sorted by creation timestamp descending."""
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        try:
            lim = int(limit)
            return jobs[:lim]
        except (TypeError, ValueError):
            return jobs[:20]

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
        """Mark a queued or running job as cancelled."""
        with self._jobs_lock:
            rec = self._jobs.get(job_id)
            if rec and rec.status in (JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.VALIDATING):
                rec.status = JobStatus.CANCELLED
                rec.stage = "CANCELLED"
                rec.message = "Job cancelled by operator."
                rec.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                logger.info(f"Job {job_id} set to CANCELLED state.")
                return True
        return False

    def delete_job(self, job_id: str, delete_disk: bool = True) -> bool:
        """Cancel if running, remove from memory, and optionally delete on-disk artifacts."""
        with self._jobs_lock:
            rec = self._jobs.pop(job_id, None)

        # Check if directory exists on disk even if not in memory
        target_dir = self.jobs_dir / job_id
        if delete_disk and target_dir.exists() and target_dir.is_dir():
            try:
                shutil.rmtree(target_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to delete disk directory for job {job_id}: {e}")

        if rec and delete_disk and rec.results_dir:
            try:
                r_dir = Path(rec.results_dir)
                if r_dir.exists() and r_dir.is_dir() and r_dir != target_dir:
                    shutil.rmtree(r_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to delete results_dir for job {job_id}: {e}")

        logger.info(f"Deleted job {job_id} (delete_disk={delete_disk})")
        return rec is not None or target_dir.exists()

    def clean_stale_jobs(self) -> int:
        """Remove sample, dummy, or corrupted test jobs from memory and disk."""
        stale_ids = []
        with self._jobs_lock:
            for jid, rec in self._jobs.items():
                is_sample = (
                    "sample" in rec.filename.lower()
                    or "scenario_a" in rec.filename.lower()
                    or "dummy" in rec.filename.lower()
                )
                is_unbacked = not rec.has_source_video and (
                    rec.frames_processed == 0 or rec.status == JobStatus.FAILED
                )
                if is_sample or is_unbacked:
                    stale_ids.append(jid)

        count = 0
        for jid in stale_ids:
            if self.delete_job(jid, delete_disk=True):
                count += 1
        return count


def get_job_manager() -> JobManager:
    """Dependency helper providing singleton JobManager."""
    return JobManager()
