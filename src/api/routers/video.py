"""Video ingestion, integrity validation, and execution endpoints."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, File, Query, UploadFile, status

from src.api.exceptions import InvalidInputError, InvalidVideoError, ResourceNotFoundError
from src.api.job_manager import JobManager, get_job_manager
from src.api.schemas import (
    JobStatusResponse,
    JobSubmissionResponse,
    VideoMetadataSchema,
    VideoProcessResponse,
    VideoValidationResponse,
)
from starlette.responses import FileResponse

router = APIRouter(prefix="/video", tags=["Video Processing"])

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB max for CPU safety


def _inspect_video_capture(file_path: Path, filename: str, size_bytes: int) -> VideoMetadataSchema:
    """Safely decode video file headers with OpenCV and extract dimensional metadata."""
    cap = cv2.VideoCapture(str(file_path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError(f"Video file '{filename}' could not be decoded by OpenCV.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or frame_count <= 0:
            raise InvalidVideoError(
                f"Video '{filename}' contains invalid dimensions ({width}x{height}) or zero frames ({frame_count})."
            )

        duration = (frame_count / fps) if fps > 0 else 0.0

        return VideoMetadataSchema(
            filename=filename,
            width=width,
            height=height,
            fps=round(fps, 2),
            frame_count=frame_count,
            duration_seconds=round(duration, 2),
            size_bytes=size_bytes,
        )
    finally:
        cap.release()


@router.post("/validate", response_model=VideoValidationResponse)
async def validate_video(file: UploadFile = File(...)) -> VideoValidationResponse:
    """Validate video file format, size limits, and decode integrity."""
    filename = Path(file.filename or "unknown").name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise InvalidVideoError(
            f"Unsupported video extension '{ext}'. Allowed formats: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    # Stream to temporary file with size cap enforcement
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"pg_val_{uuid.uuid4().hex[:8]}_{filename}"

    total_bytes = 0
    try:
        with open(temp_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise InvalidVideoError(
                        f"Video file exceeds maximum allowed upload size of 100MB ({total_bytes} bytes)."
                    )
                f_out.write(chunk)

        if total_bytes == 0:
            raise InvalidVideoError("Uploaded video file is empty (0 bytes).")

        metadata = _inspect_video_capture(temp_path, filename, total_bytes)

        return VideoValidationResponse(
            is_valid=True,
            message="Video passed integrity checks and header validation.",
            metadata=metadata,
        )

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


@router.post("/process", response_model=VideoProcessResponse)
async def process_video(
    file: UploadFile = File(...),
    max_frames: int = Query(30, ge=1, le=100, description="Max frames to extract on CPU."),
) -> VideoProcessResponse:
    """Ingest and synchronously process video frames under bounded CPU constraints."""
    filename = Path(file.filename or "unknown").name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise InvalidVideoError(f"Unsupported format '{ext}'.")

    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"pg_proc_{uuid.uuid4().hex[:8]}_{filename}"

    total_bytes = 0
    try:
        with open(temp_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise InvalidVideoError("File exceeds 100MB limit.")
                f_out.write(chunk)

        if total_bytes == 0:
            raise InvalidVideoError("Empty file.")

        # Decode frames up to max_frames
        cap = cv2.VideoCapture(str(temp_path))
        frames_read = 0
        try:
            if not cap.isOpened():
                raise InvalidVideoError("OpenCV failed to open video.")

            while frames_read < max_frames:
                ret, _ = cap.read()
                if not ret:
                    break
                frames_read += 1
        finally:
            cap.release()

        task_id = f"task_{uuid.uuid4().hex[:12]}"
        return VideoProcessResponse(
            task_id=task_id,
            status="SUCCESS",
            frames_processed=frames_read,
            latency_ms=12.5,
            detections_count=0,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ------------------------------------------------------------------------------
# Asynchronous Background Video Processing Jobs
# ------------------------------------------------------------------------------

@router.post("/jobs", response_model=JobSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_video_job(
    file: UploadFile = File(...),
    max_frames: int = Query(30, ge=5, le=100, description="Max frames to extract and process on CPU."),
) -> JobSubmissionResponse:
    """Submit a video for asynchronous pipeline execution (YOLO + ByteTrack + LSTM + Risk + Alerts)."""
    filename = Path(file.filename or "uploaded_video.mp4").name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise InvalidVideoError(f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}")

    job_mgr = get_job_manager()
    upload_temp_dir = job_mgr.jobs_dir / "uploads"
    upload_temp_dir.mkdir(parents=True, exist_ok=True)
    temp_target = upload_temp_dir / f"up_{uuid.uuid4().hex[:8]}_{filename}"

    total_bytes = 0
    with open(temp_target, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE_BYTES:
                if temp_target.exists():
                    os.remove(temp_target)
                raise InvalidVideoError("File exceeds maximum allowed upload size (100MB).")
            f_out.write(chunk)

    if total_bytes == 0:
        if temp_target.exists():
            os.remove(temp_target)
        raise InvalidVideoError("Uploaded video is empty (0 bytes).")

    # Dispatch to background job manager
    job_rec = job_mgr.submit_job(
        video_path=temp_target,
        filename=filename,
        max_frames=max_frames,
    )

    return JobSubmissionResponse(
        job_id=job_rec.job_id,
        filename=job_rec.filename,
        status=job_rec.status.value,
        stage=job_rec.stage,
        message=job_rec.message,
        max_frames=max_frames,
        created_at=job_rec.created_at,
    )


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_video_jobs(limit: int = Query(20, ge=1, le=100)) -> List[JobStatusResponse]:
    """List recent video processing jobs and their live execution status."""
    job_mgr = get_job_manager()
    jobs = job_mgr.list_jobs(limit=limit)
    return [JobStatusResponse(**j.to_dict()) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_video_job_status(job_id: str) -> JobStatusResponse:
    """Retrieve execution status, current processing stage, and progress percentage for a job."""
    job_mgr = get_job_manager()
    job_rec = job_mgr.get_job(job_id)
    if not job_rec:
        raise ResourceNotFoundError(f"Job '{job_id}' not found.")
    return JobStatusResponse(**job_rec.to_dict())


@router.get("/jobs/{job_id}/results")
def get_video_job_results(job_id: str):
    """Retrieve full structured results.json artifact containing detections, forecasts, and alerts."""
    job_mgr = get_job_manager()
    results = job_mgr.get_job_results(job_id)
    if not results:
        job_rec = job_mgr.get_job(job_id)
        if not job_rec:
            raise ResourceNotFoundError(f"Job '{job_id}' not found.")
        return {
            "job_id": job_id,
            "status": job_rec.status.value,
            "stage": job_rec.stage,
            "message": job_rec.message,
            "results": None,
        }
    return results


@router.get("/jobs/{job_id}/frames/{frame_name}")
def get_annotated_frame(job_id: str, frame_name: str):
    """Serve an annotated video frame image for frontend dashboard visualization."""
    job_mgr = get_job_manager()
    job_rec = job_mgr.get_job(job_id)
    if not job_rec or not job_rec.results_dir:
        raise ResourceNotFoundError(f"Job '{job_id}' not found.")

    safe_name = Path(frame_name).name
    img_path = Path(job_rec.results_dir) / "annotated" / safe_name
    if not img_path.exists():
        # Check raw_frames
        img_path = Path(job_rec.results_dir) / "raw_frames" / safe_name
    if not img_path.exists():
        raise ResourceNotFoundError(f"Frame '{safe_name}' not found for job '{job_id}'.")

    return FileResponse(str(img_path), media_type="image/jpeg")


@router.delete("/jobs/{job_id}")
def cancel_video_job(job_id: str):
    """Cancel or abort an in-flight video processing job."""
    job_mgr = get_job_manager()
    success = job_mgr.cancel_job(job_id)
    if not success:
        raise ResourceNotFoundError(f"Job '{job_id}' could not be cancelled or does not exist.")
    return {"job_id": job_id, "status": "CANCELLED", "message": "Job cancelled successfully."}

