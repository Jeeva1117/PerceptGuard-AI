import type {
  JobStatusResponse,
  StartProcessingRequest,
  VideoPipelineRunResult,
  VideoUploadResponse,
} from '../types/api';

const BACKEND_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');
const API_BASE = `${BACKEND_BASE}/api/v1`;

export async function fetchHealth(): Promise<{ status: string; timestamp: string }> {
  const res = await fetch(`${BACKEND_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function listJobs(limit = 20): Promise<JobStatusResponse[]> {
  const res = await fetch(`${API_BASE}/video/jobs?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to list jobs: ${res.statusText}`);
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`Failed to get job status: ${res.statusText}`);
  return res.json();
}

export async function getJobResults(jobId: string): Promise<VideoPipelineRunResult> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/results`);
  if (!res.ok) throw new Error(`Failed to get job results: ${res.statusText}`);
  return res.json();
}

export async function uploadVideoFile(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/video/upload-file`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
  }

  return res.json();
}

export async function startProcessingVideo(
  req: StartProcessingRequest
): Promise<{ job_id: string; message: string; status: string }> {
  const res = await fetch(`${API_BASE}/video/start-processing`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Start processing failed with status ${res.status}`);
  }

  return res.json();
}

export async function submitVideoJob(
  file: File,
  maxFrames?: number,
  rotation = 0,
  analysisMode = 'FULL_BALANCED',
  sampleFps?: number
): Promise<{ job_id: string; message: string; status: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (maxFrames !== undefined && maxFrames !== null && maxFrames > 0) {
    params.append('max_frames', String(maxFrames));
  }
  params.append('rotation', String(rotation));
  params.append('analysis_mode', analysisMode);
  if (sampleFps) {
    params.append('sample_fps', String(sampleFps));
  }

  const res = await fetch(`${API_BASE}/video/jobs?${params.toString()}`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
  }

  return res.json();
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; status: string; message?: string }> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    const delRes = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
    });
    if (!delRes.ok) throw new Error(`Failed to cancel job: ${delRes.statusText}`);
    return delRes.json();
  }
  return res.json();
}

export async function deleteJob(jobId: string, deleteDisk = true): Promise<{ job_id: string; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}?delete_disk=${deleteDisk}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete job: ${res.statusText}`);
  return res.json();
}

export async function cleanStaleJobs(): Promise<{ cleaned_count: number; message: string }> {
  const res = await fetch(`${API_BASE}/video/jobs/clean-stale`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to clean stale jobs: ${res.statusText}`);
  return res.json();
}

export function getSourceVideoUrl(jobId?: string | null): string {
  if (!jobId) return '';
  return `${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/source-video`;
}

export function getVideoStreamUrl(videoIdOrJobId?: string | null): string {
  if (!videoIdOrJobId) return '';
  if (videoIdOrJobId.startsWith('vid_')) {
    return `${API_BASE}/video/stream/${encodeURIComponent(videoIdOrJobId)}`;
  }
  return `${API_BASE}/video/jobs/${encodeURIComponent(videoIdOrJobId)}/source-video`;
}

export function getAnnotatedVideoUrl(jobId?: string | null): string {
  if (!jobId) return '';
  return `${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/annotated-video`;
}

export function getUploadStreamUrl(filename?: string | null): string {
  if (!filename) return '';
  const safeName = String(filename).split(/[/\\]/).pop() || filename;
  return `${API_BASE}/video/uploads/${encodeURIComponent(safeName)}/stream`;
}

export function getFrameImageUrl(jobId?: string | null, frameName?: string | null): string {
  if (!jobId || !frameName) return '';
  // Extract basename if full path was given
  const safeName = String(frameName).split(/[/\\]/).pop() || frameName;
  return `${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/frames/${encodeURIComponent(safeName)}`;
}

export async function getJobWeather(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/weather`);
  if (!res.ok) throw new Error(`Failed to get weather data: ${res.statusText}`);
  return res.json();
}

export async function getJobTrajectories(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/trajectories`);
  if (!res.ok) throw new Error(`Failed to get trajectory data: ${res.statusText}`);
  return res.json();
}

export async function getJobCollisionRisk(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/collision-risk`);
  if (!res.ok) throw new Error(`Failed to get collision risk data: ${res.statusText}`);
  return res.json();
}

export async function getJobSafetyRecommendations(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/safety-recommendations`);
  if (!res.ok) throw new Error(`Failed to get safety recommendations: ${res.statusText}`);
  return res.json();
}

export async function getJobTimeline(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/timeline`);
  if (!res.ok) throw new Error(`Failed to get timeline data: ${res.statusText}`);
  return res.json();
}

export async function getJobTimelineTelemetry(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/video/jobs/${encodeURIComponent(jobId)}/timeline-telemetry`);
  if (!res.ok) throw new Error(`Failed to get timeline telemetry: ${res.statusText}`);
  return res.json();
}
