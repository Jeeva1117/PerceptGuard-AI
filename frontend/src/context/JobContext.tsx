import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react';
import type {
  JobStatusResponse,
  VideoPipelineRunResult,
  FrameProcessingDetail,
  VideoUploadResponse,
  AnalysisMode,
} from '../types/api';
import {
  listJobs,
  getJobResults,
  fetchHealth,
  deleteJob,
  cleanStaleJobs,
  uploadVideoFile,
  startProcessingVideo,
  getJobStatus,
  cancelJob,
  getVideoStreamUrl,
} from '../services/api';
import { useVoiceAlerts } from '../hooks/useVoiceAlerts';

export type VideoProcessingState =
  | 'NO_VIDEO'
  | 'UPLOADING'
  | 'VIDEO_READY'
  | 'PROCESSING'
  | 'COMPLETED'
  | 'UPLOAD_FAILED'
  | 'PROCESSING_FAILED'
  | 'CANCELLED';

export interface StartProcessingOptions {
  analysisMode?: AnalysisMode;
  rotation?: 0 | 90 | 180 | 270;
  sampleFps?: number;
  maxFrames?: number | null;
}

interface JobContextType {
  // Job & Results Telemetry
  jobs: JobStatusResponse[];
  activeJobId: string | null;
  activeJob: JobStatusResponse | null;
  results: VideoPipelineRunResult | null;
  currentFrameIndex: number;
  currentFrame: FrameProcessingDetail | null;

  // Unified Video & Processing State Machine
  videoProcessingState: VideoProcessingState;
  uploadProgressPct: number;
  processingProgressPct: number;
  statusMessage: string;
  errorMessage: string | null;
  uploadedFile: File | null;
  analysisMode: AnalysisMode;
  rotation: 0 | 90 | 180 | 270;
  sampleFps: number;

  // Video Media & Playback State
  isPlaying: boolean;
  playbackSpeed: number;
  currentTime: number;
  duration: number;
  playbackMode: 'ORIGINAL_LIVE' | 'ANNOTATED_PRERENDERED';
  isVoiceMuted: boolean;
  isLoading: boolean;
  isBackendOnline: boolean;
  error: string | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  selectedTrackId: number | null;
  stagedVideoSrc: string | null;
  uploadedVideoId: string | null;
  uploadedVideoMetadata: VideoUploadResponse | null;

  // Operations & State Mutators
  setAnalysisMode: (mode: AnalysisMode) => void;
  setRotation: (rotation: 0 | 90 | 180 | 270) => void;
  setSampleFps: (fps: number) => void;
  uploadFile: (file: File) => Promise<void>;
  startProcessing: (options?: StartProcessingOptions) => Promise<void>;
  cancelProcessing: () => Promise<void>;
  retryProcessing: () => Promise<void>;
  resetForNewVideo: () => void;
  setSelectedTrackId: (id: number | null) => void;
  setStagedVideo: (src: string | null, videoId?: string | null, metadata?: VideoUploadResponse | null) => void;
  clearStagedVideo: () => void;
  refreshJobs: () => Promise<void>;
  selectJob: (jobId: string) => Promise<void>;
  clearActiveJob: () => void;
  deleteJobById: (jobId: string) => Promise<void>;
  cleanStale: () => Promise<number>;
  setCurrentFrameIndex: (idx: number | ((prev: number) => number)) => void;
  setIsPlaying: (playing: boolean | ((prev: boolean) => boolean)) => void;
  setPlaybackSpeed: (speed: number) => void;
  setPlaybackMode: (mode: 'ORIGINAL_LIVE' | 'ANNOTATED_PRERENDERED') => void;
  toggleVoiceMute: () => void;
  playVideo: () => void;
  pauseVideo: () => void;
  togglePlay: () => void;
  seekToTimestamp: (timeSec: number) => void;
  seekToFrame: (frameIdx: number) => void;
  handleTimeUpdate: (timeSec: number) => void;
  handleDurationChange: (durSec: number) => void;
}

const JobContext = createContext<JobContextType | undefined>(undefined);

// Fast binary search to find closest frame by timestamp
function findClosestFrameIndex(frames: FrameProcessingDetail[], time: number): number {
  if (frames.length === 0) return 0;
  let low = 0;
  let high = frames.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const midTime = frames[mid].timestamp_seconds ?? 0;
    if (midTime === time) return mid;
    if (midTime < time) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  const idx1 = Math.max(0, Math.min(frames.length - 1, high));
  const idx2 = Math.max(0, Math.min(frames.length - 1, low));
  const diff1 = Math.abs((frames[idx1].timestamp_seconds ?? 0) - time);
  const diff2 = Math.abs((frames[idx2].timestamp_seconds ?? 0) - time);
  return diff1 <= diff2 ? idx1 : idx2;
}

export const JobProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [results, setResults] = useState<VideoPipelineRunResult | null>(null);
  const [currentFrameIndex, setCurrentFrameIndex] = useState<number>(0);

  // Unified State Machine
  const [videoProcessingState, setVideoProcessingState] = useState<VideoProcessingState>('NO_VIDEO');
  const [uploadProgressPct, setUploadProgressPct] = useState<number>(0);
  const [processingProgressPct, setProcessingProgressPct] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Analysis Configuration
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('FULL_BALANCED');
  const [rotation, setRotation] = useState<0 | 90 | 180 | 270>(0);
  const [sampleFps, setSampleFps] = useState<number>(2.0);

  // Media Player State
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeedState] = useState<number>(1);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [playbackMode, setPlaybackMode] = useState<'ORIGINAL_LIVE' | 'ANNOTATED_PRERENDERED'>('ORIGINAL_LIVE');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [stagedVideoSrc, setStagedVideoSrc] = useState<string | null>(null);
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(null);
  const [uploadedVideoMetadata, setUploadedVideoMetadata] = useState<VideoUploadResponse | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const prevJobIdRef = useRef<string | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const { isVoiceMuted, toggleVoiceMute, triggerVoiceAlert } = useVoiceAlerts();

  const activeJob = jobs.find((j) => j.job_id === activeJobId) || null;
  const currentFrame = results?.frames?.[currentFrameIndex] || null;

  const setStagedVideo = useCallback((src: string | null, videoId?: string | null, metadata?: VideoUploadResponse | null) => {
    setStagedVideoSrc(src);
    if (videoId !== undefined && videoId !== null) {
      setUploadedVideoId(videoId);
      setVideoProcessingState('VIDEO_READY');
      setStatusMessage('Video staged on server. Ready for processing.');
    }
    if (metadata !== undefined) setUploadedVideoMetadata(metadata);
  }, []);

  const clearStagedVideo = useCallback(() => {
    setStagedVideoSrc(null);
    setUploadedVideoId(null);
    setUploadedVideoMetadata(null);
  }, []);

  // Check backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await fetchHealth();
        setIsBackendOnline(true);
      } catch {
        setIsBackendOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch job list
  const refreshJobs = useCallback(async () => {
    try {
      setError(null);
      const jobList = await listJobs(30);
      setJobs(jobList);
      setIsBackendOnline(true);
    } catch (err: any) {
      console.warn('Failed to load jobs:', err);
      setIsBackendOnline(false);
      setError(`Backend stream unreachable: ${err.message || err}`);
    }
  }, []);

  useEffect(() => {
    refreshJobs();
  }, [refreshJobs]);

  // Stop polling helper
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // Polling loop for active job
  const startJobPolling = useCallback((jobId: string) => {
    stopPolling();
    const timer = window.setInterval(async () => {
      try {
        const stat = await getJobStatus(jobId);
        if (stat.status === 'PROCESSING') {
          const pct = stat.progress_pct ?? stat.progress_percentage ?? 25;
          setProcessingProgressPct(Math.max(10, Math.min(95, pct)));
          setStatusMessage(stat.message || `Processing frame ${stat.frames_processed || 0}...`);
        } else if (stat.status === 'COMPLETED') {
          stopPolling();
          setProcessingProgressPct(100);
          setStatusMessage('Analysis completed successfully.');
          setVideoProcessingState('COMPLETED');
          // Load full results
          try {
            const data = await getJobResults(jobId);
            setResults(data);
            if (data.timeline_telemetry?.source_video_duration_sec) {
              setDuration(data.timeline_telemetry.source_video_duration_sec);
            } else if (data.video_metadata?.duration_seconds) {
              setDuration(data.video_metadata.duration_seconds);
            }
          } catch (e: any) {
            console.error('Failed to load completed job results:', e);
          }
          refreshJobs();
        } else if (stat.status === 'FAILED') {
          stopPolling();
          setVideoProcessingState('PROCESSING_FAILED');
          setErrorMessage(stat.error_message || stat.message || 'Pipeline execution failed.');
          setStatusMessage('Processing failed.');
          refreshJobs();
        } else if (stat.status === 'CANCELLED') {
          stopPolling();
          setVideoProcessingState('CANCELLED');
          setStatusMessage('Processing was cancelled.');
          refreshJobs();
        }
      } catch (err: any) {
        console.warn('Job polling error:', err);
      }
    }, 1000);
    pollTimerRef.current = timer;
  }, [stopPolling, refreshJobs]);

  // Clean active job back to clean upload state
  const clearActiveJob = useCallback(() => {
    stopPolling();
    setActiveJobId(null);
    prevJobIdRef.current = null;
    setResults(null);
    setError(null);
    setIsPlaying(false);
    setCurrentFrameIndex(0);
    setCurrentTime(0);
    setDuration(0);
    setSelectedTrackId(null);
    setStagedVideoSrc(null);
    setUploadedVideoId(null);
    setUploadedVideoMetadata(null);
    setUploadedFile(null);
    setVideoProcessingState('NO_VIDEO');
    setStatusMessage('');
    setErrorMessage(null);
    setUploadProgressPct(0);
    setProcessingProgressPct(0);
    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      } catch {
        // ignore
      }
    }
  }, [stopPolling]);

  const resetForNewVideo = useCallback(() => {
    clearActiveJob();
  }, [clearActiveJob]);

  // Select a historical job or reset
  const selectJob = useCallback(async (jobId: string) => {
    stopPolling();
    if (!jobId) {
      clearActiveJob();
      return;
    }

    setActiveJobId(jobId);
    setResults(null);
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);
    setCurrentFrameIndex(0);
    setCurrentTime(0);

    try {
      const stat = await getJobStatus(jobId);
      if (stat.status === 'COMPLETED') {
        const data = await getJobResults(jobId);
        setResults(data);
        setVideoProcessingState('COMPLETED');
        setStatusMessage(`Restored completed session: ${stat.filename || jobId}`);
        if (data.timeline_telemetry?.source_video_duration_sec) {
          setDuration(data.timeline_telemetry.source_video_duration_sec);
        } else if (data.video_metadata?.duration_seconds) {
          setDuration(data.video_metadata.duration_seconds);
        }
      } else if (stat.status === 'PROCESSING' || stat.status === 'QUEUED') {
        setVideoProcessingState('PROCESSING');
        setStatusMessage(stat.message || 'Processing in progress...');
        startJobPolling(jobId);
      } else if (stat.status === 'FAILED') {
        setVideoProcessingState('PROCESSING_FAILED');
        setErrorMessage(stat.error_message || stat.message || 'Pipeline execution failed.');
        setStatusMessage('Historical job failed.');
      } else if (stat.status === 'CANCELLED') {
        setVideoProcessingState('CANCELLED');
        setStatusMessage('Job was cancelled.');
      }
    } catch (err: any) {
      setResults(null);
      setError(`Failed to load job telemetry: ${err.message || err}`);
      setVideoProcessingState('PROCESSING_FAILED');
    } finally {
      setIsLoading(false);
    }
  }, [stopPolling, clearActiveJob, startJobPolling]);

  // File Upload Action
  const uploadFile = useCallback(async (file: File) => {
    stopPolling();
    clearActiveJob();

    // 1. Validate extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    if (!allowed.includes(ext)) {
      setVideoProcessingState('UPLOAD_FAILED');
      setErrorMessage(`Unsupported video extension '${ext}'. Supported formats: ${allowed.join(', ')}`);
      return;
    }

    // 2. Validate size
    if (file.size === 0) {
      setVideoProcessingState('UPLOAD_FAILED');
      setErrorMessage('Selected video file is empty (0 bytes).');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setVideoProcessingState('UPLOAD_FAILED');
      setErrorMessage(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds 100 MB limit.`);
      return;
    }

    setUploadedFile(file);
    setVideoProcessingState('UPLOADING');
    setUploadProgressPct(25);
    setStatusMessage(`Uploading '${file.name}' to server storage...`);
    setErrorMessage(null);

    // Staged local preview immediately
    const localUrl = URL.createObjectURL(file);
    setStagedVideoSrc(localUrl);

    try {
      const resp = await uploadVideoFile(file);
      setUploadProgressPct(100);
      setUploadedVideoId(resp.video_id);
      setUploadedVideoMetadata(resp);
      const serverStreamUrl = getVideoStreamUrl(resp.video_id);
      setStagedVideoSrc(serverStreamUrl);
      if (resp.metadata?.duration_seconds) {
        setDuration(resp.metadata.duration_seconds);
      }
      setVideoProcessingState('VIDEO_READY');
      setStatusMessage(`Video '${file.name}' verified on server. Ready for processing.`);
    } catch (err: any) {
      setVideoProcessingState('UPLOAD_FAILED');
      setErrorMessage(err.message || 'Server rejected video file upload.');
      setStatusMessage('Upload failed. Please check file format and try again.');
    }
  }, [stopPolling, clearActiveJob]);

  // Start Processing Action
  const startProcessing = useCallback(async (options?: StartProcessingOptions) => {
    const targetVideoId = uploadedVideoId || activeJob?.filename;
    if (!targetVideoId) {
      setErrorMessage('No valid video staged. Please upload a video first.');
      return;
    }

    const mode = options?.analysisMode || analysisMode;
    const rot = options?.rotation !== undefined ? options.rotation : rotation;
    const fps = options?.sampleFps !== undefined ? options.sampleFps : sampleFps;
    const maxFr = options?.maxFrames !== undefined ? options.maxFrames : (mode === 'QUICK_PREVIEW' ? 30 : null);

    setVideoProcessingState('PROCESSING');
    setProcessingProgressPct(5);
    setStatusMessage('Initiating perception pipeline (YOLO11 + ByteTrack + Trajectory + Collision Risk)...');
    setErrorMessage(null);

    try {
      const resp = await startProcessingVideo({
        video_id: targetVideoId,
        analysis_mode: mode,
        rotation: rot,
        sample_fps: fps,
        max_frames: maxFr,
      });

      setActiveJobId(resp.job_id);
      setStatusMessage('Pipeline executing across video timeline...');
      setProcessingProgressPct(10);
      await refreshJobs();
      startJobPolling(resp.job_id);
    } catch (err: any) {
      setVideoProcessingState('PROCESSING_FAILED');
      setErrorMessage(err.message || 'Failed to dispatch video processing job.');
      setStatusMessage('Processing failed to start.');
    }
  }, [uploadedVideoId, activeJob, analysisMode, rotation, sampleFps, refreshJobs, startJobPolling]);

  // Retry Processing Action (reuses staged video!)
  const retryProcessing = useCallback(async () => {
    return startProcessing();
  }, [startProcessing]);

  // Cancel Processing Action
  const cancelProcessing = useCallback(async () => {
    if (!activeJobId) return;
    try {
      await cancelJob(activeJobId);
      stopPolling();
      setVideoProcessingState('CANCELLED');
      setStatusMessage('Processing cancelled by user.');
      await refreshJobs();
    } catch (err: any) {
      console.warn('Cancel error:', err);
    }
  }, [activeJobId, stopPolling, refreshJobs]);

  const deleteJobById = useCallback(async (jobId: string) => {
    try {
      setIsLoading(true);
      await deleteJob(jobId);
      if (activeJobId === jobId) {
        clearActiveJob();
      }
      await refreshJobs();
    } catch (err: any) {
      setError(`Failed to delete job: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  }, [activeJobId, clearActiveJob, refreshJobs]);

  const cleanStale = useCallback(async () => {
    try {
      setIsLoading(true);
      const res = await cleanStaleJobs();
      await refreshJobs();
      return res.cleaned_count;
    } catch (err: any) {
      setError(`Failed to clean stale jobs: ${err.message || err}`);
      return 0;
    } finally {
      setIsLoading(false);
    }
  }, [refreshJobs]);

  useEffect(() => {
    if (activeJobId && activeJobId !== prevJobIdRef.current) {
      prevJobIdRef.current = activeJobId;
      selectJob(activeJobId);
    }
  }, [activeJobId, selectJob]);

  // Handle video element play / pause / seek
  const playVideo = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.play().catch(() => {});
    }
    setIsPlaying(true);
  }, []);

  const pauseVideo = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause();
    }
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        playVideo();
      } else {
        pauseVideo();
      }
    } else {
      setIsPlaying((prev) => !prev);
    }
  }, [playVideo, pauseVideo]);

  const setPlaybackSpeed = useCallback((spd: number) => {
    setPlaybackSpeedState(spd);
    if (videoRef.current) {
      videoRef.current.playbackRate = spd;
    }
  }, []);

  const seekToTimestamp = useCallback((timeSec: number) => {
    const safeTime = Math.max(0, timeSec);
    setCurrentTime(safeTime);
    if (videoRef.current) {
      try {
        videoRef.current.currentTime = safeTime;
      } catch {
        // ignore
      }
    }
    if (results?.frames?.length) {
      const idx = findClosestFrameIndex(results.frames, safeTime);
      setCurrentFrameIndex(idx);
    }
  }, [results]);

  const seekToFrame = useCallback((frameIdx: number) => {
    if (!results?.frames?.length) {
      setCurrentFrameIndex(frameIdx);
      return;
    }
    const safeIdx = Math.max(0, Math.min(results.frames.length - 1, frameIdx));
    setCurrentFrameIndex(safeIdx);
    const targetTs = results.frames[safeIdx].timestamp_seconds ?? 0;
    setCurrentTime(targetTs);
    if (videoRef.current) {
      try {
        videoRef.current.currentTime = targetTs;
      } catch {
        // ignore
      }
    }
  }, [results]);

  const handleTimeUpdate = useCallback((timeSec: number) => {
    setCurrentTime(timeSec);
    if (results?.frames?.length) {
      const idx = findClosestFrameIndex(results.frames, timeSec);
      setCurrentFrameIndex(idx);
    }
  }, [results]);

  const handleDurationChange = useCallback((durSec: number) => {
    if (durSec && durSec > 0 && !isNaN(durSec)) {
      setDuration(durSec);
    }
  }, []);

  // Voice alert evaluation on frame change during playback
  useEffect(() => {
    if (!isPlaying || !currentFrame) return;

    const recState = currentFrame.safety_recommendation?.recommendation_state ||
                     currentFrame.safety_recommendation?.primary_action;
    const isCriticalRisk = currentFrame.collision_risk?.risk_state === 'CRITICAL';
    const isHighRisk = currentFrame.collision_risk?.risk_state === 'HIGH';
    const isDegraded = currentFrame.weather_intelligence?.is_degraded;
    const pedState = currentFrame.pedestrian_awareness?.primary_state;
    const pedVoice = currentFrame.pedestrian_awareness?.primary_voice_alert;

    if (pedVoice && pedState === 'POTENTIAL PEDESTRIAN COLLISION RISK') {
      triggerVoiceAlert(pedVoice, {
        category: 'PED_CRITICAL',
        priority: 'P1',
      });
    } else if (pedVoice && pedState === 'PEDESTRIAN CROSSING DETECTED') {
      triggerVoiceAlert(pedVoice, {
        category: 'PED_CROSSING',
        priority: 'P2',
      });
    } else if (pedVoice && pedState === 'PEDESTRIAN APPROACHING ROAD') {
      triggerVoiceAlert(pedVoice, {
        category: 'PED_APPROACH',
        priority: 'P3',
      });
    } else if (recState === 'STOP_RECOMMENDED' || currentFrame.safety_recommendation?.stop_recommendation) {
      triggerVoiceAlert('Warning: Stop recommended. Critical hazard detected ahead.', {
        category: 'STOP_REC',
        priority: 'P1',
      });
    } else if (recState === 'PREPARE_TO_STOP') {
      triggerVoiceAlert('Caution: Prepare to stop. Obstacle in projected trajectory.', {
        category: 'PREPARE_STOP',
        priority: 'P2',
      });
    } else if (isCriticalRisk) {
      triggerVoiceAlert('Collision alert: Critical trajectory intersection detected.', {
        category: 'CRITICAL_RISK',
        priority: 'P1',
      });
    } else if (isHighRisk) {
      triggerVoiceAlert('Caution: Elevated collision risk on roadway.', {
        category: 'HIGH_RISK',
        priority: 'P2',
      });
    } else if (isDegraded && (currentFrame.weather_intelligence?.visibility_score ?? 100) < 50) {
      triggerVoiceAlert('Optical visibility low. Maintain safe headway.', {
        category: 'VIS_LOW',
        priority: 'P3',
      });
    }
  }, [currentFrame, isPlaying, triggerVoiceAlert]);

  return (
    <JobContext.Provider
      value={{
        jobs,
        activeJobId,
        activeJob,
        results,
        currentFrameIndex,
        currentFrame,
        videoProcessingState,
        uploadProgressPct,
        processingProgressPct,
        statusMessage,
        errorMessage,
        uploadedFile,
        analysisMode,
        rotation,
        sampleFps,
        setAnalysisMode,
        setRotation,
        setSampleFps,
        uploadFile,
        startProcessing,
        cancelProcessing,
        retryProcessing,
        resetForNewVideo,
        isPlaying,
        playbackSpeed,
        currentTime,
        duration,
        playbackMode,
        isVoiceMuted,
        isLoading,
        isBackendOnline,
        error,
        videoRef,
        selectedTrackId,
        stagedVideoSrc,
        uploadedVideoId,
        uploadedVideoMetadata,
        setSelectedTrackId,
        setStagedVideo,
        clearStagedVideo,
        refreshJobs,
        selectJob,
        clearActiveJob,
        deleteJobById,
        cleanStale,
        setCurrentFrameIndex,
        setIsPlaying,
        setPlaybackSpeed,
        setPlaybackMode,
        toggleVoiceMute,
        playVideo,
        pauseVideo,
        togglePlay,
        seekToTimestamp,
        seekToFrame,
        handleTimeUpdate,
        handleDurationChange,
      }}
    >
      {children}
    </JobContext.Provider>
  );
};

export const useJob = () => {
  const context = useContext(JobContext);
  if (!context) {
    throw new Error('useJob must be used within a JobProvider');
  }
  return context;
};
