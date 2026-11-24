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
} from '../types/api';
import { listJobs, getJobResults, fetchHealth, deleteJob, cleanStaleJobs } from '../services/api';
import { useVoiceAlerts } from '../hooks/useVoiceAlerts';

interface JobContextType {
  jobs: JobStatusResponse[];
  activeJobId: string | null;
  activeJob: JobStatusResponse | null;
  results: VideoPipelineRunResult | null;
  currentFrameIndex: number;
  currentFrame: FrameProcessingDetail | null;
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

  const { isVoiceMuted, toggleVoiceMute, triggerVoiceAlert } = useVoiceAlerts();

  const activeJob = jobs.find((j) => j.job_id === activeJobId) || null;
  const currentFrame = results?.frames?.[currentFrameIndex] || null;

  const setStagedVideo = useCallback((src: string | null, videoId?: string | null, metadata?: VideoUploadResponse | null) => {
    setStagedVideoSrc(src);
    if (videoId !== undefined) setUploadedVideoId(videoId);
    if (metadata !== undefined) setUploadedVideoMetadata(metadata);
  }, []);

  const clearStagedVideo = useCallback(() => {
    setStagedVideoSrc(null);
    setUploadedVideoId(null);
    setUploadedVideoMetadata(null);
  }, []);

  // Check health periodically
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

  // Clear active job back to clean upload state
  const clearActiveJob = useCallback(() => {
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
    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      } catch {
        // ignore
      }
    }
  }, []);

  // Fetch job list without automatically forcing a historical job on initial boot
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

  const selectJob = useCallback(async (jobId: string) => {
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
      const data = await getJobResults(jobId);
      if (!data || !data.frames || data.frames.length === 0 || (data as any).results === null) {
        setResults(null);
        if ((data as any).status === 'FAILED') {
          setError((data as any).message || 'Video pipeline execution failed.');
        }
      } else {
        setResults(data);
        if (data.timeline_telemetry?.source_video_duration_sec) {
          setDuration(data.timeline_telemetry.source_video_duration_sec);
        } else if (data.video_metadata?.duration_seconds) {
          setDuration(data.video_metadata.duration_seconds);
        } else if (data.frames.length > 0) {
          const lastTs = data.frames[data.frames.length - 1].timestamp_seconds || 0;
          setDuration(lastTs);
        }
      }
    } catch (err: any) {
      setResults(null);
      setError(`Failed to load job telemetry: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  }, [clearActiveJob]);

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
