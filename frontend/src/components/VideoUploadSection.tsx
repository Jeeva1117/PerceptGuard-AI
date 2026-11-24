import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  UploadCloud,
  FileVideo,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Sparkles,
  ArrowRight,
  Film,
  OctagonAlert,
  RotateCw,
  Ban,
  RefreshCw,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  uploadVideoFile,
  startProcessingVideo,
  getJobStatus,
  cancelJob,
  getSourceVideoUrl,
  getVideoStreamUrl,
} from '../services/api';
import { useJob } from '../context/JobContext';
import type { JobStatusResponse, AnalysisMode, VideoUploadResponse } from '../types/api';

export type UploadStateMachine =
  | 'IDLE'
  | 'FILE_SELECTED'
  | 'VALIDATING'
  | 'UPLOADING'
  | 'UPLOAD_COMPLETED'
  | 'UPLOAD_FAILED';

export type JobExecutionState =
  | 'IDLE'
  | 'QUEUED'
  | 'PROCESSING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED';

const ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB max for CPU safety

export const VideoUploadSection: React.FC = () => {
  const {
    refreshJobs,
    selectJob,
    clearActiveJob,
    activeJob,
    activeJobId,
    setStagedVideo,
    clearStagedVideo,
    stagedVideoSrc,
    uploadedVideoId: contextVideoId,
  } = useJob();

  // Local file & upload state machine
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [uploadState, setUploadState] = useState<UploadStateMachine>('IDLE');
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(contextVideoId || null);
  const [serverMetadata, setServerMetadata] = useState<VideoUploadResponse | null>(null);

  // Video properties
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [rotation, setRotation] = useState<0 | 90 | 180 | 270>(0);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('FULL_BALANCED');
  const [sampleFps, setSampleFps] = useState<number>(2.0);

  // Execution state
  const [jobState, setJobState] = useState<JobExecutionState>('IDLE');
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [completedJob, setCompletedJob] = useState<JobStatusResponse | null>(null);

  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pollTimerRef = useRef<number | null>(null);

  // Derive active stream for the preview window
  const hasLocalPreview = Boolean(selectedFile && localPreviewUrl);
  const hasServerStream = Boolean(uploadedVideoId);
  const hasHistoricalStream = Boolean(!selectedFile && activeJob?.has_source_video);

  const effectiveVideoSrc = hasServerStream
    ? getVideoStreamUrl(uploadedVideoId)
    : hasLocalPreview
    ? localPreviewUrl
    : hasHistoricalStream
    ? getSourceVideoUrl(activeJob?.job_id)
    : stagedVideoSrc;

  // Cleanup local object URLs
  const cleanupPreviewUrl = useCallback(() => {
    if (localPreviewUrl) {
      URL.revokeObjectURL(localPreviewUrl);
      setLocalPreviewUrl(null);
    }
  }, [localPreviewUrl]);

  useEffect(() => {
    return () => {
      cleanupPreviewUrl();
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, [cleanupPreviewUrl]);

  // Sync with active job state when browsing historical records
  useEffect(() => {
    if (activeJob && !selectedFile) {
      if (activeJob.status === 'COMPLETED') {
        setJobState('COMPLETED');
        setCompletedJob(activeJob);
        setStatusMessage(`Restored completed session: ${activeJob.filename}`);
      } else if (activeJob.status === 'FAILED') {
        setJobState('FAILED');
        setErrorMessage(activeJob.error_message || activeJob.message);
      } else if (activeJob.status === 'PROCESSING' || activeJob.status === 'QUEUED') {
        setJobState('PROCESSING');
        setActiveSessionId(activeJob.job_id);
        setStatusMessage(activeJob.message || `Stage: ${activeJob.stage}`);
        startStatusPolling(activeJob.job_id);
      }
    }
  }, [activeJob, selectedFile]);

  // Helpers
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  // Perform upload to FastAPI backend
  const executeUploadToServer = async (file: File) => {
    setUploadState('UPLOADING');
    setStatusMessage(`Uploading '${file.name}' to server storage...`);
    setProgressPct(20);
    setErrorMessage(null);

    try {
      // Simulate stepped progress during chunk streaming
      const progressInterval = setInterval(() => {
        setProgressPct((prev) => (prev < 85 ? prev + 15 : prev));
      }, 300);

      const resp = await uploadVideoFile(file);
      clearInterval(progressInterval);

      setProgressPct(100);
      setUploadState('UPLOAD_COMPLETED');
      setUploadedVideoId(resp.video_id);
      setServerMetadata(resp);
      setStatusMessage('Video successfully uploaded and verified on server. Ready for processing.');

      // Update global context so the persistent video player immediately renders the uploaded stream
      const streamUrl = getVideoStreamUrl(resp.video_id);
      setStagedVideo(streamUrl, resp.video_id, resp);

      if (resp.metadata?.duration_seconds) {
        setVideoDuration(resp.metadata.duration_seconds);
      }
      if (resp.metadata?.width && resp.metadata?.height) {
        setVideoDimensions({ width: resp.metadata.width, height: resp.metadata.height });
      }
    } catch (err: any) {
      setUploadState('UPLOAD_FAILED');
      setUploadedVideoId(null);
      setErrorMessage(err.message || 'Server rejected uploaded video file.');
      setStatusMessage('Upload failed. Please check file format or select a valid video.');
    }
  };

  // Process selected file
  const processSelectedFile = (file: File) => {
    // 1. Session Isolation: Clean previous job results and alerts
    if (activeJobId) {
      clearActiveJob();
    }
    clearStagedVideo();
    setErrorMessage(null);
    setCompletedJob(null);
    setJobState('IDLE');
    setUploadedVideoId(null);
    setServerMetadata(null);

    // 2. Validate extension
    setUploadState('VALIDATING');
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setUploadState('UPLOAD_FAILED');
      setErrorMessage(
        `Unsupported video format '${ext}'. Supported extensions: ${ALLOWED_EXTENSIONS.join(', ')}`
      );
      return;
    }

    // 3. Validate size
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setUploadState('UPLOAD_FAILED');
      setErrorMessage(
        `Video size (${formatFileSize(file.size)}) exceeds maximum allowed upload limit of 100 MB.`
      );
      return;
    }

    if (file.size === 0) {
      setUploadState('UPLOAD_FAILED');
      setErrorMessage('Selected video file is empty (0 bytes).');
      return;
    }

    // 4. File selected valid - create local preview URL and start upload to backend
    cleanupPreviewUrl();
    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setLocalPreviewUrl(objectUrl);
    setUploadState('FILE_SELECTED');

    // Stage video preview in global context right away
    setStagedVideo(objectUrl, null, null);

    // 5. Automatically initiate server upload
    executeUploadToServer(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleClearFile = () => {
    cleanupPreviewUrl();
    setSelectedFile(null);
    setUploadedVideoId(null);
    setServerMetadata(null);
    setVideoDuration(null);
    setVideoDimensions(null);
    setRotation(0);
    setUploadState('IDLE');
    setJobState('IDLE');
    setStatusMessage('');
    setErrorMessage(null);
    setCompletedJob(null);
    clearStagedVideo();
    clearActiveJob();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleVideoLoadedMetadata = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
      setVideoDimensions({
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight,
      });
    }
  };

  // Poll backend for real background processing status
  const startStatusPolling = (jobId: string) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    pollTimerRef.current = window.setInterval(async () => {
      try {
        const job = await getJobStatus(jobId);
        setStatusMessage(job.message || `Processing Stage: ${job.stage}`);
        const pct = job.progress_pct ?? job.progress_percentage ?? (job.status === 'COMPLETED' ? 100 : 35);
        setProgressPct(pct);

        if (job.status === 'COMPLETED') {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setJobState('COMPLETED');
          setStatusMessage('Pipeline execution completed successfully!');
          setCompletedJob(job);
          await refreshJobs();
          await selectJob(jobId);
        } else if (job.status === 'FAILED') {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setJobState('FAILED');
          setErrorMessage(job.error_message || 'Video processing encountered an error on the backend.');
          await refreshJobs();
          await selectJob(jobId);
        } else if (job.status === 'CANCELLED') {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setJobState('CANCELLED');
          setStatusMessage('Processing job was cancelled.');
          await refreshJobs();
        }
      } catch (err: any) {
        console.warn('Status poll warning:', err);
      }
    }, 1200);
  };

  // Start Processing Handler
  const handleStartProcessing = async () => {
    if (!uploadedVideoId) {
      if (selectedFile && uploadState !== 'UPLOADING') {
        executeUploadToServer(selectedFile);
      }
      return;
    }

    setJobState('PROCESSING');
    setStatusMessage('Initiating perception pipeline (YOLOv11 + ByteTrack + LSTM)...');
    setErrorMessage(null);
    setCompletedJob(null);
    setProgressPct(5);

    try {
      const resp = await startProcessingVideo({
        video_id: uploadedVideoId,
        analysis_mode: analysisMode,
        rotation: rotation,
        sample_fps: sampleFps,
        max_frames: analysisMode === 'QUICK_PREVIEW' ? 30 : null,
      });

      setActiveSessionId(resp.job_id);
      setStatusMessage('Pipeline executing across video timeline...');
      setProgressPct(10);
      await refreshJobs();
      await selectJob(resp.job_id);
      startStatusPolling(resp.job_id);
    } catch (err: any) {
      setJobState('FAILED');
      setErrorMessage(err.message || 'Failed to dispatch processing job.');
    }
  };

  // Cancel Running Job Handler
  const handleCancelJob = async () => {
    if (!activeSessionId) return;
    try {
      await cancelJob(activeSessionId);
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      setJobState('CANCELLED');
      setStatusMessage('Job cancellation requested.');
      await refreshJobs();
    } catch (err: any) {
      console.warn('Cancel error:', err);
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400">
            <UploadCloud className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Upload Road Video</h3>
            <p className="text-xs text-slate-400">
              Ingest raw traffic video for optical visibility analysis, YOLO detection, ByteTrack & collision forecasting
            </p>
          </div>
        </div>

        {/* Upload & Job State Badges */}
        <div className="flex items-center gap-2">
          {/* Upload Status */}
          <span
            className={`px-3 py-1 rounded-full text-xs font-mono font-bold border flex items-center gap-1.5 ${
              uploadState === 'UPLOAD_COMPLETED'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : uploadState === 'UPLOADING' || uploadState === 'VALIDATING'
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 animate-pulse'
                : uploadState === 'UPLOAD_FAILED'
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                : uploadState === 'FILE_SELECTED'
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            {uploadState === 'UPLOADING' || uploadState === 'VALIDATING' ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : uploadState === 'UPLOAD_COMPLETED' ? (
              <CheckCircle2 className="w-3.5 h-3.5" />
            ) : uploadState === 'UPLOAD_FAILED' ? (
              <AlertCircle className="w-3.5 h-3.5" />
            ) : null}
            UPLOAD: {uploadState}
          </span>

          {/* Job Execution Status */}
          {jobState !== 'IDLE' && (
            <span
              className={`px-3 py-1 rounded-full text-xs font-mono font-bold border flex items-center gap-1.5 ${
                jobState === 'COMPLETED'
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : jobState === 'PROCESSING' || jobState === 'QUEUED'
                  ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 animate-pulse'
                  : jobState === 'FAILED'
                  ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                  : 'bg-slate-800 border-slate-700 text-slate-400'
              }`}
            >
              {jobState === 'PROCESSING' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
              JOB: {jobState}
            </span>
          )}
        </div>
      </div>

      {/* Main Grid: Upload Dropzone & Video Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Dropzone, Metadata Card & Processing Action (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv,.webm"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Drag & Drop Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center select-none ${
              isDragOver
                ? 'border-cyan-400 bg-cyan-950/20 shadow-lg shadow-cyan-950/30'
                : selectedFile
                ? 'border-slate-700 bg-slate-950/40 hover:border-slate-600'
                : 'border-slate-700/80 hover:border-cyan-500/50 hover:bg-slate-950/60'
            }`}
          >
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-3">
              <FileVideo className="w-6 h-6" />
            </div>

            <h4 className="text-sm font-semibold text-white mb-1">
              {selectedFile ? 'Replace Selected Road Video' : 'Drag & Drop Video or Browse Files'}
            </h4>
            <p className="text-xs text-slate-400 max-w-sm mb-3">
              Supports standard road traffic recordings (.mp4, .mov, .avi, .mkv, .webm) up to 100 MB.
            </p>

            <button
              type="button"
              className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md shadow-cyan-950/40 transition-colors flex items-center gap-2"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              <UploadCloud className="w-4 h-4" />
              <span>Browse Files</span>
            </button>
          </div>

          {/* Selected & Uploaded File Metadata Card */}
          {selectedFile && (
            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 shrink-0">
                  <Film className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-white truncate font-mono">
                    {selectedFile.name}
                  </div>
                  <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                    <span>{formatFileSize(selectedFile.size)}</span>
                    <span>•</span>
                    <span className="uppercase text-cyan-300 font-mono">
                      {selectedFile.name.split('.').pop()}
                    </span>
                    {videoDimensions && (
                      <>
                        <span>•</span>
                        <span>{videoDimensions.width}x{videoDimensions.height}</span>
                      </>
                    )}
                    {videoDuration !== null && (
                      <>
                        <span>•</span>
                        <span>{formatDuration(videoDuration)}</span>
                      </>
                    )}
                    {uploadedVideoId && (
                      <>
                        <span>•</span>
                        <span className="text-emerald-400 font-mono text-[10px]">
                          Server Verified {serverMetadata?.metadata?.fps ? `(${serverMetadata.metadata.fps} FPS)` : `(${uploadedVideoId})`}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  disabled={uploadState === 'UPLOADING' || jobState === 'PROCESSING'}
                  title="Replace with another video"
                  className="px-2.5 py-1 rounded-lg text-xs font-mono text-cyan-400 hover:text-cyan-300 bg-slate-900 border border-slate-700 transition-colors disabled:opacity-40"
                >
                  Replace
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClearFile();
                  }}
                  disabled={uploadState === 'UPLOADING' || jobState === 'PROCESSING'}
                  title="Remove selected video"
                  className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-900 transition-colors disabled:opacity-40"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Active Historical Stream Card */}
          {!selectedFile && activeJob && (
            <div className="bg-slate-950 rounded-xl p-4 border border-cyan-500/30 flex items-center justify-between gap-3 animate-in fade-in duration-150">
              <div className="flex items-center gap-3 min-w-0">
                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 shrink-0">
                  <Film className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-white truncate font-mono">
                    Stream: {activeJob.filename.startsWith('up_') ? activeJob.filename.slice(12) : activeJob.filename}
                  </div>
                  <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                    <span className="text-cyan-300 font-mono">Status: {activeJob.status}</span>
                    <span>•</span>
                    <span>{activeJob.frames_processed} frames analyzed</span>
                    {activeJob.has_source_video && (
                      <>
                        <span>•</span>
                        <span className="text-emerald-400 font-mono">Video Stream Ready</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() => {
                  handleClearFile();
                  clearActiveJob();
                }}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 transition-colors shrink-0"
              >
                + New Upload
              </button>
            </div>
          )}

          {/* Execution Settings & Start Processing Bar */}
          <div className="bg-slate-950/70 rounded-xl p-4 border border-slate-800 flex flex-col gap-3">
            <div className="flex flex-wrap items-center justify-between gap-4 w-full">
              <div className="flex flex-col sm:flex-row sm:items-center gap-2 flex-1 min-w-[280px]">
                <span className="text-xs font-semibold text-slate-400 shrink-0">Analysis Mode:</span>
                <select
                  value={analysisMode}
                  onChange={(e) => {
                    const m = e.target.value as AnalysisMode;
                    setAnalysisMode(m);
                    if (m === 'FULL_BALANCED') setSampleFps(2.0);
                    else if (m === 'FULL_DETAILED') setSampleFps(5.0);
                  }}
                  disabled={uploadState === 'UPLOADING' || jobState === 'PROCESSING'}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono w-full sm:w-auto"
                >
                  <option value="FULL_BALANCED">Full Video — CPU Balanced (2 FPS, Full Duration Coverage, Recommended)</option>
                  <option value="FULL_DETAILED">Full Video — Higher Detail (5 FPS, Full Duration Coverage)</option>
                  <option value="QUICK_PREVIEW">Quick Preview (First 30 Frames / Partial Inspection)</option>
                </select>
              </div>

              <div className="flex items-center gap-3">
                {/* Orientation Selector */}
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-400 shrink-0">Orientation:</span>
                  <button
                    type="button"
                    onClick={() => setRotation((prev) => ((prev + 90) % 360) as 0 | 90 | 180 | 270)}
                    disabled={uploadState === 'UPLOADING' || jobState === 'PROCESSING'}
                    className="bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-mono flex items-center gap-1.5 transition-colors disabled:opacity-50"
                    title="Cycle rotation clockwise: 0°, 90°, 180°, 270°"
                  >
                    <RotateCw className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{rotation === 0 ? '0° (Upright)' : `${rotation}° CW`}</span>
                  </button>
                </div>

                {/* Start Processing / Cancel Action Button */}
                {jobState === 'PROCESSING' ? (
                  <button
                    type="button"
                    onClick={handleCancelJob}
                    className="px-4 py-2 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-950/50 transition-all"
                  >
                    <Ban className="w-4 h-4" />
                    <span>Cancel Processing</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleStartProcessing}
                    disabled={
                      uploadState === 'UPLOADING' ||
                      uploadState === 'VALIDATING' ||
                      (uploadState !== 'UPLOAD_COMPLETED' && !uploadedVideoId && !selectedFile)
                    }
                    className={`px-5 py-2 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg transition-all ${
                      uploadState === 'UPLOADING' || uploadState === 'VALIDATING'
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/60'
                        : uploadState === 'UPLOAD_COMPLETED' || uploadedVideoId
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-950/50'
                        : selectedFile
                        ? 'bg-amber-600 hover:bg-amber-500 text-white'
                        : 'bg-slate-800 hover:bg-slate-700 text-slate-500 border border-slate-700 cursor-not-allowed'
                    }`}
                  >
                    {uploadState === 'UPLOADING' ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                        <span>Uploading Video ({progressPct}%)...</span>
                      </>
                    ) : uploadState === 'VALIDATING' ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                        <span>Validating Video...</span>
                      </>
                    ) : uploadState === 'UPLOAD_COMPLETED' || uploadedVideoId ? (
                      <>
                        <Sparkles className="w-4 h-4" />
                        <span>Start Processing</span>
                      </>
                    ) : selectedFile ? (
                      <>
                        <UploadCloud className="w-4 h-4" />
                        <span>Upload Video</span>
                      </>
                    ) : (
                      <>
                        <UploadCloud className="w-4 h-4" />
                        <span>Start Processing (Upload Video First)</span>
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>

            {/* Dynamic Timeline Coverage Pill */}
            <div className="flex items-center gap-2 text-[11px] font-mono px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400">
              <span className="text-cyan-400 font-semibold">Timeline Coverage:</span>
              {analysisMode === 'FULL_BALANCED' ? (
                <span>
                  100% Full Video ({videoDuration ? `00:00 - ${formatDuration(videoDuration)}` : 'Full Duration'}) • ~{videoDuration ? Math.max(1, Math.round(videoDuration * 2)) : '2 frames/s'} frames @ 2 FPS [CPU Optimized]
                </span>
              ) : analysisMode === 'FULL_DETAILED' ? (
                <span>
                  100% Full Video ({videoDuration ? `00:00 - ${formatDuration(videoDuration)}` : 'Full Duration'}) • ~{videoDuration ? Math.max(1, Math.round(videoDuration * 5)) : '5 frames/s'} frames @ 5 FPS [High Fidelity]
                </span>
              ) : (
                <span className="text-amber-400">
                  Partial Preview: First 30 frames (~{Math.min(videoDuration || 1, 1).toFixed(1)}s of video)
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Video Preview Player (5 cols) */}
        <div className="lg:col-span-5 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase tracking-wider">
            <span className="flex items-center gap-1.5">
              <Film className="w-4 h-4 text-cyan-400" />
              Source Video Preview
            </span>
            <div className="flex items-center gap-2">
              {rotation !== 0 && (
                <span className="text-[10px] text-cyan-300 font-mono bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/60 flex items-center gap-1">
                  <RotateCw className="w-3 h-3 text-cyan-400" />
                  +{rotation}°
                </span>
              )}
              {uploadState === 'UPLOAD_COMPLETED' && (
                <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Verified on Server
                </span>
              )}
              {hasLocalPreview && uploadState !== 'UPLOAD_COMPLETED' && (
                <span className="text-[10px] text-amber-400 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Local File Staged
                </span>
              )}
            </div>
          </div>

          <div className="w-full aspect-video bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-inner flex items-center justify-center relative">
            {effectiveVideoSrc ? (
              <div className="w-full h-full flex items-center justify-center overflow-hidden">
                <video
                  key={effectiveVideoSrc}
                  ref={videoRef}
                  src={effectiveVideoSrc}
                  controls
                  playsInline
                  onLoadedMetadata={handleVideoLoadedMetadata}
                  className="max-w-full max-h-full object-contain transition-transform duration-200"
                  style={{
                    transform: rotation !== 0 ? `rotate(${rotation}deg)` : undefined,
                  }}
                />
              </div>
            ) : (
              <div className="p-6 text-center text-slate-600 flex flex-col items-center">
                <Film className="w-10 h-10 mb-2 stroke-[1.5]" />
                <span className="text-xs font-semibold text-slate-500">
                  {activeJob ? 'Source Video Stored Offline' : 'No Video Loaded for Preview'}
                </span>
                <span className="text-[11px] text-slate-600 mt-1 max-w-xs">
                  {activeJob
                    ? 'Individual extracted frames and perception telemetry for this session are available.'
                    : 'Select a video file on the left to verify footage before dispatching to the inference engine.'}
                </span>
              </div>
            )}
          </div>
          {effectiveVideoSrc && rotation !== 0 && (
            <p className="text-[11px] text-cyan-400/90 font-mono flex items-center gap-1.5 px-1">
              <RotateCw className="w-3 h-3 shrink-0" />
              <span>Rotation active (+{rotation}°). Frames will be rotated upright before inference.</span>
            </p>
          )}
        </div>
      </div>

      {/* Progress & Live Status Feedback Strip */}
      {(uploadState === 'UPLOADING' ||
        uploadState === 'VALIDATING' ||
        jobState === 'PROCESSING' ||
        jobState === 'COMPLETED' ||
        jobState === 'FAILED' ||
        jobState === 'CANCELLED' ||
        uploadState === 'UPLOAD_FAILED') && (
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-2">
              {uploadState === 'UPLOADING' || jobState === 'PROCESSING' ? (
                <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              ) : jobState === 'COMPLETED' || uploadState === 'UPLOAD_COMPLETED' ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertCircle className="w-4 h-4 text-rose-400" />
              )}
              {statusMessage}
            </span>

            {activeSessionId && (
              <span className="text-[11px] font-mono text-cyan-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                Session: {activeSessionId}
              </span>
            )}
          </div>

          {/* Progress bar */}
          {(uploadState === 'UPLOADING' || jobState === 'PROCESSING') && (
            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
                style={{ width: `${Math.max(10, progressPct)}%` }}
              />
            </div>
          )}

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-300 font-mono flex items-center justify-between gap-3">
              <div>
                <strong>Error:</strong> {errorMessage}
              </div>
              <button
                type="button"
                onClick={() => {
                  if (selectedFile) executeUploadToServer(selectedFile);
                }}
                className="px-2.5 py-1 rounded bg-rose-950 border border-rose-700 text-rose-200 text-xs font-semibold flex items-center gap-1.5 hover:bg-rose-900 transition-colors shrink-0"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Retry</span>
              </button>
            </div>
          )}

          {/* Processing Completed Summary Card */}
          {jobState === 'COMPLETED' && completedJob && (
            <div className="pt-2 border-t border-slate-800/80 space-y-3 animate-in fade-in duration-200">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Frames Processed</span>
                  <strong className="text-sm font-mono text-cyan-400">{completedJob.frames_processed}</strong>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Unique Tracks</span>
                  <strong className="text-sm font-mono text-emerald-400">{completedJob.unique_tracks ?? 0}</strong>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Alerts Triggered</span>
                  <strong className="text-sm font-mono text-amber-400">{completedJob.alerts_count ?? 0}</strong>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Telemetry Stream</span>
                  <span className="text-xs font-mono text-slate-300 block mt-0.5 truncate">
                    {completedJob.filename}
                  </span>
                </div>
              </div>

              {/* Hand-off Navigation to Other Four Features */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                <span className="text-xs text-slate-400 font-medium">
                  Analysis stream loaded into current workspace. Inspect downstream modules:
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  <Link
                    to="/object-detection"
                    className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-cyan-300 border border-slate-700 transition-colors"
                  >
                    <span>Object Detection</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                  <Link
                    to="/trajectory-prediction"
                    className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-cyan-300 border border-slate-700 transition-colors"
                  >
                    <span>Trajectories</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                  <Link
                    to="/collision-risk"
                    className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-cyan-300 border border-slate-700 transition-colors"
                  >
                    <span>Collision Risk</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                  <Link
                    to="/safety-alerts"
                    className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-xs font-semibold text-amber-300 border border-amber-500/30 transition-colors"
                  >
                    <OctagonAlert className="w-3 h-3" />
                    <span>Safety Alerts</span>
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
