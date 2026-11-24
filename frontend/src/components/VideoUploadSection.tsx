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
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { submitVideoJob, getJobStatus, getSourceVideoUrl } from '../services/api';
import { useJob } from '../context/JobContext';
import type { JobStatusResponse, AnalysisMode } from '../types/api';

export type UploadProcessingState =
  | 'IDLE'
  | 'FILE_SELECTED'
  | 'UPLOADING'
  | 'PROCESSING'
  | 'COMPLETED'
  | 'FAILED';

const ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'];
const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB

export const VideoUploadSection: React.FC = () => {
  const { refreshJobs, selectJob, clearActiveJob, activeJob, activeJobId } = useJob();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [rotation, setRotation] = useState<0 | 90 | 180 | 270>(0);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('FULL_BALANCED');
  const [sampleFps, setSampleFps] = useState<number>(2.0);

  const [processingState, setProcessingState] = useState<UploadProcessingState>('IDLE');
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [completedJob, setCompletedJob] = useState<JobStatusResponse | null>(null);

  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pollTimerRef = useRef<number | null>(null);

  // Derive dual-mode video source: local staged upload OR historical remote stream
  const hasLocalVideo = Boolean(selectedFile && previewUrl);
  const hasRemoteVideo = Boolean(!selectedFile && activeJob?.has_source_video);
  const effectiveVideoSrc = hasLocalVideo
    ? previewUrl
    : hasRemoteVideo
    ? getSourceVideoUrl(activeJob?.job_id)
    : null;

  // Clean up object URLs to prevent browser memory leaks
  const cleanupPreviewUrl = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      cleanupPreviewUrl();
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, [cleanupPreviewUrl]);

  // Format file size helper
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Format duration helper (mm:ss)
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  // Validate and apply selected file
  const processSelectedFile = (file: File) => {
    // Clear historical job results from the active view immediately to isolate the new session
    if (activeJobId) {
      clearActiveJob();
    }
    setErrorMessage(null);
    setCompletedJob(null);

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMessage(
        `Invalid video format '${ext}'. Allowed extensions: ${ALLOWED_EXTENSIONS.join(', ')}`
      );
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage(
        `Video size (${formatFileSize(file.size)}) exceeds the maximum CPU upload limit of 100 MB.`
      );
      return;
    }

    cleanupPreviewUrl();
    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);
    setProcessingState('FILE_SELECTED');
    setStatusMessage('Road video selected and ready for upload.');
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
    setVideoDuration(null);
    setVideoDimensions(null);
    setRotation(0);
    setProcessingState('IDLE');
    setStatusMessage('');
    setErrorMessage(null);
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

  // Poll backend for real processing status
  const startStatusPolling = (jobId: string) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    pollTimerRef.current = window.setInterval(async () => {
      try {
        const job = await getJobStatus(jobId);
        setStatusMessage(job.message || `Processing Stage: ${job.stage}`);
        setProgressPct(job.progress_pct || (job.status === 'COMPLETED' ? 100 : 45));

        if (job.status === 'COMPLETED') {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setProcessingState('COMPLETED');
          setStatusMessage('Pipeline execution completed successfully!');
          setCompletedJob(job);
          await refreshJobs();
          await selectJob(jobId);
        } else if (job.status === 'FAILED') {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setProcessingState('FAILED');
          setErrorMessage(job.error_message || 'Video processing encountered an error on the backend.');
          await refreshJobs();
          await selectJob(jobId);
        }
      } catch (err: any) {
        console.warn('Status poll warning:', err);
      }
    }, 1200);
  };

  // Submit file to backend
  const handleStartProcessing = async () => {
    if (!selectedFile) return;

    setProcessingState('UPLOADING');
    setStatusMessage('Streaming video file to FastAPI backend...');
    setErrorMessage(null);
    setCompletedJob(null);
    setProgressPct(10);

    try {
      const resp = await submitVideoJob(
        selectedFile,
        analysisMode === 'QUICK_PREVIEW' ? 30 : undefined,
        rotation,
        analysisMode,
        sampleFps
      );
      setActiveSessionId(resp.job_id);
      setProcessingState('PROCESSING');
      setStatusMessage('Video received. Dispatched to YOLO + ByteTrack + Weather Pipeline...');
      setProgressPct(25);
      await refreshJobs();
      await selectJob(resp.job_id);
      startStatusPolling(resp.job_id);
    } catch (err: any) {
      setProcessingState('FAILED');
      setErrorMessage(err.message || 'Failed to submit video for processing.');
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400">
            <UploadCloud className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Upload Road Video</h3>
            <p className="text-xs text-slate-400">
              Ingest raw road traffic recordings to run optical visibility scoring, YOLO detection & multi-agent forecasting
            </p>
          </div>
        </div>

        {/* State Indicator Badge */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Status:</span>
          <span
            className={`px-3 py-1 rounded-full text-xs font-mono font-bold border flex items-center gap-1.5 ${
              processingState === 'COMPLETED'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : processingState === 'PROCESSING' || processingState === 'UPLOADING'
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 animate-pulse'
                : processingState === 'FAILED'
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                : processingState === 'FILE_SELECTED'
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            {processingState === 'PROCESSING' || processingState === 'UPLOADING' ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : processingState === 'COMPLETED' ? (
              <CheckCircle2 className="w-3.5 h-3.5" />
            ) : processingState === 'FAILED' ? (
              <AlertCircle className="w-3.5 h-3.5" />
            ) : null}
            {processingState}
          </span>
        </div>
      </div>

      {/* Main Grid: Upload Dropzone & Video Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: File Dropzone & Configuration (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Hidden native file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Interactive Drag & Drop Area */}
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
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-3 group-hover:scale-105 transition-transform">
              <FileVideo className="w-6 h-6" />
            </div>

            <h4 className="text-sm font-semibold text-white mb-1">
              {selectedFile ? 'Change Selected Road Video' : 'Click to Browse or Drag & Drop Video'}
            </h4>
            <p className="text-xs text-slate-400 max-w-sm mb-3">
              Supports standard dashcam and traffic recordings (.mp4, .avi, .mov, .mkv) up to 100 MB.
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
              <span>Choose Video File</span>
            </button>
          </div>

          {/* Selected File Metadata Card */}
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
                        <span>
                          {videoDimensions.width}x{videoDimensions.height}
                        </span>
                      </>
                    )}
                    {videoDuration !== null && (
                      <>
                        <span>•</span>
                        <span>{formatDuration(videoDuration)}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleClearFile();
                }}
                disabled={processingState === 'UPLOADING' || processingState === 'PROCESSING'}
                title="Remove selected video"
                className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-900 transition-colors disabled:opacity-40"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Active Remote Stream Card when viewing historical completed job */}
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
                title="Deselect historical stream and return to clean upload mode"
              >
                + New Upload
              </button>
            </div>
          )}

          {/* Execution Settings & Start Button */}
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
                  disabled={processingState === 'UPLOADING' || processingState === 'PROCESSING'}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono w-full sm:w-auto"
                >
                  <option value="FULL_BALANCED">Full Video — CPU Balanced (2 FPS, Full Duration Coverage, Recommended)</option>
                  <option value="FULL_DETAILED">Full Video — Higher Detail (5 FPS, Full Duration Coverage)</option>
                  <option value="QUICK_PREVIEW">Quick Preview (First 30 Frames / Partial Inspection)</option>
                </select>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-400 shrink-0">Orientation:</span>
                  <button
                    type="button"
                    onClick={() => setRotation((prev) => ((prev + 90) % 360) as 0 | 90 | 180 | 270)}
                    disabled={processingState === 'UPLOADING' || processingState === 'PROCESSING'}
                    className="bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-mono flex items-center gap-1.5 transition-colors disabled:opacity-50"
                    title="Cycle rotation: 0°, 90°, 180°, 270° clockwise"
                  >
                    <RotateCw className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{rotation === 0 ? '0° (Upright)' : `${rotation}° CW`}</span>
                  </button>
                </div>

                <button
                  onClick={() => {
                    if (selectedFile) {
                      handleStartProcessing();
                    } else {
                      fileInputRef.current?.click();
                    }
                  }}
                  disabled={processingState === 'UPLOADING' || processingState === 'PROCESSING'}
                  className={`px-5 py-2 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg transition-all ${
                    processingState === 'UPLOADING' || processingState === 'PROCESSING'
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/60'
                      : selectedFile
                      ? 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-950/50'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
                  }`}
                >
                  {processingState === 'UPLOADING' || processingState === 'PROCESSING' ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>{processingState === 'UPLOADING' ? 'Uploading...' : 'Analyzing Timeline...'}</span>
                    </>
                  ) : selectedFile ? (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Start Processing</span>
                    </>
                  ) : activeJob ? (
                    <>
                      <UploadCloud className="w-4 h-4 text-cyan-400" />
                      <span>Upload New Video</span>
                    </>
                  ) : (
                    <>
                      <UploadCloud className="w-4 h-4 text-cyan-400" />
                      <span>Choose Video File</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Dynamic Timeline Coverage Estimation Pill */}
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

        {/* Right Column: Local/Remote Video Preview Player (5 cols) */}
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
              {hasLocalVideo && (
                <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Staged Upload
                </span>
              )}
              {hasRemoteVideo && (
                <span className="text-[10px] text-cyan-400 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  Stream ({activeJob?.status})
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
                    ? 'Individual extracted frames and perception telemetry for this historical session are displayed below.'
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
      {(processingState === 'UPLOADING' ||
        processingState === 'PROCESSING' ||
        processingState === 'COMPLETED' ||
        processingState === 'FAILED') && (
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-2">
              {processingState === 'UPLOADING' || processingState === 'PROCESSING' ? (
                <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              ) : processingState === 'COMPLETED' ? (
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
          {(processingState === 'UPLOADING' || processingState === 'PROCESSING') && (
            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
                style={{ width: `${Math.max(15, progressPct)}%` }}
              />
            </div>
          )}

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-300 font-mono">
              <strong>Error:</strong> {errorMessage}
            </div>
          )}

          {/* Processing Completed Summary Card & Quick Handoff Navigation */}
          {processingState === 'COMPLETED' && completedJob && (
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
