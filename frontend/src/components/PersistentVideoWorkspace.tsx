import React, { useState, useRef } from 'react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from './VideoCanvas';
import { PlaybackControls } from './PlaybackControls';
import { InteractiveTimeline } from './InteractiveTimeline';
import { VideoUploadSection } from './VideoUploadSection';
import type { AnalysisMode } from '../types/api';
import {
  UploadCloud,
  FileVideo,
  Sparkles,
  Ban,
  RotateCw,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Film,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  History,
  X,
} from 'lucide-react';

const formatDuration = (seconds: number): string => {
  if (!seconds || isNaN(seconds) || seconds <= 0) return '00:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
};

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

export const PersistentVideoWorkspace: React.FC = () => {
  const {
    activeJobId,
    activeJob,
    results,
    stagedVideoSrc,
    uploadedVideoId,
    videoProcessingState,
    uploadProgressPct,
    processingProgressPct,
    statusMessage,
    errorMessage,
    uploadedFile,
    analysisMode,
    rotation,
    duration,
    setAnalysisMode,
    setRotation,
    uploadFile,
    startProcessing,
    cancelProcessing,
    retryProcessing,
    resetForNewVideo,
  } = useJob();

  const [showAdvancedPanel, setShowAdvancedPanel] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  };

  const effectiveDuration =
    duration ||
    results?.timeline_telemetry?.source_video_duration_sec ||
    results?.video_metadata?.duration_seconds ||
    activeJob?.timeline_telemetry?.source_video_duration_sec ||
    0;

  const displayFilename =
    uploadedFile?.name ||
    activeJob?.filename ||
    (uploadedVideoId ? `stream_${uploadedVideoId.slice(0, 10)}` : 'Driving Stream');

  const cleanFilename = displayFilename.startsWith('up_')
    ? displayFilename.slice(12)
    : displayFilename;

  const isVideoStaged = Boolean(
    stagedVideoSrc || uploadedVideoId || activeJobId || results || uploadedFile
  );

  return (
    <div className="space-y-3">
      {/* Hidden File Picker Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp4,.avi,.mov,.mkv,.webm"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Stream Failure Alert Banner */}
      {(videoProcessingState === 'PROCESSING_FAILED' || activeJob?.status === 'FAILED') && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-lg animate-in fade-in duration-200">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-rose-500/20 text-rose-400 rounded-xl shrink-0">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-rose-300">Pipeline Execution Error</h4>
              <p className="text-xs text-rose-400/80 font-mono mt-0.5">
                {errorMessage ||
                  activeJob?.error_message ||
                  activeJob?.message ||
                  'Processing was halted due to an unhandled pipeline error.'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 self-start sm:self-auto">
            <button
              type="button"
              onClick={() => retryProcessing()}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs flex items-center gap-1.5 transition-colors shadow-md"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Processing</span>
            </button>
            <button
              type="button"
              onClick={() => resetForNewVideo()}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs transition-colors"
            >
              Upload New Video
            </button>
          </div>
        </div>
      )}

      {/* Upload Failed Alert Banner */}
      {videoProcessingState === 'UPLOAD_FAILED' && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-2xl p-4 flex items-center justify-between gap-4 shadow-lg">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <div>
              <h4 className="text-sm font-bold text-rose-300">Video Upload Failed</h4>
              <p className="text-xs text-rose-400/80 font-mono mt-0.5">
                {errorMessage || 'File upload failed. Ensure the video is in MP4/MOV/AVI/WEBM format and under 100MB.'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs flex items-center gap-1.5 transition-colors shrink-0"
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Select Another File</span>
          </button>
        </div>
      )}

      {/* UNIFIED VIDEO CONTROL & PROCESSING ACTION BAR (ALWAYS VISIBLE & UNCOLLAPSED) */}
      {!isVideoStaged || videoProcessingState === 'NO_VIDEO' ? (
        /* State 1: Dropzone for Initial Video Ingestion */
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setIsDragOver(false);
          }}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center select-none bg-slate-900/90 shadow-xl ${
            isDragOver
              ? 'border-cyan-400 bg-cyan-950/20 shadow-cyan-950/30 scale-[1.005]'
              : 'border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900'
          }`}
        >
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-3">
            <UploadCloud className="w-6 h-6" />
          </div>
          <h4 className="text-sm font-bold text-white mb-1">
            Upload Road Video to Begin Perception Analysis
          </h4>
          <p className="text-xs text-slate-400 max-w-md mb-3">
            Drag & drop raw driving video (.mp4, .mov, .avi, .mkv, .webm up to 100 MB) or browse local files.
          </p>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-bold shadow-lg shadow-cyan-950/40 transition-all flex items-center gap-2"
          >
            <FileVideo className="w-4 h-4" />
            <span>Browse Driving Video</span>
          </button>
        </div>
      ) : videoProcessingState === 'UPLOADING' ? (
        /* State 2: Upload in Progress Bar */
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-cyan-300 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span>Uploading video '{cleanFilename}' to server storage...</span>
            </span>
            <span className="font-mono text-cyan-400 font-bold">{uploadProgressPct}%</span>
          </div>
          <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(10, uploadProgressPct)}%` }}
            />
          </div>
        </div>
      ) : videoProcessingState === 'PROCESSING' ? (
        /* State 3: Active Video Perception Pipeline Running */
        <div className="bg-slate-900 border border-cyan-500/30 rounded-2xl p-4 shadow-xl shadow-cyan-950/20 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 shrink-0">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Analyzing Video
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="text-[11px] font-mono text-cyan-400 truncate">
                    {cleanFilename}
                  </span>
                </div>
                <p className="text-xs text-slate-400 truncate font-mono mt-0.5">
                  {statusMessage || 'Executing YOLO11 + ByteTrack + Trajectory + Collision Risk pipeline...'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
              <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-lg border border-cyan-800/60">
                {processingProgressPct}%
              </span>
              <button
                type="button"
                onClick={() => cancelProcessing()}
                className="px-3 py-1.5 rounded-lg bg-rose-600/90 hover:bg-rose-500 text-white font-semibold text-xs flex items-center gap-1.5 shadow transition-colors"
                title="Halt current processing pipeline"
              >
                <Ban className="w-3.5 h-3.5" />
                <span>Cancel</span>
              </button>
            </div>
          </div>

          {/* Real-time Progress Bar */}
          <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 via-sky-400 to-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(5, processingProgressPct)}%` }}
            />
          </div>
        </div>
      ) : (
        /* State 4: Video Loaded / Staged (VIDEO_READY, COMPLETED, or CANCELLED) -> PROMINENT ACTION BAR */
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            {/* Stream Identification Badge */}
            <div className="flex items-center gap-3 min-w-0">
              <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 shrink-0">
                <Film className="w-4 h-4" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-white truncate max-w-[200px] sm:max-w-[280px] font-mono">
                    {cleanFilename}
                  </span>
                  {videoProcessingState === 'COMPLETED' ? (
                    <span className="px-2 py-0.5 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-[10px] font-bold flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      ANALYZED
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-md bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono text-[10px] font-bold flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                      READY
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono mt-0.5">
                  <span>Duration: {formatDuration(effectiveDuration)}</span>
                  {uploadedFile && <span>• {formatFileSize(uploadedFile.size)}</span>}
                  {results?.frames_processed && (
                    <span>• {results.frames_processed} frames analyzed</span>
                  )}
                </div>
              </div>
            </div>

            {/* Quick Actions: Replace Video & Toggle Drawer */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                title="Select a different video file"
              >
                <UploadCloud className="w-3.5 h-3.5 text-cyan-400" />
                <span className="hidden sm:inline">Change Video</span>
              </button>

              <button
                type="button"
                onClick={() => setShowAdvancedPanel((prev) => !prev)}
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 text-xs flex items-center gap-1 transition-colors"
                title="Toggle advanced upload & batch details"
              >
                <span className="hidden sm:inline">Advanced</span>
                {showAdvancedPanel ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Processing Settings & THE PROMINENT "START PROCESSING" BUTTON */}
          <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              {/* Analysis Mode Selector */}
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-semibold text-slate-400">Mode:</span>
                <select
                  value={analysisMode}
                  onChange={(e) => setAnalysisMode(e.target.value as AnalysisMode)}
                  className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                >
                  <option value="FULL_BALANCED">Full Video (2 FPS, CPU Optimized, Recommended)</option>
                  <option value="FULL_DETAILED">Full Video (5 FPS, High Detail)</option>
                  <option value="QUICK_PREVIEW">Quick Preview (30 Frames)</option>
                </select>
              </div>

              {/* Rotation Selector */}
              <button
                type="button"
                onClick={() => setRotation(((rotation + 90) % 360) as 0 | 90 | 180 | 270)}
                className="bg-slate-950 hover:bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-mono flex items-center gap-1.5 transition-colors"
                title="Cycle rotation clockwise: 0°, 90°, 180°, 270°"
              >
                <RotateCw className="w-3.5 h-3.5 text-cyan-400" />
                <span>{rotation === 0 ? '0° Upright' : `+${rotation}°`}</span>
              </button>
            </div>

            {/* THE PROMINENT, UN-COLLAPSIBLE START PROCESSING BUTTON */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => startProcessing()}
                className="px-5 py-2.5 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-xl shadow-cyan-950/60 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 hover:from-cyan-400 hover:via-sky-400 hover:to-blue-500 text-white"
              >
                <Sparkles className="w-4 h-4 text-cyan-100" />
                <span>{videoProcessingState === 'COMPLETED' ? 'RE-RUN PROCESSING' : 'START PROCESSING'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Advanced Details Collapsible Panel (Optional) */}
      {showAdvancedPanel && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <History className="w-4 h-4 text-cyan-400" />
              Advanced Video Ingestion & Stream Details
            </span>
            <button
              type="button"
              onClick={() => setShowAdvancedPanel(false)}
              className="text-slate-400 hover:text-white p-1 rounded-lg"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <VideoUploadSection />
        </div>
      )}

      {/* Persistent Canvas and Video Element (Never Unmounts Across Navigation!) */}
      <div className="space-y-3">
        <VideoCanvas />
        <PlaybackControls />
        <InteractiveTimeline />
      </div>
    </div>
  );
};
