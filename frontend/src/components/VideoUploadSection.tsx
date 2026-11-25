import React, { useState, useRef } from 'react';
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
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { useJob } from '../context/JobContext';
import type { AnalysisMode } from '../types/api';

export const VideoUploadSection: React.FC = () => {
  const {
    activeJobId,
    activeJob,
    results,
    uploadedVideoId,
    uploadedVideoMetadata,
    uploadedFile,
    videoProcessingState,
    uploadProgressPct,
    processingProgressPct,
    statusMessage,
    errorMessage,
    analysisMode,
    rotation,
    setAnalysisMode,
    setRotation,
    setSampleFps,
    uploadFile,
    startProcessing,
    cancelProcessing,
    resetForNewVideo,
  } = useJob();

  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
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
      uploadFile(e.dataTransfer.files[0]);
    }
  };

  const displayFilename =
    uploadedFile?.name ||
    activeJob?.filename ||
    (uploadedVideoId ? `stream_${uploadedVideoId.slice(0, 10)}` : null);

  const cleanFilename = displayFilename
    ? displayFilename.startsWith('up_')
      ? displayFilename.slice(12)
      : displayFilename
    : null;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400">
            <UploadCloud className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Upload Road Video</h3>
            <p className="text-xs text-slate-400">
              Ingest driving video for visibility analysis, YOLO detection, ByteTrack & collision forecasting
            </p>
          </div>
        </div>

        {/* State Badge */}
        <div className="flex items-center gap-2">
          <span
            className={`px-3 py-1 rounded-full text-xs font-mono font-bold border flex items-center gap-1.5 ${
              videoProcessingState === 'COMPLETED'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : videoProcessingState === 'PROCESSING'
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 animate-pulse'
                : videoProcessingState === 'UPLOADING'
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 animate-pulse'
                : videoProcessingState === 'VIDEO_READY'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : videoProcessingState === 'PROCESSING_FAILED' || videoProcessingState === 'UPLOAD_FAILED'
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            {videoProcessingState === 'PROCESSING' || videoProcessingState === 'UPLOADING' ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : videoProcessingState === 'COMPLETED' || videoProcessingState === 'VIDEO_READY' ? (
              <CheckCircle2 className="w-3.5 h-3.5" />
            ) : videoProcessingState === 'PROCESSING_FAILED' || videoProcessingState === 'UPLOAD_FAILED' ? (
              <AlertCircle className="w-3.5 h-3.5" />
            ) : null}
            STATE: {videoProcessingState}
          </span>
        </div>
      </div>

      {/* Hidden File Input */}
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
            : cleanFilename
            ? 'border-slate-700 bg-slate-950/40 hover:border-slate-600'
            : 'border-slate-700/80 hover:border-cyan-500/50 hover:bg-slate-950/60'
        }`}
      >
        <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-3">
          <FileVideo className="w-6 h-6" />
        </div>

        <h4 className="text-sm font-semibold text-white mb-1">
          {cleanFilename ? 'Replace Road Video' : 'Drag & Drop Video or Browse Files'}
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

      {/* Selected / Uploaded File Metadata Card */}
      {cleanFilename && (
        <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 shrink-0">
              <Film className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <div className="text-xs font-semibold text-white truncate font-mono">
                {cleanFilename}
              </div>
              <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                {uploadedFile && <span>{formatFileSize(uploadedFile.size)}</span>}
                {uploadedVideoMetadata?.metadata?.fps && (
                  <>
                    <span>•</span>
                    <span className="text-cyan-300 font-mono">
                      {uploadedVideoMetadata.metadata.fps} FPS
                    </span>
                  </>
                )}
                {uploadedVideoMetadata?.metadata?.duration_seconds && (
                  <>
                    <span>•</span>
                    <span>{uploadedVideoMetadata.metadata.duration_seconds.toFixed(1)}s</span>
                  </>
                )}
                {uploadedVideoId && (
                  <>
                    <span>•</span>
                    <span className="text-emerald-400 font-mono text-[10px]">
                      Server Verified
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
              disabled={videoProcessingState === 'PROCESSING' || videoProcessingState === 'UPLOADING'}
              title="Replace with another video"
              className="px-2.5 py-1 rounded-lg text-xs font-mono text-cyan-400 hover:text-cyan-300 bg-slate-900 border border-slate-700 transition-colors disabled:opacity-40"
            >
              Replace
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                resetForNewVideo();
              }}
              disabled={videoProcessingState === 'PROCESSING' || videoProcessingState === 'UPLOADING'}
              title="Remove selected video"
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-900 transition-colors disabled:opacity-40"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
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
              disabled={videoProcessingState === 'UPLOADING' || videoProcessingState === 'PROCESSING'}
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
                onClick={() => setRotation(((rotation + 90) % 360) as 0 | 90 | 180 | 270)}
                disabled={videoProcessingState === 'UPLOADING' || videoProcessingState === 'PROCESSING'}
                className="bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-mono flex items-center gap-1.5 transition-colors disabled:opacity-50"
                title="Cycle rotation clockwise: 0°, 90°, 180°, 270°"
              >
                <RotateCw className="w-3.5 h-3.5 text-cyan-400" />
                <span>{rotation === 0 ? '0° (Upright)' : `${rotation}° CW`}</span>
              </button>
            </div>

            {/* Start Processing / Cancel Action Button */}
            {videoProcessingState === 'PROCESSING' ? (
              <button
                type="button"
                onClick={() => cancelProcessing()}
                className="px-4 py-2 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-950/50 transition-all"
              >
                <Ban className="w-4 h-4" />
                <span>Cancel Processing</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => startProcessing()}
                disabled={
                  videoProcessingState === 'UPLOADING' ||
                  (!uploadedVideoId && !uploadedFile && !activeJob)
                }
                className={`px-5 py-2.5 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg transition-all ${
                  videoProcessingState === 'UPLOADING'
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/60'
                    : uploadedVideoId || uploadedFile || activeJob
                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-950/50'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-500 border border-slate-700 cursor-not-allowed'
                }`}
              >
                {videoProcessingState === 'UPLOADING' ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                    <span>Uploading Video ({uploadProgressPct}%)...</span>
                  </>
                ) : uploadedVideoId || uploadedFile || activeJob ? (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>
                      {videoProcessingState === 'COMPLETED' ? 'Re-run Processing' : 'Start Processing'}
                    </span>
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
      </div>

      {/* Progress & Live Status Feedback Strip */}
      {(videoProcessingState === 'UPLOADING' ||
        videoProcessingState === 'PROCESSING' ||
        videoProcessingState === 'COMPLETED' ||
        videoProcessingState === 'PROCESSING_FAILED' ||
        videoProcessingState === 'UPLOAD_FAILED') && (
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-2">
              {videoProcessingState === 'UPLOADING' || videoProcessingState === 'PROCESSING' ? (
                <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              ) : videoProcessingState === 'COMPLETED' ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertCircle className="w-4 h-4 text-rose-400" />
              )}
              {statusMessage || (videoProcessingState === 'COMPLETED' ? 'Pipeline execution completed.' : '')}
            </span>

            {activeJobId && (
              <span className="text-[11px] font-mono text-cyan-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                Session: {activeJobId}
              </span>
            )}
          </div>

          {/* Progress bar */}
          {(videoProcessingState === 'UPLOADING' || videoProcessingState === 'PROCESSING') && (
            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
                style={{
                  width: `${Math.max(
                    10,
                    videoProcessingState === 'UPLOADING' ? uploadProgressPct : processingProgressPct
                  )}%`,
                }}
              />
            </div>
          )}

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-300 font-mono flex items-center justify-between gap-3">
              <div>
                <strong>Error:</strong> {errorMessage}
              </div>
            </div>
          )}

          {/* Processing Completed Summary Card */}
          {videoProcessingState === 'COMPLETED' && (
            <div className="pt-2 border-t border-slate-800/80 space-y-3 animate-in fade-in duration-200">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Frames Processed</span>
                  <strong className="text-sm font-mono text-cyan-400">
                    {results?.frames_processed ?? activeJob?.frames_processed ?? 0}
                  </strong>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Unique Tracks</span>
                  <strong className="text-sm font-mono text-emerald-400">
                    {results?.unique_tracks ?? activeJob?.unique_tracks ?? 0}
                  </strong>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Alerts Triggered</span>
                  <strong className="text-sm font-mono text-amber-400">
                    {results?.total_alerts ?? activeJob?.alerts_count ?? 0}
                  </strong>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Telemetry Stream</span>
                  <span className="text-xs font-mono text-slate-300 block mt-0.5 truncate">
                    {cleanFilename}
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
