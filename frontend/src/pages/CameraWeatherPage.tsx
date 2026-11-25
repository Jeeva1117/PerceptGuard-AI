import React from 'react';
import {
  CloudSun,
  Gauge,
  Film,
  Activity,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export interface CameraWeatherPageProps {
  isEmbedded?: boolean;
}

export const CameraWeatherPage: React.FC<CameraWeatherPageProps> = ({ isEmbedded = false }) => {
  const {
    results,
    currentFrame,
    activeJob,
    videoProcessingState,
    uploadedVideoMetadata,
    duration,
  } = useJob();

  const weather = currentFrame?.weather_intelligence;
  const visScore = weather?.visibility_score ?? null;
  const primaryCondition = weather?.primary_condition || weather?.condition || 'CLEAR';
  const camQuality = weather?.camera_quality_score ?? null;
  const metrics = weather?.metrics;
  const reasons = weather?.degradation_reasons || weather?.evidence || [];

  // Visibility category
  const visibilityCategory =
    results?.final_summary?.visibility ||
    (visScore == null
      ? 'UNKNOWN'
      : visScore >= 75
      ? 'CLEAR'
      : visScore >= 50
      ? 'MODERATE'
      : visScore >= 35
      ? 'DEGRADED'
      : 'LOW VISIBILITY');

  // Stream metadata
  const videoMeta = results?.video_metadata || uploadedVideoMetadata?.metadata;
  const videoDuration =
    results?.timeline_telemetry?.source_video_duration_sec ||
    videoMeta?.duration_seconds ||
    duration ||
    null;
  const videoFps = videoMeta?.fps || results?.timeline_telemetry?.source_fps || null;
  const videoResolution =
    videoMeta?.width && videoMeta?.height ? `${videoMeta.width}x${videoMeta.height}` : null;
  const framesProcessedCount =
    results?.frames_processed ??
    activeJob?.frames_processed ??
    (currentFrame ? currentFrame.frame_index + 1 : 0);

  const getConditionColor = (cond: string) => {
    switch (cond.toUpperCase()) {
      case 'CLEAR':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      case 'RAIN':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
      case 'FOG':
        return 'bg-slate-500/10 text-slate-300 border-slate-500/30';
      case 'SUN_GLARE':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'LOW_LIGHT':
        return 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30';
      case 'BLUR':
      case 'OBSTRUCTION':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };


  return (
    <div className="space-y-4 select-none">
      {/* Module Title Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400 shrink-0">
            <CloudSun className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-wide">
                Camera & Weather
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                MODULE 1
              </span>
            </div>
            <p className="text-[11px] text-slate-400 truncate">
              Optical visibility scoring & sensor health
            </p>
          </div>
        </div>

        <span
          className={`px-2.5 py-1 rounded-full text-[10px] font-bold font-mono border shrink-0 ${
            activeJob?.status === 'COMPLETED' || videoProcessingState === 'COMPLETED'
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : videoProcessingState === 'PROCESSING' || activeJob?.status === 'PROCESSING'
              ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30 animate-pulse'
              : 'bg-slate-800 text-slate-400 border-slate-700'
          }`}
        >
          {activeJob?.status === 'COMPLETED' || videoProcessingState === 'COMPLETED'
            ? 'ANALYZED'
            : videoProcessingState === 'PROCESSING'
            ? 'ANALYZING'
            : 'READY'}
        </span>
      </div>

      {/* Standalone Video Player (Only when viewed directly outside master workspace) */}
      {!isEmbedded && (
        <div className="space-y-3">
          <VideoCanvas />
          <PlaybackControls />
        </div>
      )}

      {/* Video Stream & Processing Telemetry Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Film className="w-3.5 h-3.5 text-cyan-400" />
            Stream Telemetry
          </span>
          <span className="text-[11px] font-mono text-cyan-400">
            {framesProcessedCount} frames processed
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Duration</span>
            <strong className="text-white text-xs block mt-0.5">
              {videoDuration != null ? `${videoDuration.toFixed(1)}s` : 'NOT AVAILABLE'}
            </strong>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Resolution</span>
            <strong className="text-white text-xs block mt-0.5">
              {videoResolution || 'NOT AVAILABLE'}
            </strong>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Frame Rate</span>
            <strong className="text-white text-xs block mt-0.5">
              {videoFps != null ? `${videoFps.toFixed(0)} FPS` : 'NOT AVAILABLE'}
            </strong>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Pipeline State</span>
            <strong className="text-emerald-400 text-xs block mt-0.5">
              {activeJob?.status || 'READY'}
            </strong>
          </div>
        </div>
      </div>

      {/* Optical Visibility & Atmospheric Condition Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Gauge className="w-3.5 h-3.5 text-cyan-400" />
            Optical Visibility
          </span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getConditionColor(primaryCondition)}`}>
            {primaryCondition}
          </span>
        </div>

        <div className="flex items-baseline justify-between pt-1">
          <div>
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Visibility Index</span>
            <div className="flex items-baseline gap-2 mt-0.5">
              <span className="text-3xl font-extrabold text-white font-mono">
                {visScore != null ? `${visScore.toFixed(0)}%` : 'NOT AVAILABLE'}
              </span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getConditionColor(primaryCondition)}`}>
                {visibilityCategory}
              </span>
            </div>
          </div>

          <div className="text-right">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Camera Quality</span>
            <strong className="text-sm font-mono text-cyan-300 block mt-0.5">
              {camQuality != null ? `${camQuality.toFixed(0)} / 100` : 'NOT AVAILABLE'}
            </strong>
          </div>
        </div>

        {/* Visibility Gauge Bar */}
        <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              visScore != null && visScore >= 75
                ? 'bg-gradient-to-r from-emerald-500 to-cyan-400'
                : visScore != null && visScore >= 50
                ? 'bg-gradient-to-r from-amber-500 to-yellow-400'
                : 'bg-gradient-to-r from-rose-500 to-red-600'
            }`}
            style={{ width: `${visScore != null ? Math.max(5, Math.min(100, visScore)) : 0}%` }}
          />
        </div>
      </div>

      {/* Validated Physical Optical Metrics */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            Sensor & Image Metrics
          </span>
          <span className="text-[10px] text-slate-500 font-mono">Physical Analysis</span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Laplacian Blur</span>
            <strong className="text-slate-200 text-xs block mt-0.5">
              {metrics?.blur_laplacian != null ? metrics.blur_laplacian.toFixed(1) : 'NOT AVAILABLE'}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Mean Brightness</span>
            <strong className="text-slate-200 text-xs block mt-0.5">
              {metrics?.brightness != null ? `${metrics.brightness.toFixed(1)} / 255` : 'NOT AVAILABLE'}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Contrast</span>
            <strong className="text-slate-200 text-xs block mt-0.5">
              {metrics?.contrast != null ? metrics.contrast.toFixed(1) : 'NOT AVAILABLE'}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Glare Ratio</span>
            <strong className="text-slate-200 text-xs block mt-0.5">
              {metrics?.glare_ratio != null ? `${(metrics.glare_ratio * 100).toFixed(1)}%` : 'NOT AVAILABLE'}
            </strong>
          </div>
        </div>
      </div>

      {/* Explainable Perception Diagnostics */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-2">
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
          Perception Evidence & Degradation:
        </span>
        {reasons && reasons.length > 0 ? (
          <div className="space-y-1.5">
            {reasons.map((r, i) => (
              <div key={i} className="text-xs text-amber-300 bg-amber-950/30 border border-amber-800/40 rounded-lg p-2 flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                <span>{r}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-400 font-sans italic bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            Nominal optical clarity; zero atmospheric sensor attenuation detected.
          </p>
        )}
      </div>
    </div>
  );
};
