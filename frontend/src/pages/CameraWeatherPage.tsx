import React from 'react';
import {
  CloudSun,
  Gauge,
  Activity,
  Droplets,
  CloudFog,
  Eye,
  Moon,
  SunMedium,
  TrendingDown,
  TrendingUp,
  Minus,
  ShieldAlert,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';
import { VideoUploadSection } from '../components/VideoUploadSection';
import {
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';

export interface CameraWeatherPageProps {
  isEmbedded?: boolean;
}

export const CameraWeatherPage: React.FC<CameraWeatherPageProps> = ({ isEmbedded = false }) => {
  const { results, currentFrame, activeJob } = useJob();

  const weather = currentFrame?.weather_intelligence;
  const hasData = Boolean(weather);
  const visScore = weather?.visibility_score ?? null;
  const visLevel = weather?.visibility_level ?? (visScore != null ? (visScore >= 80 ? 'EXCELLENT' : visScore >= 65 ? 'GOOD' : visScore >= 45 ? 'DEGRADED' : 'CRITICAL') : 'UNKNOWN');
  const camQuality = weather?.camera_quality_score ?? null;
  const primaryCondition = weather?.primary_condition || weather?.condition || (activeJob?.status === 'FAILED' ? 'UNAVAILABLE' : hasData ? 'CLEAR' : 'UNKNOWN');
  const confidence = weather?.confidence != null ? Math.round(weather.confidence * 100) : null;
  const uncertainty = weather?.uncertainty != null ? Math.round(weather.uncertainty * 100) : null;
  const degradationReasons: string[] = weather
    ? (Array.isArray(weather?.degradation_reasons)
      ? weather.degradation_reasons
      : Array.isArray(weather?.evidence)
      ? weather.evidence
      : typeof weather?.evidence === 'string'
      ? [weather.evidence]
      : typeof weather?.degradation_reasons === 'string'
      ? [weather.degradation_reasons]
      : ['Optical clarity and atmospheric transmission within optimal parameters.'])
    : (activeJob?.status === 'FAILED'
      ? ['Video processing pipeline execution failed. Review session logs.']
      : ['Awaiting video stream telemetry ingestion.']);

  const rain = weather?.rain;
  const fog = weather?.fog;
  const blur = weather?.blur;
  const lighting = weather?.lighting;
  const glare = weather?.glare;
  const forecast = weather?.forecast;
  const perceptionImpact = weather?.perception_impact;

  // Prepare chart series: Historical observed points up to current frame,
  // then future forecast points projected ahead
  const allFrames = results?.frames || [];
  const chartData: any[] = [];

  // Observed points
  allFrames.forEach((f) => {
    chartData.push({
      frame: f.frame_index,
      observed: f.weather_intelligence?.visibility_score ?? 100,
      predicted: null,
      type: 'Observed',
    });
  });

  // Append forecast horizons from current frame if available
  if (forecast?.horizons && allFrames.length > 0) {
    const curIdx = currentFrame?.frame_index ?? (allFrames.length - 1);
    const hShort = forecast.horizons['h_short'];
    const hMed = forecast.horizons['h_med'];
    const hLong = forecast.horizons['h_long'];

    if (hShort) {
      chartData.push({
        frame: curIdx + hShort.horizon_frames,
        observed: null,
        predicted: hShort.predicted_score,
        type: `Forecast (+${hShort.horizon_seconds}s)`,
      });
    }
    if (hMed) {
      chartData.push({
        frame: curIdx + hMed.horizon_frames,
        observed: null,
        predicted: hMed.predicted_score,
        type: `Forecast (+${hMed.horizon_seconds}s)`,
      });
    }
    if (hLong) {
      chartData.push({
        frame: curIdx + hLong.horizon_frames,
        observed: null,
        predicted: hLong.predicted_score,
        type: `Forecast (+${hLong.horizon_seconds}s)`,
      });
    }
  }

  const getConditionColor = (cond: string) => {
    switch (cond.toUpperCase()) {
      case 'RAIN':
        return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      case 'FOG':
        return 'text-slate-300 bg-slate-500/10 border-slate-500/30';
      case 'BLUR':
        return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
      case 'LOW_LIGHT':
        return 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30';
      case 'SUN_GLARE':
        return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
      case 'OBSTRUCTION':
        return 'text-red-400 bg-red-500/10 border-red-500/30';
      default:
        return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    }
  };

  const getTrendIcon = (trend?: string) => {
    switch (trend) {
      case 'Improving':
        return <TrendingUp className="w-4 h-4 text-emerald-400" />;
      case 'Gradually degrading':
      case 'Rapidly degrading':
        return <TrendingDown className="w-4 h-4 text-rose-400" />;
      default:
        return <Minus className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Module Scope */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-cyan-500/10 rounded-xl text-cyan-400">
              <CloudSun className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Camera & Weather Intelligence
              </h2>
              <p className="text-xs text-slate-400">
                Continuous optical visibility scoring, atmospheric degradation detection & predictive horizon forecasting
              </p>
            </div>
          </div>
        </div>

        {/* Global Summary Badge */}
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
              visScore == null
                ? 'bg-slate-800 text-slate-400 border-slate-700'
                : visScore >= 75
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : visScore >= 50
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                : 'bg-red-500/10 text-red-400 border-red-500/30'
            }`}
          >
            {visScore == null
              ? (hasData ? 'CLEAR' : 'UNKNOWN')
              : visScore >= 75
              ? 'CLEAR'
              : visScore >= 50
              ? 'MODERATE'
              : 'LOW'}
          </span>
          {results?.weather_summary && (
            <div className="flex items-center gap-3 text-xs bg-slate-900 border border-slate-800 rounded-xl px-4 py-2">
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Mean Visibility</span>
                <strong className="text-cyan-400 font-mono text-sm">
                  {results.weather_summary.mean_visibility_score}%
                </strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Degraded Frames</span>
                <strong className="text-amber-400 font-mono text-sm">
                  {results.weather_summary.degraded_frames_count ?? 0} / {results.frames_processed ?? allFrames.length}
                </strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Predominant Condition</span>
                <strong className="text-white font-mono text-sm">
                  {results.weather_summary.predominant_condition || 'CLEAR'}
                </strong>
              </div>
              {results?.timeline_telemetry && (
                <>
                  <div className="w-px h-6 bg-slate-800" />
                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">Timeline Coverage</span>
                    <strong className="text-emerald-400 font-mono text-sm">
                      {results.timeline_telemetry.timeline_coverage_percentage}% ({results.timeline_telemetry.total_frames_analyzed} frames @ {results.timeline_telemetry.effective_sampling_fps} FPS)
                    </strong>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Primary Video Upload & Ingestion Interface */}
      {!isEmbedded && <VideoUploadSection />}

      {/* Stream Failure Alert Banner */}
      {!isEmbedded && activeJob?.status === 'FAILED' && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-rose-500/20 text-rose-400 rounded-xl shrink-0">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-rose-300">Pipeline Execution Halted on Selected Stream</h4>
              <p className="text-xs text-rose-400/80 font-mono mt-0.5">
                {activeJob.error_message || activeJob.message || 'Processing was halted due to an unhandled pipeline error.'}
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-rose-400 bg-rose-950 px-3 py-1 rounded-lg border border-rose-800 shrink-0 self-start sm:self-auto">
            Stream: {activeJob.filename}
          </span>
        </div>
      )}

      {/* Main Grid: Visual Feed & Playback Scrubber */}
      {isEmbedded ? (
        <div className="space-y-4">
          {/* Section B: Camera Condition Summary */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Gauge className="w-4 h-4 text-cyan-400" />
                Optical Visibility Index
              </span>
              <span
                className={`px-3 py-1 rounded-full text-xs font-bold border ${getConditionColor(
                  primaryCondition
                )}`}
              >
                {primaryCondition}
              </span>
            </div>

            <div className="flex items-baseline justify-between mb-2">
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-extrabold text-white font-mono tracking-tight">
                  {visScore != null ? `${visScore.toFixed(0)}%` : '--'}
                </span>
                <span
                  className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                    visLevel === 'EXCELLENT'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : visLevel === 'GOOD'
                      ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                      : visLevel === 'DEGRADED'
                      ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      : visLevel === 'CRITICAL'
                      ? 'bg-red-500/10 text-red-400 border-red-500/20 animate-pulse'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {visLevel}
                </span>
              </div>

              <div className="text-right">
                <span className="text-[10px] text-slate-500 uppercase block">Camera Quality</span>
                <strong className="text-sm font-mono text-cyan-300">
                  {camQuality != null ? `${camQuality.toFixed(0)} / 100` : '-- / 100'}
                </strong>
              </div>
            </div>

            {/* Visibility Gauge Bar */}
            <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800 mb-4">
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  visScore != null && visScore >= 80
                    ? 'bg-gradient-to-r from-emerald-500 to-cyan-500'
                    : visScore != null && visScore >= 65
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500'
                    : visScore != null && visScore >= 45
                    ? 'bg-gradient-to-r from-amber-500 to-yellow-500'
                    : 'bg-gradient-to-r from-red-600 to-rose-500'
                }`}
                style={{ width: `${visScore != null ? Math.max(5, Math.min(100, visScore)) : 0}%` }}
              />
            </div>

            {/* Confidence & Uncertainty Chips */}
            <div className="grid grid-cols-2 gap-2 pb-3 mb-3 border-b border-slate-800/80 text-xs">
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Confidence</span>
                <strong className="text-xs font-mono text-emerald-400">
                  {confidence != null ? `${confidence}%` : '--%'}
                </strong>
              </div>
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Uncertainty</span>
                <strong className="text-xs font-mono text-amber-400">
                  {uncertainty != null ? `±${uncertainty}%` : '±--%'}
                </strong>
              </div>
            </div>

            {/* Explainable Reasons */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold uppercase text-slate-500">
                Explainable Diagnostics
              </span>
              <div className="space-y-1 bg-slate-950/70 rounded-xl p-3 border border-slate-800 text-xs text-slate-300">
                {degradationReasons.map((line, i) => (
                  <p key={i} className="flex items-start gap-2">
                    <span className="text-cyan-400 font-mono mt-0.5">•</span>
                    <span>{line}</span>
                  </p>
                ))}
              </div>
            </div>
          </div>

          {/* Section E: Perception Impact Interface Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <ShieldAlert className="w-4 h-4 text-cyan-400" />
                Downstream Perception Impact
              </span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800/60">
                Estimated Indicator
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Conf Penalty</span>
                <strong className="text-sm font-mono text-rose-400">
                  {hasData ? `-${((perceptionImpact?.expected_confidence_penalty ?? 0.05) * 100).toFixed(1)}%` : '--'}
                </strong>
              </div>
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Boundary Risk</span>
                <span
                  className={`text-[11px] font-mono font-bold block mt-0.5 ${
                    !hasData
                      ? 'text-slate-500'
                      : perceptionImpact?.boundary_degradation_risk === 'CRITICAL'
                      ? 'text-red-400'
                      : perceptionImpact?.boundary_degradation_risk === 'HIGH'
                      ? 'text-orange-400'
                      : perceptionImpact?.boundary_degradation_risk === 'MODERATE'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {hasData ? (perceptionImpact?.boundary_degradation_risk ?? 'LOW') : 'UNKNOWN'}
                </span>
              </div>
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Small Target Risk</span>
                <span
                  className={`text-[11px] font-mono font-bold block mt-0.5 ${
                    !hasData
                      ? 'text-slate-500'
                      : perceptionImpact?.small_object_risk === 'CRITICAL'
                      ? 'text-red-400'
                      : perceptionImpact?.small_object_risk === 'HIGH'
                      ? 'text-orange-400'
                      : perceptionImpact?.small_object_risk === 'MODERATE'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {hasData ? (perceptionImpact?.small_object_risk ?? 'LOW') : 'UNKNOWN'}
                </span>
              </div>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-3 border border-slate-800 text-[11px] text-slate-300 flex items-start gap-2">
              <span className="font-semibold text-cyan-400 uppercase tracking-wider shrink-0">Mode:</span>
              <span>
                <strong>{hasData ? (perceptionImpact?.suggested_detection_sensitivity ?? 'NORMAL') : 'INACTIVE'}</strong> —{' '}
                {hasData ? (perceptionImpact?.rationale || 'Nominal optical conditions.') : 'No frame telemetry available.'}
              </span>
            </div>
          </div>
        </div>
      ) : (
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Video Canvas & Scrubber */}
        <div className="lg:col-span-7 space-y-4">
          <VideoCanvas />
          <PlaybackControls />
        </div>

        {/* Right 5 Cols: Camera Condition Summary & Visibility Gauge */}
        <div className="lg:col-span-5 space-y-4">
          {/* Section B: Camera Condition Summary */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Gauge className="w-4 h-4 text-cyan-400" />
                Optical Visibility Index
              </span>
              <span
                className={`px-3 py-1 rounded-full text-xs font-bold border ${getConditionColor(
                  primaryCondition
                )}`}
              >
                {primaryCondition}
              </span>
            </div>

            <div className="flex items-baseline justify-between mb-2">
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-extrabold text-white font-mono tracking-tight">
                  {visScore != null ? `${visScore.toFixed(0)}%` : '--'}
                </span>
                <span
                  className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                    visLevel === 'EXCELLENT'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : visLevel === 'GOOD'
                      ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                      : visLevel === 'DEGRADED'
                      ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      : visLevel === 'CRITICAL'
                      ? 'bg-red-500/10 text-red-400 border-red-500/20 animate-pulse'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {visLevel}
                </span>
              </div>

              <div className="text-right">
                <span className="text-[10px] text-slate-500 uppercase block">Camera Quality</span>
                <strong className="text-sm font-mono text-cyan-300">
                  {camQuality != null ? `${camQuality.toFixed(0)} / 100` : '-- / 100'}
                </strong>
              </div>
            </div>

            {/* Visibility Gauge Bar */}
            <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800 mb-4">
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  visScore != null && visScore >= 80
                    ? 'bg-gradient-to-r from-emerald-500 to-cyan-500'
                    : visScore != null && visScore >= 65
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500'
                    : visScore != null && visScore >= 45
                    ? 'bg-gradient-to-r from-amber-500 to-yellow-500'
                    : 'bg-gradient-to-r from-red-600 to-rose-500'
                }`}
                style={{ width: `${visScore != null ? Math.max(5, Math.min(100, visScore)) : 0}%` }}
              />
            </div>

            {/* Confidence & Uncertainty Chips */}
            <div className="grid grid-cols-2 gap-2 pb-3 mb-3 border-b border-slate-800/80 text-xs">
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Confidence</span>
                <strong className="text-xs font-mono text-emerald-400">
                  {confidence != null ? `${confidence}%` : '--%'}
                </strong>
              </div>
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Uncertainty</span>
                <strong className="text-xs font-mono text-amber-400">
                  {uncertainty != null ? `±${uncertainty}%` : '±--%'}
                </strong>
              </div>
            </div>

            {/* Explainable Reasons */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold uppercase text-slate-500">
                Explainable Diagnostics
              </span>
              <div className="space-y-1 bg-slate-950/70 rounded-xl p-3 border border-slate-800 text-xs text-slate-300">
                {degradationReasons.map((line, i) => (
                  <p key={i} className="flex items-start gap-2">
                    <span className="text-cyan-400 font-mono mt-0.5">•</span>
                    <span>{line}</span>
                  </p>
                ))}
              </div>
            </div>
          </div>

          {/* Section E: Perception Impact Interface Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <ShieldAlert className="w-4 h-4 text-cyan-400" />
                Downstream Perception Impact
              </span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800/60">
                Estimated Indicator
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Conf Penalty</span>
                <strong className="text-sm font-mono text-rose-400">
                  {hasData ? `-${((perceptionImpact?.expected_confidence_penalty ?? 0.05) * 100).toFixed(1)}%` : '--'}
                </strong>
              </div>
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Boundary Risk</span>
                <span
                  className={`text-[11px] font-mono font-bold block mt-0.5 ${
                    !hasData
                      ? 'text-slate-500'
                      : perceptionImpact?.boundary_degradation_risk === 'CRITICAL'
                      ? 'text-red-400'
                      : perceptionImpact?.boundary_degradation_risk === 'HIGH'
                      ? 'text-orange-400'
                      : perceptionImpact?.boundary_degradation_risk === 'MODERATE'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {hasData ? (perceptionImpact?.boundary_degradation_risk ?? 'LOW') : 'UNKNOWN'}
                </span>
              </div>
              <div className="bg-slate-950 rounded-xl p-2.5 border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Small Target Risk</span>
                <span
                  className={`text-[11px] font-mono font-bold block mt-0.5 ${
                    !hasData
                      ? 'text-slate-500'
                      : perceptionImpact?.small_object_risk === 'CRITICAL'
                      ? 'text-red-400'
                      : perceptionImpact?.small_object_risk === 'HIGH'
                      ? 'text-orange-400'
                      : perceptionImpact?.small_object_risk === 'MODERATE'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {hasData ? (perceptionImpact?.small_object_risk ?? 'LOW') : 'UNKNOWN'}
                </span>
              </div>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-3 border border-slate-800 text-[11px] text-slate-300 flex items-start gap-2">
              <span className="font-semibold text-cyan-400 uppercase tracking-wider shrink-0">Mode:</span>
              <span>
                <strong>{hasData ? (perceptionImpact?.suggested_detection_sensitivity ?? 'NORMAL') : 'INACTIVE'}</strong> —{' '}
                {hasData ? (perceptionImpact?.rationale || 'Nominal optical conditions.') : 'No frame telemetry available.'}
              </span>
            </div>
          </div>
        </div>
      </div>
      )}

      {/* Section C: Multi-Condition Analysis Grid (Rain, Fog, Blur, Darkness, Glare) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
        {/* 1. Rain Analysis */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Droplets className="w-3.5 h-3.5 text-blue-400" />
              Precipitation
            </span>
            <span className="text-[10px] font-mono text-blue-400 font-bold">
              {hasData && rain ? `${Math.round(rain.severity * 100)}%` : '--'}
            </span>
          </div>
          <strong className="text-xs font-semibold text-white block truncate">
            {hasData ? (rain?.classification || 'No visible rain') : 'Telemetry Unavailable'}
          </strong>
          <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-blue-500 h-full rounded-full transition-all"
              style={{ width: `${hasData ? Math.max(4, (rain?.severity ?? 0.05) * 100) : 0}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-500 truncate">{hasData ? (rain?.details || 'Lens clear') : 'No input'}</p>
        </div>

        {/* 2. Fog Analysis */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <CloudFog className="w-3.5 h-3.5 text-slate-400" />
              Fog & Haze
            </span>
            <span className="text-[10px] font-mono text-slate-300 font-bold">
              {hasData && fog ? `${Math.round(fog.severity * 100)}%` : '--'}
            </span>
          </div>
          <strong className="text-xs font-semibold text-white block truncate">
            {hasData ? (fog?.classification || 'Clear') : 'Telemetry Unavailable'}
          </strong>
          <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-slate-400 h-full rounded-full transition-all"
              style={{ width: `${hasData ? Math.max(4, (fog?.severity ?? 0.05) * 100) : 0}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-500 truncate">{hasData ? (fog?.details || 'Atmosphere clear') : 'No input'}</p>
        </div>

        {/* 3. Blur Analysis */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-amber-400" />
              Camera Blur
            </span>
            <span className="text-[10px] font-mono text-amber-400 font-bold">
              {hasData && blur ? `${Math.round(blur.severity * 100)}%` : '--'}
            </span>
          </div>
          <strong className="text-xs font-semibold text-white block truncate">
            {hasData ? (blur?.classification || 'Sharp') : 'Telemetry Unavailable'}
          </strong>
          <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-amber-500 h-full rounded-full transition-all"
              style={{ width: `${hasData ? Math.max(4, (blur?.severity ?? 0.05) * 100) : 0}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-500 truncate">
            Laplacian: {hasData ? (blur?.laplacian_variance?.toFixed(1) ?? 'N/A') : '--'}
          </p>
        </div>

        {/* 4. Low-Light Analysis */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Moon className="w-3.5 h-3.5 text-indigo-400" />
              Illumination
            </span>
            <span className="text-[10px] font-mono text-indigo-400 font-bold">
              {hasData && lighting ? `${Math.round(lighting.severity * 100)}%` : '--'}
            </span>
          </div>
          <strong className="text-xs font-semibold text-white block truncate">
            {hasData ? (lighting?.classification || 'Normal lighting') : 'Telemetry Unavailable'}
          </strong>
          <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-indigo-500 h-full rounded-full transition-all"
              style={{ width: `${hasData ? Math.max(4, (lighting?.severity ?? 0.05) * 100) : 0}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-500 truncate">
            Luminance: {hasData ? (lighting?.mean_luminance?.toFixed(1) ?? 'N/A') : '--'}
          </p>
        </div>

        {/* 5. Sun Glare Analysis */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <SunMedium className="w-3.5 h-3.5 text-yellow-400" />
              Glare / Exposure
            </span>
            <span className="text-[10px] font-mono text-yellow-400 font-bold">
              {hasData && glare ? `${Math.round(glare.severity * 100)}%` : '--'}
            </span>
          </div>
          <strong className="text-xs font-semibold text-white block truncate">
            {hasData ? (glare?.classification || 'Normal exposure') : 'Telemetry Unavailable'}
          </strong>
          <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-yellow-500 h-full rounded-full transition-all"
              style={{ width: `${hasData ? Math.max(4, (glare?.severity ?? 0.05) * 100) : 0}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-500 truncate">
            Saturated: {hasData && glare ? `${(glare.saturated_pixel_ratio * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
      </div>

      {/* Section D: Temporal Visibility Trend & Multi-Horizon Forecast Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-cyan-400" />
              Temporal Visibility Trend & Lookahead Horizon Forecast
            </span>
            <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-[10px]">
              {getTrendIcon(forecast?.degradation_trend)}
              <span className="font-semibold text-slate-300">{forecast?.degradation_trend || 'Stable'}</span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-cyan-400">
              <span className="w-2.5 h-0.5 bg-cyan-400" /> Observed Frames
            </span>
            <span className="flex items-center gap-1.5 text-amber-400">
              <span className="w-2.5 h-0.5 border-t-2 border-dashed border-amber-400" /> Predicted (+0.17s, +0.50s, +1.00s)
            </span>
          </div>
        </div>

        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="visGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="frame" stroke="#475569" fontSize={10} tickLine={false} />
              <YAxis stroke="#475569" fontSize={10} domain={[0, 100]} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#020617',
                  border: '1px solid #1e293b',
                  borderRadius: '8px',
                  fontSize: '11px',
                }}
              />
              <Area
                type="monotone"
                dataKey="observed"
                stroke="#06b6d4"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#visGrad)"
                name="Observed"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 4, fill: '#f59e0b' }}
                name="Forecast"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
