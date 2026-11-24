import React from 'react';
import {
  AlertTriangle,
  Zap,
  Clock,
  ShieldAlert,
  ArrowRight,
  Sparkles,
  Gauge,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export const CollisionRiskPage: React.FC = () => {
  const { results, currentFrame } = useJob();

  const colRisk = currentFrame?.collision_risk;
  const hasData = Boolean(colRisk);
  const minTTC = colRisk?.min_ttc_seconds ?? null;
  const riskState = colRisk?.risk_state ?? (hasData ? 'LOW' : 'UNKNOWN');
  const riskScore = colRisk?.risk_score ?? colRisk?.fused_risk_score ?? null;
  const explanation = colRisk?.explanation || colRisk?.rationale || (hasData ? 'Nominal clearance maintained; zero collision trajectory intersections.' : 'No active stream telemetry.');
  const conflictPairs = colRisk?.conflicting_pairs || [];
  const uncertaintyScore = colRisk?.uncertainty_score ?? colRisk?.perception_uncertainty_factor ?? 0.15;
  const uncertaintyLevel = colRisk?.uncertainty_level ?? 'LOW';
  const closestTrack = colRisk?.closest_track_id ?? null;
  const closestDistM = colRisk?.closest_distance_m ?? null;
  const closestDistPx = colRisk?.closest_distance_px ?? null;
  const relativeSpeedKmh = colRisk?.relative_speed_kmh ?? null;
  const contributingFactors = colRisk?.primary_contributing_factors || [];

  const getRiskStateBadge = (state?: string | null) => {
    switch ((state || '').toUpperCase()) {
      case 'CRITICAL':
        return 'bg-red-500 text-white border-red-400 animate-pulse';
      case 'HIGH':
        return 'bg-orange-500 text-white border-orange-400';
      case 'MODERATE':
        return 'bg-amber-500 text-slate-950 border-amber-400';
      case 'LOW':
        return 'bg-emerald-500 text-slate-950 border-emerald-400';
      case 'UNKNOWN':
        return 'bg-purple-950 text-purple-300 border-purple-500/80';
      default:
        return 'bg-slate-700 text-slate-200 border-slate-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Module Scope */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-cyan-500/10 rounded-xl text-cyan-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Collision & Road Risk Assessment
              </h2>
              <p className="text-xs text-slate-400">
                Multi-factor trajectory convergence, mathematically validated Time-To-Collision (TTC), and uncertainty fusion
              </p>
            </div>
          </div>
        </div>

        {/* Stream Collision Summary */}
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
              riskState === 'UNKNOWN'
                ? 'bg-purple-950 text-purple-300 border-purple-500/80'
                : riskState === 'LOW'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-red-500/10 text-red-400 border-red-500/30'
            }`}
          >
            {riskState === 'UNKNOWN'
              ? 'UNKNOWN'
              : riskState === 'LOW'
              ? 'NO COLLISION RISK DETECTED'
              : 'COLLISION RISK DETECTED'}
          </span>
          {results?.collision_summary && (
            <div className="flex items-center gap-3 text-xs bg-slate-900 border border-slate-800 rounded-xl px-4 py-2">
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Min Stream TTC</span>
                <strong className="text-red-400 font-mono text-sm">
                  {results.collision_summary.min_ttc_seconds !== null
                    ? `${results.collision_summary.min_ttc_seconds}s`
                    : 'N/A'}
                </strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Critical Frames</span>
                <strong className="text-red-400 font-mono text-sm">
                  {results.collision_summary.critical_risk_frames}
                </strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">High Risk Frames</span>
                <strong className="text-orange-400 font-mono text-sm">
                  {results.collision_summary.high_risk_frames}
                </strong>
              </div>
              {results.collision_summary.total_conflicts_detected !== undefined && (
                <>
                  <div className="w-px h-6 bg-slate-800" />
                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">Conflicts</span>
                    <strong className="text-amber-400 font-mono text-sm">
                      {results.collision_summary.total_conflicts_detected}
                    </strong>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Grid: Visual Feed & Playback Scrubber */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Video Canvas & Scrubber */}
        <div className="lg:col-span-7 space-y-4">
          <VideoCanvas highlightTrackId={closestTrack} />
          <PlaybackControls />

          {/* Explainable Contributing Factors Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Info className="w-4 h-4 text-cyan-400" />
                Explainable Contributing Factors & Rationale
              </span>
              <span className="text-[10px] font-mono text-slate-500">
                Transparent Multi-Evidence Safety Engine
              </span>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800 text-xs text-slate-200">
              <p className="leading-relaxed font-sans">{explanation}</p>
            </div>

            {contributingFactors.length > 0 && (
              <div className="space-y-1.5 pt-1">
                <span className="text-[10px] font-semibold uppercase text-slate-500 block">
                  Contributing Factors:
                </span>
                <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  {contributingFactors.map((factor, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-2 bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 text-slate-300"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                      <span className="text-[11px] leading-tight">{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Right 5 Cols: TTC & Fused Collision Risk Telemetry */}
        <div className="lg:col-span-5 space-y-4">
          {/* TTC Primary Hazard Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Clock className="w-4 h-4 text-cyan-400" />
                Time-To-Collision (TTC)
              </span>
              <span
                className={`px-3 py-1 rounded-full text-xs font-extrabold font-mono tracking-wider border ${getRiskStateBadge(
                  riskState
                )}`}
              >
                {riskState} RISK
              </span>
            </div>

            {/* TTC Gauge and Value */}
            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
              <div className="flex items-baseline justify-between mb-1">
                <div>
                  <span className="text-4xl font-extrabold text-white font-mono tracking-tight">
                    {minTTC !== null ? `${minTTC.toFixed(1)}s` : '--'}
                  </span>
                  <span className="text-xs text-slate-400 block mt-1">
                    {minTTC !== null ? (
                      minTTC < 1.8 ? (
                        <span className="text-red-400 font-bold flex items-center gap-1">
                          <ShieldAlert className="w-3.5 h-3.5" /> Immediate Collision Hazard
                        </span>
                      ) : minTTC < 3.5 ? (
                        <span className="text-orange-400 font-semibold flex items-center gap-1">
                          <AlertTriangle className="w-3.5 h-3.5" /> Elevated Trajectory Convergence
                        </span>
                      ) : (
                        <span className="text-emerald-400 font-medium">Safe Headway Buffer</span>
                      )
                    ) : (
                      <span className="text-slate-500 italic">
                        TTC not applicable (diverging or lateral crossing motion)
                      </span>
                    )}
                  </span>
                </div>

                {riskScore !== null ? (
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 uppercase block">Risk Index</span>
                    <strong className="text-2xl font-mono text-cyan-400">
                      {(riskScore * 100).toFixed(0)}%
                    </strong>
                  </div>
                ) : (
                  <div className="text-right">
                    <span className="text-[10px] text-purple-400 uppercase block">Risk Score</span>
                    <strong className="text-sm font-mono text-purple-300">WITHHELD</strong>
                  </div>
                )}
              </div>
            </div>

            {/* Separation & Critical Track Grid */}
            <div className="grid grid-cols-2 gap-2.5">
              <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Critical Agent</span>
                <strong className="text-sm font-mono text-cyan-400">
                  {closestTrack !== null ? `Track #${closestTrack}` : 'None'}
                </strong>
              </div>

              <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Relative Speed</span>
                <strong className="text-sm font-mono text-white flex items-center gap-1">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                  {relativeSpeedKmh !== null ? `${relativeSpeedKmh.toFixed(1)} km/h` : '--'}
                </strong>
              </div>

              <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Separation (Pixels)</span>
                <strong className="text-sm font-mono text-slate-200">
                  {closestDistPx !== null ? `${closestDistPx.toFixed(0)} px` : '--'}
                </strong>
              </div>

              <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Ground Metric (Pinhole)</span>
                <strong className="text-sm font-mono text-emerald-400">
                  {closestDistM !== null ? `${closestDistM.toFixed(1)} m` : '--'}
                </strong>
              </div>
            </div>
          </div>

          {/* Perception & Environmental Uncertainty Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Multi-Source Uncertainty Rating
              </span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase font-bold ${
                  uncertaintyLevel === 'CRITICAL'
                    ? 'bg-red-950 text-red-400 border-red-800'
                    : uncertaintyLevel === 'HIGH'
                    ? 'bg-amber-950 text-amber-400 border-amber-800'
                    : 'bg-emerald-950 text-emerald-400 border-emerald-800'
                }`}
              >
                {uncertaintyLevel}
              </span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center text-slate-300">
                <span>Fused Uncertainty Factor:</span>
                <strong className="font-mono text-amber-400">
                  {(uncertaintyScore * 100).toFixed(1)}%
                </strong>
              </div>
              <div className="flex justify-between items-center text-slate-300">
                <span>Camera Visibility Score:</span>
                <strong className="font-mono text-cyan-400">
                  {colRisk?.camera_visibility_score !== undefined
                    ? `${colRisk.camera_visibility_score.toFixed(0)}%`
                    : '--'}
                </strong>
              </div>
              <div className="flex justify-between items-center text-slate-300">
                <span>Detection Reliability:</span>
                <strong className="font-mono text-emerald-400">
                  {colRisk?.detection_reliability !== undefined
                    ? `${(colRisk.detection_reliability * 100).toFixed(0)}%`
                    : '--'}
                </strong>
              </div>
            </div>

            <p className="text-[11px] text-slate-500 pt-1 leading-relaxed">
              When sensor visibility or tracking maturity falls below defensible thresholds, PerceptGuard AI automatically withholds confident scores and assigns an UNKNOWN state to avoid misleading drivers.
            </p>
          </div>
        </div>
      </div>

      {/* Trajectory Conflicts List */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Zap className="w-4 h-4 text-amber-400" />
            Synchronous Spatio-Temporal Trajectory Conflicts
          </span>
          <span className="text-xs font-mono text-slate-500">
            {conflictPairs.length} conflict(s) detected
          </span>
        </div>

        {conflictPairs.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800">
            No trajectory conflict points predicted within the multi-horizon lookahead envelope.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {conflictPairs.map((pair, idx) => (
              <div
                key={idx}
                className="bg-slate-950 p-4 rounded-xl border border-amber-500/30 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-white">
                    <span>Track #{pair.track_a} ({pair.class_a})</span>
                    <ArrowRight className="w-3.5 h-3.5 text-amber-400" />
                    <span>Track #{pair.track_b} ({pair.class_b})</span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono text-[10px] font-bold uppercase">
                    {pair.conflict_type ? pair.conflict_type.replace('_', ' ') : 'CONVERGENCE'}
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-1">
                  <span>
                    Horizon: <strong className="text-cyan-300">{pair.horizon_label || pair.horizon || 'H3'}</strong>{' '}
                    ({pair.time_to_conflict_seconds !== undefined ? `${pair.time_to_conflict_seconds}s` : '--'})
                  </span>
                  <span className="text-amber-400 font-bold">
                    Δ {pair.separation_pixels} px
                  </span>
                </div>

                <p className="text-[11px] text-slate-500">
                  Predicted convergence centroid: ({pair.conflict_point[0]}, {pair.conflict_point[1]})
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
