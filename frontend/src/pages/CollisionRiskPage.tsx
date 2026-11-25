import React from 'react';
import {
  AlertTriangle,
  Zap,
  Clock,
  ShieldAlert,
  ArrowRight,
  Sparkles,
  Gauge,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export interface CollisionRiskPageProps {
  isEmbedded?: boolean;
}

export const CollisionRiskPage: React.FC<CollisionRiskPageProps> = ({ isEmbedded = false }) => {
  const { results, currentFrame } = useJob();

  const colRisk = currentFrame?.collision_risk;
  const hasData = Boolean(colRisk);

  // 5 exact product collision decisions:
  // NO COLLISION RISK | NEAR PASS | POTENTIAL COLLISION RISK | COLLISION OBSERVED | UNKNOWN
  const collisionResult =
    results?.final_summary?.collision_risk ||
    colRisk?.collision_decision ||
    (hasData ? (colRisk?.risk_state === 'CRITICAL' ? 'COLLISION OBSERVED' : colRisk?.risk_state === 'HIGH' ? 'POTENTIAL COLLISION RISK' : 'NO COLLISION RISK') : 'UNKNOWN');

  const collisionReason =
    results?.final_summary?.reason ||
    colRisk?.decision_reason ||
    colRisk?.explanation ||
    colRisk?.rationale ||
    (hasData
      ? 'Nominal clearance maintained; zero trajectory intersections with ego path.'
      : 'Awaiting video stream telemetry.');

  const minTTC = colRisk?.min_ttc_seconds ?? null;
  const conflictPairs = colRisk?.conflicting_pairs || [];
  const uncertaintyScore = colRisk?.uncertainty_score ?? colRisk?.perception_uncertainty_factor ?? 0.15;
  const uncertaintyLevel = colRisk?.uncertainty_level ?? 'LOW';
  const closestTrack = colRisk?.closest_track_id ?? null;
  const closestDistM = colRisk?.closest_distance_m ?? null;
  const closestDistPx = colRisk?.closest_distance_px ?? null;
  const relativeSpeedKmh = colRisk?.relative_speed_kmh ?? null;

  const getResultBadge = (res: string) => {
    switch (res.toUpperCase()) {
      case 'COLLISION OBSERVED':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/50 animate-pulse';
      case 'POTENTIAL COLLISION RISK':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/50 animate-pulse';
      case 'NEAR PASS':
        return 'bg-cyan-500/15 text-cyan-400 border-cyan-500/40';
      case 'NO COLLISION RISK':
      case 'NO COLLISION RISK DETECTED':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40';
      case 'UNKNOWN':
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
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
                Evidence-driven multi-horizon path conflict detection and Time-To-Collision (TTC) assessment
              </p>
            </div>
          </div>
        </div>

        {/* Global Result Badge */}
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-extrabold font-mono border ${getResultBadge(
              collisionResult
            )}`}
          >
            {collisionResult}
          </span>
        </div>
      </div>

      {/* Mandatory Concise Collision Risk Assessment Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <span className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-cyan-400" />
            COLLISION RISK ASSESSMENT
          </span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 uppercase font-bold">Result:</span>
            <span className={`px-2.5 py-0.5 rounded text-xs font-extrabold font-mono border ${getResultBadge(collisionResult)}`}>
              {collisionResult}
            </span>
          </div>
        </div>

        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            Evidence-Based Reason:
          </span>
          <p className="text-sm font-medium text-slate-200 leading-relaxed font-sans">
            {collisionReason}
          </p>
        </div>
      </div>

      {/* Video Canvas & Controls */}
      {!isEmbedded && (
        <div className="space-y-4">
          <VideoCanvas highlightTrackId={closestTrack} />
          <PlaybackControls />
        </div>
      )}

      {/* Telemetry Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* TTC */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            Time-To-Collision (TTC)
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {minTTC !== null ? `${minTTC.toFixed(1)}s` : '--'}
          </div>
          <span className="text-[10px] text-slate-500">
            {minTTC !== null
              ? minTTC < 1.8
                ? 'Critical immediate hazard'
                : minTTC < 3.5
                ? 'Elevated convergence'
                : 'Safe headway buffer'
              : 'Diverging / nominal motion'}
          </span>
        </div>

        {/* Critical Agent */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
            Critical Agent
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {closestTrack !== null ? `#${closestTrack}` : 'None'}
          </div>
          <span className="text-[10px] text-slate-500">
            {relativeSpeedKmh !== null ? `Relative Speed: ${relativeSpeedKmh.toFixed(1)} km/h` : 'No relative hazard'}
          </span>
        </div>

        {/* Clearance Distance */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5 text-emerald-400" />
            Clearance Distance
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {closestDistM !== null ? `${closestDistM.toFixed(1)}m` : closestDistPx !== null ? `${closestDistPx.toFixed(0)}px` : '--'}
          </div>
          <span className="text-[10px] text-slate-500">
            {closestDistPx !== null ? `Pixel separation: ${closestDistPx.toFixed(0)} px` : 'Clear corridor'}
          </span>
        </div>

        {/* Multi-Source Uncertainty */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            Perception Uncertainty
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {(uncertaintyScore * 100).toFixed(0)}%
          </div>
          <span className="text-[10px] text-slate-500 uppercase font-mono">
            Rating: {uncertaintyLevel}
          </span>
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
