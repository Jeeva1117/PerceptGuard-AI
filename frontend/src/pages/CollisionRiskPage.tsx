import React from 'react';
import {
  AlertTriangle,
  Zap,
  Clock,
  ShieldAlert,
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
    (hasData
      ? colRisk?.risk_state === 'CRITICAL'
        ? 'COLLISION OBSERVED'
        : colRisk?.risk_state === 'HIGH'
        ? 'POTENTIAL COLLISION RISK'
        : 'NO COLLISION RISK'
      : 'UNKNOWN');

  const collisionReason =
    results?.final_summary?.reason ||
    colRisk?.decision_reason ||
    colRisk?.explanation ||
    colRisk?.rationale ||
    (hasData
      ? 'Nominal clearance maintained; zero trajectory intersections with ego path.'
      : 'Awaiting video stream telemetry.');

  const relevantTimestamp =
    results?.final_summary?.event_timestamp != null
      ? `${results.final_summary.event_timestamp.toFixed(2)}s`
      : currentFrame?.timestamp_seconds != null
      ? `${currentFrame.timestamp_seconds.toFixed(2)}s`
      : '0.00s';

  const relevantObjectId =
    results?.final_summary?.primary_object ??
    (colRisk?.closest_track_id != null ? `#${colRisk.closest_track_id}` : 'None');

  const minTTC = colRisk?.min_ttc_seconds ?? null;
  const conflictPairs = colRisk?.conflicting_pairs || [];
  const closestTrack = colRisk?.closest_track_id ?? null;
  const closestDistM = colRisk?.closest_distance_m ?? null;
  const closestDistPx = colRisk?.closest_distance_px ?? null;
  const relativeSpeedKmh = colRisk?.relative_speed_kmh ?? null;
  const closingVelocityPxS = (colRisk as any)?.closing_velocity_px_s ?? (colRisk as any)?.closing_velocity ?? null;

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
    <div className="space-y-4 select-none">
      {/* Module Title Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400 shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-wide">
                Collision Risk
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                MODULE 4
              </span>
            </div>
            <p className="text-[11px] text-slate-400 truncate">
              Path conflict detection & Time-To-Collision (TTC)
            </p>
          </div>
        </div>

        <span
          className={`px-2.5 py-1 rounded-full text-[10px] font-extrabold font-mono border shrink-0 ${getResultBadge(
            collisionResult
          )}`}
        >
          {collisionResult}
        </span>
      </div>

      {/* Standalone Video Player (Only outside master workspace) */}
      {!isEmbedded && (
        <div className="space-y-3">
          <VideoCanvas highlightTrackId={closestTrack} />
          <PlaybackControls />
        </div>
      )}

      {/* Final 5-State Decision & Evidence Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
            Decision Taxonomy
          </span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold font-mono border ${getResultBadge(collisionResult)}`}>
            {collisionResult}
          </span>
        </div>

        {/* 3 Metric Pills: Risk State, Timestamp, Relevant Object */}
        <div className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Risk State</span>
            <strong className="text-white text-[11px] block mt-0.5 truncate" title={collisionResult}>
              {collisionResult}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Timestamp</span>
            <strong className="text-cyan-300 text-[11px] block mt-0.5">
              {relevantTimestamp}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Object ID</span>
            <strong className="text-amber-400 text-[11px] block mt-0.5">
              {relevantObjectId}
            </strong>
          </div>
        </div>

        {/* Evidence-Based Rationale */}
        <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            Evidence-Based Reason:
          </span>
          <p className="text-xs font-medium text-slate-200 leading-relaxed font-sans">
            {collisionReason}
          </p>
        </div>
      </div>

      {/* Spatio-Temporal Collision Metrics Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Gauge className="w-3.5 h-3.5 text-cyan-400" />
            Spatio-Temporal Metrics
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            Physical Corridor Gating
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          {/* Time-To-Collision (TTC) */}
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <div className="flex items-center justify-between text-slate-500 text-[10px] uppercase font-sans font-semibold">
              <span>TTC</span>
              <Clock className="w-3 h-3 text-cyan-400" />
            </div>
            <strong className="text-white text-base block mt-0.5">
              {minTTC !== null && !isNaN(minTTC) ? `${minTTC.toFixed(1)}s` : 'N/A (Diverging)'}
            </strong>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              {minTTC !== null && minTTC <= 3.5 ? 'TTC <= 3.5s Threshold' : 'Nominal clearance'}
            </span>
          </div>

          {/* Closing Velocity */}
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <div className="flex items-center justify-between text-slate-500 text-[10px] uppercase font-sans font-semibold">
              <span>Closing Speed</span>
              <Zap className="w-3 h-3 text-amber-400" />
            </div>
            <strong className="text-white text-base block mt-0.5">
              {relativeSpeedKmh !== null
                ? `${relativeSpeedKmh.toFixed(1)} km/h`
                : closingVelocityPxS !== null
                ? `${closingVelocityPxS.toFixed(0)} px/s`
                : 'Nominal'}
            </strong>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              {closingVelocityPxS != null && closingVelocityPxS > 5.0 ? 'Positive closing velocity' : 'Safe separation'}
            </span>
          </div>

          {/* Corridor Clearance */}
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Corridor Clearance</span>
            <strong className="text-white text-xs block mt-0.5">
              {closestDistM !== null ? `${closestDistM.toFixed(1)}m` : closestDistPx !== null ? `${closestDistPx.toFixed(0)} px` : 'Clear Corridor'}
            </strong>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              {closestTrack !== null ? `Closest: #${closestTrack}` : 'Zero adjacent hazards'}
            </span>
          </div>

          {/* Decision Category */}
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Risk Classification</span>
            <strong className="text-cyan-400 text-xs block mt-0.5 truncate" title={collisionResult}>
              {collisionResult}
            </strong>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              Discrete 5-State Engine
            </span>
          </div>
        </div>
      </div>

      {/* Spatio-Temporal Trajectory Conflicts */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            Predicted Path Conflicts
          </span>
          <span className="text-[10px] font-mono text-slate-400">
            {conflictPairs.length} conflict point(s)
          </span>
        </div>

        {conflictPairs.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800/80">
            No trajectory conflict points predicted within lookahead corridor.
          </div>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {conflictPairs.map((pair: any, idx: number) => (
              <div
                key={idx}
                className="bg-slate-950 p-3 rounded-xl border border-amber-500/30 space-y-1 text-xs font-mono"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-amber-300 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    Conflict: Track #{pair.track_a} ↔ Track #{pair.track_b}
                  </span>
                  <span className="text-[10px] font-bold text-slate-400 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800">
                    {pair.conflict_type || 'CONVERGING'}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                  <span>Delta Time: {pair.delta_time_seconds != null ? `${pair.delta_time_seconds.toFixed(2)}s` : 'N/A'}</span>
                  <span>Point: ({pair.conflict_point?.x?.toFixed(0) ?? 0}, {pair.conflict_point?.y?.toFixed(0) ?? 0})px</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
