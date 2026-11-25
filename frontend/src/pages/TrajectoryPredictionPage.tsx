import React, { useState } from 'react';
import {
  TrendingUp,
  Compass,
  Zap,
  Target,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export interface TrajectoryPredictionPageProps {
  isEmbedded?: boolean;
}

export const TrajectoryPredictionPage: React.FC<TrajectoryPredictionPageProps> = ({ isEmbedded = false }) => {
  const { currentFrame, selectedTrackId, setSelectedTrackId } = useJob();
  const [selectedHorizon, setSelectedHorizon] = useState<'h1' | 'h3' | 'h5'>('h3');

  const trajectories = currentFrame?.trajectories || [];
  const activeSelectedTrack =
    trajectories.find((t) => t.track_id === selectedTrackId) ||
    trajectories[0] ||
    null;

  const horizonLabels: Record<'h1' | 'h3' | 'h5', { label: string; time: string }> = {
    h1: { label: 'H1', time: '1.0s' },
    h3: { label: 'H3', time: '3.0s' },
    h5: { label: 'H5', time: '5.0s' },
  };

  const getTrackingStatus = (trk: any): { status: 'CONFIRMED' | 'LOST' | 'INSUFFICIENT DATA'; badgeClass: string } => {
    const historyLen = trk.history_length || trk.history?.length || 0;
    if (trk.prediction_status === 'INSUFFICIENT_HISTORY' || historyLen < 3) {
      return {
        status: 'INSUFFICIENT DATA',
        badgeClass: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
      };
    }
    if (trk.track_lifecycle === 'LOST') {
      return {
        status: 'LOST',
        badgeClass: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      };
    }
    return {
      status: 'CONFIRMED',
      badgeClass: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    };
  };

  const getMovementDirection = (heading?: number, vx?: number, vy?: number) => {
    if (heading !== undefined && !isNaN(heading)) {
      if (heading >= 315 || heading < 45) return 'Forward (North)';
      if (heading >= 45 && heading < 135) return 'Right (East)';
      if (heading >= 135 && heading < 225) return 'Approaching (South)';
      if (heading >= 225 && heading < 315) return 'Left (West)';
    }
    if (vx !== undefined && vy !== undefined) {
      const dirX = Math.abs(vx) > 5 ? (vx > 0 ? 'Right' : 'Left') : '';
      const dirY = Math.abs(vy) > 5 ? (vy > 0 ? 'Approaching' : 'Forward') : '';
      return [dirY, dirX].filter(Boolean).join(' ') || 'Stationary';
    }
    return 'Stationary';
  };

  return (
    <div className="space-y-4 select-none">
      {/* Module Title Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400 shrink-0">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-wide">
                Trajectory Prediction
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                MODULE 3
              </span>
            </div>
            <p className="text-[11px] text-slate-400 truncate">
              Multi-horizon path extrapolation & kinematics
            </p>
          </div>
        </div>

        <span className="px-2.5 py-1 rounded-full text-[10px] font-bold font-mono bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shrink-0">
          {trajectories.length} ACTIVE PATHS
        </span>
      </div>

      {/* Standalone Video Player (Only outside master workspace) */}
      {!isEmbedded && (
        <div className="space-y-3">
          <VideoCanvas highlightTrackId={activeSelectedTrack?.track_id ?? null} />
          <PlaybackControls />
        </div>
      )}

      {/* Lookahead Horizon Selector */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Compass className="w-3.5 h-3.5 text-cyan-400" />
            Prediction Horizon
          </span>
          <div className="flex items-center gap-1.5">
            {(['h1', 'h3', 'h5'] as const).map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setSelectedHorizon(h)}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                  selectedHorizon === h
                    ? 'bg-cyan-600 text-white shadow-md'
                    : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                {horizonLabels[h].label} ({horizonLabels[h].time})
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Focused Road User Kinematics & Prediction Card */}
      {activeSelectedTrack ? (
        (() => {
          const tracking = getTrackingStatus(activeSelectedTrack);
          const pred = activeSelectedTrack.predicted_horizons?.[selectedHorizon];
          const direction = getMovementDirection(
            activeSelectedTrack.heading_degrees,
            activeSelectedTrack.velocity?.vx,
            activeSelectedTrack.velocity?.vy
          );

          return (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-bold text-white font-mono">
                    Track #{activeSelectedTrack.track_id}
                  </span>
                  <span className="text-xs capitalize font-sans text-slate-300">
                    ({activeSelectedTrack.class_name})
                  </span>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${tracking.badgeClass}`}>
                  {tracking.status}
                </span>
              </div>

              {/* Kinematic telemetry */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Speed</span>
                  <strong className="text-white text-xs block mt-0.5">
                    {activeSelectedTrack.speed_kmh != null
                      ? `${activeSelectedTrack.speed_kmh.toFixed(1)} km/h`
                      : 'NOT AVAILABLE'}
                  </strong>
                </div>

                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Heading / Vector</span>
                  <strong className="text-white text-xs block mt-0.5 truncate" title={direction}>
                    {direction}
                  </strong>
                </div>

                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Track History</span>
                  <strong className="text-cyan-400 text-xs block mt-0.5">
                    {activeSelectedTrack.history_length ?? activeSelectedTrack.history?.length ?? 0} frames
                  </strong>
                </div>

                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">
                    {selectedHorizon.toUpperCase()} Lookahead
                  </span>
                  <strong className="text-cyan-300 text-xs block mt-0.5">
                    +{horizonLabels[selectedHorizon].time}
                  </strong>
                </div>
              </div>

              {/* Predicted Waypoint Projection */}
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
                  Projected Position ({selectedHorizon.toUpperCase()}):
                </span>
                {tracking.status === 'INSUFFICIENT DATA' || !pred ? (
                  <p className="text-xs text-amber-400 font-mono italic">
                    INSUFFICIENT DATA — Awaiting mature track history (&gt;= 3 frames).
                  </p>
                ) : (
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-200">
                      Pixel Coords: ({pred.x.toFixed(0)}, {pred.y.toFixed(0)}) px
                    </span>
                    <span className="text-cyan-400 text-[11px]">
                      ±{pred.uncertainty_radius.toFixed(0)}px radius
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })()
      ) : (
        <div className="p-6 text-center text-slate-500 text-xs italic bg-slate-900 border border-slate-800 rounded-2xl">
          No tracked road users on current frame.
        </div>
      )}

      {/* Active Trajectories List (Click to Focus) */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            All Active Paths ({trajectories.length})
          </span>
          <span className="text-[10px] font-mono text-slate-400">
            Click to Highlight
          </span>
        </div>

        {trajectories.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800/80">
            No trajectory vectors available on current frame.
          </div>
        ) : (
          <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
            {trajectories.map((trk) => {
              const tracking = getTrackingStatus(trk);
              const isSelected = selectedTrackId === trk.track_id;
              return (
                <div
                  key={trk.track_id}
                  onClick={() => setSelectedTrackId(isSelected ? null : trk.track_id)}
                  className={`p-2.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-2 ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500 ring-1 ring-cyan-500 shadow-md'
                      : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    <strong className="text-white font-mono text-xs">#{trk.track_id}</strong>
                    <span className="capitalize font-sans text-slate-300 text-xs">
                      {trk.class_name}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold border ${tracking.badgeClass}`}>
                      {tracking.status === 'INSUFFICIENT DATA' ? 'INSUFFICIENT' : tracking.status}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-slate-300">{trk.speed_kmh.toFixed(0)} km/h</span>
                    <button
                      type="button"
                      className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors ${
                        isSelected
                          ? 'bg-cyan-500 text-slate-950'
                          : 'bg-slate-800 text-slate-300 hover:text-white'
                      }`}
                    >
                      {isSelected ? 'Focused' : 'Focus'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
