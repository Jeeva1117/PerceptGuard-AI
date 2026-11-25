import React, { useState } from 'react';
import {
  TrendingUp,
  Compass,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export interface TrajectoryPredictionPageProps {
  isEmbedded?: boolean;
}

export const TrajectoryPredictionPage: React.FC<TrajectoryPredictionPageProps> = ({ isEmbedded = false }) => {
  const { results, currentFrame, selectedTrackId: globalSelectedTrackId, setSelectedTrackId } = useJob();
  const [selectedHorizon, setSelectedHorizon] = useState<'h1' | 'h3' | 'h5'>('h3');

  const trajectories = currentFrame?.trajectories || [];
  const selectedTrack =
    trajectories.find((t) => t.track_id === globalSelectedTrackId) ||
    trajectories[0] ||
    null;

  // Horizon labels
  const horizonLabels: Record<'h1' | 'h3' | 'h5', { label: string; time: string }> = {
    h1: { label: 'H1', time: '1.0s' },
    h3: { label: 'H3', time: '3.0s' },
    h5: { label: 'H5', time: '5.0s' },
  };

  // Helper to resolve tracking status
  const getTrackingStatus = (trk: any): { status: 'Confirmed' | 'Lost' | 'Insufficient Data'; badgeClass: string } => {
    const historyLen = trk.history_length || trk.history?.length || 0;
    if (trk.prediction_status === 'INSUFFICIENT_HISTORY' || historyLen < 3) {
      return {
        status: 'Insufficient Data',
        badgeClass: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
      };
    }
    if (trk.track_lifecycle === 'LOST') {
      return {
        status: 'Lost',
        badgeClass: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      };
    }
    return {
      status: 'Confirmed',
      badgeClass: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    };
  };

  // Helper for movement direction
  const getMovementDirection = (heading?: number, vx?: number, vy?: number) => {
    if (heading !== undefined && !isNaN(heading)) {
      if (heading >= 315 || heading < 45) return 'North (Forward)';
      if (heading >= 45 && heading < 135) return 'East (Right)';
      if (heading >= 135 && heading < 225) return 'South (Approaching)';
      if (heading >= 225 && heading < 315) return 'West (Left)';
    }
    if (vx !== undefined && vy !== undefined) {
      const dirX = Math.abs(vx) > 5 ? (vx > 0 ? 'Right' : 'Left') : '';
      const dirY = Math.abs(vy) > 5 ? (vy > 0 ? 'Approaching' : 'Forward') : '';
      return [dirY, dirX].filter(Boolean).join(' ') || 'Stationary';
    }
    return 'Stationary';
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Module Scope */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-cyan-500/10 rounded-xl text-cyan-400">
              <TrendingUp className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Object & Trajectory Prediction
              </h2>
              <p className="text-xs text-slate-400">
                Kinematic forward trajectory projection and path conflict anticipation
              </p>
            </div>
          </div>
        </div>

        {/* Global Stream Stats */}
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
              currentFrame?.collision_risk?.conflicting_pairs && currentFrame.collision_risk.conflicting_pairs.length > 0
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                : trajectories.length > 0
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-slate-800 text-slate-400 border-slate-700'
            }`}
          >
            {currentFrame?.collision_risk?.conflicting_pairs && currentFrame.collision_risk.conflicting_pairs.length > 0
              ? 'POTENTIAL CONFLICT'
              : trajectories.length > 0
              ? 'NORMAL PATH'
              : 'INSUFFICIENT HISTORY'}
          </span>
          {results?.trajectory_summary && (
            <div className="flex items-center gap-3 text-xs bg-slate-900 border border-slate-800 rounded-xl px-4 py-2">
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Mean Speed</span>
                <strong className="text-slate-200 font-mono text-sm">
                  {results.trajectory_summary.mean_speed_kmh?.toFixed(1) ?? '0.0'} km/h
                </strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Peak Speed</span>
                <strong className="text-cyan-400 font-mono text-sm">
                  {results.trajectory_summary.max_speed_kmh?.toFixed(1) ?? '0.0'} km/h
                </strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Observed Tracks</span>
                <strong className="text-emerald-400 font-mono text-sm">
                  {results.trajectory_summary.total_observed_trajectories ?? results.trajectory_summary.total_predicted_tracks ?? 0}
                </strong>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Video Canvas & Controls */}
      {!isEmbedded && (
        <div className="space-y-4">
          <VideoCanvas highlightTrackId={selectedTrack?.track_id ?? null} />
          <PlaybackControls />
        </div>
      )}

      {/* Mandatory Concise Trajectory Prediction Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <span className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            TRAJECTORY PREDICTION — Active Track Telemetry
          </span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase font-bold text-slate-500">Horizon:</span>
            {(['h1', 'h3', 'h5'] as const).map((h) => (
              <button
                key={h}
                onClick={() => setSelectedHorizon(h)}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                  selectedHorizon === h
                    ? 'bg-cyan-600 text-white'
                    : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                {horizonLabels[h].label} ({horizonLabels[h].time})
              </button>
            ))}
          </div>
        </div>

        {/* Per Active Track Summary Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Object ID</th>
                <th className="py-2.5 px-3">Class</th>
                <th className="py-2.5 px-3">Tracking</th>
                <th className="py-2.5 px-3">Movement (Speed & Direction)</th>
                <th className="py-2.5 px-3">Prediction ({selectedHorizon.toUpperCase()})</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {trajectories.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-slate-500 italic">
                    No trajectory telemetry available for this frame.
                  </td>
                </tr>
              ) : (
                trajectories.map((trk) => {
                  const tracking = getTrackingStatus(trk);
                  const pred = trk.predicted_horizons?.[selectedHorizon];
                  const isSelected = selectedTrack?.track_id === trk.track_id;
                  const direction = getMovementDirection(trk.heading_degrees, trk.velocity?.vx, trk.velocity?.vy);

                  const predictionText =
                    tracking.status === 'Insufficient Data' || !pred
                      ? 'Insufficient Data'
                      : `(${pred.x.toFixed(0)}, ${pred.y.toFixed(0)}) px ±${pred.uncertainty_radius.toFixed(0)}px`;

                  return (
                    <tr
                      key={trk.track_id}
                      onClick={() => setSelectedTrackId(trk.track_id)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-cyan-950/40 text-cyan-300'
                          : 'hover:bg-slate-800/40 text-slate-300'
                      }`}
                    >
                      <td className="py-2.5 px-3 font-bold text-white flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-cyan-400" />#{trk.track_id}
                      </td>
                      <td className="py-2.5 px-3 capitalize font-sans">{trk.class_name}</td>
                      <td className="py-2.5 px-3 font-sans">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${tracking.badgeClass}`}>
                          {tracking.status}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">
                        <span className="font-bold text-white mr-1.5">{trk.speed_kmh.toFixed(1)} km/h</span>
                        <span className="text-[11px] text-slate-400 font-sans">({direction})</span>
                      </td>
                      <td className="py-2.5 px-3">
                        <span
                          className={
                            predictionText === 'Insufficient Data'
                              ? 'text-amber-400 italic'
                              : 'text-cyan-300 font-bold'
                          }
                        >
                          {predictionText}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedTrackId(trk.track_id);
                          }}
                          className={`px-2.5 py-1 rounded text-[11px] font-sans font-medium transition-colors ${
                            isSelected
                              ? 'bg-cyan-600 text-white'
                              : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                          }`}
                        >
                          {isSelected ? 'Focused' : 'Inspect'}
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Focused Track Details (if track selected) */}
        {selectedTrack && (
          <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 mt-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Compass className="w-4 h-4 text-cyan-400" />
                Focused Kinematics: Track #{selectedTrack.track_id} ({selectedTrack.class_name})
              </span>
              <span className="text-xs font-mono text-cyan-400">
                History: {selectedTrack.history_length || selectedTrack.history?.length || 1} frames
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Speed</span>
                <strong className="text-white text-sm">{selectedTrack.speed_kmh.toFixed(1)} km/h</strong>
              </div>
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Heading</span>
                <strong className="text-amber-400 text-sm">
                  {selectedTrack.heading_degrees !== undefined ? `${selectedTrack.heading_degrees.toFixed(0)}°` : '--'}
                </strong>
              </div>
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Velocity (vx, vy)</span>
                <strong className="text-slate-300 text-xs">
                  ({selectedTrack.velocity.vx.toFixed(0)}, {selectedTrack.velocity.vy.toFixed(0)}) px/s
                </strong>
              </div>
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 uppercase block">Acceleration (ax, ay)</span>
                <strong className="text-slate-300 text-xs">
                  ({selectedTrack.acceleration.ax.toFixed(1)}, {selectedTrack.acceleration.ay.toFixed(1)}) px/s²
                </strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
