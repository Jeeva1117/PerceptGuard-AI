import React, { useState } from 'react';
import {
  TrendingUp,
  Navigation,
  Compass,
  Activity,
  ArrowUpRight,
  Gauge,
  GitFork,
  Users,
  CheckCircle2,
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
  const [selectedHypothesisMode, setSelectedHypothesisMode] = useState<string>('PRIMARY_CONTINUATION');

  const trajectories = currentFrame?.trajectories || [];
  const selectedTrack =
    trajectories.find((t) => t.track_id === globalSelectedTrackId) ||
    trajectories[0] ||
    null;

  // Selected hypothesis
  const activeHypothesis =
    selectedTrack?.alternative_hypotheses?.find(
      (h) => h.mode === selectedHypothesisMode
    ) ||
    selectedTrack?.alternative_hypotheses?.[0] ||
    null;

  const interaction = selectedTrack?.interaction;

  // Horizon details map
  const horizonLabels: Record<'h1' | 'h3' | 'h5', { label: string; time: string; desc: string }> = {
    h1: { label: 'H1', time: '1.0s', desc: 'Short-range reactive motion' },
    h3: { label: 'H3', time: '3.0s', desc: 'Mid-range tactical maneuvers' },
    h5: { label: 'H5', time: '5.0s', desc: 'Long-range strategic trajectory' },
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
                Multi-horizon Kalman motion projection, multi-agent interaction modeling, and multimodal intention forecasting
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
              {(results.trajectory_summary.mean_ade_px !== undefined || results.trajectory_summary.overall_ade_px !== undefined) && (
                <>
                  <div className="w-px h-6 bg-slate-800" />
                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">ADE / FDE</span>
                    <strong className="text-amber-400 font-mono text-sm">
                      {(results.trajectory_summary.mean_ade_px ?? results.trajectory_summary.overall_ade_px)?.toFixed(1) ?? '--'} / {(results.trajectory_summary.mean_fde_px ?? results.trajectory_summary.overall_fde_px)?.toFixed(1) ?? '--'} px
                    </strong>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Grid: Visual Feed & Playback Scrubber */}
      <div className={isEmbedded ? "space-y-4" : "grid grid-cols-1 lg:grid-cols-12 gap-6"}>
        {/* Left 7 Cols: Video Canvas & Scrubber */}
        {!isEmbedded && (
          <div className="lg:col-span-7 space-y-4">
            <VideoCanvas highlightTrackId={selectedTrack?.track_id ?? null} />
            <PlaybackControls />
          </div>
        )}

        {/* Right 5 Cols: Horizon Selector & Kinematics Telemetry */}
        <div className={isEmbedded ? "space-y-4" : "lg:col-span-5 space-y-4"}>
          {/* Multimodal Hypotheses Card */}
          {selectedTrack && selectedTrack.alternative_hypotheses && selectedTrack.alternative_hypotheses.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <GitFork className="w-4 h-4 text-cyan-400" />
                  Multimodal Intent Hypotheses (Track #{selectedTrack.track_id})
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                  Bayesian Kinematic Probability
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {selectedTrack.alternative_hypotheses.map((hyp) => {
                  const isSelected = (activeHypothesis?.mode === hyp.mode) || (selectedHypothesisMode === hyp.mode);
                  const pct = Math.round(hyp.mode_probability * 100);
                  return (
                    <button
                      key={hyp.mode}
                      onClick={() => setSelectedHypothesisMode(hyp.mode)}
                      className={`p-3 rounded-xl text-left border transition-all ${
                        isSelected
                          ? 'bg-cyan-950/60 border-cyan-500 text-white shadow-md'
                          : 'bg-slate-950 border-slate-800 hover:border-slate-700 text-slate-400'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[11px] font-bold font-mono uppercase">
                          {hyp.mode.replace('_', ' ')}
                        </span>
                        <span
                          className={`text-xs font-bold font-mono ${
                            isSelected ? 'text-cyan-400' : 'text-slate-400'
                          }`}
                        >
                          {pct}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden mb-2">
                        <div
                          className="bg-cyan-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="block text-[10px] text-slate-400 line-clamp-2">
                        {hyp.intent_description}
                      </span>
                    </button>
                  );
                })}
              </div>

              {activeHypothesis && (
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 text-xs flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                    <span className="text-slate-300">
                      <strong>Active Mode:</strong> {activeHypothesis.intent_description}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-cyan-400 font-bold ml-2 shrink-0">
                    Confidence: {(activeHypothesis.mode_probability * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Horizon Selection Pill Group */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Navigation className="w-4 h-4 text-cyan-400" />
                Prediction Horizon
              </span>
              <span className="text-[10px] font-mono text-slate-500">
                Constant-Acceleration + Kalman
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {(['h1', 'h3', 'h5'] as const).map((id) => {
                const info = horizonLabels[id];
                const isSelected = selectedHorizon === id;
                return (
                  <button
                    key={id}
                    onClick={() => setSelectedHorizon(id)}
                    className={`p-2.5 rounded-xl text-center transition-all ${
                      isSelected
                        ? 'bg-cyan-600 text-white font-bold shadow-md shadow-cyan-950/50'
                        : 'bg-slate-950 hover:bg-slate-800 text-slate-400 border border-slate-800'
                    }`}
                  >
                    <span className="block text-sm font-mono font-bold">
                      {info.label} ({info.time})
                    </span>
                    <span className="block text-[9px] opacity-80 mt-0.5">{info.desc}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Focused Track Kinematic Card */}
          {selectedTrack ? (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 font-mono font-bold text-xs">
                    ID #{selectedTrack.track_id}
                  </span>
                  <span className="text-sm font-bold text-white capitalize">
                    {selectedTrack.class_name}
                  </span>
                  {selectedTrack.motion_state && (
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded-full font-bold border ${
                        selectedTrack.motion_state === 'CRUISING'
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : selectedTrack.motion_state === 'ACCELERATING'
                          ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30'
                          : selectedTrack.motion_state === 'DECELERATING'
                          ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                          : selectedTrack.motion_state.includes('TURNING')
                          ? 'bg-purple-500/10 text-purple-400 border-purple-500/30'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}
                    >
                      {selectedTrack.motion_state.replace('_', ' ')}
                    </span>
                  )}
                </div>
                <span className="text-xs font-mono text-slate-400">
                  History: {selectedTrack.history_length || selectedTrack.history?.length || 1} pts
                </span>
              </div>

              {/* Kinematics Telemetry Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block uppercase flex items-center gap-1">
                    <Gauge className="w-3 h-3 text-cyan-400" />
                    Estimated Speed
                  </span>
                  <strong className="text-xl font-mono text-cyan-400">
                    {selectedTrack.speed_kmh.toFixed(1)} <span className="text-xs text-slate-400">km/h</span>
                  </strong>
                </div>

                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block uppercase flex items-center gap-1">
                    <Compass className="w-3 h-3 text-amber-400" />
                    Heading Angle
                  </span>
                  <strong className="text-xl font-mono text-amber-400">
                    {selectedTrack.heading_degrees !== undefined
                      ? `${selectedTrack.heading_degrees.toFixed(0)}°`
                      : '--'}
                  </strong>
                </div>

                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block uppercase flex items-center gap-1">
                    <ArrowUpRight className="w-3 h-3 text-slate-400" />
                    Velocity (vx, vy)
                  </span>
                  <span className="text-xs font-mono text-white block mt-1">
                    ({selectedTrack.velocity.vx.toFixed(1)}, {selectedTrack.velocity.vy.toFixed(1)}) px/s
                  </span>
                </div>

                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block uppercase flex items-center gap-1">
                    <Activity className="w-3 h-3 text-slate-400" />
                    Acceleration (ax, ay)
                  </span>
                  <span className="text-xs font-mono text-white block mt-1">
                    ({selectedTrack.acceleration.ax.toFixed(1)}, {selectedTrack.acceleration.ay.toFixed(1)}) px/s²
                  </span>
                </div>
              </div>

              {/* Projected Coordinates for Selected Horizon */}
              {selectedTrack.predicted_horizons?.[selectedHorizon] && (
                <div className="bg-slate-950/70 rounded-xl p-3 border border-slate-800/80 text-xs space-y-1.5 font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">
                      Projected at {selectedHorizon.toUpperCase()} ({horizonLabels[selectedHorizon].time}):
                    </span>
                    <span className="text-cyan-300 font-bold">
                      X: {selectedTrack.predicted_horizons[selectedHorizon].x.toFixed(1)}, Y:{' '}
                      {selectedTrack.predicted_horizons[selectedHorizon].y.toFixed(1)} px
                    </span>
                  </div>
                  {selectedTrack.predicted_horizons[selectedHorizon].x_m !== undefined &&
                    selectedTrack.predicted_horizons[selectedHorizon].x_m !== null && (
                      <div className="flex items-center justify-between text-[11px] text-slate-500">
                        <span>Pinhole Ground Plane:</span>
                        <span>
                          {selectedTrack.predicted_horizons[selectedHorizon].x_m?.toFixed(1)}m lat,{' '}
                          {selectedTrack.predicted_horizons[selectedHorizon].y_m?.toFixed(1)}m fwd
                        </span>
                      </div>
                    )}
                  <div className="flex items-center justify-between text-[11px] text-amber-400/90">
                    <span>Uncertainty Radius (Covariance):</span>
                    <span>±{selectedTrack.predicted_horizons[selectedHorizon].uncertainty_radius.toFixed(1)} px</span>
                  </div>
                </div>
              )}

              {/* Multi-Agent Interaction Panel */}
              {interaction && (
                <div className="bg-slate-950 rounded-xl p-3.5 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-cyan-400" />
                      Interaction: {interaction.interaction_type.replace('_', ' ')}
                    </span>
                    {interaction.nearest_agent_id !== null && (
                      <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-900 border border-slate-800 rounded text-slate-300">
                        Target: #{interaction.nearest_agent_id}
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800/60">
                      <span className="text-[10px] text-slate-500 block">Separation Distance</span>
                      <span className="text-slate-200 font-bold">
                        {interaction.nearest_agent_distance_m !== null
                          ? `${interaction.nearest_agent_distance_m.toFixed(1)} m`
                          : interaction.nearest_agent_distance_px !== null
                          ? `${interaction.nearest_agent_distance_px.toFixed(0)} px`
                          : '--'}
                      </span>
                    </div>

                    <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800/60">
                      <span className="text-[10px] text-slate-500 block">Relative Speed</span>
                      <span
                        className={`font-bold ${
                          interaction.relative_speed_kmh !== null && interaction.relative_speed_kmh < 0
                            ? 'text-amber-400'
                            : 'text-slate-200'
                        }`}
                      >
                        {interaction.relative_speed_kmh !== null
                          ? `${interaction.relative_speed_kmh.toFixed(1)} km/h`
                          : '--'}
                      </span>
                    </div>
                  </div>

                  {interaction.time_to_headway_seconds !== null && (
                    <div className="flex items-center justify-between text-xs font-mono bg-slate-900/60 px-2.5 py-1.5 rounded-lg border border-slate-800/40">
                      <span className="text-slate-400">Time-to-Headway (TTH):</span>
                      <span
                        className={`font-bold ${
                          interaction.time_to_headway_seconds < 1.5
                            ? 'text-red-400'
                            : interaction.time_to_headway_seconds < 2.5
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                        }`}
                      >
                        {interaction.time_to_headway_seconds.toFixed(2)}s
                      </span>
                    </div>
                  )}

                  <p className="text-[11px] text-slate-400 italic">
                    {interaction.interaction_notes}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500 text-xs italic">
              No moving targets tracked in the current frame.
            </div>
          )}
        </div>
      </div>

      {/* Trajectories Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Compass className="w-4 h-4 text-cyan-400" />
            Active Track Telemetry & Multi-Horizon Forward Predictions
          </span>
          <span className="text-xs font-mono text-slate-500">
            Showing horizon: {selectedHorizon.toUpperCase()} ({horizonLabels[selectedHorizon].time})
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Track ID</th>
                <th className="py-2.5 px-3">Class</th>
                <th className="py-2.5 px-3">Motion State</th>
                <th className="py-2.5 px-3">Current Centroid</th>
                <th className="py-2.5 px-3">Speed (km/h)</th>
                <th className="py-2.5 px-3">Velocity (vx, vy)</th>
                <th className="py-2.5 px-3">Pred. Coords ({selectedHorizon.toUpperCase()})</th>
                <th className="py-2.5 px-3">Uncertainty</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {trajectories.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-6 text-center text-slate-500 italic">
                    No trajectory telemetry available for this frame.
                  </td>
                </tr>
              ) : (
                trajectories.map((trk) => {
                  const pred = trk.predicted_horizons?.[selectedHorizon];
                  const isSelected = selectedTrack?.track_id === trk.track_id;

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
                        <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                          {trk.motion_state ? trk.motion_state.replace('_', ' ') : 'CRUISING'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        ({trk.center[0].toFixed(0)}, {trk.center[1].toFixed(0)})
                      </td>
                      <td className="py-2.5 px-3 font-bold text-white">
                        {trk.speed_kmh.toFixed(1)} km/h
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        ({trk.velocity.vx.toFixed(0)}, {trk.velocity.vy.toFixed(0)})
                      </td>
                      <td className="py-2.5 px-3 text-cyan-300">
                        {pred ? `(${pred.x.toFixed(0)}, ${pred.y.toFixed(0)})` : '--'}
                      </td>
                      <td className="py-2.5 px-3 text-amber-400">
                        {pred ? `±${pred.uncertainty_radius.toFixed(0)} px` : '--'}
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
                          Select
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
