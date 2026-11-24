import React, { useState } from 'react';
import {
  Crosshair,
  Car,
  Truck,
  Bus,
  Bike,
  User,
  Layers,
  Filter,
  Sparkles,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';
import { FilterBar, DEFAULT_FILTER_STATE, type FilterState } from '../components/FilterBar';

export const ObjectDetectionPage: React.FC = () => {
  const { results, currentFrame, currentFrameIndex } = useJob();
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    ...DEFAULT_FILTER_STATE,
    minConfidence: 0.25,
  });

  const rawTracks = currentFrame?.tracks || [];
  const tracks = rawTracks.filter((t) => {
    if (filters.minConfidence > 0 && t.confidence < filters.minConfidence) return false;
    if (filters.objectClass !== 'ALL' && t.class_name.toLowerCase() !== filters.objectClass.toLowerCase()) return false;
    if (filters.trackId.trim() !== '' && String(t.track_id) !== filters.trackId.trim()) return false;
    return true;
  });
  const relScores = currentFrame?.predicted_reliability || {};

  // Class breakdown counts
  const classCounts: Record<string, number> = {};
  tracks.forEach((t) => {
    classCounts[t.class_name] = (classCounts[t.class_name] || 0) + 1;
  });

  const getClassIcon = (cls?: string | null) => {
    if (!cls) return <Crosshair className="w-3.5 h-3.5 text-cyan-400" />;
    switch (cls.toLowerCase()) {
      case 'car':
        return <Car className="w-3.5 h-3.5 text-blue-400" />;
      case 'truck':
        return <Truck className="w-3.5 h-3.5 text-amber-400" />;
      case 'bus':
        return <Bus className="w-3.5 h-3.5 text-emerald-400" />;
      case 'motorcycle':
      case 'bicycle':
        return <Bike className="w-3.5 h-3.5 text-purple-400" />;
      case 'person':
      case 'pedestrian':
        return <User className="w-3.5 h-3.5 text-rose-400" />;
      default:
        return <Crosshair className="w-3.5 h-3.5 text-cyan-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Module Scope */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-cyan-500/10 rounded-xl text-cyan-400">
              <Crosshair className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Object Detection & Multi-Object Tracking
              </h2>
              <p className="text-xs text-slate-400">
                YOLO11n detector paired with ByteTrack motion continuity and LSTM reliability forecasting
              </p>
            </div>
          </div>
        </div>

        {/* Global Detections / Tracks Telemetry */}
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
              tracks.length > 0
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : results
                ? 'bg-slate-800 text-slate-300 border-slate-700'
                : 'bg-purple-950 text-purple-300 border-purple-500/80'
            }`}
          >
            {tracks.length > 0 ? 'DETECTED' : results ? 'NOT DETECTED' : 'UNKNOWN'}
          </span>
          {results && (
            <div className="flex items-center gap-3 text-xs bg-slate-900 border border-slate-800 rounded-xl px-4 py-2">
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Unique Tracks</span>
                <strong className="text-cyan-400 font-mono text-sm">{results.unique_tracks}</strong>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <span className="text-slate-500 block text-[10px] uppercase">Total Detections</span>
                <strong className="text-slate-200 font-mono text-sm">{results.total_detections}</strong>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Multi-Dimensional Filter Bar */}
      <FilterBar
        filters={filters}
        onChange={setFilters}
        showWeatherFilters={false}
        showEventTypeFilters={false}
        showObjectFilters={true}
        showSeverityFilters={false}
      />

      {/* Main Grid: Visual Feed & Playback Scrubber */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Video Canvas & Scrubber */}
        <div className="lg:col-span-7 space-y-4">
          <VideoCanvas highlightTrackId={selectedTrackId} />
          <PlaybackControls />
        </div>

        {/* Right 5 Cols: Object Breakdown & Reliability Forecasts */}
        <div className="lg:col-span-5 space-y-4">
          {/* Reliability Forecasts Card (H1, H3, H5) */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Anticipated Reliability Forecast
              </span>
              <span className="text-[10px] font-mono text-slate-500 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                LSTM CPU
              </span>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { horizon: 'H1 (+1 Frame)', key: 'h1', time: '+0.03s' },
                { horizon: 'H3 (+3 Frames)', key: 'h3', time: '+0.10s' },
                { horizon: 'H5 (+5 Frames)', key: 'h5', time: '+0.17s' },
              ].map(({ horizon, key, time }) => {
                const score = relScores[key] ?? 0.85;
                const pct = Math.round(score * 100);
                return (
                  <div key={key} className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3 text-center">
                    <span className="text-[10px] text-slate-500 block uppercase font-mono">{horizon} ({time})</span>
                    <span className="text-2xl font-extrabold font-mono text-white block my-1">
                      {pct}%
                    </span>
                    <span
                      className={`text-[10px] font-medium px-2 py-0.5 rounded-full inline-block ${
                        pct >= 80
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : pct >= 60
                          ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          : 'bg-red-500/10 text-red-400 border border-red-500/20'
                      }`}
                    >
                      {pct >= 80 ? 'Reliable' : pct >= 60 ? 'Uncertain' : 'Failure Risk'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Active Classes Breakdown Chips */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-cyan-400" />
                Active Class Distribution
              </span>
              <span className="text-xs font-mono text-cyan-400 font-semibold">
                {tracks.length} active tracks
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              {Object.keys(classCounts).length === 0 ? (
                <p className="text-xs text-slate-500 italic">No targets above confidence threshold in this frame.</p>
              ) : (
                Object.entries(classCounts).map(([cls, count]) => (
                  <div
                    key={cls}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-medium text-slate-200"
                  >
                    {getClassIcon(cls)}
                    <span className="capitalize">{cls}</span>
                    <span className="px-1.5 py-0.2 rounded bg-slate-800 text-cyan-400 font-mono text-[11px] font-bold">
                      {count}
                    </span>
                  </div>
                ))
              )}
            </div>

            {/* Confidence Slider Filter */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="flex justify-between items-center text-xs mb-1.5">
                <span className="text-slate-400 flex items-center gap-1">
                  <Filter className="w-3.5 h-3.5 text-slate-500" />
                  Min Confidence Filter:
                </span>
                <span className="font-mono text-cyan-400 font-bold">{Math.round(filters.minConfidence * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.85"
                step="0.05"
                value={filters.minConfidence}
                onChange={(e) => setFilters((prev) => ({ ...prev, minConfidence: Number(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Tracked Objects Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Crosshair className="w-4 h-4 text-cyan-400" />
            Track Continuity Telemetry (Frame #{currentFrameIndex})
          </span>
          <span className="text-xs font-mono text-slate-500">
            ByteTrack Association Matrix
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Track ID</th>
                <th className="py-2.5 px-3">Class</th>
                <th className="py-2.5 px-3">Confidence</th>
                <th className="py-2.5 px-3">BBox Coordinates (x1, y1, x2, y2)</th>
                <th className="py-2.5 px-3">Centroid (cx, cy)</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {tracks.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-slate-500 italic">
                    No active tracks meeting filter criteria in this frame.
                  </td>
                </tr>
              ) : (
                tracks.map((t) => {
                  const [x1, y1, x2, y2] = t.bbox_xyxy;
                  const cx = Math.round((x1 + x2) / 2);
                  const cy = Math.round((y1 + y2) / 2);
                  const isSelected = selectedTrackId === t.track_id;

                  return (
                    <tr
                      key={t.track_id}
                      onClick={() =>
                        setSelectedTrackId(isSelected ? null : t.track_id)
                      }
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-cyan-950/40 text-cyan-300'
                          : 'hover:bg-slate-800/40 text-slate-300'
                      }`}
                    >
                      <td className="py-2.5 px-3 font-bold text-white flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-cyan-400" />#{t.track_id}
                      </td>
                      <td className="py-2.5 px-3 capitalize font-sans flex items-center gap-1.5">
                        {getClassIcon(t.class_name)}
                        {t.class_name}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-200">
                          {(t.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        [{x1.toFixed(0)}, {y1.toFixed(0)}, {x2.toFixed(0)}, {y2.toFixed(0)}]
                      </td>
                      <td className="py-2.5 px-3 text-cyan-400">
                        ({cx}, {cy})
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedTrackId(isSelected ? null : t.track_id);
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
      </div>
    </div>
  );
};
