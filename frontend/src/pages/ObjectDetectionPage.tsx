import React, { useState } from 'react';
import {
  Crosshair,
  Car,
  Truck,
  Bus,
  Bike,
  User,
  Filter,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';
import { FilterBar, DEFAULT_FILTER_STATE, type FilterState } from '../components/FilterBar';

export interface ObjectDetectionPageProps {
  isEmbedded?: boolean;
}

export const ObjectDetectionPage: React.FC<ObjectDetectionPageProps> = ({ isEmbedded = false }) => {
  const { results, currentFrame, currentFrameIndex, selectedTrackId: globalTrackId, setSelectedTrackId: setGlobalTrackId } = useJob();
  const selectedTrackId = globalTrackId;
  const setSelectedTrackId = setGlobalTrackId;
  const [filters, setFilters] = useState<FilterState>({
    ...DEFAULT_FILTER_STATE,
    minConfidence: 0.20,
  });

  const rawTracks = currentFrame?.tracks || [];
  const tracks = rawTracks.filter((t) => {
    if (filters.minConfidence > 0 && t.confidence < filters.minConfidence) return false;
    if (filters.objectClass !== 'ALL' && t.class_name.toLowerCase() !== filters.objectClass.toLowerCase()) return false;
    if (filters.trackId.trim() !== '' && String(t.track_id) !== filters.trackId.trim()) return false;
    return true;
  });

  // Road user unique counts from backend summary or computed across all frames
  const summaryCounts = results?.final_summary?.road_user_counts;
  const counts = {
    car: summaryCounts?.car ?? 0,
    motorcycle: summaryCounts?.motorcycle ?? 0,
    bus: summaryCounts?.bus ?? 0,
    truck: summaryCounts?.truck ?? 0,
    bicycle: summaryCounts?.bicycle ?? 0,
    pedestrian: summaryCounts?.pedestrian ?? 0,
  };

  // If no summary was generated, compute unique tracks from frames
  if (!summaryCounts && results?.frames) {
    const seenTracks = new Map<number, string>();
    results.frames.forEach((f) => {
      f.tracks?.forEach((t) => {
        if (!seenTracks.has(t.track_id)) {
          seenTracks.set(t.track_id, t.class_name.toLowerCase());
        }
      });
    });
    seenTracks.forEach((cls) => {
      if (cls === 'car') counts.car += 1;
      else if (cls === 'motorcycle') counts.motorcycle += 1;
      else if (cls === 'bus') counts.bus += 1;
      else if (cls === 'truck') counts.truck += 1;
      else if (cls === 'bicycle') counts.bicycle += 1;
      else if (cls === 'person' || cls === 'pedestrian') counts.pedestrian += 1;
    });
  }

  const getClassIcon = (cls?: string | null) => {
    if (!cls) return <Crosshair className="w-4 h-4 text-cyan-400" />;
    switch (cls.toLowerCase()) {
      case 'car':
        return <Car className="w-4 h-4 text-blue-400" />;
      case 'truck':
        return <Truck className="w-4 h-4 text-amber-400" />;
      case 'bus':
        return <Bus className="w-4 h-4 text-emerald-400" />;
      case 'motorcycle':
      case 'bicycle':
        return <Bike className="w-4 h-4 text-purple-400" />;
      case 'person':
      case 'pedestrian':
        return <User className="w-4 h-4 text-rose-400" />;
      default:
        return <Crosshair className="w-4 h-4 text-cyan-400" />;
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
                YOLO11 object detection with ByteTrack persistent identity continuity across occlusions
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

      {/* Mandatory Concise Object Detection Breakdown: Detected Objects */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <Crosshair className="w-4 h-4 text-cyan-400" />
            OBJECT DETECTION — Detected Objects
          </span>
          <span className="text-xs font-mono text-slate-400">
            Total Unique Road Users: <strong className="text-cyan-400">{results?.unique_tracks ?? Object.values(counts).reduce((a, b) => a + b, 0)}</strong>
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
              <Car className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-500 block">Car</span>
              <strong className="text-lg font-mono text-white">{counts.car}</strong>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
              <Bike className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-500 block">Motorcycle</span>
              <strong className="text-lg font-mono text-white">{counts.motorcycle}</strong>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Bus className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-500 block">Bus</span>
              <strong className="text-lg font-mono text-white">{counts.bus}</strong>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
              <Truck className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-500 block">Truck</span>
              <strong className="text-lg font-mono text-white">{counts.truck}</strong>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
              <Bike className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-500 block">Bicycle</span>
              <strong className="text-lg font-mono text-white">{counts.bicycle}</strong>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400">
              <User className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-500 block">Pedestrian</span>
              <strong className="text-lg font-mono text-white">{counts.pedestrian}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <FilterBar
        filters={filters}
        onChange={setFilters}
        showWeatherFilters={false}
        showEventTypeFilters={false}
        showObjectFilters={true}
        showSeverityFilters={false}
      />

      {/* Video Canvas & Controls */}
      {!isEmbedded && (
        <div className="space-y-4">
          <VideoCanvas highlightTrackId={selectedTrackId} />
          <PlaybackControls />
        </div>
      )}

      {/* Tracked Objects Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Crosshair className="w-4 h-4 text-cyan-400" />
            Tracked Objects Telemetry (Frame #{currentFrameIndex})
          </span>
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1 text-slate-400">
              <Filter className="w-3.5 h-3.5 text-slate-500" />
              <span>Min Conf:</span>
              <span className="font-mono text-cyan-400 font-bold">{Math.round(filters.minConfidence * 100)}%</span>
            </div>
            <span className="text-xs font-mono text-slate-500">
              {tracks.length} track(s) in frame
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Track ID</th>
                <th className="py-2.5 px-3">Class</th>
                <th className="py-2.5 px-3">Confidence</th>
                <th className="py-2.5 px-3">Lifecycle State</th>
                <th className="py-2.5 px-3">BBox Coordinates (x1, y1, x2, y2)</th>
                <th className="py-2.5 px-3">Centroid (cx, cy)</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {tracks.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-slate-500 italic">
                    No active tracks meeting filter criteria in this frame.
                  </td>
                </tr>
              ) : (
                tracks.map((t) => {
                  const [x1, y1, x2, y2] = t.bbox_xyxy;
                  const cx = Math.round((x1 + x2) / 2);
                  const cy = Math.round((y1 + y2) / 2);
                  const isSelected = selectedTrackId === t.track_id;
                  const lifecycle = t.track_lifecycle || 'CONFIRMED';

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
                      <td className="py-2.5 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                            lifecycle === 'CONFIRMED'
                              ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                              : lifecycle === 'TENTATIVE'
                              ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                              : lifecycle === 'REIDENTIFIED'
                              ? 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
                              : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                        >
                          {lifecycle}
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
