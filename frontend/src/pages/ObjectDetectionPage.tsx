import React, { useState } from 'react';
import {
  Crosshair,
  Car,
  Truck,
  Bus,
  Bike,
  User,
  Clock,
  Play,
  Layers,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export interface ObjectDetectionPageProps {
  isEmbedded?: boolean;
}

export const ObjectDetectionPage: React.FC<ObjectDetectionPageProps> = ({ isEmbedded = false }) => {
  const {
    results,
    currentFrame,
    selectedTrackId,
    setSelectedTrackId,
    seekToTimestamp,
  } = useJob();

  const [classFilter, setClassFilter] = useState<string>('ALL');

  // Road-user unique counts from backend summary
  const summaryCounts = results?.final_summary?.road_user_counts;
  const counts = {
    car: summaryCounts?.car ?? 0,
    motorcycle: summaryCounts?.motorcycle ?? 0,
    bus: summaryCounts?.bus ?? 0,
    truck: summaryCounts?.truck ?? 0,
    bicycle: summaryCounts?.bicycle ?? 0,
    pedestrian: summaryCounts?.pedestrian ?? 0,
  };

  // Active detections on current frame
  const currentTracks = currentFrame?.tracks || [];
  const filteredCurrentTracks = currentTracks.filter((t) => {
    if (classFilter !== 'ALL' && t.class_name.toLowerCase() !== classFilter.toLowerCase()) {
      return false;
    }
    return true;
  });

  // Extract all unique tracks observed across the video
  const allUniqueTracks = React.useMemo(() => {
    if (!results?.frames) return [];
    const map = new Map<number, {
      track_id: number;
      class_name: string;
      first_seen_ts: number;
      first_seen_frame: number;
      max_confidence: number;
      speed_kmh?: number;
    }>();

    results.frames.forEach((f) => {
      f.tracks?.forEach((t: any) => {
        if (!map.has(t.track_id)) {
          map.set(t.track_id, {
            track_id: t.track_id,
            class_name: t.class_name,
            first_seen_ts: f.timestamp_seconds,
            first_seen_frame: f.frame_index,
            max_confidence: t.confidence,
            speed_kmh: t.speed_kmh,
          });
        } else {
          const existing = map.get(t.track_id)!;
          if (t.confidence > existing.max_confidence) {
            existing.max_confidence = t.confidence;
          }
          if (t.speed_kmh && (!existing.speed_kmh || t.speed_kmh > existing.speed_kmh)) {
            existing.speed_kmh = t.speed_kmh;
          }
        }
      });
    });

    return Array.from(map.values()).sort((a, b) => a.track_id - b.track_id);
  }, [results?.frames]);

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
        return <Bike className="w-3.5 h-3.5 text-purple-400" />;
      case 'bicycle':
        return <Bike className="w-3.5 h-3.5 text-cyan-400" />;
      case 'person':
      case 'pedestrian':
        return <User className="w-3.5 h-3.5 text-rose-400" />;
      default:
        return <Crosshair className="w-3.5 h-3.5 text-cyan-400" />;
    }
  };

  return (
    <div className="space-y-4 select-none">
      {/* Module Title Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400 shrink-0">
            <Crosshair className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-wide">
                Object Detection
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                MODULE 2
              </span>
            </div>
            <p className="text-[11px] text-slate-400 truncate">
              YOLO11 road-user perception & ByteTrack tracking
            </p>
          </div>
        </div>

        <span className="px-2.5 py-1 rounded-full text-[10px] font-bold font-mono bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shrink-0">
          {results?.unique_tracks ?? allUniqueTracks.length} TRACKS
        </span>
      </div>

      {/* Standalone Video Player (Only outside master workspace) */}
      {!isEmbedded && (
        <div className="space-y-3">
          <VideoCanvas />
          <PlaybackControls />
        </div>
      )}

      {/* Six Supported Road-User Classes Distribution */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            Detected Road Users
          </span>
          <span className="text-[10px] font-mono text-slate-400">
            Total: {results?.unique_tracks ?? Object.values(counts).reduce((a, b) => a + b, 0)}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <button
            type="button"
            onClick={() => setClassFilter(classFilter === 'car' ? 'ALL' : 'car')}
            className={`p-2 rounded-xl border text-left transition-all ${
              classFilter === 'car'
                ? 'bg-blue-950/40 border-blue-500 ring-1 ring-blue-500'
                : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Car</span>
              <Car className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <strong className="text-sm font-mono text-white block mt-0.5">{counts.car}</strong>
          </button>

          <button
            type="button"
            onClick={() => setClassFilter(classFilter === 'motorcycle' ? 'ALL' : 'motorcycle')}
            className={`p-2 rounded-xl border text-left transition-all ${
              classFilter === 'motorcycle'
                ? 'bg-purple-950/40 border-purple-500 ring-1 ring-purple-500'
                : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Motorcycle</span>
              <Bike className="w-3.5 h-3.5 text-purple-400" />
            </div>
            <strong className="text-sm font-mono text-white block mt-0.5">{counts.motorcycle}</strong>
          </button>

          <button
            type="button"
            onClick={() => setClassFilter(classFilter === 'bus' ? 'ALL' : 'bus')}
            className={`p-2 rounded-xl border text-left transition-all ${
              classFilter === 'bus'
                ? 'bg-emerald-950/40 border-emerald-500 ring-1 ring-emerald-500'
                : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Bus</span>
              <Bus className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <strong className="text-sm font-mono text-white block mt-0.5">{counts.bus}</strong>
          </button>

          <button
            type="button"
            onClick={() => setClassFilter(classFilter === 'truck' ? 'ALL' : 'truck')}
            className={`p-2 rounded-xl border text-left transition-all ${
              classFilter === 'truck'
                ? 'bg-amber-950/40 border-amber-500 ring-1 ring-amber-500'
                : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Truck</span>
              <Truck className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <strong className="text-sm font-mono text-white block mt-0.5">{counts.truck}</strong>
          </button>

          <button
            type="button"
            onClick={() => setClassFilter(classFilter === 'bicycle' ? 'ALL' : 'bicycle')}
            className={`p-2 rounded-xl border text-left transition-all ${
              classFilter === 'bicycle'
                ? 'bg-cyan-950/40 border-cyan-500 ring-1 ring-cyan-500'
                : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Bicycle</span>
              <Bike className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <strong className="text-sm font-mono text-white block mt-0.5">{counts.bicycle}</strong>
          </button>

          <button
            type="button"
            onClick={() => setClassFilter(classFilter === 'pedestrian' || classFilter === 'person' ? 'ALL' : 'pedestrian')}
            className={`p-2 rounded-xl border text-left transition-all ${
              classFilter === 'pedestrian' || classFilter === 'person'
                ? 'bg-rose-950/40 border-rose-500 ring-1 ring-rose-500'
                : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Pedestrian</span>
              <User className="w-3.5 h-3.5 text-rose-400" />
            </div>
            <strong className="text-sm font-mono text-white block mt-0.5">{counts.pedestrian}</strong>
          </button>
        </div>
      </div>

      {/* Active Detections on Current Timestamp */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            Active Detections (Current Frame)
          </span>
          <span className="text-[11px] font-mono text-slate-400">
            {filteredCurrentTracks.length} visible
          </span>
        </div>

        {filteredCurrentTracks.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800/80">
            No road users detected on current frame timestamp.
          </div>
        ) : (
          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {filteredCurrentTracks.map((t) => {
              const isSelected = selectedTrackId === t.track_id;
              return (
                <div
                  key={t.track_id}
                  onClick={() => setSelectedTrackId(isSelected ? null : t.track_id)}
                  className={`p-2.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-2 ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500 ring-1 ring-cyan-500 shadow-md'
                      : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 shrink-0">
                      {getClassIcon(t.class_name)}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 text-xs font-mono font-bold text-white">
                        <span>#{t.track_id}</span>
                        <span className="capitalize font-sans font-medium text-slate-300">
                          {t.class_name}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                        Conf: {(t.confidence * 100).toFixed(0)}%
                        {(t as any).speed_kmh != null && ` • ${(t as any).speed_kmh.toFixed(1)} km/h`}
                      </div>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedTrackId(isSelected ? null : t.track_id);
                    }}
                    className={`px-2 py-1 rounded-lg text-[10px] font-mono font-bold transition-colors shrink-0 ${
                      isSelected
                        ? 'bg-cyan-500 text-slate-950'
                        : 'bg-slate-800 text-slate-300 hover:text-white'
                    }`}
                  >
                    {isSelected ? 'Focused' : 'Focus'}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Historical Track Log with Instant Seek */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Play className="w-3.5 h-3.5 text-cyan-400" />
            Track History (Click to Seek)
          </span>
          <span className="text-[10px] font-mono text-slate-400">
            {allUniqueTracks.length} unique tracks
          </span>
        </div>

        {allUniqueTracks.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800/80">
            Awaiting completed video analysis to index track registry.
          </div>
        ) : (
          <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1 font-mono text-xs">
            {allUniqueTracks.map((ut) => (
              <div
                key={ut.track_id}
                onClick={() => {
                  seekToTimestamp(ut.first_seen_ts);
                  setSelectedTrackId(ut.track_id);
                }}
                className="p-2 rounded-xl bg-slate-950 border border-slate-800 hover:border-cyan-500/50 hover:bg-slate-800/40 transition-colors cursor-pointer flex items-center justify-between gap-2"
              >
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <strong className="text-white">#{ut.track_id}</strong>
                  <span className="capitalize font-sans text-slate-300 text-[11px]">
                    {ut.class_name}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-[10px] text-slate-400">
                  <span>@ {ut.first_seen_ts.toFixed(1)}s</span>
                  <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-cyan-300">
                    Seek
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
