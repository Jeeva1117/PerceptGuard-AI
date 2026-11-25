import React, { useState, useMemo } from 'react';
import {
  OctagonAlert,
  ShieldAlert,
  Info,
  Clock,
  Play,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';
import { FilterBar, DEFAULT_FILTER_STATE, type FilterState } from '../components/FilterBar';

export interface SafetyAlertsPageProps {
  isEmbedded?: boolean;
}

export const SafetyAlertsPage: React.FC<SafetyAlertsPageProps> = ({ isEmbedded = false }) => {
  const { results, currentFrame, currentFrameIndex, seekToFrame, seekToTimestamp } = useJob();
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTER_STATE);

  const safety = currentFrame?.safety_recommendation;
  const hasData = Boolean(safety);
  const collision = currentFrame?.collision_risk;

  // Collision result & reason
  const collisionResult =
    results?.final_summary?.collision_risk ||
    collision?.collision_decision ||
    (hasData ? (collision?.risk_state === 'CRITICAL' ? 'COLLISION OBSERVED' : collision?.risk_state === 'HIGH' ? 'POTENTIAL COLLISION RISK' : 'NO COLLISION RISK') : 'UNKNOWN');

  const collisionReason =
    results?.final_summary?.reason ||
    collision?.decision_reason ||
    safety?.primary_reason ||
    (hasData ? 'Nominal road clearance maintained; trajectory projections within safety envelope.' : 'Awaiting video stream telemetry.');

  const relevantTimestamp =
    results?.final_summary?.event_timestamp != null
      ? `${results.final_summary.event_timestamp.toFixed(2)}s`
      : currentFrame?.timestamp_seconds != null
      ? `${currentFrame.timestamp_seconds.toFixed(2)}s`
      : '0.00s';

  const relevantObjectId =
    results?.final_summary?.primary_object ??
    (collision?.closest_track_id != null ? `#${collision.closest_track_id}` : 'None');

  const disclaimer =
    safety?.disclaimer ||
    'PerceptGuard AI is an experimental ADAS research prototype. Operational safety remains the driver\'s responsibility. Advisories do not command vehicle actuation.';

  const allTimelineEvents = results?.event_timeline || [];
  const filteredTimelineEvents = useMemo(() => {
    return allTimelineEvents.filter((ev) => {
      if (filters.eventType !== 'ALL' && ev.event_type !== filters.eventType) return false;
      if (filters.severity !== 'ALL' && ev.severity.toUpperCase() !== filters.severity.toUpperCase()) return false;
      if (filters.visibilityCondition !== 'ALL' && ev.category.toUpperCase() !== filters.visibilityCondition.toUpperCase()) return false;
      if (filters.trackId.trim() !== '') {
        const tidStr = filters.trackId.trim();
        const hasTrack = ev.involved_tracks && ev.involved_tracks.some((t: any) => String(t) === tidStr);
        if (!hasTrack) return false;
      }
      return true;
    });
  }, [allTimelineEvents, filters]);

  const formatMmss = (sec?: number) => {
    if (sec == null) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

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
              <OctagonAlert className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Safety Alerts & Driver Advisory
              </h2>
              <p className="text-xs text-slate-400">
                Context-aware driver advisories and clickable chronological safety event timeline
              </p>
            </div>
          </div>
        </div>

        {/* Global Collision Result Badge */}
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

      {/* Mandatory Concise Safety Alerts Summary Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <span className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-cyan-400" />
            SAFETY ALERTS — Final Decision & Advisory
          </span>
          <span className={`px-2.5 py-0.5 rounded text-xs font-extrabold font-mono border ${getResultBadge(collisionResult)}`}>
            {collisionResult}
          </span>
        </div>

        {/* Concise Result Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-500 uppercase block font-bold">Collision Risk Result</span>
            <span className={`inline-block mt-1 px-2 py-0.5 rounded text-xs font-extrabold border ${getResultBadge(collisionResult)}`}>
              {collisionResult}
            </span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-500 uppercase block font-bold">Relevant Timestamp</span>
            <strong className="text-cyan-300 text-sm block mt-1">{relevantTimestamp}</strong>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-500 uppercase block font-bold">Relevant Object ID</span>
            <strong className="text-amber-400 text-sm block mt-1">{relevantObjectId}</strong>
          </div>
        </div>

        <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            Evidence-Based Reason:
          </span>
          <p className="text-xs font-medium text-slate-200 leading-relaxed font-sans">
            {collisionReason}
          </p>
        </div>
      </div>

      {/* Video Canvas & Controls */}
      {!isEmbedded && (
        <div className="space-y-4">
          <VideoCanvas />
          <PlaybackControls />
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 flex items-center gap-3 text-xs text-slate-400 select-none">
        <Info className="w-4 h-4 text-cyan-400 shrink-0" />
        <span className="flex-1">
          <strong className="text-slate-200">System Disclaimer:</strong> {disclaimer}
        </span>
      </div>

      {/* Filter Bar */}
      <FilterBar
        filters={filters}
        onChange={setFilters}
        maxDurationSec={results?.timeline_telemetry?.source_video_duration_sec || 300}
      />

      {/* Clickable Chronological Video Event Timeline */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Clickable Chronological Event Timeline
              </h3>
              <p className="text-xs text-slate-400">
                Click any event to immediately seek playback to the exact timestamp and evidence frame
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {results?.timeline_telemetry && (
              <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                {results.timeline_telemetry.timeline_coverage_percentage}% Coverage (00:00 - {formatMmss(results.timeline_telemetry.source_video_duration_sec)})
              </span>
            )}
            <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700">
              {filteredTimelineEvents.length} Event(s)
            </span>
          </div>
        </div>

        {filteredTimelineEvents.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800">
            No events match the active filter criteria. Click "Reset Filters" above to view all events.
          </div>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
            {filteredTimelineEvents.map((ev, idx) => {
              const isSelected = currentFrameIndex === ev.evidence_frame_index;
              const sev = (ev.severity || 'INFO').toUpperCase();
              const sevBadge =
                sev === 'CRITICAL'
                  ? 'bg-red-500/20 text-red-400 border-red-500/40'
                  : sev === 'HIGH'
                  ? 'bg-orange-500/20 text-orange-400 border-orange-500/40'
                  : sev === 'MODERATE'
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                  : sev === 'LOW'
                  ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40'
                  : 'bg-blue-500/20 text-blue-400 border-blue-500/40';

              return (
                <div
                  key={ev.event_id || idx}
                  className={`p-3.5 rounded-xl border transition-all text-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-3 ${
                    isSelected
                      ? 'bg-cyan-950/30 border-cyan-500 shadow-md shadow-cyan-950/40 ring-1 ring-cyan-500'
                      : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start gap-3 min-w-0 flex-1">
                    <span className="px-2.5 py-1 rounded-lg text-xs font-mono font-bold bg-slate-900 border border-slate-700 text-cyan-300 shrink-0">
                      [{ev.start_time} - {ev.end_time}]
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border shrink-0 ${sevBadge}`}>
                      {ev.severity}
                    </span>
                    <div className="min-w-0">
                      <div className="font-bold text-white flex items-center gap-2">
                        <span>{ev.title}</span>
                        <span className="text-[10px] font-mono text-slate-500 font-normal uppercase">
                          ({ev.category})
                        </span>
                      </div>
                      <p className="text-slate-400 text-xs mt-0.5">{ev.description}</p>
                      {ev.action_recommended && (
                        <div className="text-cyan-300 text-[11px] font-medium mt-1">
                          Advisory: {ev.action_recommended}
                        </div>
                      )}
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => {
                      if (ev.start_timestamp_sec != null) {
                        seekToTimestamp(ev.start_timestamp_sec);
                      } else {
                        seekToFrame(ev.evidence_frame_index);
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-colors flex items-center gap-1.5 shrink-0 ${
                      isSelected
                        ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                        : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
                    }`}
                  >
                    <Play className="w-3 h-3 fill-current" />
                    <span>{isSelected ? 'Viewing' : 'Seek to'} @ {ev.start_time || `${ev.start_timestamp_sec?.toFixed(1)}s`} (Frame #{ev.evidence_frame_index})</span>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
