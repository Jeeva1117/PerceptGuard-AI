import React, { useState, useMemo } from 'react';
import {
  OctagonAlert,
  ShieldAlert,
  Info,
  Clock,
  Play,
  Search,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from '../components/VideoCanvas';
import { PlaybackControls } from '../components/PlaybackControls';

export interface SafetyAlertsPageProps {
  isEmbedded?: boolean;
}

export const SafetyAlertsPage: React.FC<SafetyAlertsPageProps> = ({ isEmbedded = false }) => {
  const { results, currentFrame, currentFrameIndex, seekToFrame, seekToTimestamp } = useJob();
  const [selectedEventType, setSelectedEventType] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const safety = currentFrame?.safety_recommendation;
  const hasData = Boolean(safety);
  const collision = currentFrame?.collision_risk;

  // Final Risk State & Reason from actual risk engine output
  const finalRiskState =
    results?.final_summary?.collision_risk ||
    collision?.collision_decision ||
    (hasData
      ? collision?.risk_state === 'CRITICAL'
        ? 'COLLISION OBSERVED'
        : collision?.risk_state === 'HIGH'
        ? 'POTENTIAL COLLISION RISK'
        : 'NO COLLISION RISK'
      : 'UNKNOWN');

  const evidenceBasedReason =
    results?.final_summary?.reason ||
    collision?.decision_reason ||
    safety?.primary_reason ||
    (hasData
      ? 'Nominal road clearance maintained; trajectory projections within safety envelope.'
      : 'Awaiting video stream telemetry.');

  const relevantTimestamp =
    results?.final_summary?.event_timestamp != null
      ? `${results.final_summary.event_timestamp.toFixed(2)}s`
      : currentFrame?.timestamp_seconds != null
      ? `${currentFrame.timestamp_seconds.toFixed(2)}s`
      : '0.00s';

  const relevantObjectId =
    results?.final_summary?.primary_object ??
    (collision?.closest_track_id != null ? `#${collision.closest_track_id}` : 'Not Available');

  const disclaimer =
    safety?.disclaimer ||
    'PerceptGuard AI is an experimental ADAS research prototype. Operational safety remains the driver\'s responsibility. Advisories do not command vehicle actuation.';

  const allTimelineEvents = results?.event_timeline || [];

  const filteredTimelineEvents = useMemo(() => {
    return allTimelineEvents.filter((ev) => {
      if (selectedEventType !== 'ALL' && ev.event_type !== selectedEventType) return false;
      if (searchQuery.trim() !== '') {
        const q = searchQuery.toLowerCase();
        const matchesTitle = ev.title?.toLowerCase().includes(q);
        const matchesDesc = ev.description?.toLowerCase().includes(q);
        const matchesCat = ev.category?.toLowerCase().includes(q);
        if (!matchesTitle && !matchesDesc && !matchesCat) return false;
      }
      return true;
    });
  }, [allTimelineEvents, selectedEventType, searchQuery]);

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

  const getSeverityBadge = (sev: string) => {
    switch (sev.toUpperCase()) {
      case 'CRITICAL':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/40';
      case 'HIGH':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/40';
      case 'MODERATE':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/40';
      case 'LOW':
        return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40';
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <div className="space-y-4 select-none">
      {/* Module Title Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400 shrink-0">
            <OctagonAlert className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-wide">
                Safety Alerts
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                MODULE 5
              </span>
            </div>
            <p className="text-[11px] text-slate-400 truncate">
              Context-aware driver advisories & evidence chain
            </p>
          </div>
        </div>

        <span
          className={`px-2.5 py-1 rounded-full text-[10px] font-extrabold font-mono border shrink-0 ${getResultBadge(
            finalRiskState
          )}`}
        >
          {finalRiskState}
        </span>
      </div>

      {/* Standalone Video Player (Only outside master workspace) */}
      {!isEmbedded && (
        <div className="space-y-3">
          <VideoCanvas />
          <PlaybackControls />
        </div>
      )}

      {/* Task 9 Recommended Clean Final Decision & Advisory Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
            Final Decision & Advisory
          </span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold font-mono border ${getResultBadge(finalRiskState)}`}>
            {finalRiskState}
          </span>
        </div>

        {/* 3 Metric Pills: Final Risk State, Relevant Timestamp, Relevant Object */}
        <div className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Final Risk State</span>
            <strong className="text-white text-[11px] block mt-0.5 truncate" title={finalRiskState}>
              {finalRiskState}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Relevant Time</span>
            <strong className="text-cyan-300 text-[11px] block mt-0.5">
              {relevantTimestamp}
            </strong>
          </div>

          <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 uppercase block font-sans font-semibold">Relevant Object</span>
            <strong className="text-amber-400 text-[11px] block mt-0.5 truncate">
              {relevantObjectId}
            </strong>
          </div>
        </div>

        {/* Evidence-Based Reason */}
        <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            Evidence-Based Reason:
          </span>
          <p className="text-xs font-medium text-slate-200 leading-relaxed font-sans">
            {evidenceBasedReason}
          </p>
        </div>
      </div>

      {/* Task 9 Compact 1-Line Filter Bar (Replaces Bloated Multi-Dimensional Bar) */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-3 shadow-lg flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
          <button
            type="button"
            onClick={() => setSelectedEventType('ALL')}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
              selectedEventType === 'ALL'
                ? 'bg-cyan-600 text-white font-bold'
                : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            All Events
          </button>
          <button
            type="button"
            onClick={() => setSelectedEventType('COLLISION_HAZARD')}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
              selectedEventType === 'COLLISION_HAZARD'
                ? 'bg-amber-600 text-white font-bold'
                : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            Hazards
          </button>
          <button
            type="button"
            onClick={() => setSelectedEventType('SAFETY_ADVISORY')}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
              selectedEventType === 'SAFETY_ADVISORY'
                ? 'bg-cyan-600 text-white font-bold'
                : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            Advisories
          </button>
          <button
            type="button"
            onClick={() => setSelectedEventType('WEATHER_FAULT')}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
              selectedEventType === 'WEATHER_FAULT'
                ? 'bg-blue-600 text-white font-bold'
                : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            Weather
          </button>
        </div>

        <div className="relative min-w-[140px] flex-1 max-w-[200px]">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5 pointer-events-none" />
          <input
            type="text"
            placeholder="Search events..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-2.5 py-1 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
          />
        </div>
      </div>

      {/* Relevant Events (Scrollable Timestamped Alert List with Click-to-Seek) */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            Relevant Events Timeline
          </span>
          <span className="text-[10px] font-mono text-slate-400">
            {filteredTimelineEvents.length} record(s)
          </span>
        </div>

        {filteredTimelineEvents.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800/80">
            No events match the active filter criteria.
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
            {filteredTimelineEvents.map((ev, idx) => {
              const isSelected = currentFrameIndex === ev.evidence_frame_index;
              const sev = ev.severity || 'INFO';

              return (
                <div
                  key={ev.event_id || idx}
                  className={`p-3 rounded-xl border transition-all text-xs space-y-2 ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500 ring-1 ring-cyan-500 shadow-md'
                      : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-mono font-bold bg-slate-900 border border-slate-700 text-cyan-300">
                      [{ev.start_time || `${ev.start_timestamp_sec?.toFixed(1)}s`} - {ev.end_time || `${ev.end_timestamp_sec?.toFixed(1)}s`}]
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${getSeverityBadge(sev)}`}>
                      {sev}
                    </span>
                  </div>

                  <div>
                    <h4 className="font-bold text-white text-xs">{ev.title}</h4>
                    <p className="text-slate-400 text-[11px] mt-0.5 leading-relaxed font-sans">{ev.description}</p>
                    {ev.action_recommended && (
                      <div className="text-cyan-300 text-[11px] font-medium mt-1 font-mono">
                        Advisory: {ev.action_recommended}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-end pt-1">
                    <button
                      type="button"
                      onClick={() => {
                        if (ev.start_timestamp_sec != null) {
                          seekToTimestamp(ev.start_timestamp_sec);
                        } else {
                          seekToFrame(ev.evidence_frame_index);
                        }
                      }}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-mono font-semibold transition-colors flex items-center gap-1.5 ${
                        isSelected
                          ? 'bg-cyan-500 text-slate-950 font-bold'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
                      }`}
                    >
                      <Play className="w-3 h-3 fill-current" />
                      <span>{isSelected ? 'Viewing Frame' : 'Seek to Event'}</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Task 9 Compact System Disclaimer Footer */}
      <div className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3 flex items-start gap-2.5 text-[11px] text-slate-400 leading-relaxed select-none">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <span>
          <strong className="text-slate-300 font-semibold">Research Disclaimer:</strong> {disclaimer}
        </span>
      </div>
    </div>
  );
};
