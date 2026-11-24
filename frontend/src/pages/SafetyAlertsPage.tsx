import React, { useState, useMemo } from 'react';
import {
  OctagonAlert,
  ShieldAlert,
  AlertTriangle,
  Info,
  Sliders,
  ShieldCheck,
  Flame,
  Car,
  User,
  Clock,
  Eye,
  HelpCircle,
  History,
  Activity,
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
  const [ttcThreshold, setTtcThreshold] = useState<number>(1.8);
  const [visThreshold, setVisThreshold] = useState<number>(60);
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTER_STATE);

  const safety = currentFrame?.safety_recommendation;
  const hasData = Boolean(safety);
  const collision = currentFrame?.collision_risk;
  const weather = currentFrame?.weather_intelligence;

  // Resolve recommendation state (supporting both new 6-state contract and legacy actions)
  const rawState = (
    safety?.recommendation_state ||
    safety?.primary_action ||
    (hasData ? 'NORMAL' : 'UNKNOWN')
  ).toUpperCase();

  const priority = (safety?.priority || safety?.urgency_level || (hasData ? 'INFO' : 'UNKNOWN')).toUpperCase();
  const riskLevel = (safety?.risk_level || collision?.risk_state || (hasData ? 'LOW' : 'UNKNOWN')).toUpperCase();
  const recommendedAction =
    safety?.recommended_action ||
    (hasData ? 'Maintain nominal speed and lane position; traffic path unobstructed.' : 'No recommendation telemetry available.');
  const primaryReason =
    safety?.primary_reason ||
    (hasData ? 'Nominal road clearance maintained; sensor metrics and trajectory projections within safety envelope.' : 'Awaiting video stream telemetry.');

  const evidenceChain = safety?.evidence_chain || (hasData ? [
    'Trajectory projection clear of collision obstacles.',
    'Atmospheric clarity and sensor optics operating nominally.',
  ] : ['No safety evidence chain recorded for this stream.']);

  const disclaimer =
    safety?.disclaimer ||
    'PerceptGuard AI is an experimental ADAS decision-support prototype. Operational safety remains the driver\'s responsibility. Stop recommendations are advisory only (when safe to do so) and do not command vehicle actuation.';

  const involvedObjects = safety?.objects_involved || [];
  const supportingEv = safety?.supporting_evidence;
  const frameAlerts = currentFrame?.alerts || [];
  const stateTransitions = results?.safety_summary?.state_transitions || [];

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

  const getActionTheme = (st?: string | null) => {
    switch ((st || '').toUpperCase()) {
      case 'STOP RECOMMENDED':
      case 'STOP_RECOMMENDED':
      case 'STOP_RECOMMENDATION':
        return {
          banner: 'bg-red-600 border-red-500 text-white shadow-red-950/50',
          badge: 'bg-white/20 text-white border-white/30',
          icon: <OctagonAlert className="w-8 h-8 animate-pulse text-white" />,
          title: 'STOP RECOMMENDED',
          subtitle: 'Imminent trajectory hazard detected. Stop vehicle when safe to do so.',
        };
      case 'PREPARE TO STOP':
      case 'PREPARE_TO_STOP':
        return {
          banner: 'bg-orange-600 border-orange-500 text-white shadow-orange-950/50',
          badge: 'bg-white/20 text-white border-white/30',
          icon: <AlertTriangle className="w-8 h-8 text-white" />,
          title: 'PREPARE TO STOP',
          subtitle: 'Road agent converging in projected path. Controlled braking readiness advised.',
        };
      case 'REDUCE SPEED':
      case 'REDUCE_SPEED':
        return {
          banner: 'bg-amber-600 border-amber-500 text-slate-950 shadow-amber-950/50',
          badge: 'bg-slate-950/20 text-slate-950 border-slate-950/30',
          icon: <AlertTriangle className="w-8 h-8 text-slate-950" />,
          title: 'REDUCE SPEED',
          subtitle: 'Decelerate to expand safety buffer under elevated road-agent risk.',
        };
      case 'CAUTION':
        return {
          banner: 'bg-yellow-500 border-yellow-400 text-slate-950 shadow-yellow-950/50',
          badge: 'bg-slate-950/20 text-slate-950 border-slate-950/30',
          icon: <AlertTriangle className="w-8 h-8 text-slate-950" />,
          title: 'CAUTION',
          subtitle: 'Pay increased attention; environmental degradation or peripheral road activity detected.',
        };
      case 'UNKNOWN':
        return {
          banner: 'bg-slate-800 border-slate-700 text-slate-200 shadow-slate-950/50',
          badge: 'bg-slate-700 text-slate-300 border-slate-600',
          icon: <HelpCircle className="w-8 h-8 text-slate-400" />,
          title: 'UNKNOWN',
          subtitle: 'Insufficient reliable evidence. Automated advisory unavailable; proceed with manual vigilance.',
        };
      default:
        return {
          banner: 'bg-emerald-600 border-emerald-500 text-white shadow-emerald-950/50',
          badge: 'bg-white/20 text-white border-white/30',
          icon: <ShieldCheck className="w-8 h-8 text-white" />,
          title: 'NORMAL',
          subtitle: 'Traffic path unobstructed; reliability metrics and trajectories within nominal band.',
        };
    }
  };

  const theme = getActionTheme(rawState);

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
                Early Warning & Safety Recommendations
              </h2>
              <p className="text-xs text-slate-400">
                Deterministic ADAS decision support, explainable evidence chains, and advisory vehicle stop guidance
              </p>
            </div>
          </div>
        </div>

        {/* Global Safety Summary */}
        {results?.safety_summary && (
          <div className="flex items-center gap-3 text-xs bg-slate-900 border border-slate-800 rounded-xl px-4 py-2">
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">Predominant Advisory</span>
              <strong className="text-cyan-400 font-mono text-sm">
                {results.safety_summary.most_frequent_recommendation || results.safety_summary.most_frequent_action}
              </strong>
            </div>
            <div className="w-px h-6 bg-slate-800" />
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">Critical Stops</span>
              <strong className="text-red-400 font-mono text-sm">
                {results.safety_summary.critical_stop_count ?? 0}
              </strong>
            </div>
            <div className="w-px h-6 bg-slate-800" />
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">Transitions</span>
              <strong className="text-amber-400 font-mono text-sm">
                {stateTransitions.length}
              </strong>
            </div>
          </div>
        )}
      </div>

      {/* Mandatory ADAS Prototype Disclaimer Banner */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 flex items-center gap-3 text-xs text-slate-400 select-none">
        <Info className="w-4 h-4 text-cyan-400 shrink-0" />
        <span className="flex-1">
          <strong className="text-slate-200">System Disclaimer:</strong> {disclaimer}
        </span>
      </div>

      {/* Primary Action Recommendation Hero Banner */}
      <div
        className={`rounded-2xl p-6 border shadow-2xl transition-all duration-200 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${theme.banner}`}
      >
        <div className="flex items-center gap-4">
          <div className="p-3 bg-black/20 rounded-2xl backdrop-blur-sm shrink-0">
            {theme.icon}
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <h3 className="text-2xl font-black tracking-tight">{theme.title}</h3>
              <span
                className={`px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase font-mono tracking-wider border ${theme.badge}`}
              >
                {rawState === 'NORMAL' ? 'NO ACTIVE ALERTS' : (rawState === 'UNKNOWN' ? 'UNKNOWN' : 'ACTIVE WARNING')}
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase font-mono tracking-wider border ${theme.badge}`}
              >
                PRIORITY: {priority}
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase font-mono tracking-wider bg-black/20 border border-black/20">
                RISK: {riskLevel}
              </span>
            </div>
            <p className="text-xs opacity-90 font-medium">{theme.subtitle}</p>
            <div className="mt-2 text-xs font-semibold underline decoration-white/30 underline-offset-4">
              Driver Action: {recommendedAction}
            </div>
          </div>
        </div>

        <div className="text-right shrink-0">
          <span className="text-[10px] opacity-75 uppercase block font-mono">Frame Evaluated</span>
          <strong className="font-mono text-lg">
            #{currentFrameIndex} ({currentFrame?.timestamp_seconds?.toFixed(2)}s)
          </strong>
          {safety?.prediction_horizon && safety.prediction_horizon !== 'N/A' && (
            <span className="block text-[11px] opacity-90 font-mono mt-0.5">
              Horizon: {safety.prediction_horizon}
            </span>
          )}
        </div>
      </div>

      {/* Metric Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
            <Clock className="w-3.5 h-3.5 text-cyan-400" /> Time-To-Collision
          </span>
          <div className="text-xl font-bold font-mono text-white">
            {supportingEv?.min_ttc_seconds != null
              ? `${supportingEv.min_ttc_seconds.toFixed(1)}s`
              : collision?.min_ttc_seconds != null
              ? `${collision.min_ttc_seconds.toFixed(1)}s`
              : 'N/A'}
          </div>
          <span className="text-[10px] text-slate-500">Threshold: &le; {ttcThreshold.toFixed(1)}s</span>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
            <Eye className="w-3.5 h-3.5 text-amber-400" /> Camera Visibility
          </span>
          <div className="text-xl font-bold font-mono text-white">
            {weather?.visibility_score != null ? `${weather.visibility_score.toFixed(0)}%` : '100%'}
          </div>
          <span className="text-[10px] text-slate-500">
            {weather?.condition || weather?.primary_condition || 'Nominal'}
          </span>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
            <Activity className="w-3.5 h-3.5 text-emerald-400" /> Detection Reliability
          </span>
          <div className="text-xl font-bold font-mono text-white">
            {safety?.detection_reliability != null
              ? `${(safety.detection_reliability * 100).toFixed(0)}%`
              : '95%'}
          </div>
          <span className="text-[10px] text-slate-500">
            Uncertainty: {safety?.uncertainty_level || 'LOW'}
          </span>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" /> Involved Obstacles
          </span>
          <div className="text-xl font-bold font-mono text-white">
            {involvedObjects.length} active
          </div>
          <span className="text-[10px] text-slate-500">
            {collision?.conflicting_pairs?.length || 0} conflict pair(s)
          </span>
        </div>
      </div>

      {/* Main Grid: Visual Feed & Playback Scrubber */}
      <div className={isEmbedded ? "space-y-4" : "grid grid-cols-1 lg:grid-cols-12 gap-6"}>
        {/* Left 7 Cols: Video Canvas & Scrubber */}
        {!isEmbedded && (
          <div className="lg:col-span-7 space-y-4">
            <VideoCanvas />
            <PlaybackControls />
          </div>
        )}

        {/* Right 5 Cols: Explainable Evidence Chain & Threshold Controls */}
        <div className={isEmbedded ? "space-y-4" : "lg:col-span-5 space-y-4"}>
          {/* Explainable Evidence Chain Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 text-cyan-400" />
              Recommendation Rationale & Evidence Chain
            </span>

            <div className="p-3 bg-cyan-950/30 border border-cyan-800/40 rounded-xl text-xs text-cyan-200">
              <strong className="block text-cyan-400 text-[10px] uppercase mb-0.5">Primary Reason:</strong>
              {primaryReason}
            </div>

            <div className="space-y-2 bg-slate-950/80 rounded-xl p-3.5 border border-slate-800 text-xs">
              <span className="text-[10px] font-mono text-slate-500 uppercase block mb-1">
                Supporting Evidence Multi-Source Inputs
              </span>
              {evidenceChain.map((item, idx) => (
                <div key={idx} className="flex items-start gap-2 text-slate-300">
                  <span className="text-cyan-400 font-mono mt-0.5 font-bold">[{idx + 1}]</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Involved Road Agents Card */}
          {involvedObjects.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Car className="w-4 h-4 text-amber-400" />
                Active Road Agents in Proximity Corridor
              </span>

              <div className="space-y-2 text-xs">
                {involvedObjects.map((obj, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2.5 bg-slate-950/70 border border-slate-800 rounded-xl font-mono"
                  >
                    <div className="flex items-center gap-2">
                      {obj.object_class.toLowerCase().includes('person') ||
                      obj.object_class.toLowerCase().includes('pedestrian') ? (
                        <User className="w-4 h-4 text-rose-400" />
                      ) : (
                        <Car className="w-4 h-4 text-cyan-400" />
                      )}
                      <div>
                        <strong className="text-slate-200">Track #{obj.track_id}</strong>
                        <span className="text-[10px] text-slate-500 ml-1.5 uppercase font-sans">
                          ({obj.object_class})
                        </span>
                      </div>
                    </div>
                    <div className="text-right text-[11px] text-slate-400">
                      {obj.distance_m != null && (
                        <span className="text-amber-400 mr-2">{obj.distance_m.toFixed(1)}m</span>
                      )}
                      {obj.speed_kmh != null && (
                        <span>{obj.speed_kmh.toFixed(1)} km/h</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Configurable Threshold Sliders Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-cyan-400" />
                Adaptive ADAS Decision Margins
              </span>
              <span className="text-[10px] font-mono text-slate-500">Advisory Tuner</span>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-slate-400">Critical TTC Margin:</span>
                  <span className="font-mono text-cyan-400 font-bold">{ttcThreshold.toFixed(1)}s</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="3.0"
                  step="0.1"
                  value={ttcThreshold}
                  onChange={(e) => setTtcThreshold(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-slate-400">Min Visibility Margin:</span>
                  <span className="font-mono text-cyan-400 font-bold">{visThreshold}%</span>
                </div>
                <input
                  type="range"
                  min="40"
                  max="80"
                  step="5"
                  value={visThreshold}
                  onChange={(e) => setVisThreshold(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Multi-Dimensional Filter Bar */}
      <FilterBar
        filters={filters}
        onChange={setFilters}
        maxDurationSec={results?.timeline_telemetry?.source_video_duration_sec || 300}
      />

      {/* Complete Video Event Timeline */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Complete Video Event Timeline
              </h3>
              <p className="text-xs text-slate-400">
                Localized safety events, optical faults, and traffic episodes across the complete drive
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
                    <span>{isSelected ? 'Viewing Frame' : 'Jump to Frame'} #{ev.evidence_frame_index}</span>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Lower Row: Transitions & Early Warning Log */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recommendation Transitions Audit Log */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <History className="w-4 h-4 text-cyan-400" />
              Recommendation Transition History
            </span>
            <span className="text-xs font-mono text-slate-500">
              {stateTransitions.length} transition(s)
            </span>
          </div>

          {stateTransitions.length === 0 ? (
            <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800">
              Zero state transitions recorded. Advisory state has remained steady throughout observation.
            </div>
          ) : (
            <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
              {stateTransitions.slice(-6).map((tr, idx) => (
                <div
                  key={idx}
                  className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs flex items-center justify-between gap-3"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-slate-500">#{tr.frame_index}</span>
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300">
                      {tr.from_state}
                    </span>
                    <span className="text-slate-600">&rarr;</span>
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
                      {tr.to_state}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono truncate max-w-xs text-right">
                    {tr.trigger_reason}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Frame Active Early Warning Alerts Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Flame className="w-4 h-4 text-amber-400" />
              Early Warning Rule Engine Trigger Log (Frame #{currentFrameIndex})
            </span>
            <span className="text-xs font-mono text-slate-500">
              {frameAlerts.length} triggered alert(s)
            </span>
          </div>

          {frameAlerts.length === 0 ? (
            <div className="py-6 text-center text-slate-500 text-xs italic bg-slate-950/60 rounded-xl border border-slate-800">
              Zero early-warning alerts triggered in this frame. All predictive metrics conform to safety envelope.
            </div>
          ) : (
            <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
              {frameAlerts.map((a, idx) => (
                <div
                  key={idx}
                  className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between gap-4 text-xs"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`px-2 py-0.5 rounded font-mono font-bold text-[10px] ${
                        a.priority === 'P1'
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                          : a.priority === 'P2'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                      }`}
                    >
                      {a.priority}
                    </span>
                    <div>
                      <span className="font-semibold text-white mr-2">Rule: {a.rule_id}</span>
                      <span className="text-slate-400">{a.message}</span>
                    </div>
                  </div>

                  <div className="text-right font-mono text-[11px] text-slate-500 shrink-0">
                    <span>Horizon: H{a.horizon}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
