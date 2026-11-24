import React, { useMemo } from 'react';
import { useJob } from '../context/JobContext';

interface TimelineEventMarker {
  type: 'WEATHER' | 'PEDESTRIAN' | 'RISK' | 'ALERT';
  frameIndex: number;
  timestamp: number;
  label: string;
  severity: 'info' | 'warning' | 'danger';
  color: string;
}

export const InteractiveTimeline: React.FC = () => {
  const { results, currentTime, duration, seekToTimestamp, seekToFrame } = useJob();

  const totalFrames = results?.frames?.length || 0;
  const maxTime = duration > 0
    ? duration
    : (results?.frames && results.frames.length > 0
        ? results.frames[results.frames.length - 1].timestamp_seconds || 0
        : 0);

  // Extract critical timeline events across the whole processed video
  const eventMarkers = useMemo(() => {
    if (!results?.frames || results.frames.length === 0) return [];

    const markers: TimelineEventMarker[] = [];
    const frames = results.frames;

    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const ts = f.timestamp_seconds ?? (i * 0.033);

      // 1. Weather Degradation Events
      const weather = f.weather_intelligence;
      if (weather && (weather.is_degraded || (weather.visibility_score != null && weather.visibility_score < 70))) {
        // Sample every 5 frames if continuous to prevent clustering
        if (markers.length === 0 || markers[markers.length - 1].type !== 'WEATHER' || (i - markers[markers.length - 1].frameIndex) >= 4) {
          markers.push({
            type: 'WEATHER',
            frameIndex: i,
            timestamp: ts,
            label: `Atmospheric: ${weather.primary_condition || weather.condition || 'Degraded'} (${weather.visibility_score}%)`,
            severity: weather.visibility_score != null && weather.visibility_score < 50 ? 'danger' : 'warning',
            color: '#38bdf8', // Sky/cyan
          });
        }
      }

      // 2. Pedestrian Crossing / Incursion Events
      const pedAwareness = f.pedestrian_awareness;
      if (pedAwareness && pedAwareness.pedestrians_count > 0 && pedAwareness.active_pedestrians) {
        const crossingOrRisk = pedAwareness.active_pedestrians.some(
          (p) => p.state === 'PEDESTRIAN CROSSING DETECTED' || p.state === 'POTENTIAL PEDESTRIAN COLLISION RISK'
        );
        if (crossingOrRisk) {
          if (markers.length === 0 || markers[markers.length - 1].type !== 'PEDESTRIAN' || (i - markers[markers.length - 1].frameIndex) >= 4) {
            markers.push({
              type: 'PEDESTRIAN',
              frameIndex: i,
              timestamp: ts,
              label: `Pedestrian: ${pedAwareness.primary_state} (${pedAwareness.pedestrians_count} detected)`,
              severity: pedAwareness.primary_state.includes('COLLISION') ? 'danger' : 'warning',
              color: '#f97316', // Orange
            });
          }
        }
      }

      // 3. Collision Risk / High TTC Alerts
      const colRisk = f.collision_risk;
      const rec = f.safety_recommendation;
      if (colRisk && (colRisk.risk_state === 'HIGH' || colRisk.risk_state === 'CRITICAL')) {
        if (markers.length === 0 || markers[markers.length - 1].type !== 'RISK' || (i - markers[markers.length - 1].frameIndex) >= 3) {
          markers.push({
            type: 'RISK',
            frameIndex: i,
            timestamp: ts,
            label: `Risk Peak: ${colRisk.risk_state} (TTC: ${colRisk.min_ttc_seconds ? colRisk.min_ttc_seconds.toFixed(1) + 's' : 'N/A'})`,
            severity: 'danger',
            color: '#ef4444', // Red
          });
        }
      } else if (rec && (rec.urgency_level === 'WARNING' || rec.urgency_level === 'CRITICAL')) {
        if (markers.length === 0 || markers[markers.length - 1].type !== 'ALERT' || (i - markers[markers.length - 1].frameIndex) >= 4) {
          markers.push({
            type: 'ALERT',
            frameIndex: i,
            timestamp: ts,
            label: `Alert: ${rec.primary_action || rec.urgency_level}`,
            severity: rec.urgency_level === 'CRITICAL' ? 'danger' : 'warning',
            color: '#eab308', // Amber
          });
        }
      }
    }

    return markers;
  }, [results?.frames]);

  if (totalFrames === 0 || maxTime <= 0) {
    return null;
  }

  const playheadPercent = Math.min(100, Math.max(0, (currentTime / maxTime) * 100));

  const handleBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetTime = ratio * maxTime;
    seekToTimestamp(targetTime);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 shadow-lg select-none">
      <div className="flex items-center justify-between text-xs mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Interactive Safety & Perception Event Timeline
          </span>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">
            {eventMarkers.length} Events Detected
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[10px] font-medium text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-sky-400" />
            Weather
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-orange-400" />
            Pedestrian
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            Collision Risk
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            Safety Alert
          </span>
        </div>
      </div>

      {/* Interactive Timeline Bar */}
      <div
        onClick={handleBarClick}
        className="relative h-7 w-full bg-slate-950 rounded-lg cursor-pointer border border-slate-800 overflow-hidden group hover:border-slate-700 transition-colors"
        title="Click anywhere to jump to timestamp"
      >
        {/* Subtle grid background */}
        <div className="absolute inset-0 flex justify-between pointer-events-none opacity-20">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="h-full w-px bg-slate-700" />
          ))}
        </div>

        {/* Progress Fill */}
        <div
          className="absolute top-0 bottom-0 left-0 bg-cyan-500/10 border-r border-cyan-500/30 transition-all pointer-events-none"
          style={{ width: `${playheadPercent}%` }}
        />

        {/* Event Markers on Timeline */}
        {eventMarkers.map((marker, idx) => {
          const markerPercent = (marker.timestamp / maxTime) * 100;
          return (
            <div
              key={idx}
              onClick={(e) => {
                e.stopPropagation();
                seekToFrame(marker.frameIndex);
              }}
              className="absolute top-1 bottom-1 w-1.5 rounded-full hover:scale-150 hover:z-20 transition-transform cursor-pointer"
              style={{
                left: `${markerPercent}%`,
                backgroundColor: marker.color,
                boxShadow: `0 0 6px ${marker.color}`,
              }}
              title={`Jump to ${marker.timestamp.toFixed(2)}s: ${marker.label}`}
            />
          );
        })}

        {/* Playhead Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_8px_#00f0ff] pointer-events-none transition-all"
          style={{ left: `${playheadPercent}%` }}
        >
          <div className="w-2 h-2 -ml-[3px] -mt-0.5 bg-cyan-300 rounded-full shadow" />
        </div>
      </div>

      {/* Timestamp labels */}
      <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1 px-0.5">
        <span>00:00.00</span>
        <span>Click markers or timeline to seek instant event telemetry</span>
        <span>
          {Math.floor(maxTime / 60)
            .toString()
            .padStart(2, '0')}
          :
          {(maxTime % 60).toFixed(2).padStart(5, '0')}
        </span>
      </div>
    </div>
  );
};
