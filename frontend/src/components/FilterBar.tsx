import React from 'react';
import { RotateCcw, Search, SlidersHorizontal } from 'lucide-react';

export interface FilterState {
  eventType: string; // 'ALL' | 'WEATHER_FAULT' | 'COLLISION_HAZARD' | 'SAFETY_ADVISORY' | 'NORMAL_DRIVING'
  severity: string;  // 'ALL' | 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW' | 'INFO'
  minTimeSec: number;
  maxTimeSec: number;
  objectClass: string; // 'ALL' | 'car' | 'truck' | 'bus' | 'person' | 'motorcycle' | 'bicycle'
  trackId: string;
  minConfidence: number; // 0.0 to 1.0
  visibilityCondition: string; // 'ALL' | 'CLEAR' | 'RAIN' | 'FOG' | 'BLUR' | 'SUN_GLARE' | 'LOW_LIGHT' | 'OBSTRUCTION'
}

export const DEFAULT_FILTER_STATE: FilterState = {
  eventType: 'ALL',
  severity: 'ALL',
  minTimeSec: 0,
  maxTimeSec: 9999,
  objectClass: 'ALL',
  trackId: '',
  minConfidence: 0.0,
  visibilityCondition: 'ALL',
};

interface FilterBarProps {
  filters: FilterState;
  onChange: (updated: FilterState) => void;
  maxDurationSec?: number;
  availableClasses?: string[];
  showObjectFilters?: boolean;
  showWeatherFilters?: boolean;
  showSeverityFilters?: boolean;
  showEventTypeFilters?: boolean;
}

export const FilterBar: React.FC<FilterBarProps> = ({
  filters,
  onChange,
  maxDurationSec = 300,
  availableClasses = ['car', 'truck', 'bus', 'person', 'motorcycle', 'bicycle'],
  showObjectFilters = true,
  showWeatherFilters = true,
  showSeverityFilters = true,
  showEventTypeFilters = true,
}) => {
  // Count active non-default filters
  let activeCount = 0;
  if (filters.eventType !== 'ALL') activeCount++;
  if (filters.severity !== 'ALL') activeCount++;
  if (filters.minTimeSec > 0 || filters.maxTimeSec < maxDurationSec) activeCount++;
  if (filters.objectClass !== 'ALL') activeCount++;
  if (filters.trackId.trim() !== '') activeCount++;
  if (filters.minConfidence > 0.0) activeCount++;
  if (filters.visibilityCondition !== 'ALL') activeCount++;

  const handleReset = () => {
    onChange({
      ...DEFAULT_FILTER_STATE,
      maxTimeSec: maxDurationSec,
    });
  };

  const update = (patch: Partial<FilterState>) => {
    onChange({ ...filters, ...patch });
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg mb-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-white">
            Multi-Dimensional Filter Bar
          </span>
          {activeCount > 0 && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
              {activeCount} Active
            </span>
          )}
        </div>

        <button
          type="button"
          onClick={handleReset}
          disabled={activeCount === 0}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-colors bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed border border-slate-700"
          title="Reset all filter fields to nominal defaults"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Filters</span>
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
        {/* 1. Event Type Filter */}
        {showEventTypeFilters && (
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-slate-400">Event Type</label>
            <select
              value={filters.eventType}
              onChange={(e) => update({ eventType: e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="ALL">All Event Types</option>
              <option value="WEATHER_FAULT">Weather & Optical Fault</option>
              <option value="COLLISION_HAZARD">Collision Hazard</option>
              <option value="SAFETY_ADVISORY">Safety Advisory</option>
              <option value="NORMAL_DRIVING">Normal Driving</option>
            </select>
          </div>
        )}

        {/* 2. Severity Filter */}
        {showSeverityFilters && (
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-slate-400">Severity</label>
            <select
              value={filters.severity}
              onChange={(e) => update({ severity: e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MODERATE">Moderate</option>
              <option value="LOW">Low</option>
              <option value="INFO">Info</option>
            </select>
          </div>
        )}

        {/* 3. Visibility Condition Filter */}
        {showWeatherFilters && (
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-slate-400">Visibility Condition</label>
            <select
              value={filters.visibilityCondition}
              onChange={(e) => update({ visibilityCondition: e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="ALL">All Conditions</option>
              <option value="CLEAR">Clear</option>
              <option value="RAIN">Rain / Wet Windshield</option>
              <option value="FOG">Fog / Atmospheric Haze</option>
              <option value="SUN_GLARE">Sun Glare / Blinding</option>
              <option value="BLUR">Blur / Defocus</option>
              <option value="LOW_LIGHT">Low Light / Dim</option>
              <option value="OBSTRUCTION">Camera Obstruction</option>
            </select>
          </div>
        )}

        {/* 4. Object Class Filter */}
        {showObjectFilters && (
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-slate-400">Object Class</label>
            <select
              value={filters.objectClass}
              onChange={(e) => update({ objectClass: e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="ALL">All Object Classes</option>
              {availableClasses.map((cls) => (
                <option key={cls} value={cls}>
                  {cls.charAt(0).toUpperCase() + cls.slice(1)}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* 5. Track ID Filter */}
        {showObjectFilters && (
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-slate-400">Track ID</label>
            <div className="relative">
              <input
                type="text"
                placeholder="e.g. 1, 4"
                value={filters.trackId}
                onChange={(e) => update({ trackId: e.target.value })}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-7 pr-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
              />
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2 top-2" />
            </div>
          </div>
        )}

        {/* 6. Confidence Threshold */}
        {showObjectFilters && (
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-[11px] font-medium text-slate-400">
              <span>Min Conf</span>
              <span className="font-mono text-cyan-400">{Math.round(filters.minConfidence * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="0.9"
              step="0.05"
              value={filters.minConfidence}
              onChange={(e) => update({ minConfidence: Number(e.target.value) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 mt-2"
            />
          </div>
        )}
      </div>
    </div>
  );
};
