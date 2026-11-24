import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  CloudSun,
  Crosshair,
  TrendingUp,
  AlertTriangle,
  OctagonAlert,
  Shield,
  Cpu,
} from 'lucide-react';
import { cn } from '../utils/cn';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  badge?: string;
}

const PRIMARY_NAV_ITEMS: NavItem[] = [
  {
    name: 'Camera & Weather',
    path: '/camera-weather',
    icon: CloudSun,
    description: 'Optical visibility, rain, fog, glare & degradation',
  },
  {
    name: 'Object Detection',
    path: '/object-detection',
    icon: Crosshair,
    description: 'YOLO detection, ByteTrack & reliability forecasting',
  },
  {
    name: 'Trajectory Prediction',
    path: '/trajectory-prediction',
    icon: TrendingUp,
    description: 'Multi-horizon future paths & kinematic telemetry',
  },
  {
    name: 'Collision Risk',
    path: '/collision-risk',
    icon: AlertTriangle,
    description: 'Time-To-Collision (TTC) & perception uncertainty',
  },
  {
    name: 'Safety Alerts',
    path: '/safety-alerts',
    icon: OctagonAlert,
    description: 'Actionable speed recommendations & early warnings',
    badge: 'ADAS',
  },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col h-screen shrink-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800/80 flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-950/40">
          <Shield className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-bold text-white tracking-wide text-base">PerceptGuard AI</h1>
            <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-800/50 text-cyan-400">
              v2.0
            </span>
          </div>
          <p className="text-xs text-slate-400">Predictive Perception Reliability</p>
        </div>
      </div>

      {/* Navigation Links — Exactly Five Features */}
      <nav className="flex-1 p-3 space-y-1.5 overflow-y-auto">
        <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Primary ADAS Modules
        </div>
        {PRIMARY_NAV_ITEMS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  'group flex items-start gap-3 px-3 py-2.5 rounded-lg transition-all duration-150',
                  isActive
                    ? 'bg-cyan-500/10 border border-cyan-500/30 text-white shadow-sm shadow-cyan-950/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <div
                    className={cn(
                      'p-1.5 rounded-md mt-0.5 transition-colors',
                      isActive
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'bg-slate-800 text-slate-400 group-hover:text-slate-200 group-hover:bg-slate-700'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium tracking-tight">
                        <span className="text-slate-500 mr-1.5 text-xs font-mono">{idx + 1}.</span>
                        {item.name}
                      </span>
                      {item.badge && (
                        <span className="text-[9px] font-bold px-1.5 py-0.2 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400">
                          {item.badge}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-500 truncate group-hover:text-slate-400 transition-colors">
                      {item.description}
                    </p>
                  </div>
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Hardware & Runtime Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/50">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
          <span className="flex items-center gap-1.5 font-medium text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            Runtime Engine
          </span>
          <span className="text-[11px] text-emerald-400 font-mono flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            CPU Native
          </span>
        </div>
        <div className="text-[11px] text-slate-500 space-y-0.5">
          <p>YOLO11n + ByteTrack + LSTM</p>
          <p>FastAPI Engine @ port 8000</p>
        </div>
      </div>
    </aside>
  );
};
