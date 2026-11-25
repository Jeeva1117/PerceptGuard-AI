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
  id: string;
  name: string;
  shortName: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}

const PRIMARY_NAV_ITEMS: NavItem[] = [
  {
    id: 'camera-weather',
    name: 'Camera & Weather',
    shortName: 'Weather',
    path: '/camera-weather',
    icon: CloudSun,
    description: 'Optical visibility, weather conditions & camera health',
  },
  {
    id: 'object-detection',
    name: 'Object Detection',
    shortName: 'Detection',
    path: '/object-detection',
    icon: Crosshair,
    description: 'Road user detection & ByteTrack continuous identities',
  },
  {
    id: 'trajectory-prediction',
    name: 'Trajectory Prediction',
    shortName: 'Trajectory',
    path: '/trajectory-prediction',
    icon: TrendingUp,
    description: 'Multi-horizon future path projection & kinematics',
  },
  {
    id: 'collision-risk',
    name: 'Collision Risk',
    shortName: 'Risk',
    path: '/collision-risk',
    icon: AlertTriangle,
    description: 'Time-To-Collision (TTC) & spatial corridor conflicts',
  },
  {
    id: 'safety-alerts',
    name: 'Safety Alerts',
    shortName: 'Alerts',
    path: '/safety-alerts',
    icon: OctagonAlert,
    description: 'Evidence-based driver advisories & event timeline',
  },
];

export const Sidebar: React.FC = () => {
  return (
    <aside
      className="w-16 lg:w-[68px] bg-slate-900 border-r border-slate-800 flex flex-col h-screen shrink-0 select-none z-30"
      aria-label="Primary Navigation Rail"
    >
      {/* Brand Icon Header */}
      <div className="h-16 flex items-center justify-center border-b border-slate-800/80 relative group">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-950/40 cursor-default">
          <Shield className="w-5 h-5 text-white" />
        </div>

        {/* Floating Tooltip */}
        <div className="absolute left-full ml-3 hidden group-hover:flex flex-col px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white shadow-2xl z-50 pointer-events-none whitespace-nowrap">
          <span className="font-bold tracking-wide">PerceptGuard AI</span>
          <span className="text-[10px] text-slate-400 font-mono">v2.0 • Predictive Road Safety</span>
        </div>
      </div>

      {/* Navigation Rail — Exactly Five Primary Modules */}
      <nav className="flex-1 py-4 flex flex-col items-center space-y-2 overflow-y-auto">
        {PRIMARY_NAV_ITEMS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              aria-label={item.name}
              title={item.name}
              className={({ isActive }) =>
                cn(
                  'group relative flex items-center justify-center w-11 h-11 rounded-xl transition-all duration-150 outline-none focus-visible:ring-2 focus-visible:ring-cyan-400',
                  isActive
                    ? 'bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 shadow-md shadow-cyan-950/40'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/80 border border-transparent'
                )
              }
            >
              {({ isActive }) => (
                <>
                  {/* Left Active Accent Indicator Bar */}
                  {isActive && (
                    <span
                      className="absolute -left-2.5 lg:-left-3 top-2 bottom-2 w-1 bg-cyan-400 rounded-r shadow-[0_0_8px_rgba(34,211,238,0.8)]"
                      aria-hidden="true"
                    />
                  )}

                  <Icon className={cn('w-5 h-5 transition-transform duration-150 group-hover:scale-110', isActive ? 'text-cyan-400' : '')} />

                  {/* Accessible Floating Tooltip On Hover */}
                  <div className="absolute left-full ml-3 hidden group-hover:flex flex-col px-3 py-1.5 rounded-lg bg-slate-900/95 backdrop-blur-md border border-slate-700 text-xs text-white shadow-2xl z-50 pointer-events-none whitespace-nowrap">
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-500 font-mono font-bold">{idx + 1}.</span>
                      <span className="font-bold text-white">{item.name}</span>
                    </div>
                    <span className="text-[11px] text-slate-400 font-normal mt-0.5">
                      {item.description}
                    </span>
                  </div>
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Hardware & CPU Indicator Footer */}
      <div className="py-3 flex flex-col items-center justify-center border-t border-slate-800/80 relative group cursor-default">
        <div className="w-9 h-9 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center text-slate-400 group-hover:text-cyan-400 group-hover:border-cyan-500/40 transition-colors">
          <Cpu className="w-4 h-4" />
        </div>
        <span className="w-2 h-2 rounded-full bg-emerald-400 absolute bottom-3 right-3 border-2 border-slate-900" />

        {/* Floating Tooltip */}
        <div className="absolute left-full ml-3 bottom-2 hidden group-hover:flex flex-col px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white shadow-2xl z-50 pointer-events-none whitespace-nowrap">
          <span className="font-bold flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            CPU Native Execution
          </span>
          <span className="text-[10px] text-slate-400 font-mono mt-0.5">
            FastAPI Pipeline • Intel i3/i5 Compatible
          </span>
        </div>
      </div>
    </aside>
  );
};
