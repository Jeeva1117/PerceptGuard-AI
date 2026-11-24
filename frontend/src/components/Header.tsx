import React, { useState } from 'react';
import {
  Upload,
  RefreshCw,
  CheckCircle2,
  Clock,
  Film,
  Layers,
  ShieldAlert,
  Trash2,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { VideoUploadModal } from './VideoUploadModal';

export const Header: React.FC = () => {
  const {
    jobs,
    activeJobId,
    activeJob,
    results,
    isBackendOnline,
    isLoading,
    refreshJobs,
    selectJob,
    deleteJobById,
  } = useJob();

  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <header className="h-16 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-6 flex items-center justify-between shrink-0 select-none">
        {/* Left Section: Active Job Selector & Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-400">Target Stream:</span>
            <select
              value={activeJobId || ''}
              onChange={(e) => selectJob(e.target.value)}
              className="bg-slate-950 border border-slate-700/80 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono max-w-[280px] truncate"
            >
              <option value="">+ New Video Upload / Clean State</option>
              {jobs.map((j) => {
                const displayName = j.filename.startsWith('up_')
                  ? j.filename.slice(12)
                  : j.filename;
                const framesText = j.frames_processed > 0 ? ` (${j.frames_processed}f)` : '';
                return (
                  <option key={j.job_id} value={j.job_id}>
                    [{j.status}] {displayName}{framesText}
                  </option>
                );
              })}
            </select>

            {activeJobId && (
              <button
                type="button"
                onClick={() => {
                  const name = activeJob?.filename || activeJobId;
                  if (window.confirm(`Delete job "${name}" and all associated results?`)) {
                    deleteJobById(activeJobId);
                  }
                }}
                title="Delete this historical job from storage"
                className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-slate-800 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Active Job State Pill */}
          {activeJob && (
            <div className="hidden lg:flex items-center gap-2 text-xs">
              <span
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${
                  activeJob.status === 'COMPLETED'
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                    : activeJob.status === 'PROCESSING'
                    ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 animate-pulse'
                    : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                }`}
              >
                {activeJob.status === 'COMPLETED' ? (
                  <CheckCircle2 className="w-3 h-3" />
                ) : (
                  <Clock className="w-3 h-3" />
                )}
                {activeJob.status}
              </span>
            </div>
          )}

          {/* Calibration Status Badge */}
          <span className="hidden xl:inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700 text-[10px] font-mono text-cyan-300">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            {results?.calibration_status || 'PROVISIONAL CALIBRATION'}
          </span>
        </div>

        {/* Right Section: Telemetry Counters, API Indicator & Upload Button */}
        <div className="flex items-center gap-3">
          {/* Quick Metrics from Current Job */}
          {results && results.frames && results.frames.length > 0 && (
            <div className="hidden md:flex items-center gap-3 text-xs text-slate-400 pr-3 border-r border-slate-800">
              <span className="flex items-center gap-1">
                <Film className="w-3.5 h-3.5 text-slate-500" />
                <strong className="text-white font-mono">{results.frames_processed ?? results.frames?.length ?? 0}</strong> frames
              </span>
              <span className="flex items-center gap-1">
                <Layers className="w-3.5 h-3.5 text-slate-500" />
                <strong className="text-white font-mono">{results.unique_tracks ?? 0}</strong> tracks
              </span>
              <span className="flex items-center gap-1">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
                <strong className="text-amber-400 font-mono">{results.total_alerts ?? 0}</strong> alerts
              </span>
            </div>
          )}

          {/* Backend Online / Offline Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-[11px]">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendOnline ? 'bg-emerald-500' : 'bg-red-500 animate-ping'
              }`}
            />
            <span className={isBackendOnline ? 'text-slate-300' : 'text-red-400 font-semibold'}>
              {isBackendOnline ? 'Backend Online' : 'Backend Disconnected'}
            </span>
          </div>

          {/* Refresh Jobs Button */}
          <button
            onClick={() => refreshJobs()}
            disabled={isLoading}
            title="Refresh jobs and status"
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
          </button>

          {/* Upload Video Button */}
          <button
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium transition-all shadow-sm shadow-cyan-950/40"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Video</span>
          </button>
        </div>
      </header>

      {/* Video Upload Modal */}
      <VideoUploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  );
};
