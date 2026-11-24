import React, { useState } from 'react';
import { useJob } from '../context/JobContext';
import { VideoCanvas } from './VideoCanvas';
import { PlaybackControls } from './PlaybackControls';
import { InteractiveTimeline } from './InteractiveTimeline';
import { VideoUploadSection } from './VideoUploadSection';
import { ShieldAlert, Upload, ChevronDown, ChevronUp } from 'lucide-react';

export const PersistentVideoWorkspace: React.FC = () => {
  const { activeJobId, activeJob, results } = useJob();
  const [showUploadDrawer, setShowUploadDrawer] = useState(false);

  const hasVideoLoaded = Boolean(activeJobId || results);

  return (
    <div className="space-y-4">
      {/* Stream Failure Alert Banner */}
      {activeJob?.status === 'FAILED' && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-rose-500/20 text-rose-400 rounded-xl shrink-0">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-rose-300">Pipeline Execution Halted on Selected Stream</h4>
              <p className="text-xs text-rose-400/80 font-mono mt-0.5">
                {activeJob.error_message || activeJob.message || 'Processing was halted due to an unhandled pipeline error.'}
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-rose-400 bg-rose-950 px-3 py-1 rounded-lg border border-rose-800 shrink-0 self-start sm:self-auto">
            Stream: {activeJob.filename}
          </span>
        </div>
      )}

      {/* Video Upload Section Drawer / Direct view */}
      {!hasVideoLoaded ? (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wide flex items-center gap-2">
              <Upload className="w-4 h-4 text-cyan-400" />
              Primary Video Input & Stream Ingestion
            </h3>
            <span className="text-xs text-slate-500 font-mono">Continuous Video Perception Mode</span>
          </div>
          <VideoUploadSection />
        </div>
      ) : (
        <div>
          {/* Quick Upload Toggle Bar */}
          <div className="flex items-center justify-between bg-slate-900/60 border border-slate-800/80 rounded-xl px-3.5 py-1.5 mb-2">
            <div className="flex items-center gap-2 text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-slate-400">Active Stream:</span>
              <strong className="text-cyan-300 font-mono text-[11px] truncate max-w-[260px]">
                {activeJob?.filename || 'Stream Telemetry Loaded'}
              </strong>
            </div>

            <button
              type="button"
              onClick={() => setShowUploadDrawer((prev) => !prev)}
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
            >
              <Upload className="w-3 h-3 text-cyan-400" />
              <span>{showUploadDrawer ? 'Hide Upload Panel' : 'Upload Another Video'}</span>
              {showUploadDrawer ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>

          {showUploadDrawer && (
            <div className="mb-4 bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
              <VideoUploadSection />
            </div>
          )}
        </div>
      )}

      {/* Persistent Canvas and Video Element (Never Unmounts Across Navigation!) */}
      <div className="space-y-3">
        <VideoCanvas />
        <PlaybackControls />
        <InteractiveTimeline />
      </div>
    </div>
  );
};
