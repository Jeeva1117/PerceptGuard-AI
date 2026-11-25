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
  const { currentFrame } = useJob();

  // Top Status Banner: Driving Status, Collision Status, Risk Level
  const hasData = Boolean(results && results.frames && results.frames.length > 0);
  
  let drivingStatus: 'NORMAL' | 'CAUTION' | 'HAZARD' | 'UNKNOWN' = 'UNKNOWN';
  let collisionStatus: string = results?.final_summary?.collision_risk || 'UNKNOWN';
  let riskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL' | 'UNKNOWN' = 'UNKNOWN';

  if (hasData && currentFrame) {
    const colRisk = currentFrame.collision_risk;
    const safety = currentFrame.safety_recommendation;

    riskLevel = (colRisk?.risk_state as any) || 'LOW';

    if (results?.final_summary?.collision_risk) {
      collisionStatus = results.final_summary.collision_risk;
    } else if (colRisk?.collision_decision) {
      collisionStatus = colRisk.collision_decision;
    } else {
      const hasCollision = riskLevel === 'CRITICAL' || (colRisk?.conflicting_pairs && colRisk.conflicting_pairs.length > 0);
      if (hasCollision || safety?.urgency_level === 'CRITICAL') {
        collisionStatus = 'POTENTIAL COLLISION RISK';
      } else {
        collisionStatus = 'NO COLLISION RISK';
      }
    }

    const recState: string = safety?.recommendation_state || 'NORMAL';
    if (riskLevel === 'CRITICAL' || safety?.urgency_level === 'CRITICAL' || recState === 'STOP RECOMMENDED' || recState === 'HAZARD' || collisionStatus === 'COLLISION OBSERVED') {
      drivingStatus = 'HAZARD';
    } else if (riskLevel === 'HIGH' || riskLevel === 'MODERATE' || safety?.urgency_level === 'WARNING' || recState === 'CAUTION' || recState === 'WARNING' || recState === 'REDUCE SPEED' || recState === 'PREPARE TO STOP' || collisionStatus === 'POTENTIAL COLLISION RISK') {
      drivingStatus = 'CAUTION';
    } else {
      drivingStatus = 'NORMAL';
    }
  } else if (hasData) {
    drivingStatus = 'NORMAL';
    collisionStatus = results?.final_summary?.collision_risk || 'NO COLLISION RISK';
    riskLevel = 'LOW';
  }

  const getDrivingBadgeColor = (s: string) => {
    switch (s) {
      case 'NORMAL':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40';
      case 'CAUTION':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/40';
      case 'HAZARD':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/50 animate-pulse';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  const getCollisionBadgeColor = (s: string) => {
    switch (s) {
      case 'NO COLLISION RISK':
      case 'NO COLLISION RISK DETECTED':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40';
      case 'NEAR PASS':
        return 'bg-cyan-500/15 text-cyan-400 border-cyan-500/40';
      case 'POTENTIAL COLLISION RISK':
      case 'COLLISION RISK DETECTED':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/40 animate-pulse';
      case 'COLLISION OBSERVED':
        return 'bg-rose-500/25 text-rose-400 border-rose-500/50 animate-pulse';
      case 'UNKNOWN':
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  const getRiskBadgeColor = (s: string) => {
    switch (s) {
      case 'LOW':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40';
      case 'MODERATE':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/40';
      case 'HIGH':
        return 'bg-orange-500/15 text-orange-400 border-orange-500/40';
      case 'CRITICAL':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/50 animate-pulse';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  return (
    <>
      <header className="h-16 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-6 flex items-center justify-between shrink-0 select-none gap-4">
        {/* Left Section: Active Job Selector & Status */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-400">Target Stream:</span>
            <select
              value={activeJobId || ''}
              onChange={(e) => selectJob(e.target.value)}
              className="bg-slate-950 border border-slate-700/80 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono max-w-[220px] truncate"
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
            <div className="hidden xl:flex items-center gap-2 text-xs">
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
        </div>

        {/* Center Section: Master Standardized Status Banner */}
        <div className="flex items-center gap-2.5 overflow-x-auto py-1">
          {/* Driving Status */}
          <div className="flex items-center gap-1.5 bg-slate-950/70 border border-slate-800 px-2.5 py-1 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Driving:</span>
            <span className={`px-2 py-0.5 rounded-md text-[11px] font-extrabold font-mono border ${getDrivingBadgeColor(drivingStatus)}`}>
              {drivingStatus}
            </span>
          </div>

          {/* Collision Status */}
          <div className="flex items-center gap-1.5 bg-slate-950/70 border border-slate-800 px-2.5 py-1 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Collision:</span>
            <span className={`px-2 py-0.5 rounded-md text-[11px] font-extrabold font-mono border ${getCollisionBadgeColor(collisionStatus)}`}>
              {collisionStatus}
            </span>
          </div>

          {/* Risk Level */}
          <div className="flex items-center gap-1.5 bg-slate-950/70 border border-slate-800 px-2.5 py-1 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Risk:</span>
            <span className={`px-2 py-0.5 rounded-md text-[11px] font-extrabold font-mono border ${getRiskBadgeColor(riskLevel)}`}>
              {riskLevel}
            </span>
          </div>
        </div>

        {/* Right Section: Telemetry Counters, API Indicator & Upload Button */}
        <div className="flex items-center gap-3 shrink-0">
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

      {/* Concise Master Analysis Result Banner (Only when analysis successfully completed) */}
      {results?.final_summary && activeJob?.status !== 'FAILED' && activeJob?.status !== 'CANCELLED' && activeJob?.status !== 'PROCESSING' && (
        <div className="bg-slate-950/95 border-b border-slate-800 px-6 py-2 flex flex-wrap items-center justify-between text-xs font-mono text-slate-300 gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-bold text-white flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              VIDEO ANALYSIS COMPLETED
            </span>
            <span className="text-slate-600">|</span>
            <span>
              Duration: <strong className="text-white">{results.final_summary.video_duration_seconds.toFixed(1)}s</strong>
            </span>
            <span className="text-slate-600">|</span>
            <span>
              Road Users: Car: <strong className="text-white">{results.final_summary.road_user_counts.car}</strong>, Motorcycle: <strong className="text-white">{results.final_summary.road_user_counts.motorcycle}</strong>, Bus: <strong className="text-white">{results.final_summary.road_user_counts.bus}</strong>, Truck: <strong className="text-white">{results.final_summary.road_user_counts.truck}</strong>, Bicycle: <strong className="text-white">{results.final_summary.road_user_counts.bicycle}</strong>, Pedestrian: <strong className="text-white">{results.final_summary.road_user_counts.pedestrian}</strong>
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-slate-600">|</span>
            <span>
              Collision Risk:{' '}
              <span className={`px-2 py-0.5 rounded text-[11px] font-bold border ${getCollisionBadgeColor(results.final_summary.collision_risk)}`}>
                {results.final_summary.collision_risk}
              </span>
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400 italic font-sans max-w-md truncate" title={results.final_summary.reason}>
              Reason: {results.final_summary.reason}
            </span>
          </div>
        </div>
      )}

      {/* Video Upload Modal */}
      <VideoUploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  );
};
