import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Film,
  Compass,
  Volume2,
  VolumeX,
  Layers,
  Route,
  Video,
} from 'lucide-react';
import { useJob } from '../context/JobContext';
import { getSourceVideoUrl, getAnnotatedVideoUrl, getFrameImageUrl } from '../services/api';

interface VideoCanvasProps {
  showTrajectories?: boolean;
  showDetections?: boolean;
  highlightTrackId?: number | null;
}

// Deterministic vibrant color for track IDs
const TRACK_PALETTE = [
  '#00f0ff', // Cyan
  '#00ff66', // Neon Green
  '#ffb700', // Amber
  '#ff007f', // Magenta
  '#a855f7', // Purple
  '#3b82f6', // Blue
  '#f97316', // Orange
  '#10b981', // Emerald
  '#ec4899', // Pink
  '#06b6d4', // Sky
];

function getTrackColor(trackId: number): string {
  const safeId = Math.abs(trackId);
  return TRACK_PALETTE[safeId % TRACK_PALETTE.length];
}

export const VideoCanvas: React.FC<VideoCanvasProps> = ({
  showTrajectories: initialShowTrajectories = true,
  showDetections: initialShowDetections = true,
  highlightTrackId = null,
}) => {
  const {
    activeJobId,
    results,
    currentFrame,
    currentFrameIndex,
    currentTime,
    playbackMode,
    isVoiceMuted,
    videoRef,
    setPlaybackMode,
    toggleVoiceMute,
    handleTimeUpdate,
    handleDurationChange,
    setIsPlaying,
  } = useJob();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [videoError, setVideoError] = useState(false);
  const [showOverlays, setShowOverlays] = useState(initialShowDetections);
  const [showTrails, setShowTrails] = useState(true);
  const [showTrajectories, setShowTrajectories] = useState(initialShowTrajectories);
  const [useFallbackImage, setUseFallbackImage] = useState(false);

  // Reset video error state on job or mode switch
  useEffect(() => {
    setVideoError(false);
    setUseFallbackImage(false);
  }, [activeJobId, playbackMode]);

  // Resolve media URLs
  const sourceVideoUrl = activeJobId ? getSourceVideoUrl(activeJobId) : '';
  const annotatedVideoUrl = activeJobId ? getAnnotatedVideoUrl(activeJobId) : '';
  const activeVideoUrl = playbackMode === 'ORIGINAL_LIVE' ? sourceVideoUrl : annotatedVideoUrl;

  const frame = currentFrame || (results?.frames && results.frames.length > 0 ? results.frames[0] : null);

  // Draw overlays on transparent canvas synced with currentFrame
  const renderCanvasOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Synchronize canvas resolution with video or container
    const width = video?.videoWidth || canvas.clientWidth || 1280;
    const height = video?.videoHeight || canvas.clientHeight || 720;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.clearRect(0, 0, width, height);

    // If pre-rendered mode or overlays toggled off, skip drawing boxes/trails
    if (playbackMode === 'ANNOTATED_PRERENDERED' || !showOverlays || !frame) {
      return;
    }

    // 1. Draw motion history trails
    if (showTrails && frame.trajectories) {
      frame.trajectories.forEach((traj) => {
        const hist = traj.history || traj.observed_history;
        if (!hist || hist.length < 2) return;

        const color = getTrackColor(traj.track_id);
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        hist.forEach((pt, i) => {
          if (i === 0) {
            ctx.moveTo(pt.x, pt.y);
          } else {
            ctx.lineTo(pt.x, pt.y);
          }
        });
        ctx.stroke();

        // Draw trail dots
        hist.forEach((pt) => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        });
      });
    }

    // 2. Draw predicted future trajectories
    if (showTrajectories && frame.trajectories) {
      frame.trajectories.forEach((traj) => {
        const horizons = traj.predicted_horizons || traj.horizons;
        if (!horizons) return;

        const cx = traj.center?.[0] ?? (traj.bbox[0] + traj.bbox[2]) / 2;
        const cy = traj.center?.[1] ?? (traj.bbox[1] + traj.bbox[3]) / 2;

        const h1 = horizons['h1'];
        const h3 = horizons['h3'];
        const h5 = horizons['h5'];

        const futurePoints = [h1, h3, h5].filter(Boolean);
        if (futurePoints.length === 0) return;

        // Draw dashed prediction line
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = '#ffd700'; // Gold/Yellow
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        futurePoints.forEach((p) => {
          if (p) ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
        ctx.restore();

        // Arrowhead on final horizon point
        const endPt = futurePoints[futurePoints.length - 1];
        if (endPt) {
          ctx.save();
          ctx.fillStyle = '#ffd700';
          ctx.beginPath();
          ctx.arc(endPt.x, endPt.y, 4.5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.restore();
        }
      });
    }

    // 2.5 Draw lane corridor guidelines
    if (frame.road_context && (frame.road_context.lane_state === 'LANE DETECTED' || frame.road_context.lane_state === 'LANE CHANGE OBSERVED')) {
      const { left_boundary_x, right_boundary_x, lane_center_px } = frame.road_context;
      const horizonY = height * 0.45;
      ctx.save();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)'; // Subtle emerald
      ctx.lineWidth = 1.5;
      ctx.setLineDash([8, 6]);
      // Left lane boundary
      ctx.beginPath();
      ctx.moveTo(lane_center_px, horizonY);
      ctx.lineTo(left_boundary_x, height);
      ctx.stroke();
      // Right lane boundary
      ctx.beginPath();
      ctx.moveTo(lane_center_px, horizonY);
      ctx.lineTo(right_boundary_x, height);
      ctx.stroke();
      ctx.restore();
    }

    // Pedestrian state lookup
    const pedMap = new Map<number, any>();
    if (frame.pedestrian_awareness?.active_pedestrians) {
      frame.pedestrian_awareness.active_pedestrians.forEach((p) => {
        pedMap.set(p.track_id, p);
      });
    }

    // 3. Draw tracked bounding boxes
    const tracksToDraw = frame.tracks && frame.tracks.length > 0 ? frame.tracks : [];
    tracksToDraw.forEach((trk) => {
      const [x1, y1, x2, y2] = trk.bbox_xyxy;
      const boxW = x2 - x1;
      const boxH = y2 - y1;
      const isFocused = highlightTrackId !== null && highlightTrackId === trk.track_id;
      const pedInfo = pedMap.get(trk.track_id);

      let color = getTrackColor(trk.track_id);
      let label = `ID:${trk.track_id} ${trk.class_name} ${(trk.confidence * 100).toFixed(0)}%`;
      let badgeBg = 'rgba(15, 23, 42, 0.85)';
      let boxLineWidth = isFocused ? 4 : 2;

      if (pedInfo) {
        if (pedInfo.state === 'POTENTIAL PEDESTRIAN COLLISION RISK') {
          color = '#ef4444'; // Red
          badgeBg = 'rgba(185, 28, 28, 0.95)';
          boxLineWidth = 3.5;
          const ttcStr = pedInfo.ttc_seconds ? ` TTC:${pedInfo.ttc_seconds.toFixed(1)}s` : '';
          label = `[COLLISION RISK] PED #${trk.track_id}${ttcStr}`;
        } else if (pedInfo.state === 'PEDESTRIAN CROSSING DETECTED') {
          color = '#f97316'; // Orange
          badgeBg = 'rgba(194, 65, 12, 0.95)';
          boxLineWidth = 2.5;
          label = `[CROSSING] PED #${trk.track_id} (${pedInfo.road_position})`;
        } else if (pedInfo.state === 'PEDESTRIAN APPROACHING ROAD') {
          color = '#f59e0b'; // Amber
          badgeBg = 'rgba(180, 83, 9, 0.95)';
          label = `[APPROACHING] PED #${trk.track_id} (${pedInfo.side})`;
        } else {
          color = '#06b6d4'; // Cyan
          badgeBg = 'rgba(14, 116, 144, 0.90)';
          label = `PED #${trk.track_id} — SIDEWALK`;
        }
      }

      ctx.save();
      ctx.lineWidth = boxLineWidth;
      ctx.strokeStyle = isFocused ? '#00f0ff' : color;

      // Draw box
      ctx.strokeRect(x1, y1, boxW, boxH);

      // Corner accent brackets for focused track
      if (isFocused) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#ffffff';
        const bracketLen = Math.min(14, boxW / 3, boxH / 3);
        // Top-left
        ctx.beginPath();
        ctx.moveTo(x1, y1 + bracketLen);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + bracketLen, y1);
        ctx.stroke();
        // Top-right
        ctx.beginPath();
        ctx.moveTo(x2 - bracketLen, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + bracketLen);
        ctx.stroke();
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x1, y2 - bracketLen);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + bracketLen, y2);
        ctx.stroke();
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x2 - bracketLen, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - bracketLen);
        ctx.stroke();
      }

      // Draw label badge
      ctx.font = 'bold 12px monospace';
      const textMetrics = ctx.measureText(label);
      const badgeW = textMetrics.width + 10;
      const badgeH = 18;
      const badgeY = Math.max(0, y1 - badgeH - 2);

      ctx.fillStyle = badgeBg;
      ctx.fillRect(x1, badgeY, badgeW, badgeH);

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, badgeY, badgeW, badgeH);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + 5, badgeY + 13);
      ctx.restore();
    });
  }, [
    playbackMode,
    showOverlays,
    showTrails,
    showTrajectories,
    frame,
    highlightTrackId,
    videoRef,
  ]);

  // Re-render canvas overlays whenever frame or visibility options change
  useEffect(() => {
    renderCanvasOverlays();
  }, [renderCanvasOverlays, currentFrameIndex]);

  // Video event listeners
  const onTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    handleTimeUpdate(v.currentTime);
  };

  const onLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    handleDurationChange(v.duration);
    renderCanvasOverlays();
  };

  const onEnded = () => {
    setIsPlaying(false);
  };

  const onPlay = () => {
    setIsPlaying(true);
  };

  const onPause = () => {
    setIsPlaying(false);
  };

  const handleVideoError = () => {
    setVideoError(true);
    setUseFallbackImage(true);
  };

  // If no job or results loaded
  if (!activeJobId || !results || !results.frames || results.frames.length === 0) {
    return (
      <div className="w-full aspect-video bg-slate-950/80 rounded-2xl border border-slate-800 flex flex-col items-center justify-center p-8 text-center text-slate-500 shadow-inner">
        <Film className="w-12 h-12 mb-3 text-slate-600 stroke-[1.5]" />
        <h4 className="text-sm font-semibold text-slate-400 mb-1">No Active Continuous Stream</h4>
        <p className="text-xs max-w-sm">
          Select a completed video session from the stream selector above or upload a dashcam video to launch the continuous perception engine.
        </p>
      </div>
    );
  }

  // Fallback frame image url if video streaming fails
  const fallbackImagePath = frame?.annotated_frame_path || frame?.raw_frame_path;
  const fallbackImageUrl = fallbackImagePath ? getFrameImageUrl(activeJobId, fallbackImagePath) : '';

  return (
    <div className="relative w-full aspect-video bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center group select-none">
      {/* 1. Continuous Video Element */}
      {!useFallbackImage && activeVideoUrl ? (
        <video
          ref={videoRef}
          src={activeVideoUrl}
          playsInline
          preload="auto"
          onTimeUpdate={onTimeUpdate}
          onLoadedMetadata={onLoadedMetadata}
          onPlay={onPlay}
          onPause={onPause}
          onEnded={onEnded}
          onError={handleVideoError}
          className="w-full h-full object-contain bg-black"
        />
      ) : fallbackImageUrl ? (
        <img
          key={fallbackImageUrl}
          src={fallbackImageUrl}
          alt={`Frame #${frame?.frame_index}`}
          className="w-full h-full object-contain"
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center text-slate-500 bg-slate-950/90">
          <Film className="w-10 h-10 mb-2 text-slate-600 stroke-[1.5]" />
          <span className="text-xs font-semibold text-slate-400">
            {frame ? `Telemetry available for Frame #${frame.frame_index}` : 'No Video Loaded'}
          </span>
          <span className="text-[11px] text-slate-600 mt-1 max-w-xs">
            {videoError
              ? 'Video stream format is processing or requires container decode.'
              : 'Continuous video stream ready.'}
          </span>
        </div>
      )}

      {/* 2. Synchronized Live Overlay Canvas */}
      {!useFallbackImage && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      )}

      {/* 3. Top HUD Telemetry Ribbon */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none select-none z-10">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded bg-slate-950/85 backdrop-blur-md border border-slate-800 text-[11px] font-mono text-cyan-400 font-bold shadow-md">
            FRAME #{String(frame?.frame_index ?? 0).padStart(4, '0')}
          </span>
          <span className="px-2.5 py-1 rounded bg-slate-950/85 backdrop-blur-md border border-slate-800 text-[11px] font-mono text-slate-300 shadow-md">
            {currentTime != null ? `${currentTime.toFixed(2)}s` : `${(frame?.timestamp_seconds ?? 0).toFixed(2)}s`}
          </span>
          {frame?.weather_intelligence && (
            <span className="px-2.5 py-1 rounded bg-slate-950/85 backdrop-blur-md border border-slate-800 text-[11px] font-medium text-amber-300 shadow-md">
              {frame.weather_intelligence.primary_condition || frame.weather_intelligence.condition} ({frame.weather_intelligence.visibility_score?.toFixed(0)}%)
            </span>
          )}
        </div>

        {/* Dynamic Risk Indicator */}
        {frame?.collision_risk && (
          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded-full text-[11px] font-bold font-mono tracking-wider shadow-lg backdrop-blur-md border ${
                frame.collision_risk.risk_state === 'CRITICAL'
                  ? 'bg-red-500/80 border-red-400 text-white animate-pulse'
                  : frame.collision_risk.risk_state === 'HIGH'
                  ? 'bg-orange-500/80 border-orange-400 text-white'
                  : frame.collision_risk.risk_state === 'MODERATE'
                  ? 'bg-amber-500/80 border-amber-400 text-slate-950'
                  : 'bg-emerald-500/80 border-emerald-400 text-slate-950'
              }`}
            >
              RISK: {frame.collision_risk.risk_state}
            </span>
          </div>
        )}
      </div>

      {/* 4. Focused Target Reticle */}
      {highlightTrackId !== null && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 px-3 py-1 bg-cyan-950/90 border border-cyan-500/60 rounded-full text-cyan-300 text-xs font-mono font-bold flex items-center gap-1.5 shadow-lg z-10 pointer-events-none">
          <Compass className="w-3.5 h-3.5 animate-spin" />
          <span>FOCUSED TRACK #{highlightTrackId}</span>
        </div>
      )}

      {/* 5. Bottom Overlay Layer Controls & Playback Mode Bar */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-2 pointer-events-auto z-10">
        {/* Left: Stream Mode Toggle */}
        <div className="flex items-center gap-1 bg-slate-950/85 backdrop-blur-md p-1 rounded-lg border border-slate-800 shadow-lg">
          <button
            type="button"
            onClick={() => setPlaybackMode('ORIGINAL_LIVE')}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5 ${
              playbackMode === 'ORIGINAL_LIVE'
                ? 'bg-cyan-600 text-white shadow-sm font-semibold'
                : 'text-slate-400 hover:text-white'
            }`}
            title="Play original road video with synchronized live canvas overlays"
          >
            <Video className="w-3.5 h-3.5" />
            <span>Original + Overlays</span>
          </button>
          <button
            type="button"
            onClick={() => setPlaybackMode('ANNOTATED_PRERENDERED')}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5 ${
              playbackMode === 'ANNOTATED_PRERENDERED'
                ? 'bg-cyan-600 text-white shadow-sm font-semibold'
                : 'text-slate-400 hover:text-white'
            }`}
            title="Stream pre-rendered annotated continuous MP4 video"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Annotated Video</span>
          </button>
        </div>

        {/* Right: Layer Toggles & Voice Mute */}
        <div className="flex items-center gap-1 bg-slate-950/85 backdrop-blur-md p-1 rounded-lg border border-slate-800 shadow-lg">
          {playbackMode === 'ORIGINAL_LIVE' && (
            <>
              <button
                type="button"
                onClick={() => setShowOverlays((p) => !p)}
                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                  showOverlays ? 'bg-slate-800 text-cyan-300 font-semibold' : 'text-slate-500 hover:text-slate-300'
                }`}
                title="Toggle object bounding boxes"
              >
                Boxes
              </button>
              <button
                type="button"
                onClick={() => setShowTrails((p) => !p)}
                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                  showTrails ? 'bg-slate-800 text-emerald-300 font-semibold' : 'text-slate-500 hover:text-slate-300'
                }`}
                title="Toggle movement trails"
              >
                Trails
              </button>
              <button
                type="button"
                onClick={() => setShowTrajectories((p) => !p)}
                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                  showTrajectories ? 'bg-slate-800 text-amber-300 font-semibold' : 'text-slate-500 hover:text-slate-300'
                }`}
                title="Toggle predicted trajectories (H1-H5)"
              >
                <span className="flex items-center gap-1">
                  <Route className="w-3 h-3" />
                  Paths
                </span>
              </button>
              <div className="w-px h-4 bg-slate-800 mx-1" />
            </>
          )}

          {/* Voice Alert Mute / Unmute Button */}
          <button
            type="button"
            onClick={toggleVoiceMute}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5 ${
              isVoiceMuted
                ? 'bg-slate-800 text-slate-400 hover:text-slate-200'
                : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-semibold'
            }`}
            title={isVoiceMuted ? 'Voice alerts muted (click to unmute)' : 'Voice alerts active (click to mute)'}
          >
            {isVoiceMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
            <span>{isVoiceMuted ? 'Voice Muted' : 'Voice Active'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
