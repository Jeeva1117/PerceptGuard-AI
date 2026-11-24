import React from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import { useJob } from '../context/JobContext';

function formatTime(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export const PlaybackControls: React.FC = () => {
  const {
    results,
    currentFrameIndex,
    currentTime,
    duration,
    isPlaying,
    playbackSpeed,
    isVoiceMuted,
    togglePlay,
    seekToTimestamp,
    seekToFrame,
    setPlaybackSpeed,
    toggleVoiceMute,
  } = useJob();

  const totalFrames = results?.frames?.length || 0;
  const maxTime = duration > 0
    ? duration
    : (results?.frames && results.frames.length > 0
        ? results.frames[results.frames.length - 1].timestamp_seconds || 0
        : 0);

  const handlePrevFrame = () => {
    seekToFrame(Math.max(0, currentFrameIndex - 1));
  };

  const handleNextFrame = () => {
    seekToFrame(Math.min(totalFrames - 1, currentFrameIndex + 1));
  };

  const handleRestart = () => {
    seekToTimestamp(0);
  };

  const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = Number(e.target.value);
    seekToTimestamp(newTime);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col md:flex-row items-center gap-4 select-none shadow-lg">
      {/* Playback Action Buttons */}
      <div className="flex items-center gap-1.5 shrink-0">
        <button
          type="button"
          onClick={handleRestart}
          disabled={totalFrames === 0}
          title="Restart video from 00:00"
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        <button
          type="button"
          onClick={handlePrevFrame}
          disabled={totalFrames === 0 || currentFrameIndex === 0}
          title="Previous frame step"
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition-colors"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={togglePlay}
          disabled={totalFrames === 0}
          title={isPlaying ? 'Pause' : 'Play continuous video'}
          className="p-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white disabled:bg-slate-800 disabled:text-slate-600 transition-colors shadow-sm shadow-cyan-950"
        >
          {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
        </button>

        <button
          type="button"
          onClick={handleNextFrame}
          disabled={totalFrames === 0 || currentFrameIndex >= totalFrames - 1}
          title="Next frame step"
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition-colors"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {/* Continuous Time & Frame Scrubber */}
      <div className="flex-1 w-full flex items-center gap-3">
        <span className="text-xs font-mono text-cyan-400 font-semibold w-12 text-right">
          {formatTime(currentTime)}
        </span>

        <input
          type="range"
          min="0"
          max={maxTime > 0 ? maxTime : 1}
          step="0.02"
          value={currentTime}
          onChange={handleScrubberChange}
          disabled={totalFrames === 0}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-40"
        />

        <span className="text-xs font-mono text-slate-400 w-12">
          {formatTime(maxTime)}
        </span>
      </div>

      {/* Frame Counter & Playback Speed Selector */}
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="text-[11px] font-mono text-slate-400 px-2.5 py-1 rounded bg-slate-950 border border-slate-800 flex items-center gap-1.5">
          <span className="text-slate-500">FRAME</span>
          <strong className="text-slate-200">
            #{String(currentFrameIndex).padStart(3, '0')} / {String(Math.max(0, totalFrames - 1)).padStart(3, '0')}
          </strong>
        </div>

        {/* Speed Chips */}
        <div className="flex items-center gap-0.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
          {[0.5, 1, 1.5, 2].map((spd) => (
            <button
              type="button"
              key={spd}
              onClick={() => setPlaybackSpeed(spd)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                playbackSpeed === spd
                  ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {spd}x
            </button>
          ))}
        </div>

        {/* Voice Alert Quick Toggle */}
        <button
          type="button"
          onClick={toggleVoiceMute}
          className={`p-1.5 rounded-lg border transition-colors ${
            isVoiceMuted
              ? 'bg-slate-950 border-slate-800 text-slate-500 hover:text-slate-300'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
          }`}
          title={isVoiceMuted ? 'Voice alerts muted (click to unmute)' : 'Voice alerts active (click to mute)'}
        >
          {isVoiceMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
};
