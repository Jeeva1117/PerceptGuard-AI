import React, { useState, useRef } from 'react';
import { UploadCloud, X, AlertCircle, CheckCircle, Loader2, Film } from 'lucide-react';
import { submitVideoJob } from '../services/api';
import { useJob } from '../context/JobContext';

interface VideoUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const VideoUploadModal: React.FC<VideoUploadModalProps> = ({ isOpen, onClose }) => {
  const { refreshJobs, selectJob } = useJob();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [maxFrames, setMaxFrames] = useState<number>(30);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successJobId, setSuccessJobId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const validExts = ['.mp4', '.avi', '.mov', '.mkv'];
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!validExts.includes(ext)) {
        setErrorMessage(`Invalid video format '${ext}'. Supported: ${validExts.join(', ')}`);
        return;
      }
      setSelectedFile(file);
      setErrorMessage(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMessage(null);

    try {
      const resp = await submitVideoJob(selectedFile, maxFrames);
      setSuccessJobId(resp.job_id);
      await refreshJobs();
      await selectJob(resp.job_id);
      setTimeout(() => {
        setIsUploading(false);
        onClose();
      }, 1200);
    } catch (err: any) {
      setIsUploading(false);
      setErrorMessage(err.message || 'Failed to submit video processing job.');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-cyan-500/10 rounded-lg text-cyan-400">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-white text-base">Upload Video for Analysis</h3>
              <p className="text-xs text-slate-400">YOLO + ByteTrack + LSTM Reliability Pipeline</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isUploading}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          {/* File Selection Box */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 rounded-xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40 hover:bg-slate-950/70"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv"
              onChange={handleFileChange}
              className="hidden"
            />
            {selectedFile ? (
              <div className="flex items-center justify-center gap-3 text-cyan-400">
                <Film className="w-6 h-6" />
                <div className="text-left">
                  <p className="text-sm font-semibold text-white truncate max-w-xs">{selectedFile.name}</p>
                  <p className="text-xs text-slate-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <UploadCloud className="w-10 h-10 text-slate-500 mx-auto" />
                <p className="text-sm font-medium text-slate-300">
                  Click or drag video file here
                </p>
                <p className="text-xs text-slate-500">
                  MP4, AVI, MOV, MKV (Max 100 MB, CPU-safe)
                </p>
              </div>
            )}
          </div>

          {/* Max Frames Constraint Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300">Max Frames to Process</span>
              <span className="font-mono text-cyan-400 font-bold">{maxFrames} frames</span>
            </div>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={maxFrames}
              onChange={(e) => setMaxFrames(Number(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>10 frames (Fastest)</span>
              <span>30 frames (Balanced)</span>
              <span>60 frames (Deep)</span>
            </div>
          </div>

          {/* Error Message */}
          {errorMessage && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Success Notification */}
          {successJobId && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs">
              <CheckCircle className="w-4 h-4 shrink-0" />
              <span>Job submitted! Processing initialized: {successJobId}</span>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isUploading}
              className="px-4 py-2 text-xs font-medium rounded-lg text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!selectedFile || isUploading}
              className="px-5 py-2 text-xs font-semibold rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-600 text-white transition-all shadow-md shadow-cyan-950 flex items-center gap-2"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Submitting Pipeline...</span>
                </>
              ) : (
                <span>Launch Analysis</span>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
