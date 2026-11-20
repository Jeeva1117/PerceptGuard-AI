"""Comprehensive Real-World Video Validation Script for PerceptGuard AI (Day 14).

Executes the complete 12-stage pipeline across:
Scenario A: Authentic BDD100K Daytime Driving Sequence (Normal Lighting)
Scenario B: Low-Lighting / Twilight Driving Sequence (Real BDD100K frames darkened)
Scenario C: Motion-Heavy / High-Speed Driving Sequence (Real BDD100K high-displacement frames)
Scenario D: Degraded Visual Quality (Gaussian Blur & Contrast Reduction)
Scenario E: Edge Case - Minimalist 1-Frame Driving Sequence
Scenario F: Edge Case - Corrupted Video Payload (Graceful Rejection Verification)

Records genuine metrics, latency, CPU utilization, and telemetry outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd().resolve()))

import cv2
import numpy as np
import psutil

from src.video_processing.video_pipeline_runner import VideoPipelineRunner, VideoPipelineRunResult


def extract_clip(src_video: str, dst_video: str, start_frame: int = 0, num_frames: int = 15, transform=None) -> dict:
    """Extract and optionally transform a slice of a real video."""
    cap = cv2.VideoCapture(src_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if transform:
            frame = transform(frame)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"Could not extract frames from {src_video}")

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_video, fourcc, 10.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    size_bytes = os.path.getsize(dst_video)
    return {
        "path": dst_video,
        "width": w,
        "height": h,
        "fps": 10.0,
        "frame_count": len(frames),
        "duration_seconds": round(len(frames) / 10.0, 2),
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2),
    }


def main():
    root = Path.cwd()
    raw_video = str(root / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4")
    if not os.path.exists(raw_video):
        print(f"ERROR: Raw video {raw_video} not found!")
        return

    work_dir = root / "outputs" / "day14_validation"
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("PERCEPTGUARD AI — DAY 14 REAL-WORLD VALIDATION SUITE")
    print("================================================================================")

    # 1. Prepare Scenarios
    scenarios = {}

    # Scenario A: Normal Daytime Driving
    print("\n[1/6] Preparing Scenario A: Normal Daytime Driving...")
    clip_a = str(clips_dir / "scenario_a_normal_daytime.mp4")
    scenarios["Scenario A: Normal Daytime Driving"] = extract_clip(
        raw_video, clip_a, start_frame=100, num_frames=15
    )

    # Scenario B: Low Lighting
    print("[2/6] Preparing Scenario B: Low Lighting (Twilight / Night)...")
    clip_b = str(clips_dir / "scenario_b_low_lighting.mp4")
    def _darken(f):
        return cv2.convertScaleAbs(f, alpha=0.35, beta=-20)
    scenarios["Scenario B: Low Lighting"] = extract_clip(
        raw_video, clip_b, start_frame=100, num_frames=15, transform=_darken
    )

    # Scenario C: Motion-Heavy
    print("[3/6] Preparing Scenario C: Motion-Heavy / High-Displacement...")
    clip_c = str(clips_dir / "scenario_c_motion_heavy.mp4")
    cap = cv2.VideoCapture(raw_video)
    frames_c = []
    # Sample every 3rd frame to simulate rapid motion
    for i in range(15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100 + i * 4)
        ret, f = cap.read()
        if ret and f is not None:
            frames_c.append(f)
    cap.release()
    h, w, _ = frames_c[0].shape
    writer = cv2.VideoWriter(clip_c, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames_c:
        writer.write(f)
    writer.release()
    scenarios["Scenario C: Motion-Heavy Driving"] = {
        "path": clip_c,
        "width": w,
        "height": h,
        "fps": 10.0,
        "frame_count": len(frames_c),
        "duration_seconds": round(len(frames_c) / 10.0, 2),
        "size_bytes": os.path.getsize(clip_c),
        "size_kb": round(os.path.getsize(clip_c) / 1024, 2),
    }

    # Scenario D: Visual Quality Degradation (Blur & Low Contrast)
    print("[4/6] Preparing Scenario D: Degraded Visual Quality (Blur & Low Contrast)...")
    clip_d = str(clips_dir / "scenario_d_blur_and_contrast.mp4")
    def _degrade(f):
        blurred = cv2.GaussianBlur(f, (11, 11), 0)
        return cv2.convertScaleAbs(blurred, alpha=0.6, beta=10)
    scenarios["Scenario D: Visual Quality Degradation"] = extract_clip(
        raw_video, clip_d, start_frame=100, num_frames=15, transform=_degrade
    )

    # Scenario E: Edge Case - 1 Frame Minimal Sequence
    print("[5/6] Preparing Scenario E: Edge Case - 1 Frame Minimal Sequence...")
    clip_e = str(clips_dir / "scenario_e_single_frame.mp4")
    scenarios["Scenario E: Minimalist 1-Frame Sequence"] = extract_clip(
        raw_video, clip_e, start_frame=100, num_frames=1
    )

    # Scenario F: Edge Case - Corrupted Video
    print("[6/6] Preparing Scenario F: Edge Case - Corrupted Video Stream...")
    clip_f = str(clips_dir / "scenario_f_corrupted.mp4")
    with open(clip_f, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE_DATA")
    scenarios["Scenario F: Corrupted Stream Rejection"] = {
        "path": clip_f,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "size_bytes": len(b"NOT_A_VALID_MP4_HEADER_GARBAGE_DATA"),
        "size_kb": 0.03,
    }

    # 2. Run Pipeline Runner across all scenarios
    runner = VideoPipelineRunner(project_root=root)
    results = {}

    process = psutil.Process(os.getpid())
    base_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\nInitial Host Process RAM: {base_ram_mb:.1f} MB")

    for scenario_name, meta in scenarios.items():
        print(f"\n--------------------------------------------------------------------------------")
        print(f"RUNNING: {scenario_name}")
        print(f"Video File: {Path(meta['path']).name} ({meta['size_kb']} KB, {meta['width']}x{meta['height']}, {meta['frame_count']} frames)")
        print(f"--------------------------------------------------------------------------------")

        job_id = f"job_day14_{Path(meta['path']).stem}"
        job_out = work_dir / job_id

        t0 = time.perf_counter()
        ram_before = process.memory_info().rss / (1024 * 1024)

        try:
            res: VideoPipelineRunResult = runner.run(
                video_path=meta["path"],
                job_id=job_id,
                output_dir=job_out,
                max_frames=meta["frame_count"] or 5,
            )
            t_total = time.perf_counter() - t0
            ram_after = process.memory_info().rss / (1024 * 1024)

            ms_per_frame = (t_total * 1000.0) / max(1, res.frames_processed)

            print(f"Status: COMPLETED")
            print(f"Frames Processed: {res.frames_processed}")
            print(f"Detections Count: {res.total_detections}")
            print(f"Unique Tracks: {res.unique_tracks}")
            print(f"Total Alerts: {res.total_alerts}")
            print(f"Active Alerts: {res.active_alerts}")
            print(f"Execution Latency: {t_total:.2f}s ({ms_per_frame:.1f} ms/frame)")
            print(f"RAM Delta: +{max(0.0, ram_after - ram_before):.1f} MB (Current: {ram_after:.1f} MB)")
            print(f"Per-Horizon Risk Scores:")
            for h_key in ("h1", "h3", "h5"):
                h_data = res.per_horizon_risk[h_key]
                print(f"  {h_key.upper()}: latest={h_data['latest_risk']:.4f}, mean={h_data['mean_risk']:.4f}, max={h_data['max_risk']:.4f}, category={h_data['category']}")

            results[scenario_name] = {
                "status": "COMPLETED",
                "video_metadata": meta,
                "frames_processed": res.frames_processed,
                "total_detections": res.total_detections,
                "unique_tracks": res.unique_tracks,
                "total_alerts": res.total_alerts,
                "active_alerts": res.active_alerts,
                "duration_seconds": round(t_total, 3),
                "ms_per_frame": round(ms_per_frame, 1),
                "ram_rss_mb": round(ram_after, 1),
                "per_horizon_risk": res.per_horizon_risk,
                "stage_durations_ms": res.stage_durations_ms,
            }

        except Exception as exc:
            t_total = time.perf_counter() - t0
            print(f"Status: EXPECTED REJECTION / FAILURE: {exc}")
            results[scenario_name] = {
                "status": "FAILED",
                "video_metadata": meta,
                "error_message": str(exc),
                "duration_seconds": round(t_total, 3),
            }

    # Save validation summary
    summary_file = work_dir / "real_world_validation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================================================================================")
    print(f"VALIDATION SUMMARY WRITTEN TO: {summary_file}")
    print("================================================================================")


if __name__ == "__main__":
    main()
