import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.video_processing.video_pipeline_runner import VideoPipelineRunner

def main():
    video_path = PROJECT_ROOT / "outputs" / "jobs" / "uploads" / "up_0a1dacef_gemini_generated_video_4cd01709.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    out_dir = PROJECT_ROOT / "outputs" / "gemini_video_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = VideoPipelineRunner(project_root=PROJECT_ROOT, device="cpu")
    print(f"Running pipeline on {video_path.name} with rotation=90, max_frames=20...")
    result = runner.run(
        video_path=video_path,
        job_id="job_gemini_test_01",
        output_dir=out_dir,
        max_frames=20,
        rotation=90,
    )

    print("\n--- GEMINI VIDEO RUN SUCCESS ---")
    print(f"Frames processed: {result.frames_processed}")
    print(f"Total detections: {result.total_detections}")
    print(f"Unique tracks: {result.unique_tracks}")
    print(f"Total alerts: {result.total_alerts}")
    print(f"Weather summary: {result.weather_summary}")
    print(f"Trajectory summary: {result.trajectory_summary}")
    print(f"Collision summary: {result.collision_summary}")
    print(f"Safety summary: {result.safety_summary}")

    # Inspect first frame dimensions
    first_frame = cv2.imread(str(out_dir / "raw_frames" / "frame_000000.jpg"))
    print(f"Extracted frame shape: {first_frame.shape} (Expected (1280, 720, 3) upright)")

if __name__ == "__main__":
    main()
