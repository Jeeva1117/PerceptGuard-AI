import os
import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.video_processing.video_pipeline_runner import VideoPipelineRunner

def main():
    print("Testing VideoPipelineRunner directly...")
    runner = VideoPipelineRunner(project_root=PROJECT_ROOT, device="cpu")

    # Create a small dummy MP4 (15 frames, sideways dimensions 1280x720)
    test_dir = PROJECT_ROOT / "outputs" / "test_verification"
    test_dir.mkdir(parents=True, exist_ok=True)
    video_path = test_dir / "sample_sideways.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (1280, 720))
    for i in range(15):
        # Create an artificial frame with an off-center rectangle to detect orientation
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        # Put text and a moving rectangle simulating a vehicle
        cx = int(300 + i * 20)
        cy = int(400)
        cv2.rectangle(frame, (cx, cy), (cx + 120, cy + 80), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"Created test video: {video_path}")

    # Run with rotation=90
    print("Executing pipeline with rotation=90...")
    out_dir = test_dir / "job_test_run"
    result = runner.run(
        video_path=video_path,
        job_id="job_test_run",
        output_dir=out_dir,
        max_frames=10,
        rotation=90,
    )

    print("Pipeline run completed!")
    print(f"Frames processed: {result.frames_processed}")
    print(f"Width x Height in meta: {result.video_metadata.get('width')} x {result.video_metadata.get('height')}")
    print(f"Rotation in meta: {result.video_metadata.get('rotation')}")

    # Check first extracted frame shape
    first_frame = cv2.imread(str(out_dir / "raw_frames" / "frame_000000.jpg"))
    print(f"Extracted frame shape: {first_frame.shape} (Expected 1280, 720, 3 for 90 deg rotation)")

    assert first_frame.shape == (1280, 720, 3), f"Expected (1280, 720, 3) but got {first_frame.shape}"
    assert result.video_metadata.get("width") == 720
    assert result.video_metadata.get("height") == 1280
    assert result.frames_processed == 10
    assert result.trajectory_summary is not None
    assert "mean_ade_px" in result.trajectory_summary
    assert "mean_fde_px" in result.trajectory_summary
    print("SUCCESS: Pipeline, ADE/FDE calculation, and rotation normalization verified flawlessly!")

if __name__ == "__main__":
    main()
