"""PerceptGuard-AI: Predictive Object Detection Reliability System.

Staff Machine Learning Architecture for Forecasting Future Detection Reliability
in Autonomous Driving Computer Vision Pipelines.

Usage:
    python main.py --help
    python main.py --info
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for modular execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BASE_DIR,
    DEVICE,
    DEFAULT_YOLO_MODEL,
    SEQUENCE_LENGTH,
    FORECAST_HORIZONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PerceptGuard")


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="PerceptGuard-AI",
        description="Predictive Object Detection Reliability for Autonomous Driving Perception",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print system configuration, device info, and Day 1 foundation status",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Target compute device ('cpu' or 'cuda') [default: {DEVICE}]",
    )
    return parser


def display_system_info(target_device: str) -> None:
    """Display runtime system architecture parameters."""
    logger.info("=" * 65)
    logger.info("PerceptGuard-AI: Predictive Object Detection Reliability")
    logger.info("Autonomous Driving Computer Vision Engineering Foundation")
    logger.info("=" * 65)
    logger.info("Base Directory       : %s", BASE_DIR)
    logger.info("Selected Device      : %s", target_device)
    logger.info("Base Detection Model : %s", DEFAULT_YOLO_MODEL)
    logger.info("Temporal History Len : %d frames", SEQUENCE_LENGTH)
    logger.info("Forecasting Horizons : %s frames ahead", FORECAST_HORIZONS)
    logger.info("-" * 65)
    logger.info("Status: Day 1 Foundation Initialized & Configured.")
    logger.info("Next Milestone: Video Frame Ingestion & ByteTrack Integration.")
    logger.info("=" * 65)


def main() -> int:
    """Main execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.info or len(sys.argv) == 1:
        display_system_info(args.device)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

