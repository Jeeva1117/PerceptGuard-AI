"""PerceptGuard-AI: Video Processing Package.

Provides production modules for video ingestion, validation, frame decoding,
and metadata extraction for autonomous driving computer vision pipelines.
"""

from src.video_processing.video_loader import (
    VideoLoader,
    VideoMetadata,
    VideoProcessingError,
    VideoNotFoundError,
    InvalidVideoPathError,
    VideoOpenError,
    CorruptedVideoError,
    InvalidVideoMetadataError,
)

__all__ = [
    "VideoLoader",
    "VideoMetadata",
    "VideoProcessingError",
    "VideoNotFoundError",
    "InvalidVideoPathError",
    "VideoOpenError",
    "CorruptedVideoError",
    "InvalidVideoMetadataError",
]

