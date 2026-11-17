"""PerceptGuard-AI: Video Processing Package.

Provides production modules for video ingestion, validation, frame decoding,
temporal sampling, and frame extraction for autonomous driving computer vision pipelines.
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

from src.video_processing.frame_extractor import (
    ExtractedFrameInfo,
    ExtractionSummary,
    FrameExtractionError,
    FrameExtractor,
    FrameSaveError,
    InvalidExtractionConfigError,
    VideoReadError,
)

__all__ = [
    # Video Ingestion & Metadata
    "VideoLoader",
    "VideoMetadata",
    "VideoProcessingError",
    "VideoNotFoundError",
    "InvalidVideoPathError",
    "VideoOpenError",
    "CorruptedVideoError",
    "InvalidVideoMetadataError",
    # Frame Extraction & Sampling
    "FrameExtractor",
    "ExtractedFrameInfo",
    "ExtractionSummary",
    "FrameExtractionError",
    "InvalidExtractionConfigError",
    "FrameSaveError",
    "VideoReadError",
]
