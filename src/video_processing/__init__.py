"""PerceptGuard-AI: Video Processing Package.

Provides production modules for video ingestion, validation, frame decoding,
temporal sampling, frame extraction, image preprocessing, and metadata generation
for autonomous driving computer vision pipelines.
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

from src.video_processing.frame_preprocessor import (
    CorruptedImageError,
    FramePreprocessor,
    ImageSaveError,
    InvalidColorChannelError,
    InvalidImageDimensionsError,
    InvalidPreprocessingConfigError,
    PreprocessedFrameRecord,
    PreprocessingError,
    PreprocessingSummary,
)

from src.video_processing.metadata_generator import (
    DatasetSummary,
    FrameMappingError,
    FrameMetadataRecord,
    MetadataError,
    MetadataFileError,
    MetadataGenerator,
    SchemaValidationError,
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
    # Frame Preprocessing & Quality Diagnostics
    "FramePreprocessor",
    "PreprocessedFrameRecord",
    "PreprocessingSummary",
    "PreprocessingError",
    "CorruptedImageError",
    "InvalidImageDimensionsError",
    "InvalidColorChannelError",
    "ImageSaveError",
    "InvalidPreprocessingConfigError",
    # Structured Metadata Generation
    "MetadataGenerator",
    "FrameMetadataRecord",
    "DatasetSummary",
    "MetadataError",
    "SchemaValidationError",
    "MetadataFileError",
    "FrameMappingError",
]
