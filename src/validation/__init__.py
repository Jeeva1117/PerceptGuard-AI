"""PerceptGuard-AI: Dataset Validation & Quality Assurance Package.

Provides production modules for non-destructive dataset auditing, multi-tier
integrity verification, and Day 3 perception readiness certification.
"""

from src.validation.dataset_validator import (
    CategoryValidationResult,
    DatasetValidationError,
    DatasetValidationReport,
    DatasetValidator,
    ValidationConfigError,
)

__all__ = [
    "DatasetValidator",
    "DatasetValidationReport",
    "CategoryValidationResult",
    "DatasetValidationError",
    "ValidationConfigError",
]
