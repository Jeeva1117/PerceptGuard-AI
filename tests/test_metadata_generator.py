"""PerceptGuard-AI: Unit and Integration Test Suite for MetadataGenerator.

Covers:
1. Metadata record creation from valid data.
2. CSV generation and column integrity.
3. CSV read-back validation via Pandas.
4. JSON summary generation and schema conformance.
5. Missing image handling (failure recording).
6. Missing optional metrics (handling null/NaN gracefully).
7. Duplicate frame ID detection (SchemaValidationError).
8. Duplicate source frame index detection (SchemaValidationError).
9. Invalid timestamp detection (negative or non-monotonic timestamps).
10. Incorrect image dimensions validation (zero/negative dimensions).
11. Original to preprocessed frame mapping integrity.
12. Failed preprocessing record retention and status tracking.
13. Empty dataset handling (SchemaValidationError on empty table).
14. Numeric quality score bounds checking (brightness out of [0, 255]).
15. Strict RFC 8259 JSON serialization of missing/NaN values as null.
16. Real-dataset integration test (Tasks 1-4 driving dataset assets).
"""

import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_processing.metadata_generator import (
    DatasetSummary,
    FrameMetadataRecord,
    MetadataError,
    MetadataFileError,
    MetadataGenerator,
    SchemaValidationError,
)


class TestMetadataGenerator(unittest.TestCase):
    """Test suite for MetadataGenerator module."""

    def setUp(self) -> None:
        """Create a temporary test directory for outputs and fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="perceptguard_test_meta_"))

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_record(
        self,
        extracted_frame_id: int = 1,
        source_frame_index: int = 0,
        timestamp_seconds: float = 0.0,
        is_valid: bool = True,
        blur_score: Optional[float] = 150.0,
        brightness_score: Optional[float] = 110.0,
        contrast_score: Optional[float] = 55.0,
    ) -> FrameMetadataRecord:
        """Helper to create a deterministic FrameMetadataRecord."""
        status = "SUCCESS" if is_valid else "FAILED"
        val_status = "VALID" if is_valid else "INVALID"
        fail_msg = None if is_valid else "Simulated preprocessing failure"
        return FrameMetadataRecord(
            dataset_id="test_dataset",
            source_video="test_video.mp4",
            extracted_frame_id=extracted_frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=timestamp_seconds,
            original_image_filename=f"frame_{extracted_frame_id:06d}.jpg",
            original_image_path=f"frames/frame_{extracted_frame_id:06d}.jpg",
            preprocessed_image_filename=f"frame_{extracted_frame_id:06d}.jpg" if is_valid else None,
            preprocessed_image_path=f"preprocessed_frames/frame_{extracted_frame_id:06d}.jpg" if is_valid else None,
            file_size_bytes=35000 if is_valid else None,
            original_width=720 if is_valid else None,
            original_height=1280 if is_valid else None,
            processed_width=640 if is_valid else None,
            processed_height=360 if is_valid else None,
            channel_count=3 if is_valid else None,
            blur_score=blur_score if is_valid else None,
            brightness_score=brightness_score if is_valid else None,
            contrast_score=contrast_score if is_valid else None,
            preprocessing_status=status,
            validation_status=val_status,
            failure_reason=fail_msg,
        )

    def test_01_valid_record_creation(self) -> None:
        """Verify instantiation and dictionary serialization of a valid record."""
        rec = self._create_sample_record()
        generator = MetadataGenerator(dataset_id="test_dataset")
        generator.validate_metadata_record(rec)
        d = rec.to_dict()
        self.assertEqual(d["dataset_id"], "test_dataset")
        self.assertEqual(d["extracted_frame_id"], 1)
        self.assertEqual(d["source_frame_index"], 0)
        self.assertEqual(d["preprocessing_status"], "SUCCESS")

    def test_02_csv_generation(self) -> None:
        """Verify CSV persistence with correct schema columns and row count."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        records = [
            self._create_sample_record(1, 0, 0.0),
            self._create_sample_record(2, 6, 0.2),
            self._create_sample_record(3, 12, 0.4),
        ]
        generator._records = records
        csv_path = self.temp_dir / "test_metadata.csv"
        saved = generator.save_metadata_csv(csv_path)

        self.assertTrue(saved.is_file())
        self.assertGreater(saved.stat().st_size, 0)

        # Inspect headers
        with open(saved, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        headers = header_line.split(",")
        self.assertEqual(headers, generator.CSV_COLUMNS)

    def test_03_csv_read_back_validation(self) -> None:
        """Verify that written CSV can be read back via Pandas with matching data."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        records = [
            self._create_sample_record(1, 0, 0.0, blur_score=200.5),
            self._create_sample_record(2, 6, 0.2, blur_score=350.2),
        ]
        generator._records = records
        csv_path = self.temp_dir / "readback_test.csv"
        generator.save_metadata_csv(csv_path)

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["extracted_frame_id"]), [1, 2])
        self.assertEqual(list(df["source_frame_index"]), [0, 6])
        self.assertAlmostEqual(df.loc[0, "blur_score"], 200.5, places=1)

    def test_04_json_summary_generation(self) -> None:
        """Verify summary.json structure, calculations, and JSON parsing."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        records = [
            self._create_sample_record(1, 0, 0.0, blur_score=100.0, brightness_score=100.0, contrast_score=50.0),
            self._create_sample_record(2, 6, 0.2, blur_score=200.0, brightness_score=120.0, contrast_score=60.0),
        ]
        generator._records = records
        json_path = self.temp_dir / "summary.json"
        saved = generator.save_summary_json(json_path)

        self.assertTrue(saved.is_file())
        with open(saved, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["dataset_id"], "test_dataset")
        self.assertEqual(summary["total_extracted_frames"], 2)
        self.assertEqual(summary["total_preprocessed_frames"], 2)
        self.assertEqual(summary["total_failed_frames"], 0)
        self.assertAlmostEqual(summary["average_blur_score"], 150.0)
        self.assertAlmostEqual(summary["average_brightness_score"], 110.0)
        self.assertAlmostEqual(summary["average_contrast_score"], 55.0)

    def test_05_missing_image_handling(self) -> None:
        """Verify handling of frames that failed preprocessing or are missing."""
        rec_failed = self._create_sample_record(1, 0, 0.0, is_valid=False)
        generator = MetadataGenerator(dataset_id="test_dataset")
        generator.validate_metadata_record(rec_failed)
        self.assertEqual(rec_failed.preprocessing_status, "FAILED")
        self.assertEqual(rec_failed.validation_status, "INVALID")
        self.assertIsNotNone(rec_failed.failure_reason)

    def test_06_missing_optional_metrics(self) -> None:
        """Verify record with None/null metrics passes validation and serializes correctly."""
        rec = self._create_sample_record(1, 0, 0.0, blur_score=None, brightness_score=None, contrast_score=None)
        generator = MetadataGenerator(dataset_id="test_dataset")
        generator.validate_metadata_record(rec)
        d = rec.to_dict()
        self.assertIsNone(d["blur_score"])
        self.assertIsNone(d["brightness_score"])
        self.assertIsNone(d["contrast_score"])

    def test_07_duplicate_frame_id_detection(self) -> None:
        """Verify that duplicate extracted_frame_id values raise SchemaValidationError."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        duplicate_records = [
            self._create_sample_record(1, 0, 0.0),
            self._create_sample_record(1, 6, 0.2),  # Duplicate ID
        ]
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_table(duplicate_records)

    def test_08_duplicate_source_frame_index_detection(self) -> None:
        """Verify that duplicate source_frame_index values raise SchemaValidationError."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        duplicate_records = [
            self._create_sample_record(1, 6, 0.0),
            self._create_sample_record(2, 6, 0.2),  # Duplicate source index
        ]
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_table(duplicate_records)

    def test_09_invalid_timestamps_detection(self) -> None:
        """Verify that negative or non-monotonic timestamps raise SchemaValidationError."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        # Negative timestamp
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_record(self._create_sample_record(1, 0, -1.5))

        # Non-monotonic timestamp sequence
        non_monotonic = [
            self._create_sample_record(1, 0, 1.0),
            self._create_sample_record(2, 6, 0.5),  # Decreasing timestamp
        ]
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_table(non_monotonic)

    def test_10_incorrect_image_dimensions(self) -> None:
        """Verify that zero or negative spatial dimensions raise SchemaValidationError."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        bad_rec = FrameMetadataRecord(
            dataset_id="test",
            source_video="test.mp4",
            extracted_frame_id=1,
            source_frame_index=0,
            timestamp_seconds=0.0,
            original_image_filename="f.jpg",
            original_image_path="f.jpg",
            preprocessed_image_filename="f.jpg",
            preprocessed_image_path="f.jpg",
            file_size_bytes=100,
            original_width=0,  # Invalid zero width
            original_height=1280,
            processed_width=640,
            processed_height=360,
            channel_count=3,
            blur_score=100.0,
            brightness_score=100.0,
            contrast_score=50.0,
            preprocessing_status="SUCCESS",
            validation_status="VALID",
        )
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_record(bad_rec)

    def test_11_original_to_preprocessed_mapping(self) -> None:
        """Verify that records preserve matching original and preprocessed asset filenames."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        rec = self._create_sample_record(5, 24, 0.8)
        self.assertEqual(rec.original_image_filename, "frame_000005.jpg")
        self.assertEqual(rec.preprocessed_image_filename, "frame_000005.jpg")

    def test_12_failed_preprocessing_records_retention(self) -> None:
        """Verify that table validation retains failed records while enforcing schema integrity."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        mixed_records = [
            self._create_sample_record(1, 0, 0.0, is_valid=True),
            self._create_sample_record(2, 6, 0.2, is_valid=False),  # Failed record
            self._create_sample_record(3, 12, 0.4, is_valid=True),
        ]
        generator.validate_metadata_table(mixed_records)
        generator._records = mixed_records
        summary = generator.generate_summary()
        self.assertEqual(summary.total_extracted_frames, 3)
        self.assertEqual(summary.total_preprocessed_frames, 2)
        self.assertEqual(summary.total_failed_frames, 1)

    def test_13_empty_dataset_handling(self) -> None:
        """Verify that attempting to validate an empty table raises SchemaValidationError."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_table([])

    def test_14_numeric_quality_bounds(self) -> None:
        """Verify that out-of-range quality scores raise SchemaValidationError."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        # Brightness > 255.0
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_record(self._create_sample_record(1, 0, 0.0, brightness_score=300.0))

        # Negative blur score
        with self.assertRaises(SchemaValidationError):
            generator.validate_metadata_record(self._create_sample_record(1, 0, 0.0, blur_score=-5.0))

    def test_15_rfc8259_json_serialization_nulls(self) -> None:
        """Verify strict RFC 8259 JSON serialization: null values instead of NaN."""
        generator = MetadataGenerator(dataset_id="test_dataset")
        records = [
            self._create_sample_record(1, 0, 0.0, is_valid=False),  # No metrics
        ]
        generator._records = records
        json_path = self.temp_dir / "null_summary.json"
        generator.save_summary_json(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Confirm strict null tokens and absence of JavaScript NaN
        self.assertIn('"average_blur_score": null', raw_text)
        self.assertNotIn("NaN", raw_text)
        self.assertNotIn("Infinity", raw_text)

    def test_16_real_dataset_integration(self) -> None:
        """Integration test compiling and serializing metadata for actual Tasks 1-4 dataset."""
        raw_frames_dir = PROJECT_ROOT / "datasets" / "processed" / "driving_sample" / "frames"
        preproc_frames_dir = PROJECT_ROOT / "datasets" / "processed" / "driving_sample" / "preprocessed_frames"
        video_path = PROJECT_ROOT / "datasets" / "raw" / "converted_026c7465-309f6d33.mp4"

        if not raw_frames_dir.is_dir() or not preproc_frames_dir.is_dir():
            self.skipTest("Real dataset frames directories not found. Skipping live integration test.")

        generator = MetadataGenerator(
            dataset_id="driving_sample",
            source_video_path=video_path,
            raw_frames_dir=raw_frames_dir,
            preprocessed_frames_dir=preproc_frames_dir,
        )

        records = generator.generate_metadata()
        self.assertEqual(len(records), 200)

        # Check first and last frame mapping
        self.assertEqual(records[0].extracted_frame_id, 1)
        self.assertEqual(records[0].source_frame_index, 0)
        self.assertAlmostEqual(records[0].timestamp_seconds, 0.0, places=3)
        self.assertEqual(records[-1].extracted_frame_id, 200)
        self.assertEqual(records[-1].source_frame_index, 1198)
        self.assertAlmostEqual(records[-1].timestamp_seconds, 39.8139, places=3)

        # Check summary
        summary = generator.summary
        self.assertIsNotNone(summary)
        self.assertEqual(summary.total_extracted_frames, 200)
        self.assertEqual(summary.total_preprocessed_frames, 200)
        self.assertEqual(summary.total_failed_frames, 0)
        self.assertAlmostEqual(summary.average_blur_score, 436.75, places=0)
        self.assertAlmostEqual(summary.average_brightness_score, 102.05, places=0)

        # Write to temporary outputs
        temp_csv = self.temp_dir / "real_metadata.csv"
        temp_json = self.temp_dir / "real_summary.json"
        generator.save_metadata_csv(temp_csv)
        generator.save_summary_json(temp_json)

        self.assertTrue(temp_csv.is_file())
        self.assertTrue(temp_json.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
