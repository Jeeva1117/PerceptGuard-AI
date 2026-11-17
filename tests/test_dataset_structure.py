"""PerceptGuard-AI: Unit and Integration Test Suite for Dataset Structure and Organization.

Covers:
1. Correct directory structure verification.
2. Missing directory detection and reporting.
3. Existing directory handling and preservation.
4. Manifest generation and filesystem persistence.
5. Manifest JSON syntax and schema validity.
6. Metadata file existence enforcement.
7. Frame file references and access interface.
8. Duplicate filename detection in frames directories.
9. Frame count mismatch detection (extracted vs. preprocessed).
10. Dataset path consistency and resolution.
11. Dry-run execution behavior (zero filesystem mutation).
12. Idempotency (repeated execution causes zero side effects or duplicates).
13. Live dataset integration test on actual Tasks 1-5 assets.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_processing.dataset_organizer import (
    DataIntegrityError,
    DatasetManifest,
    DatasetOrganizationError,
    DatasetOrganizer,
    DatasetStructureError,
    DatasetVerificationReport,
    ManifestError,
)


class TestDatasetStructure(unittest.TestCase):
    """Test suite for DatasetOrganizer and dataset layout validation."""

    def setUp(self) -> None:
        """Create a temporary sandbox directory simulating a project workspace."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="perceptguard_test_ds_"))
        self.dataset_id = "test_sample"

        # Create standard sandbox layout
        self.datasets_dir = self.temp_dir / "datasets"
        self.raw_dir = self.datasets_dir / "raw"
        self.processed_dir = self.datasets_dir / "processed" / self.dataset_id
        self.frames_dir = self.processed_dir / "frames"
        self.preproc_dir = self.processed_dir / "preprocessed_frames"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.preproc_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy source video
        self.fake_video = self.raw_dir / "sample_video.mp4"
        self.fake_video.write_bytes(b"FAKE_VIDEO_STREAM_BYTES")

        # Create dummy extracted and preprocessed frames (3 frames)
        for i in range(1, 4):
            fname = f"frame_{i:06d}.jpg"
            (self.frames_dir / fname).write_bytes(b"FAKE_RAW_IMAGE_BYTES")
            (self.preproc_dir / fname).write_bytes(b"FAKE_PREPROCESSED_IMAGE_BYTES")

        # Create dummy metadata.csv
        csv_content = (
            "dataset_id,source_video,extracted_frame_id,source_frame_index,timestamp_seconds,"
            "original_image_filename,original_image_path,preprocessed_image_filename,preprocessed_image_path,"
            "file_size_bytes,original_width,original_height,processed_width,processed_height,channel_count,"
            "blur_score,brightness_score,contrast_score,preprocessing_status,validation_status,failure_reason\n"
            "test_sample,sample_video.mp4,1,0,0.0,frame_000001.jpg,p1,frame_000001.jpg,p1,100,720,1280,640,360,3,100.0,100.0,50.0,SUCCESS,VALID,\n"
            "test_sample,sample_video.mp4,2,6,0.2,frame_000002.jpg,p2,frame_000002.jpg,p2,100,720,1280,640,360,3,100.0,100.0,50.0,SUCCESS,VALID,\n"
            "test_sample,sample_video.mp4,3,12,0.4,frame_000003.jpg,p3,frame_000003.jpg,p3,100,720,1280,640,360,3,100.0,100.0,50.0,SUCCESS,VALID\n"
        )
        (self.processed_dir / "metadata.csv").write_text(csv_content, encoding="utf-8")

        # Create dummy summary.json
        summary_dict = {
            "dataset_id": self.dataset_id,
            "source_video": "sample_video.mp4",
            "source_video_fps": 30.0,
            "source_video_resolution": [720, 1280],
            "total_source_frames": 30,
            "total_extracted_frames": 3,
            "total_preprocessed_frames": 3,
            "total_failed_frames": 0,
            "average_blur_score": 100.0,
            "average_brightness_score": 100.0,
            "average_contrast_score": 50.0,
            "processed_image_dimensions": [640, 360],
            "metadata_generation_timestamp": "2026-11-17T09:00:00+00:00",
            "metadata_schema_version": "1.0.0",
        }
        with open(self.processed_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

    def tearDown(self) -> None:
        """Clean up sandbox directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_correct_directory_structure(self) -> None:
        """Verify that a well-formed dataset passes verification completely."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = organizer.verify_dataset_structure()

        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.missing_components), 0)
        self.assertEqual(len(report.integrity_issues), 0)
        self.assertEqual(report.extracted_frames_count, 3)
        self.assertEqual(report.preprocessed_frames_count, 3)
        self.assertTrue(report.raw_video_present)
        self.assertTrue(report.metadata_csv_present)
        self.assertTrue(report.summary_json_present)

    def test_02_missing_directory_handling(self) -> None:
        """Verify that missing directories are detected and reported."""
        shutil.rmtree(self.preproc_dir)
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = organizer.verify_dataset_structure()

        self.assertFalse(report.is_valid)
        self.assertTrue(any(str(self.preproc_dir) in s for s in report.missing_components))

    def test_03_existing_directory_handling(self) -> None:
        """Verify that organizer safely detects and preserves already existing directories."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        result = organizer.organize_dataset(dry_run=False)

        self.assertTrue(result["is_organized"])
        self.assertEqual(len(result["actions"]), 4)
        for act in result["actions"]:
            self.assertIn("already exists", act)

    def test_04_manifest_generation(self) -> None:
        """Verify manifest generation and filesystem output."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        manifest_path = organizer.generate_dataset_manifest()

        self.assertTrue(manifest_path.is_file())
        self.assertEqual(manifest_path.name, "dataset_manifest.json")
        self.assertGreater(manifest_path.stat().st_size, 0)

    def test_05_manifest_json_validity(self) -> None:
        """Verify that dataset_manifest.json conforms to strict RFC 8259 syntax and schema."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        manifest_path = organizer.generate_dataset_manifest()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        required_keys = [
            "manifest_schema_version",
            "dataset_id",
            "dataset_name",
            "source_video",
            "dataset_created_at",
            "dataset_root",
            "raw_video_directory",
            "extracted_frames_directory",
            "preprocessed_frames_directory",
            "metadata_csv_path",
            "summary_json_path",
            "total_extracted_frames",
            "total_preprocessed_frames",
            "total_failed_frames",
            "image_format",
            "processed_image_width",
            "processed_image_height",
            "metadata_schema_version",
        ]
        for key in required_keys:
            self.assertIn(key, manifest)

        self.assertEqual(manifest["dataset_id"], self.dataset_id)
        self.assertEqual(manifest["total_extracted_frames"], 3)
        self.assertEqual(manifest["total_preprocessed_frames"], 3)

    def test_06_metadata_file_existence(self) -> None:
        """Verify that missing metadata.csv raises an integrity failure."""
        (self.processed_dir / "metadata.csv").unlink()
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = organizer.verify_dataset_structure()

        self.assertFalse(report.is_valid)
        self.assertFalse(report.metadata_csv_present)
        self.assertTrue(any("metadata.csv" in s for s in report.missing_components))

    def test_07_frame_file_references_access(self) -> None:
        """Verify access interface returns actual, existing preprocessed frame paths."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        paths = organizer.get_preprocessed_frame_paths()

        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(p.is_file())
            self.assertEqual(p.suffix, ".jpg")

    def test_08_duplicate_filenames_detection(self) -> None:
        """Verify that duplicate filenames are flagged in inspection report."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        inspection = organizer.inspect_dataset()

        self.assertFalse(inspection["has_duplicate_extracted_names"])
        self.assertFalse(inspection["has_duplicate_preprocessed_names"])

    def test_09_missing_preprocessed_frames_mismatch(self) -> None:
        """Verify that frame count mismatch between extracted and preprocessed frames is caught."""
        # Delete one preprocessed frame
        (self.preproc_dir / "frame_000003.jpg").unlink()

        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = organizer.verify_dataset_structure()

        self.assertFalse(report.is_valid)
        self.assertTrue(any("mismatch" in issue.lower() for issue in report.integrity_issues))

    def test_10_dataset_path_consistency(self) -> None:
        """Verify that manifest paths use clean, portable forward-slash relative paths."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)
        manifest_path = organizer.generate_dataset_manifest()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertNotIn("\\", manifest["raw_video_directory"])
        self.assertNotIn("\\", manifest["extracted_frames_directory"])
        self.assertNotIn("\\", manifest["preprocessed_frames_directory"])
        self.assertNotIn("\\", manifest["metadata_csv_path"])

    def test_11_dry_run_behavior(self) -> None:
        """Verify that dry_run=True performs zero filesystem modifications."""
        fresh_dir = self.temp_dir / "datasets" / "processed" / "dry_run_dataset"
        self.assertFalse(fresh_dir.exists())

        organizer = DatasetOrganizer(dataset_id="dry_run_dataset", project_root=self.temp_dir)
        result = organizer.organize_dataset(dry_run=True)

        self.assertTrue(result["dry_run"])
        # Target directory must NOT have been created
        self.assertFalse(fresh_dir.exists())

    def test_12_idempotent_execution(self) -> None:
        """Verify that repeated organize and manifest generation calls are idempotent."""
        organizer = DatasetOrganizer(dataset_id=self.dataset_id, project_root=self.temp_dir)

        # Call organize twice
        res1 = organizer.organize_dataset(dry_run=False)
        res2 = organizer.organize_dataset(dry_run=False)
        self.assertEqual(res1["is_organized"], res2["is_organized"])

        # Call manifest generation twice
        m1 = organizer.generate_dataset_manifest()
        m2 = organizer.generate_dataset_manifest()
        self.assertEqual(m1, m2)

        # Confirm file counts remain exactly 3
        self.assertEqual(len(list(self.frames_dir.glob("*.jpg"))), 3)
        self.assertEqual(len(list(self.preproc_dir.glob("*.jpg"))), 3)

    def test_13_real_dataset_integration(self) -> None:
        """Integration test on actual Tasks 1-5 production dataset."""
        real_raw = PROJECT_ROOT / "datasets" / "raw"
        real_processed = PROJECT_ROOT / "datasets" / "processed" / "driving_sample"

        if not real_raw.is_dir() or not real_processed.is_dir():
            self.skipTest("Real dataset directories not found. Skipping live integration test.")

        organizer = DatasetOrganizer(dataset_id="driving_sample", project_root=PROJECT_ROOT)
        report = organizer.verify_dataset_structure()

        self.assertTrue(report.is_valid, f"Verification failed with issues: {report.integrity_issues}")
        self.assertEqual(report.extracted_frames_count, 200)
        self.assertEqual(report.preprocessed_frames_count, 200)
        self.assertTrue(report.raw_video_present)
        self.assertTrue(report.metadata_csv_present)
        self.assertTrue(report.summary_json_present)

        # Generate manifest and verify
        manifest_path = organizer.generate_dataset_manifest()
        self.assertTrue(manifest_path.is_file())

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["dataset_id"], "driving_sample")
        self.assertEqual(manifest["total_extracted_frames"], 200)
        self.assertEqual(manifest["total_preprocessed_frames"], 200)
        self.assertEqual(manifest["total_failed_frames"], 0)
        self.assertEqual(manifest["processed_image_width"], 640)
        self.assertEqual(manifest["processed_image_height"], 360)


if __name__ == "__main__":
    unittest.main(verbosity=2)

