"""PerceptGuard-AI: Unit and Integration Test Suite for DatasetValidator.

Covers:
1. Valid dataset validation (overall PASS).
2. Missing raw video detection (FAIL).
3. Missing frames directory detection (FAIL).
4. Missing preprocessed frames detection (FAIL).
5. Corrupted image detection (FAIL).
6. Missing metadata.csv detection (FAIL).
7. Missing metadata schema columns detection (FAIL).
8. Duplicate frame IDs in metadata (FAIL).
9. Invalid / non-monotonic timestamps (FAIL).
10. Broken physical frame file references in metadata (FAIL).
11. Invalid summary.json syntax/keys (FAIL).
12. Invalid dataset_manifest.json (FAIL).
13. Extracted vs. preprocessed frame count mismatch (FAIL).
14. Broken source-to-frame traceability chain (FAIL).
15. Empty dataset handling (FAIL).
16. Non-blocking warning status behavior (WARNING).
17. Critical integrity failure behavior (FAIL).
18. Overall status calculation hierarchy (FAIL > WARNING > PASS).
19. Validation report JSON schema and persistence.
20. Read-only non-destructive idempotency (source data unmutated).
21. Real-dataset integration test on actual Tasks 1-6 driving dataset.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.dataset_validator import (
    CategoryValidationResult,
    DatasetValidationError,
    DatasetValidationReport,
    DatasetValidator,
)


class TestDatasetValidator(unittest.TestCase):
    """Test suite covering categorical audits and overall status decisions."""

    def setUp(self) -> None:
        """Create a mock dataset workspace sandbox."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="perceptguard_test_val_"))
        self.dataset_id = "test_sample"

        # Standard layout
        self.datasets_dir = self.temp_dir / "datasets"
        self.raw_dir = self.datasets_dir / "raw"
        self.processed_dir = self.datasets_dir / "processed" / self.dataset_id
        self.frames_dir = self.processed_dir / "frames"
        self.preproc_dir = self.processed_dir / "preprocessed_frames"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.preproc_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create a minimal valid synthetic video in raw/
        self.video_path = self.raw_dir / "sample_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, 30.0, (64, 64))
        for _ in range(10):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()

        # 2. Create 3 valid frames
        self.num_frames = 3
        for i in range(1, self.num_frames + 1):
            fname = f"frame_{i:06d}.jpg"
            img_raw = np.full((100, 100, 3), 128, dtype=np.uint8)
            img_proc = np.full((360, 640, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(self.frames_dir / fname), img_raw)
            cv2.imwrite(str(self.preproc_dir / fname), img_proc)

        # 3. Create metadata.csv
        csv_rows = [
            "dataset_id,source_video,extracted_frame_id,source_frame_index,timestamp_seconds,"
            "original_image_filename,original_image_path,preprocessed_image_filename,preprocessed_image_path,"
            "file_size_bytes,original_width,original_height,processed_width,processed_height,channel_count,"
            "blur_score,brightness_score,contrast_score,preprocessing_status,validation_status,failure_reason"
        ]
        for i in range(1, self.num_frames + 1):
            fname = f"frame_{i:06d}.jpg"
            raw_p = str(self.frames_dir / fname)
            proc_p = str(self.preproc_dir / fname)
            row = (
                f"{self.dataset_id},sample_video.mp4,{i},{(i-1)*6},{(i-1)*0.2:.4f},"
                f"{fname},{raw_p},{fname},{proc_p},1000,100,100,640,360,3,150.0,128.0,50.0,SUCCESS,VALID,"
            )
            csv_rows.append(row)
        (self.processed_dir / "metadata.csv").write_text("\n".join(csv_rows), encoding="utf-8")

        # 4. Create summary.json
        summary_data = {
            "dataset_id": self.dataset_id,
            "source_video": "sample_video.mp4",
            "source_video_fps": 30.0,
            "source_video_resolution": [64, 64],
            "total_source_frames": 10,
            "total_extracted_frames": self.num_frames,
            "total_preprocessed_frames": self.num_frames,
            "total_failed_frames": 0,
            "average_blur_score": 150.0,
            "average_brightness_score": 128.0,
            "average_contrast_score": 50.0,
            "processed_image_dimensions": [640, 360],
            "metadata_generation_timestamp": "2026-11-17T09:00:00+00:00",
            "metadata_schema_version": "1.0.0",
        }
        with open(self.processed_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 5. Create dataset_manifest.json
        manifest_data = {
            "manifest_schema_version": "1.0.0",
            "dataset_id": self.dataset_id,
            "dataset_name": "Test Dataset",
            "source_video": "sample_video.mp4",
            "dataset_created_at": "2026-11-17T09:00:00+00:00",
            "dataset_root": "datasets",
            "raw_video_directory": "datasets/raw",
            "extracted_frames_directory": f"datasets/processed/{self.dataset_id}/frames",
            "preprocessed_frames_directory": f"datasets/processed/{self.dataset_id}/preprocessed_frames",
            "metadata_csv_path": f"datasets/processed/{self.dataset_id}/metadata.csv",
            "summary_json_path": f"datasets/processed/{self.dataset_id}/summary.json",
            "total_extracted_frames": self.num_frames,
            "total_preprocessed_frames": self.num_frames,
            "total_failed_frames": 0,
            "image_format": "jpg",
            "processed_image_width": 640,
            "processed_image_height": 360,
            "metadata_schema_version": "1.0.0",
        }
        with open(self.processed_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def tearDown(self) -> None:
        """Clean up test sandbox."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_valid_dataset(self) -> None:
        """Verify that a complete, valid dataset yields overall status PASS."""
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = validator.validate_dataset()

        self.assertEqual(report.overall_status, "PASS")
        self.assertTrue(report.is_ready_for_day_3)
        self.assertEqual(len(report.issues), 0)

    def test_02_missing_raw_video(self) -> None:
        """Verify that missing raw video fails source_video check."""
        self.video_path.unlink()
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_source_video()
        self.assertEqual(res.status, "FAIL")
        self.assertGreater(len(res.errors), 0)

    def test_03_missing_frames_directory(self) -> None:
        """Verify that missing extracted frames directory causes FAIL."""
        shutil.rmtree(self.frames_dir)
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_directory_structure()
        self.assertEqual(res.status, "FAIL")

    def test_04_missing_preprocessed_frames(self) -> None:
        """Verify that missing preprocessed frames directory causes FAIL."""
        shutil.rmtree(self.preproc_dir)
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_preprocessed_frames()
        self.assertEqual(res.status, "FAIL")

    def test_05_corrupted_image(self) -> None:
        """Verify that corrupted image file causes preprocessed_frames check to FAIL."""
        corrupt_target = self.preproc_dir / "frame_000001.jpg"
        corrupt_target.write_bytes(b"CORRUPTED_NON_IMAGE_DATA_BYTES")
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_preprocessed_frames()
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("failed to decode" in err.lower() for err in res.errors))

    def test_06_missing_metadata_csv(self) -> None:
        """Verify that missing metadata.csv causes metadata check to FAIL."""
        (self.processed_dir / "metadata.csv").unlink()
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_metadata()
        self.assertEqual(res.status, "FAIL")

    def test_07_missing_metadata_columns(self) -> None:
        """Verify that metadata.csv missing required schema columns fails validation."""
        bad_csv = "dataset_id,source_video,extracted_frame_id\ntest,sample.mp4,1\n"
        (self.processed_dir / "metadata.csv").write_text(bad_csv, encoding="utf-8")
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_metadata()
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("missing required columns" in err.lower() for err in res.errors))

    def test_08_duplicate_frame_ids(self) -> None:
        """Verify that duplicate extracted_frame_id values fail validation."""
        bad_csv = (
            "dataset_id,source_video,extracted_frame_id,source_frame_index,timestamp_seconds,"
            "original_image_filename,original_image_path,preprocessed_image_filename,preprocessed_image_path,"
            "file_size_bytes,original_width,original_height,processed_width,processed_height,channel_count,"
            "blur_score,brightness_score,contrast_score,preprocessing_status,validation_status,failure_reason\n"
            "test,v.mp4,1,0,0.0,f1,p1,f1,p1,100,10,10,640,360,3,10,10,10,SUCCESS,VALID,\n"
            "test,v.mp4,1,6,0.2,f2,p2,f2,p2,100,10,10,640,360,3,10,10,10,SUCCESS,VALID,\n"
        )
        (self.processed_dir / "metadata.csv").write_text(bad_csv, encoding="utf-8")
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_metadata()
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("duplicate" in err.lower() for err in res.errors))

    def test_09_invalid_timestamps(self) -> None:
        """Verify that negative or non-monotonic timestamps fail metadata check."""
        bad_csv = (
            "dataset_id,source_video,extracted_frame_id,source_frame_index,timestamp_seconds,"
            "original_image_filename,original_image_path,preprocessed_image_filename,preprocessed_image_path,"
            "file_size_bytes,original_width,original_height,processed_width,processed_height,channel_count,"
            "blur_score,brightness_score,contrast_score,preprocessing_status,validation_status,failure_reason\n"
            "test,v.mp4,1,0,1.0,f1,p1,f1,p1,100,10,10,640,360,3,10,10,10,SUCCESS,VALID,\n"
            "test,v.mp4,2,6,0.5,f2,p2,f2,p2,100,10,10,640,360,3,10,10,10,SUCCESS,VALID,\n"
        )
        (self.processed_dir / "metadata.csv").write_text(bad_csv, encoding="utf-8")
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_metadata()
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("timestamp" in err.lower() for err in res.errors))

    def test_10_broken_frame_paths(self) -> None:
        """Verify that references to missing image files in metadata fail traceability."""
        bad_csv = (
            "dataset_id,source_video,extracted_frame_id,source_frame_index,timestamp_seconds,"
            "original_image_filename,original_image_path,preprocessed_image_filename,preprocessed_image_path,"
            "file_size_bytes,original_width,original_height,processed_width,processed_height,channel_count,"
            "blur_score,brightness_score,contrast_score,preprocessing_status,validation_status,failure_reason\n"
            "test,v.mp4,1,0,0.0,f1,non_existent_folder/f1.jpg,f1,non_existent_folder/proc1.jpg,100,10,10,640,360,3,10,10,10,SUCCESS,VALID,\n"
        )
        (self.processed_dir / "metadata.csv").write_text(bad_csv, encoding="utf-8")
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_frame_traceability()
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("missing" in err.lower() for err in res.errors))

    def test_11_invalid_summary_json(self) -> None:
        """Verify that broken JSON syntax in summary.json fails manifest check."""
        (self.processed_dir / "summary.json").write_text("{BROKEN_SYNTAX_JSON", encoding="utf-8")
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_manifest()
        self.assertEqual(res.status, "FAIL")

    def test_12_invalid_manifest(self) -> None:
        """Verify that dataset_manifest.json with missing keys or mismatch fails check."""
        bad_manifest = {"manifest_schema_version": "1.0.0", "dataset_id": "wrong_dataset_id"}
        with open(self.processed_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(bad_manifest, f)
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_manifest()
        self.assertEqual(res.status, "FAIL")

    def test_13_frame_counts_mismatch(self) -> None:
        """Verify that count mismatch between extracted and preprocessed frames fails check."""
        # Add an extra raw frame
        img_raw = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(self.frames_dir / "frame_000099.jpg"), img_raw)

        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = validator.validate_dataset()
        self.assertEqual(report.overall_status, "FAIL")

    def test_14_broken_source_to_frame_mapping(self) -> None:
        """Verify that missing preprocessed frame on SUCCESS record causes traceability FAIL."""
        (self.preproc_dir / "frame_000001.jpg").unlink()
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_frame_traceability()
        self.assertEqual(res.status, "FAIL")

    def test_15_empty_dataset(self) -> None:
        """Verify that empty frames directory causes FAIL."""
        for f in self.frames_dir.glob("*.jpg"):
            f.unlink()
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_extracted_frames()
        self.assertEqual(res.status, "FAIL")

    def test_16_warning_status_behavior(self) -> None:
        """Verify that non-critical issue (e.g. 1-channel image) produces WARNING, not FAIL."""
        # Replace 3-channel preprocessed image with 1-channel grayscale
        gray_img = np.full((360, 640), 128, dtype=np.uint8)
        cv2.imwrite(str(self.preproc_dir / "frame_000001.jpg"), gray_img)

        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        res = validator.validate_preprocessed_frames()
        self.assertEqual(res.status, "WARNING")
        self.assertEqual(len(res.errors), 0)
        self.assertGreater(len(res.warnings), 0)

    def test_17_critical_failure_behavior(self) -> None:
        """Verify that any critical check failure sets overall_status to FAIL and blocks Day 3."""
        # Unlink metadata.csv
        (self.processed_dir / "metadata.csv").unlink()
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        report = validator.validate_dataset()

        self.assertEqual(report.overall_status, "FAIL")
        self.assertFalse(report.is_ready_for_day_3)

    def test_18_overall_status_calculation(self) -> None:
        """Verify decision hierarchy: FAIL overrides WARNING, WARNING overrides PASS."""
        # Test 1: all pass -> PASS
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        rep1 = validator.validate_dataset()
        self.assertEqual(rep1.overall_status, "PASS")

        # Test 2: induce a warning -> WARNING
        gray_img = np.full((360, 640), 128, dtype=np.uint8)
        cv2.imwrite(str(self.preproc_dir / "frame_000001.jpg"), gray_img)
        rep2 = validator.validate_dataset()
        self.assertEqual(rep2.overall_status, "WARNING")

        # Test 3: induce a failure -> FAIL
        (self.processed_dir / "metadata.csv").unlink()
        rep3 = validator.validate_dataset()
        self.assertEqual(rep3.overall_status, "FAIL")

    def test_19_validation_report_generation(self) -> None:
        """Verify validation report generation, structure, and JSON persistence."""
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)
        out_report = self.temp_dir / "test_report.json"
        saved = validator.save_validation_report(out_report)

        self.assertTrue(saved.is_file())
        with open(saved, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["dataset_id"], self.dataset_id)
        self.assertEqual(data["overall_status"], "PASS")
        self.assertTrue(data["is_ready_for_day_3"])
        self.assertIn("checks", data)
        self.assertIn("statistics", data)
        self.assertIn("recommendations", data)

    def test_20_repeated_validation_non_destructive(self) -> None:
        """Verify that running validation repeatedly does not modify source dataset files."""
        validator = DatasetValidator(dataset_id=self.dataset_id, project_root=self.temp_dir)

        # Record file modification times
        mtimes_before = {
            f: f.stat().st_mtime
            for f in [self.video_path] + list(self.frames_dir.glob("*.jpg")) + list(self.preproc_dir.glob("*.jpg"))
        }

        # Run validation 3 times
        validator.validate_dataset()
        validator.validate_dataset()
        validator.validate_dataset()

        mtimes_after = {
            f: f.stat().st_mtime
            for f in [self.video_path] + list(self.frames_dir.glob("*.jpg")) + list(self.preproc_dir.glob("*.jpg"))
        }

        self.assertEqual(mtimes_before, mtimes_after)

    def test_21_real_dataset_integration(self) -> None:
        """Integration test against actual Tasks 1-6 driving dataset."""
        real_raw = PROJECT_ROOT / "datasets" / "raw"
        real_processed = PROJECT_ROOT / "datasets" / "processed" / "driving_sample"

        if not real_raw.is_dir() or not real_processed.is_dir():
            self.skipTest("Real dataset directories not found. Skipping live integration test.")

        validator = DatasetValidator(dataset_id="driving_sample", project_root=PROJECT_ROOT)
        report = validator.validate_dataset()

        self.assertEqual(report.overall_status, "PASS")
        self.assertTrue(report.is_ready_for_day_3)
        self.assertEqual(len(report.issues), 0)

        # Verify all category checks passed
        for cat, res in report.checks.items():
            self.assertEqual(res["status"], "PASS", f"Category '{cat}' failed: {res['errors']}")

        # Verify statistics
        self.assertEqual(report.statistics["total_extracted_frames"], 200)
        self.assertEqual(report.statistics["total_preprocessed_frames"], 200)
        self.assertEqual(report.statistics["total_failed_frames"], 0)
        self.assertEqual(report.statistics["completeness_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

