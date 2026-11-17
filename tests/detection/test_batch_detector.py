"""PerceptGuard-AI: Automated Test Suite for Batch Object Detection Engine.

Covers:
1. Dataset discovery in standard processed layout.
2. Natural chronological sorting of frame paths.
3. Supported image extensions discovery (.jpg, .jpeg, .png).
4. Non-image files filtered out cleanly.
5. Missing dataset directory (DatasetNotFoundError).
6. Empty dataset directory (DatasetEmptyError).
7. Metadata association from metadata.csv (timestamps, frame IDs).
8. Missing metadata handled gracefully without inventing values.
9. Duplicate metadata entries handled safely.
10. Corrupted image handling: batch continues and logs to error record.
11. Model loaded only once across the entire batch run.
12. Sequential processing loop execution.
13. Multi-frame successful detections aggregation.
14. Frames with zero detections handled as successful.
15. Frames with multiple detections extracted accurately.
16. Bounding box coordinates validated and clamped.
17. Confidence score bounds verified.
18. Semantic class name catalog mapping.
19. Per-frame inference exception recovery (fault containment).
20. Model initialization failure halts batch execution.
21. Output multi-file JSON schemas (results, summary, errors, metadata).
22. Summary statistical aggregation (averages, medians, class distribution).
23. Diagnostic error catalog recording failure causes.
24. Unique run ID generation and directory naming.
25. Existing destination directory overwrite protection.
26. Partial smoke run status set to WARNING.
27. Full dataset completion status set to PASS.
28. Frame range slicing (start_index, end_index, max_frames).
29. Real-model integration test on actual Day 2 preprocessed frames.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.detection.batch_detector import (
    BatchDetectionError,
    BatchDetector,
    BatchErrorRecord,
    BatchExecutionError,
    BatchSummary,
    DatasetEmptyError,
    DatasetNotFoundError,
    FrameDetectionRecord,
)
from src.detection.single_image_detector import (
    DetectedObject,
    DetectionInferenceError,
    ImageDecodeError,
    SingleImageDetectionResult,
    SingleImageDetector,
)
from src.detection.yolo_config import YOLOConfig, load_yolo_config
from src.detection.yolo_model_loader import (
    YOLOModelLoadError,
    YOLOModelLoader,
)


class TestBatchDetector(unittest.TestCase):
    """Test suite covering unit and integration testing for BatchDetector."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create synthetic dataset sandbox and locate real assets."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Real assets
        cls.real_weights_path = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
        cls.real_dataset_dir = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / "driving_sample"
            / "preprocessed_frames"
        )

        # Synthetic mock dataset directory
        cls.mock_dataset_dir = cls.temp_path / "mock_dataset" / "preprocessed_frames"
        cls.mock_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 synthetic test images
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        for i in range(1, 6):
            p = cls.mock_dataset_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(p), img)

        # Create synthetic metadata.csv
        meta_csv = cls.mock_dataset_dir.parent / "metadata.csv"
        meta_csv.write_text(
            "dataset_id,source_video,extracted_frame_id,source_frame_index,timestamp_seconds,preprocessed_image_filename\n"
            "mock_dataset,test.mp4,1,0,0.0,frame_000001.jpg\n"
            "mock_dataset,test.mp4,2,6,0.2,frame_000002.jpg\n"
            "mock_dataset,test.mp4,3,12,0.4,frame_000003.jpg\n"
            "mock_dataset,test.mp4,4,18,0.6,frame_000004.jpg\n"
            "mock_dataset,test.mp4,5,24,0.8,frame_000005.jpg\n",
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary sandbox."""
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        """Configure mock detector and loader for isolated unit tests."""
        self.mock_detector = MagicMock(spec=SingleImageDetector)
        self.mock_loader = MagicMock(spec=YOLOModelLoader)
        self.mock_loader.model_name = "yolo11n"
        self.mock_loader.task = "detect"
        self.mock_loader.device = "cpu"
        self.mock_loader.confidence_threshold = 0.25
        self.mock_loader.iou_threshold = 0.45
        self.mock_loader.image_size = 640
        self.mock_loader.max_detections = 100
        self.mock_loader.weights_path = Path("models/yolo/yolo11n.pt")
        self.mock_loader.config = YOLOConfig()

        self.mock_detector.loader = self.mock_loader
        self.mock_detector.config = self.mock_loader.config

    # --------------------------------------------------------------------------
    # 1-6. Dataset Discovery & Directory Handling Tests
    # --------------------------------------------------------------------------

    def test_01_dataset_discovery(self) -> None:
        """Verify discovery of image frames within a valid dataset directory."""
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        ds_id, ds_dir, frames = detector.discover_frames(self.mock_dataset_dir)
        self.assertEqual(len(frames), 5)
        self.assertEqual(ds_dir, self.mock_dataset_dir.resolve())

    def test_02_deterministic_image_ordering(self) -> None:
        """Verify natural alphanumeric sorting of frame paths."""
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, _, frames = detector.discover_frames(self.mock_dataset_dir)
        filenames = [f.name for f in frames]
        self.assertEqual(filenames, [
            "frame_000001.jpg",
            "frame_000002.jpg",
            "frame_000003.jpg",
            "frame_000004.jpg",
            "frame_000005.jpg",
        ])

    def test_03_supported_extensions(self) -> None:
        """Verify that .jpg, .jpeg, and .png images are discovered."""
        ext_dir = self.temp_path / "mixed_ext" / "preprocessed_frames"
        ext_dir.mkdir(parents=True, exist_ok=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(ext_dir / "f1.jpg"), img)
        cv2.imwrite(str(ext_dir / "f2.jpeg"), img)
        cv2.imwrite(str(ext_dir / "f3.png"), img)

        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, _, frames = detector.discover_frames(ext_dir)
        self.assertEqual(len(frames), 3)

    def test_04_unsupported_extensions_ignored_or_handled(self) -> None:
        """Verify that non-image files (.txt, .json) are ignored during discovery."""
        mixed_dir = self.temp_path / "mixed_files" / "preprocessed_frames"
        mixed_dir.mkdir(parents=True, exist_ok=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(mixed_dir / "f1.jpg"), img)
        (mixed_dir / "notes.txt").write_text("info")
        (mixed_dir / "data.json").write_text("{}")

        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, _, frames = detector.discover_frames(mixed_dir)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, "f1.jpg")

    def test_05_missing_dataset_directory(self) -> None:
        """Verify that non-existent dataset directory raises DatasetNotFoundError."""
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        with self.assertRaises(DatasetNotFoundError):
            detector.discover_frames("non_existent_dataset_9999")

    def test_06_empty_dataset_directory(self) -> None:
        """Verify that an empty dataset directory raises DatasetEmptyError."""
        empty_dir = self.temp_path / "empty_dataset" / "preprocessed_frames"
        empty_dir.mkdir(parents=True, exist_ok=True)
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        with self.assertRaises(DatasetEmptyError):
            detector.discover_frames(empty_dir)

    # --------------------------------------------------------------------------
    # 7-10. Metadata Association & Error Containment Tests
    # --------------------------------------------------------------------------

    def test_07_metadata_association(self) -> None:
        """Verify metadata extraction and association from CSV."""
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        catalog = detector.load_metadata_catalog(self.mock_dataset_dir)
        self.assertIn("frame_000001.jpg", catalog)
        meta = catalog["frame_000001.jpg"]
        self.assertEqual(meta["extracted_frame_id"], 1)
        self.assertEqual(meta["source_frame_index"], 0)
        self.assertEqual(meta["timestamp_seconds"], 0.0)

    def test_08_missing_metadata(self) -> None:
        """Verify that missing metadata file returns empty catalog without crashing."""
        no_meta_dir = self.temp_path / "no_meta" / "preprocessed_frames"
        no_meta_dir.mkdir(parents=True, exist_ok=True)
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        catalog = detector.load_metadata_catalog(no_meta_dir)
        self.assertEqual(catalog, {})

    def test_09_duplicate_metadata_entries(self) -> None:
        """Verify that duplicate CSV rows are handled safely."""
        dup_dir = self.temp_path / "dup_meta" / "preprocessed_frames"
        dup_dir.mkdir(parents=True, exist_ok=True)
        csv_file = dup_dir.parent / "metadata.csv"
        csv_file.write_text(
            "dataset_id,extracted_frame_id,preprocessed_image_filename\n"
            "dup_ds,1,f1.jpg\n"
            "dup_ds,2,f1.jpg\n",
            encoding="utf-8",
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        catalog = detector.load_metadata_catalog(dup_dir)
        self.assertIn("f1.jpg", catalog)

    def test_10_corrupt_image_recovery(self) -> None:
        """Verify that corrupt image does not crash batch run and is logged in errors."""
        corrupt_ds = self.temp_path / "corrupt_ds" / "preprocessed_frames"
        corrupt_ds.mkdir(parents=True, exist_ok=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(corrupt_ds / "frame_000001.jpg"), img)
        (corrupt_ds / "frame_000002.jpg").write_bytes(b"NOT_A_VALID_IMAGE_BYTES")
        cv2.imwrite(str(corrupt_ds / "frame_000003.jpg"), img)

        def mock_detect(image_path, **kwargs):
            if "frame_000002.jpg" in str(image_path):
                raise ImageDecodeError("Corrupted image")
            return SingleImageDetectionResult(
                image_filename=Path(image_path).name,
                image_path=str(image_path),
                width=640,
                height=360,
                detection_count=0,
                detections=[],
            )

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, frames, errors, _ = detector.process_batch(
            dataset=corrupt_ds,
            output_dir=self.temp_path / "corrupt_runs",
            overwrite=True,
        )

        self.assertEqual(summary.total_frames_attempted, 3)
        self.assertEqual(summary.successful_frames, 2)
        self.assertEqual(summary.failed_frames, 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_category, "ImageDecodeError")

    # --------------------------------------------------------------------------
    # 11-15. Model Lifecycle & Detection Aggregation Tests
    # --------------------------------------------------------------------------

    def test_11_model_initialized_only_once(self) -> None:
        """Verify that YOLOModelLoader.load_model() is invoked exactly once per batch run."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_init",
            overwrite=True,
        )
        self.assertEqual(self.mock_loader.load_model.call_count, 1)

    def test_12_sequential_processing(self) -> None:
        """Verify that frames are processed sequentially in chronological order."""
        call_order = []
        def mock_detect(image_path, **kwargs):
            call_order.append(Path(image_path).name)
            return SingleImageDetectionResult(width=640, height=360, detection_count=0, detections=[])

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_seq",
            overwrite=True,
        )
        self.assertEqual(call_order, [
            "frame_000001.jpg",
            "frame_000002.jpg",
            "frame_000003.jpg",
            "frame_000004.jpg",
            "frame_000005.jpg",
        ])

    def test_13_successful_inference(self) -> None:
        """Verify multi-frame successful inference aggregation."""
        def mock_detect(image_path, **kwargs):
            return SingleImageDetectionResult(
                image_filename=Path(image_path).name,
                image_path=str(image_path),
                width=640,
                height=360,
                inference_time_ms=45.0,
                detection_count=1,
                detections=[DetectedObject(0, 2, "car", 0.85, [10.0, 10.0, 50.0, 50.0])],
            )

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, frames, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_succ",
            overwrite=True,
        )

        self.assertEqual(summary.successful_frames, 5)
        self.assertEqual(summary.total_detections, 5)
        self.assertEqual(summary.class_distribution.get("car"), 5)

    def test_14_zero_detection_frames(self) -> None:
        """Verify that frames with zero detections are counted accurately as successful."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_zero",
            overwrite=True,
        )
        self.assertEqual(summary.successful_frames, 5)
        self.assertEqual(summary.frames_without_detections, 5)
        self.assertEqual(summary.frames_with_detections, 0)
        self.assertEqual(summary.total_detections, 0)

    def test_15_multiple_detections(self) -> None:
        """Verify multi-class detection extraction across frames."""
        def mock_detect(image_path, **kwargs):
            return SingleImageDetectionResult(
                image_filename=Path(image_path).name,
                image_path=str(image_path),
                width=640,
                height=360,
                detection_count=2,
                detections=[
                    DetectedObject(0, 9, "traffic light", 0.90, [1.0, 1.0, 10.0, 20.0]),
                    DetectedObject(1, 7, "truck", 0.75, [100.0, 50.0, 300.0, 250.0]),
                ],
            )

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_mult",
            overwrite=True,
        )
        self.assertEqual(summary.total_detections, 10)
        self.assertEqual(summary.class_distribution["traffic light"], 5)
        self.assertEqual(summary.class_distribution["truck"], 5)

    # --------------------------------------------------------------------------
    # 16-20. Calibration, Exceptions & Lifecycle Halts
    # --------------------------------------------------------------------------

    def test_16_bounding_box_validation(self) -> None:
        """Verify that bounding boxes in frame records retain calibrated coordinates."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=1,
            detections=[DetectedObject(0, 2, "car", 0.8, [10.5, 20.2, 100.0, 150.0])]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, frames, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_bbox",
            overwrite=True,
        )
        self.assertEqual(frames[0].detections[0].bbox_xyxy, [10.5, 20.2, 100.0, 150.0])

    def test_17_confidence_validation(self) -> None:
        """Verify that confidence scores in frame records match detector outputs."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=1,
            detections=[DetectedObject(0, 2, "car", 0.4381, [10.0, 10.0, 50.0, 50.0])]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, frames, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_conf",
            overwrite=True,
        )
        self.assertAlmostEqual(frames[0].detections[0].confidence, 0.4381, places=4)

    def test_18_class_mapping(self) -> None:
        """Verify semantic class mapping preservation."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=1,
            detections=[DetectedObject(0, 0, "person", 0.95, [5.0, 5.0, 25.0, 80.0])]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_cls",
            overwrite=True,
        )
        self.assertIn("person", summary.class_distribution)

    def test_19_per_frame_inference_recovery(self) -> None:
        """Verify that single frame inference exception is recorded and batch finishes."""
        def mock_detect(image_path, **kwargs):
            if "frame_000003.jpg" in str(image_path):
                raise DetectionInferenceError("Transient math error")
            return SingleImageDetectionResult(width=640, height=360, detection_count=0, detections=[])

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, errors, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_rec",
            overwrite=True,
        )
        self.assertEqual(summary.successful_frames, 4)
        self.assertEqual(summary.failed_frames, 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_category, "DetectionInferenceError")

    def test_20_model_init_failure_halts_run(self) -> None:
        """Verify that failure during model load halts execution immediately."""
        self.mock_loader.load_model.side_effect = YOLOModelLoadError("Weights corrupt")
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        with self.assertRaises(BatchExecutionError):
            detector.process_batch(
                dataset=self.mock_dataset_dir,
                output_dir=self.temp_path / "runs_halt",
                overwrite=True,
            )

    # --------------------------------------------------------------------------
    # 21-25. Output Schemas, Statistics & Overwrite Protection
    # --------------------------------------------------------------------------

    def test_21_output_json_schemas(self) -> None:
        """Verify that all 4 generated JSON files exist and conform to schema."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, _, _, run_dir = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_schemas",
            overwrite=True,
        )

        # 1. detection_results.json
        res_file = run_dir / "detection_results.json"
        self.assertTrue(res_file.is_file())
        res_json = json.loads(res_file.read_text(encoding="utf-8"))
        self.assertEqual(res_json["schema_version"], "1.0")
        self.assertIn("frame_results", res_json)

        # 2. detection_summary.json
        sum_file = run_dir / "detection_summary.json"
        self.assertTrue(sum_file.is_file())
        sum_json = json.loads(sum_file.read_text(encoding="utf-8"))
        self.assertEqual(sum_json["schema_version"], "1.0")
        self.assertIn("total_frames_discovered", sum_json)

        # 3. detection_errors.json
        err_file = run_dir / "detection_errors.json"
        self.assertTrue(err_file.is_file())
        err_json = json.loads(err_file.read_text(encoding="utf-8"))
        self.assertEqual(err_json["schema_version"], "1.0")

        # 4. run_metadata.json
        meta_file = run_dir / "run_metadata.json"
        self.assertTrue(meta_file.is_file())
        meta_json = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertIn("software_versions", meta_json)

    def test_22_summary_statistics(self) -> None:
        """Verify summary metric computations (averages, medians)."""
        latencies = [40.0, 50.0, 60.0, 70.0, 80.0]
        idx = [0]
        def mock_detect(image_path, **kwargs):
            cur_lat = latencies[idx[0]]
            idx[0] += 1
            return SingleImageDetectionResult(
                width=640, height=360, inference_time_ms=cur_lat,
                detection_count=2, detections=[
                    DetectedObject(0, 2, "car", 0.8, [1.0, 1.0, 10.0, 10.0]),
                    DetectedObject(1, 2, "car", 0.7, [20.0, 20.0, 30.0, 30.0]),
                ]
            )

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_stats",
            overwrite=True,
        )
        self.assertEqual(summary.average_inference_time_ms, 60.0)
        self.assertEqual(summary.median_inference_time_ms, 60.0)
        self.assertEqual(summary.average_detections_per_successful_frame, 2.0)

    def test_23_error_report_generation(self) -> None:
        """Verify that errors payload populates correctly in detection_errors.json."""
        def mock_detect(image_path, **kwargs):
            if "frame_000001.jpg" in str(image_path):
                raise ImageValidationError("Corrupt header")
            return SingleImageDetectionResult(width=640, height=360, detection_count=0, detections=[])

        self.mock_detector.detect.side_effect = mock_detect
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        _, _, errors, run_dir = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_err_rep",
            overwrite=True,
        )
        self.assertEqual(len(errors), 1)
        err_file = run_dir / "detection_errors.json"
        data = json.loads(err_file.read_text(encoding="utf-8"))
        self.assertEqual(data["error_count"], 1)

    def test_24_unique_run_id_behavior(self) -> None:
        """Verify that consecutive runs generate distinct run_ids."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        s1, _, _, dir1 = detector.process_batch(dataset=self.mock_dataset_dir, output_dir=self.temp_path / "runs_id")
        s2, _, _, dir2 = detector.process_batch(dataset=self.mock_dataset_dir, output_dir=self.temp_path / "runs_id")
        self.assertNotEqual(s1.run_id, s2.run_id)
        self.assertNotEqual(dir1, dir2)

    def test_25_existing_output_protection(self) -> None:
        """Verify that attempting to reuse an explicit run_id without overwrite=True raises error."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        fixed_run = "run_fixed_test"
        detector.process_batch(
            dataset=self.mock_dataset_dir,
            run_id=fixed_run,
            output_dir=self.temp_path / "runs_protect",
            overwrite=True,
        )
        with self.assertRaises(BatchExecutionError):
            detector.process_batch(
                dataset=self.mock_dataset_dir,
                run_id=fixed_run,
                output_dir=self.temp_path / "runs_protect",
                overwrite=False,
            )

    # --------------------------------------------------------------------------
    # 26-28. Status Modes & Range Slicing Tests
    # --------------------------------------------------------------------------

    def test_26_partial_run_status_warning(self) -> None:
        """Verify that a smoke test (max_frames < total) reports status WARNING."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            max_frames=2,
            output_dir=self.temp_path / "runs_partial",
            overwrite=True,
        )
        self.assertTrue(summary.is_partial_run)
        self.assertEqual(summary.final_status, "WARNING")

    def test_27_full_run_status_pass(self) -> None:
        """Verify that processing the complete dataset with zero failures reports status PASS."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, _, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            output_dir=self.temp_path / "runs_full",
            overwrite=True,
        )
        self.assertFalse(summary.is_partial_run)
        self.assertEqual(summary.final_status, "PASS")

    def test_28_range_processing_start_end_index(self) -> None:
        """Verify range slicing with start_index and end_index."""
        self.mock_detector.detect.return_value = SingleImageDetectionResult(
            width=640, height=360, detection_count=0, detections=[]
        )
        detector = BatchDetector(detector=self.mock_detector, project_root=PROJECT_ROOT)
        summary, frames, _, _ = detector.process_batch(
            dataset=self.mock_dataset_dir,
            start_index=1,
            end_index=3,
            output_dir=self.temp_path / "runs_range",
            overwrite=True,
        )
        self.assertEqual(summary.total_frames_attempted, 2)
        self.assertEqual([f.image_filename for f in frames], ["frame_000002.jpg", "frame_000003.jpg"])

    # --------------------------------------------------------------------------
    # 29. Real-Model Integration Test
    # --------------------------------------------------------------------------

    def test_29_real_model_smoke_batch(self) -> None:
        """Verify live YOLO11n inference on 5 actual Day 2 preprocessed frames."""
        if not self.real_weights_path.is_file():
            self.skipTest("Real weights missing at models/yolo/yolo11n.pt")
        if not self.real_dataset_dir.is_dir():
            self.skipTest("Real dataset missing at datasets/processed/driving_sample/preprocessed_frames")

        cfg = load_yolo_config(project_root=PROJECT_ROOT, validate=True, validate_weights=True)
        loader = YOLOModelLoader(config=cfg, project_root=PROJECT_ROOT)
        real_detector = SingleImageDetector(loader=loader, project_root=PROJECT_ROOT)

        engine = BatchDetector(detector=real_detector, project_root=PROJECT_ROOT)
        summary, frames, errors, run_dir = engine.process_batch(
            dataset=self.real_dataset_dir,
            max_frames=5,
            output_dir=self.temp_path / "runs_real_smoke",
            overwrite=True,
        )

        self.assertEqual(summary.total_frames_attempted, 5)
        self.assertEqual(summary.successful_frames, 5)
        self.assertEqual(summary.failed_frames, 0)
        self.assertEqual(len(errors), 0)
        self.assertEqual(summary.final_status, "WARNING")  # partial smoke run
        self.assertGreater(summary.total_detections, 0)
        self.assertTrue((run_dir / "detection_results.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)

