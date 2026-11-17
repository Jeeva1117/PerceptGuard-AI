"""PerceptGuard-AI: Root tests entrypoint for SingleImageDetector test suite.

Delegates to tests/detection/test_single_image_detector.py.
"""

from tests.detection.test_single_image_detector import TestSingleImageDetector

__all__ = ["TestSingleImageDetector"]

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
