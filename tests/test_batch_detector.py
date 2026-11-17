"""
Discoverability wrapper for batch object detector tests.
Allows running via `pytest tests/test_batch_detector.py` or `python tests/test_batch_detector.py`.
"""
from tests.detection.test_batch_detector import TestBatchDetector

__all__ = ["TestBatchDetector"]

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)

