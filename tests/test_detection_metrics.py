"""
Discoverability wrapper for detection metrics tests.
Allows running via `pytest tests/test_detection_metrics.py` or `python tests/test_detection_metrics.py`.
"""
from tests.analysis.test_detection_metrics import TestDetectionMetrics

__all__ = ["TestDetectionMetrics"]

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
