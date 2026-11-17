"""
Discoverability wrapper for detection visualizer tests.
Allows running via `pytest tests/test_detection_visualizer.py` or `python tests/test_detection_visualizer.py`.
"""
from tests.visualization.test_detection_visualizer import TestDetectionVisualizer

__all__ = ["TestDetectionVisualizer"]

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
