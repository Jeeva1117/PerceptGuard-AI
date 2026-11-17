"""PerceptGuard-AI: Root tests entrypoint for YOLO model configuration test suite.

Delegates to tests/detection/test_yolo_config.py for unified discovery.
"""

from tests.detection.test_yolo_config import TestYOLOModelConfiguration

__all__ = ["TestYOLOModelConfiguration"]

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)

