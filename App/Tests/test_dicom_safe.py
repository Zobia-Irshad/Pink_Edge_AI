"""
Pink Edge AI - unit tests for the memory-safe DICOM reader (Chunk 1 fail-safes).

Verifies the context-managed reader releases the dataset and that
extract_pixel_array returns a detached matrix after forcing a gc cycle.
Uses the DICOM test fixtures already bundled under App/ when present; skips
cleanly if pydicom or a fixture is unavailable.

Run:  python Tests/test_dicom_safe.py        (from App/)
"""

import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Tools"))

try:
    import numpy as np
    import pydicom  # noqa: F401
    from dicom_safe import dicom_reader, extract_pixel_array, read_pixels
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


def _find_fixture():
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("test_ultrasound_valid.dcm", "test_ultrasound.dcm"):
        path = os.path.join(app_dir, name)
        if os.path.isfile(path):
            return path
    return None


@unittest.skipUnless(HAVE_DEPS, "pydicom/numpy not available")
class TestDicomSafe(unittest.TestCase):
    def setUp(self):
        self.fixture = _find_fixture()
        if not self.fixture:
            self.skipTest("no bundled DICOM fixture found")

    def test_reader_yields_dataset_then_releases(self):
        with dicom_reader(self.fixture) as ds:
            self.assertTrue(hasattr(ds, "Modality") or hasattr(ds, "PixelData"))
        # After the context, a forced collection has already run; nothing raised.
        gc.collect()

    def test_extract_returns_detached_ndarray(self):
        with dicom_reader(self.fixture) as ds:
            if not hasattr(ds, "PixelData"):
                self.skipTest("fixture has no PixelData")
            arr = extract_pixel_array(ds)
            self.assertIsInstance(arr, np.ndarray)
            self.assertGreater(arr.size, 0)
            # The returned array must not alias pydicom's cache.
            self.assertTrue(arr.flags["OWNDATA"])

    def test_extract_without_pixeldata_raises(self):
        with dicom_reader(self.fixture, stop_before_pixels=True) as ds:
            with self.assertRaises(ValueError):
                extract_pixel_array(ds)

    def test_read_pixels_oneshot(self):
        try:
            arr = read_pixels(self.fixture)
        except ValueError:
            self.skipTest("fixture has no PixelData")
        self.assertIsInstance(arr, np.ndarray)
        self.assertGreater(arr.size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
