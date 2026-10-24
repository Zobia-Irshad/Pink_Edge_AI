import os
import sys
import unittest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "Tools"))

from dicom_adapter import dicom_to_pil  # noqa: E402


class TestDicomAdapter(unittest.TestCase):
    def test_converts_bundled_dicom_fixture(self):
        dicom_path = os.path.join(APP_DIR, "test_ultrasound_valid.dcm")

        image, dataset = dicom_to_pil(dicom_path)

        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)
        self.assertEqual(image.mode, "L")
        self.assertTrue(hasattr(dataset, "PixelData"))


if __name__ == "__main__":
    unittest.main()
