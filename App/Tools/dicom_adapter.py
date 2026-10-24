import numpy as np
import pydicom
from PIL import Image


def dicom_to_pil(dicom_path):
    """
    Convert a DICOM file into a PIL image that the
    existing Pink Edge inference functions can consume.
    """

    ds = pydicom.dcmread(dicom_path)

    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM file contains no PixelData.")

    pixels = ds.pixel_array.astype(np.float32)

    # Handle multi-frame DICOM.
    # Use the first frame for this prototype.
    if pixels.ndim > 2:
        pixels = pixels[0]

    # Normalize pixel values to 0-255.
    minimum = pixels.min()
    maximum = pixels.max()

    if maximum > minimum:
        pixels = (pixels - minimum) / (maximum - minimum) * 255.0
    else:
        pixels = np.zeros_like(pixels)

    pixels = pixels.astype(np.uint8)

    image = Image.fromarray(pixels)

    return image, ds
