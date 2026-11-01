"""
dicom_safe.py
-------------
Memory-safe DICOM reading helpers for Pink Edge AI (Chunk 1 fail-safes).

The edge targets (RK3588-class SBCs) have strict RAM boundaries, so every
DICOM read must release its file handle the instant extraction is done and
must not leave high-dimensional pixel matrices pinned in the heap.

This module provides:

* ``dicom_reader(path)`` - a context manager that opens a DICOM dataset and
  guarantees the underlying file handle is closed on exit, even on error.
* ``extract_pixel_array(ds)`` - pulls ``dataset.pixel_array`` out and forces a
  manual garbage-collection cycle immediately afterwards to prevent
  Out-Of-Memory (OOM) faults from the transient decode buffers pydicom builds.

No heavy dependencies: only ``gc`` (stdlib) plus ``pydicom``/``numpy`` which
the DICOM pipeline already requires.
"""

import gc
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pydicom


@contextmanager
def dicom_reader(path: str, *, stop_before_pixels: bool = False) -> Iterator["pydicom.dataset.FileDataset"]:
    """Open a DICOM file and always release its file handle on exit.

    ``pydicom.dcmread`` is not itself a context manager, so this wrapper gives
    the ``with pydicom.dcmread(path) as dcm:`` ergonomics the task calls for
    while guaranteeing cleanup via ``try/finally``:

        with dicom_reader(path) as ds:
            pixels = extract_pixel_array(ds)

    Parameters
    ----------
    path:
        Filesystem path to the DICOM instance.
    stop_before_pixels:
        When True, metadata is read but PixelData is left on disk (useful for
        cheap metadata-only passes that must not pull a large matrix into RAM).
    """
    ds = pydicom.dcmread(path, stop_before_pixels=stop_before_pixels)
    try:
        yield ds
    finally:
        # Drop any decoded pixel buffer pydicom cached on the dataset and close
        # the file-like object if one is still held open (deferred reads).
        _close_dataset(ds)
        del ds
        gc.collect()


def _close_dataset(ds) -> None:
    """Best-effort release of a dataset's file handle and cached pixel buffer."""
    # Deferred / lazy reads keep a handle on ``filename`` via an internal buffer.
    buffer = getattr(ds, "filename", None)
    if hasattr(buffer, "close"):
        try:
            buffer.close()
        except Exception:
            pass
    # pydicom caches the decoded array on ``_pixel_array``; drop it so it can be
    # collected rather than lingering for the dataset's lifetime.
    for attr in ("_pixel_array", "_pixel_id"):
        if hasattr(ds, attr):
            try:
                setattr(ds, attr, None)
            except Exception:
                pass


def extract_pixel_array(ds) -> np.ndarray:
    """Extract ``ds.pixel_array`` and immediately reclaim transient heap memory.

    pydicom decodes PixelData into intermediate buffers (handler scratch space,
    byte copies) that can several-times exceed the final array. We copy the
    final matrix, drop pydicom's cached reference, and force ``gc.collect()`` so
    those intermediates are freed before inference allocates its own tensors.

    Returns a standalone ``np.ndarray`` that does not alias the dataset.
    """
    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM dataset contains no PixelData.")

    # ``.copy()`` detaches the array from pydicom's internal cache so releasing
    # the dataset below cannot invalidate it.
    pixels = np.array(ds.pixel_array, copy=True)

    # Release pydicom's cached decode so it is eligible for collection now.
    if hasattr(ds, "_pixel_array"):
        try:
            ds._pixel_array = None
        except Exception:
            pass

    # Heap protection: force a collection cycle right after extracting the
    # high-dimensional matrix, before the caller allocates inference buffers.
    gc.collect()

    return pixels


def read_pixels(path: str) -> np.ndarray:
    """Convenience one-shot: open, extract pixels, close + collect.

    Equivalent to::

        with dicom_reader(path) as ds:
            return extract_pixel_array(ds)
    """
    with dicom_reader(path) as ds:
        return extract_pixel_array(ds)
