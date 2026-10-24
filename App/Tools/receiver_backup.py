"""
Pink Edge AI — standalone DICOM C-STORE listener (a minimal local PACS bridge).

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Accepts scans pushed from hospital imaging equipment and writes them to ./received_scans.
Standalone: the app never imports this. Needs the optional deps in requirements.txt
(pynetdicom, pydicom).

Run:  python Tools/receiver.py
"""
import os
from pynetdicom import AE, evt
import logging
from pynetdicom.sop_class import Verification, CTImageStorage, MRImageStorage, UltrasoundImageStorage, DigitalMammographyXRayImageStorageForPresentation

# Folder where incoming scans will save automatically
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

OUTPUT_DIR = os.path.join(BASE_DIR, "received_scans")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def handle_store(event):
    print(">>> C-STORE EVENT RECEIVED")

    try:
        ds = event.dataset
        ds.file_meta = event.file_meta

        print("Patient ID:", getattr(ds, "PatientID", "UNKNOWN"))
        print("Modality:", getattr(ds, "Modality", "UNKNOWN"))
        print("SOP Instance UID:", getattr(ds, "SOPInstanceUID", "UNKNOWN"))

        filepath = os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                f"{ds.SOPInstanceUID}.dcm"
            )
        )

        print("Saving to:", filepath)

        ds.save_as(filepath, write_like_original=False)

        print(">>> DICOM SAVED SUCCESSFULLY")
        print(filepath)

        return 0x0000

    except Exception as exc:
        print(">>> C-STORE ERROR")
        print(type(exc).__name__, ":", exc)

        return 0xC211
# Initialize Application Entity
ae = AE(ae_title=b'MY_LOCAL_PACS')

# Add supported contexts safely
ae.add_supported_context(Verification)
ae.add_supported_context(CTImageStorage)
ae.add_supported_context(MRImageStorage)
ae.add_supported_context(UltrasoundImageStorage)
ae.add_supported_context(DigitalMammographyXRayImageStorageForPresentation)

print("Bridge listening for hospital scans on port 11112...")

logging.basicConfig(level=logging.DEBUG)

ae.start_server(
    ('0.0.0.0', 11112),
    evt_handlers=[(evt.EVT_C_STORE, handle_store)]
)