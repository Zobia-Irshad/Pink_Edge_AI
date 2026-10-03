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
from pynetdicom.sop_class import Verification, CTImageStorage, MRImageStorage, UltrasoundImageStorage, DigitalMammographyXRayImageStorageForPresentation

# Folder where incoming scans will save automatically
OUTPUT_DIR = "./received_scans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handle_store(event):
    ds = event.dataset
    ds.file_meta = event.file_meta
    filepath = os.path.join(OUTPUT_DIR, f"{ds.SOPInstanceUID}.dcm")
    ds.save_as(filepath, write_like_original=False)
    print(f"Scan received and saved: {filepath}")
    return 0x0000

# Initialize Application Entity
ae = AE(ae_title=b'MY_LOCAL_PACS')

# Add supported contexts safely
ae.add_supported_context(Verification)
ae.add_supported_context(CTImageStorage)
ae.add_supported_context(MRImageStorage)
ae.add_supported_context(UltrasoundImageStorage)
ae.add_supported_context(DigitalMammographyXRayImageStorageForPresentation)

print("Bridge listening for hospital scans on port 11112...")
ae.start_server(('0.0.0.0', 11112), evt_handlers=[(evt.EVT_C_STORE, handle_store)])