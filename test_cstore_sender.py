from pydicom import dcmread
from pynetdicom import AE
from pynetdicom.sop_class import UltrasoundImageStorage

DICOM_FILE = r"received_scans\1.2.826.0.1.3680043.8.498.36978300989135663597145956691181645592.dcm"
# Load the DICOM file
ds = dcmread(DICOM_FILE)

print("DICOM loaded:")
print("Patient ID:", ds.PatientID)
print("Modality:", ds.Modality)
print("SOP Class:", ds.SOPClassUID)

# Create a DICOM Storage SCU
ae = AE(ae_title="TEST_SENDER")

# Request the same Storage SOP Class as the receiver
ae.add_requested_context(UltrasoundImageStorage)

print("\nConnecting to MY_LOCAL_PACS at 127.0.0.1:11112...")

assoc = ae.associate(
    "127.0.0.1",
    11112,
    ae_title="MY_LOCAL_PACS"
)

if assoc.is_established:
    print("Association established.")

    status = assoc.send_c_store(ds)

    if status:
        print("C-STORE status:", hex(status.Status))

        if status.Status == 0x0000:
            print("SUCCESS: DICOM was accepted by the receiver.")
        else:
            print("C-STORE failed.")

    else:
        print("No C-STORE response received.")

    assoc.release()
    print("Association released.")

else:
    print("FAILED: Could not establish DICOM association.")