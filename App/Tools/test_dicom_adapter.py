from dicom_adapter import dicom_to_pil

image, ds = dicom_to_pil(
    r"C:\Users\pc\Downloads\test_ultrasound.dcm"
)
)

print("DICOM converted successfully")
print("Patient ID:", getattr(ds, "PatientID", "Not available"))
print("Modality:", getattr(ds, "Modality", "Unknown"))
print("Image size:", image.size)
print("Image mode:", image.mode)

image.save("dicom_test_output.png")

print("Converted image saved as dicom_test_output.png")
