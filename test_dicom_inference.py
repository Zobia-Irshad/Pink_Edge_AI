from App.Tools.dicom_adapter import dicom_to_pil
from App.inference import predict_maternal


DICOM_FILE = "App/test_ultrasound.dcm"

print("Reading DICOM...")

image, ds = dicom_to_pil(DICOM_FILE)

print("DICOM loaded successfully")
print("Patient ID:", getattr(ds, "PatientID", "Unknown"))
print("Modality:", getattr(ds, "Modality", "Unknown"))
print("Image size:", image.size)


print("\nRunning Pink Edge maternal inference...")

result = predict_maternal(image)

if result is None:
    print("\nNo model result was returned.")
    print("The DICOM adapter works, but the maternal inference pipeline did not return a result.")
else:
    print("\nInference completed.")
    print("Result:")
    print(result)
