from dicom_adapter import dicom_to_pil
from inference import predict_maternal

DICOM_FILE = r"C:\Users\pc\Downloads\test_ultrasound.dcm"

print("Reading DICOM...")
image, ds = dicom_to_pil(DICOM_FILE)

print("DICOM converted to image.")
print("Image size:", image.size)

print("Running Pink Edge maternal inference...")

result = predict_maternal(image)

print("\n===== PINK EDGE RESULT =====")

if result is None:
    print("No model result was returned.")
else:
    print("Prediction source:", result.get("source"))
    print("Verdict:", result.get("verdict"))
    print("Confidence:", result.get("confidence"))
    print("Recommendation:", result.get("recommendation"))
