from App.clinical_workflow import (
    run_maternal_workflow,
    workflow_to_dict,
)
from App.Tools.dicom_adapter import dicom_to_pil


DICOM_FILE = r"App\test_ultrasound_valid.dcm"

print("=== PINK EDGE CLINICAL WORKFLOW TEST ===")

image, ds = dicom_to_pil(DICOM_FILE)

result = run_maternal_workflow(
    getattr(ds, "PatientID", "UNKNOWN"),
    image,
)

print("\nWorkflow result:")
print(workflow_to_dict(result))

print("\nStatus:", result.status)
print("Human review required:", result.human_review_required)
print("Next action:", result.next_action)

if result.status == "completed" and result.human_review_required:
    print("\nWORKFLOW TEST PASSED")
else:
    print("\nWORKFLOW TEST FAILED")