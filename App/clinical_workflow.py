from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class WorkflowResult:
    patient_id: str
    modality: str
    timestamp: str
    status: str
    inference: dict
    next_action: str
    human_review_required: bool

def run_workflow(patient_id, modality, image, inference_function):
    timestamp = datetime.now().isoformat(timespec='seconds')
    try:
        result = inference_function(image)
        if not result:
            return WorkflowResult(patient_id, modality, timestamp, 'inference_unavailable', {}, 'Inference unavailable. Route case for manual clinical review.', True)
        critical = bool(result.get('is_critical', False))
        try:
            confidence = float(result.get('confidence'))
        except (TypeError, ValueError):
            confidence = None
        if critical:
           action = 'AI flagged a finding. Route to a qualified clinical specialist for review.'
        elif confidence is not None and confidence < 60:
            action = 'AI confidence is low. Repeat or review the input and obtain specialist assessment.'
        else:
            action = 'AI screening did not trigger escalation. Human clinical review is still required before any clinical decision.'
        return WorkflowResult(patient_id, modality, timestamp, 'completed', result, action, True)
    except Exception as exc:
        return WorkflowResult(patient_id, modality, timestamp, 'error', {'error': str(exc)}, 'Inference error. Route case for manual clinical review.', True)

def run_maternal_workflow(patient_id, image):
    from .inference import predict_maternal
    return run_workflow(patient_id, 'maternal_ultrasound', image, predict_maternal)

def run_tb_workflow(patient_id, image):
    from .inference import predict_tb
    return run_workflow(patient_id, 'tuberculosis', image, predict_tb)

def run_mammography_workflow(patient_id, image):
    from .inference import predict_mammography
    return run_workflow(patient_id, 'mammography', image, predict_mammography)

def workflow_to_dict(result):
    return asdict(result)
