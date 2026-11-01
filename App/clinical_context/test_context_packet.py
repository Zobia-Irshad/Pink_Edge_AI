try:
    from .context_packet import build_context_packet
except ImportError:
    from context_packet import build_context_packet


def test_packet_contains_required_metadata():
    packet = build_context_packet(
        patient_id="TEST-001",
        modality="tuberculosis",
        clinical_context={},
    )

    assert packet["patient_id"] == "TEST-001"
    assert packet["modality"] == "tuberculosis"
    assert packet["schema_version"] == "1.0"
    assert "preprocessing_result" in packet
    assert "missing_fields" in packet


def test_empty_context_marks_missing_fields():
    packet = build_context_packet("TEST-002", "tuberculosis", {})

    assert set(packet["missing_fields"]) == {
        "age", "cough", "fever", "hemoptysis", "prior_tb", "smoking_status"}


def test_supplied_fields_are_not_missing():
    packet = build_context_packet(
        "TEST-003", "tuberculosis", {"age": 40, "smoking_status": "former"})

    assert "age" not in packet["missing_fields"]
    assert "smoking_status" not in packet["missing_fields"]


def test_invalid_patient_id_is_rejected():
    try:
        build_context_packet("", "tuberculosis", {})
    except ValueError:
        pass
    else:
        raise AssertionError("Empty patient ID should be rejected.")
