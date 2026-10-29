
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
    packet = build_context_packet(
        patient_id="TEST-002",
        modality="tuberculosis",
        clinical_context={},
    )

    assert set(packet["missing_fields"]) == set(
        __import__(
            build_context_packet.__module__.rsplit(".", 1)[0]
            + ".preprocessing"
            if "." in build_context_packet.__module__
            else "preprocessing"
        ).FEATURE_NAMES
    )


def test_invalid_patient_id_is_rejected():
    try:
        build_context_packet("", "tuberculosis", {})
    except ValueError:
        pass
    else:
        raise AssertionError("Empty patient ID should be rejected.")
