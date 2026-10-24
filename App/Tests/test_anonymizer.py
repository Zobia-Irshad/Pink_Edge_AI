"""
Pink Edge AI — Automated Unit Tests for DICOM Anonymization & Privacy Hashing

Tests `dicom_anonymizer.py` as it actually ships. The previous version of this file was written
against a never-implemented `DICOMAnonymizer(salt_key=...)` class whose
`anonymize_metadata_dict()` was expected to *drop* PII keys and inject `privacy_hash` /
`hipaa_gdpr_status` fields, with hashes formatted as `HEX-` + 8 chars. None of that exists, so
the file failed at import with ImportError before reaching a single assertion.

What actually ships is two pure functions:
  * `generate_hex_privacy_hash(dict)`     -> full uppercase SHA-256 hex of a canonical JSON dump
  * `anonymize_dicom_metadata(tag_dict)`  -> replaces values of known-sensitive DICOM tags with
                                             per-tag hashes, preserving every other key as-is

The original intent — PII never survives in cleartext, hashing is deterministic, clinical fields
are preserved — is kept here and asserted against that real surface.

Run:  python Tests/test_anonymizer.py        (from App/)
"""

import os
import re
import sys
import unittest

# Tests/ lives one level below the modules under test.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicom_anonymizer import (  # noqa: E402
    SENSITIVE_TAGS,
    anonymize_dicom_metadata,
    anonymize_dicom_tags,
    generate_hex_privacy_hash,
)

SHA256_HEX = re.compile(r"^[0-9A-F]{64}$")


class TestGenerateHexPrivacyHash(unittest.TestCase):
    def setUp(self):
        # The exact shape the app hashes — see streamlit_app.py's init_state().
        self.pii = {"pat_id": 98765432, "cnic": "31202-7654321-9"}

    def test_returns_uppercase_sha256_hex(self):
        self.assertRegex(generate_hex_privacy_hash(self.pii), SHA256_HEX)

    def test_is_deterministic(self):
        self.assertEqual(
            generate_hex_privacy_hash(self.pii),
            generate_hex_privacy_hash(dict(self.pii)),
        )

    def test_is_independent_of_key_insertion_order(self):
        """Canonicalization sorts keys, so the same facts always hash alike."""
        reordered = {"cnic": self.pii["cnic"], "pat_id": self.pii["pat_id"]}
        self.assertEqual(
            generate_hex_privacy_hash(self.pii),
            generate_hex_privacy_hash(reordered),
        )

    def test_distinct_pii_yields_distinct_hashes(self):
        other = {"pat_id": 98765433, "cnic": "31202-7654321-9"}
        self.assertNotEqual(
            generate_hex_privacy_hash(self.pii),
            generate_hex_privacy_hash(other),
        )

    def test_hash_does_not_leak_any_input_value(self):
        """The whole point: no identifier may be recoverable by substring from the digest."""
        digest = generate_hex_privacy_hash(self.pii)
        for value in ("98765432", "31202-7654321-9", "31202", "7654321"):
            self.assertNotIn(value, digest)

    def test_non_string_values_are_coerced_not_crashed(self):
        """`default=str` lets ints, None and nested values through unharmed."""
        self.assertRegex(
            generate_hex_privacy_hash({"a": 1, "b": None, "c": ["x", 2]}),
            SHA256_HEX,
        )

    def test_empty_dict_is_valid_input(self):
        self.assertRegex(generate_hex_privacy_hash({}), SHA256_HEX)

    def test_alternate_algorithm_is_honoured(self):
        sha1 = generate_hex_privacy_hash(self.pii, algorithm="sha1")
        self.assertRegex(sha1, r"^[0-9A-F]{40}$")
        self.assertNotEqual(sha1, generate_hex_privacy_hash(self.pii))

    def test_rejects_non_dict_input(self):
        for bad in ("a string", 1234, None, ["a", "list"]):
            with self.assertRaises(TypeError):
                generate_hex_privacy_hash(bad)

    def test_rejects_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            generate_hex_privacy_hash(self.pii, algorithm="not-a-hash")


class TestAnonymizeDicomMetadata(unittest.TestCase):
    def setUp(self):
        self.tags = {
            "PatientName": "Fatima Noor",
            "PatientID": "31202-7654321-9",
            "PatientBirthDate": "19901120",
            "PatientAddress": "Basti Malook, Multan",
            "PatientTelephoneNumbers": "+92-301-9876543",
            "ReferringPhysicianName": "Dr. Ahmed",
            "InstitutionName": "BHU Multan Rural",
            # Clinically necessary, must survive untouched:
            "PatientAge": 36,
            "PatientSex": "F",
            "Modality": "MG",
            "BodyPartExamined": "BREAST",
        }

    def test_every_sensitive_tag_is_hashed(self):
        out = anonymize_dicom_metadata(self.tags)
        for tag in SENSITIVE_TAGS:
            if tag in self.tags:
                self.assertRegex(out[tag], SHA256_HEX, tag)

    def test_no_sensitive_value_survives_in_cleartext(self):
        out = anonymize_dicom_metadata(self.tags)
        leaked = [
            str(self.tags[tag])
            for tag in SENSITIVE_TAGS
            if tag in self.tags and str(self.tags[tag]) in str(out.values())
        ]
        self.assertEqual(leaked, [], f"PII survived anonymization: {leaked}")

    def test_clinical_fields_are_preserved_exactly(self):
        out = anonymize_dicom_metadata(self.tags)
        self.assertEqual(out["PatientAge"], 36)
        self.assertEqual(out["PatientSex"], "F")
        self.assertEqual(out["Modality"], "MG")
        self.assertEqual(out["BodyPartExamined"], "BREAST")

    def test_all_keys_are_retained(self):
        """This implementation redacts values in place; it does not drop keys."""
        out = anonymize_dicom_metadata(self.tags)
        self.assertEqual(set(out), set(self.tags))

    def test_input_dict_is_not_mutated(self):
        original = dict(self.tags)
        anonymize_dicom_metadata(self.tags)
        self.assertEqual(self.tags, original)

    def test_is_deterministic_across_calls(self):
        self.assertEqual(
            anonymize_dicom_metadata(self.tags),
            anonymize_dicom_metadata(self.tags),
        )

    def test_same_value_under_different_tags_hashes_differently(self):
        """Hashes are salted by tag name, so cross-field correlation is not trivial."""
        out = anonymize_dicom_metadata({"PatientName": "X", "PatientID": "X"})
        self.assertNotEqual(out["PatientName"], out["PatientID"])

    def test_unknown_tags_pass_through(self):
        out = anonymize_dicom_metadata({"SomeVendorTag": "keep me"})
        self.assertEqual(out["SomeVendorTag"], "keep me")

    def test_empty_dict_returns_empty_dict(self):
        self.assertEqual(anonymize_dicom_metadata({}), {})

    def test_rejects_non_dict_input(self):
        for bad in ("a string", 1234, None, ["a", "list"]):
            with self.assertRaises(TypeError):
                anonymize_dicom_metadata(bad)

    def test_legacy_alias_matches(self):
        self.assertEqual(
            anonymize_dicom_tags(self.tags),
            anonymize_dicom_metadata(self.tags),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
