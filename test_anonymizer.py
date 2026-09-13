"""
Pink Edge AI — Automated Unit Tests for DICOM Anonymization & Privacy Hashing
"""

import sys
import os
import json
import unittest

# Ensure root directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dicom_anonymizer import DICOMAnonymizer, generate_hex_privacy_hash, anonymize_dicom_metadata


class TestDICOMAnonymizer(unittest.TestCase):

    def setUp(self):
        self.anonymizer = DICOMAnonymizer(salt_key="TEST_SALT_1234")
        self.raw_patient = {
            "patient_name": "Fatima Noor",
            "cnic": "31202-7654321-9",
            "exact_location": "Basti Malook, Multan (GPS: 29.98, 71.45)",
            "dob": "1990-11-20",
            "phone": "+92-301-9876543",
            "pat_id": 98765432,
            "patient_age": 36,
            "patient_sex": "F",
            "modality": "MG",
            "body_part": "BREAST",
            "institution": "BHU Multan Rural"
        }

    def test_pii_stripping(self):
        """Verify that all PII fields are completely removed."""
        anonymized = self.anonymizer.anonymize_metadata_dict(self.raw_patient)
        
        # PII fields must NOT be present
        self.assertNotIn("patient_name", anonymized)
        self.assertNotIn("cnic", anonymized)
        self.assertNotIn("exact_location", anonymized)
        self.assertNotIn("dob", anonymized)
        self.assertNotIn("phone", anonymized)

    def test_hex_privacy_hash_format(self):
        """Verify that hex privacy hash starts with HEX- and contains 8 hex chars."""
        anonymized = self.anonymizer.anonymize_metadata_dict(self.raw_patient)
        privacy_hash = anonymized["privacy_hash"]

        self.assertTrue(privacy_hash.startswith("HEX-"))
        self.assertEqual(len(privacy_hash), 12)  # "HEX-" (4 chars) + 8 hex chars = 12
        
        # Check hex part is valid hexadecimal
        hex_part = privacy_hash[4:]
        int(hex_part, 16)  # Will raise ValueError if not valid hex

    def test_deterministic_hashing(self):
        """Verify that identical PII inputs produce identical hex hashes."""
        hash1 = generate_hex_privacy_hash(self.raw_patient["cnic"], salt_key="TEST_KEY")
        hash2 = generate_hex_privacy_hash(self.raw_patient["cnic"], salt_key="TEST_KEY")
        
        self.assertEqual(hash1, hash2)

    def test_clinical_preservation(self):
        """Verify that essential clinical metrics for diagnosis are preserved."""
        anonymized = self.anonymizer.anonymize_metadata_dict(self.raw_patient)
        
        self.assertEqual(anonymized["patient_age"], 36)
        self.assertEqual(anonymized["patient_sex"], "F")
        self.assertEqual(anonymized["modality"], "MG")
        self.assertEqual(anonymized["body_part"], "BREAST")

    def test_hipaa_status(self):
        """Verify compliance status field is set."""
        anonymized = self.anonymizer.anonymize_metadata_dict(self.raw_patient)
        self.assertEqual(anonymized["hipaa_gdpr_status"], "COMPLIANT_OFFLINE_HEX_HASHED")


if __name__ == "__main__":
    unittest.main()
