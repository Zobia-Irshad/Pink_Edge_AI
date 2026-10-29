
import unittest

from preprocessing import preprocess_patient_context


class PatientContextTests(unittest.TestCase):

    def test_all_fields_missing(self):
        result = preprocess_patient_context({})

        self.assertEqual(len(result["features"]), 8)
        self.assertEqual(result["features"], [0.0] * 8)
        self.assertEqual(result["mask"], [0] * 8)
        self.assertFalse(result["has_any_clinical_data"])

    def test_valid_patient_data(self):
        result = preprocess_patient_context({
            "age": 60,
            "cough": "yes",
            "fever": "no",
            "hemoptysis": False,
            "prior_tb": True,
            "smoking_status": "current",
        })

        self.assertEqual(len(result["features"]), 8)
        self.assertEqual(result["mask"], [1] * 8)
        self.assertAlmostEqual(result["features"][0], 0.5)
        self.assertTrue(result["has_any_clinical_data"])
        self.assertEqual(result["warnings"], [])

    def test_invalid_age_is_masked(self):
        result = preprocess_patient_context({
            "age": "not-an-age"
        })

        self.assertEqual(result["mask"][0], 0)
        self.assertEqual(result["features"][0], 0.0)
        self.assertTrue(result["warnings"])

    def test_invalid_symptom_is_masked(self):
        result = preprocess_patient_context({
            "cough": ["yes", "no"]
        })

        self.assertEqual(result["mask"][1], 0)
        self.assertEqual(result["features"][1], 0.0)
        self.assertTrue(result["warnings"])

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_patient_context({
                "unknown_field": "value"
            })

    def test_wrong_input_type_is_rejected(self):
        with self.assertRaises(TypeError):
            preprocess_patient_context(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
