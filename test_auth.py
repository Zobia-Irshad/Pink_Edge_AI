"""
Pink Edge AI — Automated Unit Tests for Multi-Tenant Access Control & RBAC
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth_manager import AuthManager, ROLE_LHW, ROLE_RADIOLOGIST, check_pin, is_permission_granted


class TestAuthManager(unittest.TestCase):

    def setUp(self):
        self.auth = AuthManager(current_role=ROLE_LHW)

    def test_default_role(self):
        """Verify default active role is LHW."""
        profile = self.auth.get_active_profile()
        self.assertEqual(profile["name"], "Lady Health Worker (LHW)")
        self.assertEqual(self.auth.current_role, ROLE_LHW)

    def test_pin_authentication(self):
        """Verify PIN 1111 authenticates as LHW and PIN 9999 authenticates as Radiologist."""
        role1 = self.auth.authenticate_pin("1111")
        self.assertEqual(role1, ROLE_LHW)

        role2 = self.auth.authenticate_pin("9999")
        self.assertEqual(role2, ROLE_RADIOLOGIST)

        invalid_role = self.auth.authenticate_pin("1234")
        self.assertIsNone(invalid_role)

    def test_lhw_permissions_restriction(self):
        """Verify LHW has basic triage permissions but restricted diagnostic override/cloud permissions."""
        self.auth.current_role = ROLE_LHW

        self.assertTrue(self.auth.has_permission("view_binary_verdict"))
        self.assertTrue(self.auth.has_permission("download_reports"))
        
        # Restricted for LHW
        self.assertFalse(self.auth.has_permission("override_assessment"))
        self.assertFalse(self.auth.has_permission("view_telemetry_logs"))
        self.assertFalse(self.auth.has_permission("view_cloud_sync"))
        self.assertFalse(self.auth.has_permission("view_hospital_hub"))

    def test_radiologist_permissions_full(self):
        """Verify Radiologist profile has full unrestricted access."""
        self.auth.current_role = ROLE_RADIOLOGIST

        self.assertTrue(self.auth.has_permission("view_binary_verdict"))
        self.assertTrue(self.auth.has_permission("override_assessment"))
        self.assertTrue(self.auth.has_permission("view_telemetry_logs"))
        self.assertTrue(self.auth.has_permission("view_hardware_stats"))
        self.assertTrue(self.auth.has_permission("view_hospital_hub"))
        self.assertTrue(self.auth.has_permission("view_cloud_sync"))

    def test_biometric_login(self):
        """Verify biometric 1-touch login simulation switches roles."""
        res = self.auth.simulated_biometric_login(ROLE_RADIOLOGIST)
        self.assertTrue(res)
        self.assertEqual(self.auth.current_role, ROLE_RADIOLOGIST)


if __name__ == "__main__":
    unittest.main()
