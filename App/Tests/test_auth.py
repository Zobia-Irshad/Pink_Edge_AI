"""
Pink Edge AI — Automated Unit Tests for Multi-Tenant Access Control & RBAC

Tests `auth_manager.py` as it actually ships. The previous version of this file was written
against a richer, never-implemented API (`AuthManager(current_role=...)`, a mutable
`.current_role`, `simulated_biometric_login()`, module-level `check_pin()` /
`is_permission_granted()`, and permission names like `view_telemetry_logs`) and so failed at
import with ImportError before reaching a single assertion. The intent of those tests —
PIN auth works, LHW is restricted, Radiologist is not — is preserved here against the real
`AuthManager` surface.

Run:  python Tests/test_auth.py        (from App/)
"""

import os
import sys
import unittest

# Tests/ lives one level below the modules under test.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_manager import (  # noqa: E402
    ROLE_CONFIGS,
    ROLE_LHW,
    ROLE_RADIOLOGIST,
    AuthManager,
)


class TestAuthManagerRoles(unittest.TestCase):
    def test_default_role_is_lhw(self):
        """AuthManager() with no argument defaults to the least-privileged role."""
        auth = AuthManager()
        self.assertEqual(auth.role, ROLE_LHW)
        self.assertTrue(auth.is_lhw())
        self.assertFalse(auth.is_radiologist())

    def test_role_accepted_positionally_and_by_keyword(self):
        """Both call styles used by the app (streamlit_app.py passes positionally)."""
        self.assertEqual(AuthManager(ROLE_RADIOLOGIST).role, ROLE_RADIOLOGIST)
        self.assertEqual(AuthManager(role=ROLE_RADIOLOGIST).role, ROLE_RADIOLOGIST)

    def test_unknown_role_falls_back_to_lhw(self):
        """An unrecognized role must degrade to the restricted role, never to full access."""
        for bogus in ("admin", "", None, "RADIOLOGIST"):
            self.assertEqual(AuthManager(bogus).role, ROLE_LHW)

    def test_active_profile_shape(self):
        """get_active_profile() returns the badge/color/pin/permissions the sidebar renders."""
        profile = AuthManager(ROLE_LHW).get_active_profile()
        for field in ("badge", "color", "pin", "permissions"):
            self.assertIn(field, profile)
        self.assertIn("Lady Health Worker", profile["badge"])
        self.assertTrue(profile["color"].startswith("#"))

    def test_radiologist_profile_badge(self):
        profile = AuthManager(ROLE_RADIOLOGIST).get_active_profile()
        self.assertIn("Senior Radiologist", profile["badge"])


class TestPinAuthentication(unittest.TestCase):
    def setUp(self):
        self.auth = AuthManager(ROLE_LHW)

    def test_valid_pins_map_to_their_roles(self):
        self.assertEqual(self.auth.authenticate_pin("1111"), ROLE_LHW)
        self.assertEqual(self.auth.authenticate_pin("9999"), ROLE_RADIOLOGIST)

    def test_invalid_pin_returns_none(self):
        for bad in ("1234", "0000", "", "99999", "abcd"):
            self.assertIsNone(self.auth.authenticate_pin(bad))

    def test_pin_is_whitespace_and_type_tolerant(self):
        """The UI hands over raw text-input content, so trim and coerce."""
        self.assertEqual(self.auth.authenticate_pin("  9999  "), ROLE_RADIOLOGIST)
        self.assertEqual(self.auth.authenticate_pin(9999), ROLE_RADIOLOGIST)

    def test_authenticate_does_not_mutate_the_instance(self):
        """AuthManager is stateless by design — the caller stores the returned role."""
        self.assertEqual(self.auth.authenticate_pin("9999"), ROLE_RADIOLOGIST)
        self.assertEqual(self.auth.role, ROLE_LHW)


class TestPermissions(unittest.TestCase):
    def test_lhw_has_triage_basics(self):
        auth = AuthManager(ROLE_LHW)
        for allowed in ("view_results", "download_report", "run_triage"):
            self.assertTrue(auth.has_permission(allowed), allowed)

    def test_lhw_is_denied_privileged_actions(self):
        auth = AuthManager(ROLE_LHW)
        for denied in ("override_assessment", "view_audit_log", "export_dicom", "manage_users"):
            self.assertFalse(auth.has_permission(denied), denied)

    def test_radiologist_has_every_lhw_permission_plus_more(self):
        """Privilege must be strictly additive, so elevating never removes a capability."""
        lhw = ROLE_CONFIGS[ROLE_LHW]["permissions"]
        rad = ROLE_CONFIGS[ROLE_RADIOLOGIST]["permissions"]
        self.assertTrue(lhw.issubset(rad))
        self.assertTrue(rad - lhw)

    def test_unknown_permission_is_denied_not_errored(self):
        """A typo'd permission string must fail closed rather than raise."""
        self.assertFalse(AuthManager(ROLE_RADIOLOGIST).has_permission("no_such_permission"))

    def test_permissions_gated_by_the_live_app_all_exist(self):
        """
        Regression guard. `has_permission()` fails closed, so gating on a permission that is
        absent from ROLE_CONFIGS silently hides that feature from *every* role, including the
        Radiologist — with no error to notice. The archived pink_edge.py does exactly this with
        `view_hardware_stats` / `view_telemetry_logs`. Any permission the shipping UI checks
        must therefore be declared here.
        """
        gated_by_app = {"override_assessment"}
        declared = set().union(*(cfg["permissions"] for cfg in ROLE_CONFIGS.values()))
        self.assertTrue(
            gated_by_app.issubset(declared),
            f"UI gates on undeclared permission(s): {sorted(gated_by_app - declared)}",
        )


class TestRoleConfigIntegrity(unittest.TestCase):
    def test_every_role_is_fully_configured(self):
        for role, cfg in ROLE_CONFIGS.items():
            self.assertIsInstance(cfg["permissions"], set, role)
            self.assertTrue(cfg["badge"], role)
            self.assertTrue(cfg["color"].startswith("#"), role)

    def test_pins_are_unique_so_authentication_is_unambiguous(self):
        pins = [cfg["pin"] for cfg in ROLE_CONFIGS.values() if cfg.get("pin")]
        self.assertEqual(len(pins), len(set(pins)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
