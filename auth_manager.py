"""
Pink Edge AI — Multi-Tenant Biometric Access Control & RBAC Module
====================================================================
Establishes role-based access control (RBAC), clinical data security,
and liability protection for rural Lady Health Workers (LHW) vs urban
Senior Radiologists / Supervisors.
"""

from typing import Dict, Any, Optional

# Defined Role Profiles
ROLE_LHW = "lhw"
ROLE_RADIOLOGIST = "radiologist"

ROLE_CONFIGS = {
    ROLE_LHW: {
        "name": "Lady Health Worker (LHW)",
        "title": "Rural VHU Triage Intake Profile",
        "pin": "1111",
        "badge": "👤 LHW Rural Intake Mode",
        "icon": "👤",
        "color": "#10b981",
        "description": "Simplified rural intake, binary triage verdict ('Normal' vs 'Referral Required'), and primary screening.",
        "permissions": {
            "view_binary_verdict": True,
            "view_patient_intake": True,
            "download_reports": True,
            "view_hex_privacy_hash": True,
            "view_full_metadata": False,
            "override_assessment": False,
            "view_telemetry_logs": False,
            "view_hardware_stats": False,
            "view_hospital_hub": False,
            "view_cloud_sync": False,
            "manage_ota": False,
        }
    },
    ROLE_RADIOLOGIST: {
        "name": "Senior Radiologist",
        "title": "Urban Hub Clinical Intelligence Suite",
        "pin": "9999",
        "badge": "👨‍⚕️ Senior Radiologist Clinical Suite",
        "icon": "👨‍⚕️",
        "color": "#0284c7",
        "description": "Full clinical intelligence suite, raw NPU latency metrics, BI-RADS/ACR overrides, Telemetry Logs, Hospital Hub, & Cloud Sync.",
        "permissions": {
            "view_binary_verdict": True,
            "view_patient_intake": True,
            "download_reports": True,
            "view_hex_privacy_hash": True,
            "view_full_metadata": True,
            "override_assessment": True,
            "view_telemetry_logs": True,
            "view_hardware_stats": True,
            "view_hospital_hub": True,
            "view_cloud_sync": True,
            "manage_ota": True,
        }
    }
}


class AuthManager:
    """
    Manages multi-tenant authentication, numeric PIN validation,
    simulated biometric/RFID logins, and role permissions.
    """

    def __init__(self, current_role: str = ROLE_LHW):
        self.current_role = current_role if current_role in ROLE_CONFIGS else ROLE_LHW

    def authenticate_pin(self, pin: str) -> Optional[str]:
        """
        Validates numeric PIN and returns the matching role identifier if successful.
        '1111' -> LHW, '9999' -> Radiologist.
        """
        pin_clean = str(pin).strip()
        for role_key, config in ROLE_CONFIGS.items():
            if config["pin"] == pin_clean:
                self.current_role = role_key
                return role_key
        return None

    def simulated_biometric_login(self, role: str) -> bool:
        """
        Simulates a 1-touch fingerprint or RFID card scan authentication.
        """
        if role in ROLE_CONFIGS:
            self.current_role = role
            return True
        return False

    def get_active_profile(self) -> Dict[str, Any]:
        """Returns metadata for the currently active role profile."""
        return ROLE_CONFIGS[self.current_role]

    def has_permission(self, permission_name: str) -> bool:
        """Checks if the currently active role has a specific permission."""
        perms = ROLE_CONFIGS[self.current_role]["permissions"]
        return perms.get(permission_name, False)


# Convenience functions for single-line checks
def check_pin(pin: str) -> Optional[str]:
    mgr = AuthManager()
    return mgr.authenticate_pin(pin)


def is_permission_granted(role: str, permission_name: str) -> bool:
    if role not in ROLE_CONFIGS:
        return False
    return ROLE_CONFIGS[role]["permissions"].get(permission_name, False)
