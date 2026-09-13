"""
auth_manager.py
---------------
Role-based access control (RBAC) for Pink Edge AI.

Defines two user roles:
  - ROLE_LHW         : Lady Health Worker  — limited permissions
  - ROLE_RADIOLOGIST : Senior Radiologist  — full permissions (PIN: 9999)

No external dependencies — pure Python stdlib.
"""

# ── Role constants ──────────────────────────────────────────────────────────
ROLE_LHW         = "lhw"
ROLE_RADIOLOGIST = "radiologist"

# ── Role configuration ───────────────────────────────────────────────────────
# Each role entry contains:
#   badge       : display label shown in the sidebar
#   color       : hex accent colour for the profile card
#   pin         : numeric PIN string to elevate to this role (None = no PIN auth)
#   permissions : set of capability strings checked by has_permission()
ROLE_CONFIGS = {
    ROLE_LHW: {
        "badge": "👤 Lady Health Worker (LHW)",
        "color": "#ff69b4",
        "pin": "1111",
        "permissions": {
            "view_results",
            "download_report",
            "run_triage",
        },
    },
    ROLE_RADIOLOGIST: {
        "badge": "👨‍⚕️ Senior Radiologist",
        "color": "#a855f7",
        "pin": "9999",
        "permissions": {
            "view_results",
            "download_report",
            "run_triage",
            "override_assessment",
            "view_audit_log",
            "export_dicom",
            "manage_users",
        },
    },
}

# ── AuthManager class ────────────────────────────────────────────────────────
class AuthManager:
    """
    Lightweight stateless auth helper.

    Usage
    -----
    auth = AuthManager(current_role)
    auth.has_permission("override_assessment")  # → True / False
    auth.authenticate_pin("9999")               # → ROLE_RADIOLOGIST or None
    auth.get_active_profile()                   # → dict with badge, color, …
    """

    def __init__(self, role: str = ROLE_LHW):
        if role not in ROLE_CONFIGS:
            role = ROLE_LHW
        self._role = role

    # ── Properties ──────────────────────────────────────────────────────────
    @property
    def role(self) -> str:
        return self._role

    # ── Public API ───────────────────────────────────────────────────────────
    def get_active_profile(self) -> dict:
        """Return the full config dict for the current role."""
        return ROLE_CONFIGS[self._role]

    def has_permission(self, permission: str) -> bool:
        """Return True if the current role has the requested permission."""
        return permission in ROLE_CONFIGS[self._role]["permissions"]

    def authenticate_pin(self, pin: str):
        """
        Try to match *pin* against all role PINs.

        Returns
        -------
        str or None
            The matching role string if the PIN is valid, otherwise None.
        """
        pin = str(pin).strip()
        for role, cfg in ROLE_CONFIGS.items():
            if cfg.get("pin") and cfg["pin"] == pin:
                return role
        return None

    # ── Convenience helpers ──────────────────────────────────────────────────
    def is_radiologist(self) -> bool:
        return self._role == ROLE_RADIOLOGIST

    def is_lhw(self) -> bool:
        return self._role == ROLE_LHW

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuthManager(role={self._role!r})"
