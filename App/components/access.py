"""Role-gated navigation helpers (auth_manager-backed)."""
from __future__ import annotations

from auth_manager import AuthManager, ROLE_LHW, ROLE_RADIOLOGIST

# Pages locked for LHW — require Senior Radiologist (PIN 9999)
RADIOLOGIST_ONLY_PAGES = frozenset({"lab", "ops"})


def current_auth() -> AuthManager:
    import streamlit as st

    return AuthManager(st.session_state.get("user_role", ROLE_LHW))


def can_view_page(page_key: str, auth=None) -> bool:
    auth = auth or current_auth()
    if page_key == "lab":
        return auth.has_permission("view_ai_lab")
    if page_key == "ops":
        return auth.has_permission("view_edge_ops")
    return True


def visible_page_keys(all_keys, auth=None):
    auth = auth or current_auth()
    return [k for k in all_keys if can_view_page(k, auth)]


def enforce_page_access(page_key: str) -> str:
    """If current role cannot view page, bounce to overview."""
    import streamlit as st

    if can_view_page(page_key):
        return page_key
    st.session_state.active_page = "overview"
    st.warning("🔒 AI Lab / Edge Ops require Senior Radiologist PIN (9999).")
    return "overview"
