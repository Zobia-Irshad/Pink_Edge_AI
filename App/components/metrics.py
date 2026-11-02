"""Local offline usage metrics — SQLite only (no Mixpanel/PostHog)."""
from __future__ import annotations

import GUI as core


def track(event_name: str, detail: str = "", page: str = ""):
    try:
        import streamlit as st

        role = str(st.session_state.get("user_role", ""))
        page = page or str(st.session_state.get("active_page", ""))
    except Exception:
        role, page = "", page
    try:
        core.log_usage_event(event_name, detail=detail, role=role, page=page)
    except Exception:
        pass
