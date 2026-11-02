"""Case Builder page — session enrichment only; triage still uses GUI.run_triage()."""
from __future__ import annotations

import time

import streamlit as st

import GUI as core

from components.metrics import track


def render_case_builder(selected_model: str, uploaded, run_triage_fn):
    track("nav_case_builder")
    st.markdown(
        """
    <div class="builder-shell">
      <h3>Case Builder</h3>
      <p style="margin:0;color:#9aabbf;font-size:0.88rem;font-family:'Sora',sans-serif;">
        Assemble patient context before running triage. Offline models stay unchanged — this sheet only enriches the session.
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.pat_age = st.number_input(
            "Patient age", min_value=1, max_value=100, value=int(st.session_state.pat_age), step=1
        )
        st.session_state.case_gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
            index=["Female", "Male", "Other"].index(st.session_state.get("case_gender", "Female")),
        )
        st.session_state.case_bhu = st.text_input(
            "BHU / facility", value=st.session_state.get("case_bhu", "BHU-14 Faisalabad")
        )
    with c2:
        modality = st.selectbox(
            "Modality for this case",
            core.MODELS,
            index=core.MODELS.index(selected_model) if selected_model in core.MODELS else 0,
            key="builder_modality",
        )
        st.session_state.case_urgency = st.select_slider(
            "Urgency",
            options=["Routine", "Soon", "Urgent", "Critical"],
            value=st.session_state.get("case_urgency", "Routine"),
        )
        st.session_state.network_mode = st.radio(
            "Network for run",
            ["Fully Offline", "GSM Failover"],
            index=0 if "Offline" in st.session_state.network_mode else 1,
            horizontal=True,
            key="builder_net",
        )

    st.session_state.case_notes = st.text_area(
        "Clinical notes (optional)",
        value=st.session_state.get("case_notes", ""),
        placeholder="Symptoms, referral reason, image quality notes…",
        height=100,
    )

    st.markdown('<div class="section-label">Case preview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
    <div class="builder-shell">
      <div class="activity-row"><span>Privacy hash</span><b style="font-family:ui-monospace,monospace;font-size:0.78rem;">{st.session_state.privacy_hash}</b></div>
      <div class="activity-row"><span>Age / gender</span><b>{st.session_state.pat_age} Y · {st.session_state.case_gender}</b></div>
      <div class="activity-row"><span>Facility</span><b>{st.session_state.case_bhu}</b></div>
      <div class="activity-row"><span>Modality / urgency</span><b>{modality.split('(')[0].strip()} · {st.session_state.case_urgency}</b></div>
      <div class="activity-row"><span>Notes</span><b style="font-size:0.78rem;">{(st.session_state.case_notes or '—')[:120]}</b></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    if b1.button("Save case sheet", width="stretch", key="builder_save"):
        st.session_state.case_built = True
        st.session_state.last_model = modality
        st.session_state.log_entries.append(
            f"[{time.strftime('%H:%M:%S')}] [BUILDER] Case sheet saved · {st.session_state.case_bhu} · {st.session_state.case_urgency}"
        )
        track("case_sheet_saved", detail=st.session_state.case_bhu)
        st.toast("Case sheet saved", icon="🧩")
    if b2.button("Save & open triage", width="stretch", type="primary", key="builder_to_dash"):
        st.session_state.case_built = True
        st.session_state.last_model = modality
        st.session_state.active_page = "dashboard"
        st.toast("Case ready — run triage", icon="🩺")
        st.rerun()
    if b3.button("Build + Run Triage now", width="stretch", type="primary", key="builder_run"):
        st.session_state.case_built = True
        st.session_state.last_model = modality
        st.session_state.active_page = "dashboard"
        track("case_builder_run_triage", detail=modality)
        run_triage_fn(modality, uploaded)
