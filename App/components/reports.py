"""Shared SQLite reports grid — PII-stripped patient refs via dicom_anonymizer."""
from __future__ import annotations

import streamlit as st

import GUI as core
from dicom_anonymizer import anonymize_dicom_metadata, generate_hex_privacy_hash

from components.metrics import track


def _privacy_ref(patient_id) -> str:
    """Display-safe patient reference (hash), never raw identifiers."""
    raw = str(patient_id or "")
    if not raw or raw.upper().startswith("PE-") or len(raw) >= 32:
        # Already looks like a hash / privacy token
        return raw[:16] + "…" if len(raw) > 18 else raw
    return generate_hex_privacy_hash({"PatientID": raw})[:16] + "…"


def render_reports_center():
    track("nav_reports")
    total, unsynced = core.counts()
    st.markdown(
        f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="k">Total reports</div><div class="v">{total}</div><div class="s">SQLite cache</div></div>
      <div class="kpi-card"><div class="k">Unsynced</div><div class="v">{unsynced}</div><div class="s">Pending cloud</div></div>
      <div class="kpi-card"><div class="k">Synced</div><div class="v">{max(total-unsynced,0)}</div><div class="s">Marked synced</div></div>
      <div class="kpi-card"><div class="k">Session alerts</div><div class="v">{len(st.session_state.sms_alerts)}</div><div class="s">Hospital Hub</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.caption("Patient column shows privacy hashes only (dicom_anonymizer). Shared `pink_edge_cache.db` with desktop UI.")

    rows = core.get_cached_reports()
    if not rows:
        st.info("No reports cached yet — run triage and Save to Cache.")
        if st.button("Go to Clinical Dashboard", type="primary"):
            st.session_state.active_page = "dashboard"
            st.rerun()
        return

    f1, f2, f3 = st.columns([2, 1, 1])
    modalities = sorted({r[3] for r in rows if r[3]})
    pick_mod = f1.multiselect("Filter modality", modalities, default=modalities)
    sync_filter = f2.selectbox("Sync status", ["All", "Unsynced only", "Synced only"])
    q = f3.text_input("Search hash / verdict", placeholder="Type to filter…")

    filtered = []
    for r in rows:
        if pick_mod and r[3] not in pick_mod:
            continue
        synced = bool(r[12])
        if sync_filter == "Unsynced only" and synced:
            continue
        if sync_filter == "Synced only" and not synced:
            continue
        pref = _privacy_ref(r[1])
        blob = f"{pref} {r[7]} {r[5]} {r[3]}".lower()
        if q and q.strip().lower() not in blob:
            continue
        filtered.append(r)

    grid = [
        {
            "ID": r[0],
            "Patient (privacy hash)": _privacy_ref(r[1]),
            "Age": r[2],
            "Modality": r[3],
            "Model": r[4],
            "Assessment": r[5],
            "Verdict": r[7],
            "Confidence": r[9],
            "Latency": r[10],
            "Time": r[11],
            "Synced": "✅" if r[12] else "⏳",
        }
        for r in filtered
    ]
    st.dataframe(grid, width="stretch", hide_index=True)

    if not filtered:
        st.warning("No rows match the current filters.")
        return

    pick_id = st.selectbox("Open report", [r[0] for r in filtered], format_func=lambda i: f"#{i}")
    row = next(r for r in filtered if r[0] == pick_id)

    # Strip any residual PII tags if report_json embeds DICOM-like fields
    safe_patient = _privacy_ref(row[1])
    meta_preview = anonymize_dicom_metadata(
        {"PatientID": str(row[1]), "PatientName": "REDACTED", "Modality": row[3] or ""}
    )

    report_data = {
        "patient_id": safe_patient,
        "patient_age": row[2],
        "modality": row[3],
        "model_used": row[4],
        "bi_rads": row[5],
        "acr_density": row[6],
        "verdict": row[7],
        "localization": row[8],
        "confidence": row[9],
        "inference_time": row[10],
        "timestamp": row[11],
        "network_mode": st.session_state.network_mode,
        "model_source": "cached_report",
        "synced": row[12],
        "image_quality": "From cache",
        "recommendation": "See original triage notes.",
        "referral_priority": "Review",
    }
    st.markdown(
        f"""
        <div class="builder-shell">
          <h3>Report #{row[0]}</h3>
          <div class="activity-row"><span>Patient ref</span><b style="font-family:ui-monospace,monospace;font-size:0.78rem;">{safe_patient}</b></div>
          <div class="activity-row"><span>Verdict</span><b>{row[7]}</b></div>
          <div class="activity-row"><span>Assessment</span><b>{row[5]}</b></div>
          <div class="activity-row"><span>Confidence / latency</span><b>{row[9]}% · {row[10]}s</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("PII strip preview (dicom_anonymizer)"):
        st.json(meta_preview)

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download text",
        core.generate_text_report(report_data),
        file_name=f"report_{safe_patient.replace('…','')[:12]}.txt",
        width="stretch",
    )
    pdf = core.generate_pdf_bytes(report_data)
    if pdf:
        d2.download_button(
            "Download PDF",
            pdf,
            file_name=f"report_{safe_patient.replace('…','')[:12]}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    if d3.button("Open Cloud Sync", width="stretch"):
        st.session_state.active_page = "cloud"
        st.rerun()
