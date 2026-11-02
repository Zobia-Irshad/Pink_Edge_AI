"""Overview / Command Center page."""
from __future__ import annotations

import streamlit as st

import GUI as core

from components.access import can_view_page
from components.metrics import track


def _session_kpis():
    total, unsynced = core.counts()
    alerts = st.session_state.get("sms_alerts") or []
    crit = sum(1 for a in alerts if a.get("is_critical"))
    return {
        "alerts": len(alerts),
        "critical": crit,
        "cached": total,
        "unsynced": unsynced,
        "latency": st.session_state.get("inference_latency", 0),
        "train": st.session_state.get("train_score", 0),
        "done": bool(st.session_state.get("inference_done")),
    }


def render_overview(selected_model: str):
    track("nav_overview")
    k = _session_kpis()
    st.markdown(
        """
    <div class="brand-bar">
      <div class="brand-mark">PE</div>
      <div>
        <div class="brand-title">Command Center</div>
        <div class="brand-sub">Production clinical workspace · build a case · triage · escalate · sync</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="k">Hub alerts</div><div class="v">{k['alerts']}</div><div class="s">{k['critical']} critical</div></div>
      <div class="kpi-card"><div class="k">Cached reports</div><div class="v">{k['cached']}</div><div class="s">{k['unsynced']} unsynced</div></div>
      <div class="kpi-card"><div class="k">Last latency</div><div class="v">{k['latency']}s</div><div class="s">{'Ready' if k['done'] else 'Awaiting triage'}</div></div>
      <div class="kpi-card"><div class="k">Training score</div><div class="v">{k['train']}</div><div class="s">LHW arena</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Workspace modules</div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div class="feature-grid">
      <div class="feature-tile"><div class="ft-icon">🧩</div><div class="ft-title">Case Builder</div><div class="ft-desc">Compose patient context, urgency, and BHU metadata before triage.</div></div>
      <div class="feature-tile"><div class="ft-icon">🩺</div><div class="ft-title">Clinical Triage</div><div class="ft-desc">Run offline-first models with bbox overlay and clinician confirm.</div></div>
      <div class="feature-tile"><div class="ft-icon">✨</div><div class="ft-title">AI Lab</div><div class="ft-desc">Explainable AI templates + guided demo (radiologist).</div></div>
      <div class="feature-tile"><div class="ft-icon">🏥</div><div class="ft-title">Hospital Hub</div><div class="ft-desc">GSM alert stream with agree / override / approve workflow.</div></div>
      <div class="feature-tile"><div class="ft-icon">📄</div><div class="ft-title">Reports</div><div class="ft-desc">Shared SQLite cache grid with PII-stripped patient refs.</div></div>
      <div class="feature-tile"><div class="ft-icon">☁️</div><div class="ft-title">Cloud Sync</div><div class="ft-desc">Simulated edge→cloud sync, voice library, session feedback.</div></div>
      <div class="feature-tile"><div class="ft-icon">⚙️</div><div class="ft-title">Edge Ops</div><div class="ft-desc">DICOM route, shield, multi-task, cascade, telemetry (radiologist).</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Open Case Builder", width="stretch", type="primary", key="ov_builder"):
        st.session_state.active_page = "builder"
        st.rerun()
    if c2.button(f"Run Triage · {selected_model.split('(')[0].strip()}", width="stretch", key="ov_triage"):
        st.session_state.active_page = "dashboard"
        st.rerun()
    if c3.button("Hospital Hub", width="stretch", key="ov_hub"):
        st.session_state.active_page = "hub"
        st.rerun()
    if c4.button("Reports Library", width="stretch", key="ov_reports"):
        st.session_state.active_page = "reports"
        st.rerun()
    if c5.button("Edge Ops", width="stretch", key="ov_ops", disabled=not can_view_page("ops")):
        st.session_state.active_page = "ops"
        st.rerun()

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown('<div class="section-label">Live activity</div>', unsafe_allow_html=True)
        logs = st.session_state.log_entries[-8:] or ["> System initialized. Awaiting commands..."]
        rows = "".join(f"<div class='activity-row'><span>{e}</span></div>" for e in reversed(logs))
        st.markdown(f"<div class='builder-shell'>{rows}</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-label">Edge node</div>', unsafe_allow_html=True)
        hw, net = st.session_state.hw_stats, st.session_state.net_stats
        case_bits = (
            f"BHU {st.session_state.get('case_bhu', '—')} · "
            f"{st.session_state.get('case_urgency', 'Routine')} · "
            f"{'case sheet ready' if st.session_state.get('case_built') else 'no case sheet yet'}"
        )
        st.markdown(
            f"""
        <div class="builder-shell">
          <h3>RK3588 edge</h3>
          <div class="activity-row"><span>NPU load</span><b>{hw['load']}</b></div>
          <div class="activity-row"><span>Power / temp</span><b>{hw['power']} · {hw['temp']}</b></div>
          <div class="activity-row"><span>GSM</span><b>{net['module']} · {net['signal']}</b></div>
          <div class="activity-row"><span>Case context</span><b style="font-size:0.75rem;">{case_bits}</b></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
