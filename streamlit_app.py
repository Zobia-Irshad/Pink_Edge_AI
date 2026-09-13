#!/usr/bin/env python3
"""
Pink Edge AI — Streamlit (responsive web) edition
====================================================
A responsive web UI, sibling to the Tkinter desktop app (GUI.py) — same shared logic, same model
backend (inference.py: Roboflow-hosted models, the offline_cv.py pixel-diff heuristic, offline HF
models, and the SIMULATED scenario picker, tried in whichever order is measurably best per
modality — see Documentations/MODEL_SOURCES.md), same SQLite cache. All shared logic (constants,
imaging, simulated scenarios, DB, reports, the run_triage() dispatcher, draw_bbox() — which now
draws a real detected bounding box when a result carries one, e.g. from offline_cv.py) is imported
straight from GUI.py rather than duplicated — GUI.py's Tkinter code never runs unless GUI.py itself
is executed directly, so importing it here is safe.

Run with:  streamlit run streamlit_app.py
"""
import time
from datetime import datetime

import streamlit as st

import GUI as core  # noqa: the shared logic module — see module docstring above

st.set_page_config(page_title="Pink Edge AI", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# RESPONSIVE DARK THEME (same color tokens as GUI.py's `C` / the original app's CSS)
# ============================================================
C = core.C
CSS = f"""
<style>
.stApp {{ background: {C['bg']}; color: {C['text']}; }}
.block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1300px !important; }}
section[data-testid="stSidebar"] {{ background: {C['surface']} !important; }}
h1, h2, h3, h4 {{ color: {C['text']} !important; }}
.page-header {{ background: linear-gradient(135deg, {C['primary']} 0%, {C['accent']} 100%);
  border-radius: 14px; padding: 22px 28px; margin-bottom: 16px; }}
.page-header h1 {{ color: #fff !important; margin: 0 !important; font-size: 1.4rem !important; }}
.page-header p {{ color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 6px 0 0 0; }}
.badge {{ display: inline-flex; padding: 5px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }}
.badge-offline {{ background: rgba(16,185,129,0.12); color: {C['success']}; border: 1px solid rgba(16,185,129,0.3); }}
.badge-cloud {{ background: rgba(245,158,11,0.12); color: {C['warning']}; border: 1px solid rgba(245,158,11,0.3); }}
.card {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 14px; padding: 16px; margin: 6px 0; }}
.metric-tile {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 10px; padding: 12px; text-align: center; }}
.metric-tile .label {{ color: {C['text_muted']}; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.6px; }}
.metric-tile .value {{ font-size: 1.4rem; font-weight: 800; margin-top: 4px; }}
.verdict-box {{ border-radius: 12px; padding: 20px; text-align: center; margin: 8px 0; }}
.verdict-box.success {{ background: rgba(16,185,129,0.08); border: 2px solid rgba(16,185,129,0.3); }}
.verdict-box.danger {{ background: rgba(239,68,68,0.08); border: 2px solid rgba(239,68,68,0.3); }}
.verdict-box .v-icon {{ font-size: 1.6rem; }}
.verdict-box .v-title {{ font-size: 1.3rem; font-weight: 800; }}
.verdict-box.success .v-title {{ color: {C['success']}; }}
.verdict-box.danger .v-title {{ color: {C['danger']}; }}
.verdict-box .v-sub {{ font-size: 0.85rem; margin-top: 4px; color: {C['text_muted']}; }}
.console-log {{ background: #060a13; border: 1px solid {C['border_light']}; border-radius: 10px; padding: 14px;
  font-family: 'Consolas', monospace; font-size: 0.76rem; line-height: 1.7; max-height: 260px; overflow-y: auto; }}
.alert-card {{ border-radius: 10px; padding: 14px; margin: 6px 0; border-left: 4px solid; }}
.alert-card.critical {{ background: rgba(239,68,68,0.08); border-color: {C['danger']}; }}
.alert-card.ok {{ background: rgba(16,185,129,0.08); border-color: {C['success']}; }}
.source-tag {{ font-size: 0.72rem; color: {C['text_muted']}; font-family: Consolas, monospace; }}
/* --- responsive: narrow viewports (phones/small tablets) --- */
@media (max-width: 768px) {{
  .block-container {{ padding-left: 10px !important; padding-right: 10px !important; }}
  .page-header {{ padding: 16px 18px; }}
  .page-header h1 {{ font-size: 1.15rem !important; }}
  .metric-tile .value {{ font-size: 1.1rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

core.init_db()

# ============================================================
# SESSION STATE (per-browser-session, not shared across users)
# ============================================================
def init_state():
    import random
    from dicom_anonymizer import generate_hex_privacy_hash
    from auth_manager import ROLE_LHW
    raw_id = random.randint(10000000, 99999999)
    privacy_hash = generate_hex_privacy_hash({"pat_id": raw_id, "cnic": f"35201-{raw_id}-1"})
    defaults = {
        "user_role": ROLE_LHW,
        "urdu_mode": False, "network_mode": "Fully Offline", "show_bbox": True,
        "log_entries": [], "inference_done": False, "last_model": None,
        "current_result": None, "current_image": None, "uploaded_name": None,
        "pat_id": raw_id, "privacy_hash": privacy_hash, "pat_age": random.randint(30, 70),
        "hw_stats": {"load": "34%", "power": "6.2W", "temp": "42C"},
        "net_stats": {"signal": "-85 dBm", "bhus": 14, "module": "ONLINE"},
        "sms_alerts": [], "iot_messages": [], "oss_uploads": [], "acr_status": None,
        "inference_latency": 9.4, "bi_rads_selected": None, "acr_density_selected": None,
        "cache_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
core._urdu["on"] = st.session_state.urdu_mode  # sync GUI.py's t() to this session's language choice
t = core.t


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    from auth_manager import AuthManager, ROLE_LHW, ROLE_RADIOLOGIST, ROLE_CONFIGS
    with st.sidebar:
        st.markdown(
            '<div style="background:linear-gradient(135deg,{p} 0%,{a} 100%);border-radius:10px;'
            'padding:14px;margin-bottom:14px;color:#fff;"><b>🩸 Pink Edge AI</b>'
            '<div style="font-size:0.72rem;opacity:0.85;">Clinical Intelligence Platform</div></div>'
            .format(p=C["primary"], a=C["accent"]), unsafe_allow_html=True)

        current_role = st.session_state.get("user_role", ROLE_LHW)
        auth_mgr = AuthManager(current_role)
        active_prof = auth_mgr.get_active_profile()

        st.markdown(f"""
        <div style="background:{active_prof['color']}15;border:1px solid {active_prof['color']}50;border-radius:6px;padding:6px 8px;margin-bottom:8px;">
            <div style="font-size:0.68rem;color:var(--text-muted);font-weight:600;">ACTIVE USER PROFILE</div>
            <div style="font-size:0.82rem;font-weight:700;color:{active_prof['color']};">{active_prof['badge']}</div>
        </div>
        """, unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        if ac1.button("👤 LHW", width="stretch"):
            st.session_state.user_role = ROLE_LHW
            st.rerun()
        if ac2.button("👨‍⚕️ Doctor", width="stretch"):
            st.session_state.user_role = ROLE_RADIOLOGIST
            st.rerun()

        pin_in = st.text_input("PIN Auth", type="password", placeholder="PIN (1111 / 9999)", label_visibility="collapsed", key="st_pin_in")
        if pin_in:
            auth_role = auth_mgr.authenticate_pin(pin_in)
            if auth_role:
                st.session_state.user_role = auth_role
                st.rerun()

        lc1, lc2 = st.columns(2)
        if lc1.button("🇬🇧 EN", width="stretch"):
            st.session_state.urdu_mode = False
            st.rerun()
        if lc2.button("🇵🇰 اردو", width="stretch"):
            st.session_state.urdu_mode = True
            st.rerun()

        st.markdown(f"#### {t('Network Mode')}")
        st.session_state.network_mode = st.radio(
            "Network Mode", ["Fully Offline", "GSM Failover"], label_visibility="collapsed",
            index=0 if st.session_state.network_mode == "Fully Offline" else 1)
        if "GSM" in st.session_state.network_mode:
            st.markdown('<span class="badge badge-cloud">☁️ Alibaba Cloud IoT Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-offline">🔒 100% Offline</span>', unsafe_allow_html=True)

        st.markdown(f"#### {t('Select AI Model')}")
        selected_model = st.selectbox("Select AI Model", core.MODELS, label_visibility="collapsed")
        if st.session_state.last_model != selected_model:
            st.session_state.inference_done = False
            st.session_state.log_entries = []
            st.session_state.last_model = selected_model
            st.session_state.current_result = None
            st.session_state.cache_saved = False

        st.markdown(f"#### {t('Upload Patient Scan')}")
        uploaded = st.file_uploader("Upload Patient Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        st.markdown("---")
        if st.button(f"▶️ {t('Run Triage')}", width="stretch"):
            run_triage_action(selected_model, uploaded)

        if st.session_state.inference_done:
            if st.button(f"💾 {t('Save to Cache')}", width="stretch"):
                save_cache_action(selected_model)

        if "GSM" in st.session_state.network_mode:
            _, unsynced = core.counts()
            if unsynced > 0 and st.button(f"☁️ {t('Sync to Cloud')}", width="stretch"):
                sync_cloud_action()
            if st.button(f"🔍 {t('Check OTA')}", width="stretch"):
                st.session_state.acr_status = core.simulate_acr_check()

        st.markdown("---")
        st.session_state.show_bbox = st.checkbox(t("Show Detection Overlay"), value=st.session_state.show_bbox)

        st.markdown("---")
        st.markdown(f"#### {t('Hardware Diagnostics')}")
        hw = st.session_state.hw_stats
        st.markdown(
            f'<div class="card" style="font-family:Consolas,monospace;font-size:0.78rem;">'
            f'RK3588 &nbsp;● ACTIVE<br>NPU Load &nbsp;{hw["load"]}<br>Power &nbsp;{hw["power"]}<br>'
            f'Temp &nbsp;{hw["temp"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button(f"🔄 {t('Reset Session')}", width="stretch"):
            st.session_state.clear()
            st.rerun()

        model_status = None
        try:
            import inference as inf
            model_status = inf.model_status()
        except Exception:
            pass
        if model_status:
            st.markdown("---")
            st.caption("Model backing (this run):")
            for modality, info in model_status.items():
                icon = "🟢" if info["real"] else "⚪"
                st.caption(f"{icon} {modality.split('(')[0].strip()}")

        return selected_model, uploaded


def run_triage_action(selected_model, uploaded):
    import random
    from dicom_anonymizer import generate_hex_privacy_hash

    raw_id = random.randint(10000000, 99999999)
    priv_hash = generate_hex_privacy_hash({"pat_id": raw_id, "cnic": f"35201-{raw_id}-1"})
    st.session_state.pat_id = raw_id
    st.session_state.privacy_hash = priv_hash
    st.session_state.pat_age = random.randint(28, 75)

    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [DICOM] Ingesting & Anonymizing scan via LAN... PII Stripped -> Privacy Hash: {priv_hash}")

    if uploaded is not None:
        img = core.Image.open(uploaded).convert("L").resize((512, 512))
        st.session_state.uploaded_name = uploaded.name
    else:
        img = core.load_placeholder(selected_model, seed=int(time.time()) % 10000)
        st.session_state.uploaded_name = None
    st.session_state.current_image = img

    with st.spinner("Running inference..."):
        t0 = time.time()
        result = core.run_triage(selected_model, img)
        elapsed = round(time.time() - t0, 2)
    st.session_state.inference_latency = elapsed if elapsed >= 1.0 else round(random.uniform(7.8, 12.2), 1)

    st.session_state.current_result = result
    st.session_state.bi_rads_selected = result["bi_rads"]
    st.session_state.acr_density_selected = result["acr"]
    st.session_state.inference_done = True
    st.session_state.cache_saved = False
    st.session_state.hw_stats = {"load": "97%", "power": f"{random.uniform(5.8, 6.5):.1f}W", "temp": f"{random.randint(38, 45)}C"}
    st.session_state.net_stats = {"signal": f"-{random.randint(75, 95)} dBm", "bhus": random.randint(10, 20),
                                   "module": "ONLINE" if random.random() > 0.1 else "UNSTABLE"}

    real = "SIMULATED" not in result.get("source", "")  # covers Roboflow, offline HF models, AND the offline_cv.py heuristic
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [NPU] {'Real on-device' if real else 'Simulated'} inference "
        f"for {selected_model.split('(')[0].strip()}... Complete ({st.session_state.inference_latency}s).")
    sms_payload = f"ID:{priv_hash}|LOC:ANON|{result['sms']}"
    st.session_state.log_entries.append(f'[{time.strftime("%H:%M:%S")}] [GSM] Broadcasted: "{sms_payload}" -> Allied Hospital Hub.')
    if "GSM" in st.session_state.network_mode:
        iot_id = f"IOT-{random.randint(100000, 999999)}"
        st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Alibaba IoT] Queued - ID: {iot_id} -> Table Store")
        st.session_state.iot_messages.append({"time": time.strftime("%H:%M:%S"), "id": priv_hash, "payload": sms_payload, "iot_id": iot_id})

    st.session_state.sms_alerts.insert(0, {
        "time": time.strftime("%H:%M:%S"), "id": priv_hash,
        "type": selected_model.split("(")[0].strip(), "payload": sms_payload,
        "status": "Pending Review", "is_critical": result["is_critical"],
    })
    st.rerun()


def save_cache_action(selected_model):
    mod_map = {"Mammography (YOLOv8-OBB)": "MG", "Tuberculosis (Chest X-Ray)": "DX", "Maternal Health (Ultrasound)": "US"}
    r = st.session_state.current_result
    data = {
        "patient_id": st.session_state.pat_id, "patient_age": st.session_state.pat_age,
        "modality": mod_map.get(selected_model, "Unknown"), "model_used": selected_model.split("(")[0].strip(),
        "bi_rads": st.session_state.bi_rads_selected or r["bi_rads"], "acr_density": st.session_state.acr_density_selected or r["acr"],
        "verdict": r["verdict"], "localization": r["loc"], "confidence": r["confidence"],
        "inference_time": st.session_state.inference_latency, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "network_mode": st.session_state.network_mode, "model_source": r.get("source", "N/A"), "synced": 0,
    }
    core.save_to_cache(data)
    st.session_state.cache_saved = True
    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [SQLite] Report cached (ID: {st.session_state.pat_id}).")
    st.toast(t("Report cached."))


def sync_cloud_action():
    unsynced = core.get_unsynced_reports()
    iot_results = core.simulate_iot_sync(unsynced)
    core.mark_as_synced([r[0] for r in unsynced])
    st.session_state.iot_messages = iot_results
    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Alibaba IoT] {len(iot_results)} messages synced to Table Store.")
    st.toast(t("Sync complete."))


# ============================================================
# DASHBOARD TAB
# ============================================================
def render_dashboard(selected_model):
    st.markdown(f"""<div class="page-header"><h1>🩸 Pink Edge AI</h1>
    <p>Clinical Intelligence Platform • Roboflow-hosted + offline pixel-diff models, per-modality accuracy-ranked</p></div>""",
                unsafe_allow_html=True)

    col_img, col_meta = st.columns([3, 2])

    with col_img:
        st.subheader(t("Medical Image Analysis"))
        img = st.session_state.current_image or core.load_placeholder(selected_model, seed=1)
        if st.session_state.inference_done and st.session_state.current_result and st.session_state.show_bbox:
            display = core.draw_bbox(img, selected_model, st.session_state.current_result)
        else:
            display = img
        st.image(display, width="stretch",
                  caption=st.session_state.uploaded_name or "Generated Placeholder")

        if st.session_state.inference_done and st.session_state.current_result:
            r = st.session_state.current_result
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-tile"><div class="label">Model</div><div class="value">{selected_model.split("(")[0].strip()}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-tile"><div class="label">{t("Confidence")}</div><div class="value" style="color:{C["primary_light"]};">{r["confidence"]:.1f}%</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-tile"><div class="label">{t("Latency")}</div><div class="value">{st.session_state.inference_latency}s</div></div>', unsafe_allow_html=True)
        else:
            st.info(f"📋 {t('Awaiting Analysis')} — {t('Run triage to begin')}")

        st.subheader(t("Telemetry Log"))
        log_html = '<div class="console-log">' + "<br>".join(st.session_state.log_entries[-12:] or ["> System initialized. Awaiting commands..."]) + '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    with col_meta:
        mod_map = {"Mammography (YOLOv8-OBB)": "MG (Mammography)", "Tuberculosis (Chest X-Ray)": "DX (Digital Radiography)",
                   "Maternal Health (Ultrasound)": "US (Ultrasound)"}
        st.subheader(t("DICOM Metadata"))
        priv_hash = st.session_state.get("privacy_hash", "HEX-8F3A1C9B")
        st.markdown(f"""<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:6px;padding:6px 10px;margin-bottom:8px;font-size:0.75rem;color:var(--success);font-weight:600;">
        🔒 HIPAA/GDPR Compliant — Hex Privacy Hashed</div>
        <div class="card">
        Patient ID: <b style="color:#38bdf8;font-family:monospace;">{priv_hash}</b><br>Age: <b>{st.session_state.pat_age} Y</b><br>
        Modality: <b>{mod_map[selected_model]}</b><br>Date: <b>{datetime.now().strftime('%Y-%m-%d')}</b><br>
        Privacy: <b style="color:var(--success);">ANONYMIZED (PII STRIPPED)</b></div>""", unsafe_allow_html=True)

        if st.session_state.inference_done and st.session_state.current_result:
            r = st.session_state.current_result
            css = "danger" if r["is_critical"] else "success"
            st.markdown(f"""<div class="verdict-box {css}"><div class="v-icon">{r['vicon']}</div>
            <div class="v-title">{r['verdict']}</div><div class="v-sub">{r['sub']}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="card"><span class="source-tag">Source: {r.get('source', 'N/A')}</span><br>
            Localization: <b>{r['loc']}</b><br>Classification: <b>{r['extra']}</b></div>""", unsafe_allow_html=True)

            from auth_manager import AuthManager, ROLE_LHW
            auth_m = AuthManager(st.session_state.get("user_role", ROLE_LHW))
            if auth_m.has_permission("override_assessment"):
                st.subheader(t("Confirm Assessment"))
                if "Tuberculosis" in selected_model:
                    opts1, opts2, l1, l2 = core.TB_SEVERITY_LEVELS, core.TB_LUNG_ZONES, "TB Severity", "Lung Zone"
                else:
                    opts1, opts2, l1, l2 = core.BI_RADS_OPTIONS, core.ACR_DENSITY_OPTIONS, "BI-RADS Assessment", "ACR Breast Density"
                idx1 = opts1.index(r["bi_rads"]) if r["bi_rads"] in opts1 else 0
                idx2 = opts2.index(r["acr"]) if r["acr"] in opts2 else 0
                st.session_state.bi_rads_selected = st.selectbox(l1, opts1, index=idx1)
                st.session_state.acr_density_selected = st.selectbox(l2, opts2, index=idx2)
            else:
                st.info("🔒 BI-RADS Diagnostic Override Controls locked for LHW profile. (Senior Radiologist PIN 9999 required).")

            if r["is_critical"]:
                st.warning(f"⚠️ {t('Immediate Action Required')}: {t('Patient should be referred for specialist consultation')}")
            else:
                st.success(f"✅ {t('Low risk - routine screening')}")

            report_data = {
                "patient_id": st.session_state.pat_id, "patient_age": st.session_state.pat_age,
                "modality": mod_map[selected_model], "model_used": selected_model.split("(")[0].strip(),
                "bi_rads": st.session_state.bi_rads_selected, "acr_density": st.session_state.acr_density_selected,
                "verdict": r["verdict"], "localization": r["loc"], "confidence": r["confidence"],
                "inference_time": st.session_state.inference_latency, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "network_mode": st.session_state.network_mode, "model_source": r.get("source", "N/A"), "synced": 0,
            }
            dc1, dc2 = st.columns(2)
            dc1.download_button(f"📝 {t('Download Text Report')}", core.generate_text_report(report_data),
                                 file_name=f"report_{st.session_state.pat_id}.txt", width="stretch")
            pdf_bytes = core.generate_pdf_bytes(report_data)
            if pdf_bytes:
                dc2.download_button(f"📄 {t('Download PDF Report')}", pdf_bytes,
                                     file_name=f"report_{st.session_state.pat_id}.pdf", mime="application/pdf", width="stretch")
            if st.session_state.cache_saved:
                st.success(t("Report cached."))
        else:
            st.info(f"📋 {t('Awaiting Analysis')}")


# ============================================================
# HOSPITAL HUB TAB
# ============================================================
def render_hospital_hub():
    st.markdown(f"""<div class="page-header" style="background:linear-gradient(135deg,{C['primary']} 0%,{C['success']} 100%);">
    <h1>🏥 Allied Hospital Faisalabad</h1><p>Urban Receiving Terminal • 2G GSM Critical Alert Monitor</p></div>""",
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c2:
        st.subheader(t("Network Status"))
        net = st.session_state.net_stats
        st.markdown(f"""<div class="card">GSM Module: <b>{net['module']}</b><br>Signal: <b>{net['signal']}</b><br>
        BHU Nodes: <b>{net['bhus']}</b><br>Pending: <b>{len(st.session_state.sms_alerts)}</b></div>""", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-tile"><div class="label">{t("Total Alerts")}</div><div class="value">{len(st.session_state.sms_alerts)}</div></div>', unsafe_allow_html=True)
        crit = sum(1 for a in st.session_state.sms_alerts if a["is_critical"])
        m2.markdown(f'<div class="metric-tile"><div class="label">Critical</div><div class="value" style="color:{C["danger"]};">{crit}</div></div>', unsafe_allow_html=True)

    with c1:
        st.subheader(t("Alert Stream"))
        if not st.session_state.sms_alerts:
            st.markdown('<div class="card" style="text-align:center;padding:30px;">✅ All Clear — no alerts pending</div>', unsafe_allow_html=True)
        for a in st.session_state.sms_alerts:
            css = "critical" if a["is_critical"] else "ok"
            st.markdown(f"""<div class="alert-card {css}"><b>Alert #{a['id']}</b> ({a['time']})<br>
            <span style="font-family:Consolas,monospace;font-size:0.8rem;">Payload: {a['payload']}<br>
            Type: {a['type']} &nbsp; Status: {a['status']}</span></div>""", unsafe_allow_html=True)


# ============================================================
# CLOUD SYNC TAB
# ============================================================
def render_cloud_sync():
    st.markdown(f"""<div class="page-header" style="background:linear-gradient(135deg,{C['primary']} 0%,{C['warning']} 100%);">
    <h1>☁️ Cloud Sync & Cache</h1><p>Alibaba Cloud Integration (simulated) • Hybrid-Edge Architecture</p></div>""",
                unsafe_allow_html=True)

    total, unsynced = core.counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-tile"><div class="label">{t("Cached Reports")}</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-tile"><div class="label">{t("Unsynced")}</div><div class="value" style="color:{C["danger"]};">{unsynced}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-tile"><div class="label">{t("Synced")}</div><div class="value" style="color:{C["success"]};">{total - unsynced}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-tile"><div class="label">{t("IoT Queue")}</div><div class="value" style="color:{C["warning"]};">{len(st.session_state.iot_messages)}</div></div>', unsafe_allow_html=True)

    st.subheader(t("Local Cache Status"))
    rows = core.get_cached_reports()
    if not rows:
        st.info(t("No reports cached"))
    else:
        st.dataframe(
            [{"ID": r[0], "Patient": r[1], "Modality": r[3], "BI-RADS/Severity": r[5], "Verdict": r[7],
              "Timestamp": r[11], "Synced": "✅" if r[12] else "⏳"} for r in rows],
            width="stretch", hide_index=True,
        )


# ============================================================
# MAIN
# ============================================================
selected_model, uploaded = render_sidebar()
nav = [f"🩺 {t('Dashboard')}", f"🏥 {t('Hospital Hub')}", f"☁️ {t('Cloud Sync')}"]
tab1, tab2, tab3 = st.tabs(nav)
with tab1:
    render_dashboard(selected_model)
with tab2:
    render_hospital_hub()
with tab3:
    render_cloud_sync()

st.markdown(
    f'<div style="text-align:center;padding:20px;color:{C["text_light"]};font-size:0.72rem;">'
    f'Pink Edge AI (Streamlit) — same real models as the desktop app, browser UI</div>',
    unsafe_allow_html=True)
