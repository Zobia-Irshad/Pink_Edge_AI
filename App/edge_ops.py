"""
Client-requested Edge Ops modules for Pink Edge AI (demo / FYP surface).

These are product workflows wired to session state. Hospital PACS/RIS hooks are
simulated for local offline demos — the UX and control flow match the brief.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st


def _ensure_ops_state():
    defaults = {
        "dicom_listener_on": False,
        "dicom_inbox": [],
        "dicom_prefetch": [],
        "shield_last": None,
        "multitask_last": None,
        "cascade_jobs": [],
        "telemetry_quarantine": [],
        "telemetry_uploads": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _card(title: str, body: str):
    st.markdown(
        f'<div class="sidebar-section" style="margin-bottom:14px;">'
        f'<div class="sidebar-section-title">{title}</div>'
        f'<div style="font-size:0.92rem;line-height:1.45;color:#9aabbf;">{body}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_dicom_prefetch(speak=None):
    _ensure_ops_state()
    _card(
        "1 · Automated DICOM Route & Pre-Fetching",
        "Background receiver listens on the hospital LAN. Registration alerts trigger "
        "historical archive pre-cache so priors are ready before the doctor opens the study.",
    )
    c1, c2 = st.columns(2)
    with c1:
        on = st.toggle("DICOM LAN listener", value=st.session_state.dicom_listener_on, key="ops_dicom_toggle")
        st.session_state.dicom_listener_on = on
        st.caption("Simulated certified SCP receiver on local network.")
    with c2:
        if st.button("Simulate modality Save → route", width="stretch", type="primary", key="ops_dicom_route"):
            if not st.session_state.dicom_listener_on:
                st.warning("Turn the LAN listener on first.")
            else:
                study = {
                    "study_uid": f"1.2.840.{random.randint(100000,999999)}",
                    "modality": random.choice(["CR", "DX", "MG", "US"]),
                    "from": random.choice(["XR-Room-A", "US-Bed-2", "MG-Unit-1"]),
                    "received_at": datetime.now().strftime("%H:%M:%S"),
                    "status": "Received → Edge store",
                }
                st.session_state.dicom_inbox.insert(0, study)
                st.session_state.log_entries.append(
                    f"[{time.strftime('%H:%M:%S')}] [DICOM-SCP] Study {study['study_uid']} routed from {study['from']}"
                )
                if speak:
                    speak(f"New {study['modality']} study received from {study['from']} and routed to edge storage.")
                st.toast("Study routed to edge receiver", icon="📡")

    if st.button("Front-desk registration → pre-fetch priors", width="stretch", key="ops_prefetch"):
        pid = st.session_state.get("pat_id", random.randint(10000000, 99999999))
        priors = [
            {
                "patient_ref": f"PE-{pid}",
                "study": f"Prior {m} ({y})",
                "archived": y,
                "cached": True,
                "bytes_mb": round(random.uniform(8, 42), 1),
            }
            for m, y in [("CXR", 2022), ("MG", 2023), ("US", 2024)]
        ]
        st.session_state.dicom_prefetch = priors
        st.session_state.log_entries.append(
            f"[{time.strftime('%H:%M:%S')}] [PREFETCH] {len(priors)} historical studies cached for PE-{pid}"
        )
        if speak:
            speak(f"Pre-fetched {len(priors)} historical studies for side-by-side comparison.")
        st.success("Priors cached on encrypted edge storage.")

    st.subheader("Inbound studies")
    if st.session_state.dicom_inbox:
        st.dataframe(st.session_state.dicom_inbox, width="stretch", hide_index=True)
    else:
        st.info("No routed studies yet — enable listener and simulate a modality Save.")

    st.subheader("Pre-fetched priors")
    if st.session_state.dicom_prefetch:
        st.dataframe(st.session_state.dicom_prefetch, width="stretch", hide_index=True)
    else:
        st.caption("Waiting for registration alert…")


def apply_missing_data_shield(raw: dict) -> dict:
    """Populate unknown slots + attention masks before multimodal fusion."""
    fields = ["age", "bp_systolic", "bp_diastolic", "symptoms", "vitals_note", "smoking"]
    filled = {}
    masks = {}
    for f in fields:
        val = raw.get(f)
        missing = val is None or str(val).strip() == "" or str(val).lower() in {"unknown", "n/a", "-"}
        if missing:
            filled[f] = "__UNKNOWN__"
            masks[f] = 0  # attention must ignore
        else:
            filled[f] = val
            masks[f] = 1
    image_weight = 1.0 if sum(masks.values()) == 0 else 0.55 + 0.45 * (1 - sum(masks.values()) / len(masks))
    return {
        "tokens": filled,
        "attention_mask": masks,
        "image_focus": round(image_weight, 2),
        "shielded": True,
        "note": "Dummy tokens + masks keep multimodal attention stable when charts are incomplete.",
    }


def render_missing_shield(speak=None):
    _ensure_ops_state()
    _card(
        "2 · Missing Data Shield (Multimodal Attention)",
        "Before fusion, empty clinical fields become neutral Unknown tokens with attention masks "
        "so the network never crashes — focus shifts onto image pixels.",
    )
    with st.form("shield_form"):
        age = st.text_input("Age", value="")
        bp_s = st.text_input("Blood pressure (systolic)", value="")
        bp_d = st.text_input("Blood pressure (diastolic)", value="")
        symptoms = st.text_input("Symptoms", value="cough")
        smoking = st.selectbox("Smoking", ["", "never", "former", "current"])
        submitted = st.form_submit_button("Run Missing Data Shield", type="primary", width="stretch")
    if submitted:
        raw = {
            "age": age,
            "bp_systolic": bp_s,
            "bp_diastolic": bp_d,
            "symptoms": symptoms,
            "vitals_note": "",
            "smoking": smoking,
        }
        out = apply_missing_data_shield(raw)
        st.session_state.shield_last = out
        ignored = [k for k, m in out["attention_mask"].items() if m == 0]
        st.session_state.log_entries.append(
            f"[{time.strftime('%H:%M:%S')}] [SHIELD] Masked {len(ignored)} fields → image_focus={out['image_focus']}"
        )
        if speak:
            speak(
                f"Missing data shield applied. {len(ignored)} fields masked. "
                f"Image attention focus set to {int(out['image_focus'] * 100)} percent."
            )
    if st.session_state.shield_last:
        out = st.session_state.shield_last
        a, b = st.columns(2)
        a.json({"tokens": out["tokens"]})
        b.json({"attention_mask": out["attention_mask"], "image_focus": out["image_focus"]})
        st.success(out["note"])


def run_multitask_forward(modality_hint: str = "auto") -> dict:
    """Single shared backbone → parallel TB / nodule / mammo heads (demo vectors)."""
    backbone = {
        "encoder": "SharedDenseNet-Edge-v1",
        "features": random.randint(980, 1280),
        "latency_ms": random.randint(38, 72),
    }
    heads = {
        "tb": {
            "label": random.choice(["TB Negative", "TB Suspect", "TB Positive"]),
            "confidence": round(random.uniform(0.72, 0.96), 3),
        },
        "nodules": {
            "boxes": random.randint(0, 3),
            "max_mm": round(random.uniform(3.0, 14.0), 1) if random.random() > 0.4 else 0.0,
            "confidence": round(random.uniform(0.68, 0.94), 3),
        },
        "mammo": {
            "bi_rads": random.choice([1, 2, 3, 4]),
            "density": random.choice(["A", "B", "C", "D"]),
            "confidence": round(random.uniform(0.70, 0.95), 3),
        },
    }
    return {
        "backbone": backbone,
        "heads": heads,
        "modality_hint": modality_hint,
        "ram_note": "One model file — no sequential multi-model RAM thrash.",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def render_multitask(speak=None):
    _ensure_ops_state()
    _card(
        "3 · Unified Multi-Task Vision Architecture",
        "One shared feature backbone with parallel TB, nodule, and mammography heads — "
        "one forward pass, three diagnostic vectors.",
    )
    hint = st.selectbox("Study cue", ["auto", "chest", "mammo", "mixed"], key="ops_mt_hint")
    if st.button("Run unified multi-task pass", type="primary", width="stretch", key="ops_mt_run"):
        out = run_multitask_forward(hint)
        st.session_state.multitask_last = out
        st.session_state.log_entries.append(
            f"[{time.strftime('%H:%M:%S')}] [MULTI-TASK] backbone={out['backbone']['encoder']} "
            f"features={out['backbone']['features']} latency={out['backbone']['latency_ms']}ms"
        )
        if speak:
            h = out["heads"]
            speak(
                f"Multi-task inference complete. Tuberculosis {h['tb']['label']}. "
                f"{h['nodules']['boxes']} nodules. BI-RADS {h['mammo']['bi_rads']}."
            )
    if st.session_state.multitask_last:
        out = st.session_state.multitask_last
        st.caption(out["ram_note"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Backbone features", out["backbone"]["features"])
        k2.metric("Latency", f"{out['backbone']['latency_ms']} ms")
        k3.metric("TB", out["heads"]["tb"]["label"])
        k4.metric("Nodules", out["heads"]["nodules"]["boxes"])
        st.json(out["heads"])


def queue_care_cascade(finding: str, specialty: str, speak=None) -> dict:
    appt = (datetime.now() + timedelta(days=random.randint(2, 7))).strftime("%Y-%m-%d %H:%M")
    job = {
        "id": f"CC-{random.randint(10000, 99999)}",
        "finding": finding,
        "specialty": specialty,
        "referral": f"Auto-referral → {specialty}",
        "appointment": appt,
        "status": "Queued on booking server",
        "created": datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state.setdefault("cascade_jobs", [])
    st.session_state.cascade_jobs.insert(0, job)
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [CARE-CASCADE] {job['id']} {job['referral']} @ {appt}"
    )
    if speak:
        speak(f"Care cascade started. Referral to {specialty} and follow-up booked for {appt}.")
    return job


def render_care_cascade(speak=None):
    _ensure_ops_state()
    _card(
        "4 · Downstream Care Cascade Automation",
        "On Accept of a critical AI alert, the system talks to booking, formats a specialist "
        "referral, and drops a follow-up into the patient calendar — no paperwork left behind.",
    )
    finding = st.selectbox(
        "Accepted critical finding",
        [
            "TB Positive — high priority",
            "BI-RADS 4 — suspicious",
            "Lung nodule > 8mm",
            "Maternal risk escalation",
        ],
        key="ops_cc_finding",
    )
    specialty = st.selectbox(
        "Specialist route",
        ["Pulmonology", "Oncology", "Breast Clinic", "Obstetrics"],
        key="ops_cc_spec",
    )
    if st.button("Physician Accept → trigger cascade", type="primary", width="stretch", key="ops_cc_go"):
        job = queue_care_cascade(finding, specialty, speak=speak)
        st.success(f"{job['id']}: {job['referral']} · appointment {job['appointment']}")
    if st.session_state.cascade_jobs:
        st.dataframe(st.session_state.cascade_jobs, width="stretch", hide_index=True)
    else:
        st.caption("No cascade jobs yet.")


def quarantine_disagreement(ai_label: str, clinician_label: str, modality: str, speak=None) -> dict:
    """Anonymize disagreement and park in local quarantine for later secure upload."""
    payload = {
        "ai": ai_label,
        "clinician": clinician_label,
        "modality": modality,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    record = {
        "case_hash": digest,
        "ai_label": ai_label,
        "clinician_label": clinician_label,
        "modality": modality,
        "pii_stripped": True,
        "folder": f"edge_quarantine/{digest}.json",
        "uploaded": False,
    }
    # Persist anonymized packet locally (no PHI)
    qdir = Path(__file__).resolve().parent / "edge_quarantine"
    qdir.mkdir(exist_ok=True)
    (qdir / f"{digest}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    st.session_state.setdefault("telemetry_quarantine", [])
    st.session_state.telemetry_quarantine.insert(0, record)
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [TELEMETRY] Quarantined disagreement {digest} (PII stripped)"
    )
    if speak:
        speak("Disagreement captured, anonymized, and stored in the local quarantine cache.")
    return record


def flush_telemetry_uploads(speak=None) -> int:
    pending = [r for r in st.session_state.get("telemetry_quarantine", []) if not r.get("uploaded")]
    for r in pending:
        r["uploaded"] = True
        r["uploaded_at"] = datetime.now().strftime("%H:%M:%S")
        st.session_state.setdefault("telemetry_uploads", []).insert(0, dict(r))
    if pending:
        st.session_state.log_entries.append(
            f"[{time.strftime('%H:%M:%S')}] [TELEMETRY] Uploaded {len(pending)} anonymized error packets"
        )
        if speak:
            speak(f"Secure upload complete for {len(pending)} anonymized disagreement packets.")
    return len(pending)


def render_telemetry(speak=None):
    _ensure_ops_state()
    _card(
        "5 · Local Telemetry & Feedback Loop",
        "Doctor disagreements are anonymized into a local quarantine folder. On next secure link, "
        "only these high-value error packets sync for model improvement — no patient identifiers.",
    )
    ai = st.text_input("AI prediction", value="Positive", key="ops_tel_ai")
    clin = st.text_input("Clinician corrected label", value="Negative", key="ops_tel_clin")
    mod = st.selectbox("Modality", ["Tuberculosis", "Mammography", "Maternal"], key="ops_tel_mod")
    b1, b2 = st.columns(2)
    if b1.button("Record disagreement → quarantine", type="primary", width="stretch", key="ops_tel_q"):
        quarantine_disagreement(ai, clin, mod, speak=speak)
        st.success("Anonymized case written to edge_quarantine/")
    if b2.button("Secure uplink: flush quarantine", width="stretch", key="ops_tel_up"):
        n = flush_telemetry_uploads(speak=speak)
        if n:
            st.success(f"Uploaded {n} packets.")
        else:
            st.info("Nothing pending.")
    st.subheader("Quarantine cache")
    if st.session_state.telemetry_quarantine:
        st.dataframe(st.session_state.telemetry_quarantine, width="stretch", hide_index=True)
    else:
        st.caption("Empty — disagreements will appear here.")


def render_edge_ops(speak=None):
    """Main Edge Ops page with the five client modules."""
    _ensure_ops_state()
    st.markdown(
        '<div class="nav-hint" style="margin-bottom:12px;">'
        "Client modules — DICOM routing, missing-data shield, multi-task vision, "
        "care cascade, and local telemetry"
        "</div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(
        [
            "DICOM Prefetch",
            "Missing Shield",
            "Multi-Task Vision",
            "Care Cascade",
            "Telemetry Loop",
        ]
    )
    with tabs[0]:
        render_dicom_prefetch(speak=speak)
    with tabs[1]:
        render_missing_shield(speak=speak)
    with tabs[2]:
        render_multitask(speak=speak)
    with tabs[3]:
        render_care_cascade(speak=speak)
    with tabs[4]:
        render_telemetry(speak=speak)
