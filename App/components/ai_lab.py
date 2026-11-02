"""AI Lab — XAI panel + offline_cv template visualizer. No in-app model training."""
from __future__ import annotations

import streamlit as st

import GUI as core

from components.metrics import track


def _modality_key(model_name: str) -> str:
    name = (model_name or "").lower()
    if "tuberculosis" in name or "chest" in name:
        return "tb"
    if "maternal" in name:
        return "maternal"
    return "mammography"


def render_template_visualizer(modality_model: str):
    """Side-by-side positive vs negative offline_cv templates + optional patient image."""
    st.markdown("### Offline CV template visualizer")
    st.caption(
        "Pixel-diff heuristic references from `offline_cv.py` — explains why a bounding box "
        "leans positive vs negative. Not a neural heatmap."
    )
    try:
        import offline_cv
        from PIL import Image
        import numpy as np
    except Exception as e:
        st.warning(f"offline_cv unavailable: {e}")
        return

    key = _modality_key(modality_model)
    # Public wrapper preferred; fall back to internal getter
    get_tpl = getattr(offline_cv, "get_reference_templates", None) or getattr(offline_cv, "_get_templates")
    pos, neg = get_tpl(key)
    if pos is None or neg is None:
        st.info(
            f"No local templates for `{key}` yet. Place/build templates under "
            f"Models/*/templates (positive.png / negative.png)."
        )
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Positive template**")
        st.image(Image.fromarray(pos.astype("uint8")), use_container_width=True)
        st.caption("Average abnormal / positive reference")
    with c2:
        st.markdown("**Negative template**")
        st.image(Image.fromarray(neg.astype("uint8")), use_container_width=True)
        st.caption("Average healthy / negative reference")
    with c3:
        st.markdown("**Patient scan**")
        img = st.session_state.get("current_image")
        if img is not None:
            st.image(img, use_container_width=True)
            try:
                # Show lean map briefly if analyze helpers exist
                if hasattr(offline_cv, "_analyze"):
                    score, pct, bbox, orient = offline_cv._analyze(key, img, pos, neg)
                    st.caption(f"Lean score {score:.2f} · {pct:.1f}% positive pixels · {orient}")
                    if bbox:
                        st.caption(f"Heuristic bbox (norm xywh): {tuple(round(x, 3) for x in bbox)}")
            except Exception:
                st.caption("Patient image loaded — run triage for full result fields.")
        else:
            st.info("Run triage (or FYP Demo) to load a patient image here.")


def render_ai_lab(selected_model: str, *, render_xai_panel, render_training_arena, launch_fyp_demo):
    track("nav_ai_lab")
    st.markdown(
        '<div class="nav-hint" style="margin-bottom:12px;">'
        "Radiologist AI Lab — Explainable AI, template visualizer, guided demo, LHW training. "
        "Model training stays CLI-only (not launched from this UI)."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.demo_mode and st.session_state.demo_step:
        st.markdown(
            f'<div class="demo-banner"><b>Guided demo active</b><br><span>{st.session_state.demo_step}</span></div>',
            unsafe_allow_html=True,
        )

    d1, d2 = st.columns([2, 1])
    with d1:
        st.markdown("### One-click FYP Demo")
        st.write(
            "Loads a real Test Data scan, runs the **offline** triage pipeline "
            "(`GUI.run_triage` / `inference.py`), builds an Explainable AI trace, "
            "and queues a Hospital Hub alert."
        )
    with d2:
        demo_mod = st.selectbox(
            "Demo modality",
            core.MODELS,
            index=1 if len(core.MODELS) > 1 else 0,
            key="demo_modality_select",
        )
        if st.button("🚀 Launch FYP Demo", type="primary", width="stretch"):
            track("fyp_demo_launch", detail=demo_mod)
            launch_fyp_demo(demo_mod)

    st.markdown("---")
    st.markdown("### Explainable AI (live case)")
    if st.session_state.inference_done and st.session_state.current_result:
        render_xai_panel(
            st.session_state.current_result,
            selected_model or st.session_state.last_model or core.MODELS[0],
        )
    else:
        st.info("Run triage on Dashboard (or Launch FYP Demo) to populate the XAI panel.")

    st.markdown("---")
    render_template_visualizer(selected_model or st.session_state.last_model or core.MODELS[0])

    st.markdown("---")
    # Training arena = judgment drill, NOT train_local_model / train_tb_classifier
    render_training_arena()
