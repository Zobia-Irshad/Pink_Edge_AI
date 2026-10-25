#!/usr/bin/env python3
"""
Pink Edge AI — smoke test.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Drives the real Streamlit app end to end the way a user would (via
`streamlit.testing.v1.AppTest`: it executes streamlit_app.py for real, no browser), then checks
that the theme is internally consistent across the four places it's defined.

This is deliberately *not* a substitute for `Validation/validate.py` (19 checks, real model
inference, ground-truth cross-checks, both UIs). This is the fast one: ~30s, no model downloads,
answers "is the app standing up and is it still the same app" before you touch anything else.

Run:  python Tests/smoke_test.py          (from App/)
      python Tests/smoke_test.py -v       (show each check as it runs)

Exits non-zero if any check fails.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

RESULTS = []


def check(name):
    """Register a check; same reporting shape as Validation/validate.py."""
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
            if VERBOSE:
                print(f"  [pass] {name} {detail}")
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            if VERBOSE:
                print(f"  [FAIL] {name} — {type(e).__name__}: {e}")
        return fn
    return wrap


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ============================================================
# 1. The app itself — does it stand up, and do the core journeys work?
# ============================================================
_app = {}


def _fresh_app(timeout=90):
    """A freshly-run AppTest instance. Cached per-process for the first run so the
    several checks below don't each pay the startup cost."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(APP_DIR, "streamlit_app.py"))
    at.run(timeout=timeout)
    return at


def _sidebar_button(at, substring):
    """Look up a sidebar button by label, never by index — the sidebar's button order has
    already shifted once in this project's history (a role switcher was added ahead of the
    language toggles), silently breaking positional lookups."""
    matches = [b for b in at.sidebar.button if substring in b.label]
    require(matches, f"no sidebar button containing {substring!r} (have: {[b.label for b in at.sidebar.button]})")
    return matches[0]


@check("app starts with no exception and renders its 3 tabs")
def _():
    at = _fresh_app()
    require(not at.exception, f"startup raised: {list(at.exception)}")
    require(len(at.tabs) == 3, f"expected 3 tabs (Dashboard/Hospital Hub/Cloud Sync), got {len(at.tabs)}")
    _app["at"] = at
    return f"({len(at.sidebar.button)} sidebar controls)"


@check("branding is Pink Edge AI everywhere the user can see it")
def _():
    src = open(os.path.join(APP_DIR, "streamlit_app.py"), encoding="utf-8").read()
    m = re.search(r'page_title\s*=\s*["\']([^"\']+)["\']', src)
    require(m, "no st.set_page_config(page_title=...) found")
    require("Pink Edge" in m.group(1), f"page_title is {m.group(1)!r}, expected to contain 'Pink Edge'")
    # The 'Medical Radiology AI' rebrand was reverted; nothing should reintroduce it.
    for path in ("streamlit_app.py", "GUI.py", "inference.py"):
        body = open(os.path.join(APP_DIR, path), encoding="utf-8").read()
        require("Medical Radiology" not in body, f"{path} still references 'Medical Radiology'")
        for dead in ("radiology_console", "radiology_web"):
            require(dead not in body, f"{path} still references the retired module {dead!r}")
    return f"(page_title={m.group(1)!r})"


@check("role switch, language toggle and modality select all re-run cleanly")
def _():
    at = _app.get("at") or _fresh_app()
    for label in ("Doctor", "LHW", "اردو", "EN"):
        _sidebar_button(at, label).click().run(timeout=30)
        require(not at.exception, f"clicking {label!r} raised: {list(at.exception)}")
    sel = next(s for s in at.sidebar.selectbox if s.label == "Select AI Model")
    require(len(sel.options) == 3, f"expected 3 modalities, got {sel.options}")
    for modality in sel.options:
        sel.set_value(modality).run(timeout=30)
        require(not at.exception, f"selecting {modality!r} raised: {list(at.exception)}")
    _app["at"] = at
    return f"({len(sel.options)} modalities)"


@check("Run Triage completes and produces a well-formed result")
def _():
    at = _app.get("at") or _fresh_app()
    _sidebar_button(at, "Run Triage").click().run(timeout=120)
    require(not at.exception, f"Run Triage raised: {list(at.exception)}")
    require(at.session_state["inference_done"], "Run Triage did not set inference_done")
    r = at.session_state["current_result"]
    for field in ("verdict", "confidence", "source", "is_critical", "css"):
        require(field in r, f"result is missing {field!r}")
    require(0.0 <= float(r["confidence"]) <= 100.0, f"confidence out of range: {r['confidence']}")
    require(r["css"] in ("success", "warning", "danger"), f"unexpected css token {r['css']!r}")
    _app["at"] = at
    return f"({r['source'][:48]}...)"


@check("Save to Cache persists the result")
def _():
    import GUI as core

    at = _app.get("at") or _fresh_app()
    require(at.session_state.get("inference_done") if hasattr(at.session_state, "get") else at.session_state["inference_done"],
            "no triage result to save — the Run Triage check must pass first")
    before, _ = core.counts()
    _sidebar_button(at, "Save to Cache").click().run(timeout=30)
    require(not at.exception, f"Save to Cache raised: {list(at.exception)}")
    after, _ = core.counts()
    require(after == before + 1, f"cache went from {before} to {after} rows, expected +1")
    return f"({before} -> {after} rows)"


# ============================================================
# 2. Theme consistency — the palette is defined in four places; they must agree.
# ============================================================
@check("desktop and web editions share one palette object")
def _():
    import GUI as core

    src = open(os.path.join(APP_DIR, "streamlit_app.py"), encoding="utf-8").read()
    require(re.search(r"^C\s*=\s*core\.C\s*$", src, re.M),
            "streamlit_app.py no longer does `C = core.C` — it has its own palette again, which is "
            "how the two editions drifted apart last time")
    for key in ("primary", "accent", "success", "warning", "danger", "bg", "surface", "text"):
        require(key in core.C, f"GUI.py's C is missing {key!r}")
    return f"({len(core.C)} tokens)"


@check(".streamlit/config.toml matches the shared palette")
def _():
    import GUI as core

    cfg_path = os.path.join(APP_DIR, ".streamlit", "config.toml")
    cfg = open(cfg_path, encoding="utf-8").read()

    def cfg_value(key):
        m = re.search(rf'^{key}\s*=\s*"([^"]+)"', cfg, re.M)
        require(m, f"config.toml has no {key}")
        return m.group(1).lower()

    for cfg_key, token in (("primaryColor", "primary"), ("backgroundColor", "bg"),
                           ("secondaryBackgroundColor", "surface"), ("textColor", "text")):
        want = core.C[token].lower()
        got = cfg_value(cfg_key)
        require(got == want, f"config.toml {cfg_key}={got} but C[{token!r}]={want}")
    return "(4 keys)"


@check("the retired teal/hotpink theme is fully gone")
def _():
    # The three palettes that were in play before unification. If any reappears, something was
    # pasted back in from an old file or an old commit.
    retired = {
        "#0d9488": "teal primary (old web edition)",
        "#0f766e": "teal hover (old web edition)",
        "#1a237e": "indigo primary (old desktop edition)",
        "#ff69b4": "hotpink (old web brand gradient)",
        "#ff1493": "deeppink (old web brand gradient)",
    }
    offenders = []
    for path in ("streamlit_app.py", "GUI.py", os.path.join(".streamlit", "config.toml")):
        body = open(os.path.join(APP_DIR, path), encoding="utf-8").read()
        for hexval, what in retired.items():
            # GUI.py's THEME comment cites the old values deliberately; ignore comment lines.
            hits = [ln for ln in body.splitlines()
                    if hexval in ln.lower() and not ln.lstrip().startswith("#")]
            if hits:
                offenders.append(f"{path}: {hexval} ({what})")
    require(not offenders, "retired theme colours are back: " + "; ".join(offenders))
    return f"({len(retired)} colours checked)"


@check("every palette colour used as text clears WCAG AA on its background")
def _():
    import GUI as core

    def rel_lum(hexval):
        r, g, b = (int(hexval[i:i + 2], 16) / 255 for i in (1, 3, 5))
        r, g, b = ((c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def ratio(fg, bg):
        l1, l2 = sorted((rel_lum(fg), rel_lum(bg)), reverse=True)
        return (l1 + 0.05) / (l2 + 0.05)

    failures = []
    # Foreground tokens rendered as text/icons on the light surfaces.
    for token in ("text", "text_muted", "text_light", "primary", "primary_light",
                  "accent", "accent_light", "success", "warning", "danger"):
        for ground in ("surface", "bg"):
            r = ratio(core.C[token], core.C[ground])
            if r < 4.5:
                failures.append(f"{token} on {ground}: {r:.2f}:1")
    # The console panels are the one dark surface in the app.
    r = ratio(core.C["console_text"], core.C["console_bg"])
    if r < 4.5:
        failures.append(f"console_text on console_bg: {r:.2f}:1")
    require(not failures, "below WCAG AA (4.5:1): " + "; ".join(failures))
    return "(11 pairs >= 4.5:1)"


# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PINK EDGE AI — SMOKE TEST")
    print("=" * 70)
    if not VERBOSE:
        print("(run with -v to see checks as they execute)\n")

    width = max(len(n) for n, _, _ in RESULTS) + 2
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)} {detail}")

    print(f"\n{n_pass}/{len(RESULTS)} checks passed.")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
