# Accessibility

Pink Edge AI is a hackathon-grade clinical-triage **prototype**, built for frontline Lady Health
Workers as much as for Senior Radiologists. This document describes, honestly, what the code
actually does for accessibility today and what it does not — in the same spirit as
[SECURITY.md](SECURITY.md) and `App/Documentations/PRIVACY_POLICY.md`: a claim here corresponds to
something you can run and check, and a gap is stated as a gap rather than left implied.

**This is not a conformance statement.** It has not been through an accessibility audit, there is
no VPAT, and no claim below should be read as WCAG 2.1 AA certification — only one specific,
testable slice of it (§1.1) is actually enforced by an automated check.

---

## 1. What's implemented and verified

### 1.1 Color contrast — WCAG 2.1 AA, enforced by a test, not a style guide

`App/Tests/smoke_test.py` computes the real WCAG relative-luminance contrast ratio (the same
formula as the spec: linearized sRGB, `0.2126R + 0.7152G + 0.0722B`) for every text/icon color the
UI actually uses against the background it's actually drawn on, and fails the suite if any pair
falls below **4.5:1**. It checks 11 pairs: `text`, `text_muted`, `text_light`, `primary`,
`primary_light`, `accent`, `accent_light`, `success`, `warning`, and `danger` against both
`surface` and `bg`, plus `console_text` on `console_bg` for the one dark panel in the app.

This isn't theoretical — it has already caught a real violation: an earlier violet accent
(`#a855f7`) measured 3.5:1 on white and was replaced with `#6d28d9` (violet-700) once the check
existed (see the comment in `App/auth_manager.py`'s `ROLE_CONFIGS`).

Both UI editions read color from **one palette object** — `GUI.py`'s `C` dict — rather than each
keeping its own copy; `streamlit_app.py` does `C = core.C` and `.streamlit/config.toml` is checked
against the same source. The same suite verifies that sharing hasn't drifted apart, so a contrast
fix made once in `GUI.py` can't silently fail to reach one of the three places color is defined.

### 1.2 Status and severity are not color-only

Triage severity (`Low` / `Moderate` / `High`) and modality status are shown as colored badges, but
always paired with an explicit text label and/or icon (e.g. `🟢`/`⚪` plus the word "Routine
Screening", BI-RADS/severity rendered as text inside the badge, not just its background color) —
satisfying WCAG's "don't use color as the only visual means of conveying information" principle,
even though nothing automatically checks this one.

### 1.3 Bilingual UI

Both live editions (desktop and Streamlit) support a full English/Urdu toggle through one shared
translation function, `t(s)` in `GUI.py`, backed by a single string-lookup dictionary — so a
string translated once is translated in both editions. The Voice Guidance panel's text templates
go further and include a third language, Punjabi, for the message library (see §1.4 and §2).

### 1.4 Text-first fallback for voice guidance

The Voice Guidance feature is designed so a missing audio engine degrades to text, not to nothing:
`play_voice_message()` is a documented stub that returns `None`, and the UI's own fallback path
shows the message text and an explicit "🔇 No audio engine configured yet — showing text only"
caption rather than a silent or broken control.

### 1.5 This is a regression-tested property, not a one-time pass

The contrast and theme-consistency checks in §1.1 run as part of the same suite described in
[SECURITY.md §1.5](SECURITY.md#15-automated-verification-not-self-assessment) and
[CONTRIBUTING.md §4](CONTRIBUTING.md#4-testing-your-change). A future color or palette change that
breaks AA contrast fails `Tests/smoke_test.py`, the same way a privacy regression fails
`privacy_leak_test.py` — it doesn't rely on someone noticing in review.

---

## 2. Known gaps, stated plainly

- **No RTL layout or Urdu-appropriate font in the current rebuild.** Urdu strings are translated
  correctly by `t(s)`, but render in the UI's default left-to-right layout and font. The
  *superseded* fork this project was rebuilt from (`App/Archive/pink_edge.py`) did define
  `.urdu { direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', serif; }` — that
  CSS did not carry over into `GUI.py` / `streamlit_app.py` during the rebuild. Verified by
  searching both live files for `rtl` and `Nastaliq`: zero matches in either.
- **No screen-reader support has been implemented or audited.** Tkinter (the desktop edition)
  exposes very little to assistive technology on its own, and nothing in `GUI.py` adds accessible
  names or platform accessibility (MSAA/UIA) metadata. The Streamlit edition inherits whatever
  ARIA roles Streamlit's own components ship with; nothing in `streamlit_app.py` adds more.
- **No keyboard-only navigation audit.** There is no custom tab order, no skip links, and no
  documented keyboard shortcuts in either edition — both rely entirely on their framework's and
  the browser's/OS's default tab/focus behavior, untested.
- **Clinical images have no descriptive alt text.** `st.image(...)` is given a `caption` — the
  uploaded filename, or a placeholder label — which is visible, real text, but it describes the
  *file*, not the finding (e.g., it doesn't say where the bounding-box overlay is or what it
  marks). A screen-reader user does not get the information a sighted user gets from the overlay.
- **Voice Guidance has no audio yet.** As in §1.4, the fallback is graceful, but today there is no
  TTS engine wired in at all — `play_voice_message()` always returns `None`. For a feature whose
  stated purpose is supporting low-literacy or vision-impaired use in the field, text-only is a
  real functional gap, not just a missing nice-to-have.
- **Only color contrast and theme drift are automated.** §1.1's suite does not check keyboard
  operability, focus order, touch-target size, or screen-reader labeling. It is one real, narrow
  slice of WCAG 2.1 AA — not a stand-in for full conformance testing.
- **The desktop edition has no text-scaling control of its own.** `GUI.py` does not implement
  zoom or font-scaling; it inherits whatever display scaling the OS provides.
- **No third-party accessibility audit or VPAT exists for this project.**

---

## 3. Verifying the contrast claim yourself

```
cd App
python Tests/smoke_test.py        # -v to watch each check run
```

This is the same smoke test described in `SECURITY.md`/`CONTRIBUTING.md`; its WCAG check is one of
the ~9 checks it runs. A failing pair prints the exact token names and the measured ratio, e.g.
`accent on surface: 3.80:1`.

---

## 4. If you're contributing an accessibility fix

- **Change color in one place.** The palette lives in `GUI.py`'s `C` dict; `streamlit_app.py` and
  `.streamlit/config.toml` derive from or are checked against it (§1.1). Don't add a second copy.
  Re-run `python Tests/smoke_test.py` after any palette change.
- **Fixing Urdu RTL/typography:** `App/Archive/pink_edge.py`'s retired CSS (§2) is a reasonable
  starting reference for the font and `direction: rtl` rule, but it targeted a single-edition,
  Streamlit-only UI — adapt it for the shared palette/two-edition structure rather than copying it
  wholesale, and check it against both `GUI.py` and `streamlit_app.py`.
- **Closing a screen-reader or keyboard gap:** there is no existing test to extend, so add one —
  this project's convention (see
  [SECURITY.md §1.5](SECURITY.md#15-automated-verification-not-self-assessment)) is that a fix
  ships with the check that would have failed before it, not just a code change.
- Follow the file-header and PR conventions in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Contact

**Imaad Ullah Khan** — yameenimaad@gmail.com — for accessibility feedback, including if you use
assistive technology with this project and hit something not listed above.
