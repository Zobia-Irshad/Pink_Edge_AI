# Pink Edge AI — Frontend Documentation

The entire UI is built with **Streamlit**, styled with a large block of injected custom CSS (dark clinical theme), and rendered through a small set of `render_*()` functions. There is no separate JS/React frontend — Streamlit generates the DOM from Python calls each re-run.

## 1. Page Setup

```python
st.set_page_config(page_title="Pink Edge AI", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
```

Wide layout, sidebar expanded by default, custom CSS injected once at the top of `main()`.

## 2. Design System (CSS tokens)

Defined as CSS custom properties on `:root`:

| Token | Value | Purpose |
|---|---|---|
| `--bg` | `#0a0e1a` | Page background |
| `--surface` / `--surface-alt` / `--surface-hover` | `#111827` / `#1e293b` / `#334155` | Card/panel backgrounds |
| `--border` / `--border-light` | `#1e293b` / `#334155` | Dividers |
| `--text` / `--text-muted` / `--text-light` | `#f1f5f9` / `#94a3b8` / `#64748b` | Typography hierarchy |
| `--primary` / `--primary-light` | `#14b8a6` / `#2dd4bf` | Brand teal (buttons, active states, key metrics) |
| Status colors (used inline) | success `#10b981`, danger `#ef4444`, warning `#f59e0b` | Verdict boxes, alert severity |

**Fonts**: Inter (UI text), JetBrains Mono (telemetry/console log and data values), Noto Nastaliq Urdu (loaded for the Urdu translation mode).

## 3. Layout: Sidebar (`render_sidebar()`)

Persistent left rail, present on every tab:

1. **Brand header** — logo emoji + "Pink Edge AI / Clinical Intelligence Platform".
2. **Language switch** — two buttons, 🇬🇧 EN / 🇵🇰 اردو, toggling `st.session_state.urdu_mode`.
3. **Network Mode** — radio: `Fully Offline` / `GSM Failover`. Shows a colored badge reflecting the current mode.
4. **Select AI Model** — dropdown of the 3 `MODELS`.
5. **Upload Patient Scan** — `st.file_uploader` (jpg/jpeg/png).
6. **Actions**:
   - **Ingest DICOM** — cosmetic log entry only.
   - **Run Triage** — the core action; see `BACKEND_DOCUMENTATION.md` §5 for what it triggers. Shows a 2-second spinner ("Running INT8 on NPU…") to sell the on-device-inference illusion.
   - **Save to Cache** — appears only after a triage has run; persists the confirmed report.
   - **Sync to Cloud** — appears only in GSM Failover mode with unsynced reports pending.
   - **Check OTA** — appears only in GSM Failover mode; simulated ACR version check.
7. **Show Detection Overlay** — checkbox toggling the bounding-box overlay on the analyzed image.
8. **Hardware Diagnostics** — live-updating panel (RK3588 status, NPU load %, power, temperature, active model).
9. **Reset Session** — clears all `st.session_state` and reruns.

## 4. Tab 1 — Dashboard (`render_dashboard()`)

Two-column layout (`3:2` split):

**Left column — Medical Image Analysis**
- Procedurally generated or uploaded image, rendered inside a styled `.image-viewer` frame with a caption overlay.
- After a triage run, shows the image with an optional AI bounding-box overlay (`draw_bbox()`), plus three metric tiles: Model, Confidence, Latency.
- Before any triage, shows the plain image with an "Awaiting Analysis" info banner.

**Right column — DICOM Metadata + Verdict**
- A metadata grid (Patient ID, Age, Modality, Date, Body Part, Institution) styled to look like a DICOM header.
- After triage: a color-coded **verdict box** (green/red) with icon, title, and sub-text; a detail card (Localization, Classification, Inference Time, GSM Alert status); an editable **Confirm Assessment** section — BI-RADS + ACR dropdowns for Mammography/Maternal Health, or TB Severity + Lung Zone dropdowns for the TB pathway (the UI deliberately swaps the field set per modality, since BI-RADS doesn't apply to chest X-ray); a warning/success banner based on `is_critical`; and **Download Text Report** / **Download PDF Report** buttons.

**Bottom — Telemetry Log**
A monospace "console" (`.console-log`) that appends color-coded lines per pipeline stage (`[DICOM]`, `[NPU]`, `[GSM]`, `[Alibaba IoT]`, `[SQLite]`) as the user progresses through Ingest → Triage → Cache → Sync.

## 5. Tab 2 — Hospital Hub (`render_hospital_hub()`)

Represents the urban receiving terminal ("Allied Hospital Faisalabad"). Two-column layout (`2:1`):

- **Alert Stream** (main column) — a card per incoming SMS/GSM alert, color-coded critical (red) vs. routine (green), showing the raw payload string, alert type, and status, with an "Acknowledge" button. Shows an "All Clear" empty state when there are no alerts.
- **Network Status + Stats** (side column) — GSM module status, signal strength, connected-BHU node count, pending-alert count, and two metric tiles (Total Alerts, Critical Cases).

## 6. Tab 3 — Cloud Sync (`render_cloud_sync()`)

- Cloud-sync status badge (GSM active vs. offline/disabled).
- Four metric tiles: Cached Reports, Unsynced, Synced, IoT Queue.
- **Alibaba Cloud IoT Platform** card — endpoint, message count, target table.
- **Alibaba Cloud OSS** card — bucket, region, upload count (only high-risk BI-RADS 4/5 patches are "uploaded").
- **Alibaba Cloud ACR — OTA** card — current vs. available model version once a check has been run.
- **Hybrid-Edge Architecture** card — the ASCII architecture diagram reproduced inline.
- **Local Cache Status** table — every cached report (ID, Patient ID, Modality, BI-RADS, ACR, Verdict, Timestamp, Synced ✅/⏳).

## 7. Shared UI Components (CSS classes)

| Class | Used for |
|---|---|
| `.page-header` | Gradient banner heading each tab |
| `.badge` (`.badge-cloud` / `.badge-offline`) | Small pill indicating network mode |
| `.section-header` | Icon + title row above each content block |
| `.metric-tile` | Small KPI card (label + big value, color variants: `primary`/`success`/`danger`/`warning`/`accent`) |
| `.verdict-box` (`.success` / `.danger`) | Large AI verdict callout |
| `.dicom-grid` / `.dicom-item` | Metadata grid |
| `.card` / `.card-primary` / `.card-success` | Generic bordered content panels |
| `.cloud-card` | Alibaba service status cards |
| `.hw-panel` / `.hw-row` | Sidebar hardware/network diagnostics rows |
| `.console-log` | Monospace telemetry log |
| `.alert-card` (`.critical` / `.ok`) | Hospital Hub alert entries |
| `.data-table` | Cached reports table |
| `.app-footer` | Version/footer line at the bottom of the page |

## 8. Internationalization in the UI

Every visible label calls `t("...")`. Toggling the language buttons flips `st.session_state.urdu_mode` and triggers `st.rerun()`, so the whole page re-renders in the selected language on the next pass — there is no partial/localized re-render.

## 9. Image Handling

- `generate_mammogram(size=512, seed=42)` (and sibling generators for X-ray/ultrasound, referenced via `load_image()`) procedurally synthesize a plausible grayscale medical image using NumPy when no file is uploaded, cached via `@st.cache_data`.
- `draw_bbox()` overlays a detection rectangle (OpenCV) on the image when "Show Detection Overlay" is enabled and a triage result exists.
