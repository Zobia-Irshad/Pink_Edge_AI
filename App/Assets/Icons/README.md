# Assets/Icons/

Real, downloaded icon files used by `radiology_console.py` (the Tkinter desktop console) in place
of relying on the system font to render emoji glyphs — inconsistent across Windows builds/fonts,
and Tk's own emoji rendering is notoriously patchy (often shows monochrome "tofu" outlines instead
of full-color glyphs). Every call site in the app still has a text/emoji fallback if a file here is
missing, so a partial or deleted `Assets/Icons/` folder never breaks the UI, just makes it fall back
to emoji text like before.

## Source and license

[Twemoji](https://github.com/twitter/twemoji) (Twitter's open-source emoji set, now maintained by
jdecked/twemoji), licensed **CC-BY 4.0** — graphics were downloaded, unmodified other than resizing,
from `https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/<codepoint>.png`. Attribution
per the license: "Graphics Titled Twemoji" — Copyright Twitter, Inc and other contributors,
Licensed under CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

## Files

| File | Emoji | Used for |
|---|---|---|
| `xray.png` | 🩻 | App/window icon, topbar brand |
| `stethoscope.png` | 🩺 | Dashboard tab |
| `hospital.png` | 🏥 | Hospital Hub tab + header |
| `cloud.png` | ☁️ | Cloud Sync tab + header, Sync to Cloud button |
| `gear.png` | ⚙️ | Sidebar "Console Menu" header |
| `info.png` | ℹ️ | Sidebar "About" header |
| `check.png` | ✅ | Verdict icon (normal/OK result) |
| `warning.png` | ⚠️ | Verdict icon (critical/positive result) |
| `prohibited.png` | 🚫 | Verdict icon (invalid/wrong image type) |
| `clipboard.png` | 📋 | Verdict icon (awaiting analysis) |
| `memo.png` | 📝 | "Text Report" button |
| `page.png` | 📄 | "PDF Report" button |
| `hourglass.png` | ⏳ | reserved — not yet wired in (pending-sync marker still uses text) |
| `app_icon.ico` | — | Windows title-bar/taskbar icon, generated locally from `xray.png` (PIL, multi-resolution 16-72px) — not a separate download |

Loaded via `_load_icon(name, size)` in `radiology_console.py`, which resizes on load and caches the
result (Tk requires a persistent reference to a `PhotoImage` or it silently vanishes from the
widget once garbage-collected).
