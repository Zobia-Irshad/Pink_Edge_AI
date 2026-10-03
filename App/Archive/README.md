# Archive — superseded code, kept for reference

Nothing in this folder is imported, launched, or tested by the shipping app. It is kept because it
documents where the project came from, not because it runs. Neither `Start.bat` nor `Start_Web.bat`
nor `Validation/validate.py` touches anything here.

## `pink_edge.py` (1675 lines)

A **third** Streamlit UI, superseded by `../streamlit_app.py`. It is a fork of the original
hackathon app (`Pink_Edge_AI-main/pink_edge.py` below) that was partially re-themed in place, and
it diverges from that original by ~484 lines. No launcher has ever referenced it.

Do not revive it as-is — it has a latent access-control bug. It gates two panels on permissions
that do not exist in `auth_manager.ROLE_CONFIGS`:

| Line | Gate | Status |
|---|---|---|
| `pink_edge.py:1168` | `has_permission("view_hardware_stats")` | not a declared permission |
| `pink_edge.py:1373` | `has_permission("view_telemetry_logs")` | not a declared permission |

Because `has_permission()` fails closed (returns `False` for anything it doesn't recognize), those
panels are silently hidden from **every** role including Senior Radiologist, with no error raised.
`Tests/test_auth.py::test_permissions_gated_by_the_live_app_all_exist` is the regression guard that
now catches this class of mistake; if you port these panels forward, either declare the permissions
in `ROLE_CONFIGS` or gate them on existing ones, and add them to that test's `gated_by_app` set.

The live app (`../streamlit_app.py`) only gates on `override_assessment`, which is declared.

## `fix_theme.py` (24 lines)

A one-shot codemod that rewrote `pink_edge.py`'s CSS custom properties **in place** (dark theme →
light). It is spent: its replacements have already been applied to the `pink_edge.py` next to it,
so re-running it is a no-op at best. It also shipped with a UTF-8 BOM that made it a hard
`SyntaxError: invalid non-printable character U+FEFF` — it could not have run at all in that state.
The BOM has been stripped so the file at least parses, but it remains archived, not wired up.

Theme colors for the shipping app are **not** managed this way. They live in:
- `../streamlit_app.py` — the `C = {...}` color-token dict
- `../GUI.py` — its own `C = {...}` dict
- `../.streamlit/config.toml` — Streamlit's own theme block

## `orphaned_tb_classifier.pt` (2.9 MB)

Found sitting directly in `Models/` (not inside `Models/TB/`, the convention every other weight
file follows) with no code anywhere referencing it. `train_tb_classifier.py` writes its output to
`App/models/tb_classifier.pt` (lowercase `models/`, a directory that doesn't otherwise exist in
this tree) — not `App/Models/` — so this is most likely a one-off training run whose output ended
up in the wrong directory and was never cleaned up. Its content differs from
`Archive/Misc/Pink_Edge_AI-main/models/tb_classifier.pt` (different size, different hash), so it
isn't simply a duplicate of the original hackathon submission's checkpoint either. Moved here
rather than deleted, since nothing confirms it's safe to discard — if it turns out to be a real
checkpoint worth keeping, the fix is probably to point `train_tb_classifier.py`'s `MODEL_OUT` at
`Models/TB/` instead, matching every other model file's location.

## `Misc/`

Unsorted project scratch, carried over as-is:

- **`Pink_Edge_AI-main/`** — the original Alibaba-Cloud-Hackathon submission this whole project was
  built from: `pink_edge.py` (1499 lines), `Pink_Edge_AI.ipynb`, `models/tb_classifier.pt`, and its
  own `requirements.txt` + empty `pink_edge_cache.db`. This is the true upstream ancestor.
- **`Links.txt`** — reference URLs collected during the build.
- **`Pink Edge A1.pdf  1234.pdf`** — a pitch/report PDF (filename mangled by a double save).
- **`WhatsApp Image *.jpeg`** (8 files) — screenshots shared during development. Several filenames
  are visibly truncated or corrupted (`WhatsApp Image -09-12 at 7.21.23 PM.jpeg`,
  `WhatsApp Image 2026-09-125 PM.jpeg`, `WhatsApp Image 202609-12 at 7.21.24 PM.jpeg`). They were
  left byte-for-byte untouched rather than renamed, since nothing references them and guessing the
  intended names would only invent information.
