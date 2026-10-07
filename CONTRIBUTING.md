# Contributing to Pink Edge AI

Thanks for taking a look at this project. Pink Edge AI is a hackathon-grade clinical-triage
**prototype** (mammography, tuberculosis chest X-ray, maternal-health ultrasound) — not a
certified medical device — and it stays useful only as long as its documentation and tests match
what the code actually does. That's the bar for a contribution here: a checkable claim, not a
better-sounding one.

Read [SECURITY.md](SECURITY.md) first if what you have is a vulnerability report rather than a
code change — it has its own private-reporting process and must not go through a public issue or
PR.

---

## 0. Ground rules

- **Never commit real patient data, PHI, or PII.** Every identifier this project handles today is
  synthetic (`random.randint(10000000, 99999999)`, generated per session). Keep it that way. Use
  the samples in `App/Test Data/`, a public de-identified dataset you record the source of, or
  data you generate yourself.
- **Never commit a secret.** No API keys, tokens, or credentials in any file, commit message, or
  issue — see [SECURITY.md §1.4](SECURITY.md#14-secrets-handling) for exactly how this project
  supplies the Roboflow key without ever putting it in source. `Tests/secret_scan_test.py` checks
  committed *content*, not just filenames, so don't rely on a clever filename to hide one.
- **Be respectful.** There's no separate `CODE_OF_CONDUCT.md` yet; until there is, the expectation
  is the ordinary one — no harassment, personal attacks, or discriminatory language in issues,
  PRs, commits, or review comments. Disagree with the work, not the person.
.

---

## 1. Setting up a development environment

Requires **Python 3.10+** (the devcontainer uses 3.11; CI's old draft config uses 3.10).

```
git clone <your fork's URL>
cd Pink_Edge_AI
cd App
pip install -r requirements.txt
```

`requirements.txt` installs everything needed to run either UI edition (~1-2 GB: PyTorch,
Ultralytics, OpenCV headless, Streamlit, Flask). If you're touching the training scripts
(`train_tb_classifier.py`, `train_local_model.py`) or the standalone DICOM receiver
(`Tools/receiver.py`), also install the extras:

```
pip install -r requirements.txt -r requirements-dev.txt
```

**Run an edition** (each shares the same backend and the same `App/pink_edge_cache.db`):

| Edition | One-click | Manual (from `App/`) |
|---|---|---|
| Desktop (Tkinter) | `Start.bat` | `python GUI.py` |
| Web (Streamlit) | `Start_Web.bat` | `streamlit run streamlit_app.py` |
| Web (HTML/JS + Flask) | `Start_Web_GUI.bat` | `python web_api.py`, then open `http://127.0.0.1:5000` |

**Codespaces / `.devcontainer`:** opening this repo in a devcontainer installs `App/packages.txt`'s
apt packages (headless OpenCV's system libs) and `App/requirements.txt`, then launches the
Streamlit edition automatically on port 8501.

**Roboflow key (optional):** none of the above needs one — Mammography and Tuberculosis both
reach a real, non-simulated result offline. If you want the Roboflow-hosted tiers, get a key from
roboflow.com and either set the `ROBOFLOW_API_KEY` environment variable or drop it in a file named
`App/roboflow_key.txt` (already gitignored — do not remove that ignore rule).

---

## 2. Finding your way around

The README's [Project layout](README.md#project-layout) section is the authoritative map and is
kept current — read it before adding a new file so it lands next to its siblings
(`Tests/` for a new test, `Tools/` for a standalone script, `Documentations/` for reference docs,
etc.) rather than at the top level. Three things worth knowing up front:

- **One backend, three UIs.** `GUI.py` and `inference.py`/`offline_cv.py` hold the real logic;
  `streamlit_app.py` and `web_api.py` both import from `GUI.py` rather than re-implementing
  anything. If you fix a bug in triage logic, fix it once, in the shared module — don't patch it
  separately in each UI.
- **`Archive/`** holds superseded code (an old Streamlit fork with a known latent RBAC bug, a
  one-shot codemod, the original hackathon submission). It's kept for reference, not maintained —
  don't build new features on top of it.
- **Bilingual UI.** Both live editions support English/Urdu. If you add or change user-facing
  text, add both language strings and exercise the language toggle — `Tests/smoke_test.py` drives
  it.

---

## 3. Making a change

There's one branch (`main`) and no tagged releases yet (see
[SECURITY.md §2.1](SECURITY.md#21-supported-versions)), so:

1. Fork the repo, branch off `main` with a descriptive name (e.g. `fix/mammography-offline-fallback`).
2. Make the smallest change that fixes the thing — this project's own documentation gets corrected
   when a claim turns out to be stale (see `App/Documentations/PRIVACY_POLICY.md`'s approach); treat
   your change the same way: update the doc or evidence report that describes the behavior you
   touched, in the same PR.
3. Run the relevant checks locally (§4) — there is currently **no CI step that runs the test
   suites for you**. `.github/workflows/codeql.yml` runs automatically on push/PR and does static
   analysis only; `.github/workflows/python-app.txt` is a disabled draft (GitHub Actions only
   executes `.yml`/`.yaml` files in that folder), so nothing currently lints or tests a PR
   automatically. Until that's fixed, you are the test runner — say in your PR description which
   suites you ran and that they passed.
4. Open the PR against `main`. Describe what changed, why, and which suites you ran (§4). If your
   change touches a security- or privacy-relevant path (auth, the anonymizer, secrets, any data
   sink), say so explicitly — see the list in
   [SECURITY.md §2.4](SECURITY.md#24-scope).

---

## 4. Testing your change

All commands below run from inside `App/`. Pick the suite that matches what you touched; when in
doubt, or before a PR, run all of them:

| You changed… | Run |
|---|---|
| Anything — fast sanity check (~30s, no downloads) | `python Tests/smoke_test.py` |
| `auth_manager.py` | `python Tests/test_auth.py` |
| `dicom_anonymizer.py` | `python Tests/test_anonymizer.py` |
| Any data sink (cache, reports, GSM/IoT broadcast, logs) | `python Tests/privacy_leak_test.py` |
| Anything that could touch a credential | `python Tests/secret_scan_test.py` |
| `inference.py` / model fallback logic | `python Validation/validate.py` and `python Validation/validate_offline.py` |
| A result field a user could mistake for model output | `python Tests/fabrication_audit.py` |
| Multiple of the above, or before opening a PR | `python Evidence/generate_evidence.py` (`--quick` to skip the two slow model suites) |

`generate_evidence.py` is the one to run before submitting anything non-trivial: it runs every
suite, records actual exit codes (not a hand-written summary), and regenerates
`App/Evidence/EVIDENCE_REPORT.md` with a SHA-256 of every file it attests to. **Regenerate that
report rather than hand-editing it**, and include the updated report in your PR if your change
touched any of the files it hashes.

A failing suite is a blocker, not a note for later — fix the regression or explain in the PR why
the suite's expectation itself needs to change.

---

## 5. Conventions this codebase actually follows

- **Every substantive source file opens with an attribution header.** Not a formality — it's how
  this project discloses AI assistance per file, consistently, across all of it:
  ```python
  """
  <filename>
  ---------------
  <one-line description of what this file does>

  Author Name:  <your real name>
  Author Email: <your email>
  AI Helper:    <tool name, or "None">
  """
  ```
  Add this to every new `.py` file you contribute, filled in honestly — if you used an AI tool,
  name it; if you didn't, say "None." Don't carry over someone else's name.
- **Security-sensitive modules stay dependency-free.** `auth_manager.py` and
  `dicom_anonymizer.py` are both pure stdlib by design (see their own docstrings) — a smaller
  dependency surface for the two files everything else's privacy and access guarantees rest on.
  Don't add a PyPI dependency to either without discussing it first.
- **No hardcoded secrets or PINs, ever** — not even as a "temporary" default. Role PINs are read
  from environment variables / Streamlit secrets and only ever kept as a salted `scrypt` digest
  (see [SECURITY.md §1.1](SECURITY.md#11-access-control-appauth_managerpy)). If you need a value
  for local testing, set the environment variable; don't put it in code.
- **UI color changes must clear WCAG AA contrast.** `Tests/smoke_test.py` checks every themed text
  color against its background; a color that fails the check fails the suite, by design (this is
  how an earlier low-contrast accent color got caught and replaced — see the comment in
  `auth_manager.py`'s `ROLE_CONFIGS`).
- **Dataset/model folder layout follows `App/Models/README.md`.** If you're adding labeled images
  for the offline heuristic or the local classifiers, use the `positive/` / `negative/` /
  `validate/` convention described there so `offline_cv.py` and `train_*.py` pick them up correctly.
- **Dependencies are floor-pinned (`>=`), not lock-filed.** Match that style in `requirements.txt`
  additions unless you have a specific reason to pin exactly (note the reason in a comment if you
  do, the way the existing file explains each of its own pins).

---

## 6. Reporting bugs vs. reporting vulnerabilities

- **Bug, missing feature, bad model accuracy, docs issue:** open a regular GitHub issue. Never
  attach real patient data to it, even as a "realistic" example — use synthetic data.
- **Security or privacy vulnerability** (RBAC bypass, a de-identification gap, a credential
  exposure, anything in [SECURITY.md §2.4](SECURITY.md#24-scope)): **do not** open a public issue
  or PR. Follow [SECURITY.md §2.2](SECURITY.md#22-reporting-a-vulnerability) instead.

---

## 7. Contact

**Imaad Ullah Khan** — yameenimaad@gmail.com — for anything not covered above, or if you're unsure
which category your contribution falls into.
