# Security Policy

Pink Edge AI is an offline-first clinical-triage **prototype** for mammography, tuberculosis
(chest X-ray) and maternal-health (ultrasound) imaging. Three UIs — a Tkinter desktop app
(`App/GUI.py`), a Streamlit web app (`App/streamlit_app.py`), and a static HTML/JS app backed by a
small Flask API (`App/web_api.py`) — share one backend (`App/inference.py`, `App/offline_cv.py`),
one local SQLite report cache, role-based access control (`App/auth_manager.py`), and a DICOM
de-identification utility (`App/dicom_anonymizer.py`).

> **This is a hackathon-grade prototype, not a certified medical device.** It has not undergone
> clinical or regulatory review (no FDA/CE/DRAP clearance), and it **must not be used to make real
> clinical decisions or to process real patient data.** Confidence scores and severity mappings are
> illustrative. Every claim in this document describes *mechanism* — what the code actually does —
> and is backed by a test suite you can run yourself (§1.5); it is not a certification of fitness
> for any particular use.

---

## 1. What security Pink Edge AI actually provides

This section describes real, checkable behavior of the code in this repository today — not
aspirations. Each subsection names the file that implements it and the test that verifies it.

### 1.1 Access control (`App/auth_manager.py`)

- Two roles — Lady Health Worker (view/run/download) and Senior Radiologist (adds override,
  audit-log, DICOM export, user management) — gate who may override an AI assessment.
- PINs are **never hardcoded and never committed**. Each role's PIN is read once, at import, from
  an environment variable (`PINK_EDGE_LHW_PIN` / `PINK_EDGE_RADIOLOGIST_PIN`) or Streamlit secrets.
  Only a **salted `scrypt` digest** of the PIN is kept in memory; the plaintext is discarded
  immediately after hashing.
- If a deployment is started with no PIN configured, a random 6-digit PIN is generated for that
  process and printed **once, to the server console only** — it is never shown in the UI and never
  defaults to a well-known value, so a deployment someone forgot to configure cannot be elevated by
  guessing `1111`/`9999` or any other "default."
- PIN checks use `hmac.compare_digest` (constant-time) against every role's digest with no early
  exit, so a wrong guess costs the same time regardless of which role it was closest to.
- Five failed attempts trigger a 15-minute lockout, tracked process-wide (not per browser session,
  since a session identifier is attacker-controlled and resettable).
- Verified by `App/Tests/test_auth.py` (16 checks: hashing, salting, constant-time comparison,
  lockout, permission sets) and exercised end-to-end by `App/Tests/smoke_test.py`.

### 1.2 De-identification and privacy hashing (`App/dicom_anonymizer.py`)

- Seven sensitive DICOM tag categories (`PatientName`, `PatientID`, `PatientBirthDate`,
  `PatientAddress`, `PatientTelephoneNumbers`, `ReferringPhysicianName`, `InstitutionName`) are
  replaced with a deterministic, one-way SHA-256 hex digest before a scan's metadata is displayed,
  cached, or transmitted anywhere.
- The hash is deterministic across both editions, so repeat scans of the same (pseudonymous)
  patient correlate at the hospital hub without the hub ever learning who the patient is.
- Clinically necessary fields (age, sex, modality, body part) are preserved unhashed.
- Verified by `App/Tests/test_anonymizer.py` (21 checks) and, end-to-end, by
  `App/Tests/privacy_leak_test.py` (§1.5), which checks every place data can actually leave the
  patient's record — not just the hashing function in isolation.

### 1.3 Offline-first; the cloud is opt-in

- Every modality has at least one fully local inference path (an offline pixel-diff heuristic, a
  locally-trained classifier, or cached Hugging Face weights) and runs with **zero internet
  access**.
- `App/Validation/validate_offline.py` proves this rather than asserting it: it blocks every
  outbound socket, confirms the block is real by attempting three escape routes and watching them
  fail, and only then runs all three modalities through to a result (13/13 checks passing as of
  the evidence report in §1.5).
- The Roboflow-hosted model tier — the one path where the actual image leaves the device — is used
  only if a deployer explicitly configures an API key. No key, no outbound image data.

### 1.4 Secrets handling

- The Roboflow API key is resolved in this order: `ROBOFLOW_API_KEY` environment variable →
  Streamlit `st.secrets` → a local `App/roboflow_key.txt` (development only). All three keep the
  key out of source code and out of git; `.gitignore` excludes `App/roboflow_key.txt`,
  `App/.streamlit/secrets.toml`, and every `*.db` cache file.
- `App/Tests/secret_scan_test.py` checks the thing that actually matters — whether a secret value
  is present in committed *content* — rather than only whether one known filename was staged. It
  searches every tracked file for the live key's own value (if one exists locally), known provider
  token shapes (GitHub, AWS, Google, Slack, OpenAI, PEM blocks), and hardcoded-looking
  `KEY = "..."` assignments. This suite exists because a GitHub web upload once bypassed
  `.gitignore` entirely on an earlier copy of this project — see the suite's docstring for the
  incident this was built to catch.

### 1.5 Automated verification, not self-assessment

Every claim above is backed by a suite you can run yourself from `App/`:

| Suite | What it checks |
|---|---|
| `Tests/test_auth.py` | RBAC hashing, lockout, permissions (16 checks) |
| `Tests/test_anonymizer.py` | Privacy-hash correctness (21 checks) |
| `Tests/privacy_leak_test.py` | Every real data sink — GSM broadcast, cache, reports, credentials (10 checks) |
| `Tests/secret_scan_test.py` | No credential anywhere in committed content |
| `Tests/fabrication_audit.py` | Which displayed fields are model output vs. generated filler (diagnostic) |
| `Validation/validate.py` | End-to-end triage against real model backends + ground truth (19 checks) |
| `Validation/validate_offline.py` | Same, with network physically blocked (13 checks) |
| `Tests/smoke_test.py` | Both UIs start, render, and complete a full triage + cache round trip (9 checks) |

`App/Evidence/generate_evidence.py` runs all of these and writes
`App/Evidence/EVIDENCE_REPORT.md`, which is generated from actual exit codes (not hand-written) and
closes with a SHA-256 hash of every file it attests to, so the report cannot silently drift from
the code it describes.

### 1.6 Static analysis

A CodeQL workflow (`.github/workflows/codeql.yml`) scans Python and JavaScript/TypeScript on every
push to `main`, every pull request, and weekly on a schedule, independent of the test suites above.

### 1.7 Repository hygiene

`.gitignore` excludes the runtime SQLite cache, secrets files, `__pycache__`/`*.pyc`, and large
model/dataset binaries, so none of these can land in the repository through a normal commit.

---

## 2. Policies this project operates under

### 2.1 Supported versions

There are no tagged releases. Security fixes are applied only to the latest commit on `main`.

| Version | Supported |
| ------- | :-------: |
| `main` branch (latest commit) | :white_check_mark: |
| Older commits on `main` | :x: — please update |
| Forks, mirrors, and copies maintained by others | :x: — report to their owners |

### 2.2 Reporting a vulnerability

**Do not report security vulnerabilities through public GitHub issues, pull requests, discussions,
or social media.**

1. Preferred: open this repository's **Security** tab → **Report a vulnerability**, or go directly
   to `https://github.com/<owner>/<repo>/security/advisories/new`. Only the reporter and the
   maintainer can see the report. (This requires the repository owner to have enabled private
   vulnerability reporting under Settings → Advanced Security.)
2. If that is unavailable, email **Imaad Ullah Khan at yameenimaad@gmail.com** and ask for a
   private channel. Do not put vulnerability details in the initial message.

Include: a short description and impact (RBAC bypass, PHI leakage, key exposure, etc.); the
affected file/function and commit hash; which edition (desktop, Streamlit, or HTML/Flask), OS, and
Python version; reproduction steps and a minimal proof of concept; and any suggested fix.

**Never include real patient data, PHI, or PII in a report** — use synthetic data, the samples in
`App/Test Data/`, or your own fabricated DICOM metadata. If a report concerns a leak, describe
*which field leaked and how*, not the real values. If you encounter what looks like real patient
data anywhere in this repository, stop, do not copy or share it, and report its location to us
privately immediately — we will treat it as a priority incident.

### 2.3 What to expect

This is a small, mostly single-maintainer project. Targets, not guarantees:

| Stage | Target |
| ----- | ------ |
| Acknowledgement | Within 5 business days |
| Initial assessment | Within 14 days |
| Status updates | At least every 14 days |
| Fix for confirmed issues | Within 90 days (PHI leakage, credential exposure, or auth bypass prioritized) |
| Public disclosure | After a fix ships, or at 90 days, coordinated with the reporter |

### 2.4 Scope

**In scope:** RBAC and PIN handling (`auth_manager.py`); DICOM anonymization and privacy-hash
reversibility (`dicom_anonymizer.py`); the local report cache; Roboflow key handling; the GSM/IoT
alert payload; XSS or injection in either web UI; path traversal or unsafe file handling; unsafe
model/weight loading or deserialization; exploitable vulnerabilities in pinned dependencies; any
secret or real patient data found anywhere in the repository or its history.

**Out of scope:** the simulated Alibaba Cloud sync and hardware-diagnostics panels (no real
credentials or endpoints); vulnerabilities in third-party services themselves (Roboflow, Hugging
Face, Streamlit Community Cloud, GitHub — report those to the vendor); model accuracy or
misclassification with no security component; findings from automated scanners with no
demonstrated impact; denial of service by resource exhaustion; social engineering or physical
attacks; issues requiring an already-compromised host or admin access; the documented limitations
in §3 below, unless you show a *new* way to exploit them; forks or copies maintained by others.

### 2.5 Safe harbor

Good-faith security research is authorized and will not be pursued legally if you: follow this
policy; avoid privacy violations, data destruction, and service disruption; use only synthetic data
or data you own; test only your own local copy, never someone else's deployment; stop and report
immediately if you encounter sensitive data; and give us reasonable time to fix an issue before
public disclosure. This safe harbor covers this project's maintainers only — not Roboflow, Hugging
Face, or hosting providers, which have their own policies.

---

## 3. Honest security posture — how vulnerable is this in practice?

A short, calibrated answer, not marketing: **the mechanisms in §1 are real and tested, and they
hold up for what this project actually is — an offline single-user demo processing synthetic
identifiers.** They are not sufficient on their own for a networked, multi-user deployment or for
real patient data. Specifically:

- **The PIN gate is a workflow control, not an identity/access-control system.** It stops someone
  from casually clicking "override" in the UI; it does nothing against someone with access to the
  Python process or the SQLite file directly. There is one shared secret per *role*, not per
  *person* — so there is no way to know, from an audit perspective, which radiologist actually
  performed a given override. Lockout is process-wide, so it protects the deployment, not a given
  attacker specifically.
- **The privacy hash is unsalted and unkeyed (plain SHA-256).** That's adequate for this
  prototype's synthetic, randomly-generated session IDs, but it is a cryptographic promise it
  cannot keep for real low-entropy identifiers (an 8-digit ID, a CNIC number): without a secret key,
  anyone who can guess the input format can brute-force the mapping offline. Real deployment would
  need an HMAC with a secret key, not a bare hash.
- **The local report cache is an unencrypted SQLite file.** The schema itself holds no name, CNIC,
  phone, DOB, or address — but anyone with filesystem access to the host can open it and read
  cached triage results. Full-disk or database-level encryption is not implemented.
- **Cloud inference, when enabled, sends the image itself off the device.** This is the one
  intentional exception to "offline-first" — it only activates if a deployer supplies a Roboflow
  key, but once it does, clinical image content leaves the device over the network to a third
  party.
- **The Flask edition (`web_api.py`) has permissive CORS and no per-endpoint auth check**, by
  design, for its intended use as a same-machine `127.0.0.1` API. It binds to localhost and
  disables Flask's debugger by default — but if someone rebinds it to `0.0.0.0` for LAN access (as
  the README's self-hosting notes describe), those endpoints become reachable from any device on
  the network with no authentication layer of their own.
- **Neither web UI terminates TLS itself.** Streamlit and Flask both serve plain HTTP; anyone
  running either beyond localhost is responsible for putting HTTPS and real authentication in
  front of it.
- **Dependencies are floor-pinned (`>=`), not exact-pinned or lock-filed.** Builds are reproducible
  in spirit but not byte-for-byte; the project relies on CodeQL and Dependabot-style alerts rather
  than a lockfile to catch a vulnerable transitive dependency.
- **No LICENSE file is published yet.** This doesn't weaken the running code, but it leaves the
  legal terms under which a security researcher may use, patch, or redistribute a fix ambiguous.
- **No independent security or clinical audit has been performed.** The test suites in §1.5 verify
  that the project does what its own documentation claims; they are not a substitute for
  third-party penetration testing or regulatory review.

**Bottom line:** safe to run as a local, single-user, offline demo with synthetic data on a
trusted machine. Not safe to expose on a shared network, use with real patient identifiers, or
treat as a compliance control, without the hardening in §4 below first.

---

## 4. Guidance for deployers

If you run Pink Edge AI anywhere other than your own machine for a demo:

1. **Do not use real patient data** until the project has been independently validated and
   security-reviewed under the regulations that apply to you.
2. **Set real PINs via environment variables or Streamlit secrets** (`PINK_EDGE_LHW_PIN`,
   `PINK_EDGE_RADIOLOGIST_PIN`) — never fall back to an auto-generated demo PIN in anything other
   than a local trial, and treat the PIN gate as a workflow convenience, not a security boundary.
3. **Keep API keys out of git and out of logs.** Use the `ROBOFLOW_API_KEY` environment variable
   or Streamlit secrets; if a key is ever exposed, rotate it at the provider — deleting the file is
   not enough once it has been public.
4. **Protect the report cache.** Put `App/pink_edge_cache.db` on an encrypted volume, restrict its
   file permissions, and delete it when it's no longer needed.
5. **Keep every edition bound to `localhost` unless it sits behind HTTPS and real authentication.**
   This applies to the Streamlit app's `--server.address` flag and to `web_api.py` alike.
6. **If you need real-world identifier protection, replace the bare SHA-256 hash with an HMAC
   keyed by a deployment-specific secret** before any real patient identifier reaches
   `dicom_anonymizer.py`.
7. **Keep dependencies current** and act on CodeQL/Dependabot alerts promptly.
8. **Re-run `python App/Evidence/generate_evidence.py` after any change** to the files it attests
   to, and treat a failing suite as a release blocker, not a note for later.

---

## Acknowledgements

We are grateful to anyone who helps keep Pink Edge AI and its users safe. With permission, valid
reporters are credited in the resulting GitHub Security Advisory and in this section.

*No vulnerabilities have been reported yet.*

---

*This policy is reviewed as the project changes. Last updated: 2026-10-07.*
